# Inkling-2-Mini MTP ("DSpark" block drafter): graph, validation, CUDA implementation

Status 2026-11-23, branch p19/ink-mtp. The Hugging Face reference has no MTP code (docs/INKLING_SPEC.md section 14 had the graph as an open question).
This document replaces that open question. Source tags: **[vLLM]** = read in vllm-project/vllm `main` (fetched 2026-11-23 from raw.githubusercontent.com;
files `vllm/models/inkling2/nvidia/dspark.py`, `.../ops/dspark.py`, `vllm/models/inkling2/speculator.py`, `speculator_config.py`, `configs.py`,
`.../nvidia/loop.py` (target feature capture), `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, `tests/models/inkling2/test_dspark.py`),
**[DeepSpec]** = github.com/deepseek-ai/DeepSpec (`deepspec/modeling/dspark/{markov_head,common}.py`, training-side reference of DSpark),
**[CKPT]** = `mtp.safetensors` header / `config.json mtp_config`, **[ours]** = measured here, **[inferred]** = not stated by a source.
Confidence in the graph: high. It was reconstructed from vLLM's implementation (the recipe page
https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini calls the speculator `dspark`, `num_speculative_tokens` 7) and then validated empirically (below):
on code the first drafted token matches the target's greedy token 88 % of the time and 3.5 tokens are accepted per cycle on average, which a wrong graph cannot do.
The vLLM kernels run bf16 with specific rounding points; our reference is fp32 and the CUDA path int8-activation GEMV; both only change acceptance, never the output
(verification is exact).

## 1. What it is

DSpark [DeepSpec] = DFlash block drafter (one parallel forward drafts a whole block from the target's hidden states) + a low-rank sequential **Markov head**
(`markov_rank` 256) that makes draft position k depend on the previously drafted token. The Inkling checkpoint's head type is `rnn_hidden`. One cycle:

```
committed context: positions 0..p-1 (target features known), anchor token a = token at p (sampled from the target at p-1, not yet run through the target)
draft:   block of 8 query rows = [embed_norm(embed[a]) , mask_embed x7], positions p..p+7  ->  7 draft tokens d1..d7 for positions p+1..p+7
verify:  target runs rows [a, d1..dk] (positions p..p+k), greedy argmax y_0..y_k
accept:  n = longest prefix with d_{i+1} == y_i ; emit d_1..d_n, y_n ; rows 0..n are committed (n+1 target steps for 1 verify pass)
```
[vLLM `speculator.py`, `configs.py`]: `sample_from_anchor = False` (DFlash layout: the anchor is the bonus token, masks sit AT the predicted positions),
`num_speculative_tokens = block_size - 1 = 7`; the anchor row's own output is not used.

## 2. Target features ([vLLM] `loop.py` `aux_hidden_state_layers`, `configs.py`)

`mtp_config.target_layer_ids = [5, 23, 41, 48]` are **layer visits** (48 applications, loop passes counted, 0-based, order: layers 0-5, loop layers 6-17 x 3 passes, layers 18-23).
Features captured per token position:
* visit 5 = layer 5 pass 0, visit 23 = layer 11 pass 1, visit 41 = layer 17 pass 2: the **layer INPUT** (running partial block / residual stream `h` that enters the visit;
  `aux_hidden_states.append(stack[num_blocks].clone())` before `_run_layer`) = the output of applications 4, 22, 40 in our numbering,
* id 48 = `num_layer_visits`: the final output after the last attention-residual mix and `norm` (the vector the unembedding sees), raw, before softcap.
All four are bf16 `[2304]` per position. In `tools/inkling_ref.py` trace keys: `app04_L4_p0`, `app22_L10_p1`, `app40_L16_p2`, `final_norm`. [vLLM + ours]

## 3. Draft model (`mtp.safetensors`, 54 tensors, bf16) ([vLLM] `model_dspark.py`, [CKPT])

`fuse_alpha = softmax(fuse_weight, dim=-1)` with `fuse_weight [4,4]`: row i (i = 0..2) mixes the 4 target features for draft layer i, row 3 for the Markov head.
`mix(feat, row, w) = rms_w( sum_t alpha[row,t] * feat_t , w)` (fp32 math, eps 1e-6, gamma = the per-layer `dflare_norm.weight`, or `markov_head.dflare_norm.weight` for row 3).

Each of the 3 layers (hidden 2304, 24 q heads / 8 kv heads, head_dim 128, SwiGLU 9216, layers 0 and 2 sliding window 1024, layer 1 global; **RoPE neox half-split, theta 1e6, full 128 dims**, applied to q and k):
```
context K/V of every committed position t (computed once, cached, RoPE at position t):
    c = mix(feat[t], i, layers.i.dflare_norm);  k = rope(rms_w(c @ k_proj_ctx.T per head, k_norm_ctx), t);  v = c @ v_proj_ctx.T      (no norm on v)
block rows (8 rows, positions p..p+7), x = [rms_w(embed[a], embed_norm), mask_embed x7]:
    res = x; xn = rms_w(x, input_layernorm)
    q = rope(rms_w(xn @ q_proj.T per head, q_norm)), k = rope(rms_w(xn @ k_proj.T per head, k_norm)), v = xn @ v_proj.T   (own K/V of the 8 rows)
    attention: scale 128^-0.5, BIDIRECTIONAL inside the block (every row sees all 8 block keys), plus all context keys at positions < p
               (local layers: only the last 1024 context positions, i.e. positions p-1024..p-1); no sink, no relative bias, no gate, no clip
    x = res + o_proj(attn);  x = x + down(silu(gate(rms_w(x, post_attention_layernorm))) * up(...))
hidden = rms_w(x, norm)        # [8, 2304]; rows 1..7 are used
```
Note the two K/V projection sets: `k_proj/v_proj` for the 8 block rows, `k_proj_ctx/v_proj_ctx` (+`k_norm_ctx`) for the context. The block's own K/V is **not** kept: when
position p+j is later committed, its context K/V (from target features) takes its place.

## 4. Markov head (`rnn_hidden`) ([vLLM] `Inkling2MarkovHead`, [DeepSpec] `RNNHead`)

```
base_j   = hidden[j+1] @ unembed.T                           # target LM head (bf16 GEMM in vLLM), j = 0..6
f_rnn    = mix(feat[p-1], row 3, markov_head.dflare_norm)    # features of the LAST COMMITTED position (p-1)
ctx_j    = rnn_proj[:, 512:] @ [hidden[j+1], f_rnn] + rnn_proj.bias        # [768]; rnn_proj [768, 5120] = [state 256 | W1[prev] 256 | hidden 2304 | f_rnn 2304]
state_0 = 0, prev_0 = a
for j in 0..6:
    raw = rnn_proj[:, :512] @ [state, markov_w1[prev]] + ctx_j ; (gate, cand, out) = raw.chunk(3)
    state = sigmoid(gate)*state + (1-sigmoid(gate))*tanh(cand) ;  latent = tanh(out)
    bias = unembed @ (hidden_out(latent))                    # hidden_out [2304,256]+bias; vLLM folds unembed@hidden_out once (fp64) into a [V,256] matrix
    d_{j+1} = argmax softcap_30(base_j + bias) ;  prev = d_{j+1}
```
Greedy acceptance [vLLM `_sample_sequential`, `markov_argmax`]: draft = argmax; the verifier accepts while the target's argmax equals the draft token.

## 5. Empirical validation of the reconstruction (torch reference)

`tools/inkling_mtp_extract.py` runs `tools/inkling_ref.py` (fp32, CPU) over a token sequence, saving the 4 features per position, the target argmax and the prompt+greedy
continuation; `tools/inkling_mtp_ref.py` is the drafter (fp32, from the bf16 weights). Anchors p at stride 48, acceptance = accepted prefix length of the 7 drafts against the
target's own argmax chain `argmax[p .. p+6]`. Inside the generated continuation of a prompt this is exactly the greedy chain; inside a teacher-forced text it is the target's
prediction given the text prefix (an approximation after the first mismatch; the CUDA numbers below are the exact ones). Results `benchmarks/results/inkling_mtp_ref_20261123.json`:

| sequence | anchors | mean accepted (of 7) | position-1 acceptance | conditional acceptance at positions 1..7 |
|---|---|---|---|---|
| code4k (python tools, 4096 tok) | 85 | 3.53 | 0.88 | .88 .84 .73 .83 .90 .74 .76 |
| cu4k (CUDA/C++ source, 4096 tok) | 84 | 2.75 | 0.76 | .76 .72 .80 .78 .79 .74 .88 |
| p4 (model README, tables + prose, 4096 tok) | 85 | 0.89 | 0.44 | .44 .65 .33 .38 .67 .50 1.0 |
| p2_code + 100 greedy tokens (anchors every 4) | 30 | 3.57 | 0.83 | .83 .76 .84 .94 .87 .85 .73 |
| p1_en + 100 greedy tokens (anchors every 4) | 28 | 1.39 | 0.68 | .68 .42 .63 .60 .67 .50 1.0 |

Ablations (code4k, same anchors): **no Markov head** (`base` argmax only): 1.26 accepted (vs 3.53; position 1 0.71) - the head is what makes positions 2..7 work;
**f_rnn zeroed**: 3.55 (no measurable effect; kept as in vLLM). Chance level would be ~0, so the reconstruction (feature taps, norm placement, RoPE, bidirectional block
attention, token shift anchor/p+1, Markov recurrence) is right. No open choice had to be iterated.

## 6. CUDA implementation (this branch)

Files: `src/runtime/cuda_inkling_mtp.inc` (included in the engine class), `src/backends/cuda/inkling/ink_ops.{h,cu}` (row kernels), `tools/convert_inkling_mtp.py`
(sidecar), `tests/test_ink_mtp.cpp`, flags `qv_ink --mtp-depth N --mtp-file F`, `qv_run --mtp --mtp-depth N [--mtp-file F]`.

* **Sidecar** `inkling2-mini-mtp.qv` (next to the model; `<model>-mtp.qv` is the default): transformer matrices RTN+clip g64 meta8 at 5 bit, `markov_w1` and the folded
  `unembed @ hidden_out` ([200058,256]) at 6 bit, norms/`fuse_alpha`/`rnn_wa`,`rnn_wb` f16/f32. The main file is untouched. The base logits use the main model's own quantised unembedding.
* **Exact verify**: R rows (R in {2,3,4,8}, `depth = R-1`) through all 48 applications in one CUDA graph. Every row-batched kernel computes per row exactly the arithmetic of the
  1-token kernel (same reduction order): `mgemv_rows` (weights loaded once, `nr` int8 activation rows), `attn_prep` rows (the short convs read the previous rows' raw inputs),
  `attn_part/combine` rows (each row has its own position; keys written by earlier rows of the same pass come from a staging tail for the sliding layers), router rows, a **union of
  the rows' experts** (`moe_union`; each distinct expert's w13 and w2 weights are read once per matrix, `k_w13_u` applies them to all rows that selected it, w2 partials are summed
  per row in slot order), `finish`/attention-residual with `ntok`>1. Final logits: `gemv_q8_multi` (existing bit-exact multi-token GEMV).
* **State and rollback**: nothing is snapshotted. During verify the 4 short-conv states stay untouched and every row's pre-conv inputs are logged per application
  (`khist/vhist/hist_a/hist_m`, 48 apps x 8 rows); after acceptance (device-side, `k_accept`) `conv_commit_n` sets each state to the last 3 of `[old state, accepted rows]`.
  Sliding-window layers write the new rows to a per-application tail buffer, `ring_commit` copies the accepted rows into the 1024-slot ring (the ring must keep the positions the
  rejected rows would have overwritten). Global-layer K/V (int4) is written in place; entries beyond the committed position are never read.
* **Drafter**: context K/V of each committed position (3 layers, fp16: ring of 1024 for layers 0/2, `max_ctx` entries for layer 1) is computed right after the commit from the
  features captured in the verify pass (and, for the prompt, from features captured in batched prefill); the 8-row block forward and the 7 sequential Markov steps (tiny GEMV +
  6-bit `fold` GEMV over the vocabulary + argmax) are plain kernels. One CUDA graph = draft + verify + accept + commit + ingest, one host sync per cycle to read the count.
* **Tail**: with fewer than R tokens left in a `decode_greedy` call, plain 1-token steps run (and still feed the drafter), so the engine state after any call equals plain decode's.

## 7. Results

See `docs/PERFORMANCE.md` ("Inkling MTP") and `benchmarks/results/inkling_mtp_20261123.json` for tok/s, acceptance per position and verify cost.

## 8. Measured results (V100, device 0, int4 KV, ctx ~40, 3 runs each, 128 timed tokens; ids FNV hash identical to plain in every run)

| text | depth | plain tok/s | MTP tok/s | tokens per cycle (1 + accepted) |
|---|---|---|---|---|
| code | 7 | 118.6 | 148.8 | 4.63 |
| code | 3 | 118.6 | 149.0 | 3.21 |
| code | 1 | 118.6 | 114.1 | 1.90 |
| prose | 7 | 118.6 | 151.0 | 4.73 |
| prose | 3 | 118.6 | 152.2 | 3.29 |
| prose | 1 | 118.6 | 114.6 | 1.91 |

Also measured earlier (code, ctx 512, depth 7, 2 runs): plain 115.4, MTP 134.4 tok/s, 4.32 tokens per cycle.
Exactness (`tests/test_ink_mtp.cpp`, 80 tokens, depths 7/3/1, prompts p1_en, p2_code, p3_multi and a 4096-token code prompt): ids and next-step logits identical in all 12 cases.
Implied cycle time (tokens per cycle / MTP tok/s): about 31 ms at depth 7, 21 ms at depth 3, 17 ms at depth 1, against 8.4 ms for one plain step. The cycle is the cost
driver: draft + 8-row verify is about 3.7 plain steps, so the gain is 1.25x, not the ~4x token yield. Per-position acceptance on the CUDA path is not broken out
(only totals per cycle); the torch reference table in section 5 has per-position numbers.

Not done: the ctx 512/4K/32K x code/prose matrix (the GPU lock was held by another agent for the rest of the session), verify cost at 2/4/9 rows in isolation,
adaptive depth, acceptance versus sidecar bits (only 5/6-bit was built), CPU engine, `qv_server --mtp`.

## 9. Per-stage GPU time (QV_MTP_PROF=1, each stage as its own CUDA graph, ctx ~512, code, V100)

| stage | ms |
|---|---|
| plain 1-token step | 8.62 |
| drafter block (8 rows + 7 Markov steps) | 3.21 |
| verify R=2 / 3 / 4 / 8 rows | 13.26 / 15.66 / 18.31 / 28.38 |
| accept + commit + ingest (R=8) | 0.11 (measured on dummy state, so a lower bound) |

Verify adds about 2.5 ms per extra row and a 2-row verify already costs 1.54 plain steps, so the verify pass, not the drafter or rollback, is the cost driver
(drafter 3.2 ms + verify 28.4 ms = 31.6 ms, matching the 31 ms measured cycle). Consistent with the MoE expert weights being read per row: rows rarely share experts.
Open optimisation targets: expert-weight traffic in the row path, adaptive depth (depth 3 = 3.21+15.7 ms for 3.2 tokens).
