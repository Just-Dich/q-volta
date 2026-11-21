# Inkling-2-Mini in the Q-Volta format (plan step 2: converter + format + CPU reference loader)

Branch p13/ink-conv. Scope: converter (`tools/convert_inkling.py`), container additions (`tools/qv_format.py`, `include/qv/model.h`,
`src/model/model_file.cpp`), a calibration run and a quality evaluation on the torch reference. **No GPU kernels.**
Companion docs: `docs/INKLING_SPEC.md` (layer semantics), `docs/INKLING_DESIGN.md` (plan, hypotheses H-I*), `docs/QUANTIZATION.md`.
Every number below was measured in this branch unless it is labelled otherwise; raw results are in
`benchmarks/results/inkling_conv_20261121/`.

## 1. Container additions (format stays `qvolta-v1`; Qwen files load unchanged)

Single-file `.qv` as before: `"QVOLTA1\0" | u64 json_len | u64 data_off | json | pad | data`. Inkling files are written directly in
this layout (`qv_format.ContainerWriter`: 8 MiB header region reserved up front, data appended, JSON written last), so a 13 GB model never
exists twice on disk (the old `.qv.json + .qv.bin` then `qv_pack` path would). `qv_format.Reader` memory-maps a container (used by the evaluator).

* `config.arch == "inkling2_loop"` selects the Inkling branch of the loader; `config.inkling{...}` carries the architecture (table in 3).
  The generic keys `hidden_size, num_hidden_layers, vocab_size (padded, 201024), num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps,
  eos_token_id` are shared with Qwen files; the Qwen-only keys (`rope_parameters`, `linear_*`, `layer_types`, `intermediate_size`) are no longer read
  for Inkling and still required for Qwen.
* **Stacked expert tensors**: a `"q"` entry whose `shape` is `[E, N, K]`. `qweight` is `uint32[E][N][K/32][bits]` (bit-planes exactly as v1), `sm` is
  `[E][N][K/G]` (meta8 `uint16 = s8 | z8 << 8`), `rs` fp16 `[E][N]` (meta8 row scale). Every expert slice is a multiple of 256 bytes, so
  expert `e` starts at `off + e * nbytes / E` (`TensorInfo::expert_off_q/sm/rs`). A grouped kernel needs one base pointer per tensor and the
  expert id.
* **Passthrough kinds** `"nvfp4"` (group 16, bits 4) and `"mxfp8"` (group 32, bits 8): checkpoint-native expert tensors, copied bit-exactly:
  `off_q` = codes (`uint8 [E,N,K/2]` two e2m1 per byte, low nibble = even k; MXFP8: e4m3 bytes `[E,N,K]`), `off_sm` = block scales (e4m3 bytes `[E,N,K/16]`
  resp. e8m0 bytes `[E,N,K/32]`), `off_rs` = fp32 `amax/outer_denominator` per expert (NVFP4 only). Dequantisation = spec section 11. The loader parses
  them (`Kind::NVFP4/MXFP8`); there is no kernel yet (H-I4). A full native file is 17.4 GiB (does not fit the 13 GB output budget of this task), so
  the writer path was exercised only on layer subsets (`--layers`).
* Loader hardening: `ModelFile::open` now throws if any tensor extends past the end of the file (partial / truncated writes);
  `ModelFile::validate_inkling()` checks every expected tensor name, shape and kind against the file's own config (694 tensors for a full 24-layer file).
* Unchanged: bit-plane layout, asymmetric per-group `(scale, min)`, meta8, per-tensor `bits`/`group`.

## 2. Tensor table (names = checkpoint names, `model.llm.` prefix; text model only, no vision tower, **no MTP head**)

| tensor (per layer `layers.L.`) | shape stored | storage |
|---|---|---|
| `mlp.experts.w13_weight` | [128, 2304, 2304] | stacked `q` (profile bits, g128, meta8); rows **interleaved** (even = gate_i, odd = up_i) as in the checkpoint |
| `mlp.experts.w2_weight` | [128, 2304, 1152] | stacked `q` (same bits) |
| `attn.wq_du`, `attn.gate_attn` | [3072, 2304] | q 4-bit g64 meta8 (q3/q4 profiles) or f16 (native) |
| `attn.wk_dv`, `attn.wv_dv` | [1024, 2304] | same |
| `attn.wo_ud` | [2304, 3072] | same |
| `attn.wr_du` [384,2304], `attn.attn_sink_proj.weight` [24,2304], `mlp.gate.weight` [128,2304] | | **f16** (small; router and relative-bias inputs stay exact) |
| `controllers.{attn,k,v,mlp}.{down_weight [64,4608], forget_weight, gain_weight [2304,64]}` (layers 6-17) | | f16 |
| `attn.k_sconv.weight_proj`, `attn.v_sconv.weight_proj` [32,2304] (global layers) | | f16 |
| `attn.q_l2_clip.scale`, `rel_l2_clip.scale`, `attn.rel_logits_proj.proj`, `*_norm.weight`, `mlp.gate.bias` | | f32 |
| `attn.{k,v}_sconv.weight` (local), `attn_sconv.weight`, `mlp_sconv.weight` | **[D, 4]** (checkpoint [D,1,4], the singleton dim is dropped) | f32 |
| `attn.{k,v}_sconv.static_w` (global) | [4, 1024] | f32 |
| global: `q/k/v_input_norm`; local: `attn_norm` | [2304] | f32 |
| top level: `embed.weight`, `unembed.weight` | **[200058, 2304]** (padding rows >= unpadded vocab dropped) | q 4-bit g64 meta8 (q3/q4) or f16 |
| top level: `embed_norm`, `norm`, `attn_res.{attn_queries.N (N=0..22; [3,2304] for N=4,10,16,22), mlp_queries.N, output_query}`, `middle_cycle_cross_pass_{,kv_}queries` | | f32 |

## 3. `config.inkling` (mirrored in `qv::InklingConfig`)

`unpadded_vocab 200058, loop_start 6, loop_end 18, loop_passes 3, loop_gate_rank 64, global_layers [5,11,17,23], local_layers (20), window 1024,
d_rel 16, rel_extent 1024, qk_l2_clip 22.627417, rel_l2_clip 8.0, log_scaling_n_floor 8192, log_scaling_alpha 0.1, attn_res_block 6,
sconv_kernel 4, n_experts 128, topk 8, n_shared_experts 0, expert_ff 1152, route_scale 8.0, gate_softcap 4.0, up_softcap 25.0,
final_logit_softcap 30.0, w13_layout "interleaved", max_position 1048576`. Helpers: `is_global(L)`, `in_loop(L)`, `passes(L)`,
`n_applications(n_layers)` (= 48). The `quant` block records profile, bits, group, Hessian prior, calibration sequences and layer range.

## 4. Decisions

**Dense tensors.** bf16 -> fp16 is lossless wherever the value is in the fp16 range, and the range was checked on every tensor stored from bf16
(`<out>.report.json`): max |w| over all non-expert tensors is 76.5 (`attn_sconv.weight`), **no tensor has a value above 65504**, subnormal-fp16 values
(|w| < 6.1e-5, reduced precision) are 0.23 % of the elements of the f16-stored matrices and the fp16 relative L2 error of the worst tensor is
6e-8 (queries/biases that came as fp32 would be 2e-4 in fp16, so they stay fp32). Small vectors/convs/queries (< 1 M elements) are kept f32 (conv
state math is fp32 in the reference anyway), matrices that are not 4-bit are f16 (the V100 path is fp16/dp4a, there is no bf16 tensor core).
**Activation ranges (fp16 overflow risk on sm_70)**, max |value| over the 10 240 calibration/validation tokens (4 calibration texts + 1 held-out, 48 layer
applications; `calib/stats_*.json`): residual stream layer outputs 485, MoE output 112, expert activation 66, expert gate/up pre-activations 20.5, normalised
MoE input 14.5, router logits 13.7. All are >= 135x below the fp16 limit; the spec's softcaps (4/25/30) and the q/k/rel L2 clips bound the rest.
This is a margin measured on 10K tokens of wiki/code/chat text, not a proof: massive-activation outliers on other inputs are possible, so the engine
should keep the residual stream in fp32 (as the HF flow does inside a layer) and use fp16 only for matmul operands.

**Expert row layout: interleaved gate/up, kept as in the checkpoint.** Reasons: a fused kernel computes `act(gate_i, up_i)`, and with interleaved rows the two
operands of output i are adjacent (one warp reads both, adjacent accumulator columns in an mma fragment epilogue); the converter and the loader do not
touch the permutation; GPTQ/RTN are row-independent so quality is unaffected. `config.inkling.w13_layout` records it; a CPU/torch reader de-interleaves
(`inkling_ckpt.deinterleave`).

**Experts are quantised from the exact dequantisation rounded to bf16** (what the reference multiplies), not from a copy of the codes.

## 5. Profiles and converter

`python3 -I tools/convert_inkling.py --profile native|q4|q3 --out X.qv [--calib DIR] [--layers a:b] [--prior c] [--max-seconds s]`; resumable
(`X.qv.state.json`; slices under `flock /tmp/qv_bench.lock`), ~12 min per full model on the shared V100 (GPTQ batched over 8 experts, `qv_quant.gptq_batched`).

| profile | experts | dense matrices (q,k,v,o,gate_attn, embed, unembed) | small tensors |
|---|---|---|---|
| native | NVFP4 / MXFP8 passthrough kinds (17.4 GiB total: not built in full) | f16 | f16 / f32 |
| q4 | 4-bit g128 meta8, GPTQ+clip | 4-bit g64 meta8 RTN+clip | f16 / f32 |
| q3 | 3-bit g128 meta8, GPTQ+clip | same | same |

Calibration (`tools/inkling_calib.py`): the torch reference forward, fp32, on 4 sequences x 2048 tokens (2 wikitext-train, 1 python source, 1 chat-template
text) = 8192 tokens, plus a held-out sequence val0 (1024 wiki + 1024 code). It dumps per layer application the exact MoE input `x` (fp16) and the routed ids; 
the converter derives per-expert Hessians offline: `H13_e = X_e^T X_e / n_e` over the tokens routed to expert e (all 3 passes of a loop layer pooled, since the
weights are shared), and `H2_e` from the expert activations recomputed with the exact w13. The golden prompts and the README are the evaluation set and are
deliberately **not** calibrated on. Hessian shrinkage (below): `H = (n_e H_e + c H_pool) / (n_e + c)`, default `c = 1024`.

## 6. Hessian prior study (held-out output error, layer 3 / 14, relative to ||W x||, val0 tokens)

3-bit g128 meta8 (layer 3): RTN+clip w13 0.1512 / w2 0.1809; GPTQ per-expert Hessian only (c=0) 0.1551 / 0.2541 (worse than RTN: it overfits, the calibration-set error
of the same run is 0.042 / 0.084); c=64 0.1431 / 0.2035; c=256 0.1381 / 0.1858; c=1024 **0.1345 / 0.1770**; pooled only (c=inf) 0.1352 / 0.1818.
4-bit g128 (layer 3 | layer 14): RTN 0.0764 / 0.0919 | 0.0760 / 0.0966; GPTQ c=256 0.0701 / 0.0947 | 0.0742 / 0.1019; c=1024 **0.0683 / 0.0901** | **0.0708 / 0.0967**; c=inf 0.0686 / 0.0924 | 0.0694 / 0.0972.
So with ~190 routed tokens per expert and pass, pure per-expert GPTQ is **falsified** (overfits, worse than RTN); a pooled/shrunk Hessian gives -10 % on w13 and -2..-18 % on w2 against RTN.
GPTQ helps much less here than on the Qwen weights (-35 % KL, QUANTIZATION.md) because the source is already a 4-bit block-scaled format.
