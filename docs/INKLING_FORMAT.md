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
(|w| < 6.1e-5, reduced precision) are 0.23 % of the elements of all non-quantised tensors, and the fp16 relative L2 error of every f16-stored matrix is <= 6e-8
(the fp32 source tensors - queries, router bias - would lose 2e-4 in fp16, so they stay fp32). Small vectors/convs/queries (< 1 M elements) are kept f32 (conv
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
| **q3a5** (recommended) | 3-bit g128 meta8, GPTQ+clip | q/k/v/o/gate_attn **5-bit** g64 meta8; embed/unembed 4-bit g64 | same |

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

## 7. Results

Quality reference = the torch reference on the exact dequantised checkpoint (tests/golden/inkling, produced by the same code). All rows: the reference
with the weights read back from the real `.qv` (real pack/unpack/dequant path) or, marked *sim*, quantised on the fly (RTN+clip meta8, same quantiser).
Teacher forced on the golden greedy ids; `top1` = agreement with the 32 golden ids for p1 / p2 / p3 / p4 (4096-token prompt); `top20 |d|` = mean
absolute logit error on the golden top-20 ids (max over the 4 prompts in brackets); `klT20` = KL over the golden top-20 ids renormalised (the golden files hold only the
top-20 values; the full-vocabulary KL at the first step is <= 1.8e-8 for every row because that distribution is extremely peaked, so it says nothing);
`NLL README` = the 511-token set of the spec (reference 1.7561, spec 1.756 reproduced); `NLL val` = held-out 2047 tokens (wiki+code, not in calibration; reference 1.4479).
Differences below about 0.005 nats are noise-level (README row: 3-bit RTN experts give the identical 1.7561 while the logits clearly move).

| profile | file GiB (bytes) | top1 p1/p2/p3/p4 | top20 \|d\| mean (max) | klT20 | NLL README (ref 1.7561) | NLL val (ref 1.4479) |
|---|---:|---|---:|---:|---:|---:|
| reference | 18.2 stored (17.6 text) | 32/32/32/32 | 0 | 0 | 1.7561 | 1.4479 |
| N5 *sim*: experts exact -> 5-bit g64 RTN+clip, dense exact | would be ~15 (not written) | 32/32/32/- | 0.48 (5.8) | 0.0015 | 1.7570 | 1.4452 |
| Q4-experts *sim*: 4-bit g128 RTN+clip, dense exact | - | 31/32/32/- | 0.67 (9.7) | 0.0045 | 1.7560 | 1.4501 |
| Q3-experts *sim*: 3-bit g128 RTN+clip, dense exact | - | 31/32/32/- | 1.25 (9.4) | 0.0192 | 1.7561 | 1.4751 |
| dense-only *sim*: 4-bit g64 RTN on attention+embed+unembed, experts exact | - | 31/32/32/- | 0.82 (7.4) | 0.0056 | 1.7807 | 1.4630 |
| **Q4** real (4-bit experts GPTQ, dense 4-bit) | 12.661 (13,594,480,128) | 29/32/32/31 | 1.03 (17.1) | 0.0104 | 1.7824 | 1.4636 |
| **Q3** real (3-bit experts GPTQ, dense 4-bit) | 9.813 (10,536,832,512) | 31/32/32/30 | 1.39 (8.7) | 0.0287 | 1.7754 | 1.4741 |
| **Q3A5** real (3-bit experts GPTQ, attention 5-bit, embed/unembed 4-bit) | **9.886 (10,614,689,280)** | 32/32/32/29 | 1.33 (10.3) | 0.0245 | 1.7620 | 1.4643 |

Dense-tensor ablation (experts exact, *sim*, NLL README / val): attention q,k,v,o,gate_attn 4-bit g64 1.7822 / 1.4589; 5-bit 1.7619 / 1.4468; embed 4-bit 1.7665 / 1.4493;
unembed 4-bit 1.7608 / 1.4546.

Weight-space error (relative RMS ||Wq-W||/||W||, mean over layers, source = bf16 dequantised checkpoint): Q4 experts w13 0.121 / w2 0.103 (GPTQ trades weight error for output error:
RTN 4-bit g128 on sampled layers is 0.095), Q3 w13 0.235 / w2 0.200, dense 4-bit g64 0.087-0.097. Held-out relative output error of the experts (||(W-Wq)x||/||Wx||, val0 tokens,
mean over layers): Q4 w13 0.070 / w2 0.094, Q3 see `q3.qv.report.json`. Profile (a) native re-pack, RTN+clip on the exact NVFP4 / MXFP8 weights
(sampled layers 0,10,19 NVFP4 and 20,23 MXFP8, 8 experts each; `native_err.json`): **no uniform repack of a 16/32-block-scaled format is lossless** - w13 relative RMS 4-bit g64 0.087, 5-bit g64 0.044,
5-bit g16 0.031 (meta8 itself costs error), 6-bit g64 0.021, 8-bit g64 0.0056; the same for NVFP4 and MXFP8 layers. 5-bit g64 meta8 is 5.25 bpw: 14.9 GiB of experts, i.e. *larger* than
the native 4.5-bpw NVFP4 passthrough (which is exact) and unusable on a 16 GB card, but it is the only repack that is quality-neutral (row N5).

## 8. Findings

1. **Routing skew (H-I5), from the 8192 calibration tokens + goldens (`routing.json`)**: per layer, pooled over passes: the top-16 experts (12.5 %) take on average **37 %** of the routed slots
   (uniform 12.5 %); Gini 0.45; the most loaded expert gets 5.8x the mean load (worst layer 10.7x, layer 17); on average 18 of 128 experts per layer receive < 25 % of the mean load; unused experts are rare with 8K
   tokens (0.08 per layer). On the 4096-token golden prompt: Gini 0.52, top-16 share 42 %. On the 25-37 token prompts 48-58 experts per layer are unused (just too few tokens). So the load is **skewed but not
   cache-friendly enough on its own**: ~190 routed tokens per expert and pass on average, but the rare experts see < 50, which is why per-expert GPTQ overfits (section 6). Same-token overlap between pass 0 and pass 1 of loop layers
   is 4.96 of 8 experts (uniform 0.5), and consecutive tokens share 3.34 of 8 (uniform 0.5): the distinct-expert count of a verify batch (H-I8) will be far below the uniform model of the design doc (to be measured at kernel level).
2. **Dense 4-bit RTN, not the experts, was the quality bottleneck.** Experts at 4-bit g128 RTN or even 3-bit g128 RTN are invisible in README NLL (1.7560 / 1.7561) and cost +0.002 / +0.027 nats on the held-out set;
   4-bit RTN on the attention matrices alone costs +0.026 / +0.011 nats (README / val), 5-bit costs +0.006 / -0.001. Attention is only 0.65 B parameters: 5 bits cost **+0.08 GB**. (H-I falsifies "4-bit dense is free".)
3. **Q3 vs Q4 (the profile choice is falsified in one direction, not the other).** With identical dense treatment Q3 loses +0.010 nats on val vs Q4 (1.4741 vs 1.4636) and has a 2.7x larger klT20 (top-1 agreement is 125/128 for Q3, 124/128 for Q4: not a discriminator),
   in exchange for -2.85 GiB. After the 5-bit attention fix Q3A5 (1.4643 val, +0.0164 = +1.1 %) equals Q4 (1.4636, +0.0157) at 9.89 vs 12.66 GiB. top-1 over the 32 golden ids is 32/32/32/29 (the three flips of p4 start at step 4 and look
   like near ties, not verified); the top-20 logit tolerance is **not** met tightly: mean |dlogit| 1.3, max 10 on low-ranked ids (the spec noted the bf16 flow alone moves logits up to 0.8). The Q3 profile therefore stays a quality-for-size trade, not "lossless".
4. GPTQ with the shrunk Hessian pays for the experts: 3-bit GPTQ (Q3A5, which also carries 4-bit embed/unembed) is +0.016 nats on val against +0.027 for the 3-bit RTN experts alone; the prior matters (c=0 is worse than RTN).
5. The README-511 NLL is a weak discriminator for expert noise (identical for RTN 3-bit and exact); use the held-out set (and top-20/klT20) for decisions.

## 9. Recommendation for the engine

**Q3A5**: experts 3-bit g128 meta8 GPTQ (prior c=1024), attention matrices 5-bit g64 meta8, embed/unembed 4-bit g64, everything small in f16/f32; 9.886 GiB, `--profile q3a5`.
It keeps the design-doc fit (H-I3: ~4.7 GiB left for KV on 16 GB) and matches Q4 quality on every held-out measure; Q4 (12.66 GiB, 13.6 GB file, slightly above the 13 GB budget of this task) is the fallback if the
engine tests show that 3-bit experts hurt agentic text (not measured: no agentic calibration or eval here). For the lossless route the passthrough `nvfp4`/`mxfp8` kinds exist in the container and loader; their kernels are H-I4.
Not done: GPTQ for the dense matrices (the 5-bit RTN fix makes it unnecessary for quality, but a 4-bit GPTQ attention would save another 0.08 GB: not tried), MTP head conversion (graph unknown, spec 14), the HF bf16 parity, a CUDA/CPU engine check, ARM layout (the format is layout-agnostic as for Qwen).

## 10. Loader / tests / reproduce

* `include/qv/model.h` (`InklingConfig`, `Config::arch`, `Kind::NVFP4/MXFP8`, `TensorInfo::stacked/n_experts/expert_off_*`), `src/model/model_file.cpp` (parse, truncation check, `validate_inkling`).
* `tests/test_model_inkling.cpp`: ctest `model_inkling` builds a tiny synthetic container with the full Inkling tensor table (6 layers, loop [2,4)x3), checks the config fields, the table, expert slice offsets and that a truncated file and a mis-shaped expert tensor are rejected;
  `test_model_inkling FILE.qv` validates a real file (cmake `-DQV_INKLING_TEST_MODEL=FILE` registers ctest `model_inkling_file`). Checked on the real q4, q3 and q3a5 files (694 tensors each, header only). Build used: `-DQV_CUDA=OFF -DQV_CPU=ON`, ctest 6/6.
* Reproduce: `inkling_calib.py --out calib` (5 sequences, ~15 min), `convert_inkling.py --profile q3a5 --out X.qv` (~13 min), `inkling_qeval.py --qv X.qv --tasks p1,p2,p3,p4,nll,val --out res.json` (~25 min, 26-55 GB RAM), `inkling_gptq_study.py`, `inkling_routing_stats.py`, `inkling_native_err.py`.
  Disk: calibration dump 2.2 GB, one model file at a time (13.6 / 10.5 / 10.6 GB). Evaluating needs the reference's 3-pass expert cache: set `--cache-gb`.
