# Inkling-2-Mini on Q-Volta: design and feasibility (Gen 7, p12/ink-design, 2026-11-21)

Target: `thinkingmachines/Inkling-2-Mini` (26.4B total, 3.4B active, looped MoE, 1M context advertised). Goal from the user: run **very fast with large context** on
(a) one Tesla V100-16GB (sm_70) and (b) the CPU backend (AVX2 now, ARM later). This is a pure design/analysis document: **no model code was written and no benchmark was run.**
Companion: `docs/INKLING_SPEC.md` (branch p12/ink-spec) owns the exact layer semantics and the golden reference; this document does not duplicate it and flags every place where a number depends on a semantic that the spec must confirm.

Labels used for every number: **[A]** arithmetic on exact shapes (config.json + the safetensors header, read by HTTP range request, 148,408-byte header, 1,245 tensors);
**[M]** measured earlier on this machine and quoted from docs/PERFORMANCE.md, docs/LONG_CONTEXT.md, docs/CPU_OPTIMIZATION.md, CLAUDE.md or logs/gemv_sweep.txt (all Qwen-27B kernels, not yet Inkling);
**[E]** estimate / assumption (stated); **[N]** the one new measurement made for this document (section 1.4, a weight-requantization error study on 6 real expert matrices).
Every table below is reproducible with `python3 -I tools/ink_model.py` (it prints all of them).

## 0. Verdict

| question | answer | basis |
|---|---|---|
| Fits one V100-16GB? | **Yes, with 3-bit experts (about 9.7 GiB of weights, 4.7 GiB left for KV).** Native NVFP4 experts (13.7 GiB) fit but leave only ~0.7 GiB for KV (~70K-124K tokens). | [A] |
| Decode ceiling, V100, plain | Pure bandwidth 392 tok/s (Q3 profile, 2.27 GB/token at 890 GB/s); with measured GEMV efficiency (0.81) 324; with a per-kernel overhead model (680 kernels x 2.6 us) **206 tok/s**. Realistic target 130-200 tok/s at short context. | [A]+[M]+[E] |
| Decode ceiling, CPU (54 GB/s) | 23.8 tok/s at 100% of DRAM bandwidth for the Q3 profile; **17-20 tok/s realistic** (Qwen reached 98% of the DRAM ceiling). | [A]+[M] |
| Max context on V100 (all-GPU) | **~478K tokens** (Q3 weights, int4 KV, per-pass KV, MTP layer incl.), **~860K if the KV of the looped global layers is shared across passes**. With the non-core experts in host RAM (hybrid) ~930K. Nothing is claimed as validated. **1M is reachable only with a host-RAM KV tier or the hybrid residency**, and 1M quality is unvalidated. | [A] |
| "Very fast with large context", realistic | Attention is only 8 global layer applications, so decode stays fast: with the *current* split-K kernel rate (5.8 ns/key/layer [M-derived]) 91 tok/s at 128K, 59 tok/s at 256K, 37 tok/s at 466K; if the kernel became bandwidth-bound (1.44 ns/key [A]) ~2x-4x more at 256K+. MTP helps *more* at long context (shared KV read). | [A]+[M] |
| Prefill | ~3.8-4.4K tok/s short prompts (chunk 2-4K) if expert GEMMs reach 40 TFLOP/s [E]; whole 256K prompt ~190 s (1.4K tok/s average) and 1M ~38 min; global attention dominates beyond 128K. Qwen's 250K prefill took 1,701 s [M]. | [A]+[M]+[E] |
| Biggest risks | (1) attention/KV semantics of the loop (KV per pass or shared) and the positional scheme (no RoPE in the main model); (2) requantizing an NVFP4 model to 3 bits adds noise comparable to the original quantization noise ([N]); (3) the MoE draft-verify penalty (distinct experts grow with drafted tokens); (4) fp16-only tensor cores for a bf16-trained model; (5) the 1M tier is unvalidated. | see 9 |

## 1. What the checkpoint is (exact, from the safetensors headers)

### 1.1 Files (HF repo tree, sizes in bytes)
`model-00001-of-00001.safetensors` 19,526,019,168 (not 28 GB), `mtp.safetensors` 635,801,416 (**the MTP/draft head is a separate file**), `tokenizer.json` 26,178,678, `tokenizer_config.json`, `chat_template.jinja` (6.9 KB), `hf_quant_config.json`, `config.json`, `generation_config.json`, `processor_config.json`. The local `tiktoken/` directory in the download is empty; there is no `.py` modeling code in the repo (the spec agent must take semantics from transformers `inkling2_loop`).
Vision tower (`model.visual.*`): 636.5 MB, excluded everywhere below. Text tensors: 18.89 GB.

### 1.2 Tensor inventory (per layer unless stated; all dims are [out, in])
| group | tensors and shapes | dtype in checkpoint |
|---|---|---|
| embed / unembed | `embed.weight` [201,024, 2304], `unembed.weight` [201,024, 2304] (**untied**, each 926 MB), `embed_norm`, `norm` | BF16 |
| attention (all 24 layers) | `wq_du` [3072,2304], `gate_attn` [3072,2304] (output gate), `wk_dv` [1024,2304], `wv_dv` [1024,2304], `wo_ud` [2304,3072], `wr_du` [384,2304] (24 heads x d_rel 16), `attn_sink_proj` [24,2304], `rel_logits_proj.proj` [16,1024] (rel_extent 1024), `q_l2_clip.scale` [24,128], `rel_l2_clip.scale` [24,16] | BF16 |
| short convs (sconv) | `attn_sconv`, `mlp_sconv` [2304,1,4] (depthwise k=4 on the hidden stream), `k_sconv`, `v_sconv` [1024,1,4] on K and V (local layers); **global layers 5/11/17/23 use input-dependent taps**: `static_w` [4,1024] + `weight_proj` [32,2304], plus `q/k/v_input_norm` | BF16 |
| router | `mlp.gate.weight` [128,2304] BF16, `mlp.gate.bias` [128] F32 | |
| experts | `w13_weight` [128, 2304, 2304-in] = gate+up fused (2 x 1152 out), `w2_weight` [128, 2304 out, 1152 in]; layers **0-19 NVFP4** (U8 packed 2 per byte, `w13` [128,2304,1152] bytes; `weight_scale` F8_E4M3 per 16 along K [128,2304,144]; `weight_amax` F32 [128] per expert; `outer_scale_denominator` = 2688 = 6x448); layers **20-23 MXFP8** (F8_E4M3 weights [128,2304,2304], UE8M0 scale per 32 along K [128,2304,72]) | NVFP4 / MXFP8 |
| loop controllers (12 core layers 6..17) | `controllers.{attn,k,v,mlp}.{down_weight [64,4608], forget_weight [2304,64], gain_weight [2304,64]}` (gate_rank 64) | BF16 |
| attention residual (global) | `attn_res.attn_queries.N` [2304] (layers 4,10,16,22: [3,2304]), `attn_res.mlp_queries.N` [2304], `output_query`, `middle_cycle_cross_pass_queries` [24,2304], `middle_cycle_cross_pass_kv_queries` [24,2304] | F32 |
| MTP (`mtp.safetensors`, 317.9 M params) | 3 dense decoder layers (q [3072,2304], k/v [1024,2304], **k_proj_ctx / v_proj_ctx** [1024,2304], o [2304,3072], MLP 9216: gate/up [9216,2304], down [2304,9216]), `fuse_weight` [4,4], `mask_embed` [2304], `markov_head`: `markov_w1` [201,024, 256], `rnn_proj` [768,5120], `hidden_out` [2304,256] | BF16 |

Structure facts used below: 24 layers, layers 6..17 run three times (**48 layer applications per token**); global (full) attention layers are 5, 11, 17, 23 (**2 inside the loop, so 8 global layer applications** if the KV of every pass is separate; 4 if it is shared, see 4.1); the other 20 layers are local with window 1024 (40 local applications). 8 KV heads x 128 = 2048 K+V elements per token per layer application.

### 1.3 Parameter and byte budget [A]
Counts are exact (they reproduce the advertised 26.4B: 26,386 M with MTP, and "+1.0B per extra pass": we get 1.087 B).

| group | params | BF16 GB | as shipped |
|---|---:|---:|---|
| routed experts (24 x 128 x 7.963 M) | 24,461 M | 48.92 | NVFP4 20 layers 11.47 GB + MXFP8 4 layers 4.20 GB = **15.67 GB** |
| attention (24 x 26.89 M) | 645 M | 1.29 | BF16 |
| unembed (lm_head) | 463 M | 0.93 | BF16 |
| embed | 463 M | 0.93 | BF16 (one row read per token: can live in host RAM) |
| router (24 x 295 K) | 7 M | 0.014 | BF16 |
| loop controllers (12 x 2.36 M) | 28 M | 0.057 | BF16 |
| MTP drafter | 318 M | 0.64 | BF16 |

Q-Volta sizes (GB, 1e9; bits per weight include metadata: g64 fp16 s/z = +0.5, g128 fp16 = +0.25, g64 meta8 = +0.25, g128 meta8 = +0.125; **expert shapes are valid for all of them**: K=2304 = 36x64 = 18x128 and K=1152 = 18x64 = 9x128, both multiples of the 32-weight chunk):

| scheme | experts 3b / 4b / 5b | attention 3b / 4b / 5b | lm_head 3b / 4b / 5b |
|---|---|---|---|
| g64 fp16 (E0 style) | 10.70 / 13.76 / 16.82 | 0.28 / 0.36 / 0.44 | 0.20 / 0.26 / 0.32 |
| g128 fp16 | 9.94 / 13.00 / 16.05 | 0.26 / 0.34 / 0.42 | 0.19 / 0.25 / 0.30 |
| g64 meta8 | 9.94 / 13.00 / 16.05 | 0.26 / 0.34 / 0.42 | 0.19 / 0.25 / 0.30 |
| **g128 meta8** | **9.56 / 12.61 / 15.67** | 0.25 / 0.33 / 0.41 | 0.18 / 0.24 / 0.30 |
| shipped NVFP4/MXFP8 | 15.67 (11.47 + 4.20) | 1.29 (bf16) | 0.93 (bf16) |

Note the attention + head are tiny in storage (<1.5 GB at 4-5 bit) but **not in decode traffic**: attention is read 48 times per token (see 2).

VRAM-resident profiles (embed on host, MTP at 4.5 bpw, router fp16, controllers 8-bit; weights only, no KV):

| profile (experts / attention / head bits, g128 meta8 unless noted) | weights | GiB |
|---|---:|---:|
| **Q3** (3 / 4 / 5) | 10.40 GB | **9.69** |
| **Q4** (4 / 5 / 6) | 13.60 GB | 12.67 |
| **FP4** native NVFP4 experts, the 4 MXFP8 layers requantized to 4.5 bpw (4.5 / 5 / 6) | 14.75 GB | 13.73 |
| Q5 (5 / 6 / 6) | 16.74 GB | 15.59 (does not fit with any KV) |
| as shipped (NVFP4 + MXFP8 experts, 5/6-bit rest) | 16.65 GB | ~15.5: does not fit with KV |

V100 exposes 15.77 GiB. Fixed overhead = 0.72 GiB measured on Qwen (13.35 - 12.63, CUDA context + scratch) plus **[E] ~0.5 GiB** for MoE routing buffers, MTP snapshots and the larger prefill scratch: 1.2 GiB used below.

### 1.4 [N] Requantizing an NVFP4 checkpoint: how much noise does it add?
This is the one new measurement. Experts 0 and 5 of layer 0 (`w13`) and layer 12 (`w13`, `w2`) were fetched by HTTP range request (6 matrices of 2304 x 2304 or 2304 x 1152), decoded (assumed standard E2M1 codes {0,.5,1,1.5,2,3,4,6} x e4m3 block scale per 16; the global scale amax/2688 is a per-expert constant that cancels in relative error), and requantized with Q-Volta's asymmetric uniform per-group quantizer (RTN; "clip" = our per-group MSE shrink search). Error = ||W_nvfp4 - W_requant||^2 / ||W_nvfp4||^2, mean over the 6 matrices (spread between matrices <= 3% relative):

| requant | RTN rel-MSE (rel RMS) | with clip search |
|---|---|---|
| 4-bit g16 (reference: the NVFP4 group size) | 0.0048 (6.9%) | 0.0045 |
| 4-bit g64 | 0.0085 (9.2%) | 0.0076 |
| 4-bit g128 | 0.0104 (10.2%) | 0.0092 |
| 5-bit g128 | 0.0025 (5.0%) | 0.0024 |
| 3-bit g64 | 0.0377 (19.4%) | 0.0308 (17.6%) |
| 3-bit g128 | 0.0472 (21.7%) | 0.0351 (18.7%) |

Readings: (1) 4-bit uniform g128 adds ~10% relative RMS noise on top of the checkpoint; 3-bit adds ~19-22% (RTN) which is 4.5x the MSE of 4-bit g128. (2) The NVFP4 file's *own* error vs the unreleased BF16 weights cannot be measured here (BF16 weights are not public), so the comparison "requant noise vs NVFP4 noise" is open; typical FP4-g16 noise is of the same order as the 4-bit row above [E]. (3) Block scales within one 128-group vary little (std of log2 scale 0.31), so a g128 uniform grid is not catastrophically mismatched to the FP4 blocks. Caveat: the E2M1 decode and nibble order are assumed, not validated against the HF reference (nibble order does not change these statistics; a wrong code table would). The code histogram of layer 12 w13 expert 0 is nearly flat over magnitudes 0.5..6, which is unusual for Gaussian weights and is a reason to re-run this after the spec agent validates dequantization.
Consequence: **native NVFP4 passthrough (4.5 bpw, zero extra noise) is the quality reference; any 3-bit profile stacks ~20% RMS noise on the experts and needs GPTQ (halved the loss on Qwen [M]) and a KL check against the dequantized checkpoint** (and ideally against the official BF16 behavior through the HF reference).

## 2. Decode cost model, one token

### 2.1 Bytes per token [A]
48 layer applications x (attention 26.89 M + 8 experts x 7.963 M = 63.7 M) + head 463 M + router + controllers. Per token: attention 1.291 B params read, experts 3.058 B, head 0.463 B (a 2.64 B "unique active" vs ~4.8 B read per token because the 12 core layers are read three times; the advertised 5.4B effective is consistent).
Controllers: 28.3 M params per pass-set; **[E] assumed read at every pass (3x) at 8 bits = 85 MB**, the spec may show they run only on passes 2-3.

| profile | experts | attention | head | router+ctrl | total GB/token | BW-only @890 GB/s | at 0.81 efficiency | + kernel overhead model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q3 | 1.194 | 0.666 | 0.295 | 0.113 | **2.269** | 2.55 ms (392 tok/s) | 3.09 ms (324) | 4.86 ms (**206**) |
| Q4 | 1.577 | 0.827 | 0.353 | 0.113 | 2.870 | 3.22 ms (310) | 3.91 ms (256) | 5.68 ms (176) |
| FP4 native | 1.720 | 0.827 | 0.353 | 0.113 | 3.013 | 3.39 ms (295) | 4.11 ms (243) | 5.88 ms (170) |
| Q5 | 1.959 | 0.988 | 0.353 | 0.113 | 3.413 | 3.84 ms (261) | 4.67 ms (214) | 6.43 ms (155) |

Efficiency inputs are measured: 720 GB/s = 0.81 of the 890 GB/s read peak for a 15.7 MB GEMV (gdn_z 4-bit 24.6 us), 850 GB/s = 0.955 for the 0.95 GB lm_head [M]. Overhead model **[E]**: ~14 dependent kernels per layer application x 48 + 8 = 680 kernels, 2.6 us each (derived from the one measured small GEMV: 1024x5120 4-bit, 2.95 MB, best config 5.9 us vs 3.3 us of pure transfer; a single data point, so this is the least certain term: 2 us gives 4.4 ms = 227 tok/s, 4 us gives 5.7 ms = 175 tok/s). Plain decode expectation for Q3: **130-200 tok/s** (hypothesis H-I2 in 9: falsified if < 120 at short context).
The CUDA-graph captured decode on Qwen was GPU-bound (graph gain +0.3% [M]) because kernels were big; with 680 small kernels per token graphs matter more here (launch gaps 2-3 us dominate), see 2.3.

Compare with the PLAN.md H-I2 guess (250 tok/s ceiling): the exact byte count (2.27 GB, not 2.5-3 GB) gives a higher pure-bandwidth ceiling but the realistic number is lower, set by the kernel count.

### 2.2 CPU (AVX2, 54 GB/s sustained read [M])
Same bytes / 54 GB/s: Q3 42.0 ms = **23.8 tok/s ceiling**; Q4 18.8; FP4 native 17.9; Q5 15.8. At 85% of the ceiling (the engine reached 98% on Qwen: 3.93 tok/s x 13.56 GB = 53.3 GB/s [M]): Q3 20.2, Q4 16.0, FP4 15.2 tok/s; at 70% 16.7 / 13.2 / 12.5. The expert slices (2.1 MB for w13 at 3.125 bpw, per expert) are small but a layer application reads ~25 MB across 8 experts, enough for 32 threads; the per-layer barrier cost (48 apps x ~10 barriers x a few us) is ~1-3 ms [E], so expect 15-19 tok/s for Q3. ARM: no measurement possible here; the same byte arithmetic applies to LPDDR5X at its sustained bandwidth (the format stays layout-agnostic, as CLAUDE.md requires).

### 2.3 How batch-1 expert GEMV should be organized
* **Router on the GPU**, no host round trip: GEMV [128x2304] (fp16), bias-adjusted top-8 (selection uses `gate.bias`, weights use the unbiased scores, per spec), `route_scale` 8, write `ids[8]` and `w[8]` to device memory. A fixed-shape grid reads `ids[]` and indexes `base + id * expert_stride`. Data-dependent expert ids therefore change **data, not graph topology**: fully CUDA-graph capturable (as the existing verify pass is), no `cudaMemcpy` of ids.
* **One grouped launch per matrix per layer**, never 8 launches: w13 grouped (8 experts x 2304x2304, 16.6 MB at 3.125 bpw) and w2 grouped (8 x 2304x1152, 8.3 MB). Estimated time **[E] from the measured small-GEMV data point**: w13 grouped ~26 us (648 GB/s effective), w2 grouped ~14 us (589 GB/s); 8 separate launches would cost 44 us for w13. The 2.6 us fixed term is the only extrapolated number; the grouped sizes sit between the measured attn_kv (2.95 MB, 6 us) and gdn_z (15.7 MB, 24.6 us).
* **w13 kernel**: interleave gate and up rows (row 2i = gate_i, 2i+1 = up_i, done by the converter) so that one warp owns both; the epilogue applies softcaps (gate 4.0, up 25.0, per config; the exact activation is in the spec), the activation, and quantizes the 1152-wide intermediate to int8 with per-32 scales in the same kernel (no extra launch). The input vector is quantized **once** and shared by the 8 experts (same x).
* **w2 kernel**: one block owns a set of output rows for **all 8 experts and accumulates in fixed slot order** (sorted by router rank, not by expert id) into an fp32 accumulator multiplied by the routing weight: deterministic, no atomics, and identical in the 1-row and T-row (MTP verify) variants, which is what keeps the "exact greedy" guarantee. Grid = 2304 rows / R blocks (R=8: 288 blocks, enough for 80 SMs).
* Matrices that can be fused: q, output gate, k, v, rel (r) and the 24 sink logits in one N=8,600 x K=2304 GEMV (18.4 MB at 4.25 bpw, ~27 us [E]); `wo` N=2304 x K=3072 stays separate.
* Expected efficiency for the small matrices: ~0.60-0.65 of HBM peak for the grouped expert kernels, ~0.4-0.5 for attn k/v/r, versus 0.81 for 15 MB GEMVs and 0.95 for the head [M]. If this is wrong by 2x on every expert kernel the Q3 token time grows by ~1.2 ms (206 -> ~165 tok/s): the design is not sensitive to it. Falsification in 9 (H-I2).
* Non-GEMV work per layer application (not in the byte model): attention residual (softmax over <= ~5-9 block summaries per token), 3 short convs (hidden-stream k=4 conv at attn and mlp, K/V conv), qk/rel l2 clip, value norm, log-scaling, sink, controllers: all elementwise or tiny GEMVs; they are why the kernel count is ~14 per application. Fusing them (graph or megakernel per layer application) is the main lever after the grouped GEMV exists.
* FP4-native GEMV on sm_70: E2M1 values x2 are integers {0,1,2,3,4,6,8,12}, i.e. **exact int8**, so NVFP4 weights can feed dp4a with int8 activations (our existing path) using a 16-entry byte LUT (prmt-based) plus the e4m3 block scale (LUT of 256 floats) applied once per 16 weights. That is the lossless "FP4" profile (4.5 bpw). It needs ~2-3x the ALU instructions per weight of the uniform-int4 bit-plane path; since our Qwen decode attention kernel was instruction-bound [M], this must be measured (H-I8).

## 3. Prefill cost model

### 3.1 FLOPs and dequantization [A]
GEMM work per token: attention weights 2.58 GFLOP + experts 6.12 GFLOP = **8.70 GFLOP** (+0.2 router/controllers; lm_head only for the last token). At the Qwen-measured cuBLAS rate 77 TFLOP/s [M] the compute-only asymptote is 8.9K tok/s.
**MoE-specific costs**: with 128 experts, a chunk of C tokens gives C/16 tokens per expert on average (C=2048: 128, C=512: 32), so (i) all 128 experts are touched in every layer application at any useful chunk size, and (ii) the grouped GEMMs have M = C/16 per expert: small-M tensor-core GEMMs run well below the 77 TFLOP/s large-GEMM rate **[E]: I use 40 TFLOP/s for experts (sensitivity: 25-77)**.
Dequantizing all 128 experts of one layer application to an fp16 slab = 1.02 B params: read 0.40 GB + write 2.04 GB at 751 GB/s [M, dequant_rows_f16] = **3.24 ms per layer application, 160 ms per chunk (48 applications + attention weights)**, independent of C. The 3 passes dequantize the same weights three times (a cache would need 24 GB of fp16).

| chunk C | expert GEMM 77 TF: tok/s (dequant share) | 40 TF | 25 TF |
|---|---|---|---|
| 512 | 2350 (73%) | 2004 (63%) | 1693 (53%) |
| 1024 | 3714 (58%) | 2918 (46%) | 2302 (36%) |
| 2048 | 5233 (41%) | 3780 (30%) | 2807 (22%) |
| 4096 | 6578 (26%) | **4435 (17%)** | 3152 (12%) |
| 8192 | 7548 (15%) | 4856 (9%) | 3359 (7%) |
| 16384 | 8149 (8%) | 5097 (5%) | 3473 (3%) |
| fused dequant-in-GEMM asymptote | 8854 | 5365 | 3595 |

Chunk size needed for dequant <= 10% of time: **C >= 7.7K tokens at 40 TFLOP/s (5.2K at 25, 12.8K at 77)**. Scratch at C=8192 is ~1-1.5 GiB (sorted activations 8192 x 8 x 2304 x 2 B = 302 MB, attention-residual block summaries ~300 MB, hidden/fp32 buffers) which comes straight out of the KV budget: this is a real trade-off against context length on a 16 GB card; C=4096 (17% dequant, ~0.6 GiB scratch) is the practical choice, with a **fused dequant-in-GEMM grouped kernel as the structural fix** (removes the 160 ms/chunk and the fp16 slabs; H-I5). Slab memory without fusion: ping-pong slabs of 16 experts (255 MB each) instead of one 2.04 GB slab.
Compare Qwen: 1,205 tok/s at chunk 2048 [M]; the design above is 3-4x faster per token because only 8.7 GFLOP (vs ~54 GFLOP) are executed per token.
Routing for prefill: counting sort of (token, slot) pairs by expert on the GPU, segment offsets in device memory; grouped GEMM either by one cuBLAS call per expert (128 x 2 x 48 = 12.3K launches per chunk, ~60 ms of launch cost at ~5 us each, acceptable for a first version and needs the per-expert counts on the host) or `cublasGemmGroupedBatchedEx` if it supports sm_70 (to be checked) or a custom WMMA grouped kernel.

### 3.2 Prefill attention at long context
Global attention FLOPs per application: 6144 T^2 (= sum over positions p of 4 x 24 heads x 128 x p), counted like the measured kernel (causal-useful). Eight applications (per-pass KV) at the measured 27 TFLOP/s [M, head_dim 256; head_dim 128 with 3 q heads per KV head is untested, so 18 TFLOP/s is shown as a sensitivity]:

| prompt | 32K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|
| global attention, 8 apps, 27 TFLOP/s | 2 s | 31 s | 125 s | 500 s | 2,002 s |
| 8 apps, 18 TFLOP/s | 3 s | 47 s | 188 s | 751 s | 3,002 s |
| 4 apps (KV shared across passes), 27 TFLOP/s | 1 s | 16 s | 63 s | 250 s | 1,001 s |
| local layers (40 apps, window 1024): 0.50 GFLOP/token | 1 s | 2 s | 5 s | 10 s | 20 s |
| GEMM + dequant (chunk 4096, 40 TF experts) | 7 s | 30 s | 59 s | 118 s | 236 s |
| **total, 8 apps** | 10 s (3.3K tok/s) | 63 s (2.1K) | **189 s (1.4K)** | 628 s (0.83K) | **2,258 s = 38 min (0.46K)** |

For scale: Qwen's measured 250K prefill is 1,701 s [M] and its attention-only 256K estimate was 499 s (16 layers, head_dim 256). Prefill attention is quadratic: beyond ~128K it is the dominant cost and the lever is the attention kernel (raw `mma.sync` flash kernel instead of WMMA, which we expect to help but have not tried [M: "not tried"]) or reducing the number of global applications (the sharing question in 4.1). The ring-buffered local layers are negligible. CPU prefill: Qwen reached 30 tok/s at 54 GFLOP/token = 1.6 TFLOP/s effective [M]; at 8.9 GFLOP/token the same compute rate would give **~100-150 tok/s [E]** (MoE small-M efficiency unknown, chunk 256 gives only 16 tokens per expert: use larger chunks since all experts are read per chunk anyway: 19 GB of weight reads per chunk at Q3, 0.35 s at 54 GB/s, so C >= ~512 for the CPU).

## 4. Long context

### 4.1 Which layers hold full KV
Global layers are 5, 11, 17, 23. Layers 11 and 17 are inside the loop (6..17) and are **applied three times with different inputs**, so by default each application has its own K and V (they are functions of that pass's hidden state): **8 global KV layer-applications**, 40 local applications with a 1024-window ring (fp16: 40 x 1024 x 4,096 B = **160 MiB constant**). The tensors `middle_cycle_cross_pass_kv_queries` [24,2304] and `controllers.k/v` (gain/forget/down, rank 64) suggest passes may exchange K/V information, which could mean the K/V of later passes are *derived* from, or shared with, earlier ones. **This is the most valuable question for the spec**: if KV is shared across passes, global KV halves (4 applications; every KV number below is halved, 860K tokens instead of 478K, and prefill attention halves).
Per token per application (2048 elements): fp16 4,096 B, int8 with fp16 s/z per 64 about 2,176 B, int4 (the p9/kv4 block record) **1,152 B**.

| KV format | per token, 8 apps | per token, 4 apps | 128K | 256K | 512K | 1M (8 / 4 apps) |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 32 KiB | 16 KiB | 4.0 GiB | 8.0 | 16.0 | 32.0 / 16.0 GiB |
| int8 | 17 KiB | 8.5 KiB | 2.1 | 4.25 | 8.5 | 17.0 / 8.5 |
| **int4** | **9 KiB** | 4.5 KiB | 1.1 | **2.25** | 4.5 | **9.0 / 4.5 GiB** |

The MTP drafter adds one more global KV layer (its `k_proj_ctx`/`v_proj_ctx` over the target hidden states) [E]: +1,152 B/token (int4). Qwen was 18 KiB/token for 16 layers [M]; Inkling is 2x smaller per token, so context-per-GiB doubles.

### 4.2 Context achievable on the V100 after weights [A]
15.77 GiB - 1.2 GiB overhead [E] - 0.16 GiB local rings = free for KV (int4, 9 applications incl. the MTP layer):

| profile | weights | free for KV | max context, per-pass KV | shared KV | fp16 KV |
|---|---:|---:|---:|---:|---:|
| Q3 | 9.69 GiB | 4.72 GiB | **478K** | 860K | 134K |
| Q4 | 12.67 GiB | 1.75 GiB | 177K | 318K | 50K |
| FP4 native | 13.73 GiB | 0.68 GiB | 69K | 124K | 19K |
| Q5 | 15.59 GiB | does not fit | - | - | - |
| **hybrid** (experts of the 12 non-core layers in host RAM, CPU-computed, see 7): Q3 / Q4 / FP4 | 5.24 / 6.79 / 7.33 GiB | 9.17 / 7.62 / 7.09 GiB | 928K / 771K / 717K | | |

So: **256K fits with Q3 easily (2.25 GiB of KV), with Q4 only if the KV is shared across passes (1.4 GiB incl. the MTP layer < 1.75 GiB), and with expert bits <= ~3.8 bpw otherwise**. 1M needs 9 GiB of int4 KV (8 applications) or 4.5 GiB (4): it does not fit next to any all-GPU profile except with shared KV + Q3 (860K, still short of 1M with MTP; ~1M without the MTP layer), or with the hybrid residency (928K with MTP, ~1M without) , or with a host KV tier.
Everything above is arithmetic; quality of int4 KV at head_dim 128 (K has qk l2-clip, value norm, and the keys are post-sconv) and at long contexts is untested (the p9/kv4 passkey/ppl evidence is Qwen up to 32K, with a 250K passkey run on 1 prompt [M]). The ring/fp16-window rule of kv4 (128..191 recent tokens in fp16) stays; the local layers need no quantization at all (160 MiB).

### 4.3 Decode speed at long context
Attention bytes per token = apps x ctx x 1,152 B (at 256K and 8 apps: 2.4 GB = 3.0 ms at 800 GB/s [A]). But the measured Qwen split-K kernel is **instruction-bound, not bandwidth-bound**: Qwen decode at 262,001 tokens was 22.9 tok/s vs 51.7 at 512 (int4 KV) [M], i.e. (43.7 - 19.3) ms / 16 layers / 262K = **5.8 ns per key per layer** [M-derived], 4x slower than the bandwidth floor (1,152 B / 800 GB/s = 1.44 ns) [A]. Inkling has the same 1,152 B per key-layer and the same 24 query heads, so the same kernel rate is the starting assumption **[E]** (3 q heads per KV head instead of 6 and head_dim 128 instead of 256 might change it either way).

| context | 8 apps, kernel as measured | 8 apps, bandwidth-bound | plain tok/s (Q3 base 4.92 ms): as measured / bw-bound | CPU exact int4 attention (10.4 ms per app per 256K [M], linear) |
|---|---:|---:|---:|---:|
| 32K | 1.5 ms | 0.4 ms | 155 / 188 | 10.4 ms |
| 128K | 6.1 | 1.5 | 91 / 155 | 41.6 |
| 256K | 12.2 | 3.0 | **59** / 126 | 83.2 |
| 512K (only hybrid/shared) | 24.3 | 6.0 | 34 / 91 | 166 |
| 1M (host tier only) | 48.6 | 12.1 | 18 / 59 | 333 |

(Halve the attention columns for shared KV.) **What "very fast with large context" realistically means**: ~90 tok/s at 128K and ~60 tok/s at 256K plain with today's kernel, 125+ tok/s at 256K after a bandwidth-bound attention kernel (hypothesis H-I6: its mechanism is the cheaper per-key reduction already noted in PERFORMANCE.md: not done on Qwen); MTP raises these (section 5: 1.8x at 256K for alpha 0.75). At 1M, all-GPU decode would be 18-59 tok/s plain, and the 1M profile does not fit without a tier.

### 4.4 Reaching 1M
1. **Dense hybrid residency** (7): GPU keeps the 12 core layers' experts (the ones read three times per token), all attention, the int4 KV (up to ~930K tokens); non-core experts are computed on the CPU (8 ms/token at 54 GB/s for NVFP4-size experts [A]). Plain decode at 900K+: attention 43 ms (current kernel) / 11 ms (bw-bound) + ~13 ms base = **~18 tok/s now, ~40 tok/s with the kernel fix** [E]; exact, but no headroom to 1M with the MTP layer and needs shared-KV or int3 KV for the last ~100K.
2. **Host-RAM KV tier with sparse retrieval** (our p9/tier design, [M] gather 11-12.2 GB/s, hot window + top-K 64-token blocks): KV of 1M is 9 GiB of the 76 GiB host RAM. Fetch for K=4K tokens/kv head/app: 8 apps x 4096 x 1,152 B = **37.7 MB/token = 3.0 ms of PCIe, prefetched under the ~5 ms step** (Qwen doc: +6% at 16 apps). Selection summaries at 1M: 512 MiB fp16 (256 MiB int8) [A]. **Validation is the open issue**: on Qwen the mass-recall criterion failed already at 16K-32K (0.83 at 32K) while the end-to-end KL stayed small [M]. Inkling may behave differently (it applies `log_scaling` (alpha 0.1, n_floor 8192) which sharpens attention logits with context length, and has attention sinks), but that is a hypothesis, not evidence. Plan: dump real K/queries of Inkling's 4 global layers at 32K from the golden reference and re-run `tools/analyze_recall.py` before building anything (H-I7).
3. **Exact CPU attention over cold blocks** (H-C): 333 ms/token at 1M (8 apps); a slow exact fallback only.
4. CPU backend at 1M: KV 9 GiB fits host RAM; decode = 47 ms (Q3 weights) + 333 ms attention = ~2.6 tok/s exact; at 256K 83 ms + 47 = 7.7 tok/s; use sparse retrieval on CPU too (loads KV from DRAM anyway).

## 5. MTP (3 dense layers, block_size 8, markov rank 256)

What the file shows: a **block drafter**: 3 dense decoder layers (MLP 9216, local/global/local per `local_layer_ids [0,2]`), `mask_embed` (positions of the block are masked tokens, one forward drafts a whole block), context K/V projections `k_proj_ctx`/`v_proj_ctx` fed by the target hidden states of **four tapped layer-applications `target_layer_ids [5, 23, 41, 48]`** (fused by `fuse_weight` [4,4]), and a **markov head** (rank-256 low-rank logits `markov_w1` [201024,256] + an RNN projection) that adds left-to-right dependence among the drafted tokens. It is not an autoregressive chain like Qwen's MTP head: one pass of 0.64 GB (bf16; ~0.18 GB at 4.5 bpw) drafts up to 8 tokens. Cost [E, byte model]: 0.18 GB + lm_head 0.30 GB over the block ~ **0.76 ms** per draft pass vs a 4.9 ms plain step. Needs the 4 hidden-state taps saved during the target pass (4 x 2304 x 2 B per token, trivial; for prefill the drafter KV of the whole prompt is built from them).

**Verification with batched MoE**: verifying T tokens in one pass reads the union of their routed experts. Distinct experts per layer for independent uniform top-8-of-128 routing, T = 1/2/3/4/5/9: **8 / 15.5 / 22.5 / 29.1 / 35.3 / 56.4** [A] (a dense model's 1.11x / 1.14x / 1.25x verify cost [M] does not apply). Verify cost relative to a plain step (Q3, byte model + 5%/row kernel growth): T=2 1.33x, T=3 1.65x, T=4 1.94x, T=6 2.49x, T=9 3.18x; if half of the extra experts are shared between adjacent tokens (likely for related text, **unmeasured**): 1.18x / 1.34x / 1.50x / 1.79x / 2.16x.
Speedup = tokens per cycle x t_plain / (t_draft + t_verify), tokens per cycle = 1 + sum alpha^i (geometric acceptance, alpha = per-position acceptance; Qwen measured 0.9-1.0 code, 0.5-0.8 prose, 0.4-0.75 at 2-32K [M]; Inkling's alpha is unknown):

| alpha | D=1 | D=2 | D=3 | D=4 | D=6 | D=8 |
|---|---|---|---|---|---|---|
| 0.60 | 1.07 / 1.20 | 1.09 / 1.31 | 1.04 / 1.32 | 0.97 / 1.28 | 0.84 / 1.17 | 0.74 / 1.07 |
| 0.75 | 1.18 / 1.31 | 1.28 / 1.54 | 1.30 / 1.65 | 1.28 / 1.69 | 1.20 / 1.67 | 1.11 / 1.60 |
| 0.90 | 1.28 / 1.43 | 1.50 / 1.81 | 1.64 / 2.08 | 1.72 / 2.27 | 1.81 / 2.51 | 1.83 / 2.64 |
| 0.95 | 1.31 / 1.46 | 1.58 / 1.91 | 1.77 / 2.24 | 1.90 / 2.51 | 2.09 / 2.91 | 2.21 / 3.19 |

(each cell: independent routing / half-shared routing; short context.) **Expected speedup 1.3-1.7x at alpha 0.75, 1.6-2.3x at alpha 0.9, best depth 3-4 at alpha <= 0.9; with low acceptance (<= 0.6) MTP is break-even or slower, so it needs an adaptive depth/off switch** (Qwen had a fixed depth: -13% vs the best choice [M]).
**MTP gets better with context**, because the K/V read is shared by all verified rows while experts are not: with the 5%/row model, speedup at alpha 0.75 / D=3 is 1.32x (4K), 1.66x (128K), **1.81x (256K)**, 1.92x (466K); alpha 0.9 / D=3: 1.66 / 2.09 / 2.27 / 2.41x [E].
CPU: bytes scale with distinct experts and nothing else is cheap: T=2/3/4 cost 1.52x / 2.01x / 2.46x a plain step, so CPU MTP gives only 1.0-1.3x at alpha 0.75-0.9 and loses at alpha 0.6 (**CPU: skip MTP for Inkling unless acceptance and expert overlap are high; per-position routing overlap is the number to measure first**).
Exactness: kept by construction if the batched expert kernel's per-token arithmetic equals the 1-token kernel (2.3: fixed slot order, same dp4a partial-sum order), router top-k evaluated per row with the same function, sconv state (last 3 pre-conv values per site) kept in a small ring and rolled back by position like the KV; the loop adds no extra state (a token's 3 passes happen inside one pass over the 48 applications; rows are independent apart from causal KV).

## 6. Expert residency (weights larger than VRAM)
Needed for: FP4-native or Q5 profiles (13.7-16.7 GiB), or to buy KV room (4.2 hybrid) on a 16 GB card. An expert (w13 + w2) is 7.96 M params = 4.48 MB at 4.5 bpw [A]; a token touches 8 x 48 = 384 expert slots = 1.72 GB (139 ms of PCIe if all missed: unusable). PCIe 12.4 GB/s [M]; CPU DRAM 54 GB/s [M] (4.4x faster than PCIe: **compute missed experts on the CPU instead of moving them**; only the 4.6 KB activation vectors cross PCIe, round trip 14-18 us [M]).

| hit rate of a GPU expert cache | miss traffic MB/token | PCIe fetch time | CPU computes the misses |
|---|---:|---:|---:|
| 0.50 | 860 | 69 ms | 15.9 ms |
| 0.70 | 516 | 42 ms | 9.6 ms |
| 0.80 | 344 | 28 ms | 6.4 ms |
| 0.90 | 172 | 14 ms | 3.2 ms |
| 0.95 | 86 | 6.9 ms | 1.6 ms |

Uniform routing gives hit = cached fraction (4 GB of 13.8 GB: 0.29; 10 GB: 0.73; 12 GB: 0.87) [A]; real routing is skewed and temporally correlated, so an LRU/frequency cache does better, **unmeasured**. Because routing is only known after the router of that layer runs, expert prefetch needs a predictor (previous layer's / previous token's expert sets): H-I9. Two structural options:
* **Layer-structured residency (recommended first)**: the 12 core layers' experts live in VRAM (each byte is used 3x per token), the 12 non-core layers' experts (6.9 GB at 4.5 bpw: layers 0-5 and 18-23, which includes the 4 MXFP8 layers 20-23 at 8 bit: 4.2 GB, so ~9 GB of host data for the lossless profile) stay in host RAM and are computed by the CPU: 96 expert slots/token = 430 MB (NVFP4 size) = **8.0 ms at 54 GB/s** + 12 x ~35 us sync, versus 34.7 ms over PCIe. The GPU part of the step (attention, core experts, head) is ~4 ms for a total of **~12-14 ms = 70-85 tok/s plain** [E], the CPU work cannot overlap the GPU inside one token (layer dependency) but can overlap with the next layer's attention by splitting experts between devices only if the router order allows it (not assumed). It uses CPU cores and DRAM bandwidth that would also be needed by anything else on the host.
* **Hot-expert cache + CPU fallback** (dynamic): GPU holds a frequency-ranked subset; misses are executed on the CPU. Only valuable if routing is strongly skewed: first collect expert-usage histograms from the golden reference over real prompts (H-I9 gate).
Do not use CPU weights "in parallel" for throughput on the same expert set (the Qwen finding in CPU_OPTIMIZATION.md still holds: 54 GB/s is 6% of HBM).

## 7. Reuse vs new (what exists in the code base)

| component | reuse | change / new |
|---|---|---|
| `.qv` bit-plane format, `QMat`, qpack/qv_format.py, per-tensor bits and group 64/128, meta8 | **as is** for attention, head, router (fp16), MTP, and for experts | add a **stacked expert tensor** kind: `[E, N, K]` with a per-expert stride so a kernel indexes `base + id * stride`; interleave gate/up rows; optional new kind `nvfp4` (4-bit codes + e4m3 scale per 16 + per-expert fp32 amax) for the FP4-native profile |
| dp4a int8-activation GEMV `gemv_q8` (+ `gemv_q8_multi` rows 2/3/4/8, bit-exact verify) | dense GEMVs (qkvg+r fused, wo, router, head, MTP) | **grouped expert GEMV** (ids from device memory, fixed slot order, T-row variant), fused softcap/activation/int8-requant epilogue |
| dequant `dequant_rows_f16` + cuBLAS slab GEMM prefill | attention-weight GEMMs | **MoE grouped GEMM** with token-sorted expert segments (counting sort), dequant ping-pong slabs, later fused dequant-in-GEMM |
| split-K decode attention, kv4 block records, fp16 recent ring, `attn_seq_tc` WMMA kernel, kv4 prefill fetch | structure, int4 format (1,152 B per token-layer is the same: 2048 elements) | head_dim 256 -> 128 and 6 -> 3 q heads per kv head (tile shapes: 48 rows = 8 queries x 6 heads becomes e.g. 16 queries x 3 heads); **sliding-window ring kernel** for the 40 local applications (fp16, 1024); **relative-position bias** (q_rel . rel_logits_proj over the last 1024 distances, added to scores), **attention sink** (extra softmax term from `attn_sink_proj`), output gate (same epilogue idea as Qwen's gated attention), log-scaling of q for long context, qk l2-clip, value norm without gamma; no RoPE in the main model (positions come from rel bias + sconv: to be confirmed by the spec) |
| MTP verify infrastructure (snapshot/rollback by position, per-row exact kernels, graph per depth) | rollback pattern, `--mtp-depth`, graph capture per T | block drafter (mask_embed, ctx K/V from 4 hidden taps, markov head), distinct-expert-aware verify, per-site sconv ring rollback; the MTP weights file is separate (`mtp.safetensors`) |
| CUDA graphs, split-K combine, int4 KV tier design (docs/LONG_CONTEXT.md) | yes | graph per T, per-application KV addressing (pass index), selection summaries for 8 apps |
| converter `tools/convert.py` (streaming, GPTQ, clip, meta8, LoRA merge) | streaming/GPTQ/meta8 machinery | **NVFP4 and MXFP8 ingest** (dequantize to fp32/bf16: E2M1 table x e4m3 block scale x amax/2688 per expert; fp8 e4m3 x ue8m0 per 32), per-expert layer-wise GPTQ (Hessian only from tokens routed to that expert: with N calibration tokens an expert sees ~N/16, so N >= 64K tokens for K=2304; rare experts fall back to RTN+clip), layer-wise on the GPU (one layer's bf16 experts = 2 GB at a time) because the BF16 model (49 GB experts) does not fit the V100 |
| new layer types | - | **loop schedule** (layers 6..17 x 3 with controllers), **attention-residual blocks** (n_layers_per_attn_res_block 6; queries per layer; the loop adds cross-pass queries), **sconv-like state** (hidden k=4 conv at attn and mlp, K/V conv, dynamic taps on global layers), final **logit softcap 30**, `embed_norm`, **bf16-trained / fp16-only sm_70**: fp32 residual stream and accumulators, the softcaps (gate 4, up 25) bound the expert activations, verify overflow on real activations |
| tokenizer (`src/tokenizer`, Qwen BPE: NFC normalizer + hand-written Qwen split scanner + GPT-2 byte map + merges, loads HF `tokenizer.json`) | byte-level BPE merge core, added-token split, incremental UTF-8 detokenizer, `tokenizer.json` loader | `tokenizer.json` is a HF fast tokenizer with **no normalizer** (`"normalizer": null`, so NFC must be dropped), **a different pre-tokenizer regex** (o200k-style: case-aware letter runs `[^\r\n\p{L}\p{N}]?[\p{Lu}...]*[\p{Ll}...]+(?i:'s|...)?`, digits `\p{N}{1,3}`, punctuation with trailing newlines, `\s+(?!\S)`) so the hand-written scanner must be replaced and re-validated against `tokenizers`, `ignore_merges: true` (whole-word vocab hits before merges), 200,058 real ids (201,024 padded), 60 added tokens (`<|message_user|>` 200000 ... `<|content_model_end_sampling|>` 200006 = EOS, `<|unused|>` 199998, `<|endoftext|>` 199999 = pad, image token `<|unused_200054|>`) |
| chat template (`chat_template.cpp`: Qwen-specific port) | the jinja-subset machinery | new template: roles via `<|message_user|>`/`<|message_model|>`/`<|message_system|>`/`<|message_tool|>`, `<|content_text|>`, thinking blocks (`<|content_thinking|>`) with `reasoning_effort` (none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99), tool calls (`<|content_invoke_tool_json|>` / `_text`); `response_template` regexes in tokenizer_config.json describe how replies are parsed |
| server | OpenAI-compatible server, streaming, prefix reuse | reasoning_content field, tool-call parsing (not in the Qwen server), no vision (reject image parts: image tower excluded); sampling defaults temperature 1.0 (generation_config) vs Qwen's top-p/top-k; the README warns about runaway reasoning loops: output-token limits |
| CPU backend (AVX2 int8 GEMV, huge pages, pinned pool) | GEMV core | grouped expert GEMV (threads over (slot, row block)), MoE prefill (token-sorted batches, 16-row unpack-once pattern of `gemm_q` generalizes), ring-buffer attention; ARM later |

## 8. Order of work (recommended) and risks

### 8.1 Hypotheses (autoresearch style: mechanism / expected delta / falsification)
* **H-I1 [semantics + golden]** spec writes exact layer math (loop, controllers, attention-residual, sconv, rel bias, sink, caps) and a reference that reproduces HF logits. Expected: KL < 1e-3 (fp32). Falsified: > 1e-3 or unresolved semantics after the HF modeling code is read. Also answers **KV per pass vs shared** and **whether controllers run on every pass** (both change numbers here by up to 2x).
* **H-I2 [grouped decode, V100]** mechanism: 14 kernels/app, grouped expert GEMV, graph. Expected: Q3 plain 130-200 tok/s at short context (model 206, BW ceiling 392). Falsified: < 120 tok/s (< 60% of the overhead model) after the kernels are tuned; the profile will say whether small-GEMV efficiency (0.4-0.65) or launch gaps are the cause.
* **H-I3 [Q3 fit + context]** Q3 (9.7 GiB) + int4 KV leaves 4.7 GiB: 478K (per-pass) / 860K (shared) tokens. Expected: engine builds at max_ctx 262,144 with MTP and CUDA graphs and decodes at position 262K. Falsified: > 15.77 GiB at 256K (overhead > 1.2 GiB estimate or KV layout 2x larger than modelled).
* **H-Q1 [requant quality]** GPTQ-g128-meta8 3-bit experts on the dequantized NVFP4 weights keep KL(bf16-dequant || quant) <= 0.10 on wikitext/code/chat (the Qwen e1 profile had 0.112 at 3.3 bpw [M]); core layers and the last 4 (MXFP8) layers at higher bits if the sensitivity run says so. Mechanism: 3-bit RTN noise is 4.5x the MSE of 4-bit [N]; GPTQ halves the loss on Qwen [M]. Falsified: KL > 0.25 or top-1 agreement < 75% on the agentic set: then ship Q4 + shared-KV or the hybrid residency instead.
* **H-I4 [FP4-native lossless profile]** dp4a with an E2M1 LUT is within 20% of the int4 GEMV speed per byte. Expected: 4.5 bpw experts decode at ~0.85 x (3.125/4.5) of Q3 speed per token (~140-170 tok/s) with zero requant noise. Falsified: kernel is ALU-bound at > 1.5x the int4 kernel time per byte.
* **H-I5 [prefill grouped GEMM]** token-sorted grouped GEMM with chunk 4096: 3.8-4.4K tok/s (40 TFLOP/s experts); fused dequant-in-GEMM removes the 17% dequant share. Falsified: < 2.5K tok/s at C=4096 or expert GEMMs < 20 TFLOP/s (then the 25 TFLOP/s row of 3.1 applies: ~3.1K).
* **H-I6 [decode attention at long context]** a bandwidth-bound split-K int4 kernel (cheaper reduction, 3 q heads per kv head) reaches <= 2.5 ns/key-app (today 5.8 [M-derived]): 256K plain 59 -> ~100 tok/s. Falsified: stays > 4 ns.
* **H-I7 [sparse/tiered 1M]** top-K block retrieval (K=4K, hot window 16K+) keeps ppl within +2% and NIAH at 256K and 1M on Inkling's global layers; PCIe 3 ms/token prefetched. Gate: recall study on dumps at 32K first (log_scaling/sinks may peak attention); falsified as Qwen's H-B was (mass recall < 0.9 at 4K fetched). Then the 1M profile is the hybrid residency (dense, ~18-40 tok/s).
* **H-I8 [MTP block drafter on MoE]** expected 1.3-1.7x (alpha 0.75) / 1.6-2.3x (alpha 0.9), larger at long context (1.8x at 256K); falsified if the measured distinct experts per verify pass exceed the uniform model (verify T=4 > 2.0x plain) or acceptance < 0.6 at D=3. Measure routing overlap first on the golden reference (cheap, no kernel).
* **H-I9 [expert residency]** core layers on GPU, non-core on CPU: 70-85 tok/s with FP4-native weights (lossless) and ~900K context; a dynamic hot-expert cache only if routing skew gives hit >= 0.9 at 4 GB. Falsified: measured CPU grouped GEMV < 35 GB/s on 4.5 MB expert slices or sync overhead > 0.1 ms/layer.
* **H-C1 [CPU decode]** Q3 on AVX2 reaches 15-19 tok/s (ceiling 23.8); falsified < 10 tok/s.

### 8.2 Ordered plan
1. **Spec + reference** (p12/ink-spec; this branch depends on it for: KV sharing, controllers per pass, attention formulas, nibble/scale layout validation). Gate: H-I1. Dump routing histograms, per-layer activations (fp16 range check), 32K K/query dumps for the recall study (H-I7) and expert-overlap statistics (H-I8) from it.
2. **Converter**: NVFP4/MXFP8 ingest + stacked expert kind + a requant-noise table on the golden (re-run section 1.4 with validated decoding), layer-wise GPTQ for experts, profiles Q3/Q4/FP4. Gate: H-Q1 on 4096-token eval.
3. **CUDA step engine, short context, bit-exact vs the qsim reference**: dense kernels reused; new: router, grouped expert GEMV, loop schedule, attention-residual, sconv, local-window + global attention (fp16 KV first), softcaps. Gate: golden + H-I2 profile.
4. **Prefill**: counting sort + grouped GEMM + dequant slabs (chunk 4096), attention prefill kernel (head_dim 128). Gate: H-I5.
5. **int4 KV for 8 global applications + 256K build** (H-I3), NIAH + ppl at 32K/64K/128K/256K before any claim; fix attention kernel (H-I6).
6. **MTP block drafter** + MoE-aware verify with adaptive depth (H-I8).
7. **CPU backend**: grouped expert GEMV, MoE prefill, local/global attention; reuse converter output. Can start after step 2 in parallel with 3-6 (format and golden are shared); ARM stays a layout constraint.
8. **Tokenizer/chat template/server** (independent of 3-7; needs only tokenizer.json and the jinja): new pre-tokenizer scanner validated on a few hundred strings against `tokenizers`, template port validated against the jinja on conversations (the Qwen validation protocol: 509 strings, 31 conversations [M]).
9. **1M**: hybrid residency (H-I9) and/or the host KV tier (H-I7) only after 5; both are opt-in profiles.

### 8.3 Risks
1. Loop/attention semantics unclear (KV per pass; whether passes attend across passes through `cross_pass_kv_queries`; positional scheme without RoPE beyond the 1024 rel_extent): can move context by 2x and kernel design by a lot. Mitigation: step 1 first.
2. Quality of 3-bit on top of NVFP4 (noise stacking, [N]); no BF16 reference, only the dequantized checkpoint, so "quality vs bf16" cannot be measured, only vs the shipped checkpoint. FP4-native avoids it at the price of context.
3. fp16 overflow on sm_70 for a bf16-trained model (no bf16 tensor cores): residual stream and softmax in fp32; the up/gate softcaps help; attention residual and value-norm change activation statistics; verify on real activations (Qwen path clamps fp16 outputs [M]).
4. MoE speculative decoding may give less than dense MTP (1.3-1.7x vs 2.5x on Qwen); CPU: probably nothing.
5. Overheads in the byte model are estimates: kernel count (680) and 2.6 us each; MoE small-M GEMM efficiency (40 TFLOP/s assumed); attention kernel at head_dim 128; controllers frequency; MTP layer KV.
6. V100 memory is tight at 256K+ (1.2 GiB overhead is an estimate; Qwen's measured 256K build used 15.2-15.4 of 15.77 GiB [M]); prefill scratch competes with KV.
7. 1M tier quality unvalidated (Qwen's sparse recall failed at 32K [M]); do not advertise 1M until NIAH/ppl at 1M exist (CLAUDE.md rule: never claim an untested context length).
8. Throughput claims depend on routing statistics (distinct experts, overlap, skew): all three are cheap to measure on the reference and should precede kernel work for MTP and residency.

## 9. Appendix: reproduction
`python3 -I tools/ink_model.py` prints every table (inputs: shapes, the measured constants listed at the top of the script). The [N] study used `curl -r` range requests against `model-00001-of-00001.safetensors` (header length 148,408; tensor `data_offsets` are relative to byte 8 + header length) and numpy; the decoded blocks were not kept in the repo. Raw checkpoint facts: `model.safetensors.index.json` total_size 19,525,870,752 bytes (text 18.89 GB + vision 0.64 GB).
