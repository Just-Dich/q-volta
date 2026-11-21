# Inkling-2-Mini on the CPU backend (branch p14/ink-cpu, 2026-11-21)

Engine: `src/runtime/cpu_inkling_engine.cpp` (dispatched from `make_cpu_engine` on `arch == "inkling2_loop"`), kernels `src/backends/cpu/ink_ops.*`
plus the existing int8-activation GEMV/GEMM (`cpu_gemv.cpp`). Model: `q3a5.qv` (9.886 GiB: experts 3-bit g128 meta8, q/k/v/o/gate_attn 5-bit g64,
embed/unembed 4-bit g64, small tensors f16/f32). Host: Xeon E5-2683 v4 (AVX2, no VNNI), 32 pinned threads, 2 MiB huge pages (AnonHugePages 10.4 GiB
of 11.3 GiB RSS). All numbers measured in this branch, one machine, shared with other jobs (see "Measurement caveat").

## Design
* One code path for decode (T = 1) and prefill chunks (T > 1): each op is written for a chunk of T tokens; only the matmul kernel (gemv_q vs gemm_q),
  the attention key-split and the MoE schedule differ. 48 layer applications, per-application KV / conv slots (40 local, 8 global), attention-residual
  blocks kept per token, loop messages from the previous pass (pre-loop state restarted per pass), exactly as `docs/INKLING_SPEC.md`.
* Scale tables: meta8 is decoded once at load to fp16 (scale,min) pairs. Group-128 tensors keep ONE pair per 128 columns (`QMatC::gs = 1`, tiny change in
  `cpu_gemv.cpp`, default 0 = unchanged for Qwen): experts read 3.25 bit/weight instead of 3.5 with the old duplicate-into-two-64-groups trick.
* Decode MoE: **one parallel region per layer application**. Work items are (expert, 64-row chunk) of w13 and then of w2, claimed dynamically from one
  atomic counter. The w13 chunk epilogue applies softcap/sigmoid/softcap on 32 gate/up pairs (interleaved rows) into the expert's activation vector; the
  chunk that completes the expert quantizes it and releases that expert's w2 chunks (per-expert flag; w2 items are always claimed after all w13 items, so
  the spin cannot deadlock). Combine (weights x 8 experts) is fused into the post-MLP sconv region.
* Prefill MoE: (token, expert) pairs are counting-sorted by expert, quantized activations gathered into expert-sorted rows of one `Q8Batch`; per expert
  `gemm_q` over its tokens (new `xoff` argument selects the rows), items = (expert by descending count, 64 rows); activation -> second batch -> w2 GEMM.
* Attention: fp16 KV. Local layers: ring of 2048 rows per (kv head, slot) (window 1024 + prefill chunk <= 1024); global layers: linear cache per slot
  (allocated lazily, untouched pages cost nothing). Online-softmax partials per (token, kv head, key split), split up to 16 ways at decode; relative bias
  computed per 64-key block from `rel . proj` (no [24,n,1024] tensor); sink as an extra softmax logit; output gate. No RoPE, tau only on global layers.
* f16 side matrices (router, wr_du, sink, loop controllers, dynamic-conv taps) use an AVX2 f16 -> fp32 kernel, split over rows (and 64-token blocks in prefill).
* NEON map: see the headers of `cpu_gemv.h` and `ink_ops.h`; the engine itself only uses ThreadPool and std C++ besides `_mm256` in the conv cores (8-lane
  float; port as 2 x float32x4_t) and `_mm_pause` (-> `yield`).

## Correctness (tests/check_inkling_cpu.py, goldens tests/golden/inkling_q3a5 = tools/inkling_ref.py with the Q3A5 weights read back from the .qv)
| prompt | tokens | greedy ids vs Q3A5 ref | prefill top-1 | KL top-20 | mean abs dlogit top-20 | router top-8 SET mismatches | hidden rel err mean / max (3 positions x 48 apps) |
|---|---:|---|---|---:|---:|---:|---|
| p1_en | 25 | 28/32 (flip at step 28, golden top-2 gap 0.056 logit) | yes | 1e-11 | 0.17 | 187/1200 | 0.043 / 0.13 |
| p2_code | 35 | 32/32 | yes | 1e-10 | 0.12 | 281/1680 | 0.028 / 0.10 |
| p3_multi | 37 | 32/32 | yes | 4e-9 | 1.14 | 358/1776 | 0.056 / 0.50 |
| p4_long4k | 4096 | 32/32 | yes | 1e-11 | 0.88 | 98862/196608 | 0.096 / >0.8 (one position) |
Held-out NLL on val0 (2047 tokens, prefill chunks of 256): engine 1.4648 / 1.4662 (two builds) vs reference-on-Q3A5 1.4643 (docs/INKLING_FORMAT.md) = +0.0005 nats.
**Router ids are NOT identical** (7-50 % of (token, application) pairs differ in one or more of the 8 experts) and hidden states are only within a few
percent: the int8-activation pipeline (32-element groups) perturbs every matmul by ~0.5 % and the looped, softcapped stack amplifies it (same phenomenon as
the Qwen CPU engine, docs/CPU_OPTIMIZATION.md). Evidence that this is noise, not a semantic error: (a) two valid engine runs (batched prefill vs token-by-token
`--chunk 1`) differ from each other by 0.4 % at application 0 growing to 9 %, the same size as their difference from the reference (1 % -> 12 %); (b) the gap to the
reference is 3-5x smaller than the Q3A5-vs-fp32-reference weight-quantization gap (5-35 %) and the ids match the Q3A5 reference at logit level; (c) the NLL is
within 0.0005 of the reference. Results are bit-identical across prefill chunk sizes (256 vs 512). Not available: an fp32-activation engine mode that would give
a tighter check; HF logit parity is still unverified (INKLING_SPEC section 15). Against the fp32 reference goldens (tests/golden/inkling) the Q3A5 weights themselves cost
the 29-32/32 token agreement documented in INKLING_FORMAT.md.

## Speed (benchmarks/results/inkling_cpu_20261121.json; tools/qv_ink_cpu --bench, real prompt tokens from p4_long4k, greedy decode of 64 tokens after prefill)
Bytes read per decoded token as the kernels see them: **2.592 GB** (experts 1.24 GB, dense 5-bit attention 0.85 GB, lm_head 0.26 GB, f16 matrices + loop
controllers 0.23 GB). DRAM ceiling at the nominal 54 GB/s: 20.8 tok/s. (The design-doc ceiling of 23.8 assumed 2.27 GB: 4-bit dense and no scale duplication.)
| ctx | decode tok/s (2 runs) | ms/token | effective GB/s | fraction of 54 GB/s | prefill tok/s (chunk 256) |
|---:|---|---:|---:|---:|---:|
| 128 | 11.27 / 11.67 | 86-89 | 29.2-30.2 | 54-56 % | 69-70 (includes first-touch) |
| 512 | 11.22 / 11.18 | 89 | 29.0-29.1 | 54 % | 77-83 |
| 2048 | 10.96 / 10.97 | 91 | 28.4 | 53 % | 66-68 |
**Measurement caveat**: `bench_cpu_gemv` run in the same sessions (same threads) streamed only 36-41 GB/s (3/4-bit GEMV 36-41 GB/s, "93-98 % of ceiling"), not
the 54 GB/s of docs/CPU_OPTIMIZATION.md, because other jobs share the box (load average 7-24). Against that measured ceiling (~39 GB/s) the engine is at ~75 %
(14.9 tok/s ceiling). The design estimate of 17-20 tok/s is therefore NOT reached here: 11.2 tok/s measured.
Decode profile at ctx 128, 32 threads (ms/token): experts 31.6 (1.24 GB: 39 GB/s, i.e. at the box ceiling), dense projections 18.1 (+ wo 7.4), lm_head 5.5, loop
message gates 5.0, router 2.4, attention 2.9 (8.9 at ctx 2048), everything else (attention-residual, norms, convs, quantization, merges: ~100 barriers per
token) about 8. At ctx 2048 attention reads 235 MB of fp16 KV per token (local rings are constant once ctx > 1024) and is bandwidth bound.
Prefill is compute bound (~0.5 TMAC/s over 32 cores = ~8 int8 MAC/cycle/core, per-group scale/convert overhead of the maddubs tile): breakdown at 512
tokens: w13 28 %, w2 16 %, q/k/v/gate 15 %, wo 7 %, f16 side matrices (gates, router) 10 %, attention 5 %.
**Expert GEMM tile finding**: the existing 4-row x 2-token tile gives ~0.58 TMAC/s on the expert GEMMs at ~16 tokens per expert (chunk 256); chunk 128/256/512/1024
gives 70/79-83/77/71 tok/s (the per-expert token count only changes the amortisation of the weight unpack, which is not the bottleneck); register tiles 2x4, 4x4, 3x3 (rows x tokens)
give 73/75/73 tok/s versus 79-83 for 4x2. Prefill scaling with chunk is flat, so the lever is the inner loop (int32 accumulation across groups is blocked by
per-group scales), not the schedule.

## Limits / not done
* No llama.cpp comparison: no Inkling GGUF exists locally and llama.cpp has no `inkling2_loop` architecture; nothing was downloaded.
* No int4/int8 KV on CPU: global KV is fp16 = 32 KiB per token (256K context = 8 GiB, i.e. NOT within the memory notes' 256K budget; not implemented or measured at long context; max_ctx beyond 4K was never run).
* Prefill chunks are capped at 1024 tokens (local KV ring). Logits are computed for the last token only (all rows with `row_logits` / `dump_hidden`).
* MTP/speculative decoding is not implemented (spec section 14 graph unknown). Pi adapter not applicable (no merged build was tested here).
* ARM/NEON: structure is portable but nothing was compiled or run for ARM.
* Run-to-run noise from other tenants is +-5 %; numbers above are single-session.

## Reproduce
`cmake -S . -B build-cpu -DQV_CUDA=OFF && cmake --build build-cpu --target qv_ink_cpu`; goldens `tools/inkling_golden_q.py --qv q3a5.qv --out tests/golden/inkling_q3a5`;
check `qv_ink_cpu --model q3a5.qv --ids ids.i32 --dump DIR --gen 32` then `tests/check_inkling_cpu.py DIR golden.npz`; bench
`flock /tmp/qv_bench.lock qv_ink_cpu --model q3a5.qv --ids p4_ids.i32 --bench --ctx 128,512,2048 --n 64`; NLL `--nll --ids val0.i32`; profile `QV_INK_PROF=1`.
