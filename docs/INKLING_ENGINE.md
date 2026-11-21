# Inkling-2-Mini CUDA engine (V100, sm_70)

Sources: `src/runtime/cuda_inkling_engine.cu` (engine, schedule, graph capture, prefill driver), `src/backends/cuda/inkling/ink_ops.{h,cu}` (decode kernels),
`src/backends/cuda/inkling/ink_prefill.{h,cu}` (batched-prefill kernels), `tools/ink_run.cpp` (`qv_ink`), `tools/ink_check.py`, `tools/ink_pfcheck.py`.
`make_cuda_engine` dispatches on the model arch (`inkling2_loop`). Model: Q3A5 (`inkling2-mini-q3a5.qv`, 9.886 GiB, all of it resident).

## Decode (step 1)
One CUDA graph per token (the loop schedule of 48 layer applications is unrolled at capture; position and token live in device memory).
Per application: attention-residual mix (+norm, int8 quantisation) -> one multi-matrix GEMV launch (q/k/v/gate dp4a GEMV plus the small fp16 rel/sink/dw projections)
-> prep (short convs, l2 clips, tau, relative-bias table) -> split-K flash-decode (8 lanes per key, 64 splits on global layers, 8 on local) -> combine with sink and
sigmoid gate -> o-proj -> post-attention conv+residual -> router (device-side top-8 with a last-block ticket, graph capturable) -> grouped-expert w13 / w2 GEMV (8 experts
in one launch) -> conv + residual. Loop passes >0 add the message gates (cat/down/forget/gain). Local layers keep a 1024-slot ring; global layers keep the full cache.
Logits: gemv_q8 on the unembedding, softcap 30, argmax fused in the graph (`--bench` replays it).

Decode result (Q3A5, V100, ctx ~25): 122.4 tok/s. At 4K real context: 113.8 tok/s. That is about 33 % of the HBM2 bound at the stored precision, so the engine is NOT
bandwidth bound: nsys (`--cuda-graph-trace=node`) shows many short latency-bound kernels (mean 7-33 us each, 20-25 launches per application). Largest decode kernels per token:
multi-GEMV 1.04 ms, w13 0.93 ms, attention-residual 0.63 ms, w2 0.56 ms, router 0.39 ms, attention split 0.36 ms (ctx 4K), gate-down 0.29 ms, prep 0.24 ms, norm 0.23 ms, finish 0.22 ms.
Remaining headroom is kernel fusion / fewer launches (e.g. fuse norm into the preceding reduction, persistent MoE), not a faster GEMV.
Profile: `benchmarks/results/inkling_gpu_20261121.json` (summary); raw report `/tmp/ink/prof_decode.nsys-rep` (not committed).

## Batched prefill (step 2)
Per chunk (default 2048 tokens, token-parallel): attention-residual and norms emit fp16 rows; projections = dequantise the weight to an fp16 slab + cuBLAS tensor-op GEMM
(fp32 accumulate); relative bias table `rl[t,h,d] = rel[t,h,:] . proj[:,d]` is one GEMM; flash attention is a WMMA two-pass kernel (pass 1 row max incl. sink, pass 2 exp and PV)
with window / causal masking and the sigmoid gate fused; the sliding-window layers stage the 1024-key ring + the chunk (`ring_to_stage`/`stage_to_ring`) so chunks
chain exactly; conv states carry across chunks. MoE: sigmoid top-8 per token, counting sort by expert, gather, per-expert dequant + cuBLAS GEMM (w13 then w2), act kernel
(softcaps 4/25), weighted combine. The router logits use an fp32 GEMM (an fp16 router flipped near-tie expert choices; found by comparing per-application router ids).
Decode graph and prefill share all state, so prefill -> decode is seamless. `QV_INK_STEP_PREFILL=1` falls back to token-by-token.

## int4 KV (step 3)
Global layers (8 applications) store K and V as int4 per token and kv head: 128 channels -> 64 B of nibbles, plus an fp16 (scale, min) pair per 64-channel group, for K and V
(144 B per head-token, 1152 B per token per application, **9216 B/token** for the 8 applications). Quantisation is asymmetric min/max per token, so there are no 64-token blocks and
no fp16 recent window (a simplification of the generic kv4 block records; chosen because the Inkling K is l2-clipped and the test below shows no measurable damage).
Local layers keep fp16 1024-slot rings (a fixed cost). Decode dequantises in registers; the prefill tile loader
dequantises into the WMMA tile. `--kv int4|fp16`.

## Limits
- Max context with int4 KV: 580K tokens fit (chunk 512, 15.68 / 15.77 GiB used); 620K does not. The design estimate was 478K; the difference is that the weights (9.89 GiB) plus
  scratch leave ~5.2 GiB and 9216 B/token. 256K uses 2.4 GiB of KV.
- fp16 KV: 131K needs 14.4 GiB of 15.77 (tight); 256K does not fit.
- Prefill is quadratic: attention dominates beyond ~16K (see table in PERFORMANCE.md).
- Decode attention at long context is 2-3x above its own HBM bound (256K: 17.4 ms/token vs 8.2 ms at 4K; the int4 KV read is 2.4 GB = 2.7 ms at 900 GB/s).
- The pi (harness-hub) LoRA is not implemented in this engine; the plan (merge before quantisation) needs no engine change. Stage B on ARM is not addressed here.

## Correctness
See PERFORMANCE.md. Quantised-reference goldens (`/tmp/ink/qg`, from the CPU Q3A5 reference) and the fp goldens in `tests/golden/inkling`.
