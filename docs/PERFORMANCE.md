# Q-Volta — Performance report (Tesla V100-SXM2-16GB, sm_70)

Status: Stage A (GPU) in progress. All numbers below were measured on this machine; nothing is projected.
Hardware/software: V100 SXM2 16 GB, SM clock 1447-1530 MHz, CUDA 12.9, driver 580.178.04. Model: Qwen3.8-27B (BF16 rev 37561816), quantized by our converter (E0: RTN g64, MLP 3-bit, GDN/attn 4-bit, embed 4-bit, lm_head 6-bit; 12.63 GiB).
Competitor: NInfer (Apache-2.0), model Qwen3.8-27B-GSQ-RCO-IQ3_S (13.99 GiB file, ~3.5 bpw average, mixed 1..6-bit per tensor). Raw logs: benchmarks/baselines/, logs/.

## Decode (greedy, 128 new tokens, thinking off, 3 repeats, deterministic)
| engine | mode | code tok/s | prose tok/s |
|---|---|---:|---:|
| NInfer | plain | 45.3 | 45.3 |
| NInfer | MTP-3 | 106.5 (282/294 drafts) | 69.0 (228/459) |
| Q-Volta E0 | plain | 52.2 | 52.1 |
| Q-Volta E0 | MTP | see the 2026-11-20 GPU MTP section below | |
Bytes-normalized plain decode: Q-Volta 52.2 x 12.63 = 659 GiB/s effective vs NInfer 45.3 x 13.99 = 634 GiB/s (+4%). Most of the raw +15% comes from reading fewer bytes (coarser quantization), not from a faster kernel. NInfer artifact may contain tensors not read per token, so treat +4% as an upper-bound-ish estimate until verified.

## Prefill (cold, no prefix cache, identical wikitext prompts)
| prompt tokens | Q-Volta (chunk 2048) | Q-Volta (chunk 512) | NInfer (chunk 1024) | Q-Volta chunk-2048 vs NInfer |
|---:|---:|---:|---:|---:|
| 512 | 1013 (one chunk) | 1013 | 837 | +21% |
| 2048 | 1205 | 971 | 1010 | +19% |
| 4088 | 1143 | 917 | 1013 | +13% |
Path: dequantize 32-MiB weight slabs to fp16 -> cuBLAS tensor-core HGEMM; GDN recurrence in-kernel with register-resident state; fp32 CUDA-core flash-style attention. Reps agree within ~1%.
Why chunk size matters: weights are re-dequantized once per chunk, so chunk 512 paid for 8 dequant passes on a 4K prompt. Profile at 4088 tokens (chunk 512): GEMM 61% (~77 TFLOP/s, near the cuBLAS ceiling), dequant 15%, attention 10%, GDN 10%. Chunk 2048 costs +0.6 GiB VRAM (14.9 GiB total at max_ctx 8192).
Multi-chunk correctness: golden passes with chunks of 16 and 23 tokens (GDN/conv/KV carry across chunks), same error level as single-chunk.
Not yet measured: prompts > 4K (attention is CUDA-core fp32 and grows quadratically; expect the lead to shrink), VRAM at long context (fp32 KV).
History: step-by-step 51 -> batched 680 -> aligned slabs 1014 -> chunk 2048 1205 tok/s. (An earlier "25.6 tok/s" figure was a benchmark-timer artifact in --graph mode; fixed.)

## Quality (the co-metric; speed wins at worse quality are not wins)
- wikitext-2 test, 4088 tokens: BF16 ppl 8.468 -> E0 9.508 (+12%), KL(ref||q) 0.122, top-1 agreement 84.6%.
- NInfer README (different protocol, 2048-token windows): ppl 7.071 vs BF16 7.05 (+0.3%). Only the ratio is comparable: we are far behind on quality.
- KL is heavy-tailed (median 0): a few confident flips on structural tokens (JSON keys) dominate the mean. Report median/p99/flip-rate next.
- Engine correctness: golden vs simulated-quant reference PASS on real weights. Step path (int8 activations): layer rel err 2.0e-2, KL 5.7e-4. Prefill path (fp16 activations): layer rel err 6.3e-4, KL 7.3e-5.

## pi adapter (harness-hub/Qwen3.8-27B-pi-adapted, LoRA r=32 alpha=64)
Merged into BF16 before quantization (256/256 modules). Engine golden PASS. Adapter's logit shift (rms 0.08 wikitext / 0.23 agentic) is 8-18x smaller than 3-bit RTN noise (0.63 / 4.08); cosine between BF16 shift and quantized shift 0.25-0.44 => behaviour is partly blurred. Plan: runtime LoRA over a once-quantized base (L1). Not yet verified on real agent trajectories (the trajectory bucket is not publicly readable).

## Context length
Current engine: fp32 KV (128 KiB/token) => ~20K tokens max in VRAM. 256K needs 4-bit KV AND weights+scratch <= ~11.3 GiB (now 13.3). Not tested at any long length yet; no claim made.

## Known limitations / not done
split-K long-context decode attention is done (see below); KV quantization, tensor-core attention, quality improvements (GPTQ/bit allocation), tokenizer/server/CLI, CPU backend (gated until the GPU completion gate).

## Decode vs context (split-K attention, 2026-11-20)
| context at decode start | before | after |
|---:|---:|---:|
| ~128 | 52 tok/s | 52.3 |
| 4088 | 36 | 50.2 |
Kernel-only (per layer, us, old -> new): pos 511 52->19.5; 4095 573->69; 16383 2278->189; 32767 4557->348. Short contexts (pos<=63) are ~4 us slower per layer (second launch). Not yet measured end-to-end at >4K; no NIAH/quality check at long context yet.

## GPU MTP speculative decoding (2026-11-20, branch p8/gpumtp)
Exact greedy speculative decoding with the checkpoint's own MTP head (same semantics as the CPU path, docs/CPU_OPTIMIZATION.md). Output is token-for-token identical to plain greedy (and the final logits are bit-identical, i.e. the rolled-back engine state equals plain decode state): test_cuda_mtp (depth 1, 2, 3; 4 prompts; 97 tokens) and every one of the 60 benchmark runs below (ids compared against the plain run).
Mechanism: verify [cur, d1..dD] in ONE pass over the weights (`gemv_q8_multi`: weight planes unpacked once, per-token arithmetic and warp split identical to the 1-token dp4a kernel, test_gemv_multi checks bit equality incl. both warp configs); rmsnorm / conv / GDN / split-K attention / f16 dual GEMV run all rows in one launch with the exact 1-token per-row arithmetic; GDN state is carried in registers across the verify rows and a snapshot after each row (and the conv state) is written for O(copy) rollback; KV is rolled back by position. Draft: mtp.fc + one attention+MLP layer + shared lm_head, chained for depth > 1 (MTP output hidden fed back). MTP KV for accepted positions is rewritten from the true main hidden after every cycle. `--mtp-depth 1..3`, `EngineOptions::mtp_depth`.
Protocol: qv_run --bench --prefill --graph, 129 tokens (128 timed), max_ctx 4608, V100 (device 0) otherwise idle, 3 interleaved runs per cell (spread = min..max), plain row re-measured with the same build. Chat prompts code/prose are the NInfer-protocol prompts (37/47 tokens); 512/2048/4088 are wikitext prefixes (real text, base-model continuation, not chat).
| prompt (ctx) | plain tok/s | MTP depth 1 | depth 2 | depth 3 | draft acceptance d1 / d2 / d3 | ids identical |
|---|---:|---:|---:|---:|---|---|
| code, chat (37) | 52.22 [52.22-52.23] | 87.16 [86.94-87.27] (1.67x) | 116.68 [116.63-116.76] (2.23x) | 129.46 [129.44-129.49] (2.48x) | 1.000 / 0.955 / 0.912 | yes |
| prose, chat (47) | 52.22 [52.21-52.22] | 79.63 [79.62-79.66] (1.52x) | 83.94 [83.93-83.96] (1.61x) | 84.33 [84.25-84.37] (1.61x) | 0.829 / 0.558 / 0.472 | yes |
| wikitext (512) | 52.05 [52.04-52.05] | 85.75 [85.67-85.82] (1.65x) | 115.24 [115.15-115.35] (2.21x) | 134.48 [134.37-134.57] (2.58x) | 0.984 / 0.955 / 0.960 | yes |
| wikitext (2048) | 50.78 [50.76-50.80] | 69.93 [69.83-70.05] (1.38x) | 76.69 [76.42-76.86] (1.51x) | 72.61 [72.59-72.64] (1.43x) | 0.693 / 0.524 / 0.409 | yes |
| wikitext (4088) | 48.58 [47.68-49.98] | 69.77 [68.84-70.83] (1.44x) | 82.35 [81.61-83.73] (1.70x) | 79.90 [79.16-80.95] (1.64x) | 0.753 / 0.670 / 0.549 | yes |
(acceptance = accepted drafts / drafted tokens. Runs are deterministic, hence the tiny spread except at 4088 where plain itself varies 47.7-50.0.)
Versus NInfer MTP-3 on the same chat prompts and V100 (NInfer numbers are from the earlier run in this document, NOT re-run today; NInfer was measured through its server timings, ours with qv_run's decode window, so the measurement paths differ): code 129.5 vs 106.5 tok/s (+22%), prose 84.3 vs 69.0 (+22%; depth 2 gives 83.9). Plain: ours 52.2 vs 45.3. Caveats: NInfer's IQ3_S file is 13.99 GiB with ppl +0.3% vs our E0 12.63 GiB at +12% ppl, so our speed is partly bought with lower quality; MTP does not change quality (output equals our plain greedy).
Cycle breakdown (QV_MTP_PROF=1, one run each, ms per cycle): verify 21.3 (depth 1, ctx 39) / 21.8 (depth 2) / 23.9 (depth 3) versus a plain step of 19.1 ms: verify costs 1.11x / 1.14x / 1.25x of a 1-token step for 2 / 3 / 4 tokens (hypothesis was ~1.1x for 2 tokens: confirmed). Draft (+MTP KV update) costs 1.6 / 3.2 / 4.7 ms per cycle, about 1.6 ms per chained step, 1.2 ms of it the 6-bit lm_head (0.95 GB) over the full vocabulary. At ctx 4088: verify 23.0 / 24.3 / 27.3 ms, plain step 20.6 ms. Tokens per cycle = 1 + accepted drafts per cycle.
Depth 1 hypothesis (acceptance ~0.8 -> ~1.5x, 52 -> ~75 tok/s): prose 1.52x (79.6) and code 1.67x (87.2) at depth 1, confirmed; it does not beat NInfer MTP-3 by itself, depth 3 does on code.
Which depth: it depends on acceptance. Code / predictable text: depth 3 (2.5x). Chat prose: depth 2 and 3 are equal (84 tok/s) because the third draft is accepted only 47% of the time. At 2048-4088 tokens of non-chat wikitext depth 2 is best and depth 3 is worse than depth 2. No adaptive depth policy is implemented (a fixed depth is chosen at engine construction); the numbers above show a per-workload choice is worth up to ~13% over a fixed depth 3.
Graphs: the verify pass and draft steps are CUDA-graph captured, but the decode is GPU-bound: graph on/off gives +0.3% (code d3 129.5 vs 129.2, prose d1 79.65 vs 79.37). The big win over the first working version (verify 22.8 ms -> 21.3 ms at depth 1, 28.7 -> 23.9 ms at depth 3) came from launching all verify rows in one kernel; the remaining gap to a plain step is the multi-token GEMV (~9% slower per call than the single-token kernel for 3-bit) and per-row argmax.
VRAM at max_ctx 4608: plain 14.51 GiB, MTP depth 1/2/3: 14.74 / 14.89 / 15.04 GiB (of 15.77): GDN+conv snapshots cost 0.15 GiB per depth step, plus verify scratch and the MTP KV slot (fp32 KV: 128 KiB/token for the 16 main attention layers, +8 KiB/token for the MTP layer). Depth 3 leaves ~0.73 GiB at 4608, i.e. by arithmetic roughly 5K more tokens of KV (max_ctx ~10K, estimate, not tested) until KV is quantized.
Prefill cost of MTP: the MTP layer also processes the prompt: prefill tok/s 1011 -> ~995 (512), 1208 -> 1187 (2048), 1131 -> 1108 (4088) = 1.5-2.0% slower.
Not done / limits: lm_head restricted to a candidate set for the draft (would cut ~1.2 ms per draft step), adaptive depth, depth > 3 (the multi-token GEMV is compiled for 2, 3, 4, 8 rows), sampling (temperature) speculation (only exact greedy is implemented), MTP with the pi LoRA merged model (acceptance with an unadapted head not measured), server integration (the GPU server path does not call decode_greedy yet), contexts > ~8K (VRAM).
Tests (this machine, 2026-11-20): ctest all pass (gemv, gemv_multi, dequant, attn_seq, gdn_seq, attn_decode_split, cuda_mtp and the CPU tests); model golden (tests/check_golden.py, 64 tokens) PASS: worst layer rel 0.033, final logits rel 0.012, KL(qsim||engine) 6.5e-4, top1 agree vs qsim 0.984. Raw data: benchmarks/results/gpu_2026-11-20.json.
Plain-decode baseline re-measured today before touching the code (build of 1946b36, graph decode, 3 runs, tok/s): ctx 37: 52.30-52.33, ctx 47: 52.28-52.32, 512: 52.10-52.14, 2048: 50.99-51.02, 4088: 48.80-50.19; prefill 512/2048/4088: 1010 / 1207 / 1136 tok/s (mean of 3). The plain rows in the table are from the MTP build and agree (the GDN kernel now has a token loop; plain arithmetic is unchanged). See the json for every run.

## Tensor-core prefill attention (H-D, p9/tcattn, 2026-11-20)
Kernel: `src/backends/cuda/attention/attn_seq_tc.cu` (WMMA 16x16x16 = Volta HMMA, fp16 inputs, fp32 accumulate and fp32 online softmax). One block = 8 queries x the 6 q heads of one kv head (48 rows) x 64-key tiles, K/V tile shared by all 6 heads and converted fp32 -> fp16 in smem (global loads register-prefetched one phase ahead), causal mask + `pos0` offset, gate/sigmoid epilogue and fp16 clamp unchanged. KV reads sit behind a `KvSrcF32` fetch/store pair so an int4-block source can replace it. `attn_seq()` dispatches to it; `QV_ATTN=fp32` selects the old CUDA-core kernel. No split over keys: with chunk 2048 the grid is 256 x 4 blocks, all equally long, enough for 80 SMs.
Attention-only, one layer, chunk T=2048, synthetic data (`tests/bench_attn_seq.cu`; TFLOP/s counts causal-useful FLOPs):
| pos0 (keys before chunk) | fp32 ms | TC ms | speedup | TC TFLOP/s |
|---:|---:|---:|---:|---:|
| 0 | 5.83 | 2.06 | 2.8x | 25.1 |
| 6144 | 40.2 | 12.4 | 3.2x | 29.0 |
| 14336 | 87.4 | 26.2 | 3.3x | 29.5 |
| 30720 | 189 | 54.1 | 3.5x | 29.5 |
| 63488 | 431 | 110 | 3.9x | 29.5 |
| 129024 | 834 | 240 | 3.5x | 27.3 |
| 260096 | 1676 | 498 | 3.4x | 26.4 |
Prefill attention for a whole prompt (16 layers, chunk 2048; per-chunk time interpolated between the measured points above, so extrapolated for 32K+ prompts, attention only): 16K 1.8 s (fp32 5.9 s), 32K 7.2 s (24.4 s), 64K 28.6 s (105.7 s), 128K 119 s (433 s), 256K 499 s = 8.3 min (fp32 1725 s = 28.7 min). That is 3.5x, not the ~5x/4 min the plan expected: the kernel reaches 27-29.5 TFLOP/s, about 23% of the 125 TFLOP/s HMMA peak.
End-to-end prefill tok/s (E0 model, wikitext prefix, 2 reps, GPU/CPU shared with other agents so +-3% noise): 4096 (chunk 2048) fp32 1101/1075 -> TC 1191/1171 (+9%); 8192 (chunk 2048) 909/913 -> 1040/1088 (+15%); 16384 (chunk 1024) 692/684 -> 888/870 (+28%). Chunk 2048 OOMs at 16K (fp32 KV 2.1 GiB + scratch) and 32K does not fit (fp32 KV 4 GiB) until int4 KV lands, so 32K end-to-end is NOT measured.
Correctness: `test_attn_seq` TC vs golden decode loop rel err 2.1e-4..4.1e-4 (fp16 out 2.9e-4..4.6e-4; tolerance 3e-3), TC vs fp32 kernel 4.2e-4 at pos0 63488 (synthetic). Full-model golden: step path KL 6.5e-4 (unchanged); prefill path KL 7.29e-5 (fp32 kernel 7.27e-5), chunk-23 prefill 7.46e-5 (fp32 7.23e-5), top-1 vs qsim 0.984 / 0.952.
What limits it (measured): removing the global K/V loads from the kernel (diagnostic build, zeros instead) gave 40 TFLOP/s vs 27, so ~30% is the fp32 KV read + convert through L2 (an fp16/int4 cache cuts this); the rest is the serial S -> softmax -> PV phases with one 8-warp block per SM (93 KB smem) and smem-bandwidth-bound WMMA fragment loads. Tried and slower: 12 warps one-tile-per-warp S (11.1 ms vs 7.95 at pos0 8192, T 1024), 96-row tile with 32-key tiles (8.7 ms, register spills). Not tried: raw mma.sync with register-resident S/P (FlashAttention-2 style) which should avoid the smem round trip.
Raw data: benchmarks/results/gpu_tcattn_2026-11-20.json.
