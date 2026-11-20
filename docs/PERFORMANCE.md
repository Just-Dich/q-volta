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
Default engine path: fp32 KV (128 KiB/token). Measured max_ctx that builds and decodes (V100, plain decode, prefill chunk 512): fp32 18,432 (20,480 OOM), fp16 38,912 (40,960 OOM), int4 139,264 (143,360 OOM); see the int4 KV section below. 256K needs 4-bit KV AND weights+scratch <= ~11.3 GiB (now ~13.3 incl. CUDA context): not reachable on this card yet; no claim made beyond the lengths tested in the int4 KV section.

## Known limitations / not done
split-K long-context decode attention is done (see below); int4 KV is done (see the int4 KV section; GPU only, no host tier yet); tensor-core attention, quality improvements (GPTQ/bit allocation), tokenizer/server/CLI, CPU backend (gated until the GPU completion gate).

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

## int4 KV cache (2026-11-20, branch p9/kv4; hypothesis H-A of PLAN.md Gen 6)
`--kv fp32|fp16|int4` (qv_run and qv_server; `EngineOptions::kv_mode` 0/1/2, `kv4_kmode`). fp32 is the unchanged default path. fp16 = the same code path as int4 with a full-length fp16 ring and no quantization (64 KiB/token; used as the exact-ish baseline where fp32 does not fit). Design and byte layout: src/backends/cuda/attention/kv4.h.
Design: KV of the 16 attention layers (4 kv heads x head_dim 256) is stored in self-contained 64-token block records of 18,432 B (K int4 + K scale/zero + V int4 + V scale/zero; scale/zero fp16 half2, asymmetric min/max, 16 levels) = 4.5 bit/element = 288 B/token/head = 18 KiB/token for all 16 layers (fp32: 128 KiB, fp16: 64 KiB). The most recent 128..191 tokens stay un-quantized in an fp16 ring (256 slots per head); block b is quantized when position 64(b+3)-1 is written, so every path (prefill, 1-token decode, MTP verify) applies the same rule "keys [0, 64*floor((n-128)/64)) quantized, the rest fp16" and sees the same bits. K quantization has two modes (`--kv4-kmode channel|token`): per-channel (KIVI style: scale/zero per channel over the 64 tokens of a block) and per-token (groups of 64 channels per token); V is always per-token groups of 64 channels. All block addressing goes through `kv4_rec()` and a record is a contiguous 18 KiB unit, so a later host-RAM tier can move cold blocks without touching the kernels. Kernels: kv4.cu (block quantize, ring commit), attn_kv4.cu (split-K decode: dequantizes in registers while streaming, same online-softmax/combine structure as the fp32 kernel, splits own whole 64-key blocks), attn_seq.cu (`attn_seq_kv4_kernel`, prefill: dequantizes tiles into shared memory, handles windows that straddle a tile with a per-row mask); the prefill chunk's post-RoPE K/V are staged in fp16 (2 MiB each at chunk 1024) and quantized after the chunk's prep.
Correctness: test_attn_kv4 (new, ctest) - int4 prefill kernel vs fp32 path (random data with outlier K channels: rel L2 0.2e-3..2.5e-2 per-channel, up to 3.7e-2 per-token; fp16 mode 1e-4), decode kernel vs prefill kernel on the same int4 cache 3e-7..1e-6, multi-chunk prefill (chunks of 193 / 64 / rest) identical to single chunk (0.0), nt=3 MTP-verify launch bit-identical to three 1-row launches. test_cuda_mtp with int4 KV (depth 1, 3; contexts 14, 37, 120 and 4088 tokens, 96 generated tokens each, crossing block-quantization boundaries): ids AND final logits identical to plain greedy on the same int4 cache in every case; also per-token-K mode (depth 3). The fp32 path is unchanged (ctest attn_seq, attn_decode_split, gdn_seq, dequant, gemv_multi pass).
### Quality (wikitext-2 test text, first 4088 / 8192 real tokens, prefill chunk 2048, E0 model; reference = same engine with fp32 KV)
| context | KV | ppl | ppl vs fp32 KV | KL(fp32-KV || KV) over top-1024 + tail | top-1 agree |
|---|---|---:|---:|---:|---:|
| 4088 | fp32 | 6.2567 | - | - | - |
| 4088 | fp16 | 6.2569 | +0.003% | 8e-6 | 99.88% |
| 4088 | int4, K per-channel | 6.2537 | -0.048% | 8.3e-4 | 98.65% |
| 4088 | int4, K per-token | 6.2605 | +0.060% | 1.22e-3 | 98.48% |
| 8192 | fp32 | 6.8793 | - | - | - |
| 8192 | fp16 | 6.8792 | -0.002% | 8e-6 | 99.82% |
| 8192 | int4, K per-channel | 6.8807 | +0.020% | 7.9e-4 | 98.75% |
| 8192 | int4, K per-token | 6.8853 | +0.087% | 1.31e-3 | 98.39% |
(Single run each, deterministic. KL: mean over positions of KL(P_ref||P_kv) computed on the reference's top-1024 tokens plus one lumped tail bucket (mean reference tail mass 0.8%), so it is a close approximation, not the full-vocabulary KL. ppl differences of +-0.05% are within what a different but equally valid quantization noise realisation would give; the robust signal is KL ~8e-4 (100x the fp16 floor) and 1.2-1.5% top-1 flips. Per-channel K is better than per-token K on both metrics (KL 8e-4 vs 1.2-1.3e-3) and is the default.) The ppl of 6.26/6.88 is for these text prefixes and is not comparable to the 9.5 of the 4088-token eval above (different text window; this eval is also teacher-forced with the KV of the *quantized-weights* model on both sides, so it isolates the KV effect only). H-A falsification threshold (ppl +>3%): not met (max +0.09%).
### Passkey retrieval (tools/niah.py; real token lengths via qv_tok; chat prompt, thinking off, greedy 16 tokens; 5-digit key hidden in wikitext filler at depth 0.1/0.5/0.9, 2 random keys each = 6 prompts per length)
| length (tokens) | fp32 KV | fp16 KV | int4 KV |
|---|---:|---:|---:|
| 4,096 | 6/6 | 6/6 | 6/6 |
| 8,192 | 6/6 | 6/6 | 6/6 |
| 16,384 | 6/6 | 6/6 | 6/6 |
| 32,768 | does not fit | 6/6 | 6/6 |
Harder variant (tools/niah.py --multi 6: six "The secret code for NAME is NNNNN." needles spread over the document, ask for one of them, so the answer must be bound to the right name; 6 prompts per length, the asked needle rotates through the document): int4 6/6 at 8K, 16K and 32K; fp16 6/6 at 8K, 16K and 32K (fp32 not run on this variant).
Caveats: single-passkey retrieval is an easy task (saturated at 100% for all three KV formats, so it can only detect a gross failure, not mild degradation); an earlier version of the generator that asked for a completion without the chat template produced <think> openings that were scored as misses for fp32 as well as int4 (format artefact, replaced by the chat prompt above). Only lengths <= 32K are tested; nothing is claimed above that. H-A falsification (NIAH drops at <= 32K): not met.
### Speed (qv_run --decode-bench: prefill the real wikitext prefix, untimed 1-token warm-up that captures the CUDA graph, then 129 timed graph-decode tokens; V100 device 0, `flock` + idle check per run, prefill chunk 1024)
| context at decode start | fp32 KV tok/s | fp16 KV tok/s | int4 KV tok/s | int4 vs fp32 | VRAM fp32 / fp16 / int4 (GiB, nvidia-smi incl. CUDA context) |
|---:|---:|---:|---:|---:|---|
| 512 | 52.03 [51.89-52.10] | 51.88 [51.87-51.88] | 51.70 [51.69-51.71] | -0.6% | 13.63 / 13.59 / 13.57 |
| 4,096 | 50.01 [50.00-50.01] | 50.36 [50.34-50.37] | 50.50 [50.37-50.59] | +1.0% | 14.06 / 13.80 / 13.63 |
| 8,192 | 46.81 [45.85-48.17] | 47.46 [45.77-49.02] | 47.87 [46.14-49.26] | +2.3% | 14.56 / 14.05 / 13.70 |
| 16,384 | 39.66 [36.45-42.87] | 42.63 [39.35-45.91] | 42.54 [38.41-46.67] | +7% | 15.56 / 14.55 / 13.84 |
| 32,768 | does not fit | 35.98 [33.56-38.41] | 38.35 [36.63-40.07] | (+6.6% vs fp16) | - / 15.55 / 14.12 |
(mean [min-max] of 3 runs at <=8K and 2 runs at 16K/32K; the GPU/CPU were shared with other jobs (all timed runs held the lock, but non-locking load exists) and the spread at >=8K is mostly drift between rounds (all three formats drop together), not between formats: in all 10 rounds at >=4K int4 was >= fp32, and int4 >= fp16 in 9 of 10 (not at 16K round 2: 38.41 vs 39.35). Differences of 1-3% are weak evidence; the robust statement is "int4 is not slower than fp32/fp16 at any context (512: -0.6%) and uses 0.5-1.4 GiB less VRAM at 16K".) Why the speed-up is small although int4 reads 3.6-7x fewer bytes: the split-K attention kernel appears to be instruction/latency bound rather than bandwidth bound. This is inferred, not profiled: the end-to-end cost of 32K context over 512 context is ~5.6 ms/token for int4 (24.96 vs 19.34 ms), i.e. ~350 us per attention layer, the same as the 348 us measured for the fp32 kernel at 32K (test_attn_decode_split), while the int4 data of one layer at 32K (37.7 MB) would take ~47 us at 800 GB/s. A kernel with a cheaper per-key reduction (the 6-head shuffle reduce) could therefore turn the byte saving into speed; not done.
Prefill: 8K prefill 9.3 s int4 vs 9.0 s fp32 (+3%, single runs, chunk 2048); in the NIAH runs (chunk 1024, mean of 6) 4K 4.5 / 4.2 / 4.8 s, 8K 10.4 / 10.0 / 10.8, 16K 25.3 / 24.8 / 25.8, 32K 67.1 / 67.6 / - s for int4 / fp16 / fp32 (i.e. 32K prefill ~490 tok/s; the attention share is not profiled here, CUDA-core attention is the suspected cause and tensor-core attention is hypothesis H-D).
MTP depth 3 with int4 KV (wikitext prefixes, base-model continuation, 129 tokens, 1 run each): 512: 133.1 tok/s (acceptance 0.962) vs fp32 129.2 (0.926); 4,096: 89.8 (0.608) vs 85.1 (0.589); 16,384: 71.4 (0.563); 32,768: 44.6 (0.375). Output ids identical to the plain int4 decode of the same prompt in all four cases (compared over the 122 tokens both runs cover); fp32 MTP at 16K does not fit (snapshots + KV). Acceptance differs slightly between KV formats because the model's own outputs differ slightly.
### Max context and VRAM
Measured (qv_run --decode-bench --synth-ctx: engine built with max_ctx N, 512-token prefill at chunk 512, then 33 graph-decode tokens at position ~N-300 over UNWRITTEN KV, so speed/VRAM are measured but the content is not meaningful; plain decode, no MTP):
| KV | largest max_ctx that fits | first size that OOMs | decode tok/s at that position | nvidia-smi used |
|---|---:|---:|---:|---:|
| fp32 | 18,432 | 20,480 | 44.85 | 15.58 GiB |
| fp16 | 38,912 | 40,960 | 41.94 | 15.71 GiB |
| int4 | 139,264 | 143,360 (147,456 and 155,648 also) | 30.28 | 15.74 GiB |
(int4 also: 65,536 -> 38.9 tok/s, 98,304 -> 34.4, 131,072 -> 31.2; sizes tested in 2K-8K steps, so the true limit lies inside the gap; MTP needs 0.15 GiB more per depth step.) Quality was verified only up to 32K (above); lengths 33K-139K fit but are NOT validated. The int4 limit is 7.6x the fp32 one and 3.6x the fp16 one (expected ~100K+; measured 139K).
256K requirement (arithmetic, not a measurement): int4 KV = 18,432 B/token x 262,144 = 4.5 GiB (+0.28 GiB for the MTP layer's KV), fp16 KV would be 16 GiB. The V100 exposes 15.77 GiB; everything except KV currently takes ~13.35 GiB (weights 12.63 + scratch/prefill buffers + CUDA context, taken from the 65,536-token probe: 14.48 used - 1.13 KV), so 256K needs ~17.85 GiB: ~2.1 GiB too much. Options: weights+scratch <= ~11.3 GiB (about -2.1 GiB of weights), or keep cold int4 blocks in host RAM (record layout is ready for that).
Reproduce: `qv_run --model M --max-ctx N --eval wiki.i32 --eval-mode write|cmp --eval-ref ref.bin --kv fp32|fp16|int4 [--kv4-kmode channel|token]` (ppl/KL), `python3 -I tools/niah.py gen ...` + `qv_run --gen --prompt a.i32,b.i32 --n 16 --kv ...` + `niah.py score` (passkey), `qv_run --decode-bench --prompt P.i32 [--synth-ctx N] [--mtp --mtp-depth 3] --kv ...` (speed / VRAM / max ctx), `QV_TEST_KV=1 QV_TEST_MAXCTX=4608 test_cuda_mtp MODEL 96 1,3 prompts...` (MTP exactness on the int4 cache). Raw data: benchmarks/results/gpu_kv4_2026-11-20.json.
