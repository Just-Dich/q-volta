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
| Q-Volta E0 | MTP | not implemented | |
Bytes-normalized plain decode: Q-Volta 52.2 x 12.63 = 659 GiB/s effective vs NInfer 45.3 x 13.99 = 634 GiB/s (+4%). Most of the raw +15% comes from reading fewer bytes (coarser quantization), not from a faster kernel. NInfer artifact may contain tensors not read per token, so treat +4% as an upper-bound-ish estimate until verified.

## Prefill (cold, no prefix cache, identical wikitext prompts)
| prompt tokens | Q-Volta | NInfer | Q-Volta vs NInfer |
|---:|---:|---:|---:|
| 512 | 1013 | 837 | +21% |
| 2048 | 971 | 1010 | -4% |
| 4088 | 917 | 1013 | -9.5% |
Path: dequantize 32-MiB weight slabs to fp16 -> cuBLAS tensor-core HGEMM; GDN recurrence in-kernel with register-resident state; fp32 CUDA-core flash-style attention. Reps agree within 1%.
Where the loss comes from: attention (8.4 TFLOP/s on CUDA cores) grows with context; NInfer is flat. Next: tensor-core attention, overlap dequant with GEMM.
History: step-by-step prefill 51 tok/s -> batched 680 -> aligned slabs 1014 (512 tokens). (An earlier "25.6 tok/s" figure was a benchmark-timer artifact in --graph mode; fixed.)

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
MTP, split-K long-context decode attention (decode drops 52 -> 36 tok/s after a 4K prompt), KV quantization, tensor-core attention, quality improvements (GPTQ/bit allocation), tokenizer/server/CLI, CPU backend (gated until the GPU completion gate).
