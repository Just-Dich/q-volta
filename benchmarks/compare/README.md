# Q-Volta vs llama.cpp on the same Tesla V100 (Qwen3.8-27B), 2026-11-21..23

Purpose: a fair, reproducible comparison at roughly equal bytes, with quality measured on one common axis. **This document does not claim that either engine is "best"**; see the verdict and the "not comparable" list.

## Setup
- GPU: Tesla V100-SXM2-16GB, device 0 only (`CUDA_VISIBLE_DEVICES=0`, the P4 is ignored), SM clock 1530 MHz during the timed runs (`raw/clocks_*.txt`, one `clocks` field per row in `raw/speed_*.jsonl`). Every GPU command ran under `flock /tmp/qv_bench.lock`, 3 rounds, interleaved (each round runs every cell once). The host is shared with other jobs (the lock serializes GPU use, CPU load is not controlled).
- llama.cpp: commit 9d6f04f (arch `qwen35`), CUDA build `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`, `-fa 1 -ngl 99`, f16 KV unless stated. llama-bench `-r 1` per round (3 rounds = 3 samples).
- Q-Volta: this repo (`qv_run --bench --prefill --graph --n 129 --max-ctx 4608`, MTP depth 2/3 with `--mtp --mtp-depth N`), models `qwen38-27b-e0.qv` (RTN g64, 12.63 GiB) and `qwen38-27b-e1.qv` (GPTQ g128 meta8, 10.44 GiB).
- Prompts: `data/code.i32`, `data/prose.i32` (chat prompts, 37/47 tokens), `data/P512.i32`, `P2048.i32`, `P4088.i32` (wikitext-2 test prefixes). The same token ids are fed to both engines (llama-server receives token arrays). 128 new tokens, greedy.
- Quality: wikitext-2 test, the same 4088-token window as Q-Volta's earlier evaluations, plus the 8x512-token protocol of `tools/convert.py` (BF16 ppl 8.4676).

## Common reference (explicit choice)
Reference logits = **llama.cpp Q8_0 GGUF** (converted from the BF16 safetensors, 29 GB, includes the MTP head), evaluated on the 4088-token window with `llama_eval --mode write` into the same file format as `qv_run --eval-mode write` (top-1024 log-probs + tail mass). Both engines are then compared against that one file with the same formula (`llama_eval --mode compare`, `qv_run --eval-mode cmp`): KL(ref||q) over the reference top-1024 plus a tail term, top-1 agreement. 
Q8_0 is a **near-lossless proxy, not BF16**: on the 8x512 protocol it gives ppl 8.467 vs BF16 8.468 (differences in KL vs true BF16 were not measured; BF16 logits for this window were not produced here). KL numbers below are therefore "KL to Q8_0", not "KL to BF16", and are not interchangeable with the KL column of docs/QUANTIZATION.md (that one is vs BF16 on the 8x512 set).
ppl on the single 4088 window is ~6.0 and not comparable to the 8x512 ppl (~8.5-9.5); both are reported.

## Results (equal-bytes pairs)
Sizes are file sizes (GiB = 2^30). llama.cpp quants were requantized from Q8_0 (`--allow-requantize`), not from BF16, so they carry the Q8_0 rounding on top.

| model | file GiB | imatrix | ppl 8x512 (BF16 8.468) | ppl 4088 (Q8_0 5.774) | KL to Q8_0 | top-1 vs Q8_0 |
|---|---:|---|---:|---:|---:|---:|
| llama.cpp Q8_0 (reference) | ~27.0 (29.0 GB) | no | 8.467 | 5.774 | 0 | 100% |
| llama.cpp Q3_K_M | 12.77 | no | 8.905 (+5.2%) | 6.047 | 0.0535 | 89.7% |
| **Q-Volta e0** (RTN g64) | 12.63 | no (RTN) | 9.511 (+12.3%) | 6.257 | 0.1018 | 86.6% |
| llama.cpp IQ3_XXS (+ imatrix) | 10.89 | yes (Q-Volta GPTQ calibration text) | 8.992 (+6.2%) | 6.102 | 0.0751 | 89.0% |
| **Q-Volta e1** (GPTQ g128) | 10.44 | GPTQ Hessian, same calibration set | 9.278 (+9.6%) | 6.281 | 0.0923 | 87.1% |
| **Q-Volta e1b1** (p21/quant2: act-order GPTQ, per-class damping, 256 calib seqs, g64 on down/o_proj, GPTQ 4-bit head) | 10.40 | GPTQ Hessian, 256 sequences (same text families) | 9.112 (+7.6%) | 6.102 | 0.0750 | 89.3% |

Pairs are not exactly equal: Q3_K_M is 1.1% larger than e0, IQ3_XXS is 4.3% larger than e1 (the nearest llama.cpp types that exist; IQ3_S was not built, see "missing"). The imatrix was computed with `llama-imatrix` on the Q8_0 model over the text decoded from `/workspace/q-volta-models/calib/calib32.npy` (the 32x1024-token set our GPTQ uses; re-tokenization shifts window boundaries slightly). For IQ3_XXS the MTP layer (blk.64, absent from the imatrix) was forced to Q4_K. Q3_K_M was built without imatrix; **without-imatrix IQ3_XXS and with-imatrix Q3_K_M were not run** (time/disk).

### Speed, V100, tok/s (mean of 3 rounds; spread in `results.json`)
Plain decode (128 new tokens, greedy) and prefill:

| model | llama-bench tg128 (depth 0 / 4088) | llama-server decode code / prose / wiki4088 | llama-bench pp512 / pp2048 / pp4096 | llama-server prefill, 4088-token prompt |
|---|---|---|---|---|
| llama.cpp Q3_K_M (12.77 GiB) | 42.3 / 40.7 | 41.0 / 39.8 / 37.7 | 930 / 931 / 920 | 829 |
| Q-Volta e0 (12.63 GiB) | - | 52.2 / 52.2 / 49.9 | (qv_run) 1008 / 1250 / 1223 (wiki 512/2048/4088) | - |
| llama.cpp IQ3_XXS+im (10.89 GiB) | 38.7 / 36.7 | 35.5 / 34.4 / 32.6 | 733 / 779 / 775 | 649 |
| Q-Volta e1 (10.44 GiB) | - | 57.6 / 57.6 / 54.7 | (qv_run) 1011 / 1250 / 1220 | - |

MTP speculation (both engines: llama.cpp `--spec-type draft-mtp --spec-draft-n-max N` in llama-server, supported for this architecture in this commit, MTP layer loaded from the Q8_0-derived GGUF; Q-Volta `--mtp --mtp-depth N`). llama.cpp's speculative output was **not** checked for equality with its plain greedy output; Q-Volta's is bit-identical to plain greedy (repo tests). Decode tok/s, best of depth 2/3 shown as depth(value):

| prompt | llama.cpp Q3_K_M MTP | Q-Volta e0 MTP | llama.cpp IQ3_XXS+im MTP | Q-Volta e1 MTP |
|---|---|---|---|---|
| code (37 tok) | 3: 74.8 (accept 0.93) | 3: 131.7 (0.94) | 3: 68.8 (0.88) | 3: 135.9 (0.94) |
| prose (47 tok) | 2: 56.9 (0.68) | 2: 86.5 (0.58) | 2: 52.6 (0.66) | 2: 93.8 (0.61) |
| wiki 512 | 2: 58.3 (0.74) | 3: 134.4 (0.96) | 3: 61.9 (0.85) | 3: 130.5 (0.89) |
| wiki 2048 | 2: 50.5 (0.59) | 2: 76.3 (0.52) | 2: 43.1 (0.52) | 2: 82.4 (0.54) |
| wiki 4088 | 2: 51.6 (0.65) | 2: 78.8 (0.59) | 2: 49.4 (0.73) | 2: 93.9 (0.73) |

Acceptance is a different statistic in the two engines (llama.cpp: accepted/drafted per its server timings; Q-Volta: accepted/drafted in its MTP stats), so the acceptance column is indicative only. Note that llama-server decode with no speculation (41.0 on code) is below llama-bench tg128 (42.3): server overhead; llama-bench is the cleaner llama.cpp plain number.

### Max context on the 16 GB V100 (nothing else resident)
Bisection to 1 K tokens; a size "fits" if the context allocates and one 2048-token prefill + decode step runs (llama.cpp, `llama_eval --probe-ctx`, flash-attn, ubatch 512) / a synthetic-KV decode at position N-8 runs (Q-Volta `--decode-bench --synth-ctx`, prefill chunk 512, no MTP). Speed and memory only: no quality claim at these lengths (Q-Volta int4 KV was quality-checked to 32K, passkey/ppl, see docs/LONG_CONTEXT.md; llama.cpp q8_0/q4_0 KV quality was not measured here).

| weights | KV cache | max ctx |
|---|---|---:|
| llama.cpp Q3_K_M 12.77 GiB | f16 / q8_0 / q4_0 | 39,936 / 74,752 / 132,096 |
| Q-Volta e0 12.63 GiB | fp16 / int4 (4.5 bit) | 38,912 / 140,288 |
| llama.cpp IQ3_XXS+im 10.89 GiB | f16 / q8_0 / q4_0 | 70,656 / 129,024 / 219,136 |
| Q-Volta e1 10.44 GiB | fp16 / int4 | 74,752 / 267,264 |

llama.cpp Q4_K_M (15.83 GiB) does not fit: the V100 exposes 15.77 GiB (arithmetic, not run on GPU; partial offload was not tried). Q-Volta's earlier 256K evidence (real 250K prefill, passkey 1/1) is in docs/LONG_CONTEXT.md.

### p21/quant2 update (2026-11-23): e1b1 at 10.40 GiB
e1b1 (recipe in docs/QUANTIZATION.md) measured with the same scripts (`quality_qv.sh`, qv_run 8x512 + 4088 cmp vs ref_q8_P4088.bin). Against IQ3_XXS+imatrix (10.89 GiB, 4.7% larger): KL to Q8_0 0.0750 vs 0.0751 (equal), ppl 4088 6.102 vs 6.102 (equal), top-1 89.3% vs 89.0%, but ppl 8x512 9.112 vs 8.992 (llama.cpp still 0.12 better, +7.6% vs +6.2%). Against e1: ppl 9.278 -> 9.112, KL 0.0923 -> 0.0750. Decode (qv_run --graph, 3 interleaved rounds, flock): plain 57.5 vs e1 57.6 tok/s, MTP-3 code 137.2 vs 138.5, prose 92.2 vs 88.4. Agentic set (convert.py eval, BF16 reference): KL 1.370 vs e1 1.34 (not better, slightly worse than e1; E0 RTN was 1.13). The 12.7 GiB target (e0b1) and a GPTQ-E0-bits row in this protocol were NOT run (disk, time).

## Verdict (what the numbers support)
- **Speed, plain decode**: Q-Volta is faster at the same file size: e0 52.2 vs llama.cpp Q3_K_M 42.3 (+23%), e1 57.6 vs IQ3_XXS 38.7 (+49%, with e1 4% smaller). Part of this is bytes read; IQ-quant dequantization on Volta also costs llama.cpp (IQ3_XXS is slower than Q3_K_M despite being smaller).
- **Speed, MTP**: both engines have MTP for this model; Q-Volta reaches 1.5-2.3x of llama.cpp's MTP tok/s on these prompts (e.g. code 131-136 vs 69-75), with caveats above (different acceptance accounting; llama.cpp output equality unchecked).
- **Prefill**: Q-Volta 1250 (2048-token) / 1220 (4088) tok/s vs llama.cpp Q3_K_M 931 / 920 (llama-bench) and IQ3_XXS 779 / 775. At 512 tokens Q-Volta ~1010 vs 930/733. Q-Volta's lead over llama.cpp is 1.1-1.6x depending on quant.
- **Quality at ~equal bytes: llama.cpp wins.** Q3_K_M (12.77 GiB) is clearly better than Q-Volta e0 (12.63 GiB): +5.2% vs +12.3% ppl, KL to Q8_0 0.054 vs 0.102. IQ3_XXS+imatrix (10.89 GiB, 4% larger) is better than e1 (10.44 GiB): +6.2% vs +9.6% ppl, KL 0.075 vs 0.092. A tok/s win at worse quality is not a clean win (PLAN rule); Q-Volta's quantization is the weak part, the engine is the fast part. A bytes-matched quality comparison for e1 would need an llama.cpp type at ~10.4 GiB (e.g. IQ3_XXS with a few tensors lowered), not built.
- **Max context**: Q-Volta's int4 KV allows longer contexts than llama.cpp's q4_0 KV at the same weights (140K vs 132K on the ~12.7 GiB weights, 267K vs 219K on the ~10.5-10.9 GiB weights), and fp16 KV is similar (39K vs 39K; 75K vs 71K). The gap is small at 12.7 GiB and largest where Q-Volta's weights are smaller; llama.cpp's compute buffers and unwritten-KV vs allocated-KV semantics differ, so treat as +-5%.
- **Not comparable / caveats**: (1) KL reference is Q8_0, not BF16; (2) llama.cpp quants come from Q8_0, Q-Volta from BF16; (3) file sizes differ by 1-4%; (4) MTP acceptance and exactness differ as noted; (5) llama.cpp ppl uses its own tokenization-free path via `llama_eval` (our harness, not `llama-perplexity`), validated by Q8_0 8.467 vs BF16 8.468 and Q-Volta e0 reproducing 9.511 vs the documented 9.508; (6) the CPU backend was not part of this stand (see README CPU table); (7) one machine, shared host.

## Other engines on sm_70 (source/wheel inspection only, not run; time-boxed)
- **ExLlamaV3 1.6.7**: the EXL3 kernels use `mma.sync` m16n8k8/m16n8k16; its lowest supported architecture is sm_75 (Turing, "EXL3_SM75" path, `arch.cuh`); Volta has no m16n8k8 tensor-core instruction. **Not runnable on V100: kernels require sm_75+** (inferred from source; not compiled here).
- **ExLlamaV2 0.3.2**: no Qwen3.5/Qwen3-Next (gated delta-net) architecture in `architecture.py` (only Qwen3 / Qwen3-MoE). **Not runnable for this model: architecture unsupported.**
- **vLLM 0.34.0 / SGLang 0.5.23**: have Qwen3.5 model code (+ MTP); vLLM's platform layer allows fp16 on Volta and its fused GDN decode kernel needs sm_80+ with a fallback; no 4-bit checkpoint of this 27B fits a 16 GB card with usable context (AWQ/GPTQ INT4 ~15-16 GB plus fp16 non-quantized parts) and the pinned torch/CUDA wheels (~GBs, disk was 20-40 GB free) were not installed. **Not run: no feasible fit on 16 GB and no install attempted; support on sm_70 unverified.**

## Missing (honest)
IQ3_S and Q3_K_S/IQ3_XXS without imatrix, Q3_K_M with imatrix, llama.cpp quants made from BF16, llama.cpp KV-quant quality, BF16-based KL for llama.cpp quants, Q-Volta pi-adapter, repeated max-ctx runs.

## Reproduce (paths in `scripts/env.sh`)
```
scripts/build_all.sh                       # llama.cpp CUDA sm_70 (+nvml.h workaround), llama_eval, qv_run
scripts/convert_q8.sh                      # BF16 -> Q8_0 GGUF (4 shards; used scripts/ramfile.py to keep 2 shards in RAM, disk was tight)
flock /tmp/qv_bench.lock $LB/llama_eval -m q8.gguf --tokens data/P4088.i32 --mode write --ref ref_q8_P4088.bin -ngl 30 -t 32
scripts/quantize.sh Q3_K_M                 # llama-quantize --allow-requantize
scripts/make_imatrix_text.py calib.txt && scripts/imatrix.sh && scripts/quantize.sh IQ3_XXS imatrix.gguf
scripts/quality_llama.sh MODEL.gguf TAG    # ppl 4088 + KL vs Q8_0, ppl 8x512
scripts/quality_qv.sh MODEL.qv TAG
flock /tmp/qv_bench.lock scripts/block_all_gguf.sh TAG MODEL.gguf MODEL.qv   # quality, speed.py (llama-bench, llama-server +/- MTP, qv_run), llama max ctx
flock /tmp/qv_bench.lock scripts/block_maxctx_qv.sh LABEL MODEL.qv
python3 -I scripts/collect.py              # raw/ -> results.json
```
`llama_eval.cpp` is our small libllama harness (teacher-forced, same file format and KL formula as `qv_run --eval`). Raw outputs: `raw/` (JSON/JSONL, stderr logs, imatrix log); aggregated: `results.json`. Note: in `raw/speed_Q3KM_vs_e0.jsonl` and the IQ3 file the Q-Volta MTP depth was recovered from run order by `collect.py` (a field overwrite in the first version of `speed.py`, fixed afterwards).
