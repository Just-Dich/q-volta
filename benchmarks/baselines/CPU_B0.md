# CPU_B0 - llama.cpp CPU baseline, Qwen3.8-27B (measured 2026-11-20)

Engine: upstream llama.cpp commit 9d6f04fa99936708b9df140b7b2df62a8925afd5 (2026-11-20), arch `qwen35`
(hybrid Gated-DeltaNet + full attention), CPU-only. Host: 1 socket Xeon E5-2683 v4, 64 logical CPUs, AVX2.

## Commands
- Build: `cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j24 --target llama-bench llama-quantize llama-cli`
- Convert: `CUDA_VISIBLE_DEVICES= python3 convert_hf_to_gguf.py /workspace/q-volta-models/Qwen3.8-27B --outtype q8_0 --outfile scratch/q8.gguf` (29.0 GB, text model)
- Quantize: `llama-quantize --allow-requantize scratch/q8.gguf scratch/<Q>.gguf <Q>`
- Bench: `CUDA_VISIBLE_DEVICES= llama-bench -m <Q>.gguf -t <T> -p 512,2048 -n 64 -r 3 -o md`
  (each thread count a separate invocation; scripts run2.sh/run3.sh; raw output in results.log)

## CAVEATS (read first)
- Quants were made from a Q8_0 GGUF (slightly lossy), NOT from BF16 (disk limit). Quality (KL) of these
  files is therefore not exactly that of a BF16->Qx quant; quality was not measured here.
- No imatrix was used (plain llama-quantize).
- Before each run the script waited for 1-min loadavg < 8 (all runs started at 7.5-7.9). But llama-bench's own
  threads raise loadavg (32 threads => ~32), and the "load_after" value mostly reflects this benchmark itself.
  ps showed a concurrent Q-Volta qv_run_cpu job for parts of the session; a first attempt (Q4_K_M, 16-32
  threads) overlapped with it at loadavg ~34 and was aborted/discarded (aborted_contended_run.log).
  During the t=32 runs loadavg was 26-32 (own 32 threads + possibly background); treat t=32 pp2048 and tg64
  as possibly slightly noisy (t=32 tg64 2.01 < t=16 tg64 2.16 may be contention or thread oversubscription
  of memory-bound decode, not resolved).
- Not run (time): IQ4_XS, IQ3_S, Q3_K_M at 16/24 threads. Threads other than 16/24/32 not tried.
- bits/weight = file GiB * 2^30 * 8 / 27.32e9 params (llama-bench param count, includes embeddings/output).

## Results (llama-bench, r=3, mean +- sd, tok/s)
| Quant | GiB | bpw | Threads | tg64 | pp512 | pp2048 | loadavg before -> after (1m) |
|---|---|---|---|---|---|---|---|
| Q4_K_M | 15.83 | 4.98 | 16 | 2.16 +-0.01 | 13.67 +-0.51 | 13.17 +-0.10 | 7.84 -> 15.5 |
| Q4_K_M | 15.83 | 4.98 | 24 | 2.14 +-0.03 | 19.21 +-0.28 | 18.66 +-0.04 | 7.96 -> 23.6 |
| Q4_K_M | 15.83 | 4.98 | 32 | 2.01 +-0.02 | 23.47 +-0.59 | 20.09 +-0.33 | 7.53 -> 31.9 (noisy?) |
| Q3_K_M | 12.76 | 4.01 | 32 | 1.86 +-0.01 | 22.97 +-0.48 | 19.87 +-0.43 | 7.89 -> 31.95 |

Best per quant: Q4_K_M decode 2.16 tok/s @16 threads, prefill 23.5 (pp512) / 20.1 (pp2048) @32 threads.
Q3_K_M @32: decode 1.86, prefill 23.0 / 19.9.

## Observations
- llama.cpp CPU decode on this box is ~2 tok/s (about 34 GB/s effective weight read for Q4_K_M, below the
  54 GB/s ceiling); Q3_K_M is slower than Q4_K_M despite being smaller (3-bit K-quant dequant cost).
- Prefill is compute-bound and scales with threads (13.7 -> 19.2 -> 23.5 tok/s at 16/24/32), ~20 tok/s at
  pp2048 for both quants.
- Comparison to Q-Volta CPU (3.93 tok/s decode @32 threads, 12.63 GiB) is left to the owner.

Intermediates (q8.gguf, quant files) were deleted; /workspace free space back to 52 GB.

---
# Q-Volta CPU vs llama.cpp — summary written by the owner (2026-11-20)

Q-Volta CPU (qv_run_cpu, E0 weights 12.63 GiB, 32 threads, `--bench --prefill`, 48 decode tokens), run right after the load
average had decayed (1-min load 5.8 at start; it rose to ~32 from the benchmark's own spinning threads):
| prompt tokens | prefill tok/s | decode tok/s (after that prompt) |
|---:|---:|---:|
| 128 | 27.2 | 3.27 |
| 512 | 26.0 | 2.89 |
| 2048 | 20.8 | 2.77 |
(An earlier, more contended run of the same engine gave decode 3.93 at ~4K context start; decode varies 2.8-3.9 with load and context.)

llama.cpp (table above, Q3_K_M 12.76 GiB, 32 threads): decode 1.86, pp512 22.97, pp2048 19.87 — measured at load ~26-32.

Reading (honest): decode Q-Volta ~1.5-2x llama.cpp; prefill roughly parity (+13% at 512, +4% at 2048, inside the noise of
these runs). The two engines were NOT measured under identical load, quality (KL to BF16) was not compared, and llama.cpp quants
came from a Q8_0 intermediate. The quant files were deleted; a clean re-measure needs regenerating the GGUF (~30 GB scratch).
