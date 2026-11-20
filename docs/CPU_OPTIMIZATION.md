# CPU backend: measurements and design constraints

Host used for all numbers here: Intel Xeon E5-2683 v4 (Broadwell-EP, AVX2+FMA+F16C, no AVX-512/VNNI), 64 logical CPUs
(likely 32 cores x 2 threads), 1 NUMA node, 16 MiB L3, DDR4, 76 GiB RAM. Measured by `benchmarks/bench_host.cpp`
(branch p3/cpuattn, commit 9deb469). Load average ~0.5 during runs. Random data, not real activations.

## Memory bandwidth (the decode ceiling)
| threads | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| sustained read GB/s (1 GiB) | 7.3 | 27.7 | 44.8 | 54.8 | 54.2 | 48.8 |

Ceiling ~55 GB/s, reached at 16-32 threads; 64 threads is not better. Decode reads every weight once per token, so with
the 12.63 GiB (13.56 GB) E0 model the bandwidth-only upper bound is ~4.0 tok/s (calculation, not a measurement of our engine).
Only fewer bytes per token (lower bits) or several tokens per weight pass (MTP / speculative) raise it.

## PCIe 3.0 x16 to the V100
H2D 11.9-12.5 GB/s, D2H 12.6-13.2 GB/s (pinned, 1-256 MB). Round trip of a 6 KB vector: 14.4 us; 25 KB: 17.6 us.

## Int4 KV attention on CPU (200K tokens, 230 MB KV per layer)
| threads | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| ms per layer | 170-180 | 26 | 14-15 | 8.4 | 6.6-8.9 |

ALU-bound per thread (~190-210 ns per token per kv head), flattens at 27-35 GB/s. 16 layers => 106-134 ms/token
(~9 tok/s from attention alone) at 200K context. Conclusion: cold KV on CPU is not a viable main path for GPU decode
(26x slower than GPU-resident int4 KV); CPU attention is only a candidate for the CPU-only backend or as a small side share.

## Int4 GEMV on CPU (32768x8192 matrix, 151 MB)
1 / 16 / 32 / 64 threads: 3.3 / 31-37 / 54.7 / 51-57 GB/s, i.e. the GEMV can saturate RAM bandwidth. Format used was the
group-64 layout with a 36 B group, not the real Q-Volta bit-plane layout (that kernel is being written, branch p4/cpugemv).

## Consequences for the plan
- CPU decode target: approach 55 GB/s of weight reads => ~4 tok/s with E0 weights; use 16-32 pinned threads, not 64.
- GDN state traffic is small: 3 MiB per layer (48x128x128 fp32), 144 MiB per token over 48 layers (288 MiB read+write); measured ~6.4 ms/token at 48 threads (~3% of a 250 ms token). Earlier draft of this doc wrongly said 144 MiB per layer.
- Using CPU weights in parallel with the GPU is pointless: 55 GB/s is ~6% of V100 HBM and both would fight for host RAM.
