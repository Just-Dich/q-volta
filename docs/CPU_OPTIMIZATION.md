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

## CPU engine v1 (branch merge 2026-11-20, commit 44f5b7f)
Engine: src/runtime/cpu_engine.cpp on AVX2 GEMV (int8 activations, 32-element groups) + spin thread pool, weights in a 2 MiB-aligned
huge-page copy (plain mmap: 2.37 tok/s vs ~3.9 with huge pages). Build without CUDA: `cmake -S . -B build-cpu -DQV_CUDA=OFF`.
| threads | decode tok/s | note |
|---:|---:|---|
| 32 | 3.93 (253.7 ms/token) | box load avg 9-17 from other jobs during runs: noisy, probably understated |
| 24 | 3.78 | |
| 16 | 3.35 | |
Time per token at 32 threads (group-64 build): gate/up 40%, in-proj 22%, down 19%, out-proj 10%, lm_head 7%, GDN 1.4%, attention 0.3%
(4K context, fp32 KV). Limiter = DRAM read bandwidth (~49-58 GB/s, ceiling ~54): fewer bytes per token or several tokens per weight pass are the only levers.
Golden (tests/check_golden.py, T=64, 32 threads): group-64 activations FAIL (top1 vs qsim 0.9365, KL 7.4e-3); 32-element groups PASS
(top1 0.9524 = 60/63 vs gate 0.95, KL 3.6e-3, worst layer rel 0.078, final logits rel 0.030). Margin is thin (one flipped token fails);
KL is ~6x the GPU path (5.7e-4), mostly int8 activation noise (fp32-activation diagnostic gives <5e-4 per layer).
Not done: batched prefill GEMM (prefill = step loop), NEON/ARM, int4 KV for long context, comparison with llama.cpp on this machine.

## CPU batched prefill (branch p5/cpuprefill)
`gemm_q` / `gemm_q_par` (src/backends/cpu/cpu_gemv.cpp): each claimed 16-row block is unpacked once per 16-group K chunk into an int8
L1 buffer and multiplied against every token of the chunk (register tile 4 rows x 2 tokens, maddubs+madd, same per-32 activation
scale/sum scheme as gemv_q; weight-min term computed separately as sum_g min*xsum). `Q8Batch` = batched activations + pre-expanded scale tables.
`CpuEngine::prefill` runs chunks of 256 tokens (`--prefill-chunk`, `QV_CPU_PREFILL_CHUNK`): gemm projections, GDN conv/recurrence parallel over
channels/v-heads, attention causal over (cache + chunk) parallel over (kv head, token), logits only for the last token unless dump_hidden.
`QV_PREFILL_STEP=1` selects the old step loop, `QV_PREFILL_PROF=1` prints a phase breakdown, `QV_DIAG_GEMV=1` swaps gemm_q for per-token gemv_q.
Correctness: tests/test_cpu_gemm (gemm vs gemv_q per token <= 6.3e-7 rel); prefill with QV_DIAG_GEMV and the step path with a single attention split are
bit-identical, i.e. the only prefill-vs-step numeric differences are gemm summation order and the attention split order. The int8-activation
pipeline amplifies such 1e-7 differences chaotically (4e-4 after the first attention layer, 4-6% at layer 64 between two valid runs), so
check_golden numbers move a little between runs (worst layer 0.078 step vs 0.097 prefill; top1 vs qsim 0.9524 for both).

## MTP speculative decoding (branch p7/mtp, `qv_run --mtp`, `EngineOptions::mtp`)
Uses the checkpoint's own MTP head (mtp.fc 10240->5120, one full-attention decoder layer with its own KV slot, mtp.norm, shared embed/lm_head).
Semantics (as in vLLM qwen3_next_mtp): for position q with main hidden h_q and next token t_{q+1}: `x = fc(concat[pre_fc_norm_embedding(embed(t_{q+1})), pre_fc_norm_hidden(h_q)])`
(embedding FIRST), layer at RoPE position q, `mtp.norm`, lm_head -> draft for t_{q+2}. h_q = final-norm output of the main model (QV_MTP_HIDDEN=pre selects the pre-norm residual; not tuned).
Prefill also fills the MTP KV for the prompt (cost: +1.3..5% prefill time, see numbers).
Decode cycle: verify `[cur, draft]` in one main-model pass (weights read once), accept if argmax(row0)==draft (then 2 tokens: row0 and row1), else 1 token and roll back.
Rollback: attention KV is truncated by position; GDN recurrent state and conv state after token 0 are snapshotted inside the verify pass (per head / per channel block, in the same parallel region) into
alt buffers and swapped in on rejection (O(1), no extra token pass).
Exactness: verify uses `gemv_multi_q` (T<=4: weight planes unpacked once, per-token arithmetic identical op-for-op to `gemv_q`, so bit-identical; `gemm_q` is NOT, its weight-min term sums in another order)
and per-token split-K attention with the same splits as the 1-token path. tests/test_cpu_gemv_multi (bit-identity), tests/test_cpu_mtp (ids MTP == plain). The last token of a generation is a plain step so engine state matches plain decode.

Measured (E0 model, 32 threads, 97 generated tokens, greedy, one run each; same machine state, plain decode today is 2.65-2.78 tok/s, lower than the 3.9 recorded earlier, probably DRAM contention from other tenants; the plain/MTP ratio is what matters):
| prompt | ctx | plain tok/s | MTP tok/s | speedup | acceptance | ids identical |
|---|---|---|---|---|---|---|
| chat-formatted (code request) | 39 | 2.78 | 4.46 | 1.60x | 0.939 (46/49) | yes |
| real text (repo doc) | 128 | 2.78 | 4.09 | 1.47x | 0.827 | yes |
| random tokens | 256 | 2.71 | 4.05 | 1.49x | 0.827 | yes |
| real text | 512 | 2.67 | 4.15 | 1.55x | 0.882 | yes |
| real text | 1536 | 2.65 | 3.86 | 1.46x | 0.793 | yes |
| real text | 1880 | 2.63 | 3.75 | 1.43x | 0.778 | yes |
(An earlier plain run at ctx 1880 gave 0.84 tok/s, a machine hiccup; it was repeated and the repeat is shown.)
Cost split (ctx 39): verify pass 397 ms per 2-token cycle vs 360 ms for a plain 1-token step (1.10x), draft+MTP-KV update 31 ms per cycle (~9%). Falsification thresholds of H-MTP1 (acceptance < 0.35, verify > 1.5x step) were not met: hypothesis confirmed on this model.
Note: "acceptance" = accepted drafts per verify cycle; tokens per cycle = 1 + acceptance. Random-token prompts still show 0.83 because the model's continuation is itself predictable text-like; a prompt of prose with high entropy would be lower (not measured). Draft lm_head restriction to top-k was not tried (draft is ~9% of a cycle; max gain <9%).
