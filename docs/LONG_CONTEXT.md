# 256K context with the cold KV in host RAM (Gen 6, hypotheses H-B and H-C)

Branch `p9/tier`. Question: can the 16 attention layers' KV of a 256K context live (mostly) in pinned host RAM while the V100 keeps
decoding at ~19 ms/token (MTP verify 21-24 ms)? Everything below is measured on this host (Xeon E5-2683 v4, 64 logical CPUs, V100-SXM2-16GB,
PCIe 3.0 x16, other agents' jobs running at the same time; timed runs under `flock /tmp/qv_bench.lock`) unless it says *model* (arithmetic on
measured inputs) or *extrapolation* (fit beyond the measured range). Raw results: `benchmarks/results/tier_20261120.json` and
`benchmarks/results/tier/`.

## Verdict (short)

| design | per-token cost at 256K (19 ms GPU step) | quality | verdict |
|---|---|---|---|
| dense streaming of the cold KV over PCIe | ~400 ms (model) | exact | refuted a priori, confirmed by arithmetic |
| **H-C**: exact CPU attention over all cold blocks, merge (m,l,acc) | **185 ms = 5.4 tok/s** (10.4 ms/layer measured at 262,144 tokens, 32 threads, 29 GB/s) | exact | works, but 9.7x slower: **refuted as a decode path** (usable as an exact slow fallback only) |
| H-C on the top-k blocks only (CPU attends 2K/4K/8K tokens) | 23.6 / 26.4 / 31.9 ms (+24..68%) (model from measured 0.25/0.43/0.77 ms/layer) | approximate (top-k) | dominated by H-B-fetch |
| **H-B**: top-k block retrieval, async gather over PCIe, prefetched one step ahead | **19.8 / 20.2 / 21.0 ms (+4 / +6 / +11%)** for 2K / 4K / 8K tokens fetched per kv head (model on the measured stand-in pipeline); 22.8-32.7 ms if the fetch sits on the critical path | approximate: **mass recall criterion FAILED** (see below), but 32K end-to-end probe is benign | speed is fine, quality does not scale to 256K as specified |

* **H-B falsification criterion (>= 0.95 softmax mass at <= 4K fetched tokens for most layers) is not met**, already at 16K-32K context
  (4K fetched, sink + 1K recent window: mean recall 0.909 at 16K, 0.829 at 32K; layers >= 0.95: 5/16 and 0/16). Recall falls roughly in
  proportion to the cold/fetched ratio, so 256K (8x longer than the longest context measured here) is much worse (*extrapolation*, below).
  Even an oracle block selector (true attention mass per 64-token block) only reaches 0.90 at 32K with 4K fetched.
* End-to-end the damage at 32K is small (teacher-forced probe, sparse attention in all 16 layers): 4K fetched, R=1K: KL(dense||sparse) 0.008
  nats, top-1 agreement 95%, ppl +1.0% (64 rows, noisy); NIAH (needle at 50% depth of 32K) is found with only **1K fetched tokens**
  (16 blocks) and with every larger setting. So mass recall is a conservative proxy at 32K, but it cannot be trusted to extrapolate to 256K.
* Recommendation: the exact route is **C2 + H-A** (weights -2 GiB, int4 KV fully resident, ideal 4.5 GiB/token of KV reads, no approximation).
  Build the host tier (H-B with a big hot window and prefetch, design below) only as an opt-in **long-context profile** for when the weights
  cannot shrink, gated by NIAH + ppl measured at 64K/128K/256K on the real engine (nothing beyond 32K is validated here). H-C stays an
  exact slow fallback (5.4 tok/s).

## 1. Data

CPU engine (`src/runtime/cpu_engine.cpp`, E0 weights, fp32 KV) with two debug hooks, both off unless an env var is set (see the comment block
in the engine): `QV_KVDUMP=dir[,qstride]` writes fp16 K/V of all 16 attention layers plus the post-norm/RoPE queries of every 64th position
(from 1023) and of the last 32 positions; `QV_SPARSE=sink,recent,block,k[,agg]` (or `qv_run --sparse-list`) switches single-token attention to
sink + recent window + top-k 64-token blocks ranked by the Quest bound. `qv_run_cpu --tail T --sparse-list "cfg;cfg" [--gen G]` prefills all
but the last T tokens densely, snapshots the recurrent state and replays the T tokens for every config (KL/top-1/ppl; greedy generation).
Prompts (`tools/make_longprompt.py`, wikitext-2 test text, Qwen tokenizer): `wiki32k` (first 32,768 tokens) and `niah32k` (32,768 tokens of the same
text with `The secret passphrase for the vault is "amber-falcon-7421". Remember it.` inserted at token 16,361 (50%), question at the end).
Prefill of 32.7K tokens took 38-43 min each on 28 threads at nice 10 (12.6-14.4 tok/s, quadratic attention): **32K is the longest context
dumped**; 64K+ was not attempted (>2 h per run on the shared CPU). Dumps are 2.1 GB each and stay outside the repo.

## 2. Offline recall study (`tools/analyze_recall.py`)

Per layer, per sampled query position p (16 queries per context bucket [C-1024, C)), all 24 q heads: exact softmax over keys [0,p] vs attending
only to sink (64 tokens) + recent window R + the top-k 64-token blocks of the middle, chosen **per kv head** (4 per layer; the 6 q heads of a group
share the set, as they would in a fetch). *Mass recall* = softmax mass inside the selected set; *err* = relative L2 error of the attention output
(renormalised sparse softmax). Selectors: Quest bound sum_d max(q_d kmin_d, q_d kmax_d) summed (`quest_sum`) or max'ed over the group,
`oracle` (true block mass, ceiling for any 64-token-block selector), `mean` (q . block-mean-key). Means over the 16 layers (wiki32k; niah32k filler
gives the same numbers within 0.003):

| context | R | fetched K (tokens/kv head) | quest_sum recall | layers >= 0.95 | worst layer | output rel err | oracle recall |
|---|---|---|---|---|---|---|---|
| 8K | 1K | 1024 / 2048 / 4096 / 8192 | .911 / .940 / .976 / 1.0 | 5 / 10 / 13 / 16 | .76 / .83 / .92 / 1 | .056 / .036 / .014 / 0 | .930 / .957 / .985 / 1 |
| 16K | 1K | 1024 / 2048 / 4096 / 8192 | .769 / .837 / .909 / .965 | 0 / 0 / 5 / 12 | .59 / .70 / .77 / .88 | .123 / .083 / .046 / .017 | .850 / .904 / .951 / .984 |
| 32K | 1K | 1024 / 2048 / 4096 / 8192 | .653 / .734 / .829 / .907 | 0 / 0 / 0 / 5 | .46 / .58 / .71 / .80 | .172 / .125 / .078 / .041 | .755 / .829 / .899 / .950 |
| 32K | 4K | 1024 / 2048 / 4096 / 8192 | .814 / .857 / .898 / .937 | 0 / 0 / 4 / 8 | .68 / .75 / .79 / .85 | .091 / .070 / .048 / .029 | |
| 32K | 16K | 1024 / 2048 / 4096 / 8192 | .962 / .969 / .978 / .989 | 11 / 12 / 14 / 16 | .92 / .93 / .95 / .97 | .021 / .017 / .012 / .006 | .972 / .979 / .987 / .995 |

* Quest (sum or max over the group: within 0.01 of each other) is within 0.02-0.10 of the oracle; the mean-key score is as good as the min/max bound (0.92 vs 0.91 at 16K, K=4K).
* Layers 6-10 are the easiest (0.90-0.93 at 32K, K=4K), layers 0-3 and 14-15 the hardest (0.71-0.77): a per-layer budget would help but the
  worst layers need K > 8K at 32K already.
* The recent window matters as much as K: R=4K at 4K fetched is worth about +2x fetched tokens; **R=16K (half of the 32K context) reaches 0.978 at
  4K fetched**. A hot window that is a large fraction of the context is not a fair preview of 256K with R=1K.
* Staleness (for a prefetch from the previous token's query; last 32 consecutive positions of niah32k, question tokens, which are harder than the
  filler: quest_sum 4K = .809 fresh): query of p-1: .792, of p-4: .775, i.e. -0.017 / -0.035 -- prefetching one step ahead costs little.
* Needle: the dumped last-32 queries are question tokens (the answer-position query is the first step of the generation, not dumped). Fraction of (query, kv head) pairs
  whose selection (quest_sum, R=1K) contains the needle's block, mean over layers [worst layer]: K=1K 0.41 [0.0], 2K 0.57 [0.04], 4K 0.79 [0.34], 8K 0.96 [0.79]. End to end see section 4.

**Extrapolation to 256K (not measured).** Pooled fit over all (context 16K/32K, R, K) points: log(missing mass) = c0 + c1 log(cold/K) + c2 log R,
log-rmse 0.24-0.25 (about +-27%), evaluated 8x beyond the largest measured context and 3-5x beyond the largest R: mass recall at 256K with R = 1K / 16K /
64K / 87K (1.5 GiB int4 hot window) and K = 4K: 0.18-0.20 / 0.74-0.84 / 0.88-0.94 / 0.90-0.95; with K = 8K: 0.53 / 0.85-0.90 / 0.93-0.96 / 0.94-0.97.
Order of magnitude only: with R=1K recall is bad at any realistic K; a hot window of tens of thousands of tokens gets close to 0.95 *if* the fit's R-exponent
survives, which is unverified.

## 3. Transfer and CPU benchmarks (all under the bench lock)

PCIe (`bench_pcie`, `bench_tier`; pinned 4.50 GiB pool = full 256K int4 store, unit = one (layer, kv head, 64-token block): K 64x144 B + V 64x144 B = 18,432 B):

| path | 1K tokens/head (1.18 MB) | 2K (2.36) | 4K (4.72) | 8K (9.44) |
|---|---|---|---|---|
| contiguous cudaMemcpyAsync, upper bound | 11.5 GB/s, 0.102 ms | 11.4, 0.207 | 12.2, 0.388 | 12.3, 0.767 |
| **zero-copy gather kernel** (mapped pinned memory, 16 blocks x 512 threads, scattered 18 KiB units) | 11.0 GB/s, 0.107 ms | 11.6, 0.203 | 12.0, 0.393 | 12.2, 0.774 |
| cudaMemcpyAsync per 18 KiB unit, 1 stream | 4.0 GB/s, 0.30 ms | 4.1, 0.58 | 4.2, 1.13 | 4.2, 2.25 |
| same, 4/8/16 streams | 3.1-3.6 | 3.7-3.9 | 3.5-3.8 | 3.5-3.8 |
| cudaMemcpyAsync per 72 KiB unit (block for all 4 heads: selection shared across kv heads) | 7.9 GB/s | 8.0 | 8.2 | 8.2 |

Pinned H2D 12.4 GB/s (64 MB), D2H 12.8 GB/s; 6 KiB D2H+sync+H2D round trip 14.5 us. Scattered copies via the API are host-launch bound (~4.5 us per
call), the gather kernel runs at the PCIe rate regardless of scatter and needs no host calls (CUDA-graph friendly: the block list lives in device memory).

Pipeline emulation (`bench_tier`, stand-in for the GPU work: 16 groups of a multi-wave 800 MB bandwidth-bound kernel = 15.14 ms/step, fetch of group i+1
issued when group i starts, high-priority stream, double buffered; not the real engine, so SM/HBM interference of the real GEMV kernels is not captured):

| fetched/kv head | gather 4 blocks | gather 16 blocks | gather 64 blocks | memcpyAsync/unit (8 streams) | fetch on critical path (16 blocks) |
|---|---|---|---|---|---|
| 2K | 15.61 (+3.1%) | 15.57 (+2.9%) | 17.51 | 16.59 | 18.58 |
| 4K | 16.02 | 15.99 (+5.6%) | 20.37 | 23.74 | 21.86 |
| 8K | 17.50 | 16.82 (+11%) | 26.02 | 41.74 | 28.47 |

A persistent small grid (4-16 blocks) overlaps almost perfectly; a grid with one block per unit (64+) steals SMs and loses most of the overlap.
A first version of this benchmark that did not gate the prefetch on compute showed no difference between overlapped and critical-path fetch;
that was a benchmark bug (fetch stream ran ahead of the compute), fixed (events `ce[l]`).

CPU int4 attention (`bench_cold`, `src/cpu/attn_cpu.cpp` from p3/cpuattn 9deb469, AVX2/F16C/FMA, row = 144 B; new `attn_cpu_partial_blocks` for scattered blocks; checked
against the scalar double reference 1e-6, `tests/test_attn_cpu.cpp` ALL PASS):

* All cold KV of one layer, 262,144 tokens (302 MB; 16 layers rotated so nothing is cached): 8 / 16 / 32 / 64 threads = 34.1 / 17.5 / **10.4** / 12.0 ms per layer
  (8.8 / 17.2 / 29.1 / 25.3 GB/s) => x16 layers = 546 / 280 / **166** / 191 ms per token. ALU-bound per thread; 64 threads are slower than 32.
* Scattered top-k blocks (4 kv heads, K tokens each): K=1K/2K/4K/8K/16K at 32 threads 0.145 / 0.248 / 0.427 / 0.768 / 1.31 ms per layer; at 16 threads 0.19 / 0.30 / 0.54 / 1.01 / 1.80.

## 4. Quality probe with real sparse attention (CPU engine, E0 weights, 32K, `tools/sparse_logits_eval.py`)

Dense 32K prefill, then teacher-forced steps with sparse attention in every attention layer (selection recomputed at every step from the current query; sink 64).

| config (R, K tokens fetched) | wiki32k, 64 rows: KL / top-1 / ppl delta | niah32k, 32 rows: KL / top-1 | NIAH answer `amber-falcon-7421"` (greedy, 14 tokens) |
|---|---|---|---|
| dense | 0 / 1.00 / 0 | 0 / 1.00 | correct |
| R=1K, K=256 | 0.101 / 0.89 / +19.1% | 0.625 / 0.81 | wrong |
| R=1K, K=1K | 0.033 / 0.92 / +6.7% | 0.069 / 0.91 | **correct** |
| R=1K, K=2K | 0.016 / 0.94 / +3.0% | 0.058 / 0.94 | correct |
| R=1K, K=4K | 0.008 / 0.95 / +1.0% | 0.033 / 0.97 | correct |
| R=1K, K=8K | 0.0045 / 0.98 / -0.0% | 0.013 / 0.97 | correct |
| R=4K, K=2K | 0.0057 / 0.97 / +0.6% | 0.048 / 0.94 | correct |
| R=1K, K=4K, max-over-group score | 0.009 / 0.98 / -1.9% | 0.033 / 0.97 | correct |

(ppl deltas of +-2% are noise at 63 rows; the niah tail is the question text, harder than prose; for scale, E0 quantisation vs bf16 is KL ~0.12 and kv4 vs
fp32 KV is ~1e-3.) The mass-recall criterion is therefore stricter than what 32K needs for this model, but 0.83 mass at 32K with R=1K and the 8x longer context
to come leave no margin to claim 256K quality. Not measured: any context > 32K, code / agentic prompts (attention is likely more peaked than on Wikipedia text),
int4-quantised K for the bounds, multi-needle / long-QA tasks.

## 5. Per-token cost model at 256K (`tools/tier_model.py`, *model*)

GPU step 19 ms (plain), 16 attention layers, K tokens per kv head per layer; selection kernel (min/max fp16 summaries read once per token: 256 MiB at 256K) 0.34 ms at 800 GB/s.

| K | MB fetched / token (16 layers) | H-B gather, prefetched | H-B gather, on critical path | H-C top-k on CPU (32 thr, critical path) |
|---|---|---|---|---|
| 2K | 37.7 | 19.8 ms (+4%) | 22.8 | 23.6 |
| 4K | 75.5 | 20.2 (+6%) | 26.1 | 26.4 |
| 8K | 151 | 21.0 (+11%) | 32.7 | 31.9 |

PCIe load of the prefetched design: 75-151 MB per 19 ms step = 4-8 GB/s of the 12.4 GB/s. Dense exact CPU: 185 ms (5.4 tok/s). Dense PCIe streaming of everything outside a 6K window: 400 ms.
Context for the speed side: the GPU attention kernel measured by p9/kv4 costs ~0.35 ms/layer at 32K regardless of format (their inference); if that scaled
linearly, resident full-context attention at 256K would add ~+50 ms/token, which sparse retrieval avoids (it attends ~6K tokens): the sparse tier is faster than any dense
resident layout *if* the attention kernel is not improved (unmeasured extrapolation).

## 6. Design (if the long-context profile is built on top of p9/kv4)

1. **Host pool** (pinned, `cudaHostAlloc(Mapped|Portable)`): layout `[layer 16][kv head 4][block 4096]`, unit = one 64-token block record in exactly the p9/kv4 GPU block
   format (K and V int4 group-64 + fp16 scale/min, 18,432 B per (layer, head, block) in the layout benchmarked here; the benchmark used 144 B rows, so convert to the kv4 record
   size). 256K = **4.50 GiB** (measured allocation). fp16 cold storage = 16 GiB host (fits the 76 GB host) but 3.6x the bytes per fetch (75 MB -> 270 MB at K=4K = 22 ms/token of pure
   PCIe): not viable except at K<=1K; recommend int4 cold (its own quality cost is H-A's ~1e-3 KL).
2. **Writing the pool**: when a 64-token block closes and leaves the hot window, a D2H copy of its record (async, off the decode critical path); during prefill the same, per chunk.
   Exact chunked prefill needs the cold blocks of all earlier tokens: stream them back per chunk, ~288 GiB of H2D in total for 256K/2048-token chunks = ~23 s (*model*), negligible
   next to the prefill time.
3. **GPU-resident**: sink block (64 tokens), hot window of R tokens (a ring of block records; all spare VRAM: 1.5 GiB / 18 KiB = ~87K tokens, 16K tokens = 0.29 GiB), per-(layer, head, block)
   fp16 min/max key summaries (256 MiB at 256K; 128 MiB if int8) computed from the dequantised K when a block closes, and a double-buffered staging slab per layer
   (K=4K: 4.7 MB x 16 layers x 2 = 150 MB).
4. **Selection kernel** (one CTA per kv head, 4 CTAs per layer, after the q/k norm+RoPE): score(block) = sum over the 6 q heads of the group of sum_d max(q_d kmin_d, q_d kmax_d)
   (max over the group is equivalent within 0.005 mass), exclude sink and hot window, top-k by a two-pass radix/threshold select over 4096 scores, write the block list (device memory).
   Cost ~0.02 ms/layer (summary read 16.8 MB; *model*).
5. **Gather**: persistent zero-copy kernel, **4-16 blocks of 512 threads**, grid-stride over the block list, on a high-priority stream, no host involvement (graph capturable); measured 11-12.2 GB/s.
6. **Prefetch overlap**: select layer L's blocks for token t+1 from the query of token t at layer L (stale-1, measured -0.017 recall) as soon as that query exists, gather into the
   staging slab while the other ~15 ms of the step runs; attention of token t+1 at layer L reads the staging slab. The whole step is the slack (75-151 MB per 19 ms). Optional fresh
   correction (select from the current q among the cached top-2K, fetch ~512 extra tokens on the critical path) was not measured.
7. **Attention**: three segments (sink | hot window | staged blocks), each a split-K partial (m, l, acc) merged exactly like the existing split-K combine, with the kv4 kernel reading a block table instead of a contiguous range.
   Keys are stored post-RoPE, so no re-rotation of fetched blocks.
8. **MTP verify (depth d, d+1 rows)**: all rows of a verify pass share one selection per layer (query of the last committed token, stale), so the fetch volume is the same as a plain step and
   the verify pass is 1.1-1.25x a plain step as before. The draft rows' own KV is in the hot window. Exactness: MTP == plain greedy holds only if plain decode uses the identical selection rule;
   a stale-1 rule (set depends on the previous token's q) differs between a 1-row and a (d+1)-row pass, so the bit-exact guarantee must be re-defined: either **windowed refresh**
   (selection recomputed at positions that are multiples of S=16 from that row's query and reused for the next S tokens, a pure function of position, one critical-path fetch of 6-13 ms every 16 tokens = ~0.4-0.8 ms/token amortised,
   but staleness up to 15 tokens, quality unmeasured) or accept "exact with respect to the sparse model" only per cycle. Decide after the quality gate.
9. **Gate before shipping**: NIAH (multiple depths, multi-needle) + ppl on PG19/wikitext at 64K/128K/256K with the profile on the real engine, against dense int4-resident KV at 64K/128K (fits); per-layer
   budgets (layers 0-3, 14-15 need more); recent-window size sweep. No context length is claimed before that.

## 7. Exact CPU merge (H-C) as the alternative, and its cost

The cold blocks are attended on the CPU in int4 (AVX2, 32 threads), the GPU attends sink + hot window, partials (m, l, acc) are merged (the split-K combine). The q is 24x256 fp32 = 24 KiB D2H and the partials
(24x(256+2) fp32 = 25 KiB) H2D per layer (round trip 15-18 us measured), the CPU time is the whole cost: **10.4 ms/layer at 256K, 166 ms/token for 16 layers; 185 ms/token with the GPU step = 5.4 tok/s, 9.7x slower**, exact. With
a hot window of 87K tokens the cold part is 175K tokens = ~7 ms/layer = ~110 ms/token (*model, linear*). CPU DRAM bandwidth is not the limit (29 of ~55 GB/s); the per-thread AVX2 int4 unpack/FMA rate is (Broadwell, no VNNI/AVX-512), which
also means an ARM NEON/dotprod port could behave differently (no ARM measurement available). Use only as an exact fallback (e.g. a `--exact-cold` flag), not as the default path.

## Files

`src/runtime/cpu_engine.cpp` (QV_KVDUMP / QV_SPARSE hooks, state_save/restore), `include/qv/engine.h`, `tools/qv_run.cpp` (`--tail`, `--sparse-list`, `--gen`), `tools/make_longprompt.py`, `tools/analyze_recall.py`,
`tools/sparse_logits_eval.py`, `tools/tier_model.py`, `src/cpu/attn_cpu.{h,cpp}` + `tests/test_attn_cpu.cpp` + `benchmarks/bench_host.cpp` + `bench_pcie.cu` (cherry-picked from p3/cpuattn 9deb469; CMake target
renamed `qv_hostattn`, the CPU backend's `qv_cpu` is untouched), `benchmarks/bench_tier.cu`, `benchmarks/bench_cold.cpp`, `benchmarks/results/tier_20261120.json`, `benchmarks/results/tier/*`.
