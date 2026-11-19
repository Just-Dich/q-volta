# Q-Volta — working notes for Claude

## Current stage
**Stage A (GPU, Tesla V100-SXM2-16GB, sm_70), Milestone 1**: correct full-model GPU forward + plain greedy decode,
benchmarked against NInfer on the same GPU.

**CPU-only backend work (Stage B) is NOT allowed yet.** It starts only after the GPU completion gate in the task spec
(correctness, <16 GB, reproducible benches, profiling, MTP, long-context measured, docs). CPU code in this stage is
limited to utilities that operate the GPU backend (loader, tokenizer, server).

## Model
Qwen/Qwen3.8-27B @ 37561816 (BF16, `qwen3_5` hybrid: 48 Gated-DeltaNet + 16 gated GQA layers, 1 MTP layer).
Local BF16 shards: /workspace/q-volta-models/Qwen3.8-27B (scripts/download_bf16.sh).
Quantized by our own converter (tools/convert.py) into the Q-Volta v1 format (tools/qv_format.py docstring):
bit-plane packed, asymmetric group-64 RTN; E0 bit plan mlp 3-bit, gdn/attn 4-bit, embed 4-bit, lm_head 6-bit.

## Last completed milestone
- E0 converter + reference pass (BF16 ref + simulated-quant golden activations) — see logs/convert-e0.log
- GEMV: dp4a int8-activation path at 720–880 GB/s (81–99 % of 890 GB/s measured HBM read peak)
- CUDA-graph greedy decode path (H1) implemented — measurement pending

## Commands
```
cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && ninja -C build
python3 -I tests/gen_gemv_cases.py build/gemv_cases && ./build/test_gemv build/gemv_cases [q8]   # kernel tests
python3 -I tests/check_golden.py /workspace/q-volta-models/qv/qwen38-27b-e0 \
        /workspace/q-volta-models/qv/eval-e0 build/qv_run 64                                  # full-model golden
./build/bench_gemv [q8]                                                                        # GEMV bandwidth
./build/qv_run --model /workspace/q-volta-models/qv/qwen38-27b-e0 --bench --prompt P.i32 --n 128 [--graph]
```
Env switches: `QV_GEMV=fp32` (A/B the fp32 GEMV path), `QV_GEMV_CFG=R,W` (force GEMV rows/warps).

## Confirmed performance (V100, measured)
- GEMV q8 (µs): mlp gate/up 3b 53.8, mlp down 3b 50.1, gdn qkv 4b 37.3, attn q 4b 44.3, lm_head 6b 1171
- Baseline to beat (NInfer, IQ3_S, same GPU, pre-existing run): 45.4 tok/s plain, 100.4 tok/s MTP-3

## Current bottlenecks / next step
1. Run golden test → fix any mismatch.  2. Bench plain decode (graph vs no graph) vs NInfer re-run.
3. Then: fuse norm→q8-quant, KV fp16, attention split-K for long context, batched prefill (WMMA), MTP.

## Rules
- Every optimisation: test_gemv / check_golden must pass; log in ~/autoresearch-runs/q-volta/EXPERIMENTS.md.
- Never report unmeasured numbers. No CPU offload in GPU benchmarks.
- The strata service was stopped to free the V100 (restart: `cd /workspace/strata-v100/dashboard && sh watchdog.sh start`).
