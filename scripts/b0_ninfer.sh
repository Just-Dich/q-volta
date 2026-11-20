#!/bin/sh
# B0: run NInfer (Apache-2.0, third-party) on THIS V100 with the same two chat prompts, greedy, 128 tokens, plain and MTP-3.
# Usage: b0_ninfer.sh   (needs the GPU idle). Logs GPU clocks/power before+after each run.
set -u
M=/workspace/q-volta-models/ninfer-iq3s/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer
BIN=/workspace/ninfer-all/build/apps/ninfer-serve
cd /workspace/q-volta
for mode in none mtp; do
  extra=""; [ $mode = mtp ] && extra="--spec mtp --draft-tokens 3"
  CUDA_VISIBLE_DEVICES=0 $BIN $M --model-id qwen3.8-27b --host 127.0.0.1 --port 8000 --max-context 8192 --kv-capacity 8192 \
      --kv-dtype rk4v4 --max-concurrency 1 --gdn-state-fp16 --prefill-chunk 512 --default-max-tokens 256 $extra \
      > logs/b0-ninfer-$mode.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 2; done
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader -i 0 > logs/b0-$mode.clocks.txt
  for rep in 1 2 3; do python3 -I benchmarks/bench_chat.py b0-$mode-r$rep > logs/b0-$mode-r$rep.out 2>&1; done
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader -i 0 >> logs/b0-$mode.clocks.txt
  kill $pid; wait $pid 2>/dev/null; sleep 5
done
echo finished > logs/b0.done
