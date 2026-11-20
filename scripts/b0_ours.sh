#!/bin/sh
# Our engine on the same two chat prompts as B0 (greedy, 128 new tokens, graph on), 3 repeats. GPU must be idle.
M=${1:-/workspace/q-volta-models/qv/qwen38-27b-e0}; cd /workspace/q-volta
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader -i 0 > logs/b0-ours.clocks.txt
for k in code prose; do for r in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/qv_run --model $M --bench --prompt logs/chat_$k.i32 --n 128 --graph 2>/dev/null | python3 -I -c "
import sys,json;d=json.loads(sys.stdin.read().strip().splitlines()[-1]);d.pop('ids');print('$k',json.dumps(d))" 
done; done | tee logs/b0-ours.out
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader -i 0 >> logs/b0-ours.clocks.txt
