#!/bin/sh
# Agentic-corpus quality eval: BF16 ref vs RTN-quant, for base and pi-merged weights (same text, same windows).
set -u
S=/workspace/q-volta-models/scratch-agent; T=/workspace/q-volta-models/eval/agentic_synth.txt
cd /workspace/q-volta
CUDA_VISIBLE_DEVICES=0 python3 -I tools/convert.py --lora /workspace/q-volta-models/pi-adapter --out $S/pi --eval-dir $S/eval-pi --eval-text $T --nseq 17 --seqlen 512 > logs/agent-eval-pi.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -I tools/convert.py --out $S/base --eval-dir $S/eval-base --eval-text $T --nseq 17 --seqlen 512 > logs/agent-eval-base.log 2>&1
echo finished > logs/agent-eval.done
