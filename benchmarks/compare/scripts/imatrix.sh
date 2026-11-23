#!/bin/bash
# llama-imatrix on the Q8_0 GGUF with the Q-Volta GPTQ calibration text (32 x 1024-token windows decoded to text), 30 layers on the V100.
. "$(dirname "$0")/env.sh"
$FL $LB/llama-imatrix -m $WORK/q8-00001-of-00004.gguf -f $WORK/calib.txt -o $WORK/imatrix.gguf -c 1024 -b 512 -ngl 30 -t 32 --no-ppl > $OUT/imatrix.log 2>&1
ls -l $WORK/imatrix.gguf
