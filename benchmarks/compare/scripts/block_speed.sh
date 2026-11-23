#!/bin/bash
# usage: flock /tmp/qv_bench.lock bash block_speed.sh LABEL MODEL.gguf MODEL.qv [speed.py flags]   (whole block under one lock hold)
export QV_HELD=1; D=$(dirname "$0"); L=$1; G=$2; Q=$3; shift 3
nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv > $OUT/clocks_$L.txt
python3 -I $D/speed.py $L $G $Q 3 "$@"
