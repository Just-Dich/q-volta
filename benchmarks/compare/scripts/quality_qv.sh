#!/bin/bash
# usage: quality_qv.sh MODEL.qv TAG   (Q-Volta; same token files, same protocol as quality_llama.sh)
. "$(dirname "$0")/env.sh"; M=$1; TAG=$2; shift 2
$FL $QV_BIN --model $M --max-ctx 4608 --eval $DATA/P4088.i32 --eval-mode cmp --eval-ref $WORK/ref_q8_P4088.bin "$@" > $OUT/q4088_$TAG.json 2> $OUT/q4088_$TAG.err
# 8 x 512: one qv_run per window (state reset), combined by collect.py (mean NLL)
for i in 0 1 2 3 4 5 6 7; do
  python3 -I -c "import numpy as np;np.fromfile('$DATA/wiki8x512.i32',dtype=np.int32).reshape(8,512)[$i].tofile('/tmp/qv_w$i.i32')"
  $FL $QV_BIN --model $M --max-ctx 1024 --eval /tmp/qv_w$i.i32 --eval-mode write --eval-ref /tmp/qv_w_ref.bin --eval-topk 16 "$@" 2>>$OUT/q8x512_$TAG.err | tail -1
done > $OUT/q8x512_$TAG.json
cat $OUT/q4088_$TAG.json $OUT/q8x512_$TAG.json
