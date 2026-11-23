#!/bin/bash
# usage: flock /tmp/qv_bench.lock bash block_maxctx_qv.sh LABEL MODEL.qv
export QV_HELD=1; . "$(dirname "$0")/env.sh"; D=$(dirname "$0")
for kv in int4 fp16; do python3 -I $D/maxctx.py qv $2 $kv; done >> $OUT/maxctx_$1.jsonl
