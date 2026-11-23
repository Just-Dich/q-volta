#!/bin/bash
# usage: flock /tmp/qv_bench.lock bash block_maxctx.sh LABEL MODEL.gguf MODEL.qv  -> raw/maxctx_LABEL.jsonl
export QV_HELD=1; . "$(dirname "$0")/env.sh"; D=$(dirname "$0"); L=$1
for kv in f16 q8_0 q4_0; do python3 -I $D/maxctx.py llama $2 $kv; done >> $OUT/maxctx_$L.jsonl
for kv in int4 fp16; do python3 -I $D/maxctx.py qv $3 $kv; done >> $OUT/maxctx_$L.jsonl
