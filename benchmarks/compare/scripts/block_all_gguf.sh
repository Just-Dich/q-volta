#!/bin/bash
# usage: flock /tmp/qv_bench.lock bash block_all_gguf.sh TAG MODEL.gguf [MODEL.qv for the speed block]  -> quality + speed + llama max ctx
export QV_HELD=1; D=$(dirname "$0"); . $D/env.sh; T=$1; G=$2
bash $D/quality_llama.sh $G $T
if [ -n "$3" ]; then bash $D/block_speed.sh ${T}_vs_$(basename $3 .qv) $G $3 --llama-mtp; fi
for kv in f16 q8_0 q4_0; do python3 -I $D/maxctx.py llama $G $kv; done >> $OUT/maxctx_llama_$T.jsonl
