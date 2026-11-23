#!/bin/bash
# usage: quality_llama.sh MODEL.gguf TAG [extra llama_eval args e.g. "-ngl 26"]
#  (1) single 4088-token wikitext-2 window: ppl + KL/top-1 vs the Q8_0 reference file (qv_run-format, top-1024 + tail)
#  (2) 8 x 512-token windows (the convert.py protocol behind BF16 ppl 8.4676): ppl only
. "$(dirname "$0")/env.sh"; M=$1; TAG=$2; shift 2
$FL $LB/llama_eval -m $M --tokens $DATA/P4088.i32 --mode compare --ref $WORK/ref_q8_P4088.bin --tag $TAG "$@" > $OUT/q4088_$TAG.json 2> $OUT/q4088_$TAG.err
$FL $LB/llama_eval -m $M --tokens $DATA/wiki8x512.i32 --window 512 --tag $TAG "$@" > $OUT/q8x512_$TAG.json 2> $OUT/q8x512_$TAG.err
cat $OUT/q4088_$TAG.json $OUT/q8x512_$TAG.json
