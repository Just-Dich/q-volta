#!/bin/bash
# usage: quantize.sh TYPE [IMATRIX.gguf]  -> $WORK/TYPE[.im].gguf (requantized from the Q8_0 GGUF: --allow-requantize)
# With an imatrix the MTP layer (blk.64, not covered by llama-imatrix) is forced to Q4_K: IQ3_* refuse tensors without imatrix data.
. "$(dirname "$0")/env.sh"; T=$1; IM=$2
if [ -n "$IM" ]; then OUTF=$WORK/$T.im.gguf; X=(--imatrix $IM --tensor-type 'blk\.64\..*=q4_K'); else OUTF=$WORK/$T.gguf; X=(); fi
df -h $WORK | tail -1
CUDA_VISIBLE_DEVICES= $LB/llama-quantize --allow-requantize "${X[@]}" $WORK/q8-00001-of-00004.gguf $OUTF $T 2>&1 | tail -n 6
ls -l $OUTF
