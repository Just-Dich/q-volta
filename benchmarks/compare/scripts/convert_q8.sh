#!/bin/bash
# BF16 safetensors -> Q8_0 GGUF (29 GB; includes the MTP/nextn head). Split in 8 GB shards so that part of it can sit in RAM
# files (scripts/ramfile.py + symlinks) when the disk is tight. Q8_0 is a near-lossless PROXY of BF16, not BF16.
. "$(dirname "$0")/env.sh"
cd $LLAMA_SRC && CUDA_VISIBLE_DEVICES= python3 convert_hf_to_gguf.py $HF_DIR --outtype q8_0 --split-max-size 8G --outfile $WORK/q8.gguf
