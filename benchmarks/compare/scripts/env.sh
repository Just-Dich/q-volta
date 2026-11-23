#!/bin/bash
export LLAMA_SRC=${LLAMA_SRC:-/workspace/q-volta-models/scratch-cpu-baseline/llama.cpp}   # commit 9d6f04f
export LLAMA_BUILD=${LLAMA_BUILD:-/tmp/llama-cuda-build}
export LB=$LLAMA_BUILD/bin
export QV_BIN=${QV_BIN:-/workspace/q-volta-wt/compare/build/qv_run}
export HF_DIR=${HF_DIR:-/workspace/q-volta-models/Qwen3.8-27B}
export QVM=${QVM:-/workspace/q-volta-models/qv}                 # qwen38-27b-e0.qv / e1.qv
export WORK=${WORK:-/workspace/q-volta-models/scratch-compare}   # big files (GGUF); keep <= ~30 GB
export CMP=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)    # benchmarks/compare
export DATA=$CMP/data
export OUT=${OUT:-$CMP/raw}                                     # raw per-run outputs
export GPU_LOCK=/tmp/qv_bench.lock
# QV_HELD=1 means the caller already holds the lock for the whole block (flock $GPU_LOCK bash block.sh)
export FL=$([ -n "$QV_HELD" ] && echo "" || echo "flock $GPU_LOCK")
export CUDA_VISIBLE_DEVICES=0                                   # V100 only (the P4 is ignored)
mkdir -p "$OUT" "$WORK"
