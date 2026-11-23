#!/bin/bash
# Build llama_eval against the CUDA llama.cpp build tree (LL=src checkout, B=build dir)
LL=${LL:-/workspace/q-volta-models/scratch-cpu-baseline/llama.cpp}; B=${B:-/tmp/llama-cuda-build}
HERE=$(cd "$(dirname "$0")/.." && pwd)
g++ -O2 -std=c++17 -Wno-deprecated-declarations -I$LL/include -I$LL/ggml/include $HERE/llama_eval.cpp -o $B/bin/llama_eval \
  -L$B/bin -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -Wl,-rpath,$B/bin
