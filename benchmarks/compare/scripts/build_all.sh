#!/bin/bash
# Build the CUDA llama.cpp (sm_70) + llama_eval + Q-Volta qv_run. nvml.h is missing from CUDA 12.9 here: copy it from 12.6.
. "$(dirname "$0")/env.sh"
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p /tmp/nvml_inc && cp /usr/local/cuda-12.6/targets/x86_64-linux/include/nvml.h /tmp/nvml_inc/
cmake -S $LLAMA_SRC -B $LLAMA_BUILD -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS=-I/tmp/nvml_inc -DCMAKE_CXX_FLAGS=-I/tmp/nvml_inc
cmake --build $LLAMA_BUILD -j24 --target llama-bench llama-quantize llama-perplexity llama-server llama-imatrix llama-cli
sh "$(dirname "$0")/build_llama_eval.sh"
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cmake -S $REPO -B $REPO/build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && ninja -C $REPO/build qv_run
