// Batched projections for prefill: Y[T][N] = X[T][K] * W[N][K]^T with fp16 inputs, fp32 accumulate, fp32 output.
// Quantized weights are dequantized to an fp16 scratch slab (a block of output rows at a time) and multiplied with
// cuBLAS tensor-core HGEMM; for T == 1..8 the int8 dp4a GEMV path stays faster and is used by the decode code.
#pragma once
#include <cublas_v2.h>

#include "../common.cuh"

namespace qv::cuda {

struct LinearPrefill {
    cublasHandle_t cublas = nullptr;
    __half* slab = nullptr;  // scratch for dequantized rows
    size_t slab_elems = 0;   // capacity in fp16 elements

    void init(cudaStream_t s, size_t slab_bytes = (size_t)32 << 20);
    void destroy();
};

// Y[T][N] = X[T][K] * dequant(W)^T
void linear_seq(LinearPrefill& lp, const QMat& W, const __half* X, int T, float* Y, cudaStream_t s);
// same for a dense fp16 weight (no dequant)
void linear_seq_dense(LinearPrefill& lp, const DMat16& W, const __half* X, int T, float* Y, cudaStream_t s);

}  // namespace qv::cuda
