#pragma once
#include "../common.cuh"
#include "gemv.h"

namespace qv::cuda {
// q8 != nullptr: also emit the int8 GEMV input (fused quantization)
void rmsnorm(const float* x, const float* w, float* out, float* xs32, int n, float eps, cudaStream_t s,
             const Q8Buf* q8 = nullptr);
void embed(const QMat& E, const int* tok, float* out, cudaStream_t s);
void silu_mul(const float* g, const float* u, float* out, float* xs32, int n, cudaStream_t s, const Q8Buf* q8 = nullptr);
void argmax(const float* x, int n, int* out, cudaStream_t s);
void argmax_advance(const float* x, int n, int* tok, int* pos, int* gen, cudaStream_t s);
}  // namespace qv::cuda
