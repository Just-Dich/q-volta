#pragma once
#include "../common.cuh"

namespace qv::cuda {
// y = W x (+ residual). x must have K floats, xsum32 = per-32 chunk sums of x (see xsum32()).
void gemv_q(const QMat& W, const float* x, const float* xsum32, float* y, const float* residual, cudaStream_t s);
void xsum32(const float* x, float* out, int K, cudaStream_t s);
void gemv_f16(const DMat16& W, const float* x, float* y, cudaStream_t s);
}  // namespace qv::cuda
