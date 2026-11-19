#pragma once
#include "../common.cuh"

namespace qv::cuda {
struct GemvCfg { int rows = 0, warps = 0; };  // rows==0 -> per-shape default
GemvCfg gemv_default_cfg(const QMat& W);
// y = W x (+ residual). x must have K floats, xsum32 = per-32 chunk sums of x (see xsum32()).
void gemv_q(const QMat& W, const float* x, const float* xsum32, float* y, const float* residual, cudaStream_t s,
            GemvCfg cfg = {});
void xsum32(const float* x, float* out, int K, cudaStream_t s);
void gemv_f16(const DMat16& W, const float* x, float* y, cudaStream_t s);
void gemv_f16_dual(const DMat16& A, const DMat16& B, const float* x, float* ya, float* yb, cudaStream_t s);

// int8 activation vector for gemv_q8 (canonical chunk permutation, see gemv_q8.cu)
struct Q8Vec { const int* q; const float* d; const float* sum; const int* qsum; };
struct Q8Buf {
    int* q = nullptr; float* d = nullptr; float* sum = nullptr; int* qsum = nullptr;  // q: K/4 ints, rest K/32
    operator Q8Vec() const { return Q8Vec{q, d, sum, qsum}; }
};
void quantize_q8(const float* x, int K, Q8Buf& out, cudaStream_t s);
GemvCfg gemv_q8_default_cfg(const QMat& W);
void gemv_q8(const QMat& W, const Q8Vec& x, float* y, const float* residual, cudaStream_t s, GemvCfg cfg = {});
// nt token vectors stored back to back (quantize_q8 over nt*K elements produces exactly this layout); y: [nt][N]
void gemv_q8_multi(const QMat& W, const Q8Vec& x, int nt, float* y, const float* residual, cudaStream_t s);
}  // namespace qv::cuda
