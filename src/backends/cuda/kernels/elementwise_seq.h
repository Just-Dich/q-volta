// Batched (multi-token) elementwise kernels for the prefill path. Token-major layout: element (t, i) at [t * n + i].
// Semantics are identical to the single-token kernels in elementwise.cu applied to each row in order.
#pragma once
#include "../common.cuh"

namespace qv::cuda {

// out[t][k] = dequantized embedding row toks[t]   (fp32 residual stream)
void embed_rows(const QMat& E, const int* toks, int T, float* out, cudaStream_t s);

// out[t][i] = fp16( clamp( x[t][i] * rsqrt(mean_i(x[t]^2)+eps) * w[i], +-65504 ) )     (w already holds 1+w)
void rmsnorm_rows_f16(const float* x, const float* w, __half* out, int n, int T, float eps, cudaStream_t s);

// out[i] = fp16(clamp(silu(g[i]) * u[i]))  over `total` elements
void silu_mul_f16(const float* g, const float* u, __half* out, size_t total, cudaStream_t s);

// x[i] += y[i]  over `total` elements (residual add)
void add_f32(float* x, const float* y, size_t total, cudaStream_t s);

}  // namespace qv::cuda
