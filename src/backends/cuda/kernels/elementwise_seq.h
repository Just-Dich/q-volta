// Batched (multi-token) elementwise kernels for the prefill path. Token-major layout: element (t, i) at [t * n + i].
// Semantics are identical to the single-token kernels in elementwise.cu applied to each row in order.
#pragma once
#include "../common.cuh"

namespace qv::cuda {

// out[t][k] = dequantized embedding row toks[t]   (fp32 residual stream)
void embed_rows(const QMat& E, const int* toks, int T, float* out, cudaStream_t s);

// out[t][i] = fp16( clamp( x[t][i] * rsqrt(mean_i(x[t]^2)+eps) * w[i], +-65504 ) )     (w already holds 1+w)
void rmsnorm_rows_f16(const float* x, const float* w, __half* out, int n, int T, float eps, cudaStream_t s);

// out[t][i] = x[t][i] * rsqrt(mean(x[t]^2)+eps) * w[i]   (fp32 out; may alias nothing)
void rmsnorm_rows_f32(const float* x, const float* w, float* out, int n, int T, float eps, cudaStream_t s);
// MTP head input rows: out[t] (2n halves) = [ rmsnorm(emb[t], w_e) | rmsnorm(hid[t], w_h) ] (fp16, clamped)
void mtp_cat_rows_f16(const float* emb, const float* hid, const float* w_e, const float* w_h, __half* out, int n, int T,
                      float eps, cudaStream_t s);

// out[i] = fp16(clamp(silu(g[i]) * u[i]))  over `total` elements
void silu_mul_f16(const float* g, const float* u, __half* out, size_t total, cudaStream_t s);

// x[i] += y[i]  over `total` elements (residual add)
void add_f32(float* x, const float* y, size_t total, cudaStream_t s);

}  // namespace qv::cuda
