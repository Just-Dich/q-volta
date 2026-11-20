#include "elementwise_seq.h"

namespace qv::cuda {

__device__ __forceinline__ float clamp_h(float v) { return fminf(fmaxf(v, -65504.f), 65504.f); }

// grid = (ceil(K/256), T)
__global__ void embed_rows_kernel(QMat E, const int* __restrict__ toks, float* __restrict__ out) {
    const int row = toks[blockIdx.y];
    const int C = E.K >> 5, G = E.K >> 6;
    float* o = out + (size_t)blockIdx.y * E.K;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < E.K; k += gridDim.x * blockDim.x) {
        const int c = k >> 5, i = k & 31;
        const uint32_t* w = E.q + ((size_t)row * C + c) * E.bits;
        uint32_t q;
        switch (E.bits) {
            case 2: q = (w[i & 1] >> (2 * (i >> 1))) & 3u; break;
            case 3: q = ((w[i & 1] >> (2 * (i >> 1))) & 3u) | (((w[2] >> i) & 1u) << 2); break;
            case 4: q = (w[i & 3] >> (4 * (i >> 2))) & 15u; break;
            case 5: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4] >> i) & 1u) << 4); break;
            case 6: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 1)] >> (2 * (i >> 1))) & 3u) << 4); break;
            default: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 3)] >> (4 * (i >> 2))) & 15u) << 4);
        }
        const float2 sm = __half22float2(E.sm[(size_t)row * G + (c >> 1)]);
        o[k] = sm.x * (float)q + sm.y;
    }
}

void embed_rows(const QMat& E, const int* toks, int T, float* out, cudaStream_t s) {
    embed_rows_kernel<<<dim3((E.K + 255) / 256, T), 256, 0, s>>>(E, toks, out);
}

// one block per token row
__global__ void __launch_bounds__(256) rmsnorm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                           __half* __restrict__ out, int n, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    __half* orow = out + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(xr)[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) orow[i] = __float2half_rn(clamp_h(xr[i] * r * w[i]));
}

void rmsnorm_rows_f16(const float* x, const float* w, __half* out, int n, int T, float eps, cudaStream_t s) {
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(x, w, out, n, eps);
}

__global__ void silu_mul_f16_kernel(const float* __restrict__ g, const float* __restrict__ u, __half* __restrict__ out,
                                    size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float a = g[i];
    out[i] = __float2half_rn(clamp_h(a / (1.f + expf(-a)) * u[i]));
}

void silu_mul_f16(const float* g, const float* u, __half* out, size_t total, cudaStream_t s) {
    silu_mul_f16_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(g, u, out, total);
}

__global__ void add_f32_kernel(float* __restrict__ x, const float* __restrict__ y, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += y[i];
}

void add_f32(float* x, const float* y, size_t total, cudaStream_t s) {
    add_f32_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(x, y, total);
}

}  // namespace qv::cuda
