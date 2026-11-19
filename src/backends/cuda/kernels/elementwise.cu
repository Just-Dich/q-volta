#include "../common.cuh"
#include "elementwise.h"

namespace qv::cuda {

// out = x * rsqrt(mean(x^2)+eps) * w   (w already holds 1+w for zero-centred norms); also emits xsum32(out).
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                               float* __restrict__ xs32, int n, float eps) {
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    float r = rsqrtf(ss / n + eps);
    // each warp handles whole 32-chunks so it can also produce the chunk sums
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int c = wid; c < n / 32; c += nw) {
        int i = c * 32 + lane;
        float v = x[i] * r * w[i];
        out[i] = v;
        float s = warp_sum(v);
        if (xs32 && lane == 0) xs32[c] = s;
    }
}

void rmsnorm(const float* x, const float* w, float* out, float* xs32, int n, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<1, 1024, 0, s>>>(x, w, out, xs32, n, eps);
}

// Dequantize one embedding row (token id read from device memory) into out[K].
__global__ void embed_kernel(QMat E, const int* __restrict__ tok, float* __restrict__ out) {
    const int row = *tok;
    const int C = E.K >> 5, G = E.K >> 6;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < E.K; k += gridDim.x * blockDim.x) {
        int c = k >> 5, i = k & 31;
        const uint32_t* w = E.q + ((size_t)row * C + c) * E.bits;
        uint32_t q;
        // generic (slow path is fine: 5120 elements once per token)
        switch (E.bits) {
            case 2: q = (w[i & 1] >> (2 * (i >> 1))) & 3u; break;
            case 3: q = ((w[i & 1] >> (2 * (i >> 1))) & 3u) | (((w[2] >> i) & 1u) << 2); break;
            case 4: q = (w[i & 3] >> (4 * (i >> 2))) & 15u; break;
            case 5: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4] >> i) & 1u) << 4); break;
            case 6: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 1)] >> (2 * (i >> 1))) & 3u) << 4); break;
            default: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 3)] >> (4 * (i >> 2))) & 15u) << 4);
        }
        float2 sm = __half22float2(E.sm[(size_t)row * G + (c >> 1)]);
        out[k] = sm.x * (float)q + sm.y;
    }
}

void embed(const QMat& E, const int* tok, float* out, cudaStream_t s) {
    embed_kernel<<<(E.K + 255) / 256, 256, 0, s>>>(E, tok, out);
}

// act = silu(g) * u ; also chunk sums for the following quantized GEMV.
__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out,
                                float* __restrict__ xs32, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (i < n) {
        v = silu(g[i]) * u[i];
        out[i] = v;
    }
    float s = warp_sum(v);
    if ((threadIdx.x & 31) == 0 && i < n) xs32[i >> 5] = s;
}

void silu_mul(const float* g, const float* u, float* out, float* xs32, int n, cudaStream_t s) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, out, xs32, n);
}

// argmax over n floats, single block (first index wins on ties, like torch.argmax).
__global__ void argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        best = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) *out = bi;
    }
}

void argmax(const float* x, int n, int* out, cudaStream_t s) { argmax_kernel<<<1, 1024, 0, s>>>(x, n, out); }

}  // namespace qv::cuda
