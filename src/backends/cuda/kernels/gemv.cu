// Batch-1 GEMV over Q-Volta quantized weights:  y[n] = sum_k W[n,k] x[k]  (+ residual[n])
// W[n,k] = scale[n,g] * q[n,k] + min[n,g]  ⇒  per 32-chunk c:  s * sum(q x) + m * xsum32[c]
#include "../common.cuh"
#include "gemv.h"

namespace qv::cuda {

// Weight index i (0..31) of a 32-weight chunk stored as BITS words of bit-planes (compile-time folded).
template <int BITS>
__device__ __forceinline__ uint32_t qval(const uint32_t* w, int i) {
    if constexpr (BITS == 2) {
        return (w[i & 1] >> (2 * (i >> 1))) & 3u;
    } else if constexpr (BITS == 3) {
        return ((w[i & 1] >> (2 * (i >> 1))) & 3u) | (((w[2] >> i) & 1u) << 2);
    } else if constexpr (BITS == 4) {
        return (w[i & 3] >> (4 * (i >> 2))) & 15u;
    } else if constexpr (BITS == 5) {
        return ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4] >> i) & 1u) << 4);
    } else if constexpr (BITS == 6) {
        return ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 1)] >> (2 * (i >> 1))) & 3u) << 4);
    } else {
        static_assert(BITS == 8, "unsupported bits");
        return ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 3)] >> (4 * (i >> 2))) & 15u) << 4);
    }
}

// int -> float without I2F: bit pattern 2^23 + q, minus 2^23 (exact for q < 2^23).
__device__ __forceinline__ float u2f(uint32_t q) { return __uint_as_float(0x4B000000u | q) - 8388608.f; }

template <int BITS, int R>
__global__ void __launch_bounds__(128) gemv_q_kernel(QMat W, const float* __restrict__ x,
                                                     const float* __restrict__ xsum32, float* __restrict__ y,
                                                     const float* __restrict__ residual) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= W.N) return;
    const int C = W.K >> 5;
    const int G = W.K >> 6;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    for (int c = lane; c < C; c += 32) {
        float xv[32];
        const float4* xp = reinterpret_cast<const float4*>(x + c * 32);
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            float4 t = __ldg(xp + v);
            xv[4 * v] = t.x; xv[4 * v + 1] = t.y; xv[4 * v + 2] = t.z; xv[4 * v + 3] = t.w;
        }
        const float xs = __ldg(xsum32 + c);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < W.N) {
                uint32_t w[BITS];
                const uint32_t* wp = W.q + ((size_t)row * C + c) * BITS;
#pragma unroll
                for (int b = 0; b < BITS; ++b) w[b] = __ldg(wp + b);
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < 32; ++i) d = fmaf(u2f(qval<BITS>(w, i)), xv[i], d);
                float2 sm = __half22float2(W.sm[(size_t)row * G + (c >> 1)]);
                acc[r] = fmaf(sm.x, d, fmaf(sm.y, xs, acc[r]));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = warp_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < W.N) y[row] = v + (residual ? residual[row] : 0.f);
    }
}

template <int BITS>
static void launch_bits(const QMat& W, const float* x, const float* xsum32, float* y, const float* res,
                        cudaStream_t s) {
    constexpr int R = 1;
    int rows_per_block = 4 * R;
    int blocks = (W.N + rows_per_block - 1) / rows_per_block;
    gemv_q_kernel<BITS, R><<<blocks, 128, 0, s>>>(W, x, xsum32, y, res);
}

void gemv_q(const QMat& W, const float* x, const float* xsum32, float* y, const float* residual, cudaStream_t s) {
    switch (W.bits) {
        case 2: launch_bits<2>(W, x, xsum32, y, residual, s); break;
        case 3: launch_bits<3>(W, x, xsum32, y, residual, s); break;
        case 4: launch_bits<4>(W, x, xsum32, y, residual, s); break;
        case 5: launch_bits<5>(W, x, xsum32, y, residual, s); break;
        case 6: launch_bits<6>(W, x, xsum32, y, residual, s); break;
        case 8: launch_bits<8>(W, x, xsum32, y, residual, s); break;
        default: std::fprintf(stderr, "gemv_q: unsupported bits %d\n", W.bits); std::abort();
    }
}

__global__ void xsum32_kernel(const float* __restrict__ x, float* __restrict__ out, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float4* p = reinterpret_cast<const float4*>(x + c * 32);
    float s = 0.f;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        float4 t = p[v];
        s += (t.x + t.y) + (t.z + t.w);
    }
    out[c] = s;
}

void xsum32(const float* x, float* out, int K, cudaStream_t s) {
    int C = K / 32;
    xsum32_kernel<<<(C + 127) / 128, 128, 0, s>>>(x, out, C);
}

// Dense fp16-weight GEMV (tiny matrices: GDN in_proj_a / in_proj_b). Warp per row.
__global__ void gemv_f16_kernel(DMat16 W, const float* __restrict__ x, float* __restrict__ y) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= W.N) return;
    const __half2* w = reinterpret_cast<const __half2*>(W.w + (size_t)warp * W.K);
    const float2* xv = reinterpret_cast<const float2*>(x);
    float acc = 0.f;
    for (int k = lane; k < W.K / 2; k += 32) {
        float2 a = __half22float2(w[k]);
        float2 b = xv[k];
        acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
    }
    acc = warp_sum(acc);
    if (lane == 0) y[warp] = acc;
}

void gemv_f16(const DMat16& W, const float* x, float* y, cudaStream_t s) {
    int threads = 128;
    int blocks = (W.N * 32 + threads - 1) / threads;
    gemv_f16_kernel<<<blocks, threads, 0, s>>>(W, x, y);
}

}  // namespace qv::cuda
