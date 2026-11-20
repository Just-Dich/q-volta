// Batch-1 GEMV over Q-Volta quantized weights:  y[n] = sum_k W[n,k] x[k]  (+ residual[n])
// W[n,k] = scale[n,g] * q[n,k] + min[n,g]  ⇒  per 32-chunk c:  s * sum(q x) + m * xsum32[c]
#include "../common.cuh"
#include "gemv.h"
#include <cstdlib>

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

// Block = WARPS warps that all work on the same R rows and split the K chunks between them (split-K inside the
// block, reduced through shared memory). R rows share every x load; WARPS trades rows-per-block for parallelism.
template <int BITS, int R, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) gemv_q_kernel(QMat W, const float* __restrict__ x,
                                                            const float* __restrict__ xsum32, float* __restrict__ y,
                                                            const float* __restrict__ residual) {
    __shared__ float red[WARPS][R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * R;
    const int C = W.K >> 5;
    const int G = W.K >> W.gshift, GS = W.gshift - 5;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    for (int c = warp * 32 + lane; c < C; c += WARPS * 32) {
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
                float2 sm = load_sm(W, row, G, c >> GS);
                acc[r] = fmaf(sm.x, d, fmaf(sm.y, xs, acc[r]));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = warp_sum(acc[r]);
        if (lane == 0) red[warp][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < R) {
        const int r = threadIdx.x, row = row0 + r;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += red[w][r];
        if (row < W.N) y[row] = v + (residual ? residual[row] : 0.f);
    }
}

template <int BITS, int R, int WARPS>
static void launch_cfg(const QMat& W, const float* x, const float* xsum32, float* y, const float* res, cudaStream_t s) {
    gemv_q_kernel<BITS, R, WARPS><<<(W.N + R - 1) / R, WARPS * 32, 0, s>>>(W, x, xsum32, y, res);
}

template <int BITS>
static void launch_bits(const QMat& W, const float* x, const float* xsum32, float* y, const float* res,
                        GemvCfg cfg, cudaStream_t s) {
#define QV_CFG(R_, W_) \
    if (cfg.rows == R_ && cfg.warps == W_) return launch_cfg<BITS, R_, W_>(W, x, xsum32, y, res, s);
    QV_CFG(1, 4) QV_CFG(2, 2) QV_CFG(2, 4) QV_CFG(2, 8) QV_CFG(4, 1) QV_CFG(4, 2) QV_CFG(4, 4) QV_CFG(4, 8)
    QV_CFG(8, 1) QV_CFG(8, 2) QV_CFG(8, 4) QV_CFG(16, 2) QV_CFG(16, 4)
#undef QV_CFG
    std::fprintf(stderr, "gemv_q: config rows=%d warps=%d not compiled\n", cfg.rows, cfg.warps);
    std::abort();
}

// Per-shape choice (filled from benchmarks/bench_gemv sweeps; see docs/GPU_OPTIMIZATION.md).
GemvCfg gemv_default_cfg(const QMat& W) {
    static const char* env = std::getenv("QV_GEMV_CFG");  // "R,W" override for sweeps
    if (env) {
        GemvCfg c;
        std::sscanf(env, "%d,%d", &c.rows, &c.warps);
        return c;
    }
    return GemvCfg{4, 4};
}

void gemv_q(const QMat& W, const float* x, const float* xsum32, float* y, const float* residual, cudaStream_t s,
            GemvCfg cfg) {
    if (cfg.rows == 0) cfg = gemv_default_cfg(W);
    switch (W.bits) {
        case 2: launch_bits<2>(W, x, xsum32, y, residual, cfg, s); break;
        case 3: launch_bits<3>(W, x, xsum32, y, residual, cfg, s); break;
        case 4: launch_bits<4>(W, x, xsum32, y, residual, cfg, s); break;
        case 5: launch_bits<5>(W, x, xsum32, y, residual, cfg, s); break;
        case 6: launch_bits<6>(W, x, xsum32, y, residual, cfg, s); break;
        case 8: launch_bits<8>(W, x, xsum32, y, residual, cfg, s); break;
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

// Two small dense fp16 matrices (GDN in_proj_a / in_proj_b, N=48 rows each) in one launch: block per row, the whole
// block splits K (latency-bound shape: 96 rows only, so each row needs many threads in flight).
__global__ void __launch_bounds__(256) gemv_f16_dual_kernel(DMat16 A, DMat16 B, const float* __restrict__ x,
                                                            float* __restrict__ ya, float* __restrict__ yb) {
    __shared__ float red[32];
    const bool first = blockIdx.x < A.N;
    const DMat16& W = first ? A : B;
    const int row = first ? blockIdx.x : blockIdx.x - A.N;
    x += (size_t)blockIdx.y * A.K;  // token row (nt > 1: MTP verify); x stride = K, outputs stride = N
    ya += (size_t)blockIdx.y * A.N;
    yb += (size_t)blockIdx.y * B.N;
    const uint2* w = reinterpret_cast<const uint2*>(W.w + (size_t)row * W.K);  // 4 halves per load
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
    for (int k = threadIdx.x; k < W.K / 4; k += blockDim.x) {
        uint2 p = __ldg(w + k);
        float2 a = __half22float2(*reinterpret_cast<__half2*>(&p.x));
        float2 b = __half22float2(*reinterpret_cast<__half2*>(&p.y));
        float4 xx = __ldg(xv + k);
        acc = fmaf(a.x, xx.x, fmaf(a.y, xx.y, fmaf(b.x, xx.z, fmaf(b.y, xx.w, acc))));
    }
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) (first ? ya : yb)[row] = acc;
}

void gemv_f16_dual(const DMat16& A, const DMat16& B, const float* x, float* ya, float* yb, cudaStream_t s, int nt) {
    gemv_f16_dual_kernel<<<dim3(A.N + B.N, nt), 256, 0, s>>>(A, B, x, ya, yb);
}

void gemv_f16(const DMat16& W, const float* x, float* y, cudaStream_t s) {
    int threads = 128;
    int blocks = (W.N * 32 + threads - 1) / threads;
    gemv_f16_kernel<<<blocks, threads, 0, s>>>(W, x, y);
}

}  // namespace qv::cuda
