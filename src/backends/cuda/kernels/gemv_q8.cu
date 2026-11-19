// H2: integer-dot GEMV. Activations are quantized to int8 per 32-chunk (scale dx = amax/127); weights stay in the
// Q-Volta bit-plane format. For every plane width, (word >> s) & 0x0X0X0X0X yields 4 weights {b, b+8, b+16, b+24}
// of a chunk in 4 bytes, so a single canonical x permutation serves all bit widths:
//     xq word b (b = 0..7) of a chunk = int8 x[b], x[b+8], x[b+16], x[b+24]
// y[n] = sum_c  s[n,g(c)] * dx[c] * dp4a(q, xq)  +  m[n,g(c)] * xsum[c]      (xsum = exact fp32 chunk sums)
#include <cstdlib>

#include "../common.cuh"
#include "gemv.h"

namespace qv::cuda {

template <int BITS>
__device__ __forceinline__ uint32_t qword(const uint32_t* w, int b) {
    if constexpr (BITS == 2) {
        return (w[b & 1] >> (2 * (b >> 1))) & 0x03030303u;
    } else if constexpr (BITS == 3) {
        return ((w[b & 1] >> (2 * (b >> 1))) & 0x03030303u) | (((w[2] >> b) & 0x01010101u) << 2);
    } else if constexpr (BITS == 4) {
        return (w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu;
    } else if constexpr (BITS == 5) {
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4] >> b) & 0x01010101u) << 4);
    } else if constexpr (BITS == 6) {
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) |
               (((w[4 + (b & 1)] >> (2 * (b >> 1))) & 0x03030303u) << 4);
    } else {
        static_assert(BITS == 8, "unsupported bits");
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) |
               (((w[4 + (b & 3)] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) << 4);
    }
}

__device__ __forceinline__ int dp4a_us(uint32_t a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int BITS, int R, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) gemv_q8_kernel(QMat W, Q8Vec xv, float* __restrict__ y,
                                                             const float* __restrict__ residual) {
    __shared__ float red[WARPS][R];
    constexpr int ZP = 1 << (BITS - 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * R;
    const int C = W.K >> 5, G = W.K >> 6;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    for (int c = warp * 32 + lane; c < C; c += WARPS * 32) {
        const int4* xp = reinterpret_cast<const int4*>(xv.q + c * 8);
        const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
        const int xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const float dx = __ldg(xv.d + c), xs = __ldg(xv.sum + c);
        const int qs = __ldg(xv.qsum + c);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < W.N) {
                uint32_t w[BITS];
                const uint32_t* wp = W.q + ((size_t)row * C + c) * BITS;
#pragma unroll
                for (int b = 0; b < BITS; ++b) w[b] = __ldg(wp + b);
                int d = 0;
#pragma unroll
                for (int b = 0; b < 8; ++b) d = dp4a_us(qword<BITS>(w, b), xw[b], d);
                // centre q on zp = 2^(BITS-1): the activation-quantization error then multiplies (q - zp), not q
                d -= ZP * qs;
                const float2 sm = __half22float2(W.sm[(size_t)row * G + (c >> 1)]);
                acc[r] = fmaf(sm.x * dx, (float)d, fmaf(fmaf(sm.x, (float)ZP, sm.y), xs, acc[r]));
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
static void launch_q8(const QMat& W, const Q8Vec& x, float* y, const float* res, cudaStream_t s) {
    gemv_q8_kernel<BITS, R, WARPS><<<(W.N + R - 1) / R, WARPS * 32, 0, s>>>(W, x, y, res);
}

template <int BITS>
static void q8_bits(const QMat& W, const Q8Vec& x, float* y, const float* res, GemvCfg cfg, cudaStream_t s) {
#define QV_CFG(R_, W_) \
    if (cfg.rows == R_ && cfg.warps == W_) return launch_q8<BITS, R_, W_>(W, x, y, res, s);
    QV_CFG(1, 4) QV_CFG(2, 2) QV_CFG(2, 4) QV_CFG(2, 8) QV_CFG(4, 1) QV_CFG(4, 2) QV_CFG(4, 4) QV_CFG(4, 8)
    QV_CFG(8, 1) QV_CFG(8, 2) QV_CFG(8, 4) QV_CFG(16, 2) QV_CFG(16, 4)
#undef QV_CFG
    std::fprintf(stderr, "gemv_q8: config rows=%d warps=%d not compiled\n", cfg.rows, cfg.warps);
    std::abort();
}

GemvCfg gemv_q8_default_cfg(const QMat& W) {
    static const char* env = std::getenv("QV_GEMV_CFG");
    if (env) {
        GemvCfg c;
        std::sscanf(env, "%d,%d", &c.rows, &c.warps);
        return c;
    }
    // from benchmarks/bench_gemv q8 sweeps on V100 (logs/gemv_sweep2.txt)
    if (W.N <= 2048) return GemvCfg{2, 4};
    if (W.bits >= 4 && W.N >= 10240) return GemvCfg{2, 2};
    return GemvCfg{1, 4};
}

void gemv_q8(const QMat& W, const Q8Vec& x, float* y, const float* residual, cudaStream_t s, GemvCfg cfg) {
    if (cfg.rows == 0) cfg = gemv_q8_default_cfg(W);
    switch (W.bits) {
        case 2: q8_bits<2>(W, x, y, residual, cfg, s); break;
        case 3: q8_bits<3>(W, x, y, residual, cfg, s); break;
        case 4: q8_bits<4>(W, x, y, residual, cfg, s); break;
        case 5: q8_bits<5>(W, x, y, residual, cfg, s); break;
        case 6: q8_bits<6>(W, x, y, residual, cfg, s); break;
        case 8: q8_bits<8>(W, x, y, residual, cfg, s); break;
        default: std::fprintf(stderr, "gemv_q8: unsupported bits %d\n", W.bits); std::abort();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Multi-token GEMV (MTP verification / short prefill): NT activation vectors share every weight load.
// x: NT Q8 vectors, token t at q + t*K/4 ints, d/sum/qsum + t*K/32; y/residual: [NT][N].
template <int BITS, int R, int WARPS, int NT>
__global__ void __launch_bounds__(WARPS * 32) gemv_q8_multi_kernel(QMat W, Q8Vec xv, float* __restrict__ y,
                                                                   const float* __restrict__ residual) {
    __shared__ float red[WARPS][NT][R];
    constexpr int ZP = 1 << (BITS - 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * R;
    const int C = W.K >> 5, G = W.K >> 6;
    float acc[NT][R];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[t][r] = 0.f;

    for (int c = warp * 32 + lane; c < C; c += WARPS * 32) {
        uint32_t qw[R][8];  // bit-plane decode done once per chunk, reused by all NT tokens
        float2 sm[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(row0 + r, W.N - 1);
            const uint32_t* wp = W.q + ((size_t)row * C + c) * BITS;
            uint32_t w[BITS];
#pragma unroll
            for (int b = 0; b < BITS; ++b) w[b] = __ldg(wp + b);
#pragma unroll
            for (int b = 0; b < 8; ++b) qw[r][b] = qword<BITS>(w, b);
            sm[r] = __half22float2(W.sm[(size_t)row * G + (c >> 1)]);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int4* xp = reinterpret_cast<const int4*>(xv.q + (size_t)t * (W.K >> 2) + c * 8);
            const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
            const int xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const size_t o = (size_t)t * C + c;
            const float dx = __ldg(xv.d + o), xs = __ldg(xv.sum + o);
            const int qs = __ldg(xv.qsum + o);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int d = 0;
#pragma unroll
                for (int b = 0; b < 8; ++b) d = dp4a_us(qw[r][b], xw[b], d);
                d -= ZP * qs;
                acc[t][r] = fmaf(sm[r].x * dx, (float)d, fmaf(fmaf(sm[r].x, (float)ZP, sm[r].y), xs, acc[t][r]));
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = warp_sum(acc[t][r]);
            if (lane == 0) red[warp][t][r] = v;
        }
    __syncthreads();
    if (threadIdx.x < NT * R) {
        const int t = threadIdx.x / R, r = threadIdx.x % R, row = row0 + r;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += red[w][t][r];
        if (row < W.N) {
            const size_t o = (size_t)t * W.N + row;
            y[o] = v + (residual ? residual[o] : 0.f);
        }
    }
}

template <int BITS, int NT>
static void multi_bits(const QMat& W, const Q8Vec& x, float* y, const float* res, cudaStream_t s) {
    // rows per block: reuse each token's x loads across R rows (x traffic is NT x the weight bytes)
    static int Rsel = [] { const char* e = std::getenv("QV_MULTI_R"); return e ? std::atoi(e) : 0; }();
    int R = Rsel ? Rsel : (NT <= 2 ? 2 : 4);
    if (R == 1) gemv_q8_multi_kernel<BITS, 1, 4, NT><<<W.N, 128, 0, s>>>(W, x, y, res);
    else if (R == 2) gemv_q8_multi_kernel<BITS, 2, 4, NT><<<(W.N + 1) / 2, 128, 0, s>>>(W, x, y, res);
    else gemv_q8_multi_kernel<BITS, 4, 4, NT><<<(W.N + 3) / 4, 128, 0, s>>>(W, x, y, res);
}

template <int NT>
static void multi_nt(const QMat& W, const Q8Vec& x, float* y, const float* res, cudaStream_t s) {
    switch (W.bits) {
        case 2: multi_bits<2, NT>(W, x, y, res, s); break;
        case 3: multi_bits<3, NT>(W, x, y, res, s); break;
        case 4: multi_bits<4, NT>(W, x, y, res, s); break;
        case 5: multi_bits<5, NT>(W, x, y, res, s); break;
        case 6: multi_bits<6, NT>(W, x, y, res, s); break;
        case 8: multi_bits<8, NT>(W, x, y, res, s); break;
        default: std::fprintf(stderr, "gemv_q8_multi: unsupported bits %d\n", W.bits); std::abort();
    }
}

void gemv_q8_multi(const QMat& W, const Q8Vec& x, int nt, float* y, const float* residual, cudaStream_t s) {
    switch (nt) {
        case 1: return gemv_q8(W, x, y, residual, s);
        case 2: return multi_nt<2>(W, x, y, residual, s);
        case 3: return multi_nt<3>(W, x, y, residual, s);
        case 4: return multi_nt<4>(W, x, y, residual, s);
        case 5: return multi_nt<5>(W, x, y, residual, s);
        case 6: return multi_nt<6>(W, x, y, residual, s);
        case 8: return multi_nt<8>(W, x, y, residual, s);
        default: std::fprintf(stderr, "gemv_q8_multi: nt=%d not compiled\n", nt); std::abort();
    }
}

// warp per 32-chunk: int8 symmetric quantization in the canonical permutation + exact fp32 chunk sum.
__global__ void quantize_q8_kernel(const float* __restrict__ x, int C, int8_t* __restrict__ q, float* __restrict__ d,
                                   float* __restrict__ sum, int* __restrict__ qsum) {
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, i = threadIdx.x & 31;
    if (c >= C) return;
    const float v = x[c * 32 + i];
    const float amax = warp_max(fabsf(v));
    const float dx = amax / 127.f;
    const int qi = dx > 0.f ? __float2int_rn(v / dx) : 0;
    q[c * 32 + (i & 7) * 4 + (i >> 3)] = (int8_t)qi;
    const float s = warp_sum(v);
    int qs = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o);
    if (i == 0) {
        d[c] = dx;
        sum[c] = s;
        qsum[c] = qs;
    }
}

void quantize_q8(const float* x, int K, Q8Buf& out, cudaStream_t s) {
    const int C = K / 32;
    quantize_q8_kernel<<<(C * 32 + 255) / 256, 256, 0, s>>>(x, C, (int8_t*)out.q, out.d, out.sum, out.qsum);
}

}  // namespace qv::cuda
