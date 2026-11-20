// Dequantize bit-plane quantized rows to fp16 (feeds cuBLAS HGEMM in batched prefill).
// Work split: 4 consecutive lanes share one 32-weight chunk; lane `sub` produces weights 8*sub..8*sub+7 and
// writes them as one 16-byte store, so a warp writes 512 contiguous bytes per store instruction.
// For weight i = 8*sub + k, in a plane of width w (1, 2 or 4): word index = i % w = k % w (static) and the
// field shift = w*(i/w) = 8*sub + k - k % w, i.e. a per-lane base plus a compile-time constant.
#include "dequant.h"

namespace qv::cuda {

template <int BITS>
__global__ void __launch_bounds__(256) dequant_kernel(QMat W, int row0, long long total_threads, __half* __restrict__ out) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_threads) return;
    const int sub = (int)(t & 3);
    const long long chunk = t >> 2;  // global chunk index within the output block
    const int C = W.K >> 5;
    const long long gchunk = chunk + (long long)row0 * C;  // chunk index in W
    const uint32_t* wp = W.q + gchunk * BITS;
    uint32_t w[BITS];
#pragma unroll
    for (int b = 0; b < BITS; ++b) w[b] = __ldg(wp + b);
    const float2 sm = __half22float2(W.sm[gchunk >> 1]);
    const int base = 8 * sub;

    // plane layout per bits: list of (width, shift, first word)
    int q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t v;
        if constexpr (BITS == 2) {
            v = (w[k & 1] >> (base + k - (k & 1))) & 3u;
        } else if constexpr (BITS == 3) {
            v = ((w[k & 1] >> (base + k - (k & 1))) & 3u) | (((w[2] >> (base + k)) & 1u) << 2);
        } else if constexpr (BITS == 4) {
            v = (w[k & 3] >> (base + k - (k & 3))) & 15u;
        } else if constexpr (BITS == 5) {
            v = ((w[k & 3] >> (base + k - (k & 3))) & 15u) | (((w[4] >> (base + k)) & 1u) << 4);
        } else if constexpr (BITS == 6) {
            v = ((w[k & 3] >> (base + k - (k & 3))) & 15u) | (((w[4 + (k & 1)] >> (base + k - (k & 1))) & 3u) << 4);
        } else {
            static_assert(BITS == 8, "unsupported bits");
            v = ((w[k & 3] >> (base + k - (k & 3))) & 15u) | (((w[4 + (k & 3)] >> (base + k - (k & 3))) & 15u) << 4);
        }
        q[k] = (int)v;
    }
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float a = __fmaf_rn(sm.x, (float)q[2 * k], sm.y);
        const float b = __fmaf_rn(sm.x, (float)q[2 * k + 1], sm.y);
        h[k] = __floats2half2_rn(a, b);
    }
    uint4 o;
    o.x = *reinterpret_cast<uint32_t*>(&h[0]);
    o.y = *reinterpret_cast<uint32_t*>(&h[1]);
    o.z = *reinterpret_cast<uint32_t*>(&h[2]);
    o.w = *reinterpret_cast<uint32_t*>(&h[3]);
    reinterpret_cast<uint4*>(out)[t] = o;
}

template <int BITS>
static void launch(const QMat& W, int row0, int nrows, __half* out, cudaStream_t s) {
    const long long total = (long long)nrows * (W.K >> 3);
    const long long blocks = (total + 255) / 256;
    dequant_kernel<BITS><<<(unsigned)blocks, 256, 0, s>>>(W, row0, total, out);
}

void dequant_rows_f16(const QMat& W, int row0, int nrows, __half* out, cudaStream_t s) {
    if (nrows <= 0) return;
    if (W.K % 64 != 0 || row0 < 0 || row0 + nrows > W.N) {
        std::fprintf(stderr, "dequant_rows_f16: bad shape K=%d row0=%d nrows=%d N=%d\n", W.K, row0, nrows, W.N);
        std::abort();
    }
    switch (W.bits) {
        case 2: launch<2>(W, row0, nrows, out, s); break;
        case 3: launch<3>(W, row0, nrows, out, s); break;
        case 4: launch<4>(W, row0, nrows, out, s); break;
        case 5: launch<5>(W, row0, nrows, out, s); break;
        case 6: launch<6>(W, row0, nrows, out, s); break;
        case 8: launch<8>(W, row0, nrows, out, s); break;
        default: std::fprintf(stderr, "dequant_rows_f16: unsupported bits %d\n", W.bits); std::abort();
    }
}

}  // namespace qv::cuda
