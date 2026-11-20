// CPU (AVX2) integer-dot GEMV over Q-Volta bit-plane weights (format: tools/qv_format.py).
//
// Numerics mirror the GPU dp4a path (src/backends/cuda/kernels/gemv_q8.cu), with activation groups of 64 (= the
// weight scale/min group instead of the GPU's 32-chunk):
//   xq[k]  = round_nearest_even(x[k] / d_g),  d_g = max|x_g| / 127           (int8, per group g of 64)
//   y[n]   = sum_g  scale[n,g] * d_g * (sum_k q[n,k] * xq[k])  +  min[n,g] * xsum_g      (xsum_g = fp32 sum of x_g)
// Weights are never materialized as fp32; q bytes are rebuilt in registers from the bit-planes.
//
// Activation byte order inside a group of 64 is a fixed permutation (see Q8ActC) so that the weight bit-planes can be
// expanded to bytes with plain shift/and ops, no cross-lane shuffles.
//
// ---------------------------------------------------------------------------------------------------------------
// ARM / NEON port map (mechanical):
//   _mm256_and_si256 / srli_epi16(.,4)       -> vandq_u8 / vshrq_n_u8(v,4)
//   _mm256_srlv_epi32(v, {0,0,2,2}) & 0x03.. -> vshlq_u32(v, vdupq of negative per-lane shift) (USHL with neg count) & vdupq_n_u8(3)
//   (any 1/2-bit plane expansion by cmpeq)   -> vtstq_u8 (compare-bit-test) or vceqq_u8(vandq_u8(v,mask),mask)
//   pshufb (byte table lookup / broadcast)   -> vqtbl1q_u8 (TBL)
//   _mm256_maddubs_epi16 + _mm256_madd_epi16 -> vdotq_s32 (SDOT/UDOT, FEAT_DotProd; Snapdragon 8 Gen 2 has it):
//                                               int8 x int8 -> int32 in one op, q (<=127) is used as signed int8,
//                                               so no saturation concern and no madd stage.
//   _mm256_cvtepi32_ps / fmadd_ps            -> vcvtq_f32_s32 / vfmaq_f32
//   _mm_cvtph_ps (F16C)                      -> vcvt_f32_f16 (FP16 scale/min)
//   _mm_prefetch                             -> __builtin_prefetch / PRFM
//   NEON registers are 128-bit: process the 64-weight group as 4x16 bytes per row (the permuted activation layout is
//   byte-for-byte the same; only the register width differs). 8-bit weights need uint8 -> vdotq_u32 with unsigned q
//   plus the usual s8 activations (use vusdotq_s32 with I8MM, or widen).
// ---------------------------------------------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <vector>

#include "threadpool.h"

namespace qv::cpu {

// Quantized matrix view (does not own memory). q: uint32[N][K/32][bits], sm: fp16[N][K/64][2] = (scale, min).
struct QMatC {
    const uint32_t* q;
    const uint16_t* sm;
    int N, K, bits;
    size_t row_words() const { return (size_t)(K / 32) * bits; }
    size_t bytes() const { return (size_t)N * row_words() * 4 + (size_t)N * (K / 64) * 4; }
};

// Quantized activation. For group g (64 values) the int8 bytes q[64g .. 64g+63] are stored permuted:
// natural index n = 32*c + 8*b + a (c: 32-chunk 0..1, b 0..3, a 0..7) lives at
//   pos = (a >= 4 ? 32 : 0) + 16*c + 4*(a & 3) + b.
// ds[2g] = d_g (scale), ds[2g+1] = xsum_g (fp32 sum of the unquantized x group).
struct Q8ActC {
    int K = 0;
    std::vector<int8_t> q;
    std::vector<float> ds;
    float d(int g) const { return ds[2 * g]; }
    float sum(int g) const { return ds[2 * g + 1]; }
};

// natural in-group index -> stored position
inline int q8_perm(int n) {
    const int c = n >> 5, r = n & 31, b = r >> 3, a = r & 7;
    return (a >= 4 ? 32 : 0) + 16 * c + 4 * (a & 3) + b;
}

void quantize_act_q8(const float* x, int K, Q8ActC& out);  // AVX2; K % 64 == 0

// Range variant for fused/parallel use: quantizes groups [g0,g1) of x (x points at the START of the vector) into a
// pre-sized `out` (q.size() >= 64*g1, ds.size() >= 2*g1). Disjoint group ranges may run concurrently.
void quantize_act_q8_range(const float* x, int g0, int g1, Q8ActC& out);

// y[row0..row1) = W[row0..row1) . x  (single thread). Optimized AVX2 for bits 2,3,4,5,6,8.
void gemv_q(const QMatC& W, const Q8ActC& x, float* y, int row0, int row1);

// Multi-threaded: rows split statically over the pool (multiples of 4 rows per thread).
void gemv_q_par(ThreadPool& pool, const QMatC& W, const Q8ActC& x, float* y);

}  // namespace qv::cpu
