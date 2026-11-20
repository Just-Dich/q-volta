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
// Scales are per 32-element sub-group (like the GPU): ds[4g..4g+3] = (d_c0, d_c1, xsum_c0, xsum_c1) of group g;
// sub-group c is natural indices 32c..32c+31. Both sub-groups share the weight (scale,min) of the 64-group.
struct Q8ActC {
    int K = 0;
    std::vector<int8_t> q;
    std::vector<float> ds;
    // sg = 32-element sub-group index (0..K/32-1)
    float d(int sg) const { return ds[4 * (sg >> 1) + (sg & 1)]; }
    float sum(int sg) const { return ds[4 * (sg >> 1) + 2 + (sg & 1)]; }
};

// Batched quantized activations for gemm_q: T rows (tokens) of K values, same per-row numerics as Q8ActC, plus two
// pre-expanded side tables so the GEMM inner loop needs no shuffles:
//   dv8[(t*G + g)*8 + i] = d of 32-sub-group (i >= 4), i.e. (d0,d0,d0,d0,d1,d1,d1,d1) of 64-group g (matches the int32
//   lane order of the dot of one group); xs[t*G + g] = sum0 + sum1 (fp32) for the weight-min term.
struct Q8Batch {
    int T = 0, K = 0, G = 0;  // capacity rows, row stride in elements, K/64
    std::vector<int8_t> q;    // [T][K] (per row: Q8ActC layout)
    std::vector<float> ds;    // [T][G*4]
    std::vector<float> dv8;   // [T][G][8]
    std::vector<float> xs;    // [T][G]
    void resize(int rows, int k) {
        T = rows; K = k; G = k / 64;
        q.assign((size_t)rows * k, 0);
        ds.assign((size_t)rows * G * 4, 0.f);
        dv8.assign((size_t)rows * G * 8, 0.f);
        xs.assign((size_t)rows * G, 0.f);
    }
    // Quantize x[0..kuse) (kuse % 64 == 0, kuse <= K) into row t. Thread-safe for distinct t.
    void quantize_row(int t, const float* x, int kuse);
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
// Same on raw buffers (q: 64 bytes per group, ds: 4 floats per group, both starting at group 0).
void quantize_act_q8_raw(const float* x, int g0, int g1, int8_t* q, float* ds);

// y[row0..row1) = W[row0..row1) . x  (single thread). Optimized AVX2 for bits 2,3,4,5,6,8.
void gemv_q(const QMatC& W, const Q8ActC& x, float* y, int row0, int row1);

// Batched (prefill) GEMM: Y[t*ldy + n] = W[n] . X[t] for rows n in [row0,row1), tokens t in [0,T). Every weight row block is
// unpacked ONCE (bit planes -> int8 in an L1-sized buffer) and multiplied against all T tokens (register tile of 2 rows x
// 4 tokens, K processed in chunks so the unpacked tile and the token tile stay in L1). Per-token results equal gemv_q up to
// fp32 summation order of the weight-min term (<1e-6 relative). bits 8 falls back to per-token gemv_q.
void gemm_q(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1);
// Multi-threaded over the pool: dynamic claims of 8 rows; each claimed block handles all T tokens.
void gemm_q_par(ThreadPool& pool, const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy);

// Multi-threaded: rows split statically over the pool (multiples of 4 rows per thread).
void gemv_q_par(ThreadPool& pool, const QMatC& W, const Q8ActC& x, float* y);

}  // namespace qv::cpu
