#include "cpu_gemv.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qv::cpu {

namespace {
[[noreturn]] void throw_too_many_tokens() { throw std::invalid_argument("gemv_multi_q: T > 4"); }
}  // namespace

void quantize_act_q8(const float* x, int K, Q8ActC& out) {
    out.K = K;
    out.q.assign((size_t)K, 0);
    out.ds.assign((size_t)(K / 64) * 4, 0.f);
    quantize_act_q8_range(x, 0, K / 64, out);
}

void quantize_act_q8_range(const float* x, int g0, int g1, Q8ActC& out) {
    quantize_act_q8_raw(x, g0, g1, out.q.data(), out.ds.data());
}

void quantize_act_q8_raw(const float* x, int g0, int g1, int8_t* qout, float* dsout) {
    static int perm[64];
    static bool init = [] { for (int n = 0; n < 64; ++n) perm[n] = q8_perm(n); return true; }();
    (void)init;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    alignas(32) int8_t nat[64];
    for (int g = g0; g < g1; ++g) {
        const float* xg = x + (size_t)g * 64;
        __m256 v[8];
        __m256 mx = _mm256_setzero_ps(), sm = _mm256_setzero_ps();
        for (int i = 0; i < 8; ++i) {
            v[i] = _mm256_loadu_ps(xg + 8 * i);
            mx = _mm256_max_ps(mx, _mm256_and_ps(v[i], absmask));
            sm = _mm256_add_ps(sm, v[i]);
        }
        // 32-element sub-groups (one per 128-bit lane of the stored layout): sub-group c = v[4c..4c+3]
        float dsub[2], ssub[2];
        for (int c = 0; c < 2; ++c) {
            __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v[4 * c], absmask), _mm256_and_ps(v[4 * c + 1], absmask)),
                                     _mm256_max_ps(_mm256_and_ps(v[4 * c + 2], absmask), _mm256_and_ps(v[4 * c + 3], absmask)));
            __m256 sm4 = _mm256_add_ps(_mm256_add_ps(v[4 * c], v[4 * c + 1]), _mm256_add_ps(v[4 * c + 2], v[4 * c + 3]));
            __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
            __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(sm4), _mm256_extractf128_ps(sm4, 1));
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
            dsub[c] = _mm_cvtss_f32(m4) / 127.f;
            ssub[c] = _mm_cvtss_f32(s4);
        }
        dsout[4 * g] = dsub[0];
        dsout[4 * g + 1] = dsub[1];
        dsout[4 * g + 2] = ssub[0];
        dsout[4 * g + 3] = ssub[1];
        if (!(dsub[0] > 0.f) && !(dsub[1] > 0.f)) std::memset(qout + (size_t)g * 64, 0, 64);
        else {
            // zero sub-group: divide by 1 (all inputs are 0 there)
            const __m256 d0 = _mm256_set1_ps(dsub[0] > 0.f ? dsub[0] : 1.f), d1 = _mm256_set1_ps(dsub[1] > 0.f ? dsub[1] : 1.f);
            __m256i qi[4];
            for (int i = 0; i < 4; ++i) {
                const __m256 dv = i < 2 ? d0 : d1;
                qi[i] = _mm256_packs_epi32(_mm256_cvtps_epi32(_mm256_div_ps(v[2 * i], dv)),
                                           _mm256_cvtps_epi32(_mm256_div_ps(v[2 * i + 1], dv)));
            }
            // packs_epi32 interleaves 128-bit lanes: fix order while packing to bytes
            __m256i a = _mm256_packs_epi16(qi[0], qi[1]);
            __m256i b = _mm256_packs_epi16(qi[2], qi[3]);
            const __m256i fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            _mm256_store_si256((__m256i*)nat, _mm256_permutevar8x32_epi32(a, fix));
            _mm256_store_si256((__m256i*)(nat + 32), _mm256_permutevar8x32_epi32(b, fix));
            int8_t* o = qout + (size_t)g * 64;
            for (int n = 0; n < 64; ++n) o[perm[n]] = nat[n];
        }
    }
}

namespace {

inline __m256i ld2(const uint32_t* p0, const uint32_t* p1) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)p0)),
                                   _mm_loadu_si128((const __m128i*)p1), 1);
}
inline __m256i ld2_64(const uint32_t* p0, const uint32_t* p1) {
    __m256i v = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*)p0)),
                                        _mm_loadl_epi64((const __m128i*)p1), 1);
    return _mm256_shuffle_epi32(v, 0x44);  // [w0 w1 w0 w1] per lane
}
inline __m256i ld2_32(const uint32_t* p0, const uint32_t* p1) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32((int)*p0)), _mm_set1_epi32((int)*p1), 1);
}

// 4-bit plane (4 words per 32-chunk) at word offset `off` -> lo/hi byte vectors (values 0..15)
template <int BITS>
inline void plane4(const uint32_t* p, int off, __m256i& lo, __m256i& hi) {
    const __m256i m = _mm256_set1_epi8(0x0f);
    __m256i w = (BITS == 4) ? _mm256_loadu_si256((const __m256i*)p) : ld2(p + off, p + BITS + off);
    lo = _mm256_and_si256(w, m);
    hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), m);
}
// 2-bit plane (2 words per chunk) at `off`, result shifted left by `sh`
template <int BITS>
inline void plane2(const uint32_t* p, int off, int sh, __m256i& lo, __m256i& hi) {
    const __m256i v = ld2_64(p + off, p + BITS + off);
    const __m256i m = _mm256_set1_epi8(3);
    lo = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(0, 0, 2, 2, 0, 0, 2, 2)), m);
    hi = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(4, 4, 6, 6, 4, 4, 6, 6)), m);
    lo = _mm256_slli_epi16(lo, sh);
    hi = _mm256_slli_epi16(hi, sh);
}
// 1-bit plane (1 word per chunk) at `off`
template <int BITS>
inline void plane1(const uint32_t* p, int off, int sh, __m256i& lo, __m256i& hi) {
    const __m256i v = ld2_32(p + off, p + BITS + off);
    const __m256i m = _mm256_set1_epi8(1);
    lo = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)), m);
    hi = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(4, 5, 6, 7, 4, 5, 6, 7)), m);
    lo = _mm256_slli_epi16(lo, sh);
    hi = _mm256_slli_epi16(hi, sh);
}

// Rebuild the 64 q bytes of one group (p = first word, 2*BITS words) as two byte vectors that pair with the activation
// halves (xlo, xhi). BITS != 8 only (q <= 63).
template <int BITS>
inline void unpack_group(const uint32_t* p, __m256i& lo, __m256i& hi) {
    if constexpr (BITS == 4) {
        plane4<4>(p, 0, lo, hi);
    } else if constexpr (BITS == 6) {
        __m256i l2, h2;
        plane4<6>(p, 0, lo, hi);
        plane2<6>(p, 4, 4, l2, h2);
        lo = _mm256_or_si256(lo, l2);
        hi = _mm256_or_si256(hi, h2);
    } else if constexpr (BITS == 5) {
        __m256i l2, h2;
        plane4<5>(p, 0, lo, hi);
        plane1<5>(p, 4, 4, l2, h2);
        lo = _mm256_or_si256(lo, l2);
        hi = _mm256_or_si256(hi, h2);
    } else if constexpr (BITS == 3) {
        __m256i l2, h2;
        plane2<3>(p, 0, 0, lo, hi);
        plane1<3>(p, 2, 2, l2, h2);
        lo = _mm256_or_si256(lo, l2);
        hi = _mm256_or_si256(hi, h2);
    } else {  // 2
        plane2<2>(p, 0, 0, lo, hi);
    }
}

// int32x8 partial dot of one 64-weight group (p = first word of the group, 2*BITS words).
template <int BITS>
inline __m256i dot_group(const uint32_t* p, __m256i xlo, __m256i xhi) {
    const __m256i ones = _mm256_set1_epi16(1);
    if constexpr (BITS == 8) {
        __m256i alo, ahi, blo, bhi;
        plane4<8>(p, 0, alo, ahi);
        plane4<8>(p, 4, blo, bhi);
        __m256i sa = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(alo, xlo), _mm256_maddubs_epi16(ahi, xhi)), ones);
        __m256i sb = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(blo, xlo), _mm256_maddubs_epi16(bhi, xhi)), ones);
        return _mm256_add_epi32(sa, _mm256_slli_epi32(sb, 4));
    } else {
        __m256i lo, hi;
        unpack_group<BITS>(p, lo, hi);
        // q <= 63: pair sums (<=16002) and the add of two (<=32004) fit in int16 without saturation.
        __m256i s = _mm256_add_epi16(_mm256_maddubs_epi16(lo, xlo), _mm256_maddubs_epi16(hi, xhi));
        return _mm256_madd_epi16(s, ones);
    }
}

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

template <int BITS, int R>
void rows_kernel(const QMatC& W, const Q8ActC& x, float* y, int r0) {
    const int G = W.K / 64;
    const size_t rw = W.row_words();
    const uint32_t* wp[R];
    const uint16_t* sp[R];
    __m256 accf[R];
    __m128 accm[R];
    for (int r = 0; r < R; ++r) {
        wp[r] = W.q + (size_t)(r0 + r) * rw;
        sp[r] = W.sm + (size_t)(r0 + r) * (G >> W.gs) * 2;
        accf[r] = _mm256_setzero_ps();
        accm[r] = _mm_setzero_ps();
    }
    const int8_t* xq = x.q.data();
    const float* ds = x.ds.data();
    constexpr int PF = 64;  // words ahead (256 B)
    for (int g = 0; g < G; ++g) {
        const __m256i xlo = _mm256_loadu_si256((const __m256i*)(xq + 64 * g));
        const __m256i xhi = _mm256_loadu_si256((const __m256i*)(xq + 64 * g + 32));
        // (d0, d1, sum0, sum1) of the two 32-element activation sub-groups; sub-group c lives in 128-bit lane c of the
        // stored layout, hence in int32 lanes 4c..4c+3 of dot_group's result
        const __m128 dsg = _mm_loadu_ps(ds + 4 * g);
        const __m256 dvec = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_shuffle_ps(dsg, dsg, 0x00)), _mm_shuffle_ps(dsg, dsg, 0x55), 1);
        const __m128 ssum = _mm_add_ss(_mm_shuffle_ps(dsg, dsg, 2), _mm_shuffle_ps(dsg, dsg, 3));
        for (int r = 0; r < R; ++r) {
            const uint32_t* p = wp[r] + (size_t)g * 2 * BITS;
            _mm_prefetch((const char*)(p + PF), _MM_HINT_T0);
            const __m256i d = dot_group<BITS>(p, xlo, xhi);
            uint32_t smw;
            std::memcpy(&smw, sp[r] + 2 * (g >> W.gs), 4);
            const __m128 t = _mm_cvtph_ps(_mm_cvtsi32_si128((int)smw));  // (scale, min)
            accm[r] = _mm_add_ps(accm[r], _mm_mul_ss(_mm_shuffle_ps(t, t, 1), ssum));
            accf[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(dvec, _mm256_broadcastss_ps(t)), accf[r]);
        }
    }
    for (int r = 0; r < R; ++r)
        y[r0 + r] = hsum(accf[r]) + _mm_cvtss_f32(accm[r]);
}


// Multi-token GEMV: identical arithmetic (operation for operation) to rows_kernel for each token, but the weight bit
// planes of a row/group are unpacked once and applied to T tokens (T <= 4). Results are bit-identical to gemv_q per token,
// which is what exact speculative-decoding verification needs (gemm_q differs in the weight-min summation order).
template <int BITS, int R, int T>
void rows_kernel_multi(const QMatC& W, const Q8Batch& X, float* Y, size_t ldy, int r0) {
    const int G = W.K / 64;
    const size_t rw = W.row_words();
    const uint32_t* wp[R];
    const uint16_t* sp[R];
    __m256 accf[R][T];
    __m128 accm[R][T];
    for (int r = 0; r < R; ++r) {
        wp[r] = W.q + (size_t)(r0 + r) * rw;
        sp[r] = W.sm + (size_t)(r0 + r) * (G >> W.gs) * 2;
        for (int t = 0; t < T; ++t) { accf[r][t] = _mm256_setzero_ps(); accm[r][t] = _mm_setzero_ps(); }
    }
    constexpr int PF = 64;
    for (int g = 0; g < G; ++g) {
        __m256i xlo[T], xhi[T];
        __m256 dvec[T];
        __m128 ssum[T];
        for (int t = 0; t < T; ++t) {
            const int8_t* xq = X.q.data() + (size_t)t * X.K;
            const float* ds = X.ds.data() + (size_t)t * X.G * 4;
            xlo[t] = _mm256_loadu_si256((const __m256i*)(xq + 64 * g));
            xhi[t] = _mm256_loadu_si256((const __m256i*)(xq + 64 * g + 32));
            const __m128 dsg = _mm_loadu_ps(ds + 4 * g);
            dvec[t] = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_shuffle_ps(dsg, dsg, 0x00)), _mm_shuffle_ps(dsg, dsg, 0x55), 1);
            ssum[t] = _mm_add_ss(_mm_shuffle_ps(dsg, dsg, 2), _mm_shuffle_ps(dsg, dsg, 3));
        }
        const __m256i ones = _mm256_set1_epi16(1);
        for (int r = 0; r < R; ++r) {
            const uint32_t* p = wp[r] + (size_t)g * 2 * BITS;
            _mm_prefetch((const char*)(p + PF), _MM_HINT_T0);
            __m256i lo, hi;
            unpack_group<BITS>(p, lo, hi);
            uint32_t smw;
            std::memcpy(&smw, sp[r] + 2 * (g >> W.gs), 4);
            const __m128 tt = _mm_cvtph_ps(_mm_cvtsi32_si128((int)smw));
            const __m256 sc = _mm256_broadcastss_ps(tt);
            for (int t = 0; t < T; ++t) {
                const __m256i s = _mm256_add_epi16(_mm256_maddubs_epi16(lo, xlo[t]), _mm256_maddubs_epi16(hi, xhi[t]));
                const __m256i d = _mm256_madd_epi16(s, ones);
                accm[r][t] = _mm_add_ps(accm[r][t], _mm_mul_ss(_mm_shuffle_ps(tt, tt, 1), ssum[t]));
                accf[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(dvec[t], sc), accf[r][t]);
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int t = 0; t < T; ++t) Y[(size_t)t * ldy + r0 + r] = hsum(accf[r][t]) + _mm_cvtss_f32(accm[r][t]);
}

template <int BITS, int T>
void multi_bits_T(const QMatC& W, const Q8Batch& X, float* Y, size_t ldy, int row0, int row1) {
    int r = row0;
    for (; r + 4 <= row1; r += 4) rows_kernel_multi<BITS, 4, T>(W, X, Y, ldy, r);
    for (; r < row1; ++r) rows_kernel_multi<BITS, 1, T>(W, X, Y, ldy, r);
}
template <int BITS>
void multi_bits(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1) {
    switch (T) {
        case 1: multi_bits_T<BITS, 1>(W, X, Y, ldy, row0, row1); break;
        case 2: multi_bits_T<BITS, 2>(W, X, Y, ldy, row0, row1); break;
        case 3: multi_bits_T<BITS, 3>(W, X, Y, ldy, row0, row1); break;
        default: multi_bits_T<BITS, 4>(W, X, Y, ldy, row0, row1); break;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Batched GEMM. Loop nest per claimed row block (RB rows, one thread):
//   for K chunk (KCG groups):  unpack RB rows x KCG groups -> int8 buffer (once)
//     for token tile (TB tokens): for row tile (RT rows): load acc, run KCG groups, store acc
// Accumulators (fp32 x8 lanes per (row, token)) live in a per-thread array between chunks; the weight-min term
// sum_g min[n,g]*xsum[t,g] is added at the end.
#ifndef QV_GEMM_TB
#define QV_GEMM_TB 2
#endif
#ifndef QV_GEMM_RT
#define QV_GEMM_RT 4
#endif
constexpr int kRT = QV_GEMM_RT;  // rows per register tile
#ifndef QV_GEMM_RB
#define QV_GEMM_RB 16
#endif
#ifndef QV_GEMM_TP
#define QV_GEMM_TP 64
#endif
constexpr int kRB = QV_GEMM_RB;  // rows per claimed block
constexpr int kTP = QV_GEMM_TP;  // tokens per panel
constexpr int kKCG = 16;  // 64-groups per K chunk (1024 weights)
constexpr int kTB = QV_GEMM_TB;

template <int RT, int TB>
inline void tile_kernel(const int8_t* wb, size_t wrow, const float* sc, size_t scrow, const int8_t* xq, size_t xqrow,
                        const float* dv, size_t dvrow, float* acc, size_t accrow, int ng) {
    __m256 a[RT][TB];
    for (int r = 0; r < RT; ++r)
        for (int t = 0; t < TB; ++t) a[r][t] = _mm256_loadu_ps(acc + r * accrow + t * 8);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int g = 0; g < ng; ++g) {
        __m256i wlo[RT], whi[RT];
        __m256 s[RT];
        for (int r = 0; r < RT; ++r) {
            wlo[r] = _mm256_load_si256((const __m256i*)(wb + r * wrow + 64 * g));
            whi[r] = _mm256_load_si256((const __m256i*)(wb + r * wrow + 64 * g + 32));
            s[r] = _mm256_broadcast_ss(sc + r * scrow + g);
        }
        for (int t = 0; t < TB; ++t) {
            const __m256i xlo = _mm256_loadu_si256((const __m256i*)(xq + t * xqrow + 64 * g));
            const __m256i xhi = _mm256_loadu_si256((const __m256i*)(xq + t * xqrow + 64 * g + 32));
            const __m256 d = _mm256_loadu_ps(dv + t * dvrow + 8 * g);
            for (int r = 0; r < RT; ++r) {
                const __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(wlo[r], xlo), _mm256_maddubs_epi16(whi[r], xhi));
                a[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), _mm256_mul_ps(d, s[r]), a[r][t]);
            }
        }
    }
    for (int r = 0; r < RT; ++r)
        for (int t = 0; t < TB; ++t) _mm256_storeu_ps(acc + r * accrow + t * 8, a[r][t]);
}

template <int RT, int TB>
inline void tile_dispatch(int tn, const int8_t* wb, size_t wrow, const float* sc, size_t scrow, const int8_t* xq, size_t xqrow,
                          const float* dv, size_t dvrow, float* acc, size_t accrow, int ng) {
    if (tn == TB) tile_kernel<RT, TB>(wb, wrow, sc, scrow, xq, xqrow, dv, dvrow, acc, accrow, ng);
    else if constexpr (TB > 1) tile_dispatch<RT, TB - 1>(tn, wb, wrow, sc, scrow, xq, xqrow, dv, dvrow, acc, accrow, ng);
}

template <int RT>
inline void tile_tokens(const int8_t* wb, size_t wrow, const float* sc, size_t scrow, const Q8Batch& X, int t0, int tn,
                        int g0, int ng, float* acc, size_t accrow) {
    const size_t xqrow = (size_t)X.K, dvrow = (size_t)X.G * 8;
    const int8_t* xq = X.q.data() + (size_t)t0 * xqrow + 64 * (size_t)g0;
    const float* dv = X.dv8.data() + (size_t)t0 * dvrow + 8 * (size_t)g0;
    tile_dispatch<RT, kTB>(tn, wb, wrow, sc, scrow, xq, xqrow, dv, dvrow, acc, accrow, ng);
}

template <int BITS>
void gemm_block(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int r0, int nr, int xoff) {
    const int G = W.K / 64;
    const size_t rw = W.row_words();
    alignas(64) static thread_local int8_t wb[kRB * kKCG * 64];
    static thread_local std::vector<float> accv, scv, mnv;
    scv.resize((size_t)nr * G);
    mnv.resize((size_t)nr * G);
    for (int r = 0; r < nr; ++r) {
        const uint16_t* sp = W.sm + (size_t)(r0 + r) * (G >> W.gs) * 2;
        for (int g = 0; g < G; ++g) {
            scv[(size_t)r * G + g] = _cvtsh_ss(sp[2 * (g >> W.gs)]);
            mnv[(size_t)r * G + g] = _cvtsh_ss(sp[2 * (g >> W.gs) + 1]);
        }
    }
    // Token panels of kTP tokens bound the accumulator array (nr x kTP x 8 fp32) and the working set of the activation
    // chunk; the weight rows are re-unpacked per panel (~1-3% of the MAC work for kTP=64).
    for (int tp0 = 0; tp0 < T; tp0 += kTP) {
        const int tpn = T - tp0 < kTP ? T - tp0 : kTP;
        const size_t accrow = (size_t)tpn * 8;
        accv.assign((size_t)nr * tpn * 8, 0.f);
        for (int g0 = 0; g0 < G; g0 += kKCG) {
            const int ng = G - g0 < kKCG ? G - g0 : kKCG;
            for (int r = 0; r < nr; ++r) {
                const uint32_t* p = W.q + (size_t)(r0 + r) * rw + (size_t)g0 * 2 * BITS;
                for (int g = 0; g < ng; ++g) {
                    __m256i lo, hi;
                    unpack_group<BITS>(p + (size_t)g * 2 * BITS, lo, hi);
                    _mm256_store_si256((__m256i*)(wb + ((size_t)r * kKCG + g) * 64), lo);
                    _mm256_store_si256((__m256i*)(wb + ((size_t)r * kKCG + g) * 64 + 32), hi);
                }
            }
            for (int tl = 0; tl < tpn; tl += kTB) {
                const int tn = tpn - tl < kTB ? tpn - tl : kTB;
                int r = 0;
                for (; r + kRT <= nr; r += kRT)
                    tile_tokens<kRT>(wb + (size_t)r * kKCG * 64, (size_t)kKCG * 64, scv.data() + (size_t)r * G + g0, G, X,
                                     xoff + tp0 + tl, tn, g0, ng, accv.data() + r * accrow + tl * 8, accrow);
                for (; r < nr; ++r)
                    tile_tokens<1>(wb + (size_t)r * kKCG * 64, (size_t)kKCG * 64, scv.data() + (size_t)r * G + g0, G, X,
                                   xoff + tp0 + tl, tn, g0, ng, accv.data() + r * accrow + tl * 8, accrow);
            }
        }
        // epilogue: horizontal sum + weight-min term
        for (int tl = 0; tl < tpn; ++tl) {
            const int t = tp0 + tl;
            const float* xs = X.xs.data() + (size_t)(xoff + t) * G;
            for (int r = 0; r < nr; ++r) {
                const float* mn = mnv.data() + (size_t)r * G;
                __m256 m = _mm256_setzero_ps();
                int g = 0;
                for (; g + 8 <= G; g += 8) m = _mm256_fmadd_ps(_mm256_loadu_ps(mn + g), _mm256_loadu_ps(xs + g), m);
                float tail = 0.f;
                for (; g < G; ++g) tail += mn[g] * xs[g];
                Y[(size_t)t * ldy + r0 + r] = hsum(_mm256_loadu_ps(accv.data() + r * accrow + (size_t)tl * 8)) + hsum(m) + tail;
            }
        }
    }
}

template <int BITS>
void gemm_bits(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1, int xoff) {
    for (int r = row0; r < row1; r += kRB) gemm_block<BITS>(W, X, T, Y, ldy, r, row1 - r < kRB ? row1 - r : kRB, xoff);
}

template <int BITS>
void gemv_bits(const QMatC& W, const Q8ActC& x, float* y, int row0, int row1) {
    int r = row0;
    for (; r + 4 <= row1; r += 4) rows_kernel<BITS, 4>(W, x, y, r);
    for (; r < row1; ++r) rows_kernel<BITS, 1>(W, x, y, r);
}

}  // namespace

void gemv_q(const QMatC& W, const Q8ActC& x, float* y, int row0, int row1) {
    switch (W.bits) {
        case 2: gemv_bits<2>(W, x, y, row0, row1); break;
        case 3: gemv_bits<3>(W, x, y, row0, row1); break;
        case 4: gemv_bits<4>(W, x, y, row0, row1); break;
        case 5: gemv_bits<5>(W, x, y, row0, row1); break;
        case 6: gemv_bits<6>(W, x, y, row0, row1); break;
        case 8: gemv_bits<8>(W, x, y, row0, row1); break;
        default: break;
    }
}

void gemv_q_par(ThreadPool& pool, const QMatC& W, const Q8ActC& x, float* y) {
    // dynamic chunks of 16 rows (a multiple of the 4-row kernel block): robust to preempted threads
    pool.parallel_for_dyn(
        (size_t)W.N, [&](size_t b, size_t e, int) { gemv_q(W, x, y, (int)b, (int)e); }, 16);
}

void Q8Batch::quantize_row(int t, const float* x, int kuse) {
    const int Gu = kuse / 64;
    quantize_act_q8_raw(x, 0, Gu, q.data() + (size_t)t * K, ds.data() + (size_t)t * G * 4);
    const float* d = ds.data() + (size_t)t * G * 4;
    float* o = dv8.data() + (size_t)t * G * 8;
    float* s = xs.data() + (size_t)t * G;
    for (int g = 0; g < Gu; ++g) {
        for (int i = 0; i < 4; ++i) { o[8 * g + i] = d[4 * g]; o[8 * g + 4 + i] = d[4 * g + 1]; }
        s[g] = d[4 * g + 2] + d[4 * g + 3];
    }
}

void gemm_q(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1, int xoff) {
    if (T <= 0) return;
    switch (W.bits) {
        case 2: gemm_bits<2>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 3: gemm_bits<3>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 4: gemm_bits<4>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 5: gemm_bits<5>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 6: gemm_bits<6>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 8: {  // q up to 255 needs the two-nibble-plane dot: reuse the GEMV kernel token by token (not optimized)
            Q8ActC a;
            a.K = X.K;
            a.q.resize((size_t)X.K);
            a.ds.resize((size_t)X.G * 4);
            std::vector<float> y((size_t)W.N);
            for (int t = 0; t < T; ++t) {
                std::memcpy(a.q.data(), X.q.data() + (size_t)(xoff + t) * X.K, X.K);
                std::memcpy(a.ds.data(), X.ds.data() + (size_t)(xoff + t) * X.G * 4, (size_t)X.G * 16);
                gemv_q(W, a, y.data(), row0, row1);
                for (int r = row0; r < row1; ++r) Y[(size_t)t * ldy + r] = y[r];
            }
            break;
        }
        default: break;
    }
}

void gemv_multi_q(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1) {
    if (T <= 0) return;
    if (T > 4) throw_too_many_tokens();
    switch (W.bits) {
        case 2: multi_bits<2>(W, X, T, Y, ldy, row0, row1); break;
        case 3: multi_bits<3>(W, X, T, Y, ldy, row0, row1); break;
        case 4: multi_bits<4>(W, X, T, Y, ldy, row0, row1); break;
        case 5: multi_bits<5>(W, X, T, Y, ldy, row0, row1); break;
        case 6: multi_bits<6>(W, X, T, Y, ldy, row0, row1); break;
        default: {  // 8 bit: per-token GEMV
            Q8ActC a;
            a.K = X.K;
            a.q.resize((size_t)X.K);
            a.ds.resize((size_t)X.G * 4);
            std::vector<float> y((size_t)W.N);
            for (int t = 0; t < T; ++t) {
                std::memcpy(a.q.data(), X.q.data() + (size_t)t * X.K, X.K);
                std::memcpy(a.ds.data(), X.ds.data() + (size_t)t * X.G * 4, (size_t)X.G * 16);
                gemv_q(W, a, y.data(), row0, row1);
                for (int r = row0; r < row1; ++r) Y[(size_t)t * ldy + r] = y[r];
            }
        }
    }
}

void gemv_multi_q_par(ThreadPool& pool, const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy) {
    pool.parallel_for_dyn((size_t)W.N, [&](size_t b, size_t e, int) { gemv_multi_q(W, X, T, Y, ldy, (int)b, (int)e); }, 16);
}

void gemm_q_par(ThreadPool& pool, const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy) {
    pool.parallel_for_dyn((size_t)W.N, [&](size_t b, size_t e, int) { gemm_q(W, X, T, Y, ldy, (int)b, (int)e); }, kRB);
}

}  // namespace qv::cpu
