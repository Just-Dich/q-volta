#include "cpu_gemv.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace qv::cpu {

void quantize_act_q8(const float* x, int K, Q8ActC& out) {
    out.K = K;
    out.q.assign((size_t)K, 0);
    out.ds.assign((size_t)(K / 64) * 4, 0.f);
    quantize_act_q8_range(x, 0, K / 64, out);
}

void quantize_act_q8_range(const float* x, int g0, int g1, Q8ActC& out) {
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
        out.ds[4 * g] = dsub[0];
        out.ds[4 * g + 1] = dsub[1];
        out.ds[4 * g + 2] = ssub[0];
        out.ds[4 * g + 3] = ssub[1];
        if (!(dsub[0] > 0.f) && !(dsub[1] > 0.f)) std::memset(out.q.data() + (size_t)g * 64, 0, 64);
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
            int8_t* o = out.q.data() + (size_t)g * 64;
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

// int32x8 partial dot of one 64-weight group (p = first word of the group, 2*BITS words).
template <int BITS>
inline __m256i dot_group(const uint32_t* p, __m256i xlo, __m256i xhi) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i lo, hi;
    if constexpr (BITS == 8) {
        __m256i alo, ahi, blo, bhi;
        plane4<8>(p, 0, alo, ahi);
        plane4<8>(p, 4, blo, bhi);
        __m256i sa = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(alo, xlo), _mm256_maddubs_epi16(ahi, xhi)), ones);
        __m256i sb = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(blo, xlo), _mm256_maddubs_epi16(bhi, xhi)), ones);
        return _mm256_add_epi32(sa, _mm256_slli_epi32(sb, 4));
    } else if constexpr (BITS == 4) {
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
    // q <= 63: pair sums (<=16002) and the add of two (<=32004) fit in int16 without saturation.
    __m256i s = _mm256_add_epi16(_mm256_maddubs_epi16(lo, xlo), _mm256_maddubs_epi16(hi, xhi));
    return _mm256_madd_epi16(s, ones);
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
        sp[r] = W.sm + (size_t)(r0 + r) * G * 2;
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
            std::memcpy(&smw, sp[r] + 2 * g, 4);
            const __m128 t = _mm_cvtph_ps(_mm_cvtsi32_si128((int)smw));  // (scale, min)
            accm[r] = _mm_add_ps(accm[r], _mm_mul_ss(_mm_shuffle_ps(t, t, 1), ssum));
            accf[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(dvec, _mm256_broadcastss_ps(t)), accf[r]);
        }
    }
    for (int r = 0; r < R; ++r)
        y[r0 + r] = hsum(accf[r]) + _mm_cvtss_f32(accm[r]);
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

}  // namespace qv::cpu
