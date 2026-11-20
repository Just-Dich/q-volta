#include "attn_cpu.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace qv::cpu {

static constexpr float kNegInf = -std::numeric_limits<float>::infinity();
static constexpr int kBlk = 32;       // tokens per online-softmax block

// ---------------------------------------------------------------- scalar helpers
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t f2h(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }

void pack_row(const float* x, uint8_t* row) {
    for (int g = 0; g < kGroups; ++g) {
        const float* xg = x + g * kGroup;
        float lo = xg[0], hi = xg[0];
        for (int i = 1; i < kGroup; ++i) { lo = std::min(lo, xg[i]); hi = std::max(hi, xg[i]); }
        uint16_t hs = f2h((hi - lo) / 15.f), hm = f2h(lo);
        float s = h2f(hs), mn = h2f(hm);
        uint8_t* cb = row + g * 32;
        for (int j = 0; j < 32; ++j) {
            int c[2];
            for (int k = 0; k < 2; ++k) {
                float v = s > 0.f ? (xg[j + 32 * k] - mn) / s : 0.f;
                c[k] = std::min(15, std::max(0, (int)std::lrintf(v)));
            }
            cb[j] = (uint8_t)(c[0] | (c[1] << 4));
        }
        std::memcpy(row + 128 + g * 4, &hs, 2);
        std::memcpy(row + 128 + g * 4 + 2, &hm, 2);
    }
}

void dequant_row(const uint8_t* row, float* x) {
    for (int g = 0; g < kGroups; ++g) {
        uint16_t hs, hm;
        std::memcpy(&hs, row + 128 + g * 4, 2);
        std::memcpy(&hm, row + 128 + g * 4 + 2, 2);
        float s = h2f(hs), mn = h2f(hm);
        for (int j = 0; j < 32; ++j) {
            uint8_t b = row[g * 32 + j];
            x[g * 64 + j] = (b & 15) * s + mn;
            x[g * 64 + j + 32] = (b >> 4) * s + mn;
        }
    }
}

// ---------------------------------------------------------------- AVX2 kernel
static inline __m256 exp_neg(__m256 x) {  // exp(x) for x <= 0, rel err ~2e-7
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.0f / 120.f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// 8 nibble codes (from the low 8 bytes of v) -> fp32
static inline __m256 codes8(__m128i v) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)); }

struct Partial {
    float m[kGqa];
    float l[kGqa];
    alignas(32) float acc[kGqa][kD];
};

// Process tokens [t0,t1) of one kv head; q already scaled, qsum[h][g] = sum of q over group g.
static void run_chunk(const float* q, const float (*qsum)[kGroups], const uint8_t* K, const uint8_t* V,
                      int64_t t0, int64_t t1, Partial& P) {
    for (int h = 0; h < kGqa; ++h) { P.m[h] = kNegInf; P.l[h] = 0.f; }
    std::memset(P.acc, 0, sizeof(P.acc));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    alignas(32) float sc[kGqa][kBlk];
    alignas(32) float vs[kBlk][kGroups * 2];  // per-token {scale,min} x 4 groups for V

    for (int64_t b0 = t0; b0 < t1; b0 += kBlk) {
        const int nt = (int)std::min<int64_t>(kBlk, t1 - b0);
        // ---- scores: s = sum_g ( scale_g * (q . codes_g) + min_g * qsum_g )
        for (int t = 0; t < nt; ++t) {
            const uint8_t* row = K + (b0 + t) * kRowBytes;
            __m256 tot[kGqa];
            for (int h = 0; h < kGqa; ++h) tot[h] = _mm256_setzero_ps();
            float msum[kGqa] = {0, 0, 0, 0, 0, 0};
            for (int g = 0; g < kGroups; ++g) {
                uint16_t hs, hm;
                std::memcpy(&hs, row + 128 + g * 4, 2);
                std::memcpy(&hm, row + 128 + g * 4 + 2, 2);
                const __m256 s = _mm256_set1_ps(h2f(hs));
                const float mn = h2f(hm);
                __m256 p[kGqa];
                for (int h = 0; h < kGqa; ++h) p[h] = _mm256_setzero_ps();
                for (int half = 0; half < 2; ++half) {
                    for (int part = 0; part < 2; ++part) {
                        __m128i raw = _mm_loadu_si128((const __m128i*)(row + g * 32 + part * 16));
                        __m128i c = half ? _mm_and_si128(_mm_srli_epi16(raw, 4), m4) : _mm_and_si128(raw, m4);
                        __m256 c0 = codes8(c), c1 = codes8(_mm_srli_si128(c, 8));
                        const int e = g * kGroup + half * 32 + part * 16;
                        for (int h = 0; h < kGqa; ++h) {
                            p[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + h * kD + e), c0, p[h]);
                            p[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + h * kD + e + 8), c1, p[h]);
                        }
                    }
                }
                for (int h = 0; h < kGqa; ++h) { tot[h] = _mm256_fmadd_ps(p[h], s, tot[h]); msum[h] += mn * qsum[h][g]; }
            }
            for (int h = 0; h < kGqa; ++h) sc[h][t] = hsum(tot[h]) + msum[h];
        }
        for (int t = nt; t < kBlk; ++t)
            for (int h = 0; h < kGqa; ++h) sc[h][t] = kNegInf;
        // ---- online softmax update
        float prob_scale[kGqa];
        for (int h = 0; h < kGqa; ++h) {
            __m256 mx = _mm256_load_ps(sc[h]);
            for (int i = 8; i < kBlk; i += 8) mx = _mm256_max_ps(mx, _mm256_load_ps(sc[h] + i));
            float bm = std::max(P.m[h], [&] {
                __m128 a = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
                a = _mm_max_ps(a, _mm_movehl_ps(a, a));
                a = _mm_max_ss(a, _mm_movehdup_ps(a));
                return _mm_cvtss_f32(a);
            }());
            prob_scale[h] = P.m[h] == kNegInf ? 0.f : std::exp(P.m[h] - bm);
            P.m[h] = bm;
            const __m256 vb = _mm256_set1_ps(bm);
            for (int i = 0; i < kBlk; i += 8) {
                __m256 e = exp_neg(_mm256_sub_ps(_mm256_load_ps(sc[h] + i), vb));
                _mm256_store_ps(sc[h] + i, e);
            }
            float lsum = 0.f;
            for (int t = 0; t < nt; ++t) lsum += sc[h][t];
            P.l[h] = P.l[h] * prob_scale[h] + lsum;
            const __m256 sv = _mm256_set1_ps(prob_scale[h]);
            for (int d = 0; d < kD; d += 8) _mm256_store_ps(&P.acc[h][d], _mm256_mul_ps(_mm256_load_ps(&P.acc[h][d]), sv));
        }
        // ---- V: per-token scale/min staging, then accumulate 16 elements x 6 heads at a time
        for (int t = 0; t < nt; ++t) {
            const uint8_t* row = V + (b0 + t) * kRowBytes;
            for (int g = 0; g < kGroups; ++g) {
                uint16_t hs, hm;
                std::memcpy(&hs, row + 128 + g * 4, 2);
                std::memcpy(&hm, row + 128 + g * 4 + 2, 2);
                vs[t][g * 2] = h2f(hs);
                vs[t][g * 2 + 1] = h2f(hm);
            }
        }
        for (int g = 0; g < kGroups; ++g) {
            for (int half = 0; half < 2; ++half) {
                for (int part = 0; part < 2; ++part) {
                    const int e = g * kGroup + half * 32 + part * 16;
                    __m256 a0[kGqa], a1[kGqa];
                    for (int h = 0; h < kGqa; ++h) { a0[h] = _mm256_load_ps(&P.acc[h][e]); a1[h] = _mm256_load_ps(&P.acc[h][e + 8]); }
                    for (int t = 0; t < nt; ++t) {
                        const uint8_t* row = V + (b0 + t) * kRowBytes;
                        __m128i raw = _mm_loadu_si128((const __m128i*)(row + g * 32 + part * 16));
                        __m128i c = half ? _mm_and_si128(_mm_srli_epi16(raw, 4), m4) : _mm_and_si128(raw, m4);
                        const __m256 s = _mm256_set1_ps(vs[t][g * 2]), mn = _mm256_set1_ps(vs[t][g * 2 + 1]);
                        __m256 v0 = _mm256_fmadd_ps(codes8(c), s, mn);
                        __m256 v1 = _mm256_fmadd_ps(codes8(_mm_srli_si128(c, 8)), s, mn);
                        for (int h = 0; h < kGqa; ++h) {
                            __m256 pb = _mm256_broadcast_ss(&sc[h][t]);
                            a0[h] = _mm256_fmadd_ps(pb, v0, a0[h]);
                            a1[h] = _mm256_fmadd_ps(pb, v1, a1[h]);
                        }
                    }
                    for (int h = 0; h < kGqa; ++h) { _mm256_store_ps(&P.acc[h][e], a0[h]); _mm256_store_ps(&P.acc[h][e + 8], a1[h]); }
                }
            }
        }
    }
}

static void merge_n(int nh, float* m_a, float* l_a, float* acc_a, const float* m_b, const float* l_b,
                    const float* acc_b) {
    for (int h = 0; h < nh; ++h) {
        if (m_b[h] == kNegInf) continue;
        float m = std::max(m_a[h], m_b[h]);
        float fa = m_a[h] == kNegInf ? 0.f : std::exp(m_a[h] - m), fb = std::exp(m_b[h] - m);
        l_a[h] = l_a[h] * fa + l_b[h] * fb;
        for (int d = 0; d < kD; ++d) acc_a[h * kD + d] = acc_a[h * kD + d] * fa + acc_b[h * kD + d] * fb;
        m_a[h] = m;
    }
}

void attn_merge(float* m_a, float* l_a, float* acc_a, const float* m_b, const float* l_b, const float* acc_b) {
    merge_n(kHq, m_a, l_a, acc_a, m_b, l_b, acc_b);
}

void attn_cpu_partial(const float* q, const uint8_t* K, const uint8_t* V, int64_t ntok, float* m, float* l,
                      float* acc, int nthreads, int chunk_tokens) {
    const int kChunk = chunk_tokens > 0 ? chunk_tokens : 2048;  // tokens per work item
    for (int h = 0; h < kHq; ++h) { m[h] = kNegInf; l[h] = 0.f; }
    std::memset(acc, 0, sizeof(float) * kHq * kD);
    if (ntok <= 0) return;
    alignas(32) float qs[kHq * kD];
    float qsum[kHq][kGroups];
    const float inv = 1.f / std::sqrt((float)kD);
    for (int h = 0; h < kHq; ++h)
        for (int g = 0; g < kGroups; ++g) {
            float s = 0.f;
            for (int i = 0; i < kGroup; ++i) { float v = q[h * kD + g * kGroup + i] * inv; qs[h * kD + g * kGroup + i] = v; s += v; }
            qsum[h][g] = s;
        }
    const int64_t nchunk = (ntok + kChunk - 1) / kChunk;
    const int64_t nitems = nchunk * kHkv;
    std::vector<Partial> parts((size_t)nitems);
    const int nt = nthreads > 0 ? nthreads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int64_t it = 0; it < nitems; ++it) {
        const int kvh = (int)(it / nchunk);
        const int64_t c = it % nchunk, t0 = c * kChunk, t1 = std::min<int64_t>(ntok, t0 + kChunk);
        const size_t base = (size_t)kvh * ntok * kRowBytes;
        run_chunk(qs + kvh * kGqa * kD, &qsum[kvh * kGqa], K + base, V + base, t0, t1, parts[it]);
    }
    for (int kvh = 0; kvh < kHkv; ++kvh)
        for (int64_t c = 0; c < nchunk; ++c) {
            const Partial& P = parts[kvh * nchunk + c];
            merge_n(kGqa, m + kvh * kGqa, l + kvh * kGqa, acc + kvh * kGqa * kD, P.m, P.l, &P.acc[0][0]);
        }
}

void attn_cpu_partial_blocks(const float* q, const uint8_t* K, const uint8_t* V, int64_t ntok_per_head, const int32_t* blocks, int nb,
                             float* m, float* l, float* acc, int nthreads, int blocks_per_item) {
    for (int h = 0; h < kHq; ++h) { m[h] = kNegInf; l[h] = 0.f; }
    std::memset(acc, 0, sizeof(float) * kHq * kD);
    if (nb <= 0) return;
    alignas(32) float qs[kHq * kD];
    float qsum[kHq][kGroups];
    const float inv = 1.f / std::sqrt((float)kD);
    for (int h = 0; h < kHq; ++h)
        for (int g = 0; g < kGroups; ++g) {
            float s = 0.f;
            for (int i = 0; i < kGroup; ++i) { float v = q[h * kD + g * kGroup + i] * inv; qs[h * kD + g * kGroup + i] = v; s += v; }
            qsum[h][g] = s;
        }
    const int ng = (nb + blocks_per_item - 1) / blocks_per_item;
    const int64_t nitems = (int64_t)ng * kHkv;
    std::vector<Partial> parts((size_t)nitems);
    const int nt = nthreads > 0 ? nthreads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int64_t it = 0; it < nitems; ++it) {
        const int kvh = (int)(it / ng), gi = (int)(it % ng);
        Partial& P = parts[it];
        for (int h = 0; h < kGqa; ++h) { P.m[h] = kNegInf; P.l[h] = 0.f; }
        std::memset(P.acc, 0, sizeof(P.acc));
        Partial T;
        const size_t base = (size_t)kvh * ntok_per_head * kRowBytes;
        for (int b = gi * blocks_per_item; b < std::min(nb, (gi + 1) * blocks_per_item); ++b) {
            const int64_t t0 = (int64_t)blocks[kvh * nb + b] * 64;
            run_chunk(qs + kvh * kGqa * kD, &qsum[kvh * kGqa], K + base, V + base, t0, t0 + 64, T);
            merge_n(kGqa, P.m, P.l, &P.acc[0][0], T.m, T.l, &T.acc[0][0]);
        }
    }
    for (int kvh = 0; kvh < kHkv; ++kvh)
        for (int gi = 0; gi < ng; ++gi) {
            const Partial& P = parts[(size_t)kvh * ng + gi];
            merge_n(kGqa, m + kvh * kGqa, l + kvh * kGqa, acc + kvh * kGqa * kD, P.m, P.l, &P.acc[0][0]);
        }
}

void attn_cpu_ref(const float* q, const uint8_t* K, const uint8_t* V, int64_t ntok, float* m, float* l,
                  float* acc) {
    std::vector<float> kd(kD), vd(kD);
    std::vector<double> sc((size_t)std::max<int64_t>(ntok, 1));
    for (int h = 0; h < kHq; ++h) {
        const int kvh = h / kGqa;
        const size_t base = (size_t)kvh * ntok * kRowBytes;
        double mx = -INFINITY;
        for (int64_t t = 0; t < ntok; ++t) {
            dequant_row(K + base + t * kRowBytes, kd.data());
            double d = 0;
            for (int i = 0; i < kD; ++i) d += (double)q[h * kD + i] * kd[i];
            sc[t] = d / std::sqrt((double)kD);
            mx = std::max(mx, sc[t]);
        }
        std::vector<double> a(kD, 0.0);
        double ls = 0;
        for (int64_t t = 0; t < ntok; ++t) {
            double p = std::exp(sc[t] - mx);
            ls += p;
            dequant_row(V + base + t * kRowBytes, vd.data());
            for (int i = 0; i < kD; ++i) a[i] += p * vd[i];
        }
        m[h] = (float)mx; l[h] = (float)ls;
        for (int i = 0; i < kD; ++i) acc[h * kD + i] = (float)a[i];
    }
}

void fill_random_cache(uint8_t* buf, int64_t ntok, uint64_t seed) {
    const int64_t rows = ntok * kHkv;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        uint64_t x = seed * 0x9E3779B97F4A7C15ull + (uint64_t)r * 0xD1B54A32D192ED03ull + 1;
        auto rnd = [&] {  // xorshift64* -> approx N(0,1) via sum of 4 uniforms
            float s4 = 0;
            for (int i = 0; i < 4; ++i) {
                x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
                s4 += (float)((x * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.f;
            }
            return (s4 - 2.f) * 1.7320508f;
        };
        float v[kD];
        for (int i = 0; i < kD; ++i) v[i] = rnd();
        pack_row(v, buf + (size_t)r * kRowBytes);
    }
}

}  // namespace qv::cpu
