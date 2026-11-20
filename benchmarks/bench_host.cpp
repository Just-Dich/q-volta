// Host RAM / CPU measurements for the "cold KV on CPU" hypothesis.
//   bench_host [h2d_GBps rt_us_small rt_us_partials]   (PCIe numbers come from bench_pcie; used only for the model)
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/cpu/attn_cpu.h"
using namespace qv::cpu;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> static double best_of(int reps, F f) {
    double b = 1e30;
    for (int i = 0; i < reps; ++i) { double t = now(); f(); b = std::min(b, now() - t); }
    return b;
}

// ---- STREAM-like read bandwidth
static double read_bw(const float* buf, size_t n, int nt) {
    std::vector<double> part(64 * 8, 0.0);
    double t = best_of(5, [&] {
#pragma omp parallel num_threads(nt)
        {
            int id = omp_get_thread_num();
            size_t lo = n / nt * id, hi = id == nt - 1 ? n : n / nt * (id + 1);
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (size_t i = lo; i + 32 <= hi; i += 32) {
                a0 = _mm256_add_ps(a0, _mm256_load_ps(buf + i));
                a1 = _mm256_add_ps(a1, _mm256_load_ps(buf + i + 8));
                a2 = _mm256_add_ps(a2, _mm256_load_ps(buf + i + 16));
                a3 = _mm256_add_ps(a3, _mm256_load_ps(buf + i + 24));
            }
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            alignas(32) float tmp[8]; _mm256_store_ps(tmp, a0);
            part[id * 8] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
    });
    volatile double sink = 0; for (int i = 0; i < nt; ++i) sink += part[i * 8];
    return n * 4 / t / 1e9;
}

// ---- int4 GEMV: W [N][K] row-major, per 64 elements: 32 code bytes (lo nibble j, hi nibble j+32) + fp16 scale,min
static constexpr int GB = 36;  // bytes per 64 weights
static void gemv_i4(const uint8_t* W, const float* x, const float* xsum /*per group*/, float* y, int N, int K, int nt) {
    const int G = K / 64;
    const __m128i m4 = _mm_set1_epi8(0x0F);
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < N; ++r) {
        const uint8_t* row = W + (size_t)r * G * GB;
        __m256 tot = _mm256_setzero_ps();
        float msum = 0;
        for (int g = 0; g < G; ++g) {
            const uint8_t* gp = row + g * GB;
            uint16_t hs, hm; std::memcpy(&hs, gp + 32, 2); std::memcpy(&hm, gp + 34, 2);
            __m256 p = _mm256_setzero_ps();
            for (int half = 0; half < 2; ++half)
                for (int part = 0; part < 2; ++part) {
                    __m128i raw = _mm_loadu_si128((const __m128i*)(gp + part * 16));
                    __m128i c = half ? _mm_and_si128(_mm_srli_epi16(raw, 4), m4) : _mm_and_si128(raw, m4);
                    const float* xp = x + g * 64 + half * 32 + part * 16;
                    p = _mm256_fmadd_ps(_mm256_loadu_ps(xp), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c)), p);
                    p = _mm256_fmadd_ps(_mm256_loadu_ps(xp + 8), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c, 8))), p);
                }
            tot = _mm256_fmadd_ps(p, _mm256_set1_ps(_cvtsh_ss(hs)), tot);
            msum += _cvtsh_ss(hm) * xsum[g];
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(tot), _mm256_extractf128_ps(tot, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_movehdup_ps(s));
        y[r] = _mm_cvtss_f32(s) + msum;
    }
}

int main(int argc, char** argv) {
    const double pcie_gbps = argc > 1 ? atof(argv[1]) : 12.0;
    const double rt_us = argc > 2 ? atof(argv[2]) : 20.0;
    const double rt_part_us = argc > 3 ? atof(argv[3]) : 25.0;
    std::printf("omp max threads %d\n", omp_get_max_threads());

    // 1. read bandwidth (1 GiB, >> L3)
    const size_t nf = (size_t)1 << 28;
    float* buf = (float*)std::aligned_alloc(64, nf * 4);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nf; ++i) buf[i] = 1.f;
    std::printf("\n[1] STREAM-like read bandwidth (1 GiB)\n");
    double bw_best = 0;
    for (int nt : {1, 4, 8, 16, 32, 64}) { double b = read_bw(buf, nf, nt); bw_best = std::max(bw_best, b); std::printf("  threads=%-2d  %6.1f GB/s\n", nt, b); }
    std::free(buf);

    // 2. attention partial
    const int64_t NT = 200000;
    const size_t layer_bytes = (size_t)kHkv * NT * kRowBytes;
    const int NL = 3;  // rotate 3 distinct layers (K+V each) so nothing stays cached
    std::vector<uint8_t*> Kb(NL), Vb(NL);
    for (int i = 0; i < NL; ++i) {
        Kb[i] = (uint8_t*)std::aligned_alloc(64, (layer_bytes + 63) / 64 * 64);
        Vb[i] = (uint8_t*)std::aligned_alloc(64, (layer_bytes + 63) / 64 * 64);
        fill_random_cache(Kb[i], NT, 100 + i); fill_random_cache(Vb[i], NT, 200 + i);
    }
    std::vector<float> q(kHq * kD); for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(0.37 * i);
    float m[kHq], l[kHq]; std::vector<float> acc(kHq * kD);
    const double kvbytes = 2.0 * layer_bytes;
    std::printf("\n[2] attn_cpu_partial, %ld tokens, KV bytes/layer = %.1f MB (K+V, row %d B)\n", (long)NT, kvbytes / 1e6, kRowBytes);
    double best_ms = 1e9; int best_nt = 0;
    for (int nt : {1, 2, 4, 8, 16, 32, 64}) {
        int it = 0;
        double t = best_of(nt <= 2 ? 2 : 6, [&] { attn_cpu_partial(q.data(), Kb[it % NL], Vb[it % NL], NT, m, l, acc.data(), nt); ++it; });
        std::printf("  threads=%-2d  %8.2f ms/layer  %6.1f GB/s KV   (%.1f ns per token-kvhead per thread)\n", nt, t * 1e3, kvbytes / t / 1e9,
                    t * nt / (NT * kHkv) * 1e9);
        if (t * 1e3 < best_ms) { best_ms = t * 1e3; best_nt = nt; }
    }
    // compute-bound check: cache-resident working set (8192 tokens = 9.4 MB KV, fits 16 MB L3), repeated
    {
        const int64_t n = 8192; const double kvb = 2.0 * kHkv * n * kRowBytes;
        std::vector<uint8_t> K((size_t)kHkv * n * kRowBytes), V(K.size());
        fill_random_cache(K.data(), n, 1); fill_random_cache(V.data(), n, 2);
        std::printf("  compute-bound check, cache-resident KV (%.1f MB, 8192 tokens), same kernel, 128-token work items so all threads get work:\n", kvb / 1e6);
        for (int nt : {1, 16, 32, 64}) {
            double t = best_of(30, [&] { attn_cpu_partial(q.data(), K.data(), V.data(), n, m, l, acc.data(), nt, 128); });
            std::printf("    threads=%-2d  %7.3f ms  %6.1f GB/s-equivalent (data from cache)\n", nt, t * 1e3, kvb / t / 1e9);
        }
    }

    // 4. int4 GEMV
    {
        const int N = 32768, K = 8192, G = K / 64;
        const size_t bytes = (size_t)N * G * GB;
        uint8_t* W = (uint8_t*)std::aligned_alloc(64, bytes);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < N; ++r) {
            uint64_t s = r * 2654435761u + 7;
            uint8_t* row = W + (size_t)r * G * GB;
            for (int g = 0; g < G; ++g) {
                uint8_t* gp = row + g * GB;
                for (int j = 0; j < 32; ++j) { s = s * 6364136223846793005ull + 1442695040888963407ull; gp[j] = s >> 56; }
                uint16_t hs = _cvtss_sh(0.01f, 0), hm = _cvtss_sh(-0.08f, 0);
                std::memcpy(gp + 32, &hs, 2); std::memcpy(gp + 34, &hm, 2);
            }
        }
        std::vector<float> x(K), xs(G), y(N);
        for (int i = 0; i < K; ++i) x[i] = std::cos(0.01 * i);
        for (int g = 0; g < G; ++g) { xs[g] = 0; for (int i = 0; i < 64; ++i) xs[g] += x[g * 64 + i]; }
        // correctness on a few rows
        gemv_i4(W, x.data(), xs.data(), y.data(), N, K, 8);
        double maxerr = 0;
        for (int r : {0, 1, 12345, N - 1}) {
            double ref = 0;
            for (int g = 0; g < G; ++g) {
                const uint8_t* gp = W + ((size_t)r * G + g) * GB; uint16_t hs, hm; std::memcpy(&hs, gp + 32, 2); std::memcpy(&hm, gp + 34, 2);
                float s = _cvtsh_ss(hs), mn = _cvtsh_ss(hm);
                for (int j = 0; j < 32; ++j) { ref += (double)x[g * 64 + j] * ((gp[j] & 15) * s + mn); ref += (double)x[g * 64 + j + 32] * ((gp[j] >> 4) * s + mn); }
            }
            maxerr = std::max(maxerr, std::fabs(ref - y[r]) / (std::fabs(ref) + 1.0));
        }
        std::printf("\n[4] int4 GEMV N=%d K=%d (%.1f MB weights incl. meta), check err %.1e\n", N, K, bytes / 1e6, maxerr);
        for (int nt : {1, 16, 32, 64}) {
            double t = best_of(nt == 1 ? 2 : 6, [&] { gemv_i4(W, x.data(), xs.data(), y.data(), N, K, nt); });
            std::printf("  threads=%-2d  %7.2f ms  %6.1f GB/s weights\n", nt, t * 1e3, bytes / t / 1e9);
        }
        std::free(W);
    }

    // 5. per-token model
    const int AL = 16;  // full-attention layers
    const double kv_layer_tok = 2.0 * kHkv * kRowBytes;  // bytes of K+V per token per layer
    std::printf("\n[5] per-token decode time model, 200000-token context, %d attention layers, int4 KV\n", AL);
    std::printf("  inputs: CPU attn best %.2f ms/layer (%d thr), PCIe H2D %.1f GB/s, small-copy round trip %.1f us (q) + %.1f us (partials)\n",
                best_ms, best_nt, pcie_gbps, rt_us, rt_part_us);
    double a = AL * (best_ms + (rt_us + rt_part_us) * 1e-3);
    double kv4k = 4096 * kv_layer_tok;
    double sel_xfer = kv4k / (pcie_gbps * 1e9) * 1e3;                // ms per layer
    double c4k_gpu = kv4k / 890e9 * 1e3;                              // GPU attention on the fetched 4K
    double b = AL * (sel_xfer + c4k_gpu);
    double allkv = 200000 * kv_layer_tok * AL;
    double c = allkv / 890e9 * 1e3;
    std::printf("  KV total %.2f GB (%.1f MB/layer); 4K-token subset = %.2f MB/layer\n", allkv / 1e9, 200000 * kv_layer_tok / 1e6, kv4k / 1e6);
    std::printf("  (a) all cold KV on CPU, serial layers:          %7.1f ms/token attn  (%.2f tok/s if attention were the only cost)\n", a, 1e3 / a);
    std::printf("  (b) Quest-style 4K subset over PCIe (no overlap): %7.1f ms/token attn  (%.2f ms/layer xfer; excludes page-score cost)\n", b, sel_xfer);
    std::printf("  (c) GPU-resident int4 KV @890 GB/s:             %7.1f ms/token attn  (not fitting in 16 GiB with weights: needs %.2f GB)\n", c, allkv / 1e9);
    return 0;
}
