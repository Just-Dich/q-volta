// CPU GEMV bandwidth bench. Rotates through >= `--gib` GiB of distinct 17408x5120 matrices (far beyond the 16 MiB
// L3) so the kernel streams from RAM; compares against a plain AVX2 streaming read at the same thread counts.
// usage: bench_cpu_gemv [--bits 4,3] [--threads 1,8,16,32,64] [--gib 1.1] [--secs 1.0] [--N 17408] [--K 5120]
#include <immintrin.h>
#include <sys/mman.h>
#include <x86intrin.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cpu_gemv.h"

using namespace qv::cpu;
using clk = std::chrono::steady_clock;

static void* big_alloc(size_t bytes) {
    const size_t al = 2u << 20;
    bytes = (bytes + al - 1) / al * al;
    void* p = std::aligned_alloc(al, bytes);
    if (!p) { std::perror("alloc"); std::exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}
static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); v.push_back(atoi(s.substr(i, j - i).c_str())); i = j + 1; }
    return v;
}
static double tsc_ghz() {
    auto t0 = clk::now(); uint64_t c0 = __rdtsc();
    while (std::chrono::duration<double>(clk::now() - t0).count() < 0.1) {}
    return (__rdtsc() - c0) / std::chrono::duration<double>(clk::now() - t0).count() / 1e9;
}

static __attribute__((noinline)) uint64_t stream_read(const void* p, size_t bytes) {
    const __m256i* a = (const __m256i*)p;
    const size_t n = bytes / 128;
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t i = 0; i < n; ++i) {
        s0 = _mm256_xor_si256(s0, _mm256_load_si256(a + 4 * i));
        s1 = _mm256_xor_si256(s1, _mm256_load_si256(a + 4 * i + 1));
        s2 = _mm256_xor_si256(s2, _mm256_load_si256(a + 4 * i + 2));
        s3 = _mm256_xor_si256(s3, _mm256_load_si256(a + 4 * i + 3));
    }
    s0 = _mm256_xor_si256(_mm256_xor_si256(s0, s1), _mm256_xor_si256(s2, s3));
    return (uint64_t)_mm256_extract_epi64(s0, 0) ^ (uint64_t)_mm256_extract_epi64(s0, 3);
}

// Same read volume, but 4 concurrent sub-streams per thread + software prefetch (like the gemv's 4 row streams).
// Gives the memory system more outstanding misses; used as the second (usually higher) ceiling variant.
static __attribute__((noinline)) uint64_t stream_read_pf(const void* p, size_t bytes) {
    const size_t sub = bytes / 4 / 64 * 64;
    const char* a0 = (const char*)p; const char* a1 = a0 + sub; const char* a2 = a1 + sub; const char* a3 = a2 + sub;
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t o = 0; o < sub; o += 64) {
        _mm_prefetch(a0 + o + 1024, _MM_HINT_T0); _mm_prefetch(a1 + o + 1024, _MM_HINT_T0);
        _mm_prefetch(a2 + o + 1024, _MM_HINT_T0); _mm_prefetch(a3 + o + 1024, _MM_HINT_T0);
        s0 = _mm256_xor_si256(s0, _mm256_xor_si256(_mm256_load_si256((const __m256i*)(a0 + o)), _mm256_load_si256((const __m256i*)(a0 + o + 32))));
        s1 = _mm256_xor_si256(s1, _mm256_xor_si256(_mm256_load_si256((const __m256i*)(a1 + o)), _mm256_load_si256((const __m256i*)(a1 + o + 32))));
        s2 = _mm256_xor_si256(s2, _mm256_xor_si256(_mm256_load_si256((const __m256i*)(a2 + o)), _mm256_load_si256((const __m256i*)(a2 + o + 32))));
        s3 = _mm256_xor_si256(s3, _mm256_xor_si256(_mm256_load_si256((const __m256i*)(a3 + o)), _mm256_load_si256((const __m256i*)(a3 + o + 32))));
    }
    s0 = _mm256_xor_si256(_mm256_xor_si256(s0, s1), _mm256_xor_si256(s2, s3));
    return (uint64_t)_mm256_extract_epi64(s0, 0) ^ (uint64_t)_mm256_extract_epi64(s0, 3);
}

struct Set {
    std::vector<uint32_t*> q;
    std::vector<uint16_t*> sm;
    int N, K, bits;
    size_t qbytes, smbytes;
};

static Set make_set(int N, int K, int bits, double gib, int nth) {
    ThreadPool fill(nth);  // scoped: spinning workers must not outlive the fill
    Set s; s.N = N; s.K = K; s.bits = bits;
    s.qbytes = (size_t)N * (K / 32) * bits * 4;
    s.smbytes = (size_t)N * (K / 64) * 4;
    const int M = (int)std::ceil(gib * (1ull << 30) / (double)(s.qbytes + s.smbytes));
    for (int m = 0; m < M; ++m) {
        s.q.push_back((uint32_t*)big_alloc(s.qbytes));
        s.sm.push_back((uint16_t*)big_alloc(s.smbytes));
        uint32_t* q = s.q.back(); uint16_t* sm = s.sm.back();
        fill.parallel_for(s.qbytes / 8, [&](size_t b, size_t e, int t) {
            uint64_t x = 0x9E3779B97F4A7C15ull * (b + 1 + m * 77 + t);
            uint64_t* d = (uint64_t*)q;
            for (size_t i = b; i < e; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; d[i] = x; }
        });
        for (size_t i = 0; i < s.smbytes / 2; i += 2) { sm[i] = 0x2400 + (i & 0xff); sm[i + 1] = 0xb000; }
    }
    return s;
}

int main(int argc, char** argv) {
    std::vector<int> bitsl = {4, 3}, thl = {1, 8, 16, 32, 64};
    double gib = 1.1, secs = 1.0; int N = 17408, K = 5120;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--bits") bitsl = parse_list(argv[i + 1]);
        else if (a == "--threads") thl = parse_list(argv[i + 1]);
        else if (a == "--gib") gib = atof(argv[i + 1]);
        else if (a == "--secs") secs = atof(argv[i + 1]);
        else if (a == "--N") N = atoi(argv[i + 1]);
        else if (a == "--K") K = atoi(argv[i + 1]);
    }
    int maxt = 1; for (int t : thl) maxt = t > maxt ? t : maxt;
    double l1[3]; FILE* f = fopen("/proc/loadavg", "r"); if (f) { if (fscanf(f, "%lf %lf %lf", &l1[0], &l1[1], &l1[2]) == 3) std::printf("loadavg at start: %.2f %.2f %.2f\n", l1[0], l1[1], l1[2]); fclose(f); }
    const double ghz = tsc_ghz();
    std::printf("TSC %.3f GHz; matrix %dx%d; set >= %.2f GiB\n", ghz, N, K, gib);

    std::vector<float> x(K); { std::mt19937 r(5); std::normal_distribution<float> nd; for (auto& v : x) v = nd(r); }
    Q8ActC xa; quantize_act_q8(x.data(), K, xa);
    std::vector<float> y(N);

    // ceilings measured on the first set's memory (any >=1 GiB buffer works)
    std::vector<double> ceil_gbs(thl.size(), 0.0), ceil_cpb(thl.size(), 0.0);
    {
        Set cs = make_set(N, K, 4, gib, maxt);
        for (size_t ti = 0; ti < thl.size(); ++ti) {
            ThreadPool pool(thl[ti]);
            volatile uint64_t sink = 0;
            const int M = (int)cs.q.size();
            // variant A: per matrix, threads claim 64 KiB chunks of q dynamically (same barrier count/access pattern as the kernel)
            // variant B: no barrier, each thread streams its own contiguous slice of every matrix
            double best = 0;
            for (int variant = 0; variant < 4; ++variant) {
                const bool pf = variant >= 2;
                auto run_pass = [&](int m) {
                    pool.parallel_for_dyn(cs.qbytes / 128, [&](size_t b, size_t e, int) {  // 64 KiB chunks, dynamic claim
                        sink = sink ^ (pf ? stream_read_pf((const char*)cs.q[m] + b * 128, (e - b) * 128)
                                          : stream_read((const char*)cs.q[m] + b * 128, (e - b) * 128));
                    }, 512);
                };
                run_pass(0);
                size_t bytes = 0; auto t0 = clk::now(); double el = 0; int it = 0;
                if (variant % 2 == 0) {
                    while (el < secs || it < M) { run_pass(it % M); bytes += cs.qbytes; ++it; el = std::chrono::duration<double>(clk::now() - t0).count(); }
                } else {
                    while (el < secs || it < 1) {
                        pool.parallel_for(pool.size(), [&](size_t, size_t, int tid) {
                            uint64_t a = 0;
                            for (int m = 0; m < M; ++m) { // thread tid's contiguous slice of matrix m
                                size_t units = cs.qbytes / 128, per = units / pool.size();
                                a ^= pf ? stream_read_pf((const char*)cs.q[m] + tid * per * 128, per * 128) : stream_read((const char*)cs.q[m] + tid * per * 128, per * 128);
                            }
                            sink = sink ^ a;
                        }, 1);
                        bytes += (cs.qbytes / 128 / pool.size()) * 128 * (size_t)pool.size() * M; ++it;
                        el = std::chrono::duration<double>(clk::now() - t0).count();
                    }
                }
                const double gbs = bytes / el / 1e9;
                if (gbs > best) { best = gbs; ceil_cpb[ti] = ghz * 1e9 * thl[ti] * el / bytes; }
                std::printf("  stream threads=%2d variant=%s%s  %.1f GB/s\n", thl[ti], variant % 2 ? "nobarrier" : "per-matrix", pf ? "+pf4" : "", gbs);
            }
            ceil_gbs[ti] = best;
        }
        for (auto p : cs.q) std::free(p);
        for (auto p : cs.sm) std::free(p);
    }

    std::printf("\n%-5s %-7s %-9s %-10s %-9s %-10s\n", "bits", "threads", "GB/s", "%ceiling", "cyc/B*", "stream cyc/B*");
    for (int bits : bitsl) {
        Set s = make_set(N, K, bits, gib, maxt);
        const int M = (int)s.q.size();
        for (size_t ti = 0; ti < thl.size(); ++ti) {
            ThreadPool pool(thl[ti]);
            auto mat = [&](int m) { return QMatC{s.q[m], s.sm[m], N, K, bits}; };
            gemv_q_par(pool, mat(0), xa, y.data());
            size_t bytes = 0; int it = 0; double el = 0; auto t0 = clk::now();
            while (el < secs || it < M) {
                QMatC W = mat(it % M);
                gemv_q_par(pool, W, xa, y.data());
                bytes += W.bytes(); ++it;
                el = std::chrono::duration<double>(clk::now() - t0).count();
            }
            const double gbs = bytes / el / 1e9;
            const double cpb = ghz * 1e9 * thl[ti] * el / bytes;
            std::printf("%-5d %-7d %-9.1f %-10.1f %-9.3f %-10.3f  (%d iters, %d mats, ms/gemv %.2f)\n", bits, thl[ti], gbs,
                        100.0 * gbs / ceil_gbs[ti], cpb, ceil_cpb[ti], it, M, 1e3 * el / it);
        }
        for (auto p : s.q) std::free(p);
        for (auto p : s.sm) std::free(p);
    }
    std::printf("* cyc/B = TSC-GHz x threads x seconds / bytes: aggregate core-cycles per byte of q+sm read\n");
    std::printf("y[0]=%g\n", y[0]);
    return 0;
}
