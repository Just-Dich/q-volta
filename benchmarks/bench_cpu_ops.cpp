// Bench for qv_cpu_ops. Usage: bench_cpu_ops <section> [threads]   sections: mem gdn attn misc
// Multi-thread numbers use a tiny persistent spin-barrier pool defined here (bench only; the engine brings its own).
#include <immintrin.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "ops.h"
using namespace qv::cpu;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Pool {
    int n;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0};
    std::atomic<bool> stop{false};
    std::function<void(int, int)> job;  // (thread idx, nthreads)
    explicit Pool(int n_) : n(n_) {
        for (int t = 1; t < n; ++t) th.emplace_back([this, t] {
            int seen = 0;
            for (;;) {
                int g;
                for (int sp = 0; (g = gen.load(std::memory_order_acquire)) == seen; ++sp) { if (stop) return; if (sp < 4000) _mm_pause(); else sched_yield(); }
                seen = g;
                job(t, n);
                done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    void run(std::function<void(int, int)> f) {
        job = std::move(f);
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        job(0, n);
        for (int sp = 0; done.load(std::memory_order_acquire) < n - 1; ++sp) { if (sp < 4000) _mm_pause(); else sched_yield(); }
    }
    ~Pool() { stop = true; for (auto& t : th) t.join(); }
};
static std::pair<int, int> part(int total, int t, int n) { return {(int)((long)total * t / n), (int)((long)total * (t + 1) / n)}; }

template <class F>
static std::pair<double, double> timeit(int reps, F f) {  // min, median seconds
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) { double t = now(); f(); v.push_back(now() - t); }
    std::sort(v.begin(), v.end());
    return {v[0], v[v.size() / 2]};
}
static void fill(float* p, size_t n, unsigned seed, float lo, float hi) {
    unsigned s = seed;
    for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; p[i] = lo + (hi - lo) * (float)(s >> 8) * (1.f / 16777216.f); }
}

int main(int argc, char** argv) {
    const std::string sec = argc > 1 ? argv[1] : "gdn";
    const int NT = argc > 2 ? atoi(argv[2]) : 64;
    Pool pool(NT);
    if (sec == "mem") {  // read-only streaming and read+write streaming bandwidth
        const size_t n = (size_t)1 << 28;  // 1 GiB of floats
        float* a = (float*)aligned_alloc(64, n * 4);
        fill(a, n, 1, 0, 1);
        for (int nt : {1, NT}) {
            Pool p(nt);
            std::vector<float> sink(nt);
            auto r = timeit(5, [&] {
                p.run([&](int t, int N) {
                    auto [b, e] = part((int)(n / 8), t, N);
                    __m256 s = _mm256_setzero_ps();
                    for (long i = b; i < e; ++i) s = _mm256_add_ps(s, _mm256_load_ps(a + i * 8));
                    sink[t] = _mm256_cvtss_f32(s);
                });
            });
            std::printf("mem read  threads=%2d  %.1f GB/s (best)\n", nt, n * 4 / r.first / 1e9);
            r = timeit(5, [&] {
                p.run([&](int t, int N) {
                    auto [b, e] = part((int)(n / 8), t, N);
                    for (long i = b; i < e; ++i) _mm256_store_ps(a + i * 8, _mm256_add_ps(_mm256_load_ps(a + i * 8), _mm256_set1_ps(1e-9f)));
                });
            });
            std::printf("mem rd+wr threads=%2d  %.1f GB/s counting read+write bytes (best)\n", nt, 2.0 * n * 4 / r.first / 1e9);
        }
        return 0;
    }
    if (sec == "gdn") {
        const int Hk = 16, Hv = 48, D = 128, L = 48;
        const size_t per = (size_t)Hv * D * D;  // floats per layer = 3 MiB
        float* S = (float*)aligned_alloc(64, per * L * 4);
        fill(S, per * L, 7, -0.1f, 0.1f);
        std::vector<float> conv(2 * Hk * D + Hv * D), alpha(Hv), beta(Hv), dtb(Hv), An(Hv), z(Hv * D), nw(D, 1.f), out(Hv * D);
        fill(conv.data(), conv.size(), 2, -1, 1); fill(alpha.data(), Hv, 3, -1, 1); fill(beta.data(), Hv, 4, -1, 1);
        fill(dtb.data(), Hv, 5, -1, 1); fill(An.data(), Hv, 6, -1, -0.1f); fill(z.data(), z.size(), 8, -2, 2);
        auto mk = [&](int l) { return GdnArgs{conv.data(), alpha.data(), beta.data(), An.data(), dtb.data(), z.data(), nw.data(), S + per * l, out.data(), Hk, Hv, D, D, 1e-6f}; };
        // single thread: 48 layers x 48 heads
        auto r1 = timeit(5, [&] { for (int l = 0; l < L; ++l) gdn_step_range(mk(l), 0, Hv); });
        const double bytes = 2.0 * per * 4 * L;  // read + write
        std::printf("GDN 48 layers, 1 thread : best %.2f ms  median %.2f ms  -> %.2f GB/s (rd+wr %.0f MiB)\n", r1.first * 1e3,
                    r1.second * 1e3, bytes / r1.first / 1e9, bytes / 1048576);
        std::printf("   per layer (1 thread, whole 3 MiB state): %.1f us ; per head %.1f us\n", r1.first / L * 1e6, r1.first / L / Hv * 1e6);
        auto rn = timeit(9, [&] {
            for (int l = 0; l < L; ++l) {
                GdnArgs a = mk(l);
                pool.run([&](int t, int N) { auto [b, e] = part(Hv, t, N); if (e > b) gdn_step_range(a, b, e); });
            }
        });
        std::printf("GDN 48 layers, %d threads (head-partitioned, %d active, barrier per layer): best %.2f ms median %.2f ms -> %.2f GB/s\n",
                    NT, std::min(NT, Hv), rn.first * 1e3, rn.second * 1e3, bytes / rn.first / 1e9);
        // convolution on the same pool
        const int C = 10240;
        std::vector<float> cin(C), cw(C * 4), cs(C * 3), cout(C);
        fill(cin.data(), C, 1, -1, 1); fill(cw.data(), C * 4, 2, -1, 1); fill(cs.data(), C * 3, 3, -1, 1);
        auto c1 = timeit(200, [&] { conv_step_range(cin.data(), cw.data(), cs.data(), cout.data(), 0, C); });
        auto cn = timeit(200, [&] { pool.run([&](int t, int N) { auto [b, e] = part(C, t, N); conv_step_range(cin.data(), cw.data(), cs.data(), cout.data(), b, e); }); });
        std::printf("conv_step C=10240: 1 thread best %.1f us (x48 = %.2f ms) ; %d threads best %.1f us (x48 = %.2f ms)\n", c1.first * 1e6,
                    c1.first * 48e3, NT, cn.first * 1e6, cn.first * 48e3);
        return 0;
    }
    if (sec == "attn") {
        const int Hq = 24, Hkv = 4, D = 256, S = 16;
        for (int n : {4096, 32768, 131072}) {
            const size_t kv = (size_t)Hkv * n * D;
            float* K = (float*)aligned_alloc(64, kv * 4);
            float* V = (float*)aligned_alloc(64, kv * 4);
            fill(K, kv, 11, -1, 1); fill(V, kv, 12, -1, 1);
            std::vector<float> q(Hq * D), qg(Hq * 2 * D), out(Hq * D), part_((size_t)Hq * S * (D + 2));
            fill(q.data(), q.size(), 3, -1, 1); fill(qg.data(), qg.size(), 4, -1, 1);
            AttnDecodeArgs a{q.data(), qg.data(), K, V, part_.data(), out.data(), Hq, Hkv, D, n, n, S};
            const int items = Hkv * S;
            auto r1 = timeit(n >= 131072 ? 2 : 4, [&] { attn_decode_range(a, 0, items); attn_merge_range(a, 0, Hq); });
            const double bytes = 2.0 * kv * 4;
            auto rn = timeit(n >= 131072 ? 5 : 15, [&] {
                pool.run([&](int t, int N) { auto [b, e] = part(items, t, N); if (e > b) attn_decode_range(a, b, e); });
                pool.run([&](int t, int N) { auto [b, e] = part(Hq, t, N); if (e > b) attn_merge_range(a, b, e); });
            });
            std::printf("attn n=%6d  KV/layer %.0f MiB | 1 thread: %.2f ms (%.1f GB/s) | %d threads (%d items, 2 barriers): best %.3f ms median %.3f ms (%.1f GB/s)\n",
                        n, bytes / 1048576, r1.first * 1e3, bytes / r1.first / 1e9, NT, items, rn.first * 1e3, rn.second * 1e3, bytes / rn.first / 1e9);
            free(K); free(V);
        }
        return 0;
    }
    if (sec == "misc") {
        const int H = 5120, I = 17408, V = 248320;
        std::vector<float> x(I), w(I, 1.f), o(I), u(I);
        fill(x.data(), I, 1, -2, 2); fill(u.data(), I, 2, -2, 2);
        auto r = timeit(2000, [&] { rmsnorm(x.data(), w.data(), o.data(), H, 1e-6f); });
        std::printf("rmsnorm n=%d: 1 thread %.2f us\n", H, r.first * 1e6);
        r = timeit(2000, [&] { silu_mul_range(x.data(), u.data(), o.data(), 0, I); });
        std::printf("silu_mul n=%d: 1 thread %.2f us\n", I, r.first * 1e6);
        r = timeit(2000, [&] { add_range(x.data(), u.data(), o.data(), 0, H); });
        std::printf("add n=%d: 1 thread %.2f us\n", H, r.first * 1e6);
        std::vector<float> lg(V);
        fill(lg.data(), V, 5, -10, 10);
        float pv; int pi;
        r = timeit(200, [&] { argmax_range(lg.data(), 0, V, &pv, &pi); });
        std::printf("argmax n=%d: 1 thread %.1f us\n", V, r.first * 1e6);
        r = timeit(200, [&] { pool.run([&](int t, int N) { auto [b, e] = part(V, t, N); float v; int i; argmax_range(lg.data(), b, e, &v, &i); }); });
        std::printf("argmax n=%d: %d threads %.1f us\n", V, NT, r.first * 1e6);
        return 0;
    }
    return 1;
}
