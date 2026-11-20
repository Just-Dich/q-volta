// Batched CPU GEMM (prefill kernel) vs per-token gemv_q: same weights, same int8 activations, T tokens.
// Usage: test_cpu_gemm            correctness
//        test_cpu_gemm bench N K bits T threads   throughput of gemm_q_par vs T x gemv_q_par
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "../src/backends/cpu/cpu_gemv.h"
#include "../src/backends/cpu/qv_pack.h"

using namespace qv::cpu;
static int g_fail = 0;

struct Case {
    std::vector<uint32_t> words;
    std::vector<uint16_t> sm;
    QMatC W;
    Q8Batch X;
    int T;
};

static Case make(int N, int K, int bits, int T, std::mt19937_64& rng) {
    Case c;
    const int G = K / 64;
    std::vector<uint8_t> q((size_t)N * K);
    for (auto& v : q) v = (uint8_t)(rng() & ((1u << bits) - 1));
    c.words = pack_q(q.data(), N, K, bits);
    c.sm.resize((size_t)N * G * 2);
    std::uniform_real_distribution<float> us(0.002f, 0.03f), um(-0.2f, 0.05f);
    for (size_t i = 0; i < c.sm.size(); i += 2) { c.sm[i] = _cvtss_sh(us(rng), 0); c.sm[i + 1] = _cvtss_sh(um(rng), 0); }
    c.W = QMatC{c.words.data(), c.sm.data(), N, K, bits};
    c.T = T;
    c.X.resize(T, K);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(K);
    for (int t = 0; t < T; ++t) {
        for (auto& v : x) v = nd(rng) * (1.f + (rng() % 7 == 0 ? 20.f : 0.f));
        if (t == 2) std::fill(x.begin(), x.end(), 0.f);  // all-zero token
        c.X.quantize_row(t, x.data(), K);
    }
    return c;
}

static Q8ActC row_of(const Q8Batch& X, int t) {
    Q8ActC a;
    a.K = X.K;
    a.q.assign(X.q.begin() + (size_t)t * X.K, X.q.begin() + (size_t)(t + 1) * X.K);
    a.ds.assign(X.ds.begin() + (size_t)t * X.G * 4, X.ds.begin() + (size_t)(t + 1) * X.G * 4);
    return a;
}

static void run_case(int N, int K, int bits, int T, std::mt19937_64& rng, ThreadPool& pool) {
    Case c = make(N, K, bits, T, rng);
    std::vector<float> y((size_t)T * N, -1.f), yp((size_t)T * N, -2.f), yr(N);
    gemm_q(c.W, c.X, T, y.data(), N, 0, N);
    gemm_q_par(pool, c.W, c.X, T, yp.data(), N);
    double worst = 0, worst_el = 0;
    for (int t = 0; t < T; ++t) {
        Q8ActC a = row_of(c.X, t);
        gemv_q(c.W, a, yr.data(), 0, N);
        double en = 0, rn = 0, mx = 0;
        for (int n = 0; n < N; ++n) {
            const double d = (double)y[(size_t)t * N + n] - yr[n];
            en += d * d; rn += (double)yr[n] * yr[n];
            mx = std::fmax(mx, std::fabs(d));
        }
        double maxref = 0;
        for (int n = 0; n < N; ++n) maxref = std::fmax(maxref, std::fabs(yr[n]));
        worst = std::fmax(worst, std::sqrt(en / (rn > 0 ? rn : 1)));
        worst_el = std::fmax(worst_el, mx / (maxref > 0 ? maxref : 1));
    }
    const bool par_same = std::memcmp(y.data(), yp.data(), y.size() * 4) == 0;
    std::printf("N=%5d K=%5d bits=%d T=%3d  rel_l2=%.2e  maxabs/maxref=%.2e  par==serial:%d\n", N, K, bits, T, worst, worst_el, par_same);
    if (!(worst < 1e-6 && worst_el < 1e-6) || !par_same) { std::printf("FAIL\n"); ++g_fail; }
}

static void bench(int N, int K, int bits, int T, int nth) {
    std::mt19937_64 rng(7);
    Case c = make(N, K, bits, T, rng);
    ThreadPool pool(nth);
    std::vector<float> y((size_t)T * N);
    using clk = std::chrono::steady_clock;
    for (int w = 0; w < 2; ++w) gemm_q_par(pool, c.W, c.X, T, y.data(), N);
    const int reps = std::getenv("QV_REPS") ? atoi(std::getenv("QV_REPS")) : 5;
    auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) gemm_q_par(pool, c.W, c.X, T, y.data(), N);
    const double s = std::chrono::duration<double>(clk::now() - t0).count() / reps;
    const double mac = (double)N * K * T;
    std::printf("gemm_q_par N=%d K=%d bits=%d T=%d threads=%d: %.2f ms  %.1f GMAC/s  (%.2f GMAC/s/thread, %.1f tok-rows/s)\n", N, K, bits,
                T, nth, s * 1e3, mac / s / 1e9, mac / s / 1e9 / nth, T / s);
    std::vector<float> y1(N);
    Q8ActC a = row_of(c.X, 0);
    auto t1 = clk::now();
    for (int i = 0; i < reps; ++i) gemv_q_par(pool, c.W, a, y1.data());
    const double sg = std::chrono::duration<double>(clk::now() - t1).count() / reps;
    std::printf("  gemv_q_par one token: %.3f ms -> %.1f GMAC/s; gemm/gemv speedup per token: %.1fx\n", sg * 1e3, (double)N * K / sg / 1e9, sg * T / s);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "bench") { bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6])); return 0; }
    std::mt19937_64 rng(99);
    ThreadPool pool(8);
    for (int bits : {2, 3, 4, 5, 6, 8})
        for (int T : {1, 3, 4, 7, 16}) run_case(13, 128, bits, T, rng, pool);
    for (int bits : {3, 4, 6}) {
        for (int T : {1, 5, 33}) run_case(1000, 5120, bits, T, rng, pool);
        run_case(64, 17408, bits, 9, rng, pool);  // K spans several chunks plus a partial one
        run_case(6144, 5120, bits, 64, rng, pool);
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
