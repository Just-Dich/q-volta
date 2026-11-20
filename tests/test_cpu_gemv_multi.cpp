// gemv_multi_q (decode-time multi-token GEMV, T <= 4) must be BIT-IDENTICAL to gemv_q per token (speculative-decoding
// verification relies on this). Also checks the threaded variant.
#include <cstdio>
#include <cstring>
#include <random>

#include "../src/backends/cpu/cpu_gemv.h"
#include "../src/backends/cpu/qv_pack.h"

using namespace qv::cpu;

int main() {
    std::mt19937_64 rng(7);
    ThreadPool pool(8, {});
    int fail = 0;
    for (int bits : {2, 3, 4, 5, 6})
        for (int T = 1; T <= 4; ++T)
            for (auto [N, K] : {std::pair{37, 192}, std::pair{128, 1024}, std::pair{515, 2560}}) {
                const int G = K / 64;
                std::vector<uint8_t> q((size_t)N * K);
                for (auto& v : q) v = (uint8_t)(rng() & ((1u << bits) - 1));
                auto words = pack_q(q.data(), N, K, bits);
                std::vector<uint16_t> sm((size_t)N * G * 2);
                std::uniform_real_distribution<float> us(0.002f, 0.03f), um(-0.2f, 0.05f);
                for (size_t i = 0; i < sm.size(); i += 2) { sm[i] = _cvtss_sh(us(rng), 0); sm[i + 1] = _cvtss_sh(um(rng), 0); }
                QMatC W{words.data(), sm.data(), N, K, bits};
                Q8Batch X;
                X.resize(T, K);
                std::normal_distribution<float> nd(0.f, 1.f);
                std::vector<float> x(K);
                for (int t = 0; t < T; ++t) { for (auto& v : x) v = nd(rng); X.quantize_row(t, x.data(), K); }
                std::vector<float> y((size_t)T * N), yp((size_t)T * N), yr(N);
                gemv_multi_q(W, X, T, y.data(), N, 0, N);
                gemv_multi_q_par(pool, W, X, T, yp.data(), N);
                bool ok = std::memcmp(y.data(), yp.data(), y.size() * 4) == 0;
                for (int t = 0; t < T; ++t) {
                    Q8ActC a;
                    a.K = K;
                    a.q.assign(X.q.begin() + (size_t)t * K, X.q.begin() + (size_t)(t + 1) * K);
                    a.ds.assign(X.ds.begin() + (size_t)t * G * 4, X.ds.begin() + (size_t)(t + 1) * G * 4);
                    gemv_q(W, a, yr.data(), 0, N);
                    ok = ok && std::memcmp(yr.data(), y.data() + (size_t)t * N, (size_t)N * 4) == 0;
                }
                if (!ok) { std::printf("FAIL bits=%d T=%d N=%d K=%d\n", bits, T, N, K); ++fail; }
            }
    std::printf(fail ? "gemv_multi: %d FAILED\n" : "gemv_multi: all bit-identical to gemv_q\n", fail);
    return fail != 0;
}
