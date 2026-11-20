// CPU int4 attention over the cold KV store (Gen 6 H-C), and over a scattered top-k subset (H-B/H-C hybrid), at 256K context.
// One layer's store = [kv head][token][144 B] for K and V (302 MB at 262144 tokens); 16 layers are allocated and rotated so the
// data is never cache-resident (host RAM 16 MiB L3). Output: JSON on stdout.
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include "../src/cpu/attn_cpu.h"
using namespace qv::cpu;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    const int64_t ntok = argc > 1 ? std::atoll(argv[1]) : 262144;
    const int NLAY = 16, nblk = (int)(ntok / 64);
    const size_t lay = (size_t)kHkv * ntok * kRowBytes;
    std::vector<uint8_t*> K(NLAY), V(NLAY);
    for (int l = 0; l < NLAY; ++l) {
        K[l] = (uint8_t*)std::aligned_alloc(4096, lay); V[l] = (uint8_t*)std::aligned_alloc(4096, lay);
        fill_random_cache(K[l], ntok, 100 + l); fill_random_cache(V[l], ntok, 200 + l);
    }
    std::mt19937 g(1); std::normal_distribution<float> nd;
    std::vector<float> q(kHq * kD), m(kHq), lse(kHq), acc(kHq * kD);
    for (auto& x : q) x = nd(g);
    std::printf("{\"ctx_tokens\": %ld, \"kv_bytes_per_layer_MB\": %.1f, \"full_scan\": [", (long)ntok, 2.0 * lay / 1e6);
    bool first = true;
    for (int nt : {8, 16, 32, 64}) {
        std::vector<double> t;
        for (int r = 0; r < 24; ++r) {
            int l = r % NLAY;
            double t0 = now();
            attn_cpu_partial(q.data(), K[l], V[l], ntok, m.data(), lse.data(), acc.data(), nt);
            if (r >= 4) t.push_back((now() - t0) * 1e3);
        }
        std::sort(t.begin(), t.end());
        std::printf("%s\n  {\"threads\": %d, \"ms_per_layer_median\": %.3f, \"ms_min\": %.3f, \"gbps\": %.1f, \"ms_per_token_16_layers\": %.1f}", first ? "" : ",", nt, t[t.size() / 2], t[0], 2.0 * lay / (t[t.size() / 2] * 1e-3) / 1e9, 16 * t[t.size() / 2]);
        first = false;
    }
    std::printf("],\n \"subset\": [");
    first = true;
    for (int fetch : {1024, 2048, 4096, 8192, 16384}) {
        const int kb = fetch / 64;
        for (int nt : {4, 8, 16, 32}) {
            std::vector<double> t;
            std::vector<int32_t> blk((size_t)kHkv * kb);
            std::vector<int> ids(nblk); std::iota(ids.begin(), ids.end(), 0);
            for (int r = 0; r < 40; ++r) {
                for (int h = 0; h < kHkv; ++h) { std::shuffle(ids.begin(), ids.end(), g); for (int i = 0; i < kb; ++i) blk[h * kb + i] = ids[i]; }
                int l = r % NLAY;
                double t0 = now();
                attn_cpu_partial_blocks(q.data(), K[l], V[l], ntok, blk.data(), kb, m.data(), lse.data(), acc.data(), nt, 2);
                if (r >= 8) t.push_back((now() - t0) * 1e3);
            }
            std::sort(t.begin(), t.end());
            const double bytes = 2.0 * kHkv * fetch * kRowBytes;
            std::printf("%s\n  {\"fetch_tokens_per_head\": %d, \"threads\": %d, \"ms_per_layer_median\": %.4f, \"ms_min\": %.4f, \"bytes_MB\": %.2f, \"gbps\": %.1f}", first ? "" : ",", fetch, nt, t[t.size() / 2], t[0], bytes / 1e6, bytes / (t[t.size() / 2] * 1e-3) / 1e9);
            first = false;
        }
    }
    std::printf("]}\n");
    return 0;
}
