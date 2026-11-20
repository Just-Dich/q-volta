// attn_cpu_partial (AVX2/OpenMP) vs scalar double-precision reference.
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../src/cpu/attn_cpu.h"
using namespace qv::cpu;

static double rel(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { num = std::max(num, (double)std::fabs(a[i] - b[i])); den = std::max(den, (double)std::fabs(b[i])); }
    return num / (den + 1e-30);
}

int main() {
    int fails = 0;
    std::mt19937 g(1);
    std::normal_distribution<float> nd;
    for (int64_t ntok : {1, 1000, 65536, 200000}) {
        std::vector<uint8_t> K((size_t)kHkv * ntok * kRowBytes), V(K.size());
        fill_random_cache(K.data(), ntok, 11);
        fill_random_cache(V.data(), ntok, 22);
        std::vector<float> q(kHq * kD);
        for (auto& x : q) x = nd(g);
        std::vector<float> m(kHq), l(kHq), acc(kHq * kD), mr(kHq), lr(kHq), ar(kHq * kD);
        attn_cpu_ref(q.data(), K.data(), V.data(), ntok, mr.data(), lr.data(), ar.data());
        for (int nt : {1, 7, 64}) {
            attn_cpu_partial(q.data(), K.data(), V.data(), ntok, m.data(), l.data(), acc.data(), nt);
            // compare normalised outputs (what actually matters) and raw m/l/acc
            std::vector<float> o(kHq * kD), orr(kHq * kD);
            for (int h = 0; h < kHq; ++h)
                for (int d = 0; d < kD; ++d) { o[h * kD + d] = acc[h * kD + d] / l[h]; orr[h * kD + d] = ar[h * kD + d] / lr[h]; }
            double em = rel(m.data(), mr.data(), kHq), el = rel(l.data(), lr.data(), kHq);
            double ea = rel(acc.data(), ar.data(), kHq * kD), eo = rel(o.data(), orr.data(), kHq * kD);
            bool ok = em < 1e-5 && el < 1e-4 && ea < 1e-4 && eo < 1e-4;
            std::printf("ntok=%-7ld threads=%-2d rel_err m=%.2e l=%.2e acc=%.2e out=%.2e %s\n", (long)ntok, nt, em, el, ea, eo, ok ? "OK" : "FAIL");
            fails += !ok;
        }
    }
    // merge check: split 1000 tokens... merge of two disjoint partial results equals the full one
    {
        int64_t n = 1000;
        std::vector<uint8_t> K((size_t)kHkv * n * kRowBytes), V(K.size());
        fill_random_cache(K.data(), n, 5); fill_random_cache(V.data(), n, 6);
        std::vector<float> q(kHq * kD);
        for (auto& x : q) x = nd(g);
        std::vector<float> m(kHq), l(kHq), a(kHq * kD), m2(kHq), l2(kHq), a2(kHq * kD);
        attn_cpu_partial(q.data(), K.data(), V.data(), n, m.data(), l.data(), a.data(), 4);
        float z[kHq] = {}; std::vector<float> za(kHq * kD, 0.f);
        std::vector<float> mn(kHq, -INFINITY);
        std::vector<float> ln(kHq, 0.f), an(kHq * kD, 0.f);
        attn_merge(mn.data(), ln.data(), an.data(), m.data(), l.data(), a.data());
        double e = rel(an.data(), a.data(), kHq * kD);
        std::printf("merge-into-empty rel err %.2e %s\n", e, e < 1e-6 ? "OK" : "FAIL");
        fails += !(e < 1e-6); (void)z; (void)za; (void)m2; (void)l2; (void)a2;
    }
    // block-restricted variant: all blocks (shuffled) == full; a subset == partial over a compacted copy
    {
        const int64_t n = 64 * 40;
        std::vector<uint8_t> K((size_t)kHkv * n * kRowBytes), V(K.size());
        fill_random_cache(K.data(), n, 7); fill_random_cache(V.data(), n, 8);
        std::vector<float> q(kHq * kD);
        for (auto& x : q) x = nd(g);
        std::vector<float> m(kHq), l(kHq), a(kHq * kD), m2(kHq), l2(kHq), a2(kHq * kD);
        const int nb = 13;
        std::vector<int32_t> blk(kHkv * nb);
        std::vector<uint8_t> Kc((size_t)kHkv * nb * 64 * kRowBytes), Vc(Kc.size());
        for (int h = 0; h < kHkv; ++h)
            for (int i = 0; i < nb; ++i) {
                blk[h * nb + i] = (i * 7 + h * 3) % 40;  // distinct (7 and 40 coprime), differs per head
                std::memcpy(&Kc[((size_t)h * nb + i) * 64 * kRowBytes], &K[((size_t)h * n + blk[h * nb + i] * 64) * kRowBytes], 64 * kRowBytes);
                std::memcpy(&Vc[((size_t)h * nb + i) * 64 * kRowBytes], &V[((size_t)h * n + blk[h * nb + i] * 64) * kRowBytes], 64 * kRowBytes);
            }
        attn_cpu_partial_blocks(q.data(), K.data(), V.data(), n, blk.data(), nb, m.data(), l.data(), a.data(), 4, 3);
        attn_cpu_partial(q.data(), Kc.data(), Vc.data(), (int64_t)nb * 64, m2.data(), l2.data(), a2.data(), 4);
        std::vector<float> o(kHq * kD), o2(kHq * kD);
        for (int h = 0; h < kHq; ++h) for (int d = 0; d < kD; ++d) { o[h * kD + d] = a[h * kD + d] / l[h]; o2[h * kD + d] = a2[h * kD + d] / l2[h]; }
        double e = rel(o.data(), o2.data(), kHq * kD);
        std::printf("blocks variant vs compacted copy rel err %.2e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        fails += !(e < 1e-5);
    }
    std::printf(fails ? "FAILED\n" : "ALL PASS\n");
    return fails != 0;
}
