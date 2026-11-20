// Split-K attn_decode vs the original single-pass kernel (attn_decode_ref): accuracy + per-call timing.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/backends/cuda/attention/attn.h"

using namespace qv::cuda;
constexpr int Hq = 24, Hkv = 4, D = 256, MAXCTX = 32768;

template <class T> static T* dev(size_t n) { T* p; QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); return p; }
static float* up(const std::vector<float>& v) {
    float* p = dev<float>(v.size());
    QV_CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}
static std::vector<float> rnd(size_t n, std::mt19937& g) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(g);
    return v;
}
struct Q8Host { std::vector<int8_t> q; std::vector<float> d, sum; std::vector<int> qsum; };
static Q8Buf q8_alloc() {
    Q8Buf b;
    b.q = dev<int>(Hq * D / 4); b.d = dev<float>(Hq * D / 32); b.sum = dev<float>(Hq * D / 32); b.qsum = dev<int>(Hq * D / 32);
    return b;
}
static Q8Host q8_down(const Q8Buf& b) {
    Q8Host h; h.q.resize(Hq * D); h.d.resize(Hq * D / 32); h.sum.resize(Hq * D / 32); h.qsum.resize(Hq * D / 32);
    cudaMemcpy(h.q.data(), b.q, Hq * D, cudaMemcpyDeviceToHost);
    cudaMemcpy(h.d.data(), b.d, h.d.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h.sum.data(), b.sum, h.sum.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h.qsum.data(), b.qsum, h.qsum.size() * 4, cudaMemcpyDeviceToHost);
    return h;
}

template <class F> static float time_us(F f, int runs = 20) {
    for (int i = 0; i < 5; ++i) f();  // warmup
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ts;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms * 1000.f);
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];  // median
}

int main() {
    std::mt19937 g(1234);
    const size_t kvn = (size_t)Hkv * MAXCTX * D;
    float *kc = up(rnd(kvn, g)), *vc = up(rnd(kvn, g));
    float *q = up(rnd(Hq * D, g)), *qg = up(rnd((size_t)Hq * 2 * D, g));
    float* scores = dev<float>((size_t)Hq * MAXCTX);
    float *out_ref = dev<float>(Hq * D), *out_new = dev<float>(Hq * D);
    float* part = dev<float>((size_t)Hq * ATTN_MAX_SPLITS * (D + 2));
    Q8Buf q8r = q8_alloc(), q8n = q8_alloc();
    int* dpos = dev<int>(1);

    const int positions[] = {0, 1, 63, 511, 4095, 16383, 32767};
    int fail = 0;
    printf("%7s %12s %10s %8s %10s %10s %8s\n", "pos", "max|d|/max|r|", "rel_l2", "q8 maxLSB", "old_us", "new_us", "speedup");
    for (int pos : positions) {
        QV_CUDA_CHECK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
        AttnDecodeArgs ar{q, qg, kc, vc, scores, out_ref, Hq, Hkv, MAXCTX, dpos, q8r};
        AttnDecodeArgs an{q, qg, kc, vc, nullptr, out_new, Hq, Hkv, MAXCTX, dpos, q8n, part, ATTN_MAX_SPLITS};
        attn_decode_ref(ar, 0);
        attn_decode(an, 0);
        QV_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> r(Hq * D), n(Hq * D);
        cudaMemcpy(r.data(), out_ref, Hq * D * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(n.data(), out_new, Hq * D * 4, cudaMemcpyDeviceToHost);
        double maxd = 0, maxr = 0, d2 = 0, r2 = 0;
        for (int i = 0; i < Hq * D; ++i) {
            maxd = std::max(maxd, (double)std::fabs(r[i] - n[i])); maxr = std::max(maxr, (double)std::fabs(r[i]));
            d2 += (double)(r[i] - n[i]) * (r[i] - n[i]); r2 += (double)r[i] * r[i];
        }
        double rel = maxd / maxr, rel2 = std::sqrt(d2 / r2);
        Q8Host a = q8_down(q8r), b = q8_down(q8n);
        int maxlsb = 0, nneq = 0;
        for (int i = 0; i < Hq * D; ++i) {
            int df = std::abs((int)a.q[i] - (int)b.q[i]); maxlsb = std::max(maxlsb, df); nneq += df != 0;
        }
        float told = time_us([&] { attn_decode_ref(ar, 0); });
        float tnew = time_us([&] { attn_decode(an, 0); });
        printf("%7d %12.3e %10.3e %4d(%d ne) %10.1f %10.1f %7.2fx\n", pos, rel, rel2, maxlsb, nneq, told, tnew, told / tnew);
        if (!(rel <= 1e-5) || maxlsb > 1) { ++fail; printf("  FAIL at pos %d\n", pos); }
    }
    printf(fail ? "FAILED\n" : "PASS\n");
    return fail != 0;
}
