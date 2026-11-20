// attn_prep_seq + attn_seq vs a loop of attn_prep + attn_decode (golden single-token path).
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "../src/backends/cuda/attention/attn.h"
#include "../src/backends/cuda/attention/attn_seq.h"

using namespace qv::cuda;
constexpr int Hq = 24, Hkv = 4, D = 256, RD = 64, MAXCTX = 9300;

template <class T> static T* dev(size_t n) { T* p; QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); return p; }
template <class T> static T* up(const std::vector<T>& v) {
    T* p = dev<T>(v.size());
    QV_CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static std::vector<float> rnd(size_t n, std::mt19937& g) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(g);
    return v;
}

struct Data {
    int N;  // total tokens (prefix + T)
    float *qg, *k, *v, *qn, *kn, *invf;
};

static int g_fail = 0;

// run reference loop for tokens [t_begin, t_end) of data d (position = token index); writes out_ref[(t-t_begin)*Hq*D]
static void ref_loop(const Data& d, int t_begin, int t_end, float* kc, float* vc, float* out_ref, int* dpos) {
    float* qbuf = dev<float>(Hq * D);
    float* scores = dev<float>((size_t)Hq * MAXCTX);
    for (int t = t_begin; t < t_end; ++t) {
        AttnPrepArgs pa{d.qg + (size_t)t * Hq * 2 * D, d.k + (size_t)t * Hkv * D, d.v + (size_t)t * Hkv * D, d.qn, d.kn, d.invf,
                        qbuf, kc, vc, Hq, Hkv, D, RD, MAXCTX, dpos + t, 1e-6f};
        attn_prep(pa, 0);
        AttnDecodeArgs da{};
        da.q = qbuf; da.qg = d.qg + (size_t)t * Hq * 2 * D; da.k_cache = kc; da.v_cache = vc; da.scores = scores;
        da.out = out_ref ? out_ref + (size_t)(t - t_begin) * Hq * D : qbuf;  // prefix: output discarded into qbuf (after use)
        da.n_heads = Hq; da.n_kv_heads = Hkv; da.max_ctx = MAXCTX; da.pos = dpos + t;
        if (!out_ref) da.out = scores + (size_t)Hq * MAXCTX - Hq * D;  // dummy, beyond any used score index
        attn_decode(da, 0);
    }
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(qbuf); cudaFree(scores);
}

static AttnSeqPrepArgs prep_args(const Data& d, int pos0, int T, float* qout, float* kc, float* vc) {
    return AttnSeqPrepArgs{d.qg + (size_t)pos0 * Hq * 2 * D, d.k + (size_t)pos0 * Hkv * D, d.v + (size_t)pos0 * Hkv * D, d.qn, d.kn, d.invf,
                           qout, kc, vc, Hq, Hkv, D, RD, MAXCTX, pos0, T, 1e-6f};
}

static void run_case(int pos0, int T, std::mt19937& g) {
    Data d; d.N = pos0 + T;
    d.qg = up(rnd((size_t)d.N * Hq * 2 * D, g));
    d.k = up(rnd((size_t)d.N * Hkv * D, g));
    d.v = up(rnd((size_t)d.N * Hkv * D, g));
    std::vector<float> qn = rnd(D, g), kn = rnd(D, g);
    for (auto& x : qn) x = 1.f + 0.1f * x;
    for (auto& x : kn) x = 1.f + 0.1f * x;
    d.qn = up(qn); d.kn = up(kn);
    std::vector<float> invf(RD / 2);
    for (int i = 0; i < RD / 2; ++i) invf[i] = powf(1e7f, -2.f * i / RD);
    d.invf = up(invf);
    std::vector<int> posv(d.N);
    for (int i = 0; i < d.N; ++i) posv[i] = i;
    int* dpos = up(posv);
    size_t csz = (size_t)Hkv * MAXCTX * D;
    float *kc_r = dev<float>(csz), *vc_r = dev<float>(csz), *kc_s = dev<float>(csz), *vc_s = dev<float>(csz);
    QV_CUDA_CHECK(cudaMemset(kc_r, 0, csz * 4)); QV_CUDA_CHECK(cudaMemset(vc_r, 0, csz * 4));
    if (pos0 > 0) ref_loop(d, 0, pos0, kc_r, vc_r, nullptr, dpos);
    QV_CUDA_CHECK(cudaMemcpy(kc_s, kc_r, csz * 4, cudaMemcpyDeviceToDevice));
    QV_CUDA_CHECK(cudaMemcpy(vc_s, vc_r, csz * 4, cudaMemcpyDeviceToDevice));
    size_t on = (size_t)T * Hq * D;
    float *out_ref = dev<float>(on), *out_f = dev<float>(on), *qprep = dev<float>(on);
    __half* out_h = dev<__half>(on);
    ref_loop(d, pos0, d.N, kc_r, vc_r, out_ref, dpos);
    attn_prep_seq(prep_args(d, pos0, T, qprep, kc_s, vc_s), 0);
    AttnSeqArgs sa{qprep, d.qg + (size_t)pos0 * Hq * 2 * D, kc_s, vc_s, out_h, out_f, Hq, Hkv, MAXCTX, pos0, T};
    attn_seq(sa, 0);
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> r(on), s(on), kr(csz), ks(csz), vr(csz), vs(csz);
    std::vector<__half> h(on);
    cudaMemcpy(r.data(), out_ref, on * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.data(), out_f, on * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h.data(), out_h, on * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(kr.data(), kc_r, csz * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ks.data(), kc_s, csz * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(vr.data(), vc_r, csz * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(vs.data(), vc_s, csz * 4, cudaMemcpyDeviceToHost);
    double en = 0, rn = 0, hn = 0;
    for (size_t i = 0; i < on; ++i) {
        double e = s[i] - r[i]; en += e * e; rn += (double)r[i] * r[i];
        double e2 = __half2float(h[i]) - r[i]; hn += e2 * e2;
    }
    double rel = std::sqrt(en / rn), relh = std::sqrt(hn / rn);
    size_t kdiff = 0; double maxd = 0;
    for (size_t i = 0; i < csz; ++i) {
        if (std::memcmp(&kr[i], &ks[i], 4) || std::memcmp(&vr[i], &vs[i], 4)) ++kdiff;
        maxd = std::fmax(maxd, std::fmax(std::fabs(kr[i] - ks[i]), std::fabs(vr[i] - vs[i])));
    }
    bool ok = rel <= 2e-4 && maxd <= 1e-5 && relh < 2e-3;
    std::printf("pos0=%5d T=%4d  out_f rel_err=%.3e  out_h rel_err=%.3e  cache non-bitwise=%zu maxabs=%.2e  %s\n", pos0, T, rel, relh, kdiff,
                maxd, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
    cudaFree(const_cast<float*>(d.qg)); cudaFree(const_cast<float*>(d.k)); cudaFree(const_cast<float*>(d.v));
    cudaFree(const_cast<float*>(d.qn)); cudaFree(const_cast<float*>(d.kn)); cudaFree(const_cast<float*>(d.invf));
    cudaFree(dpos); cudaFree(kc_r); cudaFree(vc_r); cudaFree(kc_s); cudaFree(vc_s);
    cudaFree(out_ref); cudaFree(out_f); cudaFree(qprep); cudaFree(out_h);
}

static void bench(int pos0, int T, std::mt19937& g) {
    int N = pos0 + T;
    float* qg = up(rnd((size_t)T * Hq * 2 * D, g));
    float* q = up(rnd((size_t)T * Hq * D, g));
    size_t csz = (size_t)Hkv * MAXCTX * D;
    std::vector<float> kv = rnd(csz, g);
    float *kc = up(kv), *vc = up(kv);
    float* out = dev<float>((size_t)T * Hq * D);
    __half* oh = dev<__half>((size_t)T * Hq * D);
    AttnSeqArgs sa{q, qg, kc, vc, oh, out, Hq, Hkv, MAXCTX, pos0, T};
    for (int i = 0; i < 3; ++i) attn_seq(sa, 0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) attn_seq(sa, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    double keys = (double)T * pos0 + (double)T * (T + 1) / 2;
    double fl = 4.0 * Hq * D * keys;
    std::printf("bench pos0=%5d T=%d: %.3f ms  %.2f TFLOP/s (causal-useful FLOPs)\n", pos0, T, ms, fl / (ms * 1e-3) / 1e12);
    (void)N;
}

int main() {
    std::mt19937 g(1234);
    for (int T : {1, 5, 64, 257}) run_case(0, T, g);
    run_case(1000, 64, g);
    bench(0, 1024, g);
    bench(8192, 1024, g);
    std::printf(g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail ? 1 : 0;
}
