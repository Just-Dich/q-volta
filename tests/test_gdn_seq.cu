// Batched GDN prefill kernels vs a loop of the single-token decode kernels (conv_step + gdn_step).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include "../src/backends/cuda/recurrent/gdn.h"
#include "../src/backends/cuda/recurrent/gdn_seq.h"

using namespace qv::cuda;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

constexpr int Hk = 16, Hv = 48, DK = 128, DV = 128;
constexpr int Cc = 2 * Hk * DK + Hv * DV;  // 10240

template <class V> static float* up(const V& v) {
    float* d; CK(cudaMalloc(&d, v.size() * 4)); CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return d;
}
static float* dalloc(size_t n) { float* d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemset(d, 0, n * 4)); return d; }
static std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v;
}
// max |a-b| / (max|b| + tiny): scale-relative error, robust to near-zero elements
static double relerr(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, (double)std::fabs(b[i])); d = std::max(d, (double)std::fabs(a[i] - b[i])); }
    return d / (m + 1e-12);
}

struct Params {
    std::vector<float> A, dtb, nw, convw;
    float *dA, *ddtb, *dnw, *dconvw;
};
struct Inputs {
    int T;
    std::vector<float> conv_in, conv_out, alpha, beta, z;
    float *d_conv_in, *d_conv_out, *d_alpha, *d_beta, *d_z;
};

static Inputs make_inputs(int T, std::mt19937& g) {
    std::normal_distribution<float> n01(0.f, 1.f);
    Inputs in; in.T = T;
    auto fill = [&](std::vector<float>& v, size_t n) { v.resize(n); for (auto& x : v) x = n01(g); };
    fill(in.conv_in, (size_t)T * Cc); fill(in.conv_out, (size_t)T * Cc); fill(in.alpha, (size_t)T * Hv);
    fill(in.beta, (size_t)T * Hv); fill(in.z, (size_t)T * Hv * DV);
    in.d_conv_in = up(in.conv_in); in.d_conv_out = up(in.conv_out); in.d_alpha = up(in.alpha);
    in.d_beta = up(in.beta); in.d_z = up(in.z);
    return in;
}

static GdnSeqArgs seq_args(const Inputs& in, const Params& p, float* st, __half* oh, float* of, int off, int T) {
    GdnSeqArgs a{};
    a.conv_out = in.d_conv_out + (size_t)off * Cc; a.alpha = in.d_alpha + (size_t)off * Hv;
    a.beta_logit = in.d_beta + (size_t)off * Hv; a.A_neg = p.dA; a.dt_bias = p.ddtb;
    a.z = in.d_z + (size_t)off * Hv * DV; a.norm_w = p.dnw; a.state = st;
    a.out_h = oh ? oh + (size_t)off * Hv * DV : nullptr; a.out_f = of ? of + (size_t)off * Hv * DV : nullptr;
    a.n_k_heads = Hk; a.n_v_heads = Hv; a.dk = DK; a.dv = DV; a.T = T; a.eps = 1e-6f;
    return a;
}

static void ref_gdn(const Inputs& in, const Params& p, float* st, float* out, int T) {
    for (int t = 0; t < T; ++t) {
        GdnStepArgs a{};
        a.conv_out = in.d_conv_out + (size_t)t * Cc; a.alpha = in.d_alpha + (size_t)t * Hv;
        a.beta_logit = in.d_beta + (size_t)t * Hv; a.A_neg = p.dA; a.dt_bias = p.ddtb;
        a.z = in.d_z + (size_t)t * Hv * DV; a.norm_w = p.dnw; a.state = st; a.out = out + (size_t)t * Hv * DV;
        a.n_k_heads = Hk; a.n_v_heads = Hv; a.dk = DK; a.dv = DV; a.eps = 1e-6f;
        gdn_step(a, 0);
    }
}

static int fails = 0;
static void check(const char* name, double e, double tol) {
    bool ok = e <= tol; if (!ok) ++fails;
    printf("  %-34s rel err %.3e  %s\n", name, e, ok ? "OK" : "FAIL");
}

int main() {
    std::mt19937 g(1234);
    std::normal_distribution<float> n01(0.f, 1.f), n05(0.f, 0.5f);
    Params p;
    p.A.resize(Hv); p.dtb.resize(Hv); p.nw.resize(DV); p.convw.resize((size_t)Cc * 4);
    for (auto& x : p.A) x = -std::exp(n05(g));
    for (auto& x : p.dtb) x = n05(g);
    for (auto& x : p.nw) x = 1.f + 0.1f * n01(g);
    for (auto& x : p.convw) x = 0.5f * n01(g);
    p.dA = up(p.A); p.ddtb = up(p.dtb); p.dnw = up(p.nw); p.dconvw = up(p.convw);
    std::vector<float> st0((size_t)Hv * DK * DV);
    for (auto& x : st0) x = 0.1f * n01(g);

    for (int T : {1, 2, 7, 64, 1024}) {
        printf("T=%d\n", T);
        Inputs in = make_inputs(T, g);
        // --- conv
        std::vector<float> cs0((size_t)Cc * 3); for (auto& x : cs0) x = n01(g);
        float *cs_a = up(cs0), *cs_b = up(cs0);
        float* co_a = dalloc((size_t)T * Cc); float* co_b = dalloc((size_t)T * Cc);
        conv_seq(in.d_conv_in, p.dconvw, cs_a, co_a, Cc, T, 0);
        for (int t = 0; t < T; ++t) conv_step(in.d_conv_in + (size_t)t * Cc, p.dconvw, cs_b, co_b + (size_t)t * Cc, Cc, 0);
        CK(cudaDeviceSynchronize());
        check("conv out", relerr(down(co_a, (size_t)T * Cc), down(co_b, (size_t)T * Cc)), 1e-6);
        check("conv state", relerr(down(cs_a, Cc * 3), down(cs_b, Cc * 3)), 1e-6);
        // --- gdn
        float *st_a = up(st0), *st_b = up(st0);
        size_t no = (size_t)T * Hv * DV;
        float *of_a = dalloc(no), *of_b = dalloc(no);
        __half* oh; CK(cudaMalloc(&oh, no * 2));
        gdn_seq(seq_args(in, p, st_a, oh, of_a, 0, T), 0);
        ref_gdn(in, p, st_b, of_b, T);
        CK(cudaDeviceSynchronize());
        auto fa = down(of_a, no), fb = down(of_b, no);
        check("gdn out_f", relerr(fa, fb), 1e-4);
        check("gdn state", relerr(down(st_a, st0.size()), down(st_b, st0.size())), 1e-4);
        std::vector<__half> hh(no); CK(cudaMemcpy(hh.data(), oh, no * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < no; ++i) {
            float c = std::fmin(std::fmax(fa[i], -65504.f), 65504.f);
            if (__half2float(hh[i]) != __half2float(__float2half(c))) ++bad;
        }
        printf("  %-34s mismatches %zu  %s\n", "gdn out_h == half(out_f)", bad, bad ? "FAIL" : "OK");
        if (bad) ++fails;
        // null outputs allowed
        gdn_seq(seq_args(in, p, st_a, nullptr, of_a, 0, T), 0);
        gdn_seq(seq_args(in, p, st_a, oh, nullptr, 0, T), 0);
        CK(cudaDeviceSynchronize());
        cudaFree(cs_a); cudaFree(cs_b); cudaFree(co_a); cudaFree(co_b); cudaFree(st_a); cudaFree(st_b);
        cudaFree(of_a); cudaFree(of_b); cudaFree(oh);
        cudaFree(in.d_conv_in); cudaFree(in.d_conv_out); cudaFree(in.d_alpha); cudaFree(in.d_beta); cudaFree(in.d_z);
    }

    // split calls: 64 + 64 (state carried) vs 128 at once, plus conv state carry
    {
        printf("split 64+64 vs 128\n");
        const int T = 128;
        Inputs in = make_inputs(T, g);
        size_t no = (size_t)T * Hv * DV;
        float *st_a = up(st0), *st_b = up(st0), *of_a = dalloc(no), *of_b = dalloc(no);
        gdn_seq(seq_args(in, p, st_a, nullptr, of_a, 0, 64), 0);
        gdn_seq(seq_args(in, p, st_a, nullptr, of_a, 64, 64), 0);
        gdn_seq(seq_args(in, p, st_b, nullptr, of_b, 0, 128), 0);
        std::vector<float> cs0((size_t)Cc * 3); for (auto& x : cs0) x = n01(g);
        float *cs_a = up(cs0), *cs_b = up(cs0), *co_a = dalloc((size_t)T * Cc), *co_b = dalloc((size_t)T * Cc);
        conv_seq(in.d_conv_in, p.dconvw, cs_a, co_a, Cc, 64, 0);
        conv_seq(in.d_conv_in + (size_t)64 * Cc, p.dconvw, cs_a, co_a + (size_t)64 * Cc, Cc, 64, 0);
        conv_seq(in.d_conv_in, p.dconvw, cs_b, co_b, Cc, 128, 0);
        CK(cudaDeviceSynchronize());
        check("gdn out_f", relerr(down(of_a, no), down(of_b, no)), 1e-6);
        check("gdn state", relerr(down(st_a, st0.size()), down(st_b, st0.size())), 1e-6);
        check("conv out", relerr(down(co_a, (size_t)T * Cc), down(co_b, (size_t)T * Cc)), 1e-7);
        check("conv state", relerr(down(cs_a, Cc * 3), down(cs_b, Cc * 3)), 1e-7);
    }

    // timing: T=1024, 20 runs after warmup
    {
        const int T = 1024;
        Inputs in = make_inputs(T, g);
        size_t no = (size_t)T * Hv * DV;
        float *st = up(st0), *of = dalloc(no), *cs = dalloc((size_t)Cc * 3), *co = dalloc((size_t)T * Cc);
        __half* oh; CK(cudaMalloc(&oh, no * 2));
        GdnSeqArgs a = seq_args(in, p, st, oh, of, 0, T);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 3; ++i) { gdn_seq(a, 0); conv_seq(in.d_conv_in, p.dconvw, cs, co, Cc, T, 0); }
        CK(cudaDeviceSynchronize());
        auto timeit = [&](auto fn) {
            float best = 1e30f, sum = 0;
            for (int i = 0; i < 20; ++i) {
                cudaEventRecord(e0); fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); sum += ms; best = std::min(best, ms);
            }
            printf("  mean %.2f us/token  best %.2f us/token\n", sum / 20 * 1000 / T, best * 1000 / T);
        };
        printf("timing T=%d (20 runs, 48 heads, one launch)\n gdn_seq:", T);
        timeit([&] { gdn_seq(a, 0); });
        printf(" conv_seq:");
        timeit([&] { conv_seq(in.d_conv_in, p.dconvw, cs, co, Cc, T, 0); });
        CK(cudaDeviceSynchronize());
    }
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
