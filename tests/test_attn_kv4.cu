// int4 KV: (a) accuracy vs the fp32 path (prep_seq + attn_seq / attn_decode), K with outlier channels;
// (b) decode-kernel vs prefill-kernel agreement on the same int4 cache (same window rule), incl. multi-chunk prefill and
// chunk boundaries at every phase of the 64-block; (c) nt-row (MTP verify) launch bit-identical to 1-row launches.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <vector>

#include "../src/backends/cuda/attention/attn.h"
#include "../src/backends/cuda/attention/attn_seq.h"

using namespace qv::cuda;
constexpr int Hq = 24, Hkv = 4, D = 256, RD = 64, MAXCTX = 6144;

template <class T> static T* dev(size_t n) { T* p; QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); QV_CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <class T> static T* up(const std::vector<T>& v) {
    T* p = dev<T>(v.size());
    QV_CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> static std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    QV_CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double)(a[i] - b[i]) * (a[i] - b[i]); r += (double)b[i] * b[i]; }
    return std::sqrt(d / (r + 1e-30));
}
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

struct Data { int N; float *qg, *k, *v, *qn, *kn, *invf; };

static Kv4Layer make_kv4(int kmode) {
    Kv4Layer l;
    l.nblocks = MAXCTX / 64; l.kmode = kmode & 1; l.exact = kmode == 2;  // kmode 2: fp16 KV (exact ring)
    if (l.exact) { l.nblocks = 1; l.ring_n = MAXCTX; }
    l.blocks = dev<uint8_t>((size_t)Hkv * l.nblocks * KV4_REC_BYTES);
    l.ring_k = dev<__half>((size_t)Hkv * l.ring_n * D);
    l.ring_v = dev<__half>((size_t)Hkv * l.ring_n * D);
    return l;
}
static void free_kv4(Kv4Layer& l) { cudaFree(l.blocks); cudaFree(l.ring_k); cudaFree(l.ring_v); }

// prefill [pos0, pos0+T) into cache (fp32 if !kv4) and attend; returns out_f [T][Hq*D]; qr (prepared q) optional
static std::vector<float> seq_run(const Data& d, int pos0, int T, float* kc, float* vc, const Kv4Layer* kv4, float* qr_out,
                                  __half* sk, __half* sv) {
    float* qr = dev<float>((size_t)T * Hq * D);
    float* out = dev<float>((size_t)T * Hq * D);
    AttnSeqPrepArgs p{};
    p.qg = d.qg + (size_t)pos0 * Hq * 2 * D; p.k = d.k + (size_t)pos0 * Hkv * D; p.v = d.v + (size_t)pos0 * Hkv * D;
    p.q_norm = d.qn; p.k_norm = d.kn; p.inv_freq = d.invf; p.q_out = qr; p.k_cache = kc; p.v_cache = vc;
    p.n_heads = Hq; p.n_kv_heads = Hkv; p.head_dim = D; p.rope_dim = RD; p.max_ctx = MAXCTX; p.pos0 = pos0; p.T = T; p.eps = 1e-6f;
    if (kv4) { p.kv4 = *kv4; p.stage_k = sk; p.stage_v = sv; }
    attn_prep_seq(p, 0);
    AttnSeqArgs a{};
    a.q = qr; a.qg = p.qg; a.k_cache = kc; a.v_cache = vc; a.out_f = out; a.out_h = nullptr;
    a.n_heads = Hq; a.n_kv_heads = Hkv; a.max_ctx = MAXCTX; a.pos0 = pos0; a.T = T;
    if (kv4) { a.kv4 = *kv4; a.stage_k = sk; a.stage_v = sv; }
    attn_seq(a, 0);
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    auto o = down(out, (size_t)T * Hq * D);
    if (qr_out) QV_CUDA_CHECK(cudaMemcpy(qr_out, qr, (size_t)T * Hq * D * 4, cudaMemcpyDeviceToDevice));
    cudaFree(qr); cudaFree(out);
    return o;
}

// decode attention of the single query row `row` (prepared q taken from qr_all) at position pos for the given caches
static std::vector<float> decode_run(const Data& d, const float* qr_all, int row_tok, int pos, float* kc, float* vc,
                                     const Kv4Layer* kv4, int nt = 1) {
    int* dpos = up(std::vector<int>{pos, pos + 1, pos + 2, pos + 3});
    float* part = dev<float>((size_t)nt * Hq * ATTN_MAX_SPLITS * (D + 2));
    float* out = dev<float>((size_t)nt * Hq * D);
    AttnDecodeArgs a{};
    a.q = qr_all + (size_t)row_tok * Hq * D; a.qg = d.qg + (size_t)row_tok * Hq * 2 * D;
    a.k_cache = kc; a.v_cache = vc; a.out = out; a.n_heads = Hq; a.n_kv_heads = Hkv; a.max_ctx = MAXCTX; a.pos = dpos;
    a.part = part; a.max_splits = ATTN_MAX_SPLITS;
    if (kv4) a.kv4 = *kv4;
    attn_decode(a, 0);
    QV_CUDA_CHECK(cudaDeviceSynchronize());
    auto o = down(out, (size_t)Hq * D);
    cudaFree(dpos); cudaFree(part); cudaFree(out);
    return o;
}

int main() {
    const char* qa = std::getenv("QV_ATTN");
    const bool fp32k = qa && std::string(qa) == "fp32";  // CUDA-core prefill kernel: tight tolerances; tensor-core kernel (fp16 P): looser
    const double tol = fp32k ? 2e-4 : 6e-3;
    std::printf("prefill kernel: %s\n", fp32k ? "fp32 CUDA-core" : "tensor-core");
    std::mt19937 g(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int N = 4000;
    Data d; d.N = N;
    {
        std::vector<float> qg((size_t)N * Hq * 2 * D), k((size_t)N * Hkv * D), v((size_t)N * Hkv * D);
        for (auto& x : qg) x = nd(g);
        for (size_t i = 0; i < k.size(); ++i) { int c = i % D; k[i] = nd(g) * ((c % 37 == 5) ? 12.f : 1.f) + ((c % 53 == 3) ? 4.f : 0.f); }
        for (auto& x : v) x = nd(g) * 0.5f + 0.1f;
        d.qg = up(qg); d.k = up(k); d.v = up(v);
        std::vector<float> qn(D, 1.f), kn(D, 1.f), inv(RD / 2);
        for (int i = 0; i < RD / 2; ++i) inv[i] = 1.f / std::pow(1e6f, (float)(2 * i) / RD);
        d.qn = up(qn); d.kn = up(kn); d.invf = up(inv);
    }
    float* kcf = dev<float>((size_t)Hkv * MAXCTX * D);
    float* vcf = dev<float>((size_t)Hkv * MAXCTX * D);
    __half* sk = dev<__half>((size_t)4000 * Hkv * D);
    __half* sv = dev<__half>((size_t)4000 * Hkv * D);
    float* qr_all = dev<float>((size_t)N * Hq * D);

    for (int kmode = 0; kmode < 3; ++kmode) {
        std::printf("== %s ==\n", kmode == 2 ? "fp16 KV (exact ring)" : kmode ? "int4, K per-token" : "int4, K per-channel");
        const int Ts[] = {100, 191, 192, 193, 255, 256, 319, 320, 1000, 2047, 2048, 3999};
        for (int T : Ts) {
            // fp32 reference: whole sequence as one chunk
            auto ref = seq_run(d, 0, T, kcf, vcf, nullptr, nullptr, nullptr, nullptr);
            // int4, single chunk
            Kv4Layer l = make_kv4(kmode);
            auto q1 = seq_run(d, 0, T, nullptr, nullptr, &l, qr_all, sk, sv);
            double e_seq = rel(q1, ref);
            // decode kernel at the last position on that cache vs the prefill kernel's last row
            auto dec = decode_run(d, qr_all, T - 1, T - 1, nullptr, nullptr, &l);
            std::vector<float> last(q1.end() - Hq * D, q1.end());
            double e_dec = rel(dec, last);
            // fp decode vs fp prefill reference for the same row (sanity of the harness)
            // multi-chunk int4 (chunks of 193 / 64 / rest) must reproduce the single-chunk output of the last chunk
            Kv4Layer l2 = make_kv4(kmode);
            int pos = 0; std::vector<float> lastc;
            const int sizes[] = {193, 64, 100000};
            for (int s : sizes) {
                int n = std::min(s, T - pos);
                if (n <= 0) break;
                lastc = seq_run(d, pos, n, nullptr, nullptr, &l2, nullptr, sk, sv);
                pos += n;
            }
            std::vector<float> lastc_tail(lastc.end() - Hq * D, lastc.end());
            double e_chunk = rel(lastc_tail, last);
            std::printf("T=%5d  int4 prefill vs fp32: rel L2 %.4f | decode vs prefill(int4): %.2e | multichunk vs single: %.2e\n",
                        T, e_seq, e_dec, e_chunk);
            CHECK(e_dec < tol, "T=%d decode vs prefill kernel mismatch %.3e", T, e_dec);
            CHECK(e_chunk < tol, "T=%d multi-chunk mismatch %.3e", T, e_chunk);
            CHECK(e_seq < (kmode == 2 ? 6e-3 : 0.15), "T=%d accuracy %.3e", T, e_seq);
            // MTP-style verify: nt rows (positions T-1..T+1 would need new tokens) -> use rows at the same ring content:
            // here compare nt=3 launch (same q, positions T-3..T-1) against three 1-row launches (bit-exact required)
            if (T >= 300) {
                int* dpos = up(std::vector<int>{T - 3, T - 2, T - 1});
                float* part = dev<float>((size_t)3 * Hq * ATTN_MAX_SPLITS * (D + 2));
                float* out3 = dev<float>((size_t)3 * Hq * D);
                AttnDecodeArgs a{};
                a.q = qr_all + (size_t)(T - 1) * Hq * D; a.qg = d.qg + (size_t)(T - 1) * Hq * 2 * D; a.out = out3;
                a.n_heads = Hq; a.n_kv_heads = Hkv; a.max_ctx = MAXCTX; a.pos = dpos; a.part = part; a.max_splits = ATTN_MAX_SPLITS;
                a.kv4 = l; a.nt = 3; a.q_stride = 0; a.qg_stride = 0; a.out_stride = Hq * D; a.part_stride = (size_t)Hq * ATTN_MAX_SPLITS * (D + 2);
                attn_decode(a, 0);
                QV_CUDA_CHECK(cudaDeviceSynchronize());
                auto o3 = down(out3, (size_t)3 * Hq * D);
                bool exact = true;
                for (int r = 0; r < 3; ++r) {
                    auto o1 = decode_run(d, qr_all, T - 1, T - 3 + r, nullptr, nullptr, &l);
                    exact &= std::memcmp(o1.data(), o3.data() + (size_t)r * Hq * D, (size_t)Hq * D * 4) == 0;
                }
                CHECK(exact, "T=%d nt=3 verify launch not bit-identical to 1-row launches", T);
                cudaFree(dpos); cudaFree(part); cudaFree(out3);
            }
            free_kv4(l); free_kv4(l2);
        }
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
