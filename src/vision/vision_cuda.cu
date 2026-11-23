// CUDA Inkling vision encoder (V100, sm_70): fp16 weights, fp32 residual stream, cuBLAS tensor-op GEMMs (fp32 accumulate),
// two-pass WMMA flash attention (head_dim 64, bidirectional, per image), small fused kernels. Math identical to vision_cpu.cpp.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vision.h"

using namespace nvcuda;

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("vision CUDA: ") + cudaGetErrorString(e_) + " at " #x); } while (0)
#define VBL(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("vision cuBLAS failure at " #x); } while (0)

namespace qv::vision {
namespace {

constexpr int D = 1024, HD = 64, NH = 16;

__device__ __forceinline__ float gelu_erf(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

__device__ float block_sum(float v, float* sm) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sm[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) r += sm[i];
    return r;
}

// block of 256 threads, 4 elements each; LN with fp32 two-pass statistics
__device__ void ln_row(const float v[4], const float* g, const float* b, float eps, float out[4], float* sm) {
    float s = v[0] + v[1] + v[2] + v[3];
    const float mean = block_sum(s, sm) / D;
    float s2 = 0.f;
    for (int j = 0; j < 4; ++j) { const float d = v[j] - mean; s2 += d * d; }
    const float inv = rsqrtf(block_sum(s2, sm) / D + eps);
    for (int j = 0; j < 4; ++j) { const int i = threadIdx.x + 256 * j; out[j] = (v[j] - mean) * inv * g[i] + b[i]; }
}

__global__ void k_patchify(const float* pix, __half* out, int T, int cols) {
    // out [P][1200]: P = 4T token-major; patch (y,x,c) of sub-patch s of token tk
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)T * 4 * 1200) return;
    const long p = i / 1200;
    const int e = (int)(i % 1200), c = e % 3, x = (e / 3) % 20, y = e / 60;
    const int tk = (int)(p >> 2), s = (int)(p & 3), sr = s >> 1, sc = s & 1;
    out[i] = __float2half(pix[(((long)tk * 40 + sr * 20 + y) * 40 + sc * 20 + x) * 3 + c]);
}

__device__ void axis_taps(int i, int size, int& a, int& b, float& wb) {
    const float src = ((float)i + 0.5f) * 32.f / (float)size - 0.5f;
    const float fl = floorf(src);
    a = min(max((int)fl, 0), 31);
    b = min(max((int)fl + 1, 0), 31);
    wb = fmaxf(0.f, 1.f - fabsf(src - fl - 1.f));
}

// x[0] = class token, x[p+1] += bilinear position embedding; then LN_pre (in place) and LN1 of layer 0 -> xn
__global__ void __launch_bounds__(256) k_embed_ln(float* x, __half* xn, const __half* pos, const float* cls, const float* gp, const float* bp,
                                                   const float* g1, const float* b1, int cols_tok, int rows_p, int cols_p, float eps) {
    __shared__ float sm[8];
    const int r = blockIdx.x;
    float v[4], o[4];
    if (r == 0) {
        for (int j = 0; j < 4; ++j) v[j] = cls[threadIdx.x + 256 * j];
    } else {
        const int p = r - 1, tk = p >> 2, s = p & 3;
        const int row = (tk / cols_tok) * 2 + (s >> 1), col = (tk % cols_tok) * 2 + (s & 1);
        int r0, r1, c0, c1;
        float rw, cw;
        axis_taps(row, rows_p, r0, r1, rw);
        axis_taps(col, cols_p, c0, c1, cw);
        const float a = 1.f - rw, b = rw, c = 1.f - cw, d = cw;
        for (int j = 0; j < 4; ++j) {
            const int i = threadIdx.x + 256 * j;
            v[j] = x[(long)r * D + i] + a * c * __half2float(pos[(r0 * 32 + c0) * D + i]) + a * d * __half2float(pos[(r0 * 32 + c1) * D + i]) +
                   b * c * __half2float(pos[(r1 * 32 + c0) * D + i]) + b * d * __half2float(pos[(r1 * 32 + c1) * D + i]);
        }
    }
    ln_row(v, gp, bp, eps, o, sm);
    for (int j = 0; j < 4; ++j) x[(long)r * D + threadIdx.x + 256 * j] = o[j];
    float o2[4];
    ln_row(o, g1, b1, eps, o2, sm);
    for (int j = 0; j < 4; ++j) xn[(long)r * D + threadIdx.x + 256 * j] = __float2half(o2[j]);
}

// x += lam * (t + bias); optionally xn = LN(x) (fp16)
__global__ void __launch_bounds__(256) k_resid_ln(float* x, const float* t, const float* bias, const float* lam, const float* g, const float* b, __half* xn, float eps,
                                                   float* stat) {
    __shared__ float sm[8];
    const int r = blockIdx.x;
    float v[4], o[4];
    float mx = 0.f;
    for (int j = 0; j < 4; ++j) {
        const int i = threadIdx.x + 256 * j;
        v[j] = x[(long)r * D + i] + lam[i] * (t[(long)r * D + i] + bias[i]);
        x[(long)r * D + i] = v[j];
        mx = fmaxf(mx, fabsf(v[j]));
    }
    if (stat) atomicMax((int*)stat, __float_as_int(mx));
    if (g) {
        ln_row(v, g, b, eps, o, sm);
        for (int j = 0; j < 4; ++j) xn[(long)r * D + threadIdx.x + 256 * j] = __float2half(o[j]);
    }
}

__global__ void k_rope_tab(float* cs, float* sn, int N, int cols_tok, float theta) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    float h = 0, w = 0;
    if (s > 0) {
        const int p = s - 1, tk = p >> 2, sub = p & 3;
        h = (float)((tk / cols_tok) * 2 + (sub >> 1) + 1);
        w = (float)((tk % cols_tok) * 2 + (sub & 1) + 1);
    }
    for (int j = 0; j < 16; ++j) {
        const float inv = powf(theta, -(float)(2 * j) / 32.f);
        float a, c;
        sincosf(w * inv, &a, &c);
        cs[(long)s * HD + 2 * j] = cs[(long)s * HD + 2 * j + 1] = c;
        sn[(long)s * HD + 2 * j] = sn[(long)s * HD + 2 * j + 1] = a;
        sincosf(h * inv, &a, &c);
        cs[(long)s * HD + 32 + 2 * j] = cs[(long)s * HD + 32 + 2 * j + 1] = c;
        sn[(long)s * HD + 32 + 2 * j] = sn[(long)s * HD + 32 + 2 * j + 1] = a;
    }
}

// qkv [N][3D] fp16: add bias, rotate q and k pairs (fp32 math), v bias only. one thread per (row, pair of 1536)
__global__ void k_qkv_bias_rope(__half* qkv, const float* bias, const float* cs, const float* sn, int N) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * 1536) return;
    const int r = (int)(i / 1536), pr = (int)(i % 1536), c = pr * 2;
    __half2* ptr = (__half2*)(qkv + (long)r * 3 * D + c);
    float2 v = __half22float2(*ptr);
    v.x += bias[c]; v.y += bias[c + 1];
    if (c < 2 * D) {
        const int d = c % HD;
        const float co = cs[(long)r * HD + d], si = sn[(long)r * HD + d];
        const float a = v.x * co - v.y * si, b = v.y * co + v.x * si;
        v.x = a; v.y = b;
    }
    *ptr = __floats2half2_rn(v.x, v.y);
}

__global__ void k_bias_gelu(__half* h, const float* bias, long total, int ld) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = __half2float(h[i]);
    if (bias) v += bias[i % ld];
    h[i] = __float2half(gelu_erf(v));
}

__global__ void k_f32_to_f16(const float* a, __half* b, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __float2half(a[i]);
}

// ---- two-pass WMMA flash attention: block = 4 warps x 16 query rows, key tiles of 64 ----
constexpr int QT = 64, KT = 64;
__global__ void __launch_bounds__(128) k_attn(const __half* __restrict__ qkv, __half* __restrict__ out, int N) {
    __shared__ __align__(32) __half sK[KT * HD];
    __shared__ __align__(32) __half sV[KT * HD];
    __shared__ __align__(32) float sS[4][16 * KT];
    __shared__ __align__(32) __half sP[4][16 * KT];
    const int h = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q0 = blockIdx.x * QT + w * 16;
    const long ld = 3 * D;
    const __half* Q = qkv + (long)q0 * ld + h * HD;
    const int ntile = (N + KT - 1) / KT;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[4];
    for (int k = 0; k < 4; ++k) wmma::load_matrix_sync(qf[k], Q + k * 16, (unsigned)ld);
    const int row = lane >> 1, hf = lane & 1;
    float m = -INFINITY, l = 0.f;
    for (int pass = 0; pass < 2; ++pass) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[4];
        if (pass == 1) for (int j = 0; j < 4; ++j) wmma::fill_fragment(of[j], 0.f);
        for (int t = 0; t < ntile; ++t) {
            __syncthreads();
            for (int e = threadIdx.x; e < KT * HD / 8; e += 128) {
                const int kr = e >> 3, c8 = e & 7;
                const long src = (long)(t * KT + kr) * ld + h * HD + c8 * 8;
                *(uint4*)(sK + kr * HD + c8 * 8) = *(const uint4*)(qkv + src + D);
                if (pass == 1) *(uint4*)(sV + kr * HD + c8 * 8) = *(const uint4*)(qkv + src + 2 * D);
            }
            __syncthreads();
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
                wmma::fill_fragment(sf, 0.f);
                for (int k = 0; k < 4; ++k) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
                    wmma::load_matrix_sync(kf, sK + n * 16 * HD + k * 16, HD);
                    wmma::mma_sync(sf, qf[k], kf, sf);
                }
                wmma::store_matrix_sync(&sS[w][n * 16], sf, KT, wmma::mem_row_major);
            }
            __syncwarp();
            const int kvalid = min(KT, N - t * KT);
            float* sr = &sS[w][row * KT + hf * 32];
            if (pass == 0) {
                float mx = -INFINITY;
                for (int j = 0; j < 32; ++j) { const int kc = hf * 32 + j; mx = fmaxf(mx, kc < kvalid ? sr[j] * 0.125f : -INFINITY); }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                const float mn = fmaxf(m, mx);
                float s = 0.f;
                for (int j = 0; j < 32; ++j) { const int kc = hf * 32 + j; if (kc < kvalid) s += __expf(sr[j] * 0.125f - mn); }
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                l = l * (m == -INFINITY ? 0.f : __expf(m - mn)) + s;
                m = mn;
            } else {
                __half* pr = &sP[w][row * KT + hf * 32];
                for (int j = 0; j < 32; ++j) { const int kc = hf * 32 + j; pr[j] = __float2half(kc < kvalid ? __expf(sr[j] * 0.125f - m) : 0.f); }
                __syncwarp();
                for (int k = 0; k < 4; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
                    wmma::load_matrix_sync(pf, &sP[w][k * 16], KT);
                    for (int j = 0; j < 4; ++j) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
                        wmma::load_matrix_sync(vf, sV + k * 16 * HD + j * 16, HD);
                        wmma::mma_sync(of[j], pf, vf, of[j]);
                    }
                }
            }
            __syncwarp();
        }
        if (pass == 1) {
            for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(&sS[w][j * 16], of[j], KT, wmma::mem_row_major);
            __syncwarp();
            const float inv = 1.f / l;
            const int qr = q0 + row;
            if (qr < N)
                for (int j = 0; j < 32; ++j) out[(long)qr * D + h * HD + hf * 32 + j] = __float2half(sS[w][row * KT + hf * 32 + j] * inv);
        }
    }
}

struct DLayer {
    __half *qkv, *out, *fc1, *fc2;
    float *qkv_b, *out_b, *fc1_b, *fc2_b, *ln1g, *ln1b, *ln2g, *ln2b, *lam1, *lam2;
};

class CudaEncoder : public Encoder {
   public:
    explicit CudaEncoder(std::shared_ptr<VisionFile> f) {
        cfg_ = f->cfg;
        if (cfg_.hidden != D || cfg_.heads != NH || cfg_.merge != 2 || cfg_.patch != 20) throw std::runtime_error("CUDA vision encoder is specialised for the Inkling-2-Mini tower");
        VBL(cublasCreate(&bl_));
        VBL(cublasSetMathMode(bl_, CUBLAS_TENSOR_OP_MATH));
        VCK(cudaStreamCreate(&st_));
        VBL(cublasSetStream(bl_, st_));
        VCK(cudaEventCreate(&e0_)); VCK(cudaEventCreate(&e1_));
        auto P = [&](const std::string& n) { return "model.visual." + n; };
        patch_ = up16(f->half(P("backbone.embeddings.patch_embedding.weight")));
        pos_ = up16(f->half(P("backbone.embeddings.position_embedding")));
        cls_ = upf([&] { auto a = f->floats(P("backbone.embeddings.class_embedding")), b = f->floats(P("backbone.embeddings.cls_position_embedding")); for (size_t i = 0; i < a.size(); ++i) a[i] += b[i]; return a; }());
        preg_ = upf(f->floats(P("backbone.ln_pre.weight"))); preb_ = upf(f->floats(P("backbone.ln_pre.bias")));
        L_.resize(cfg_.layers);
        for (int l = 0; l < cfg_.layers; ++l) {
            DLayer& d = L_[l];
            const std::string b = "backbone.layers." + std::to_string(l) + ".";
            std::vector<uint16_t> w;
            std::vector<float> bb;
            for (const char* nm : {"q_proj", "k_proj", "v_proj"}) {
                auto a = f->half(P(b + "self_attn." + nm + ".weight")); w.insert(w.end(), a.begin(), a.end());
                auto c = f->floats(P(b + "self_attn." + nm + ".bias")); bb.insert(bb.end(), c.begin(), c.end());
            }
            d.qkv = up16(w); d.qkv_b = upf(bb);
            d.out = up16(f->half(P(b + "self_attn.out_proj.weight"))); d.out_b = upf(f->floats(P(b + "self_attn.out_proj.bias")));
            d.fc1 = up16(f->half(P(b + "mlp.c_fc.weight"))); d.fc1_b = upf(f->floats(P(b + "mlp.c_fc.bias")));
            d.fc2 = up16(f->half(P(b + "mlp.c_proj.weight"))); d.fc2_b = upf(f->floats(P(b + "mlp.c_proj.bias")));
            d.ln1g = upf(f->floats(P(b + "ln_1.weight"))); d.ln1b = upf(f->floats(P(b + "ln_1.bias")));
            d.ln2g = upf(f->floats(P(b + "ln_2.weight"))); d.ln2b = upf(f->floats(P(b + "ln_2.bias")));
            d.lam1 = upf(f->floats(P(b + "ls_1.gamma"))); d.lam2 = upf(f->floats(P(b + "ls_2.gamma")));
        }
        m1_ = up16(f->half(P("proj_in.weight")));
        m2_ = up16(f->half(P("proj_out.weight")));
        VCK(cudaMalloc(&stat_, 64));
        VCK(cudaStreamSynchronize(st_));
    }
    ~CudaEncoder() override {
        for (void* p : allocs_) cudaFree(p);
        cublasDestroy(bl_);
        cudaStreamDestroy(st_);
    }
    const char* backend() const override { return "cuda"; }

    void encode(const Pixels& pix, float* out) override {
        const int T = pix.tokens(), P = 4 * T, N = P + 1, Np = (N + 63) / 64 * 64, tl = 40;
        const bool check = std::getenv("QV_VIS_CHECK") != nullptr;
        std::vector<void*> ws;
        auto W = [&](size_t bytes) { void* p; VCK(cudaMalloc(&p, bytes)); ws.push_back(p); return p; };
        VCK(cudaEventRecord(e0_, st_));
        float* d_pix = (float*)W((size_t)T * tl * tl * 3 * 4);
        __half* d_pv = (__half*)W((size_t)P * 1200 * 2);
        float* x = (float*)W((size_t)Np * D * 4);
        float* tmp = (float*)W((size_t)Np * D * 4);
        __half* xn = (__half*)W((size_t)Np * D * 2);
        __half* qkv = (__half*)W((size_t)Np * 3 * D * 2);
        __half* att = (__half*)W((size_t)Np * D * 2);
        __half* hid = (__half*)W((size_t)Np * 4096 * 2);
        float *cs = (float*)W((size_t)N * HD * 4), *sn = (float*)W((size_t)N * HD * 4);
        __half* mi = (__half*)W((size_t)T * 4096 * 2);
        __half* mh = (__half*)W((size_t)T * 4096 * 2);
        float* d_out = (float*)W((size_t)T * cfg_.text_hidden * 4);
        VCK(cudaMemsetAsync(qkv, 0, (size_t)Np * 3 * D * 2, st_));
        VCK(cudaMemsetAsync(att, 0, (size_t)Np * D * 2, st_));
        VCK(cudaMemsetAsync(stat_, 0, 64, st_));
        VCK(cudaMemcpyAsync(d_pix, pix.data.data(), (size_t)T * tl * tl * 3 * 4, cudaMemcpyHostToDevice, st_));
        k_patchify<<<(unsigned)(((long)P * 1200 + 255) / 256), 256, 0, st_>>>(d_pix, d_pv, T, pix.cols);
        gemm(d_pv, P, 1200, patch_, D, x + D, D, true);
        k_embed_ln<<<N, 256, 0, st_>>>(x, xn, pos_, cls_, preg_, preb_, L_[0].ln1g, L_[0].ln1b, pix.cols, pix.rows * 2, pix.cols * 2, cfg_.eps);
        k_rope_tab<<<(N + 127) / 128, 128, 0, st_>>>(cs, sn, N, pix.cols, cfg_.rope_theta);
        for (int l = 0; l < cfg_.layers; ++l) {
            const DLayer& d = L_[l];
            gemm(xn, N, D, d.qkv, 3 * D, qkv, 3 * D, false, true);
            k_qkv_bias_rope<<<(unsigned)(((long)N * 1536 + 255) / 256), 256, 0, st_>>>(qkv, d.qkv_b, cs, sn, N);
            k_attn<<<dim3(Np / QT, NH), 128, 0, st_>>>(qkv, att, N);
            gemm(att, N, D, d.out, D, tmp, D, true);
            k_resid_ln<<<N, 256, 0, st_>>>(x, tmp, d.out_b, d.lam1, d.ln2g, d.ln2b, xn, cfg_.eps, check ? stat_ : nullptr);
            gemm(xn, N, D, d.fc1, 4096, hid, 4096, false, true);
            k_bias_gelu<<<(unsigned)(((long)N * 4096 + 255) / 256), 256, 0, st_>>>(hid, d.fc1_b, (long)N * 4096, 4096);
            gemm(hid, N, 4096, d.fc2, D, tmp, D, true);
            const bool last = l + 1 == cfg_.layers;
            k_resid_ln<<<N, 256, 0, st_>>>(x, tmp, d.fc2_b, d.lam2, last ? nullptr : L_[l + 1].ln1g, last ? nullptr : L_[l + 1].ln1b, xn, cfg_.eps, check ? stat_ + 1 : nullptr);
        }
        k_f32_to_f16<<<(unsigned)(((long)T * 4096 + 255) / 256), 256, 0, st_>>>(x + D, mi, (long)T * 4096);
        gemm(mi, T, 4096, m1_, 4096, mh, 4096, false, true);
        k_bias_gelu<<<(unsigned)(((long)T * 4096 + 255) / 256), 256, 0, st_>>>(mh, nullptr, (long)T * 4096, 4096);
        gemm(mh, T, 4096, m2_, cfg_.text_hidden, d_out, cfg_.text_hidden, true);
        VCK(cudaMemcpyAsync(out, d_out, (size_t)T * cfg_.text_hidden * 4, cudaMemcpyDeviceToHost, st_));
        float hs[2] = {0, 0};
        if (check) VCK(cudaMemcpyAsync(hs, stat_, 8, cudaMemcpyDeviceToHost, st_));
        VCK(cudaEventRecord(e1_, st_));
        VCK(cudaStreamSynchronize(st_));
        VCK(cudaGetLastError());
        float ms = 0;
        VCK(cudaEventElapsedTime(&ms, e0_, e1_));
        last_ms = ms;
        if (check) std::fprintf(stderr, "[vision] max |residual| after attn/mlp residual adds: %.1f / %.1f (fp16 limit 65504)\n", hs[0], hs[1]);
        for (void* p : ws) cudaFree(p);
    }

   private:
    // C[M][N] = A[M][K] * W[N][K]^T ; C fp32 if c32 else fp16
    void gemm(const __half* A, int M, int K, const __half* Wt, int N, void* C, int ldc, bool c32, bool = false) {
        const float al = 1.f, be = 0.f;
        VBL(cublasGemmEx(bl_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, Wt, CUDA_R_16F, K, A, CUDA_R_16F, K, &be, C, c32 ? CUDA_R_32F : CUDA_R_16F, ldc,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    __half* up16(const std::vector<uint16_t>& h) { void* p; VCK(cudaMalloc(&p, h.size() * 2)); allocs_.push_back(p); VCK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); return (__half*)p; }
    float* upf(const std::vector<float>& h) { void* p; VCK(cudaMalloc(&p, h.size() * 4)); allocs_.push_back(p); VCK(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return (float*)p; }

    cublasHandle_t bl_{};
    cudaStream_t st_{};
    cudaEvent_t e0_{}, e1_{};
    std::vector<void*> allocs_;
    std::vector<DLayer> L_;
    __half *patch_ = nullptr, *pos_ = nullptr, *m1_ = nullptr, *m2_ = nullptr;
    float *cls_ = nullptr, *preg_ = nullptr, *preb_ = nullptr, *stat_ = nullptr;
};

}  // namespace

std::unique_ptr<Encoder> make_cuda_encoder(std::shared_ptr<VisionFile> f) { return std::make_unique<CudaEncoder>(f); }

}  // namespace qv::vision
