// Inkling-2-Mini (inkling2_loop) decode-step kernels (CUDA, sm_70). See docs/INKLING_ENGINE.md.
// All activations are fp32 [H]; GEMV inputs are int8-quantized (Q8Buf, canonical chunk permutation of gemv_q8).
#pragma once
#include "../common.cuh"
#include "../kernels/gemv.h"

namespace qv::cuda::ink {

constexpr int H = 2304, NH = 24, NKV = 8, HD = 128, FF = 1152, NE = 128, TOPK = 8, KW = 4, REL = 16, REXT = 1024;
constexpr int HC = H / 32;  // 72 chunks of 32

// ---- embedding + final -------------------------------------------------------------------------------------------
// h = rms_w(dequant(embed[tok]), w); also copies h to blk0.
void embed_norm(const QMat& E, const int* tok, const float* w, float* h, float* blk0, cudaStream_t s);
void softcap_inplace(float* x, int n, float cap, cudaStream_t s);

// ---- attention residual (+ optional fused sconv/residual of the attention branch, norm and int8 quantization) ----------
struct ArArgs {
    int mode = 0;       // 0: partial = h ; 1: partial = post-attention residual (sconv(araw) [+ h])
    int B = 0;          // number of stored block states before this layer
    int Q = 1;          // number of queries
    int mix = 1;        // 0: out = partial (no attention residual; layer 0)
    int epi = 0;        // 0: write raw mix to raw_out ; 1: rms_w (nw[q]) -> xn_out + q8 (stream q)
    int boundary = 0;   // mode 1: layer % 6 == 0: residual = a (no h), and h is appended to blocks[B]
    const float* h = nullptr;
    const float* araw = nullptr;
    const float* cw = nullptr;  // [H][4] attn_sconv weight
    float* cst = nullptr;       // [3][H] state
    float* res_out = nullptr;
    float* blocks = nullptr;    // [4][H]
    const float* queries = nullptr;
    long qbs = 0, rbs = 0;      // per-blockIdx.x strides (messages: one block per loop layer)
    float* raw_out = nullptr;
    const float* nw[4] = {nullptr, nullptr, nullptr, nullptr};
    float* xn_out = nullptr;
    Q8Buf q8;                   // stream q at q + q*H/4, d/sum/qsum + q*HC
};
void attn_res(const ArArgs& a, int nblocks, cudaStream_t s);

// out stream s = rms_w(in[s], nw[s]) -> xn_out + s*H, q8 stream s
struct NormArgs {
    int S = 1;
    const float* in[4] = {};
    const float* nw[4] = {};
    float* xn_out = nullptr;
    Q8Buf q8;
};
void norm_q8(const NormArgs& a, cudaStream_t s);

// ---- loop message gates ------------------------------------------------------------------------------------------
struct GateArgs {
    int S = 1;
    const float* h[4] = {};
    const float* msg[4] = {};
    const __half* down[4] = {};
    const __half* forget[4] = {};
    const __half* gain[4] = {};
    float* g = nullptr;      // [S][64]
    float* out[4] = {};
};
void gate(const GateArgs& a, cudaStream_t s);

// ---- multi-matrix GEMV (q8 quantized matrices and fp16 matrices with fp32 x) ---------------------------------------
struct GD {
    int kind = 0;  // 0: quantized (int8 x), 1: fp16 (fp32 x)
    int N = 0, K = 0, gshift = 6;
    const uint32_t* q = nullptr;
    const uint16_t* sm8 = nullptr;
    const __half* rs = nullptr;
    const __half* w16 = nullptr;
    float zlo = 0, zstep = 0;
    Q8Vec xq{};
    const float* xf = nullptr;
    float* y = nullptr;
    int nblk = 0;
};
struct GDs {
    int n = 0;
    GD d[8];
};
void mgemv(GDs& g, int bits, cudaStream_t s);  // fills nblk, launches; all kind-0 descriptors must have `bits` bits

// ---- attention ------------------------------------------------------------------------------------------------------
struct PrepArgs {
    const float *qraw, *kraw, *vraw, *relraw, *dwk, *dwv;   // GEMV outputs (dwk/dwv: global layers only)
    const float *qscale, *relscale, *relproj;               // [24,128], [24,16], [16,1024]
    const float *kw, *vw;                                   // local: [1024][4]; global: static_w [4][1024]
    float *kst, *vst;                                       // conv state [3][1024]
    int glob;
    const int* pos;
    float* qs;      // [24][128] out
    float* rl;      // [24][1024] out (relative bias by distance)
    __half *kc, *vc;
    int cap;        // slots per kv head in the cache (ring size 1024 for local layers)
    int ring;
};
void attn_prep(const PrepArgs& a, cudaStream_t s);
constexpr int PART_STRIDE = 130;
struct AttnArgs {
    const float* qs; const float* rl; const __half *kc, *vc;
    int cap, ring, window;
    const int* pos;
    float* part;     // [8*nsplit][3][130]
    int nsplit;
    const float* sink; const float* gate; Q8Buf out;   // combine: sink [24], gate [3072] -> q8 of sigmoid(gate)*o
};
void attn_part(const AttnArgs& a, cudaStream_t s);
void attn_combine(const AttnArgs& a, cudaStream_t s);

// ---- MoE --------------------------------------------------------------------------------------------------------------
// router: logits = W x (fp16 [128][H], fp32 x), top-8 on sigmoid(l)+bias, weights = sigmoid(l)/sum*route_scale. ticket: one int, zero at start.
void router(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts, int* ticket, float route_scale,
            cudaStream_t s);
struct ExpertW {  // stacked [E][N][K] bit-plane tensor
    const uint32_t* q; const uint16_t* sm8; const __half* rs;
    int N, K, bits, gshift;
    float zlo, zstep;
};
// act = softcap(g,4) sigmoid(g) softcap(u,25) for interleaved rows; output int8-quantized per slot (288 chunks total)
void moe_w13(const ExpertW& W, const Q8Vec& x, const int* ids, Q8Buf act, float gcap, float ucap, cudaStream_t s);
void moe_w2(const ExpertW& W, const Q8Vec& act, const int* ids, const float* wts, float* y, cudaStream_t s);
// h_out = res + sconv(y)
void finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, cudaStream_t s);

}  // namespace qv::cuda::ink
