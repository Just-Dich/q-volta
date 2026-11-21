// Inkling-2-Mini (inkling2_loop) decode-step kernels (CUDA, sm_70). See docs/INKLING_ENGINE.md.
// All activations are fp32 [H]; GEMV inputs are int8-quantized (Q8Buf, canonical chunk permutation of gemv_q8).
#pragma once
#include "../common.cuh"
#include "../kernels/gemv.h"

namespace qv::cuda::ink {

// ---- int4 KV rows (global layers): per token and kv head, 128 channels -> 64 B of nibbles (byte b = channels 2b | 2b+1 << 4),
// plus one (scale, min) fp16 pair per 64-channel group, separately for K and V (K 64+8 B, V 64+8 B = 144 B per head-token).
// Called by all 128 threads of a block (thread c owns channel c). sh: >= 8 floats of shared memory.
__device__ __forceinline__ void kv4_store(float val, int c, uint8_t* qrow, __half2* srow, float* sh) {
    float mn = val, mx = val;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
    const int w = c >> 5;
    __syncthreads();
    if ((c & 31) == 0) { sh[w * 2] = mn; sh[w * 2 + 1] = mx; }
    __syncthreads();
    const int g = c >> 6;
    mn = fminf(sh[g * 4], sh[g * 4 + 2]);
    mx = fmaxf(sh[g * 4 + 1], sh[g * 4 + 3]);
    const __half sh_ = __float2half(fmaxf((mx - mn) * (1.f / 15.f), 1e-5f)), mh = __float2half(mn);
    const float sc = __half2float(sh_), mf = __half2float(mh);
    int q = __float2int_rn((val - mf) / sc);
    q = max(0, min(15, q));
    const int qn = __shfl_xor_sync(0xffffffffu, q, 1);
    if ((c & 1) == 0) qrow[c >> 1] = (uint8_t)(q | (qn << 4));
    if ((c & 63) == 0) srow[g] = __halves2half2(sh_, mh);
}

constexpr int H = 2304, NH = 24, NKV = 8, HD = 128, FF = 1152, NE = 128, TOPK = 8, KW = 4, REL = 16, REXT = 1024;
constexpr int HC = H / 32;  // 72 chunks of 32

// ---- embedding + final -------------------------------------------------------------------------------------------
// h = rms_w(dequant(embed[tok]), w); also copies h to blk0.
void embed_norm(const QMat& E, const int* tok, const float* w, float* h, float* blk0, cudaStream_t s, int T = 1);
void softcap_inplace(float* x, int n, float cap, cudaStream_t s);

// ---- attention residual (+ optional fused sconv/residual of the attention branch, norm and int8 quantization) ----------
struct ArArgs {
    int mode = 0;       // 0: partial = h ; 1: partial = post-attention residual (sconv(araw) [+ h])
    int B = 0;          // number of stored block states before this layer
    int Q = 1;          // number of queries
    int mix = 1;        // 0: out = partial (no attention residual; layer 0)
    int epi = 0;        // 0: write raw mix to raw_out ; 1: rms_w (nw[q]) -> xn_out + q8 (stream q) ; 2: rms_w -> xh_out (fp16)
    int ntok = 1, tok0 = 0, allpar = 0;  // prefill: ntok rows per buffer, allpar: token = blockIdx.x (else tok0)
    __half* xh_out = nullptr;            // epi 2: fp16 normalised rows [q][ntok][H]
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
    int T = 1;                 // prefill: row stride (rows per stream); grid (Tg, S) when T > 1
    int Tg = 0;
    __half* xh_out = nullptr;  // fp16 output instead of xn_out/q8
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
    uint8_t *kq = nullptr, *vq = nullptr; __half2 *ks = nullptr, *vs = nullptr;  // int4 KV (global layers): kq/vq [8][cap][64], ks/vs [8][cap][2]
    int cap;        // slots per kv head in the cache (ring size 1024 for local layers)
    int ring;
};
void attn_prep(const PrepArgs& a, cudaStream_t s);
constexpr int PART_STRIDE = 130;
struct AttnArgs {
    const float* qs; const float* rl; const __half *kc, *vc;
    int cap, ring, window;
    const uint8_t *kq = nullptr, *vq = nullptr; const __half2 *ks = nullptr, *vs = nullptr;
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
void finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, cudaStream_t s, int T = 1);
void conv_commit(const float* rows, long rs, int T, int D, float* st, cudaStream_t s);

}  // namespace qv::cuda::ink
