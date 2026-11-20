// Batched prefill attention (gated GQA). Flash-style: a block owns TQ consecutive queries x the 6 q heads
// of one kv head (R = 48 rows), streams KT-key tiles of K and V through shared memory once for all rows,
// fp32 online softmax, CUDA-core FMA.
#include "attn_seq.h"
#include <cstdlib>
#include <string>

namespace qv::cuda {

// ---- single place that knows the KV cache layout [Hkv][max_ctx][D] ----
__device__ __forceinline__ size_t kv_off(int max_ctx, int D, int kvh, int pos) {
    return ((size_t)kvh * max_ctx + pos) * D;
}

// grid (T, Hq+Hkv), D threads. Same op order as qk_prep_kernel in attn.cu.
template <int D>
__global__ void __launch_bounds__(D) qk_prep_seq_kernel(AttnSeqPrepArgs a) {
    __shared__ float red[32];
    __shared__ float buf[D];
    const int tok = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const bool is_q = b < a.n_heads;
    const int h = is_q ? b : b - a.n_heads;
    float x = is_q ? a.qg[(size_t)tok * a.n_heads * 2 * D + h * 2 * D + t]
                   : a.k[(size_t)tok * a.n_kv_heads * D + h * D + t];
    const float* w = is_q ? a.q_norm : a.k_norm;
    float ss = block_sum(x * x, red);
    float y = x * rsqrtf(ss / D + a.eps) * w[t];
    buf[t] = y;
    __syncthreads();
    const int pos = a.pos0 + tok;
    const int half = a.rope_dim / 2;
    if (t < a.rope_dim) {
        int i = t < half ? t : t - half;
        float f = (float)pos * a.inv_freq[i];
        float c = cosf(f), s = sinf(f);
        y = t < half ? y * c - buf[t + half] * s : y * c + buf[t - half] * s;
    }
    if (is_q) {
        a.q_out[(size_t)tok * a.n_heads * D + h * D + t] = y;
    } else {
        if (a.kv4.blocks) {
            const size_t so = ((size_t)tok * a.n_kv_heads + h) * D + t;
            a.stage_k[so] = __float2half(fminf(fmaxf(y, -65504.f), 65504.f));
            a.stage_v[so] = __float2half(fminf(fmaxf(a.v[(size_t)tok * a.n_kv_heads * D + h * D + t], -65504.f), 65504.f));
        } else {
            size_t off = kv_off(a.max_ctx, D, h, pos) + t;
            a.k_cache[off] = y;
            a.v_cache[off] = a.v[(size_t)tok * a.n_kv_heads * D + h * D + t];
        }
    }
}

void attn_prep_seq(const AttnSeqPrepArgs& a, cudaStream_t s) {
    if (a.head_dim != 256) { std::fprintf(stderr, "attn_seq: head_dim 256 only\n"); std::abort(); }
    if (a.T <= 0) return;
    qk_prep_seq_kernel<256><<<dim3(a.T, a.n_heads + a.n_kv_heads), 256, 0, s>>>(a);
    if (a.kv4.blocks) kv4_quant_seq(a.kv4, a.n_kv_heads, a.stage_k, a.stage_v, a.pos0, a.T, s);
}

constexpr int AD = 256;         // head_dim
constexpr int AG = 6;           // q heads per kv head
constexpr int ATQ = 8;          // queries per block
constexpr int AR = ATQ * AG;    // 48 rows
constexpr int AKT = 32;         // keys per tile
constexpr int AKS = AD + 4;     // padded smem row stride for K/V tile (conflict-free float4 across keys)
constexpr size_t ASMEM = sizeof(float) * ((size_t)AR * AD + (size_t)AKT * AKS + (size_t)AR * AKT + 2 * AR);

__global__ void __launch_bounds__(256, 1) attn_seq_kernel(AttnSeqArgs a) {
    extern __shared__ __align__(16) float sm[];
    float* Qs = sm;                       // [AR][AD]
    float* KV = Qs + AR * AD;             // [AKT][AKS]
    float* P = KV + AKT * AKS;            // [AR][AKT]
    float* alpha_s = P + AR * AKT;        // [AR]
    float* l_s = alpha_s + AR;            // [AR]

    const int tid = threadIdx.x;
    const int kvh = blockIdx.y;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * ATQ;  // heaviest (latest) tiles first
    const int Hq = a.n_heads;
    const int qi_last = min(q0 + ATQ - 1, a.T - 1);
    const int n = a.pos0 + qi_last + 1;  // keys needed by this block

    for (int idx = tid; idx < AR * (AD / 4); idx += 256) {
        int r = idx / (AD / 4), c4 = idx % (AD / 4);
        int qi = q0 + r / AG, hq = kvh * AG + r % AG;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qi < a.T) v = *reinterpret_cast<const float4*>(a.q + (size_t)qi * Hq * AD + hq * AD + c4 * 4);
        *reinterpret_cast<float4*>(Qs + r * AD + c4 * 4) = v;
    }

    // score-phase mapping: 16 row groups (3 rows) x 16 key groups (2 keys)
    const int rg = tid >> 4, kg = tid & 15;
    // softmax-phase mapping (tid < 192): 4 threads per row
    const int srow = tid >> 2, ssub = tid & 3;
    int lim_s[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) lim_s[i] = a.pos0 + min(q0 + (rg * 3 + i) / AG, a.T - 1);
    float m_run = -INFINITY, l_run = 0.f;
    // PV-phase mapping: 64 column groups (float4) x 4 row groups (12 rows)
    const int cg = tid & 63, rp = tid >> 6;
    float acc[12][4];
#pragma unroll
    for (int i = 0; i < 12; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    for (int k0 = 0; k0 < n; k0 += AKT) {
        __syncthreads();  // previous PV done reading KV; Qs stores visible on first iteration
        for (int idx = tid; idx < AKT * (AD / 4); idx += 256) {
            int j = idx / (AD / 4), c4 = idx % (AD / 4), key = k0 + j;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (key < n) v = *reinterpret_cast<const float4*>(a.k_cache + kv_off(a.max_ctx, AD, kvh, key) + c4 * 4);
            *reinterpret_cast<float4*>(KV + j * AKS + c4 * 4) = v;
        }
        __syncthreads();
        {
            float s[3][2] = {};
#pragma unroll 4
            for (int d4 = 0; d4 < AD / 4; ++d4) {
                float4 qv[3], kv[2];
#pragma unroll
                for (int i = 0; i < 3; ++i) qv[i] = *reinterpret_cast<const float4*>(Qs + (rg * 3 + i) * AD + d4 * 4);
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) kv[jj] = *reinterpret_cast<const float4*>(KV + (kg + 16 * jj) * AKS + d4 * 4);
#pragma unroll
                for (int i = 0; i < 3; ++i)
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        s[i][jj] = fmaf(qv[i].x, kv[jj].x, s[i][jj]);
                        s[i][jj] = fmaf(qv[i].y, kv[jj].y, s[i][jj]);
                        s[i][jj] = fmaf(qv[i].z, kv[jj].z, s[i][jj]);
                        s[i][jj] = fmaf(qv[i].w, kv[jj].w, s[i][jj]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    int kk = kg + 16 * jj;
                    P[(rg * 3 + i) * AKT + kk] = (k0 + kk <= lim_s[i]) ? s[i][jj] * 0.0625f : -INFINITY;
                }
        }
        __syncthreads();  // scores complete; KV free
        if (tid < 192) {
            float sv[8];
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) { sv[u] = P[srow * AKT + ssub + 4 * u]; mx = fmaxf(mx, sv[u]); }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            float mnew = fmaxf(m_run, mx);
            float alpha = 1.f, sum = 0.f;
            if (mnew == -INFINITY) {
#pragma unroll
                for (int u = 0; u < 8; ++u) P[srow * AKT + ssub + 4 * u] = 0.f;
            } else {
                alpha = expf(m_run - mnew);  // m_run=-inf -> 0
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float p = expf(sv[u] - mnew);
                    P[srow * AKT + ssub + 4 * u] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l_run = l_run * alpha + sum;
            m_run = mnew;
            if (ssub == 0) { alpha_s[srow] = alpha; l_s[srow] = l_run; }
        }
        // V tile into KV (K reads finished before the barrier above)
        for (int idx = tid; idx < AKT * (AD / 4); idx += 256) {
            int j = idx / (AD / 4), c4 = idx % (AD / 4), key = k0 + j;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (key < n) v = *reinterpret_cast<const float4*>(a.v_cache + kv_off(a.max_ctx, AD, kvh, key) + c4 * 4);
            *reinterpret_cast<float4*>(KV + j * AKS + c4 * 4) = v;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 12; ++i) {
            float al = alpha_s[rp * 12 + i];
            acc[i][0] *= al; acc[i][1] *= al; acc[i][2] *= al; acc[i][3] *= al;
        }
#pragma unroll
        for (int j4 = 0; j4 < AKT / 4; ++j4) {
            float4 vv[4];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) vv[jj] = *reinterpret_cast<const float4*>(KV + (j4 * 4 + jj) * AKS + cg * 4);
#pragma unroll
            for (int i = 0; i < 12; ++i) {
                float4 p = *reinterpret_cast<const float4*>(P + (rp * 12 + i) * AKT + j4 * 4);
                float pp[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    acc[i][0] = fmaf(pp[jj], vv[jj].x, acc[i][0]);
                    acc[i][1] = fmaf(pp[jj], vv[jj].y, acc[i][1]);
                    acc[i][2] = fmaf(pp[jj], vv[jj].z, acc[i][2]);
                    acc[i][3] = fmaf(pp[jj], vv[jj].w, acc[i][3]);
                }
            }
        }
    }
    __syncthreads();  // l_s final
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        int r = rp * 12 + i, qi = q0 + r / AG, hq = kvh * AG + r % AG;
        if (qi >= a.T) continue;
        float inv = 1.f / l_s[r];
        float4 g = *reinterpret_cast<const float4*>(a.qg + (size_t)qi * Hq * 2 * AD + hq * 2 * AD + AD + cg * 4);
        float o[4];
        o[0] = acc[i][0] * inv * (1.f / (1.f + expf(-g.x)));
        o[1] = acc[i][1] * inv * (1.f / (1.f + expf(-g.y)));
        o[2] = acc[i][2] * inv * (1.f / (1.f + expf(-g.z)));
        o[3] = acc[i][3] * inv * (1.f / (1.f + expf(-g.w)));
        size_t off = (size_t)qi * Hq * AD + hq * AD + cg * 4;
        if (a.out_f) *reinterpret_cast<float4*>(a.out_f + off) = make_float4(o[0], o[1], o[2], o[3]);
        if (a.out_h) {
            __half2 h01 = __floats2half2_rn(fminf(fmaxf(o[0], -65504.f), 65504.f), fminf(fmaxf(o[1], -65504.f), 65504.f));
            __half2 h23 = __floats2half2_rn(fminf(fmaxf(o[2], -65504.f), 65504.f), fminf(fmaxf(o[3], -65504.f), 65504.f));
            *reinterpret_cast<__half2*>(a.out_h + off) = h01;
            *reinterpret_cast<__half2*>(a.out_h + off + 2) = h23;
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------
// int4 KV variant (kv4.h). Key j is read quantized for a query whose context length n_q has j < kv4_q(n_q), else from the
// fp16 window (chunk staging for j >= pos0, ring otherwise). A 32-key tile lies inside one 64-key block, so per row the
// variant is uniform over the tile; a tile is "mixed" only when the 8 queries of the block straddle a window boundary,
// in which case both variants are loaded and run with per-row masks.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float2 h2f_(uint32_t u) {
    __half2 h = *reinterpret_cast<__half2*>(&u);
    return make_float2(__low2float(h), __high2float(h));
}

__device__ __forceinline__ float4 load_fp16_row4(const AttnSeqArgs& a, int kvh, int key, int c4, bool isK) {
    const __half* base;
    if (key >= a.pos0) base = (isK ? a.stage_k : a.stage_v) + ((size_t)(key - a.pos0) * a.n_kv_heads + kvh) * AD;
    else base = (isK ? a.kv4.ring_k : a.kv4.ring_v) + ((size_t)kvh * a.kv4.ring_n + kv4_slot(a.kv4, key)) * AD;
    uint2 u = *reinterpret_cast<const uint2*>(base + c4 * 4);
    float2 x = h2f_(u.x), y = h2f_(u.y);
    return make_float4(x.x, x.y, y.x, y.y);
}

template <int KM>
__device__ __forceinline__ void load_k_tile(const AttnSeqArgs& a, float* KV, int kvh, int k0, int n, bool quant, int tid) {
    for (int idx = tid; idx < AKT * (AD / 4); idx += 256) {
        const int j = idx / (AD / 4), c4 = idx % (AD / 4), key = k0 + j;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (key < n) {
            if (quant) {
                const uint8_t* rec = kv4_rec(a.kv4, kvh, key >> 6);
                const int t = key & 63;
                const uint32_t w = *reinterpret_cast<const uint32_t*>(rec + KV4_REC_K + t * 128 + (c4 & 31) * 4) >> ((c4 >> 5) * 16);
                float sc[4], zr[4];
                if (KM == 0) {
                    uint4 u = reinterpret_cast<const uint4*>(rec + KV4_REC_KP)[c4];
                    float2 p0 = h2f_(u.x), p1 = h2f_(u.y), p2 = h2f_(u.z), p3 = h2f_(u.w);
                    sc[0] = p0.x; zr[0] = p0.y; sc[1] = p1.x; zr[1] = p1.y; sc[2] = p2.x; zr[2] = p2.y; sc[3] = p3.x; zr[3] = p3.y;
                } else {
                    float2 p = h2f_(reinterpret_cast<const uint32_t*>(rec + KV4_REC_KP)[t * 4 + (c4 >> 4)]);
                    sc[0] = sc[1] = sc[2] = sc[3] = p.x; zr[0] = zr[1] = zr[2] = zr[3] = p.y;
                }
                v.x = fmaf((float)(w & 15u), sc[0], zr[0]);
                v.y = fmaf((float)((w >> 4) & 15u), sc[1], zr[1]);
                v.z = fmaf((float)((w >> 8) & 15u), sc[2], zr[2]);
                v.w = fmaf((float)((w >> 12) & 15u), sc[3], zr[3]);
            } else {
                v = load_fp16_row4(a, kvh, key, c4, true);
            }
        }
        *reinterpret_cast<float4*>(KV + j * AKS + c4 * 4) = v;
    }
}

__device__ __forceinline__ void load_v_tile(const AttnSeqArgs& a, float* KV, int kvh, int k0, int n, bool quant, int tid) {
    if (quant) {  // thread = (byte column b, half of the tile); channels b (low nibble) and b+128 (high nibble)
        const int b = tid & 127, hh = tid >> 7, key0 = k0 + 16 * hh;
        const uint8_t* rec = kv4_rec(a.kv4, kvh, key0 >> 6);
        const int t0 = key0 & 63;
        const uint4 u = *reinterpret_cast<const uint4*>(rec + KV4_REC_V + b * 64 + t0);
        const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
        const uint32_t* par = reinterpret_cast<const uint32_t*>(rec + KV4_REC_VP);
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const uint32_t byte = (uw[e >> 2] >> (8 * (e & 3))) & 255u;
            float2 p0 = h2f_(par[(t0 + e) * 4 + (b >> 6)]), p1 = h2f_(par[(t0 + e) * 4 + 2 + (b >> 6)]);
            const bool ok = key0 + e < n;
            KV[(16 * hh + e) * AKS + b] = ok ? fmaf((float)(byte & 15u), p0.x, p0.y) : 0.f;
            KV[(16 * hh + e) * AKS + b + 128] = ok ? fmaf((float)(byte >> 4), p1.x, p1.y) : 0.f;
        }
    } else {
        for (int idx = tid; idx < AKT * (AD / 4); idx += 256) {
            const int j = idx / (AD / 4), c4 = idx % (AD / 4), key = k0 + j;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (key < n) v = load_fp16_row4(a, kvh, key, c4, false);
            *reinterpret_cast<float4*>(KV + j * AKS + c4 * 4) = v;
        }
    }
}

template <int KM>
__global__ void __launch_bounds__(256, 1) attn_seq_kv4_kernel(AttnSeqArgs a) {
    extern __shared__ __align__(16) float sm[];
    float* Qs = sm;
    float* KV = Qs + AR * AD;
    float* P = KV + AKT * AKS;
    float* alpha_s = P + AR * AKT;
    float* l_s = alpha_s + AR;

    const int tid = threadIdx.x;
    const int kvh = blockIdx.y;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * ATQ;
    const int Hq = a.n_heads;
    const int qi_last = min(q0 + ATQ - 1, a.T - 1);
    const int n = a.pos0 + qi_last + 1;
    const int Qlo = kv4_q(a.kv4, a.pos0 + q0 + 1), Qhi = kv4_q(a.kv4, a.pos0 + qi_last + 1);  // window starts of the first / last query

    for (int idx = tid; idx < AR * (AD / 4); idx += 256) {
        int r = idx / (AD / 4), c4 = idx % (AD / 4);
        int qi = q0 + r / AG, hq = kvh * AG + r % AG;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qi < a.T) v = *reinterpret_cast<const float4*>(a.q + (size_t)qi * Hq * AD + hq * AD + c4 * 4);
        *reinterpret_cast<float4*>(Qs + r * AD + c4 * 4) = v;
    }

    const int rg = tid >> 4, kg = tid & 15;
    const int srow = tid >> 2, ssub = tid & 3;
    int lim_s[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) lim_s[i] = a.pos0 + min(q0 + (rg * 3 + i) / AG, a.T - 1);
    float m_run = -INFINITY, l_run = 0.f;
    const int cg = tid & 63, rp = tid >> 6;
    int lim_pv[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) lim_pv[i] = a.pos0 + min(q0 + (rp * 12 + i) / AG, a.T - 1);
    float acc[12][4];
#pragma unroll
    for (int i = 0; i < 12; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    for (int k0 = 0; k0 < n; k0 += AKT) {
        __syncthreads();
        const bool qa = k0 < Qhi;   // some row reads this tile quantized
        const bool fa = k0 >= Qlo;  // some row reads it from the fp16 window
        const bool both = qa && fa;
#pragma unroll
        for (int vi = 0; vi < 2; ++vi) {
            const bool vq = vi == 0;
            if (vq ? !qa : !fa) continue;
            load_k_tile<KM>(a, KV, kvh, k0, n, vq, tid);
            __syncthreads();
            {
                float s[3][2] = {};
#pragma unroll 4
                for (int d4 = 0; d4 < AD / 4; ++d4) {
                    float4 qv[3], kv[2];
#pragma unroll
                    for (int i = 0; i < 3; ++i) qv[i] = *reinterpret_cast<const float4*>(Qs + (rg * 3 + i) * AD + d4 * 4);
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) kv[jj] = *reinterpret_cast<const float4*>(KV + (kg + 16 * jj) * AKS + d4 * 4);
#pragma unroll
                    for (int i = 0; i < 3; ++i)
#pragma unroll
                        for (int jj = 0; jj < 2; ++jj) {
                            s[i][jj] = fmaf(qv[i].x, kv[jj].x, s[i][jj]);
                            s[i][jj] = fmaf(qv[i].y, kv[jj].y, s[i][jj]);
                            s[i][jj] = fmaf(qv[i].z, kv[jj].z, s[i][jj]);
                            s[i][jj] = fmaf(qv[i].w, kv[jj].w, s[i][jj]);
                        }
                }
#pragma unroll
                for (int i = 0; i < 3; ++i)
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        const int kk = kg + 16 * jj, key = k0 + kk;
                        const bool masked = key > lim_s[i];
                        const bool row_q = key < kv4_q(a.kv4, lim_s[i] + 1);
                        if (!both || masked || row_q == vq)
                            P[(rg * 3 + i) * AKT + kk] = masked ? -INFINITY : s[i][jj] * 0.0625f;
                    }
            }
            __syncthreads();
        }
        if (tid < 192) {
            float sv[8];
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) { sv[u] = P[srow * AKT + ssub + 4 * u]; mx = fmaxf(mx, sv[u]); }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            float mnew = fmaxf(m_run, mx);
            float alpha = 1.f, sum = 0.f;
            if (mnew == -INFINITY) {
#pragma unroll
                for (int u = 0; u < 8; ++u) P[srow * AKT + ssub + 4 * u] = 0.f;
            } else {
                alpha = expf(m_run - mnew);
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float p = expf(sv[u] - mnew);
                    P[srow * AKT + ssub + 4 * u] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l_run = l_run * alpha + sum;
            m_run = mnew;
            if (ssub == 0) { alpha_s[srow] = alpha; l_s[srow] = l_run; }
        }
        bool first = true;
#pragma unroll
        for (int vi = 0; vi < 2; ++vi) {
            const bool vq = vi == 0;
            if (vq ? !qa : !fa) continue;
            load_v_tile(a, KV, kvh, k0, n, vq, tid);
            __syncthreads();
            if (first) {
#pragma unroll
                for (int i = 0; i < 12; ++i) {
                    float al = alpha_s[rp * 12 + i];
                    acc[i][0] *= al; acc[i][1] *= al; acc[i][2] *= al; acc[i][3] *= al;
                }
                first = false;
            }
            float rm[12];
#pragma unroll
            for (int i = 0; i < 12; ++i) rm[i] = (!both || ((k0 < kv4_q(a.kv4, lim_pv[i] + 1)) == vq)) ? 1.f : 0.f;
#pragma unroll
            for (int j4 = 0; j4 < AKT / 4; ++j4) {
                float4 vv[4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) vv[jj] = *reinterpret_cast<const float4*>(KV + (j4 * 4 + jj) * AKS + cg * 4);
#pragma unroll
                for (int i = 0; i < 12; ++i) {
                    float4 p = *reinterpret_cast<const float4*>(P + (rp * 12 + i) * AKT + j4 * 4);
                    float pp[4] = {p.x, p.y, p.z, p.w};
                    if (both) { pp[0] *= rm[i]; pp[1] *= rm[i]; pp[2] *= rm[i]; pp[3] *= rm[i]; }
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        acc[i][0] = fmaf(pp[jj], vv[jj].x, acc[i][0]);
                        acc[i][1] = fmaf(pp[jj], vv[jj].y, acc[i][1]);
                        acc[i][2] = fmaf(pp[jj], vv[jj].z, acc[i][2]);
                        acc[i][3] = fmaf(pp[jj], vv[jj].w, acc[i][3]);
                    }
                }
            }
            if (both && vi == 0) __syncthreads();
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        int r = rp * 12 + i, qi = q0 + r / AG, hq = kvh * AG + r % AG;
        if (qi >= a.T) continue;
        float inv = 1.f / l_s[r];
        float4 g = *reinterpret_cast<const float4*>(a.qg + (size_t)qi * Hq * 2 * AD + hq * 2 * AD + AD + cg * 4);
        float o[4];
        o[0] = acc[i][0] * inv * (1.f / (1.f + expf(-g.x)));
        o[1] = acc[i][1] * inv * (1.f / (1.f + expf(-g.y)));
        o[2] = acc[i][2] * inv * (1.f / (1.f + expf(-g.z)));
        o[3] = acc[i][3] * inv * (1.f / (1.f + expf(-g.w)));
        size_t off = (size_t)qi * Hq * AD + hq * AD + cg * 4;
        if (a.out_f) *reinterpret_cast<float4*>(a.out_f + off) = make_float4(o[0], o[1], o[2], o[3]);
        if (a.out_h) {
            __half2 h01 = __floats2half2_rn(fminf(fmaxf(o[0], -65504.f), 65504.f), fminf(fmaxf(o[1], -65504.f), 65504.f));
            __half2 h23 = __floats2half2_rn(fminf(fmaxf(o[2], -65504.f), 65504.f), fminf(fmaxf(o[3], -65504.f), 65504.f));
            *reinterpret_cast<__half2*>(a.out_h + off) = h01;
            *reinterpret_cast<__half2*>(a.out_h + off + 2) = h23;
        }
    }
}

void attn_seq_fp32(const AttnSeqArgs& a, cudaStream_t s) {
    if (a.n_heads != AG * a.n_kv_heads) { std::fprintf(stderr, "attn_seq: need 6 q heads per kv head\n"); std::abort(); }
    if (a.T <= 0) return;
    static bool init = false;
    if (!init) {
        QV_CUDA_CHECK(cudaFuncSetAttribute(attn_seq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ASMEM));
        init = true;
    }
    dim3 grid((a.T + ATQ - 1) / ATQ, a.n_kv_heads);
    if (a.kv4.blocks) {
        static bool init4 = false;
        if (!init4) {
            QV_CUDA_CHECK(cudaFuncSetAttribute(attn_seq_kv4_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ASMEM));
            QV_CUDA_CHECK(cudaFuncSetAttribute(attn_seq_kv4_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ASMEM));
            init4 = true;
        }
        if (a.kv4.kmode == 0) attn_seq_kv4_kernel<0><<<grid, 256, ASMEM, s>>>(a);
        else attn_seq_kv4_kernel<1><<<grid, 256, ASMEM, s>>>(a);
        kv4_ring_commit(a.kv4, a.n_kv_heads, a.stage_k, a.stage_v, a.pos0, a.T, s);  // after the reads of the old ring
        return;
    }
    attn_seq_kernel<<<grid, 256, ASMEM, s>>>(a);
}

void attn_seq(const AttnSeqArgs& a, cudaStream_t s) {
    static const bool fp32 = [] { const char* e = std::getenv("QV_ATTN"); return e && std::string(e) == "fp32"; }();
    if (fp32) attn_seq_fp32(a, s); else attn_seq_tc(a, s);
}

}  // namespace qv::cuda
