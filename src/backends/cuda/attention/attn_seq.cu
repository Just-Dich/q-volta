// Batched prefill attention (gated GQA). Flash-style: a block owns TQ consecutive queries x the 6 q heads
// of one kv head (R = 48 rows), streams KT-key tiles of K and V through shared memory once for all rows,
// fp32 online softmax, CUDA-core FMA.
#include "attn_seq.h"

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
        size_t off = kv_off(a.max_ctx, D, h, pos) + t;
        a.k_cache[off] = y;
        a.v_cache[off] = a.v[(size_t)tok * a.n_kv_heads * D + h * D + t];
    }
}

void attn_prep_seq(const AttnSeqPrepArgs& a, cudaStream_t s) {
    if (a.head_dim != 256) { std::fprintf(stderr, "attn_seq: head_dim 256 only\n"); std::abort(); }
    if (a.T <= 0) return;
    qk_prep_seq_kernel<256><<<dim3(a.T, a.n_heads + a.n_kv_heads), 256, 0, s>>>(a);
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

void attn_seq(const AttnSeqArgs& a, cudaStream_t s) {
    if (a.n_heads != AG * a.n_kv_heads) { std::fprintf(stderr, "attn_seq: need 6 q heads per kv head\n"); std::abort(); }
    if (a.T <= 0) return;
    static bool init = false;
    if (!init) {
        QV_CUDA_CHECK(cudaFuncSetAttribute(attn_seq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ASMEM));
        init = true;
    }
    dim3 grid((a.T + ATQ - 1) / ATQ, a.n_kv_heads);
    attn_seq_kernel<<<grid, 256, ASMEM, s>>>(a);
}

}  // namespace qv::cuda
