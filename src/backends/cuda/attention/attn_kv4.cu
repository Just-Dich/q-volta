// Split-K decode attention over the int4 KV cache (layout and window rule: kv4.h). Same structure as attn_split_kernel
// in attn.cu (CTA = (kv head, split), G=6 q heads share every K/V read, online softmax per 64-key tile, partials merged by
// attn_combine_kernel) with two differences: splits own whole 64-key blocks, and a tile is either entirely quantized
// (dequantized in registers while streaming) or entirely in the fp16 ring.
#include "../common.cuh"
#include "attn.h"

namespace qv::cuda {

constexpr int K4_TK = 64;  // keys per tile == block size
constexpr int K4_G = 6;

__device__ __forceinline__ int sk4_chunk_k(int n, int S) { return (((n + 63) / 64 + S - 1) / S) * 64; }

__device__ __forceinline__ float2 h2f(uint32_t u) {
    __half2 h = *reinterpret_cast<__half2*>(&u);
    return make_float2(__low2float(h), __high2float(h));
}

template <int KM>
__global__ void __launch_bounds__(256) attn_split_kv4_kernel(AttnDecodeArgs a) {
    constexpr int D = 256, NW = 8, G = K4_G;
    __shared__ float ps[G][K4_TK];
    __shared__ float alpha_s[G];
    a.q += (size_t)blockIdx.z * a.q_stride; a.part += blockIdx.z * a.part_stride; a.pos += blockIdx.z;
    const int g = blockIdx.x, sp = blockIdx.y, S = gridDim.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int n = *a.pos + 1;
    const int Q = kv4_q(a.kv4, n);
    const int chunk = sk4_chunk_k(n, S);
    const int j0 = sp * chunk;
    const int j1 = min(n, j0 + chunk);
    float* part_acc = a.part;
    float* part_ml = a.part + (size_t)a.n_heads * S * D;
    const int hb = g * G;

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;

    if (j0 < j1) {
        constexpr int NV = D / 128;
        float4 q[G][NV];
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int i = 0; i < NV; ++i)
                q[h][i] = *reinterpret_cast<const float4*>(a.q + (size_t)(hb + h) * D + 4 * lane + 128 * i);
        const Kv4Layer& kv = a.kv4;
        const __half* RK = kv.ring_k + (size_t)g * kv.ring_n * D;
        const __half* RV = kv.ring_v + (size_t)g * kv.ring_n * D;
        const float scale = rsqrtf((float)D);

        for (int jt = j0; jt < j1; jt += K4_TK) {
            const bool quant = jt < Q;
            const uint8_t* rec = quant ? kv4_rec(kv, g, jt >> 6) : nullptr;
            float sc[8], zr[8];
            if (quant && KM == 0) {  // per-channel K parameters of this block, for this lane's 8 channels
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    uint4 u = reinterpret_cast<const uint4*>(rec + KV4_REC_KP)[lane + 32 * i];
                    float2 p0 = h2f(u.x), p1 = h2f(u.y), p2 = h2f(u.z), p3 = h2f(u.w);
                    sc[4 * i + 0] = p0.x; zr[4 * i + 0] = p0.y; sc[4 * i + 1] = p1.x; zr[4 * i + 1] = p1.y;
                    sc[4 * i + 2] = p2.x; zr[4 * i + 2] = p2.y; sc[4 * i + 3] = p3.x; zr[4 * i + 3] = p3.y;
                }
            }
            // phase 1: scores. warp w handles keys jt+w, jt+w+NW, ...
            for (int jj = warp; jj < K4_TK; jj += NW) {
                const int j = jt + jj;
                if (j < j1) {
                    float4 kvv[NV];
                    if (quant) {
                        const uint32_t w = *reinterpret_cast<const uint32_t*>(rec + KV4_REC_K + jj * 128 + lane * 4);
                        float kf[8];
                        if (KM == 0) {
#pragma unroll
                            for (int e = 0; e < 8; ++e) kf[e] = fmaf((float)((w >> (4 * e)) & 15u), sc[e], zr[e]);
                        } else {
                            const uint32_t* kp = reinterpret_cast<const uint32_t*>(rec + KV4_REC_KP);
                            float2 p0 = h2f(kp[jj * 4 + (lane >> 4)]), p1 = h2f(kp[jj * 4 + (lane >> 4) + 2]);
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                kf[e] = fmaf((float)((w >> (4 * e)) & 15u), p0.x, p0.y);
                                kf[4 + e] = fmaf((float)((w >> (16 + 4 * e)) & 15u), p1.x, p1.y);
                            }
                        }
                        kvv[0] = make_float4(kf[0], kf[1], kf[2], kf[3]);
                        kvv[1] = make_float4(kf[4], kf[5], kf[6], kf[7]);
                    } else {
                        const __half* rp = RK + (size_t)kv4_slot(kv, j) * D;
#pragma unroll
                        for (int i = 0; i < NV; ++i) {
                            uint2 u = *reinterpret_cast<const uint2*>(rp + 4 * lane + 128 * i);
                            float2 a0 = h2f(u.x), a1 = h2f(u.y);
                            kvv[i] = make_float4(a0.x, a0.y, a1.x, a1.y);
                        }
                    }
                    float d[G];
#pragma unroll
                    for (int h = 0; h < G; ++h) {
                        float x = 0.f;
#pragma unroll
                        for (int i = 0; i < NV; ++i) {
                            x = fmaf(q[h][i].x, kvv[i].x, x);
                            x = fmaf(q[h][i].y, kvv[i].y, x);
                            x = fmaf(q[h][i].z, kvv[i].z, x);
                            x = fmaf(q[h][i].w, kvv[i].w, x);
                        }
                        d[h] = x;
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
                        for (int h = 0; h < G; ++h) d[h] += __shfl_xor_sync(0xffffffffu, d[h], off);
                    if (lane < G) {
                        float v = d[0];
#pragma unroll
                        for (int h = 1; h < G; ++h) if (lane == h) v = d[h];
                        ps[lane][jj] = v * scale;
                    }
                } else if (lane < G) {
                    ps[lane][jj] = -INFINITY;
                }
            }
            __syncthreads();
            // phase 2: online softmax bookkeeping (warp h < G handles head h)
            if (warp < G) {
                float s0 = ps[warp][lane], s1 = ps[warp][lane + 32];
                float tm = warp_max(fmaxf(s0, s1));
                float m_new = fmaxf(m_run, tm);
                float p0 = expf(s0 - m_new), p1 = expf(s1 - m_new);
                float al = expf(m_run - m_new);
                ps[warp][lane] = p0;
                ps[warp][lane + 32] = p1;
                l_run = l_run * al + warp_sum(p0 + p1);
                m_run = m_new;
                if (lane == 0) alpha_s[warp] = al;
            }
            __syncthreads();
            // phase 3: accumulate; thread t owns output channel t
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] *= alpha_s[h];
            const int cnt = min(K4_TK, j1 - jt);
            if (quant) {
                uint32_t vw[16];
                const uint4* vp = reinterpret_cast<const uint4*>(rec + KV4_REC_V + (size_t)(t & 127) * 64);
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    uint4 u = vp[x];
                    vw[4 * x] = u.x; vw[4 * x + 1] = u.y; vw[4 * x + 2] = u.z; vw[4 * x + 3] = u.w;
                }
                const int sh = (t >> 7) * 4;
                const uint32_t* vpar = reinterpret_cast<const uint32_t*>(rec + KV4_REC_VP) + (t >> 6);
#pragma unroll
                for (int g4 = 0; g4 < 16; ++g4) {
                    if (g4 * 4 >= cnt) break;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int jj = g4 * 4 + k;
                        if (jj < cnt) {
                            float2 pr = h2f(vpar[jj * 4]);
                            const float v = fmaf((float)((vw[g4] >> (8 * k + sh)) & 15u), pr.x, pr.y);
#pragma unroll
                            for (int h = 0; h < G; ++h) acc[h] = fmaf(ps[h][jj], v, acc[h]);
                        }
                    }
                }
            } else {
                const __half* vr = RV + t;
#pragma unroll 4
                for (int jj = 0; jj < cnt; ++jj) {
                    const float v = __half2float(vr[(size_t)kv4_slot(kv, jt + jj) * D]);
#pragma unroll
                    for (int h = 0; h < G; ++h) acc[h] = fmaf(ps[h][jj], v, acc[h]);
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t)(hb + h) * S + sp) * D + t] = acc[h];
    if (warp < G && lane == 0) {
        part_ml[((size_t)(hb + warp) * S + sp) * 2] = m_run;
        part_ml[((size_t)(hb + warp) * S + sp) * 2 + 1] = l_run;
    }
}

void attn_split_kv4(const AttnDecodeArgs& a, cudaStream_t s) {
    if (a.kv4.kmode == 0) attn_split_kv4_kernel<0><<<dim3(a.n_kv_heads, a.max_splits, a.nt), 256, 0, s>>>(a);
    else attn_split_kv4_kernel<1><<<dim3(a.n_kv_heads, a.max_splits, a.nt), 256, 0, s>>>(a);
}

}  // namespace qv::cuda
