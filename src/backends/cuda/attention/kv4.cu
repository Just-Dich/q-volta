// int4 KV cache: block quantization kernels (see kv4.h for the layout and the window rule).
#include <algorithm>

#include "../common.cuh"
#include "kv4.h"

namespace qv::cuda {

namespace {

struct Src {
    const __half *ring_k, *ring_v, *stage_k, *stage_v;
    int stage_pos0, Hkv;
};

// fp16 value of position p, channel c of kv head h: staged chunk if p >= stage_pos0 else the ring
__device__ __forceinline__ float fetch(const __half* ring, const __half* stage, const Src& s, int h, int p, int c) {
    const __half* ptr = (stage && p >= s.stage_pos0) ? stage + ((size_t)(p - s.stage_pos0) * s.Hkv + h) * KV4_D + c
                                                     : ring + ((size_t)h * KV4_RING + (p & (KV4_RING - 1))) * KV4_D + c;
    return __half2float(*ptr);
}

__device__ __forceinline__ void fix_params(float mn, float mx, float& sf, float& zf, __half2& out) {
    __half sh = __float2half(mx > mn ? (mx - mn) * (1.f / 15.f) : 0.f);
    __half zh = __float2half(mn);
    sf = __half2float(sh);
    zf = __half2float(zh);
    out = __halves2half2(sh, zh);
}

__device__ __forceinline__ uint8_t qval(float x, float sf, float zf) {
    if (!(sf > 0.f)) return 0;
    return (uint8_t)fminf(fmaxf(rintf((x - zf) / sf), 0.f), 15.f);
}

// one block (64 tokens) of one kv head. 256 threads.
template <int KM>
__device__ void quant_block(const Kv4Layer& l, const Src& src, int h, int b) {
    __shared__ uint8_t qb[KV4_BLK * KV4_D];
    const int tid = threadIdx.x, t0 = b * KV4_BLK;
    uint8_t* rec = kv4_rec(l, h, b);
    __half2* kp = reinterpret_cast<__half2*>(rec + KV4_REC_KP);
    __half2* vp = reinterpret_cast<__half2*>(rec + KV4_REC_VP);
    // ---- K ----
    if (KM == 0) {  // thread = channel; statistics over the 64 tokens
        float mn = INFINITY, mx = -INFINITY;
        for (int t = 0; t < KV4_BLK; ++t) {
            float x = fetch(src.ring_k, src.stage_k, src, h, t0 + t, tid);
            mn = fminf(mn, x); mx = fmaxf(mx, x);
        }
        float sf, zf; __half2 pr;
        fix_params(mn, mx, sf, zf, pr);
        kp[tid] = pr;
        for (int t = 0; t < KV4_BLK; ++t)
            qb[t * KV4_D + tid] = qval(fetch(src.ring_k, src.stage_k, src, h, t0 + t, tid), sf, zf);
    } else {  // thread = (token, group of 64 channels)
        const int t = tid >> 2, g = tid & 3;
        float mn = INFINITY, mx = -INFINITY;
        for (int e = 0; e < 64; ++e) {
            float x = fetch(src.ring_k, src.stage_k, src, h, t0 + t, g * 64 + e);
            mn = fminf(mn, x); mx = fmaxf(mx, x);
        }
        float sf, zf; __half2 pr;
        fix_params(mn, mx, sf, zf, pr);
        kp[t * 4 + g] = pr;
        for (int e = 0; e < 64; ++e)
            qb[t * KV4_D + g * 64 + e] = qval(fetch(src.ring_k, src.stage_k, src, h, t0 + t, g * 64 + e), sf, zf);
    }
    __syncthreads();
    uint32_t* kd = reinterpret_cast<uint32_t*>(rec + KV4_REC_K);
    for (int idx = tid; idx < KV4_BLK * 32; idx += 256) {
        const int t = idx >> 5, w = idx & 31;
        const uint8_t* r = qb + t * KV4_D;
        uint32_t word = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) word |= ((uint32_t)r[4 * w + k] << (4 * k)) | ((uint32_t)r[128 + 4 * w + k] << (16 + 4 * k));
        kd[idx] = word;
    }
    __syncthreads();
    // ---- V (per-token groups) ----
    {
        const int t = tid >> 2, g = tid & 3;
        float mn = INFINITY, mx = -INFINITY;
        for (int e = 0; e < 64; ++e) {
            float x = fetch(src.ring_v, src.stage_v, src, h, t0 + t, g * 64 + e);
            mn = fminf(mn, x); mx = fmaxf(mx, x);
        }
        float sf, zf; __half2 pr;
        fix_params(mn, mx, sf, zf, pr);
        vp[t * 4 + g] = pr;
        for (int e = 0; e < 64; ++e)
            qb[t * KV4_D + g * 64 + e] = qval(fetch(src.ring_v, src.stage_v, src, h, t0 + t, g * 64 + e), sf, zf);
    }
    __syncthreads();
    uint32_t* vd = reinterpret_cast<uint32_t*>(rec + KV4_REC_V);
    for (int w = tid; w < 128 * 16; w += 256) {
        const int bc = w >> 4, tt = (w & 15) * 4;
        uint32_t word = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint8_t* r = qb + (tt + k) * KV4_D;
            word |= ((uint32_t)(r[bc] | (r[bc + 128] << 4))) << (8 * k);
        }
        vd[w] = word;
    }
}

template <int KM>
__global__ void __launch_bounds__(256) quant_decode_kernel(Kv4Layer l, Src src, const int* pos) {
    const int p = pos[blockIdx.z];
    if (((p + 1) & (KV4_BLK - 1)) || p + 1 < KV4_BLK * KV4_LAG) return;
    quant_block<KM>(l, src, blockIdx.x, ((p + 1) >> 6) - KV4_LAG);
}

template <int KM>
__global__ void __launch_bounds__(256) quant_seq_kernel(Kv4Layer l, Src src, int b_lo) {
    quant_block<KM>(l, src, blockIdx.x, b_lo + blockIdx.y);
}

__global__ void ring_commit_kernel(Kv4Layer l, Src src, int p0) {
    const int h = blockIdx.x, p = p0 + blockIdx.y, t = threadIdx.x;
    const size_t so = ((size_t)(p - src.stage_pos0) * src.Hkv + h) * KV4_D + t;
    const size_t ro = ((size_t)h * l.ring_n + kv4_slot(l, p)) * KV4_D + t;
    l.ring_k[ro] = src.stage_k[so];
    l.ring_v[ro] = src.stage_v[so];
}

}  // namespace

void kv4_quant_decode(const Kv4Layer& l, int Hkv, const int* pos, int nt, cudaStream_t s) {
    if (l.exact) return;
    Src src{l.ring_k, l.ring_v, nullptr, nullptr, 0x7fffffff, Hkv};
    if (l.kmode == 0) quant_decode_kernel<0><<<dim3(Hkv, 1, nt), 256, 0, s>>>(l, src, pos);
    else quant_decode_kernel<1><<<dim3(Hkv, 1, nt), 256, 0, s>>>(l, src, pos);
}

void kv4_quant_seq(const Kv4Layer& l, int Hkv, const __half* sk, const __half* sv, int pos0, int T, cudaStream_t s) {
    if (l.exact) return;
    const int b_lo = std::max(0, pos0 / KV4_BLK - (KV4_LAG - 1));
    const int b_hi = (pos0 + T) / KV4_BLK - KV4_LAG;
    if (b_hi < b_lo) return;
    Src src{l.ring_k, l.ring_v, sk, sv, pos0, Hkv};
    dim3 grid(Hkv, b_hi - b_lo + 1);
    if (l.kmode == 0) quant_seq_kernel<0><<<grid, 256, 0, s>>>(l, src, b_lo);
    else quant_seq_kernel<1><<<grid, 256, 0, s>>>(l, src, b_lo);
}

void kv4_ring_commit(const Kv4Layer& l, int Hkv, const __half* sk, const __half* sv, int pos0, int T, cudaStream_t s) {
    const int p0 = std::max(pos0, pos0 + T - l.ring_n);
    Src src{l.ring_k, l.ring_v, sk, sv, pos0, Hkv};
    ring_commit_kernel<<<dim3(Hkv, pos0 + T - p0), 256, 0, s>>>(l, src, p0);
}

}  // namespace qv::cuda
