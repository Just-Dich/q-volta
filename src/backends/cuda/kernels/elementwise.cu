#include "../common.cuh"
#include "elementwise.h"
#include "q8.cuh"

namespace qv::cuda {

// out = x * rsqrt(mean(x^2)+eps) * w   (w already holds 1+w for zero-centred norms); also emits xsum32(out) or
// the fused int8 GEMV input. Multi-block: every block recomputes the (L2-resident) sum of squares, then handles its
// own slice of 32-chunks — trades redundant reads of n floats for parallelism (latency-bound single-block before).
constexpr int NORM_CHUNKS_PER_BLOCK = 8;
__global__ void __launch_bounds__(256) rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                      float* __restrict__ out, float* __restrict__ xs32, int n,
                                                      float eps, Q8Buf q8) {
    __shared__ float red[32];
    x += (size_t)blockIdx.y * n;  // token row (nt > 1: MTP verify); q8 chunks continue token-major
    out += (size_t)blockIdx.y * n;
    float ss = 0.f;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / n + eps);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int c = blockIdx.x * NORM_CHUNKS_PER_BLOCK + wid;  // one warp per chunk
    if (wid >= NORM_CHUNKS_PER_BLOCK || c >= n / 32) return;
    const int i = c * 32 + lane;
    const float v = x[i] * r * w[i];
    out[i] = v;
    if (q8.q) {
        q8_store_chunk(v, c + blockIdx.y * (n / 32), q8);
    } else {
        float s = warp_sum(v);
        if (xs32 && lane == 0) xs32[c] = s;
    }
}

void rmsnorm(const float* x, const float* w, float* out, float* xs32, int n, float eps, cudaStream_t s, const Q8Buf* q8,
             int nt) {
    const int blocks = (n / 32 + NORM_CHUNKS_PER_BLOCK - 1) / NORM_CHUNKS_PER_BLOCK;
    rmsnorm_kernel<<<dim3(blocks, nt), 32 * NORM_CHUNKS_PER_BLOCK, 0, s>>>(x, w, out, xs32, n, eps, q8 ? *q8 : Q8Buf{});
}

// Dequantize one embedding row (token id read from device memory) into out[K].
__global__ void embed_kernel(QMat E, const int* __restrict__ tok, float* __restrict__ out) {
    const int row = *tok;
    const int C = E.K >> 5, G = E.K >> E.gshift, GS = E.gshift - 5;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < E.K; k += gridDim.x * blockDim.x) {
        int c = k >> 5, i = k & 31;
        const uint32_t* w = E.q + ((size_t)row * C + c) * E.bits;
        uint32_t q;
        // generic (slow path is fine: 5120 elements once per token)
        switch (E.bits) {
            case 2: q = (w[i & 1] >> (2 * (i >> 1))) & 3u; break;
            case 3: q = ((w[i & 1] >> (2 * (i >> 1))) & 3u) | (((w[2] >> i) & 1u) << 2); break;
            case 4: q = (w[i & 3] >> (4 * (i >> 2))) & 15u; break;
            case 5: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4] >> i) & 1u) << 4); break;
            case 6: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 1)] >> (2 * (i >> 1))) & 3u) << 4); break;
            default: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 3)] >> (4 * (i >> 2))) & 15u) << 4);
        }
        float2 sm = load_sm(E, row, G, c >> GS);
        out[k] = sm.x * (float)q + sm.y;
    }
}

void embed(const QMat& E, const int* tok, float* out, cudaStream_t s) {
    embed_kernel<<<(E.K + 255) / 256, 256, 0, s>>>(E, tok, out);
}

// act = silu(g) * u ; also chunk sums for the following quantized GEMV.
__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out,
                                float* __restrict__ xs32, int n, Q8Buf q8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // n % 32 == 0 → whole warps in range
    if (i >= n) return;
    float v = silu(g[i]) * u[i];
    out[i] = v;
    if (q8.q) { q8_store_chunk(v, i >> 5, q8); return; }
    float s = warp_sum(v);
    if ((threadIdx.x & 31) == 0) xs32[i >> 5] = s;
}

void silu_mul(const float* g, const float* u, float* out, float* xs32, int n, cudaStream_t s, const Q8Buf* q8) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, out, xs32, n, q8 ? *q8 : Q8Buf{});
}

// argmax over n floats, single block (first index wins on ties, like torch.argmax).
__global__ void argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        best = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) *out = bi;
    }
}

// Greedy feedback for graph decode: tok = argmax(logits); gen[pos] = tok; pos += 1.
// Two stages: ARG_BLOCKS blocks reduce slices to (value, index) partials; one block finishes + feeds back.
constexpr int ARG_BLOCKS = 240;

__device__ __forceinline__ void argmax_merge(float& best, int& bi, float ov, int oi) {
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
}

__device__ void block_argmax(float& best, int& bi) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1)
        argmax_merge(best, bi, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o));
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        best = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1)
            argmax_merge(best, bi, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o));
    }
}

__global__ void argmax_partial_kernel(const float* __restrict__ x, int n, float* pv, int* pi) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) argmax_merge(best, bi, x[i], i);
    block_argmax(best, bi);
    if (threadIdx.x == 0) { pv[blockIdx.x] = best; pi[blockIdx.x] = bi; }
}

__global__ void argmax_final_kernel(const float* pv, const int* pi, int m, int* tok, int* pos, int* gen) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < m; i += blockDim.x) argmax_merge(best, bi, pv[i], pi[i]);
    block_argmax(best, bi);
    if (threadIdx.x == 0) {
        *tok = bi;
        if (pos) {
            int p = *pos;
            gen[p] = bi;
            *pos = p + 1;
        }
    }
}

static float* g_pv = nullptr;
static int* g_pi = nullptr;

void argmax_advance(const float* x, int n, int* tok, int* pos, int* gen, cudaStream_t s) {
    if (!g_pv) {  // one-time scratch (allocated before any graph capture: first call is eager)
        QV_CUDA_CHECK(cudaMalloc(&g_pv, ARG_BLOCKS * sizeof(float)));
        QV_CUDA_CHECK(cudaMalloc(&g_pi, ARG_BLOCKS * sizeof(int)));
    }
    argmax_partial_kernel<<<ARG_BLOCKS, 256, 0, s>>>(x, n, g_pv, g_pi);
    argmax_final_kernel<<<1, 256, 0, s>>>(g_pv, g_pi, ARG_BLOCKS, tok, pos, gen);
}

void argmax(const float* x, int n, int* out, cudaStream_t s) { argmax_advance(x, n, out, nullptr, nullptr, s); }

__global__ void cand_max_final_kernel(const float* pv, const int* pi, int m, CandMeta* meta) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < m; i += blockDim.x) argmax_merge(best, bi, pv[i], pi[i]);
    block_argmax(best, bi);
    if (threadIdx.x == 0) {
        meta->mx = best;
        meta->sum_exp = 0.0f;
        meta->count = 0;
    }
}

__global__ void extract_cand_kernel(const float* __restrict__ x, int n, float margin, int max_cand,
                                    int* __restrict__ cid, float* __restrict__ cval, CandMeta* meta) {
    __shared__ float red[32];
    const float mx = meta->mx;
    const bool ok = isfinite(mx);
    const float thr = mx - margin;
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
        if (ok) {
            local_sum += expf(v - mx);
            if (v >= thr) {
                const int idx = atomicAdd(&meta->count, 1);
                if (idx < max_cand) {
                    cid[idx] = i;
                    cval[idx] = v;
                }
            }
        }
    }
    local_sum = block_sum(local_sum, red);
    if (threadIdx.x == 0 && ok) atomicAdd(&meta->sum_exp, local_sum);
}

void extract_top_candidates(const float* x, int n, float margin, int max_cand,
                            int* cand_id, float* cand_val, CandMeta* meta, cudaStream_t s) {
    if (!g_pv) {
        QV_CUDA_CHECK(cudaMalloc(&g_pv, ARG_BLOCKS * sizeof(float)));
        QV_CUDA_CHECK(cudaMalloc(&g_pi, ARG_BLOCKS * sizeof(int)));
    }
    argmax_partial_kernel<<<ARG_BLOCKS, 256, 0, s>>>(x, n, g_pv, g_pi);
    cand_max_final_kernel<<<1, 256, 0, s>>>(g_pv, g_pi, ARG_BLOCKS, meta);
    extract_cand_kernel<<<ARG_BLOCKS, 256, 0, s>>>(x, n, margin, max_cand, cand_id, cand_val, meta);
}

}  // namespace qv::cuda
