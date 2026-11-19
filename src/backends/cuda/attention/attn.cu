// Gated GQA full attention, single-token decode (Qwen3.5-style: q_proj emits [q | gate] per head,
// zero-centred q/k RMSNorm, partial rotate-half RoPE on the first rope_dim dims, output * sigmoid(gate)).
#include "../common.cuh"
#include "attn.h"

namespace qv::cuda {

// grid = n_heads + n_kv_heads blocks, D threads. Blocks [0,Hq) prepare q, [Hq, Hq+Hkv) prepare k and copy v.
template <int D>
__global__ void __launch_bounds__(D) qk_prep_kernel(AttnPrepArgs a) {
    __shared__ float red[32];
    __shared__ float buf[D];
    const int b = blockIdx.x, t = threadIdx.x;
    const bool is_q = b < a.n_heads;
    const int h = is_q ? b : b - a.n_heads;
    float x = is_q ? a.qg[h * 2 * D + t] : a.k[h * D + t];
    const float* w = is_q ? a.q_norm : a.k_norm;
    float ss = block_sum(x * x, red);
    float y = x * rsqrtf(ss / D + a.eps) * w[t];
    buf[t] = y;
    __syncthreads();
    const int pos = *a.pos;
    const int half = a.rope_dim / 2;
    if (t < a.rope_dim) {
        int i = t < half ? t : t - half;
        float f = (float)pos * a.inv_freq[i];
        float c = cosf(f), s = sinf(f);
        y = t < half ? y * c - buf[t + half] * s : y * c + buf[t - half] * s;
    }
    if (is_q) {
        a.q_out[h * D + t] = y;
    } else {
        size_t off = ((size_t)h * a.max_ctx + pos) * D + t;
        a.k_cache[off] = y;
        a.v_cache[off] = a.v[h * D + t];
    }
}

void attn_prep(const AttnPrepArgs& a, cudaStream_t s) {
    if (a.head_dim != 256) { std::fprintf(stderr, "attn: head_dim 256 only\n"); std::abort(); }
    qk_prep_kernel<256><<<a.n_heads + a.n_kv_heads, 256, 0, s>>>(a);
}

// One block per q head; scores kept in a global scratch row [n_heads][max_ctx].
template <int D>
__global__ void __launch_bounds__(D) attn_decode_kernel(AttnDecodeArgs a) {
    __shared__ float qs[D];
    __shared__ float red[32];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5, nwarp = D / 32;
    const int kvh = h / (a.n_heads / a.n_kv_heads);
    const int n = *a.pos + 1;
    qs[t] = a.q[h * D + t];
    __syncthreads();
    const float* K = a.k_cache + (size_t)kvh * a.max_ctx * D;
    const float* V = a.v_cache + (size_t)kvh * a.max_ctx * D;
    float* sc = a.scores + (size_t)h * a.max_ctx;
    const float scale = rsqrtf((float)D);

    float mx = -INFINITY;
    for (int j = warp; j < n; j += nwarp) {
        const float* kr = K + (size_t)j * D;
        float d = 0.f;
#pragma unroll
        for (int e = lane; e < D; e += 32) d = fmaf(qs[e], kr[e], d);
        d = warp_sum(d) * scale;
        if (lane == 0) sc[j] = d;
        mx = fmaxf(mx, d);
    }
    // block max
    mx = warp_max(mx);
    __syncthreads();
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (t < 32) {
        float v = t < nwarp ? red[t] : -INFINITY;
        v = warp_max(v);
        if (t == 0) red[0] = v;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int j = t; j < n; j += D) {
        float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);  // contains __syncthreads → sc[] writes visible
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc = fmaf(sc[j], V[(size_t)j * D + t], acc);
    float o = acc / sum;
    float gate = a.qg[h * 2 * D + D + t];
    a.out[h * D + t] = o * (1.f / (1.f + expf(-gate)));
}

void attn_decode(const AttnDecodeArgs& a, cudaStream_t s) {
    attn_decode_kernel<256><<<a.n_heads, 256, 0, s>>>(a);
}

}  // namespace qv::cuda
