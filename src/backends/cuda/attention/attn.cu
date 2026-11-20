// Gated GQA full attention, single-token decode (Qwen3.5-style: q_proj emits [q | gate] per head,
// zero-centred q/k RMSNorm, partial rotate-half RoPE on the first rope_dim dims, output * sigmoid(gate)).
#include "../common.cuh"
#include "attn.h"
#include "../kernels/q8.cuh"

namespace qv::cuda {

// grid = n_heads + n_kv_heads blocks, D threads. Blocks [0,Hq) prepare q, [Hq, Hq+Hkv) prepare k and copy v.
template <int D>
__global__ void __launch_bounds__(D) qk_prep_kernel(AttnPrepArgs a) {
    __shared__ float red[32];
    __shared__ float buf[D];
    const int b = blockIdx.x, t = threadIdx.x;
    a.qg += (size_t)blockIdx.y * a.qg_stride; a.k += (size_t)blockIdx.y * a.k_stride; a.v += (size_t)blockIdx.y * a.k_stride;
    a.q_out += (size_t)blockIdx.y * a.q_stride; a.pos += blockIdx.y;
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
        if (a.kv4.blocks) {
            const size_t ro = ((size_t)h * a.kv4.ring_n + kv4_slot(a.kv4, pos)) * D + t;
            a.kv4.ring_k[ro] = __float2half(fminf(fmaxf(y, -65504.f), 65504.f));
            a.kv4.ring_v[ro] = __float2half(fminf(fmaxf(a.v[h * D + t], -65504.f), 65504.f));
        } else {
            size_t off = ((size_t)h * a.max_ctx + pos) * D + t;
            a.k_cache[off] = y;
            a.v_cache[off] = a.v[h * D + t];
        }
    }
}

void attn_prep(const AttnPrepArgs& a, cudaStream_t s) {
    if (a.head_dim != 256) { std::fprintf(stderr, "attn: head_dim 256 only\n"); std::abort(); }
    qk_prep_kernel<256><<<dim3(a.n_heads + a.n_kv_heads, a.nt), 256, 0, s>>>(a);
    if (a.kv4.blocks) kv4_quant_decode(a.kv4, a.n_kv_heads, a.pos, a.nt, s);
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
    const float y = o * (1.f / (1.f + expf(-gate)));
    a.out[h * D + t] = y;
    if (a.q8.q) q8_store_chunk(y, (h * D + t) >> 5, a.q8);
}

void attn_decode_ref(const AttnDecodeArgs& a, cudaStream_t s) {
    attn_decode_kernel<256><<<a.n_heads, 256, 0, s>>>(a);
}

// ---------------------------------------------------------------------------------------------------------------
// Split-K decode. Grid (n_kv_heads, S). CTA (g, s) owns a contiguous key slice for the G = Hq/Hkv q heads of kv head
// g, so each K/V row is read once per kv head. Online softmax over tiles of TK keys yields (m, l, acc[D]) partials;
// attn_combine_kernel merges them, then applies the gate and the optional int8 epilogue.
// ---------------------------------------------------------------------------------------------------------------
constexpr int SK_TK = 64;        // keys per tile
constexpr int SK_MIN_CHUNK = 32; // minimum keys per split (short contexts use few active CTAs)
constexpr int SK_G = 6;          // q heads per kv head (compile-time; checked on the host)

__device__ __forceinline__ int sk_chunk(int n, int S) {
    int c = (n + S - 1) / S;
    return c < SK_MIN_CHUNK ? SK_MIN_CHUNK : c;
}

// int4 KV: splits own whole 64-key blocks (attn_kv4.cu uses the same rule)
__device__ __forceinline__ int sk4_chunk(int n, int S) { return (((n + 63) / 64 + S - 1) / S) * 64; }

template <int D>
__global__ void __launch_bounds__(D) attn_split_kernel(AttnDecodeArgs a) {
    constexpr int NW = D / 32;
    constexpr int G = SK_G;
    __shared__ float ps[G][SK_TK];  // scores, then probabilities
    __shared__ float alpha_s[G];
    a.q += (size_t)blockIdx.z * a.q_stride; a.part += blockIdx.z * a.part_stride; a.pos += blockIdx.z;
    const int g = blockIdx.x, sp = blockIdx.y, S = gridDim.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int n = *a.pos + 1;
    const int chunk = sk_chunk(n, S);
    const int j0 = sp * chunk;
    const int j1 = min(n, j0 + chunk);
    float* part_acc = a.part;
    float* part_ml = a.part + (size_t)a.n_heads * S * D;
    const int hb = g * G;

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;  // meaningful in warp h < G (lane-uniform)

    if (j0 < j1) {
        // q slice for this lane: elements e = 4*lane + 128*i (i < D/128), all G heads, in registers
        constexpr int NV = D / 128;
        float4 q[G][NV];
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int i = 0; i < NV; ++i)
                q[h][i] = *reinterpret_cast<const float4*>(a.q + (size_t)(hb + h) * D + 4 * lane + 128 * i);
        const float* K = a.k_cache + (size_t)g * a.max_ctx * D;
        const float* V = a.v_cache + (size_t)g * a.max_ctx * D;
        const float scale = rsqrtf((float)D);

        for (int jt = j0; jt < j1; jt += SK_TK) {
            // phase 1: scores. warp w handles keys jt+w, jt+w+NW, ...
            for (int jj = warp; jj < SK_TK; jj += NW) {
                const int j = jt + jj;
                if (j < j1) {
                    const float* kr = K + (size_t)j * D;
                    float4 kv[NV];
#pragma unroll
                    for (int i = 0; i < NV; ++i) kv[i] = *reinterpret_cast<const float4*>(kr + 4 * lane + 128 * i);
                    float d[G];
#pragma unroll
                    for (int h = 0; h < G; ++h) {
                        float x = 0.f;
#pragma unroll
                        for (int i = 0; i < NV; ++i) {
                            x = fmaf(q[h][i].x, kv[i].x, x);
                            x = fmaf(q[h][i].y, kv[i].y, x);
                            x = fmaf(q[h][i].z, kv[i].z, x);
                            x = fmaf(q[h][i].w, kv[i].w, x);
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
            // phase 2: warp h < G does the online-softmax bookkeeping for head h
            if (warp < G) {
                float s0 = ps[warp][lane], s1 = ps[warp][lane + 32];
                float tm = warp_max(fmaxf(s0, s1));
                float m_new = fmaxf(m_run, tm);
                float p0 = expf(s0 - m_new), p1 = expf(s1 - m_new);
                float al = expf(m_run - m_new);  // m_run = -inf on first tile -> 0
                ps[warp][lane] = p0;
                ps[warp][lane + 32] = p1;
                l_run = l_run * al + warp_sum(p0 + p1);
                m_run = m_new;
                if (lane == 0) alpha_s[warp] = al;
            }
            __syncthreads();
            // phase 3: accumulate; thread t owns output dim t
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] *= alpha_s[h];
            const int cnt = min(SK_TK, j1 - jt);
            const float* vp = V + (size_t)jt * D + t;
#pragma unroll 4
            for (int jj = 0; jj < cnt; ++jj) {
                const float v = vp[(size_t)jj * D];
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = fmaf(ps[h][jj], v, acc[h]);
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

// grid = n_heads, D threads. Merge partials of the active splits, apply gate, optional q8 epilogue.
template <int D>
__global__ void __launch_bounds__(D) attn_combine_kernel(AttnDecodeArgs a, int S) {
    const int h = blockIdx.x, t = threadIdx.x;
    a.part += blockIdx.y * a.part_stride; a.pos += blockIdx.y; a.qg += (size_t)blockIdx.y * a.qg_stride;
    a.out += (size_t)blockIdx.y * a.out_stride;
    if (a.q8.q) { a.q8.q += (size_t)blockIdx.y * (a.q8_k >> 2); a.q8.d += (size_t)blockIdx.y * (a.q8_k >> 5);
                  a.q8.sum += (size_t)blockIdx.y * (a.q8_k >> 5); a.q8.qsum += (size_t)blockIdx.y * (a.q8_k >> 5); }
    const int n = *a.pos + 1;
    const int chunk = a.kv4.blocks ? sk4_chunk(n, S) : sk_chunk(n, S);
    const int ns = min(S, (n + chunk - 1) / chunk);  // splits that own at least one key (>= 1)
    const float* ml = a.part + (size_t)a.n_heads * S * D + (size_t)h * S * 2;
    const float* pa = a.part + (size_t)h * S * D + t;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, ml[2 * s]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float w = expf(ml[2 * s] - M);  // empty slices (m=-inf) would give 0, but are never visited
        L = fmaf(ml[2 * s + 1], w, L);
        acc = fmaf(pa[(size_t)s * D], w, acc);
    }
    const float o = acc / L;
    const float gate = a.qg[h * 2 * D + D + t];
    const float y = o * (1.f / (1.f + expf(-gate)));
    a.out[h * D + t] = y;
    if (a.q8.q) q8_store_chunk(y, (h * D + t) >> 5, a.q8);
}

void attn_decode(const AttnDecodeArgs& a, cudaStream_t s) {
    if (!a.part) { attn_decode_ref(a, s); return; }
    if (a.n_heads != a.n_kv_heads * SK_G || a.max_splits < 1) {
        std::fprintf(stderr, "attn_decode: split-K needs n_heads == 6*n_kv_heads and max_splits >= 1\n");
        std::abort();
    }
    if (a.kv4.blocks) attn_split_kv4(a, s);
    else attn_split_kernel<256><<<dim3(a.n_kv_heads, a.max_splits, a.nt), 256, 0, s>>>(a);
    attn_combine_kernel<256><<<dim3(a.n_heads, a.nt), 256, 0, s>>>(a, a.max_splits);
}

}  // namespace qv::cuda
