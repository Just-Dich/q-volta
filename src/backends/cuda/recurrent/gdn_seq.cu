// Batched Gated-DeltaNet prefill kernels. Per-token math/op order mirrors gdn.cu (conv_step / gdn_step).
#include "../common.cuh"
#include "gdn_seq.h"

namespace qv::cuda {

// ---- conv_seq: one thread per (channel, token chunk) -------------------------------------------------
constexpr int CONV_CHUNK = 32;

__global__ void conv_seq_kernel(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ state,
                                float* __restrict__ out, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int t0 = blockIdx.y * CONV_CHUNK;
    const int t1 = min(T, t0 + CONV_CHUNK);
    const float4 wk = reinterpret_cast<const float4*>(w)[c];
    // history x[t0-3], x[t0-2], x[t0-1] (from the carried state for negative indices)
    float h[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        int ti = t0 - 3 + j;
        h[j] = ti >= 0 ? in[(size_t)ti * C + c] : state[c * 3 + (ti + 3)];
    }
    float s0 = h[0], s1 = h[1], s2 = h[2];
    float xn = in[(size_t)t0 * C + c];
#pragma unroll 8
    for (int t = t0; t < t1; ++t) {
        float x = xn;
        if (t + 1 < t1) xn = in[(size_t)(t + 1) * C + c];
        float acc = wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * x;
        out[(size_t)t * C + c] = silu(acc);
        s0 = s1; s1 = s2; s2 = x;
    }
}

// Final state = last 3 inputs (or shifted old state when T < 3). Runs after conv_seq_kernel completes.
__global__ void conv_seq_state_kernel(const float* __restrict__ in, float* __restrict__ state, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float o[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        int ti = T - 3 + j;
        o[j] = ti >= 0 ? in[(size_t)ti * C + c] : 0.f;
    }
    if (T < 3) {
        // old state shifts left by T: new[j] = (j+T<3) ? old[j+T] : in[j+T-3]
        float old[3] = {state[c * 3], state[c * 3 + 1], state[c * 3 + 2]};
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            int src = j + T;
            o[j] = src < 3 ? old[src] : in[(size_t)(src - 3) * C + c];
        }
    }
    state[c * 3] = o[0]; state[c * 3 + 1] = o[1]; state[c * 3 + 2] = o[2];
}

void conv_seq(const float* in, const float* w, float* state, float* out, int C, int T, cudaStream_t s) {
    if (T <= 0) return;
    dim3 grid((C + 127) / 128, (T + CONV_CHUNK - 1) / CONV_CHUNK);
    conv_seq_kernel<<<grid, 128, 0, s>>>(in, w, state, out, C, T);
    conv_seq_state_kernel<<<(C + 127) / 128, 128, 0, s>>>(in, state, C, T);
}

// ---- gdn_seq ------------------------------------------------------------------------------------------
__device__ __forceinline__ float softplus_s(float x) { return x > 20.f ? x : log1pf(expf(x)); }

struct Tok {
    float v, z, al, bl;
};

// Sum of 4 per-warp partials in the same tree order as block_sum's final warp_sum (lanes 0..3, rest zero).
__device__ __forceinline__ float sum4(const float* p) { return (p[0] + p[2]) + (p[1] + p[3]); }

// One block per value head, 128 threads (one per value column / key row for loads). State in registers.
// Pipeline (one __syncthreads per token): the block reduction for token t's output RMSNorm also carries
// the q/k sum-of-squares of token t+2; the normalised q/k for t+2 are written right after that sync into the
// smem buffer token t just finished with, and become visible to everyone at token t+1's sync.
template <int DK, int DV>
__global__ void __launch_bounds__(128, 1) gdn_seq_kernel(GdnSeqArgs a) {
    static_assert(DK == 128 && DV == 128, "");
    __shared__ __align__(16) float qs[2][DK];
    __shared__ __align__(16) float ks[2][DK];
    __shared__ float red[2][3][4];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int T = a.T;
    const int Hv = a.n_v_heads;
    const int kh = h / (Hv / a.n_k_heads);
    const int Cc = 2 * a.n_k_heads * DK + Hv * DV;
    const float* qin = a.conv_out + kh * DK + tid;
    const float* kin = a.conv_out + a.n_k_heads * DK + kh * DK + tid;
    const float* vin = a.conv_out + 2 * a.n_k_heads * DK + h * DV + tid;
    const float* zin = a.z + h * DV + tid;
    const int HD = Hv * DV;
    const float rsDK = rsqrtf((float)DK);

    auto ldq = [&](int t) { return t < T ? qin[(size_t)t * Cc] : 0.f; };
    auto ldk = [&](int t) { return t < T ? kin[(size_t)t * Cc] : 0.f; };
    auto ldt = [&](int t) {
        Tok r{0.f, 0.f, 0.f, 0.f};
        if (t < T) {
            r.v = vin[(size_t)t * Cc];
            r.z = zin[(size_t)t * HD];
            r.al = a.alpha[(size_t)t * Hv + h];
            r.bl = a.beta_logit[(size_t)t * Hv + h];
        }
        return r;
    };

    const float A_neg = a.A_neg[h], dtb = a.dt_bias[h], nw = a.norm_w[tid];

    float* S = a.state + (size_t)h * DK * DV + tid;
    float s[DK];
#pragma unroll
    for (int k = 0; k < DK; ++k) s[k] = S[k * DV];

    // prologue: normalise q/k of tokens 0 and 1 with a single reduction
    {
        float q0 = ldq(0), k0 = ldk(0), q1 = ldq(1), k1 = ldk(1);
        float p0 = warp_sum(q0 * q0), p1 = warp_sum(k0 * k0), p2 = warp_sum(q1 * q1), p3 = warp_sum(k1 * k1);
        __shared__ float pro[4][4];
        if (lane == 0) { pro[0][wid] = p0; pro[1][wid] = p1; pro[2][wid] = p2; pro[3][wid] = p3; }
        __syncthreads();
        qs[0][tid] = q0 * rsqrtf(sum4(pro[0]) + 1e-6f) * rsDK;
        ks[0][tid] = k0 * rsqrtf(sum4(pro[1]) + 1e-6f);
        qs[1][tid] = q1 * rsqrtf(sum4(pro[2]) + 1e-6f) * rsDK;
        ks[1][tid] = k1 * rsqrtf(sum4(pro[3]) + 1e-6f);
        __syncthreads();
    }
    float rqA = ldq(2), rkA = ldk(2), rqB = ldq(3), rkB = ldk(3);
    Tok c0 = ldt(0), c1 = ldt(1);

    for (int t = 0; t < T; ++t) {
        const int b = t & 1;
        const float g = A_neg * softplus_s(c0.al + dtb);
        const float decay = expf(g);
        const float beta = 1.f / (1.f + expf(-c0.bl));
        const float4* k4 = reinterpret_cast<const float4*>(ks[b]);
        const float4* q4 = reinterpret_cast<const float4*>(qs[b]);

        float kva[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k4i = 0; k4i < DK / 4; ++k4i) {
            float4 kk = k4[k4i];
            s[4 * k4i + 0] *= decay; kva[0] = fmaf(s[4 * k4i + 0], kk.x, kva[0]);
            s[4 * k4i + 1] *= decay; kva[1] = fmaf(s[4 * k4i + 1], kk.y, kva[1]);
            s[4 * k4i + 2] *= decay; kva[2] = fmaf(s[4 * k4i + 2], kk.z, kva[2]);
            s[4 * k4i + 3] *= decay; kva[3] = fmaf(s[4 * k4i + 3], kk.w, kva[3]);
        }
        asm volatile("" ::: "memory");  // stop the compiler from keeping all of ks in registers across loops
        const float kvmem = (kva[0] + kva[1]) + (kva[2] + kva[3]);
        const float delta = (c0.v - kvmem) * beta;
        float oa[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k4i = 0; k4i < DK / 4; ++k4i) {
            float4 kk = k4[k4i];
            float4 qq = q4[k4i];
            s[4 * k4i + 0] = fmaf(kk.x, delta, s[4 * k4i + 0]); oa[0] = fmaf(s[4 * k4i + 0], qq.x, oa[0]);
            s[4 * k4i + 1] = fmaf(kk.y, delta, s[4 * k4i + 1]); oa[1] = fmaf(s[4 * k4i + 1], qq.y, oa[1]);
            s[4 * k4i + 2] = fmaf(kk.z, delta, s[4 * k4i + 2]); oa[2] = fmaf(s[4 * k4i + 2], qq.z, oa[2]);
            s[4 * k4i + 3] = fmaf(kk.w, delta, s[4 * k4i + 3]); oa[3] = fmaf(s[4 * k4i + 3], qq.w, oa[3]);
        }

        const float o = (oa[0] + oa[1]) + (oa[2] + oa[3]);
        // one fused block reduction: o^2 (token t) and q/k sum-of-squares (token t+2)
        float p0 = warp_sum(o * o), p1 = warp_sum(rqA * rqA), p2 = warp_sum(rkA * rkA);
        if (lane == 0) { red[b][0][wid] = p0; red[b][1][wid] = p1; red[b][2][wid] = p2; }
        __syncthreads();
        const float oss = sum4(red[b][0]);
        if (t + 2 < T) {
            qs[b][tid] = rqA * rsqrtf(sum4(red[b][1]) + 1e-6f) * rsDK;
            ks[b][tid] = rkA * rsqrtf(sum4(red[b][2]) + 1e-6f);
        }
        const float r = rsqrtf(oss / DV + a.eps);
        const float y = (o * r) * nw * silu(c0.z);
        const size_t oi = (size_t)t * HD + h * DV + tid;
        if (a.out_f) a.out_f[oi] = y;
        if (a.out_h) a.out_h[oi] = __float2half(fminf(fmaxf(y, -65504.f), 65504.f));

        // advance prefetch pipeline
        rqA = rqB; rkA = rkB;
        rqB = ldq(t + 4); rkB = ldk(t + 4);
        c0 = c1; c1 = ldt(t + 2);
    }

#pragma unroll
    for (int k = 0; k < DK; ++k) S[k * DV] = s[k];
}

void gdn_seq(const GdnSeqArgs& a, cudaStream_t s) {
    if (a.dk != 128 || a.dv != 128) {
        std::fprintf(stderr, "gdn_seq: only dk=dv=128 compiled\n");
        std::abort();
    }
    if (a.T <= 0) return;
    gdn_seq_kernel<128, 128><<<a.n_v_heads, 128, 0, s>>>(a);
}

}  // namespace qv::cuda
