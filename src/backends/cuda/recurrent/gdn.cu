// Gated DeltaNet single-token decode step (Qwen3.5 / Qwen3.8 linear-attention layers).
// Semantics follow transformers torch_recurrent_gated_delta_rule + Qwen3_5RMSNormGated exactly.
#include "../common.cuh"
#include "gdn.h"
#include "../kernels/q8.cuh"

namespace qv::cuda {

// Depthwise causal conv (kernel KC=4) + SiLU, with rolling state [C][KC-1] (oldest first).
__global__ void conv_step_kernel(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ state,
                                 float* __restrict__ out, int C, float* __restrict__ snap) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    float x = in[c];
    const float4 wk = reinterpret_cast<const float4*>(w)[c];
    float acc = wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * x;
    out[c] = silu(acc);
    state[c * 3 + 0] = s1;
    state[c * 3 + 1] = s2;
    state[c * 3 + 2] = x;
    if (snap) { snap[c * 3 + 0] = s1; snap[c * 3 + 1] = s2; snap[c * 3 + 2] = x; }  // MTP rollback copy
}

void conv_step(const float* in, const float* w, float* state, float* out, int C, cudaStream_t s, float* snap) {
    conv_step_kernel<<<(C + 255) / 256, 256, 0, s>>>(in, w, state, out, C, snap);
}

// nt token rows in order (state carried in registers); snap (optional, stride snap_stride floats) gets the state after
// tokens 0..nt-2. in/out: [nt][C].
__global__ void conv_step_multi_kernel(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ state,
                                       float* __restrict__ out, int C, int nt, float* __restrict__ snap, size_t snap_stride) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const float4 wk = reinterpret_cast<const float4*>(w)[c];
    for (int t = 0; t < nt; ++t) {
        const float x = in[(size_t)t * C + c];
        const float acc = wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * x;
        out[(size_t)t * C + c] = silu(acc);
        s0 = s1; s1 = s2; s2 = x;
        if (snap && t < nt - 1) {
            float* sp = snap + t * snap_stride;
            sp[c * 3 + 0] = s0; sp[c * 3 + 1] = s1; sp[c * 3 + 2] = s2;
        }
    }
    state[c * 3 + 0] = s0; state[c * 3 + 1] = s1; state[c * 3 + 2] = s2;
}

void conv_step_multi(const float* in, const float* w, float* state, float* out, int C, int nt, float* snap,
                     size_t snap_stride, cudaStream_t s) {
    conv_step_multi_kernel<<<(C + 255) / 256, 256, 0, s>>>(in, w, state, out, C, nt, snap, snap_stride);
}

__device__ __forceinline__ float softplus_t(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// One block per value head, DV threads (one per value column). State layout [Hv][DK][DV] fp32.
// nt > 1 (MTP verify): the nt token rows are processed back to back with the state held in registers (bit-identical to
// nt separate launches: the state is fp32 either way); snap, if set, receives the state after tokens 0..nt-2.
template <int DK, int DV>
__global__ void __launch_bounds__(DV) gdn_step_kernel(GdnStepArgs a) {
    __shared__ float qs[DK], ks[DK];
    __shared__ float red[32];
    const int h = blockIdx.x, t = threadIdx.x;
    const int kh = h / (a.n_v_heads / a.n_k_heads);
    float* S = a.state + (size_t)h * DK * DV + t;
    float s[DK];
    for (int tk = 0; tk < a.nt; ++tk) {
        const float* conv = a.conv_out + (size_t)tk * a.conv_stride;
        const float* qin = conv + kh * DK;
        const float* kin = conv + a.n_k_heads * DK + kh * DK;
        const float* vin = conv + 2 * a.n_k_heads * DK + h * DV;

        // l2-normalise q and k (eps inside the sqrt, like FLA), then scale q by 1/sqrt(DK)
        float qv = (t < DK) ? qin[t] : 0.f, kv = (t < DK) ? kin[t] : 0.f;
        float qss = block_sum(qv * qv, red);
        float kss = block_sum(kv * kv, red);
        if (t < DK) {
            qs[t] = qv * rsqrtf(qss + 1e-6f) * rsqrtf((float)DK);
            ks[t] = kv * rsqrtf(kss + 1e-6f);
        }
        __syncthreads();

        const float g = a.A_neg[h] * softplus_t(a.alpha[(size_t)tk * a.n_v_heads + h] + a.dt_bias[h]);
        const float decay = expf(g);
        const float beta = 1.f / (1.f + expf(-a.beta_logit[(size_t)tk * a.n_v_heads + h]));

        float kvmem = 0.f;
#pragma unroll
        for (int k = 0; k < DK; ++k) {
            if (tk == 0) s[k] = S[k * DV];
            s[k] = s[k] * decay;
            kvmem = fmaf(s[k], ks[k], kvmem);
        }
        const float delta = (vin[t] - kvmem) * beta;
        float o = 0.f;
        const bool live = tk == a.nt - 1;
        float* sn = (a.snap && !live) ? a.snap + (size_t)tk * a.snap_stride + (size_t)h * DK * DV + t : nullptr;
#pragma unroll
        for (int k = 0; k < DK; ++k) {
            s[k] = fmaf(ks[k], delta, s[k]);
            if (live) S[k * DV] = s[k];
            if (sn) sn[k * DV] = s[k];
            o = fmaf(s[k], qs[k], o);
        }

        // gated RMSNorm over the head:  (o * rsqrt(mean(o^2)+eps)) * w * silu(z)
        float oss = block_sum(o * o, red);
        float r = rsqrtf(oss / DV + a.eps);
        float z = a.z[(size_t)tk * a.n_v_heads * DV + h * DV + t];
        const float y = (o * r) * a.norm_w[t] * silu(z);
        a.out[(size_t)tk * a.out_stride + h * DV + t] = y;
        if (a.q8.q) q8_store_chunk(y, tk * (a.n_v_heads * DV / 32) + ((h * DV + t) >> 5), a.q8);
    }
}

void gdn_step(const GdnStepArgs& a, cudaStream_t s) {
    if (a.dk != 128 || a.dv != 128) {
        std::fprintf(stderr, "gdn_step: only dk=dv=128 compiled\n");
        std::abort();
    }
    gdn_step_kernel<128, 128><<<a.n_v_heads, 128, 0, s>>>(a);
}

}  // namespace qv::cuda
