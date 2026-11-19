// Gated DeltaNet single-token decode step (Qwen3.5 / Qwen3.8 linear-attention layers).
// Semantics follow transformers torch_recurrent_gated_delta_rule + Qwen3_5RMSNormGated exactly.
#include "../common.cuh"
#include "gdn.h"

namespace qv::cuda {

// Depthwise causal conv (kernel KC=4) + SiLU, with rolling state [C][KC-1] (oldest first).
__global__ void conv_step_kernel(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ state,
                                 float* __restrict__ out, int C) {
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
}

void conv_step(const float* in, const float* w, float* state, float* out, int C, cudaStream_t s) {
    conv_step_kernel<<<(C + 255) / 256, 256, 0, s>>>(in, w, state, out, C);
}

__device__ __forceinline__ float softplus_t(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// One block per value head, DV threads (one per value column). State layout [Hv][DK][DV] fp32.
template <int DK, int DV>
__global__ void __launch_bounds__(DV) gdn_step_kernel(GdnStepArgs a) {
    __shared__ float qs[DK], ks[DK];
    __shared__ float red[32];
    const int h = blockIdx.x, t = threadIdx.x;
    const int kh = h / (a.n_v_heads / a.n_k_heads);
    const float* qin = a.conv_out + kh * DK;
    const float* kin = a.conv_out + a.n_k_heads * DK + kh * DK;
    const float* vin = a.conv_out + 2 * a.n_k_heads * DK + h * DV;

    // l2-normalise q and k (eps inside the sqrt, like FLA), then scale q by 1/sqrt(DK)
    float qv = (t < DK) ? qin[t] : 0.f, kv = (t < DK) ? kin[t] : 0.f;
    float qss = block_sum(qv * qv, red);
    float kss = block_sum(kv * kv, red);
    if (t < DK) {
        qs[t] = qv * rsqrtf(qss + 1e-6f) * rsqrtf((float)DK);
        ks[t] = kv * rsqrtf(kss + 1e-6f);
    }
    __syncthreads();

    const float g = a.A_neg[h] * softplus_t(a.alpha[h] + a.dt_bias[h]);
    const float decay = expf(g);
    const float beta = 1.f / (1.f + expf(-a.beta_logit[h]));

    float* S = a.state + (size_t)h * DK * DV + t;
    float s[DK];
    float kvmem = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
        s[k] = S[k * DV] * decay;
        kvmem = fmaf(s[k], ks[k], kvmem);
    }
    const float delta = (vin[t] - kvmem) * beta;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
        s[k] = fmaf(ks[k], delta, s[k]);
        S[k * DV] = s[k];
        o = fmaf(s[k], qs[k], o);
    }

    // gated RMSNorm over the head:  (o * rsqrt(mean(o^2)+eps)) * w * silu(z)
    float oss = block_sum(o * o, red);
    float r = rsqrtf(oss / DV + a.eps);
    float z = a.z[h * DV + t];
    a.out[h * DV + t] = (o * r) * a.norm_w[t] * silu(z);
}

void gdn_step(const GdnStepArgs& a, cudaStream_t s) {
    if (a.dk != 128 || a.dv != 128) {
        std::fprintf(stderr, "gdn_step: only dk=dv=128 compiled\n");
        std::abort();
    }
    gdn_step_kernel<128, 128><<<a.n_v_heads, 128, 0, s>>>(a);
}

}  // namespace qv::cuda
