#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qv::cuda {

// Batched (multi-token) prefill versions of conv_step / gdn_step. Semantically identical to calling the
// single-token kernels T times in order.

// Depthwise causal conv (k=4) + SiLU over T tokens. in/out: [T][C]; w: [C][4]; state: [C][3] carried in/out.
void conv_seq(const float* in, const float* w, float* state, float* out, int C, int T, cudaStream_t s);

struct GdnSeqArgs {
    const float* conv_out;    // [T][Cc], Cc = 2*Hk*dk + Hv*dv, rows laid out as [q | k | v]
    const float* alpha;       // [T][Hv]
    const float* beta_logit;  // [T][Hv]
    const float* A_neg;       // [Hv]
    const float* dt_bias;     // [Hv]
    const float* z;           // [T][Hv*dv]
    const float* norm_w;      // [dv]
    float* state;             // [Hv][dk][dv] fp32 in/out
    __half* out_h;            // [T][Hv*dv], clamped to +-65504, may be null
    float* out_f;             // [T][Hv*dv], may be null
    int n_k_heads, n_v_heads, dk, dv, T;
    float eps;
};

// One block per value head; loops over all T tokens in-kernel with the state resident in registers.
void gdn_seq(const GdnSeqArgs& a, cudaStream_t s);

}  // namespace qv::cuda
