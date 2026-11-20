#pragma once
#include "../common.cuh"
#include "../kernels/gemv.h"

namespace qv::cuda {

struct GdnStepArgs {
    const float* conv_out;    // [q(Hk*dk) | k(Hk*dk) | v(Hv*dv)] after conv+silu
    const float* alpha;       // in_proj_a output [Hv]
    const float* beta_logit;  // in_proj_b output [Hv]
    const float* A_neg;       // -exp(A_log) [Hv]
    const float* dt_bias;     // [Hv]
    const float* z;           // in_proj_z output [Hv*dv]
    const float* norm_w;      // [dv]
    float* state;             // [Hv][dk][dv] fp32, updated in place
    float* out;               // [Hv*dv]
    int n_k_heads, n_v_heads, dk, dv;
    float eps;
    Q8Buf q8;  // if q8.q: also emit the out_proj int8 input
    float* snap = nullptr;  // MTP verify (nt > 1): receives the state after tokens 0..nt-2 (token tk at + tk*snap_stride)
    int nt = 1;             // token rows processed back to back; conv_out/alpha/beta/z/out are token-major
    size_t snap_stride = 0; // floats between snapshots
    int conv_stride = 0;    // floats per conv_out row
    int out_stride = 0;     // floats per out row (q8 rows use Hv*dv/32 chunks)
};

void conv_step(const float* in, const float* w, float* state, float* out, int C, cudaStream_t s, float* snap = nullptr);
void conv_step_multi(const float* in, const float* w, float* state, float* out, int C, int nt, float* snap,
                     size_t snap_stride, cudaStream_t s);
void gdn_step(const GdnStepArgs& a, cudaStream_t s);

}  // namespace qv::cuda
