#pragma once
#include "../common.cuh"

namespace qv::cuda {

struct AttnPrepArgs {
    const float* qg;        // q_proj output [Hq][2*D] = per head [q | gate]
    const float* k;         // [Hkv*D]
    const float* v;         // [Hkv*D]
    const float* q_norm;    // [D] (1+w)
    const float* k_norm;    // [D] (1+w)
    const float* inv_freq;  // [rope_dim/2]
    float* q_out;           // [Hq*D]
    float* k_cache;         // [Hkv][max_ctx][D]
    float* v_cache;
    int n_heads, n_kv_heads, head_dim, rope_dim, max_ctx, pos;
    float eps;
};

struct AttnDecodeArgs {
    const float* q;  // prepared q [Hq*D]
    const float* qg; // for the gate half
    const float* k_cache;
    const float* v_cache;
    float* scores;   // scratch [Hq][max_ctx]
    float* out;      // [Hq*D]
    int n_heads, n_kv_heads, max_ctx, pos;
};

void attn_prep(const AttnPrepArgs& a, cudaStream_t s);
void attn_decode(const AttnDecodeArgs& a, cudaStream_t s);

}  // namespace qv::cuda
