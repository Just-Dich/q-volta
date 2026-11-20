#pragma once
// Batched prefill attention for the gated GQA layers; semantically identical to running
// attn_prep + attn_decode (attn.h) for positions pos0..pos0+T-1 in order.
#include <cuda_fp16.h>
#include "../common.cuh"

namespace qv::cuda {

struct AttnSeqPrepArgs {
    const float* qg;       // [T][Hq*2D], per head [q | gate]
    const float* k;        // [T][Hkv*D]
    const float* v;        // [T][Hkv*D]
    const float* q_norm;   // [D] (1+w)
    const float* k_norm;   // [D] (1+w)
    const float* inv_freq; // [rope_dim/2]
    float* q_out;          // [T][Hq*D]
    float* k_cache;        // [Hkv][max_ctx][D]
    float* v_cache;
    int n_heads, n_kv_heads, head_dim, rope_dim, max_ctx, pos0, T;
    float eps;
};

struct AttnSeqArgs {
    const float* q;        // [T][Hq*D] prepared
    const float* qg;       // [T][Hq*2D], gate half is used
    const float* k_cache;
    const float* v_cache;
    __half* out_h;         // [T][Hq*D] fp16 clamped to +-65504 (may be null)
    float* out_f;          // [T][Hq*D] fp32 (may be null)
    int n_heads, n_kv_heads, max_ctx, pos0, T;
};

void attn_prep_seq(const AttnSeqPrepArgs& a, cudaStream_t s);
void attn_seq(const AttnSeqArgs& a, cudaStream_t s);

}  // namespace qv::cuda
