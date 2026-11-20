#pragma once
// Batched prefill attention for the gated GQA layers; semantically identical to running
// attn_prep + attn_decode (attn.h) for positions pos0..pos0+T-1 in order.
#include <cuda_fp16.h>
#include "../common.cuh"
#include "kv4.h"

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
    // int4 KV: post-RoPE K and V of the chunk are written as fp16 to stage_k/stage_v [T][Hkv*D] (instead of the fp caches),
    // then completed blocks are quantized (kv4.h). The ring is updated by attn_seq() after the chunk's attention.
    Kv4Layer kv4{};
    __half *stage_k = nullptr, *stage_v = nullptr;
};

struct AttnSeqArgs {
    const float* q;        // [T][Hq*D] prepared
    const float* qg;       // [T][Hq*2D], gate half is used
    const float* k_cache;
    const float* v_cache;
    __half* out_h;         // [T][Hq*D] fp16 clamped to +-65504 (may be null)
    float* out_f;          // [T][Hq*D] fp32 (may be null)
    int n_heads, n_kv_heads, max_ctx, pos0, T;
    Kv4Layer kv4{};                         // kv4.blocks != nullptr: read the int4 KV (+ fp16 window), k_cache/v_cache unused
    const __half *stage_k = nullptr, *stage_v = nullptr;
};

void attn_prep_seq(const AttnSeqPrepArgs& a, cudaStream_t s);
// attn_seq dispatches to the tensor-core kernel (attn_seq_tc.cu) unless env QV_ATTN=fp32 (fp32 CUDA-core reference kernel).
void attn_seq(const AttnSeqArgs& a, cudaStream_t s);
void attn_seq_fp32(const AttnSeqArgs& a, cudaStream_t s);
void attn_seq_tc(const AttnSeqArgs& a, cudaStream_t s);

}  // namespace qv::cuda
