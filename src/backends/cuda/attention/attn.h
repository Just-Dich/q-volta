#pragma once
#include "../common.cuh"
#include "../kernels/gemv.h"

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
    int n_heads, n_kv_heads, head_dim, rope_dim, max_ctx;
    const int* pos;  // device scalar (graph-capturable); row y of an nt-row launch uses pos[y]
    float eps;
    // nt > 1 (MTP verify): nt rows in one launch, row y reads/writes at +y*stride (floats); per-row positions pos[y]
    int nt = 1, qg_stride = 0, k_stride = 0, q_stride = 0;
};

struct AttnDecodeArgs {
    const float* q;  // prepared q [Hq*D]
    const float* qg; // for the gate half
    const float* k_cache;
    const float* v_cache;
    float* scores;   // scratch [Hq][max_ctx]
    float* out;      // [Hq*D]
    int n_heads, n_kv_heads, max_ctx;
    const int* pos;  // device scalar
    Q8Buf q8;        // if q8.q: also emit the o_proj int8 input
    // Split-K (flash-decoding) scratch. nullptr -> falls back to the single-pass reference kernel (attn_decode_ref).
    // Size: n_heads * max_splits * (head_dim + 2) floats (= 24*80*258*4 = 1,981,440 B for Qwen3.8-27B).
    // Layout: [Hq][S][D] partial accumulators followed by [Hq][S][2] (m, l). No zero-init and no counters needed.
    float* part = nullptr;
    int max_splits = 80;  // number of key splits launched per kv head (fixed => graph-capturable); must match scratch
    // nt > 1: nt rows in one launch (row z: q/qg/out + z*stride, pos[z], part + z*part_stride, q8 rows of q8_k elements)
    int nt = 1, q_stride = 0, qg_stride = 0, out_stride = 0, q8_k = 0;
    size_t part_stride = 0;
};

constexpr int ATTN_MAX_SPLITS = 80;  // default; total CTAs = n_kv_heads * max_splits

void attn_prep(const AttnPrepArgs& a, cudaStream_t s);  // for nt > 1 all rows' K/V are written before any row attends
void attn_decode(const AttnDecodeArgs& a, cudaStream_t s);      // split-K if a.part, else reference
void attn_decode_ref(const AttnDecodeArgs& a, cudaStream_t s);  // original one-block-per-head kernel (uses a.scores)

}  // namespace qv::cuda
