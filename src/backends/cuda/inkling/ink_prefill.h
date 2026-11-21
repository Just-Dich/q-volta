// Inkling-2-Mini batched-prefill kernels (T tokens per chunk). See docs/INKLING_ENGINE.md.
#pragma once
#include <cublas_v2.h>

#include "../common.cuh"
#include "ink_ops.h"

namespace qv::cuda::ink {

constexpr int PS = 8704;  // projection row stride (floats): q 0, k 3072, v 4096, gate 5120, rel 8192, sink 8576, dwk 8600, dwv 8632

struct PrepTArgs {
    const float* proj;                                   // [T][PS]
    const float *qscale, *relscale, *relproj;            // relproj unused here (bias table is a GEMM)
    const float *kw, *vw;
    const float *kst, *vst;                              // conv state at chunk start [3][1024]
    int glob, T, pos0;
    __half* qh;      // [T][24][128] (softmax scale and tau folded in)
    __half* relh;    // [T][24][16]
    __half *kc, *vc; // per-kv-head rows: row = glob ? pos : (t + 1024) ; head stride `cap` rows
    int cap;
};
void attn_prep_t(const PrepTArgs& a, cudaStream_t s);

struct PfAttnArgs {
    const __half* qh;   // [Tpad][3072]
    const __half *kc, *vc;
    long kbase;         // key position j lives at row (j - kbase) of its head
    int cap;            // rows per kv head
    const __half* rl;   // [T*24][1024] bias by distance
    const float* proj;  // sink / gate rows
    int T, pos0, window;
    __half* out;        // [Tpad][3072] sigmoid(gate) * attention
};
void attn_prefill(const PfAttnArgs& a, cudaStream_t s);

void ring_to_stage(const __half* ring, __half* stage, int stage_rows, int pos0, cudaStream_t s);
void stage_to_ring(const __half* stage, __half* ring, int stage_rows, int pos0, int T, cudaStream_t s);

// MoE
void topk_t(const float* logits, const float* bias, int* ids, float* wts, int T, float route_scale, cudaStream_t s);
void moe_sort(const int* ids, int T, int* cnt, int* offs, int* fill, int* perm, int* inv, cudaStream_t s);  // offs[129] ready after
void moe_gather(const __half* x, const int* perm, int n, __half* out, cudaStream_t s);
void moe_act(const __half* y13, int n, float gcap, float ucap, __half* out, cudaStream_t s);
void moe_combine(const __half* y2, const int* inv, const float* wts, int T, float* out, cudaStream_t s);

// loop gates
void gate_cat(const float* h, const float* msg, int T, __half* xcat, cudaStream_t s);
void f16_to_f32(const __half* x, long n, float* y, cudaStream_t s);
void f32_to_f16(const float* x, long n, __half* y, cudaStream_t s);
void gate_apply_t(const float* f, const float* g, const float* h, const float* msg, long n, float* out, cudaStream_t s);
void copy_f32(const float* src, float* dst, long n, cudaStream_t s);

}  // namespace qv::cuda::ink
