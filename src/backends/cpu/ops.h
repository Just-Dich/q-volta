// Q-Volta CPU decode ops (fp32, AVX2+FMA+F16C). Original implementation mirroring the semantics of the CUDA
// kernels in src/backends/cuda/{recurrent,attention,kernels}. Every kernel is single-threaded and RANGE-based:
// the engine splits [begin,end) (heads / channels / rows / work items) across its own thread pool. No kernel
// allocates, takes locks or touches shared state, so disjoint ranges may run concurrently.
#pragma once
#include <cstddef>
#include <cstdint>

namespace qv::cpu {

// ---- scalar math helpers (accurate fp32, ~1 ulp; see test_cpu_ops for the measured error) -------------------------
float exp_fast(float x);      // vectorised-kernel exp, max rel err ~2e-7 on [-87,88]; underflows to 0 below -87.3
float silu_f(float x);        // x / (1 + exp(-x))
float softplus_f(float x);    // x > 20 ? x : log1p(exp(x))   (same threshold as the GPU kernel)

// ---- elementwise (ranges are element indices) ---------------------------------------------------------------------
void silu_mul_range(const float* g, const float* u, float* out, int begin, int end);  // out = silu(g) * u
void add_range(const float* a, const float* b, float* out, int begin, int end);       // out = a + b (aliasing ok)
void silu_range(const float* x, float* out, int begin, int end);                      // out = silu(x)

// ---- RMSNorm: out = x * rsqrt(mean(x^2) + eps) * w   (w already holds 1+w for zero-centred norms) -------------------
double rmsnorm_sumsq_range(const float* x, int begin, int end);  // partial sum of squares (sum partials, then scale)
float rmsnorm_scale(double sumsq, int n, float eps);              // rsqrt(sumsq / n + eps)
void rmsnorm_apply_range(const float* x, const float* w, float* out, float r, int begin, int end);
void rmsnorm(const float* x, const float* w, float* out, int n, float eps);  // whole vector, one thread

// ---- Gated DeltaNet --------------------------------------------------------------------------------------------------
// Causal depthwise conv (k=4) + SiLU. state [C][3] oldest first (updated), w [C][4], in/out [C]. Range = channels.
void conv_step_range(const float* in, const float* w, float* state, float* out, int c_begin, int c_end);

struct GdnArgs {
    const float* conv_out;    // [q(Hk*dk) | k(Hk*dk) | v(Hv*dv)] after conv+silu
    const float* alpha;       // in_proj_a output [Hv]
    const float* beta_logit;  // in_proj_b output [Hv]
    const float* A_neg;       // -exp(A_log) [Hv]
    const float* dt_bias;     // [Hv]
    const float* z;           // in_proj_z output [Hv*dv]
    const float* norm_w;      // [dv]
    float* state;             // [Hv][dk][dv] fp32, updated in place
    float* out;               // [Hv*dv]
    int n_k_heads, n_v_heads, dk, dv;  // dk == dv == 128 only (like the GPU kernel)
    float eps;
};
// Full per-head step (l2norm q/k, decay, delta rule, readout, gated rmsnorm) for v-heads [h_begin, h_end).
void gdn_step_range(const GdnArgs& a, int h_begin, int h_end);
// Standalone gated per-head rmsnorm: out = (o * rsqrt(mean(o^2)+eps)) * w * silu(z); o,z,out [heads*dv]. Range = heads.
void gated_rmsnorm_range(const float* o, const float* z, const float* w, float* out, int dv, float eps, int h_begin,
                         int h_end);

// ---- Gated GQA attention, single-token decode -------------------------------------------------------------------------
struct AttnPrepArgs {
    const float* qg;        // [Hq][2*D] = per head [q | gate]
    const float* k;         // [Hkv*D]
    const float* v;         // [Hkv*D]
    const float* q_norm;    // [D] (1+w)
    const float* k_norm;    // [D] (1+w)
    const float* rope_cs;   // [2][rope_dim/2] cos then sin for this position (attn_rope_table)
    float* q_out;           // [Hq*D]
    float* k_cache;         // [Hkv][max_ctx][D]
    float* v_cache;
    int n_heads, n_kv_heads, head_dim, rope_dim, max_ctx, pos;
    float eps;
};
// cos/sin(pos * inv_freq[i]) in fp32 like the GPU (cosf/sinf of the fp32 product). cs has rope_dim entries.
void attn_rope_table(int pos, const float* inv_freq, int rope_dim, float* cs);
// Items [0,Hq) prepare q heads, [Hq,Hq+Hkv) prepare k heads (+ copy v) into the cache at `pos`. Range = items.
void attn_prep_range(const AttnPrepArgs& a, int item_begin, int item_end);

struct AttnDecodeArgs {
    const float* q;        // prepared q [Hq*D]
    const float* qg;       // for the gate half ([Hq][2*D])
    const float* k_cache;  // [Hkv][max_ctx][D]
    const float* v_cache;
    float* part;           // scratch, attn_part_floats(...) floats: [Hq][S][D] acc then [Hq][S][2] (m, l)
    float* out;            // [Hq*D]
    int n_heads, n_kv_heads, head_dim, max_ctx;
    int n_keys;            // pos + 1
    int n_splits;          // S: number of key chunks per kv head
};
inline size_t attn_part_floats(int n_heads, int n_splits, int head_dim) {
    return (size_t)n_heads * n_splits * (head_dim + 2);
}
int attn_chunk(int n_keys, int n_splits);    // keys per split (>= 32)
int attn_active_splits(int n_keys, int n_splits);
// Work item w = kv_head * n_splits + split; online softmax over the split's keys for all Hq/Hkv q heads of that kv
// head. Writes partials (acc, m, l). Range = work items [0, Hkv*S).
void attn_decode_range(const AttnDecodeArgs& a, int item_begin, int item_end);
// Merge the partials of q heads [h_begin,h_end), divide by l, apply out *= sigmoid(gate).
void attn_merge_range(const AttnDecodeArgs& a, int h_begin, int h_end);

// ---- embedding row dequant (format: tools/qv_format.py) ----------------------------------------------------------------
struct QRows {
    const uint32_t* q;   // [N][K/32][bits] bit-plane words
    const uint16_t* sm;  // fp16 [N][K/64][2] (scale, min)
    int K, bits;         // K % 64 == 0, bits in {2,3,4,5,6,8}
};
void embed_row(const QRows& E, int row, float* out);

// ---- argmax over fp32 logits (first index wins ties; NaN never wins) ----------------------------------------------------
void argmax_range(const float* x, int begin, int end, float* best_val, int* best_idx);
void argmax_merge(const float* vals, const int* idxs, int m, int* out_idx);

}  // namespace qv::cpu
