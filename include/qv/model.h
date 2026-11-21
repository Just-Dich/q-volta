// Q-Volta model description (backend-independent).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qv {

// Inkling-2-Mini (arch "inkling2_loop") description, filled from config["inkling"] of the .qv header (docs/INKLING_FORMAT.md).
struct InklingConfig {
    int unpadded_vocab = 0;                              // logits / embedding rows actually stored (vocab_size is the padded size)
    int loop_start = 0, loop_end = 0, loop_passes = 1;   // layers [loop_start, loop_end) run loop_passes times
    int loop_gate_rank = 0;
    std::vector<int> global_layers, local_layers;        // full-attention layer ids / sliding-window layer ids
    int window = 0, d_rel = 0, rel_extent = 0;
    float qk_l2_clip = 0, rel_l2_clip = 0;
    int log_scaling_n_floor = 0;
    float log_scaling_alpha = 0;
    int attn_res_block = 0, sconv_kernel = 0;
    int n_experts = 0, topk = 0, n_shared_experts = 0, expert_ff = 0;
    float route_scale = 1, gate_softcap = 0, up_softcap = 0, final_logit_softcap = 0;
    bool w13_interleaved = true;                         // expert w13 rows: even = gate, odd = up
    int64_t max_position = 0;
    bool is_global(int layer) const;
    bool in_loop(int layer) const { return layer >= loop_start && layer < loop_end; }
    int passes(int layer) const { return in_loop(layer) ? loop_passes : 1; }
    int n_applications(int n_layers) const { return n_layers + (loop_passes - 1) * (loop_end - loop_start); }
};

struct Config {
    std::string arch = "qwen3_5";  // "inkling2_loop" for Inkling-2-Mini files
    InklingConfig ink;             // valid when arch == "inkling2_loop"
    bool is_inkling() const { return arch == "inkling2_loop"; }
    int hidden = 0, inter = 0, n_layers = 0, vocab = 0;
    int n_heads = 0, n_kv_heads = 0, head_dim = 0;  // full attention
    int rope_dim = 0;                               // partial rotary dims
    double rope_theta = 0;
    int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_k = 0;
    float eps = 1e-6f;
    std::vector<bool> is_attn;  // per layer: true = full attention, false = gated deltanet
    int eos = -1;
};

// NVFP4 / MXFP8 = checkpoint-native passthrough expert tensors (off_q = codes, off_sm = block scales, off_rs = per-expert fp32 factor, NVFP4 only)
enum class Kind { Quant, F32, F16, NVFP4, MXFP8 };

struct TensorInfo {
    std::string name;
    Kind kind = Kind::F32;
    int bits = 0, group = 0;
    std::vector<int64_t> shape;
    uint64_t off = 0, nbytes = 0;        // dense
    uint64_t off_q = 0, nbytes_q = 0;    // quant words
    uint64_t off_sm = 0, nbytes_sm = 0;  // quant (scale,min) fp16 pairs, or meta8: uint16 (s8 | z8<<8)
    bool meta8 = false;                  // meta8 encoding: + fp16 row scales at off_rs (see tools/qv_quant.py)
    uint64_t off_rs = 0, nbytes_rs = 0;
    // Stacked expert tensors have shape [E, N, K]: the per-expert slices are contiguous, equally sized and 256-byte aligned.
    bool stacked() const { return shape.size() == 3; }
    int64_t n_experts() const { return stacked() ? shape[0] : 1; }
    uint64_t expert_off_q(int64_t e) const { return off_q + e * (nbytes_q / n_experts()); }
    uint64_t expert_off_sm(int64_t e) const { return off_sm + e * (nbytes_sm / n_experts()); }
    uint64_t expert_off_rs(int64_t e) const { return off_rs + e * (nbytes_rs / n_experts()); }
};

struct ModelFile {
    Config cfg;
    std::map<std::string, TensorInfo> tensors;
    std::string bin_path;
    uint64_t bin_size = 0;
    static ModelFile open(const std::string& prefix);  // <prefix>.qv (single file) or legacy <prefix>.qv.json + .qv.bin
    const TensorInfo& at(const std::string& n) const;
    // Inkling: throws std::runtime_error naming the first missing / mis-shaped tensor (full tensor table check from cfg.ink).
    void validate_inkling() const;
};

}  // namespace qv
