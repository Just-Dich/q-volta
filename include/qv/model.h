// Q-Volta model description (backend-independent).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qv {

struct Config {
    int hidden = 0, inter = 0, n_layers = 0, vocab = 0;
    int n_heads = 0, n_kv_heads = 0, head_dim = 0;  // full attention
    int rope_dim = 0;                               // partial rotary dims
    double rope_theta = 0;
    int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_k = 0;
    float eps = 1e-6f;
    std::vector<bool> is_attn;  // per layer: true = full attention, false = gated deltanet
    int eos = -1;
};

enum class Kind { Quant, F32, F16 };

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
};

struct ModelFile {
    Config cfg;
    std::map<std::string, TensorInfo> tensors;
    std::string bin_path;
    uint64_t bin_size = 0;
    static ModelFile open(const std::string& prefix);  // <prefix>.qv (single file) or legacy <prefix>.qv.json + .qv.bin
    const TensorInfo& at(const std::string& n) const;
};

}  // namespace qv
