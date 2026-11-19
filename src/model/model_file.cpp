#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../core/json.h"
#include "qv/model.h"

namespace qv {

const TensorInfo& ModelFile::at(const std::string& n) const {
    auto it = tensors.find(n);
    if (it == tensors.end()) throw std::runtime_error("tensor not found: " + n);
    return it->second;
}

ModelFile ModelFile::open(const std::string& prefix) {
    std::ifstream f(prefix + ".qv.json");
    if (!f) throw std::runtime_error("cannot open " + prefix + ".qv.json");
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value root = json::parse(ss.str());
    if (root["format"].str() != "qvolta-v1") throw std::runtime_error("bad format tag");
    ModelFile m;
    const auto& c = root["config"];
    Config& g = m.cfg;
    g.hidden = (int)c["hidden_size"].i64();
    g.inter = (int)c["intermediate_size"].i64();
    g.n_layers = (int)c["num_hidden_layers"].i64();
    g.vocab = (int)c["vocab_size"].i64();
    g.n_heads = (int)c["num_attention_heads"].i64();
    g.n_kv_heads = (int)c["num_key_value_heads"].i64();
    g.head_dim = (int)c["head_dim"].i64();
    const auto& rp = c["rope_parameters"];
    g.rope_theta = rp["rope_theta"].num();
    g.rope_dim = (int)(g.head_dim * rp["partial_rotary_factor"].num());
    g.lin_k_heads = (int)c["linear_num_key_heads"].i64();
    g.lin_v_heads = (int)c["linear_num_value_heads"].i64();
    g.lin_k_dim = (int)c["linear_key_head_dim"].i64();
    g.lin_v_dim = (int)c["linear_value_head_dim"].i64();
    g.conv_k = (int)c["linear_conv_kernel_dim"].i64();
    g.eps = (float)c["rms_norm_eps"].num();
    g.eos = (int)c["eos_token_id"].i64();
    for (const auto& t : c["layer_types"].a) g.is_attn.push_back(t.str() == "full_attention");
    if ((int)g.is_attn.size() != g.n_layers) throw std::runtime_error("layer_types size mismatch");

    for (const auto& [name, v] : root["tensors"].o) {
        TensorInfo ti;
        ti.name = name;
        const std::string& k = v["kind"].str();
        for (const auto& d : v["shape"].a) ti.shape.push_back(d.i64());
        if (k == "q") {
            ti.kind = Kind::Quant;
            ti.bits = (int)v["bits"].i64();
            ti.group = (int)v["group"].i64();
            ti.off_q = v["off_q"].i64();
            ti.nbytes_q = v["nbytes_q"].i64();
            ti.off_sm = v["off_sm"].i64();
            ti.nbytes_sm = v["nbytes_sm"].i64();
        } else {
            ti.kind = k == "f32" ? Kind::F32 : Kind::F16;
            ti.off = v["off"].i64();
            ti.nbytes = v["nbytes"].i64();
        }
        m.tensors.emplace(name, std::move(ti));
    }
    m.bin_path = prefix + ".qv.bin";
    m.bin_size = std::filesystem::file_size(m.bin_path);
    return m;
}

}  // namespace qv
