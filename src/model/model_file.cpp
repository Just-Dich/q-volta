#include <cstring>
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

// Single-file container (.qv): "QVOLTA1\0" | u64 json_len | u64 data_off | json | zero pad | data (data_off is 2 MiB aligned).
// Tensor offsets inside the JSON are relative to the data section; they are rebased to absolute file offsets on load,
// so engines treat bin_path + offsets identically for the container and for the legacy .qv.json + .qv.bin pair.
ModelFile ModelFile::open(const std::string& prefix_in) {
    std::string prefix = prefix_in, path;
    uint64_t rebase = 0;
    std::string text;
    auto ends = [&](const std::string& s, const char* e) {
        size_t n = std::strlen(e);
        return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
    };
    std::string single = ends(prefix, ".qv") ? prefix : prefix + ".qv";
    if (std::filesystem::is_regular_file(single)) {
        path = single;
        std::ifstream f(single, std::ios::binary);
        char magic[8];
        uint64_t jl = 0, doff = 0;
        f.read(magic, 8);
        f.read((char*)&jl, 8);
        f.read((char*)&doff, 8);
        if (!f || std::memcmp(magic, "QVOLTA1\0", 8) != 0) throw std::runtime_error("not a .qv container: " + single);
        text.resize(jl);
        f.read(text.data(), (std::streamsize)jl);
        if (!f) throw std::runtime_error("truncated .qv header: " + single);
        rebase = doff;
    } else {
        std::ifstream f(prefix + ".qv.json");
        if (!f) throw std::runtime_error("cannot open " + single + " or " + prefix + ".qv.json");
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
        path = prefix + ".qv.bin";
    }
    json::Value root = json::parse(text);
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
            ti.off_q = v["off_q"].i64() + rebase;
            ti.nbytes_q = v["nbytes_q"].i64();
            ti.off_sm = v["off_sm"].i64() + rebase;
            ti.nbytes_sm = v["nbytes_sm"].i64();
        } else {
            ti.kind = k == "f32" ? Kind::F32 : Kind::F16;
            ti.off = v["off"].i64() + rebase;
            ti.nbytes = v["nbytes"].i64();
        }
        m.tensors.emplace(name, std::move(ti));
    }
    m.bin_path = path;
    m.bin_size = std::filesystem::file_size(m.bin_path);
    return m;
}

}  // namespace qv
