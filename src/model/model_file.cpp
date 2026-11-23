#include <algorithm>
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
    if (c.has("arch")) g.arch = c["arch"].str();
    g.hidden = (int)c["hidden_size"].i64();
    g.n_layers = (int)c["num_hidden_layers"].i64();
    g.vocab = (int)c["vocab_size"].i64();
    g.n_heads = (int)c["num_attention_heads"].i64();
    g.n_kv_heads = (int)c["num_key_value_heads"].i64();
    g.head_dim = (int)c["head_dim"].i64();
    g.eps = (float)c["rms_norm_eps"].num();
    g.eos = (int)c["eos_token_id"].i64();
    if (g.is_inkling()) {
        const auto& k = c["inkling"];
        InklingConfig& n = g.ink;
        n.unpadded_vocab = (int)k["unpadded_vocab"].i64();
        n.loop_start = (int)k["loop_start"].i64();
        n.loop_end = (int)k["loop_end"].i64();
        n.loop_passes = (int)k["loop_passes"].i64();
        n.loop_gate_rank = (int)k["loop_gate_rank"].i64();
        for (const auto& v : k["global_layers"].a) n.global_layers.push_back((int)v.i64());
        for (const auto& v : k["local_layers"].a) n.local_layers.push_back((int)v.i64());
        n.window = (int)k["window"].i64();
        n.d_rel = (int)k["d_rel"].i64();
        n.rel_extent = (int)k["rel_extent"].i64();
        n.qk_l2_clip = (float)k["qk_l2_clip"].num();
        n.rel_l2_clip = (float)k["rel_l2_clip"].num();
        n.log_scaling_n_floor = (int)k["log_scaling_n_floor"].i64();
        n.log_scaling_alpha = (float)k["log_scaling_alpha"].num();
        n.attn_res_block = (int)k["attn_res_block"].i64();
        n.sconv_kernel = (int)k["sconv_kernel"].i64();
        n.n_experts = (int)k["n_experts"].i64();
        n.topk = (int)k["topk"].i64();
        n.n_shared_experts = (int)k["n_shared_experts"].i64();
        n.expert_ff = (int)k["expert_ff"].i64();
        n.route_scale = (float)k["route_scale"].num();
        n.gate_softcap = (float)k["gate_softcap"].num();
        n.up_softcap = (float)k["up_softcap"].num();
        n.final_logit_softcap = (float)k["final_logit_softcap"].num();
        n.w13_interleaved = k["w13_layout"].str() == "interleaved";
        n.max_position = k["max_position"].i64();
        if ((int)(n.global_layers.size() + n.local_layers.size()) != g.n_layers) throw std::runtime_error("global/local layer lists do not cover all layers");
    } else if (g.arch != "inkling2_mtp") {  // the MTP sidecar (tools/convert_inkling_mtp.py) only carries tensors; the engine hardcodes + checks its shapes
        g.inter = (int)c["intermediate_size"].i64();
        const auto& rp = c["rope_parameters"];
        g.rope_theta = rp["rope_theta"].num();
        g.rope_dim = (int)(g.head_dim * rp["partial_rotary_factor"].num());
        g.lin_k_heads = (int)c["linear_num_key_heads"].i64();
        g.lin_v_heads = (int)c["linear_num_value_heads"].i64();
        g.lin_k_dim = (int)c["linear_key_head_dim"].i64();
        g.lin_v_dim = (int)c["linear_value_head_dim"].i64();
        g.conv_k = (int)c["linear_conv_kernel_dim"].i64();
        for (const auto& t : c["layer_types"].a) g.is_attn.push_back(t.str() == "full_attention");
        if ((int)g.is_attn.size() != g.n_layers) throw std::runtime_error("layer_types size mismatch");
    }

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
            if (v.has("meta")) {
                if (v["meta"].str() != "s8z8") throw std::runtime_error("unknown quant meta encoding: " + v["meta"].str());
                ti.meta8 = true;
                ti.off_rs = v["off_rs"].i64() + rebase;
                ti.nbytes_rs = v["nbytes_rs"].i64();
            }
        } else if (k == "nvfp4" || k == "mxfp8") {
            ti.kind = k == "nvfp4" ? Kind::NVFP4 : Kind::MXFP8;
            ti.bits = (int)v["bits"].i64();
            ti.group = (int)v["group"].i64();
            ti.off_q = v["off_q"].i64() + rebase;
            ti.nbytes_q = v["nbytes_q"].i64();
            ti.off_sm = v["off_sm"].i64() + rebase;
            ti.nbytes_sm = v["nbytes_sm"].i64();
            if (v.has("off_rs")) {
                ti.off_rs = v["off_rs"].i64() + rebase;
                ti.nbytes_rs = v["nbytes_rs"].i64();
            }
        } else {
            ti.kind = k == "f32" ? Kind::F32 : Kind::F16;  // also the optional "lora_f16" section (dense fp16 pairs)
            ti.off = v["off"].i64() + rebase;
            ti.nbytes = v["nbytes"].i64();
        }
        m.tensors.emplace(name, std::move(ti));
    }
    m.bin_path = path;
    m.bin_size = std::filesystem::file_size(m.bin_path);
    for (const auto& [name, t] : m.tensors) {  // detect truncated / partially written files
        uint64_t end = t.kind == Kind::F32 || t.kind == Kind::F16 ? t.off + t.nbytes : std::max({t.off_q + t.nbytes_q, t.off_sm + t.nbytes_sm, t.off_rs + t.nbytes_rs});
        if (end > m.bin_size) throw std::runtime_error("tensor extends past end of file (truncated model?): " + name);
    }
    return m;
}

bool InklingConfig::is_global(int layer) const {
    return std::find(global_layers.begin(), global_layers.end(), layer) != global_layers.end();
}

// Full tensor-table check of an Inkling file against its own config (names, ranks, shapes, kinds). Names are the checkpoint names.
void ModelFile::validate_inkling() const {
    if (!cfg.is_inkling()) throw std::runtime_error("not an inkling2_loop file");
    const InklingConfig& k = cfg.ink;
    const int64_t H = cfg.hidden, NL = cfg.n_layers, QD = (int64_t)cfg.n_heads * cfg.head_dim, KD = (int64_t)cfg.n_kv_heads * cfg.head_dim;
    auto need = [&](const std::string& n, std::vector<int64_t> shape, int kinds /*bit mask over Kind*/) -> const TensorInfo& {
        const TensorInfo& t = at(n);
        if (t.shape != shape) {
            std::string s = n + ": shape [";
            for (auto d : t.shape) s += std::to_string(d) + ",";
            throw std::runtime_error(s + "] does not match the config");
        }
        if (!((1 << (int)t.kind) & kinds)) throw std::runtime_error(n + ": unexpected tensor kind");
        return t;
    };
    const int Q = 1 << (int)Kind::Quant, D32 = 1 << (int)Kind::F32, D16 = 1 << (int)Kind::F16;
    const int EXP = Q | (1 << (int)Kind::NVFP4) | (1 << (int)Kind::MXFP8);
    const std::string P = "model.llm.";
    auto matrix = [&](const std::string& n, int64_t N, int64_t Kd) {
        const TensorInfo& t = need(n, {N, Kd}, Q | D16 | D32);
        if (t.kind == Kind::Quant && (Kd % 32 != 0 || Kd % t.group != 0 || t.bits < 2 || t.bits > 8)) throw std::runtime_error(n + ": bad quant parameters");
    };
    need(P + "embed_norm.weight", {H}, D32 | D16);
    need(P + "norm.weight", {H}, D32 | D16);
    matrix(P + "embed.weight", k.unpadded_vocab, H);
    matrix(P + "unembed.weight", k.unpadded_vocab, H);
    need(P + "attn_res.output_query", {H}, D32);
    const int64_t nloop = k.loop_end - k.loop_start;
    if (k.loop_passes > 1) {
        need(P + "middle_cycle_cross_pass_queries", {2 * nloop, H}, D32);
        need(P + "middle_cycle_cross_pass_kv_queries", {2 * nloop, H}, D32);
    }
    for (int L = 0; L < NL; L++) {
        const bool glob = k.is_global(L);
        const std::string l = P + "layers." + std::to_string(L) + ".";
        const std::string a = l + "attn.";
        if (L >= 1) need(P + "attn_res.attn_queries." + std::to_string(L - 1), glob ? std::vector<int64_t>{3, H} : std::vector<int64_t>{H}, D32);
        need(P + "attn_res.mlp_queries." + std::to_string(L), {H}, D32);
        matrix(a + "wq_du.weight", QD, H);
        matrix(a + "wk_dv.weight", KD, H);
        matrix(a + "wv_dv.weight", KD, H);
        matrix(a + "wo_ud.weight", H, QD);
        matrix(a + "gate_attn.weight", QD, H);
        matrix(a + "wr_du.weight", (int64_t)cfg.n_heads * k.d_rel, H);
        matrix(a + "attn_sink_proj.weight", cfg.n_heads, H);
        need(a + "q_l2_clip.scale", {cfg.n_heads, cfg.head_dim}, D32);
        need(a + "rel_l2_clip.scale", {cfg.n_heads, k.d_rel}, D32);
        need(a + "rel_logits_proj.proj", {k.d_rel, k.rel_extent}, D32);
        if (glob) {
            for (const char* c : {"k_sconv", "v_sconv"}) {
                need(a + c + ".static_w", {k.sconv_kernel, KD}, D32);
                matrix(a + c + ".weight_proj", (int64_t)cfg.n_kv_heads * k.sconv_kernel, H);
            }
            for (const char* c : {"q_input_norm", "k_input_norm", "v_input_norm"}) need(l + c + ".weight", {H}, D32 | D16);
        } else {
            need(a + "k_sconv.weight", {KD, k.sconv_kernel}, D32);  // depthwise kernels stored as [D, taps]
            need(a + "v_sconv.weight", {KD, k.sconv_kernel}, D32);
            need(l + "attn_norm.weight", {H}, D32 | D16);
        }
        need(l + "attn_sconv.weight", {H, k.sconv_kernel}, D32);
        need(l + "mlp_sconv.weight", {H, k.sconv_kernel}, D32);
        need(l + "mlp_norm.weight", {H}, D32 | D16);
        matrix(l + "mlp.gate.weight", k.n_experts, H);
        need(l + "mlp.gate.bias", {k.n_experts}, D32);
        const TensorInfo& w13 = need(l + "mlp.experts.w13_weight", {k.n_experts, 2 * (int64_t)k.expert_ff, H}, EXP);
        const TensorInfo& w2 = need(l + "mlp.experts.w2_weight", {k.n_experts, H, k.expert_ff}, EXP);
        if (w13.kind != w2.kind || w13.bits != w2.bits) throw std::runtime_error(l + "mlp.experts: w13 / w2 kinds differ");
        if (w13.kind == Kind::Quant && (w13.nbytes_q % k.n_experts || w13.nbytes_sm % k.n_experts))
            throw std::runtime_error(l + "mlp.experts.w13_weight: expert slices are not equally sized");
        if (k.in_loop(L) && k.loop_passes > 1)
            for (const char* c : {"attn", "k", "v", "mlp"}) {
                const std::string cp = l + "controllers." + c + ".";
                matrix(cp + "down_weight", k.loop_gate_rank, 2 * H);
                matrix(cp + "forget_weight", H, k.loop_gate_rank);
                matrix(cp + "gain_weight", H, k.loop_gate_rank);
            }
    }
}

}  // namespace qv
