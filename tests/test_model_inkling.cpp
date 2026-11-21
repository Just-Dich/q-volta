// Inkling-2-Mini model description: loads a .qv header, checks the Config fields and the full tensor table.
//   test_model_inkling                -> builds a tiny synthetic Inkling container (same names/layout as tools/convert_inkling.py)
//                                        in a temp file and validates it (always run by ctest)
//   test_model_inkling FILE.qv        -> validates a real converted model (header only; no weights are read)
// Also checks that a truncated file and a mis-shaped tensor are rejected.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "qv/model.h"

using namespace qv;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
            fails++;                                                              \
        }                                                                         \
    } while (0)

struct Fx {  // tiny synthetic model
    int H = 128, NL = 6, NH = 4, NKV = 2, HD = 32, DREL = 4, REXT = 16, FF = 64, E = 4, TOPK = 2, V = 200, VP = 208, GR = 8, KS = 4;
    int ls = 2, le = 4, passes = 3, bits = 3, grp = 64;
    std::vector<int> glob{1, 3, 5};
    uint64_t off = 0;
    std::ostringstream tj;
    bool first = true;
    bool truncated_shape = false;
    void sep() { if (!first) tj << ","; first = false; }
    uint64_t take(uint64_t n) { uint64_t o = off; off += (n + 255) / 256 * 256; return o; }
    void dense(const std::string& n, std::vector<int64_t> sh, bool f32) {
        uint64_t cnt = 1; for (auto d : sh) cnt *= d;
        uint64_t nb = cnt * (f32 ? 4 : 2);
        sep();
        tj << "\"" << n << "\":{\"kind\":\"" << (f32 ? "f32" : "f16") << "\",\"shape\":[";
        for (size_t i = 0; i < sh.size(); i++) tj << (i ? "," : "") << sh[i];
        tj << "],\"off\":" << take(nb) << ",\"nbytes\":" << nb << "}";
    }
    void quant(const std::string& n, std::vector<int64_t> sh, int b, int g) {
        uint64_t E_ = sh.size() == 3 ? sh[0] : 1, N = sh[sh.size() - 2], Kd = sh.back();
        uint64_t nq = E_ * N * (Kd / 32) * b * 4, ns = E_ * N * (Kd / g) * 2, nr = E_ * N * 2;
        sep();
        tj << "\"" << n << "\":{\"kind\":\"q\",\"bits\":" << b << ",\"group\":" << g << ",\"shape\":[";
        for (size_t i = 0; i < sh.size(); i++) tj << (i ? "," : "") << sh[i];
        tj << "],\"off_q\":" << take(nq) << ",\"nbytes_q\":" << nq << ",\"off_sm\":" << take(ns) << ",\"nbytes_sm\":" << ns
           << ",\"meta\":\"s8z8\",\"off_rs\":" << take(nr) << ",\"nbytes_rs\":" << nr << "}";
    }
    std::string build() {
        auto D = [&](const std::string& n, std::vector<int64_t> s) { dense(n, s, true); };
        std::string P = "model.llm.";
        quant(P + "embed.weight", {V, H}, 4, 64);
        quant(P + "unembed.weight", {V, H}, 4, 64);
        D(P + "embed_norm.weight", {H}); D(P + "norm.weight", {H}); D(P + "attn_res.output_query", {H});
        D(P + "middle_cycle_cross_pass_queries", {2 * (le - ls), H}); D(P + "middle_cycle_cross_pass_kv_queries", {2 * (le - ls), H});
        for (int L = 0; L < NL; L++) {
            bool g = std::find(glob.begin(), glob.end(), L) != glob.end();
            std::string l = P + "layers." + std::to_string(L) + ".", a = l + "attn.";
            if (L >= 1) D(P + "attn_res.attn_queries." + std::to_string(L - 1), g ? std::vector<int64_t>{3, H} : std::vector<int64_t>{H});
            D(P + "attn_res.mlp_queries." + std::to_string(L), {H});
            quant(a + "wq_du.weight", {NH * HD, H}, 4, 64); quant(a + "wk_dv.weight", {NKV * HD, H}, 4, 64);
            quant(a + "wv_dv.weight", {NKV * HD, H}, 4, 64); quant(a + "wo_ud.weight", {H, NH * HD}, 4, 64);
            quant(a + "gate_attn.weight", {NH * HD, H}, 4, 64);
            dense(a + "wr_du.weight", {NH * DREL, H}, false); dense(a + "attn_sink_proj.weight", {NH, H}, false);
            D(a + "q_l2_clip.scale", {NH, HD}); D(a + "rel_l2_clip.scale", {NH, DREL}); D(a + "rel_logits_proj.proj", {DREL, REXT});
            if (g) {
                for (const char* c : {"k_sconv", "v_sconv"}) { D(a + c + ".static_w", {KS, NKV * HD}); dense(a + c + ".weight_proj", {NKV * KS, H}, false); }
                for (const char* c : {"q_input_norm", "k_input_norm", "v_input_norm"}) D(l + c + ".weight", {H});
            } else {
                D(a + "k_sconv.weight", {NKV * HD, KS}); D(a + "v_sconv.weight", {NKV * HD, KS}); D(l + "attn_norm.weight", {H});
            }
            D(l + "attn_sconv.weight", {H, KS}); D(l + "mlp_sconv.weight", {H, KS}); D(l + "mlp_norm.weight", {H});
            dense(l + "mlp.gate.weight", {E, H}, false); D(l + "mlp.gate.bias", {E});
            quant(l + "mlp.experts.w13_weight", {E, 2 * FF, H}, bits, grp);
            quant(l + "mlp.experts.w2_weight", {E, H, truncated_shape ? FF + 64 : FF}, bits, grp);
            if (L >= ls && L < le)
                for (const char* c : {"attn", "k", "v", "mlp"}) {
                    std::string cp = l + "controllers." + c + ".";
                    dense(cp + "down_weight", {GR, 2 * H}, false); dense(cp + "forget_weight", {H, GR}, false); dense(cp + "gain_weight", {H, GR}, false);
                }
        }
        std::ostringstream o;
        o << "{\"format\":\"qvolta-v1\",\"config\":{\"arch\":\"inkling2_loop\",\"hidden_size\":" << H << ",\"num_hidden_layers\":" << NL
          << ",\"vocab_size\":" << VP << ",\"num_attention_heads\":" << NH << ",\"num_key_value_heads\":" << NKV << ",\"head_dim\":" << HD
          << ",\"rms_norm_eps\":1e-06,\"eos_token_id\":7,\"inkling\":{\"unpadded_vocab\":" << V << ",\"loop_start\":" << ls << ",\"loop_end\":" << le
          << ",\"loop_passes\":" << passes << ",\"loop_gate_rank\":" << GR << ",\"global_layers\":[1,3,5],\"local_layers\":[0,2,4],\"window\":8,\"d_rel\":" << DREL
          << ",\"rel_extent\":" << REXT << ",\"qk_l2_clip\":22.627416997969522,\"rel_l2_clip\":8.0,\"log_scaling_n_floor\":8192,\"log_scaling_alpha\":0.1"
          << ",\"attn_res_block\":6,\"sconv_kernel\":" << KS << ",\"n_experts\":" << E << ",\"topk\":" << TOPK << ",\"n_shared_experts\":0,\"expert_ff\":" << FF
          << ",\"route_scale\":8.0,\"gate_softcap\":4.0,\"up_softcap\":25.0,\"final_logit_softcap\":30.0,\"w13_layout\":\"interleaved\",\"max_position\":1048576}},"
          << "\"tensors\":{" << tj.str() << "}}";
        return o.str();
    }
};

static std::string write_container(const std::string& js, uint64_t data_bytes, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    uint64_t jl = js.size(), doff = 2u << 20;
    f.write("QVOLTA1\0", 8);
    f.write((const char*)&jl, 8);
    f.write((const char*)&doff, 8);
    f.write(js.data(), (std::streamsize)jl);
    std::string pad(doff - 24 - jl, '\0');
    f.write(pad.data(), (std::streamsize)pad.size());
    std::string z(1 << 20, '\0');
    for (uint64_t w = 0; w < data_bytes; w += z.size()) f.write(z.data(), (std::streamsize)std::min<uint64_t>(z.size(), data_bytes - w));
    return path;
}

static void report(const ModelFile& m) {
    const auto& k = m.cfg.ink;
    std::printf("arch %s: layers %d (loop [%d,%d) x%d -> %d applications), hidden %d, heads %d/%d x %d, vocab %d (rows %d), experts %d top-%d ff %d, "
                "window %d, d_rel %d, rel_extent %d, globals %zu, tensors %zu\n",
                m.cfg.arch.c_str(), m.cfg.n_layers, k.loop_start, k.loop_end, k.loop_passes, k.n_applications(m.cfg.n_layers), m.cfg.hidden, m.cfg.n_heads,
                m.cfg.n_kv_heads, m.cfg.head_dim, m.cfg.vocab, k.unpadded_vocab, k.n_experts, k.topk, k.expert_ff, k.window, k.d_rel, k.rel_extent,
                k.global_layers.size(), m.tensors.size());
}

int main(int argc, char** argv) {
    if (argc > 1) {  // real file
        ModelFile m = ModelFile::open(argv[1]);
        report(m);
        m.validate_inkling();
        const TensorInfo& w13 = m.at("model.llm.layers.0.mlp.experts.w13_weight");
        std::printf("layer0 w13: kind %d bits %d group %d meta8 %d E %lld expert1 off_q %llu\n", (int)w13.kind, w13.bits, w13.group, (int)w13.meta8,
                    (long long)w13.n_experts(), (unsigned long long)w13.expert_off_q(1));
        std::printf("test_model_inkling (%s): %s\n", argv[1], fails ? "FAIL" : "PASS");
        return fails != 0;
    }
    auto tmp = std::filesystem::temp_directory_path();
    Fx fx;
    std::string js = fx.build();
    std::string p = write_container(js, fx.off, (tmp / "qv_inkling_fixture.qv").string());
    ModelFile m = ModelFile::open(p);
    report(m);
    const auto& k = m.cfg.ink;
    CHECK(m.cfg.is_inkling());
    CHECK(m.cfg.hidden == 128 && m.cfg.n_layers == 6 && m.cfg.vocab == 208 && k.unpadded_vocab == 200);
    CHECK(k.loop_start == 2 && k.loop_end == 4 && k.loop_passes == 3);
    CHECK(k.n_applications(m.cfg.n_layers) == 10);  // 6 + 2 extra passes x 2 loop layers
    CHECK(k.is_global(1) && k.is_global(5) && !k.is_global(0) && !k.is_global(4));
    CHECK(k.passes(2) == 3 && k.passes(4) == 1);
    CHECK(k.w13_interleaved && k.n_experts == 4 && k.topk == 2 && k.expert_ff == 64);
    CHECK(k.qk_l2_clip > 22.6f && k.qk_l2_clip < 22.7f && k.rel_l2_clip == 8.0f && k.final_logit_softcap == 30.0f);
    m.validate_inkling();
    const TensorInfo& w13 = m.at("model.llm.layers.3.mlp.experts.w13_weight");
    CHECK(w13.kind == Kind::Quant && w13.stacked() && w13.n_experts() == 4 && w13.bits == 3 && w13.group == 64 && w13.meta8);
    CHECK(w13.shape == (std::vector<int64_t>{4, 128, 128}));
    CHECK(w13.expert_off_q(2) == w13.off_q + 2 * (w13.nbytes_q / 4));
    CHECK(m.at("model.llm.layers.1.attn.k_sconv.static_w").shape == (std::vector<int64_t>{4, 64}));  // global layer: dynamic conv static taps
    CHECK(m.at("model.llm.layers.0.attn.k_sconv.weight").shape == (std::vector<int64_t>{64, 4}));    // local layer: depthwise kernel
    CHECK(m.tensors.count("model.llm.layers.3.controllers.mlp.gain_weight") == 1 && m.tensors.count("model.llm.layers.1.controllers.mlp.gain_weight") == 0);
    // negative: truncated file is rejected
    {
        std::string q = write_container(js, fx.off / 2, (tmp / "qv_inkling_trunc.qv").string());
        bool thrown = false;
        try { ModelFile::open(q); } catch (const std::exception&) { thrown = true; }
        CHECK(thrown);
        std::filesystem::remove(q);
    }
    // negative: wrong expert shape is rejected by validate_inkling
    {
        Fx bad; bad.truncated_shape = true;
        std::string bjs = bad.build();
        std::string q = write_container(bjs, bad.off, (tmp / "qv_inkling_bad.qv").string());
        ModelFile bm = ModelFile::open(q);
        bool thrown = false;
        try { bm.validate_inkling(); } catch (const std::exception&) { thrown = true; }
        CHECK(thrown);
        std::filesystem::remove(q);
    }
    std::filesystem::remove(p);
    std::printf("test_model_inkling: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
