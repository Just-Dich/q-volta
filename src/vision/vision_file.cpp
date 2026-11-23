// Sidecar container reader for the Inkling vision tower (tools/convert_inkling_vision.py, format "qvolta-vision-v1").
#include <cstdio>
#include <cstring>
#include <fstream>

#include "core/json.h"
#include "vision.h"

namespace qv::vision {

float half_to_float(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, o;
    if (e == 0) {
        if (m == 0) o = s;
        else {  // subnormal
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            m &= 0x3ff;
            o = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &o, 4);
    return f;
}

uint16_t float_to_half(float f) {  // round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 255) return (uint16_t)(s | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(s | r);
    }
    uint32_t r = ((uint32_t)e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t)(s | r);
}

const VisionFile::T& VisionFile::at(const std::string& n) const {
    for (const auto& kv : list)
        if (kv.first == n) return kv.second;
    throw std::runtime_error("vision sidecar: tensor not found: " + n);
}

std::unique_ptr<VisionFile> VisionFile::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open vision file " + path);
    char magic[8];
    uint64_t jl = 0, doff = 0;
    f.read(magic, 8); f.read((char*)&jl, 8); f.read((char*)&doff, 8);
    if (!f || std::memcmp(magic, "QVOLTA1\0", 8) != 0) throw std::runtime_error("not a .qv container: " + path);
    if (jl > (64u << 20)) throw std::runtime_error("vision sidecar: implausible header size");
    std::string text(jl, '\0');
    f.read(text.data(), (std::streamsize)jl);
    if (!f) throw std::runtime_error("truncated vision sidecar header: " + path);
    json::Value root = json::parse(text);
    if (root["format"].str() != "qvolta-vision-v1")
        throw std::runtime_error("not a vision sidecar (format tag '" + root["format"].str() + "'): " + path);
    auto v = std::make_unique<VisionFile>();
    v->path = path;
    v->data_off = doff;
    const auto& c = root["config"];
    VisionConfig& g = v->cfg;
    g.hidden = (int)c["hidden_size"].i64(); g.layers = (int)c["num_hidden_layers"].i64(); g.heads = (int)c["num_attention_heads"].i64();
    g.mlp = (int)(g.hidden * c["mlp_ratio"].num()); g.patch = (int)c["patch_size"].i64(); g.merge = (int)c["merge_size"].i64();
    g.grid = (int)c["posemb_grid_size"].i64(); g.proj = (int)c["projection_hidden_dim"].i64(); g.text_hidden = (int)c["text_hidden_size"].i64();
    g.eps = (float)c["layer_norm_eps"].num(); g.rope_theta = (float)c["rope_theta"].num(); g.layer_scale_init = (float)c["layer_scale_init"].num();
    g.image_token_id = (int)c["image_token_id"].i64(); g.min_side = (int)c["min_side"].i64(); g.max_side = (int)c["max_side"].i64();
    v->quant = c["quant"].str();
    uint64_t fsz = 0;
    { std::ifstream e(path, std::ios::binary | std::ios::ate); fsz = (uint64_t)e.tellg(); }
    for (const auto& [name, t] : root["tensors"].o) {
        T e;
        for (const auto& d : t["shape"].a) e.shape.push_back(d.i64());
        e.kind = t["kind"].str();
        e.off = t["off"].i64() + doff;
        e.nbytes = t["nbytes"].i64();
        if (e.kind == "i8") e.soff = t["scale"]["off"].i64() + doff;
        if (e.off + e.nbytes > fsz) throw std::runtime_error("vision sidecar truncated (tensor " + name + " past end of file)");
        v->list.emplace_back(name, std::move(e));
    }
    return v;
}

static void read_at(const std::string& path, uint64_t off, void* dst, uint64_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    if (fseeko(f, (off_t)off, SEEK_SET) != 0 || std::fread(dst, 1, n, f) != n) { std::fclose(f); throw std::runtime_error("short read " + path); }
    std::fclose(f);
}

std::vector<float> VisionFile::floats(const std::string& n) const {
    const T& t = at(n);
    int64_t cnt = 1;
    for (int64_t d : t.shape) cnt *= d;
    std::vector<float> out(cnt);
    if (t.kind == "f32") read_at(path, t.off, out.data(), t.nbytes);
    else if (t.kind == "f16") {
        std::vector<uint16_t> h(cnt);
        read_at(path, t.off, h.data(), t.nbytes);
        for (int64_t i = 0; i < cnt; ++i) out[i] = half_to_float(h[i]);
    } else if (t.kind == "i8") {
        std::vector<int8_t> q(cnt);
        const int64_t N = t.shape[0], K = cnt / N;
        std::vector<float> sc(N);
        read_at(path, t.off, q.data(), t.nbytes);
        read_at(path, t.soff, sc.data(), N * 4);
        for (int64_t i = 0; i < N; ++i)
            for (int64_t k = 0; k < K; ++k) out[i * K + k] = q[i * K + k] * sc[i];
    } else throw std::runtime_error("vision sidecar: unknown tensor kind " + t.kind);
    return out;
}

std::vector<uint16_t> VisionFile::half(const std::string& n) const {
    const T& t = at(n);
    if (t.kind == "f16") {
        std::vector<uint16_t> h(t.nbytes / 2);
        read_at(path, t.off, h.data(), t.nbytes);
        return h;
    }
    std::vector<float> f = floats(n);
    std::vector<uint16_t> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = float_to_half(f[i]);
    return h;
}

std::unique_ptr<Encoder> Encoder::create(const std::string& sidecar, bool cuda) {
    std::shared_ptr<VisionFile> f(VisionFile::open(sidecar).release());
    if (cuda) {
#ifdef QV_VISION_CUDA
        return make_cuda_encoder(f);
#else
        throw std::runtime_error("CUDA vision encoder not built into this binary");
#endif
    }
    return make_cpu_encoder(f);
}

}  // namespace qv::vision
