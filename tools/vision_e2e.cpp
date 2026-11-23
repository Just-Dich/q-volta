// qv_vision_e2e: image-prompt prefill (prefill_embeds) + greedy decode through an Inkling engine; used to validate against
// tests/golden/inkling_vision/e2e_*.npz (tools/vision_e2e_check.py writes the raw inputs and compares).
//   qv_vision_e2e --model X.qv --ids ids.i32 --mask mask.u8 --emb emb.f32 [--gen 32] [--cpu] [--max-ctx N] [--kv fp16|int4] [--logits-out f]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "qv/engine.h"

template <class T> static std::vector<T> rd(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::vector<T> v((size_t)f.tellg() / sizeof(T));
    f.seekg(0);
    f.read((char*)v.data(), v.size() * sizeof(T));
    return v;
}

int main(int argc, char** argv) {
    std::string model, ids_p, mask_p, emb_p, lo;
    int gen = 32, max_ctx = 4096;
    bool cpu = false;
    std::string kv = "fp16";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nx(); else if (a == "--ids") ids_p = nx(); else if (a == "--mask") mask_p = nx(); else if (a == "--emb") emb_p = nx();
        else if (a == "--gen") gen = std::stoi(nx()); else if (a == "--cpu") cpu = true; else if (a == "--max-ctx") max_ctx = std::stoi(nx());
        else if (a == "--kv") kv = nx(); else if (a == "--logits-out") lo = nx();
    }
    try {
        qv::ModelFile mf = qv::ModelFile::open(model);
        qv::EngineOptions opt;
        opt.max_ctx = max_ctx;
        opt.kv_mode = kv == "int4" ? 1 : kv == "fp16" ? 2 : 0;
        std::unique_ptr<qv::Engine> eng;
#ifdef QV_HAVE_CPU
        if (cpu) eng = qv::make_cpu_engine(mf, opt);
        else
#endif
            eng = qv::make_cuda_engine(mf, opt);
        auto ids = rd<int>(ids_p);
        auto mask = rd<unsigned char>(mask_p);
        auto emb = rd<float>(emb_p);
        const int T = (int)ids.size();
        auto t0 = std::chrono::steady_clock::now();
        eng->prefill_embeds(ids.data(), emb.data(), mask.data(), T, 0);
        eng->sync();
        const double pf = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!lo.empty()) {
            std::vector<float> lg;
            eng->logits(lg);
            std::ofstream o(lo, std::ios::binary);
            o.write((const char*)lg.data(), lg.size() * 4);
        }
        std::vector<int> out;
        int nxt = eng->argmax();
        for (int s = 0; s < gen; ++s) {
            out.push_back(nxt);
            if (s + 1 == gen) break;
            eng->step(nxt, T + s);
            nxt = eng->argmax();
        }
        std::printf("{\"prefill_ms\": %.1f, \"tokens\": %d, \"gen\": [", pf, T);
        for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? "," : "", out[i]);
        std::printf("]}\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
