// qv_run — driver for correctness dumps and decode benchmarks.
//   qv_run --model PREFIX --golden TOKENS.i32 --out DUMP.bin      (hidden states after embed+each layer, + logits)
//   qv_run --model PREFIX --bench --prompt TOKENS.i32 [--n 128] [--max-ctx 4096]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "qv/engine.h"

static std::vector<int> read_i32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}

int main(int argc, char** argv) {
    std::string model, golden, out, prompt;
    bool bench = false, graph = false;
    int n = 128, max_ctx = 4096, warm = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--golden") golden = next();
        else if (a == "--out") out = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--bench") bench = true;
        else if (a == "--graph") graph = true;
        else if (a == "--n") n = std::stoi(next());
        else if (a == "--max-ctx") max_ctx = std::stoi(next());
        else if (a == "--warmup") warm = std::stoi(next());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    auto t0 = std::chrono::steady_clock::now();
    qv::ModelFile mf = qv::ModelFile::open(model);
    qv::EngineOptions opt;
    opt.max_ctx = max_ctx;
    opt.dump_hidden = !golden.empty();
    auto eng = qv::make_cuda_engine(mf, opt);
    eng->sync();
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    std::fprintf(stderr, "loaded in %.1fs: weights %.2f GiB, state+scratch %.2f GiB, device used %.2f / %.2f GiB\n", load_s,
                 eng->device_bytes_weights() / 1073741824.0, eng->device_bytes_state() / 1073741824.0,
                 (tot - fr) / 1073741824.0, tot / 1073741824.0);

    if (!golden.empty()) {
        std::vector<int> toks = read_i32(golden);
        std::ofstream o(out, std::ios::binary);
        std::vector<float> lg;
        for (size_t p = 0; p < toks.size(); ++p) {
            eng->step(toks[p], (int)p);
            eng->logits(lg);
            o.write((const char*)lg.data(), lg.size() * 4);
        }
        const auto& h = eng->hidden_dump();
        o.write((const char*)h.data(), h.size() * 4);
        std::fprintf(stderr, "golden: %zu tokens dumped to %s\n", toks.size(), out.c_str());
        return 0;
    }
    if (bench) {
        std::vector<int> toks = prompt.empty() ? std::vector<int>{9707} : read_i32(prompt);
        // warmup (not timed), then reset
        for (int i = 0; i < warm; ++i) eng->step(toks[0], i);
        eng->sync();
        eng->reset();
        cudaEvent_t e0, e1, e2;
        cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
        cudaEventRecord(e0);
        int pos = 0;
        for (; pos < (int)toks.size(); ++pos) eng->step(toks[pos], pos);
        int next = eng->argmax();
        cudaEventRecord(e1);
        std::vector<int> gen{next}, rest;
        auto td0 = std::chrono::steady_clock::now();
        if (graph) {  // capture outside the timed region: one untimed decode, then restore by re-running the prompt
            eng->decode_greedy(next, pos, 1, rest, true);
            eng->reset();
            for (int p = 0; p < (int)toks.size(); ++p) eng->step(toks[p], p);
            eng->sync();
            cudaEventRecord(e1);
            td0 = std::chrono::steady_clock::now();
        }
        eng->decode_greedy(next, pos, n - 1, rest, graph);
        gen.insert(gen.end(), rest.begin(), rest.end());
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        double td = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        float pre_ms, dec_ms;
        cudaEventElapsedTime(&pre_ms, e0, e1);
        cudaEventElapsedTime(&dec_ms, e1, e2);
        cudaMemGetInfo(&fr, &tot);
        std::printf("{\"prompt_tokens\": %zu, \"prefill_ms\": %.2f, \"prefill_tok_s\": %.2f, \"decode_tokens\": %d, "
                    "\"decode_ms\": %.2f, \"decode_tok_s\": %.2f, \"decode_tok_s_wall\": %.2f, \"vram_used_gib\": %.3f, \"ids\": [",
                    toks.size(), pre_ms, toks.size() / (pre_ms / 1e3), n - 1, dec_ms, (n - 1) / (dec_ms / 1e3),
                    (n - 1) / td, (tot - fr) / 1073741824.0);
        for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? ", " : "", gen[i]);
        std::printf("]}\n");
        return 0;
    }
    std::fprintf(stderr, "nothing to do (use --golden or --bench)\n");
    return 2;
}
