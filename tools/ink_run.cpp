// qv_ink - Inkling engine driver (CUDA): golden dumps and benchmarks.
//   qv_ink --model X.qv --ids prompt.i32 --out dump.bin [--gen 32] [--max-ctx N] [--prefill]
//        teacher-forced pass over the prompt (dump mode: per-application hidden states / router ids), then greedy generation; writes
//        a raw dump read by tools/ink_check.py (format: see the writer below).
//   qv_ink --model X.qv --ids prompt.i32 --bench [--n 128] [--max-ctx N] [--synth-ctx C] [--prefill]
//        prefill the prompt, then time greedy decode (CUDA graph). --synth-ctx: decode at that position over unwritten KV.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
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
    std::string model, ids_path, out;
    int gen = 32, max_ctx = 8192, n = 128, synth = 0, chunk = 2048;
    bool bench = false, use_prefill = false, no_graph = false;
    std::string kv = "fp16";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--out") out = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--max-ctx") max_ctx = std::stoi(next());
        else if (a == "--n") n = std::stoi(next());
        else if (a == "--synth-ctx") synth = std::stoi(next());
        else if (a == "--bench") bench = true;
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--no-graph") no_graph = true;
        else if (a == "--prefill-chunk") chunk = std::stoi(next());
        else if (a == "--kv") kv = next();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    qv::ModelFile mf = qv::ModelFile::open(model);
    qv::EngineOptions opt;
    opt.max_ctx = max_ctx;
    opt.prefill_chunk = chunk;
    opt.kv_mode = kv == "int4" ? 1 : kv == "fp16" ? 2 : 0;
    opt.dump_hidden = !bench && !out.empty();
    auto t0 = std::chrono::steady_clock::now();
    auto eng = qv::make_cuda_engine(mf, opt);
    eng->sync();
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    std::fprintf(stderr, "loaded in %.1fs: weights %.2f GiB, state+scratch %.2f GiB, device used %.2f / %.2f GiB\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), eng->device_bytes_weights() / 1073741824.0,
                 eng->device_bytes_state() / 1073741824.0, (tot - fr) / 1073741824.0, tot / 1073741824.0);
    std::vector<int> ids = read_i32(ids_path);
    const int N = (int)ids.size();
    using clk = std::chrono::steady_clock;
    if (bench) {
        auto e0 = clk::now();
        eng->prefill(ids.data(), N, 0);
        int next = eng->argmax();
        eng->sync();
        const double pre_ms = std::chrono::duration<double, std::milli>(clk::now() - e0).count();
        int pos = N;
        if (synth > pos) pos = synth;
        std::vector<int> w, rest;
        eng->decode_greedy(next, pos, 4, w, !no_graph);
        eng->sync();
        auto t1 = clk::now();
        eng->decode_greedy(w.back(), pos + 4, n, rest, !no_graph);
        eng->sync();
        const double dec_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
        cudaMemGetInfo(&fr, &tot);
        std::printf("{\"prompt_tokens\": %d, \"decode_pos\": %d, \"synthetic_kv\": %s, \"prefill_ms\": %.1f, \"prefill_tok_s\": %.1f, "
                    "\"decode_tokens\": %d, \"decode_ms\": %.2f, \"decode_tok_s\": %.2f, \"vram_used_gib\": %.3f, \"ids\": [",
                    N, pos + 4, synth > N ? "true" : "false", pre_ms, N / (pre_ms / 1e3), n, dec_ms, n / (dec_ms / 1e3), (tot - fr) / 1073741824.0);
        for (size_t i = 0; i < rest.size() && i < 32; ++i) std::printf("%s%d", i ? ", " : "", rest[i]);
        std::printf("]}\n");
        return 0;
    }
    // ---- golden dump
    const int H = 2304;
    const int posl[3] = {0, N / 2, N - 1};
    std::vector<float> hid(3 * 50 * H);
    std::vector<int> rid((size_t)N * 48 * 8);
    auto grab = [&](int p) {
        const auto& h = eng->hidden_dump();
        const auto& r = eng->aux_dump();
        for (int k = 0; k < 3; ++k)
            if (p == posl[k]) std::memcpy(&hid[(size_t)k * 50 * H], h.data(), 50 * H * 4);
        std::memcpy(&rid[(size_t)p * 48 * 8], r.data(), 48 * 8 * 4);
    };
    if (use_prefill) {
        std::fprintf(stderr, "--prefill dump not supported: using step()\n");
    }
    for (int p = 0; p < N; ++p) {
        eng->step(ids[p], p);
        grab(p);
    }
    std::vector<float> lg;
    eng->logits(lg);
    const int V = (int)lg.size();
    std::vector<float> lastlg = lg;
    std::vector<int> gen_ids, top_ids;
    std::vector<float> top_val;
    for (int s = 0; s < gen; ++s) {
        std::vector<int> ord(V);
        std::iota(ord.begin(), ord.end(), 0);
        std::partial_sort(ord.begin(), ord.begin() + 20, ord.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
        for (int k = 0; k < 20; ++k) { top_ids.push_back(ord[k]); top_val.push_back(lg[ord[k]]); }
        gen_ids.push_back(ord[0]);
        if (s == gen - 1) break;
        eng->step(ord[0], N + s);
        eng->logits(lg);
    }
    std::ofstream o(out, std::ios::binary);
    int hdr[6] = {0x4b4e4951, N, gen, V, 50, 48};
    o.write((const char*)hdr, sizeof hdr);
    o.write((const char*)posl, sizeof posl);
    o.write((const char*)hid.data(), hid.size() * 4);
    o.write((const char*)rid.data(), rid.size() * 4);
    o.write((const char*)lastlg.data(), lastlg.size() * 4);
    o.write((const char*)gen_ids.data(), gen_ids.size() * 4);
    o.write((const char*)top_ids.data(), top_ids.size() * 4);
    o.write((const char*)top_val.data(), top_val.size() * 4);
    std::fprintf(stderr, "dump written: %s (%d prompt tokens, %d generated)\n", out.c_str(), N, gen);
    std::printf("gen_ids:");
    for (int g : gen_ids) std::printf(" %d", g);
    std::printf("\n");
    return 0;
}
