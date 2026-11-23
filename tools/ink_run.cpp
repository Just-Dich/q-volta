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
    int runs = 1, gen = 32, max_ctx = 8192, n = 128, synth = 0, chunk = 2048;
    bool bench = false, use_prefill = false, no_graph = false, pfgen = false;
    std::string kv = "fp16", mtp_file;
    int mtp_depth = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--out") out = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--max-ctx") max_ctx = std::stoi(next());
        else if (a == "--n") n = std::stoi(next());
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--synth-ctx") synth = std::stoi(next());
        else if (a == "--bench") bench = true;
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--no-graph") no_graph = true;
        else if (a == "--pf-gen") pfgen = true;
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--prefill-chunk") chunk = std::stoi(next());
        else if (a == "--kv") kv = next();
        else if (a == "--mtp-file") mtp_file = next();
        else if (a == "--mtp-depth") mtp_depth = std::stoi(next());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    qv::ModelFile mf = qv::ModelFile::open(model);
    qv::EngineOptions opt;
    opt.max_ctx = max_ctx;
    opt.prefill_chunk = chunk;
    opt.kv_mode = kv == "int4" ? 1 : kv == "fp16" ? 2 : 0;
    opt.mtp = mtp_depth > 0;
    opt.mtp_depth = mtp_depth;
    opt.mtp_file = mtp_file;
    opt.dump_hidden = !bench && !pfgen && !out.empty();
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
        // per run: (plain, then MTP when --mtp-depth > 0) x reset + prefill + untimed warm-up + timed greedy decode of n tokens; one JSON line each
        if (use_prefill) { eng->prefill(ids.data(), N, 0); eng->sync(); eng->reset(); }  // warm-up (allocations, cuBLAS)
        for (int run = 0; run < runs; ++run)
            for (int m = 0; m < (mtp_depth > 0 ? 2 : 1); ++m) {
                const bool use_mtp = m == 1;
                eng->set_mtp_enabled(use_mtp);
                eng->reset();
                auto e0 = clk::now();
                eng->prefill(ids.data(), N, 0);
                int next = eng->argmax();
                eng->sync();
                const double pre_ms = std::chrono::duration<double, std::milli>(clk::now() - e0).count();
                int pos = N;
                if (synth > pos) pos = synth;
                std::vector<int> w, rest;
                const int nw = 24;
                eng->decode_greedy(next, pos, nw, w, !no_graph);
                eng->sync();
                qv::MtpStats ms0;
                eng->mtp_stats(ms0);
                auto t1 = clk::now();
                eng->decode_greedy(w.back(), pos + nw, n, rest, !no_graph);
                eng->sync();
                const double dec_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
                qv::MtpStats ms;
                eng->mtp_stats(ms);
                uint64_t hsh = 1469598103934665603ull;
                for (int v : rest) { hsh ^= (uint32_t)v; hsh *= 1099511628211ull; }
                std::printf("{\"run\": %d, \"mode\": \"%s\", \"prompt_tokens\": %d, \"decode_pos\": %d, \"prefill_ms\": %.1f, \"decode_tokens\": %d, \"decode_ms\": %.2f, "
                            "\"decode_tok_s\": %.2f, \"ids_fnv\": \"%016llx\", \"first_ids\": [",
                            run, use_mtp ? "mtp" : "plain", N, pos + nw, pre_ms, n, dec_ms, n / (dec_ms / 1e3), (unsigned long long)hsh);
                for (size_t i = 0; i < rest.size() && i < 8; ++i) std::printf("%s%d", i ? ", " : "", rest[i]);
                std::printf("]");
                if (use_mtp)
                    std::printf(", \"mtp\": {\"depth\": %d, \"cycles\": %ld, \"drafted\": %ld, \"accepted\": %ld, \"plain_steps\": %ld, \"tokens_per_cycle\": %.3f}", mtp_depth,
                                ms.cycles - ms0.cycles, ms.drafted - ms0.drafted, ms.accepted - ms0.accepted, ms.plain_steps - ms0.plain_steps,
                                ms.cycles > ms0.cycles ? 1.0 + (double)(ms.accepted - ms0.accepted) / (ms.cycles - ms0.cycles) : 0.0);
                std::printf("}\n");
                std::fflush(stdout);
            }
        return 0;
    }
    if (pfgen) {  // batched prefill, then greedy decode (graph); prints last-position logits stats and ids
        auto e0 = clk::now();
        eng->prefill(ids.data(), N, 0);
        eng->sync();
        const double pre_s = std::chrono::duration<double>(clk::now() - e0).count();
        std::vector<float> lg;
        eng->logits(lg);
        int first = eng->argmax();
        std::vector<int> rest;
        if (gen > 1) eng->decode_greedy(first, N, gen - 1, rest, true);
        rest.insert(rest.begin(), first);
        if (!out.empty() && !eng->hidden_dump().empty()) {
            std::ofstream o(out + ".hid", std::ios::binary);
            o.write((const char*)eng->hidden_dump().data(), eng->hidden_dump().size() * 4);
        }
        if (!out.empty()) {
            std::ofstream o(out, std::ios::binary);
            o.write((const char*)lg.data(), lg.size() * 4);
            o.write((const char*)rest.data(), rest.size() * 4);
        }
        std::printf("{\"prompt_tokens\": %d, \"prefill_s\": %.3f, \"prefill_tok_s\": %.1f, \"ids\": [", N, pre_s, N / pre_s);
        for (size_t i = 0; i < rest.size(); ++i) std::printf("%s%d", i ? ", " : "", rest[i]);
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
