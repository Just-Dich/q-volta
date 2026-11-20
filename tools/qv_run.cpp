// qv_run — driver for correctness dumps and decode benchmarks.
//   qv_run --model PREFIX --golden TOKENS.i32 --out DUMP.bin      (hidden states after embed+each layer, + logits)
//   qv_run --backend cuda|cpu [--threads N] ...   (cpu: no GPU is touched; qv_run_cpu is the same source built without CUDA)
//   qv_run --model PREFIX --bench --prompt TOKENS.i32 [--n 128] [--max-ctx 4096]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef QV_HAVE_CUDA
#include <cuda_runtime.h>
#else
#define QV_HAVE_CPU 1
#endif

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
    int tail = 0, gen_n = 0;
    std::string sparse_list;  // --tail M: prefill all but the last M prompt tokens, then step() the M tokens (teacher forced) and dump
                              // their logits to --out (fp32); --gen G then greedy-decodes G tokens and prints the ids (research probe)
    std::string model, golden, out, prompt;
    bool bench = false, graph = false, use_prefill = false, mtp = false;
#ifdef QV_HAVE_CUDA
    std::string backend = "cuda";
#else
    std::string backend = "cpu";
#endif
    int threads = 32;
    int mtp_depth = 1;
    int n = 128, max_ctx = 4096, warm = 8, chunk = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--golden") golden = next();
        else if (a == "--out") out = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--backend") backend = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--bench") bench = true;
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--prefill-chunk") chunk = std::stoi(next());
        else if (a == "--mtp") mtp = true;
        else if (a == "--mtp-depth") mtp_depth = std::stoi(next());
        else if (a == "--graph") graph = true;
        else if (a == "--tail") tail = std::stoi(next());
        else if (a == "--sparse-list") sparse_list = next();
        else if (a == "--gen") gen_n = std::stoi(next());
        else if (a == "--n") n = std::stoi(next());
        else if (a == "--max-ctx") max_ctx = std::stoi(next());
        else if (a == "--warmup") warm = std::stoi(next());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    auto t0 = std::chrono::steady_clock::now();
    qv::ModelFile mf = qv::ModelFile::open(model);
    qv::EngineOptions opt;
    opt.max_ctx = max_ctx;
    opt.prefill_chunk = chunk > 0 ? chunk : (backend == "cpu" ? 256 : 2048);  // cpu: tokens per batched-prefill chunk
    opt.dump_hidden = !golden.empty();
    opt.threads = threads;
    opt.mtp = mtp;
    opt.mtp_depth = mtp_depth;
    std::unique_ptr<qv::Engine> eng;
    if (backend == "cpu") {
#ifdef QV_HAVE_CPU
        eng = qv::make_cpu_engine(mf, opt);
#else
        std::fprintf(stderr, "this build has no CPU engine\n"); return 2;
#endif
    } else {
#ifdef QV_HAVE_CUDA
        eng = qv::make_cuda_engine(mf, opt);
#else
        std::fprintf(stderr, "this build has no CUDA engine\n"); return 2;
#endif
    }
    const bool cpu = backend == "cpu";
    eng->sync();
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t fr = 0, tot = 0;
#ifdef QV_HAVE_CUDA
    if (!cpu) cudaMemGetInfo(&fr, &tot);
#endif
    std::fprintf(stderr, "loaded in %.1fs: weights %.2f GiB, state+scratch %.2f GiB, device used %.2f / %.2f GiB\n", load_s,
                 eng->device_bytes_weights() / 1073741824.0, eng->device_bytes_state() / 1073741824.0,
                 (tot - fr) / 1073741824.0, tot / 1073741824.0);

    if (tail > 0) {
        std::vector<int> toks = read_i32(prompt);
        const int N = (int)toks.size();
        auto t1 = std::chrono::steady_clock::now();
        eng->prefill(toks.data(), N - tail, 0);
        double pre_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        std::fprintf(stderr, "prefill %d tokens: %.1f s (%.1f tok/s)\n", N - tail, pre_s, (N - tail) / pre_s);
        std::vector<float> lg;
        eng->logits(lg);  // logits after token N-tail-1 (predicts toks[N-tail])
        std::vector<float> lg0 = lg;
        // configs: "off" or "sink,recent,block,k[,agg]" separated by ';' (default: whatever QV_SPARSE says = one run)
        std::vector<std::string> cfgs;
        for (size_t b = 0; b <= sparse_list.size();) {
            size_t e = sparse_list.find(';', b);
            if (e == std::string::npos) e = sparse_list.size();
            cfgs.push_back(sparse_list.substr(b, e - b));
            b = e + 1;
        }
        if (sparse_list.empty()) cfgs = {"env"};
        eng->state_save();
        for (size_t ci = 0; ci < cfgs.size(); ++ci) {
            if (cfgs[ci] != "env") eng->set_sparse(cfgs[ci]);
            eng->state_restore();
            std::ofstream o(out + (cfgs.size() > 1 ? "." + std::to_string(ci) : ""), std::ios::binary);
            o.write((const char*)lg0.data(), lg0.size() * 4);
            auto t2 = std::chrono::steady_clock::now();
            for (int p = N - tail; p < N; ++p) {
                eng->step(toks[p], p);
                eng->logits(lg);
                o.write((const char*)lg.data(), lg.size() * 4);
            }
            double st_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
            std::printf("{\"cfg\": \"%s\", \"tail_ms_per_step\": %.1f", cfgs[ci].c_str(), st_s * 1e3 / tail);
            if (gen_n > 0) {
                std::vector<int> g;
                int first = eng->argmax();
                eng->decode_greedy(first, N, gen_n, g, false);
                std::printf(", \"gen\": [%d", first);
                for (int x : g) std::printf(", %d", x);
                std::printf("]");
            }
            std::printf("}\n");
            std::fflush(stdout);
        }
        return 0;
    }
    if (!golden.empty()) {
        std::vector<int> toks = read_i32(golden);
        std::ofstream o(out, std::ios::binary);
        std::vector<float> lg;
        if (use_prefill) {
            eng->prefill(toks.data(), (int)toks.size(), 0);
            const auto& la = eng->logits_dump();
            o.write((const char*)la.data(), la.size() * 4);
        } else
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
        if (use_prefill) eng->prefill(toks.data(), (int)toks.size(), 0);  // warm: allocates prefill scratch + cuBLAS
        eng->sync();
        eng->reset();
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        auto e0 = clk::now();
        int pos = 0;
        if (use_prefill) { eng->prefill(toks.data(), (int)toks.size(), 0); pos = (int)toks.size(); }
        else for (; pos < (int)toks.size(); ++pos) eng->step(toks[pos], pos);
        int next = eng->argmax();
        eng->sync();
        auto e1 = clk::now();
        auto e1b = e1;  // decode window starts here (re-taken after the untimed graph-capture re-run)
        std::vector<int> gen{next}, rest;
        auto td0 = clk::now();
        if (graph) {  // capture outside the timed region: one untimed decode, then restore by re-running the prompt
            eng->decode_greedy(next, pos, 1, rest, true);
            eng->reset();
            if (use_prefill) eng->prefill(toks.data(), (int)toks.size(), 0);
            else for (int p = 0; p < (int)toks.size(); ++p) eng->step(toks[p], p);
            eng->sync();
            e1b = clk::now();
            td0 = e1b;
        }
        eng->decode_greedy(next, pos, n - 1, rest, graph);
        gen.insert(gen.end(), rest.begin(), rest.end());
        eng->sync();
        auto e2 = clk::now();
        double td = std::chrono::duration<double>(e2 - td0).count();
        double pre_ms = ms(e0, e1), dec_ms = ms(e1b, e2);
#ifdef QV_HAVE_CUDA
        if (!cpu) cudaMemGetInfo(&fr, &tot);
#endif
        std::printf("{\"prompt_tokens\": %zu, \"prefill_ms\": %.2f, \"prefill_tok_s\": %.2f, \"decode_tokens\": %d, "
                    "\"decode_ms\": %.2f, \"decode_tok_s\": %.2f, \"decode_tok_s_wall\": %.2f, \"vram_used_gib\": %.3f, \"ids\": [",
                    toks.size(), pre_ms, toks.size() / (pre_ms / 1e3), n - 1, dec_ms, (n - 1) / (dec_ms / 1e3),
                    (n - 1) / td, (tot - fr) / 1073741824.0);
        for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? ", " : "", gen[i]);
        std::printf("]");
        {
            qv::MtpStats ms_;
            eng->mtp_stats(ms_);
            if (mtp)
                std::printf(", \"mtp\": {\"cycles\": %ld, \"accepted\": %ld, \"accept_rate\": %.4f, \"plain_steps\": %ld, "
                            "\"drafted\": %ld, \"draft_ms\": %.1f, \"verify_ms\": %.1f, \"plain_ms\": %.1f, \"prefill_mtp_ms\": %.1f}",
                            ms_.cycles, ms_.accepted, ms_.drafted ? (double)ms_.accepted / ms_.drafted : 0.0, ms_.plain_steps,
                            ms_.drafted, ms_.draft_ms, ms_.verify_ms, ms_.plain_ms, ms_.prefill_mtp_ms);
        }
        std::printf("}\n");
        return 0;
    }
    std::fprintf(stderr, "nothing to do (use --golden or --bench)\n");
    return 2;
}
