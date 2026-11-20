// qv_run — driver for correctness dumps and decode benchmarks.
//   qv_run --model PREFIX --golden TOKENS.i32 --out DUMP.bin      (hidden states after embed+each layer, + logits)
//   qv_run --backend cuda|cpu [--threads N] ...   (cpu: no GPU is touched; qv_run_cpu is the same source built without CUDA)
//   qv_run --model PREFIX --bench --prompt TOKENS.i32 [--n 128] [--max-ctx 4096]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
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
    std::string model, golden, out, prompt;
    bool bench = false, graph = false, use_prefill = false, mtp = false;
#ifdef QV_HAVE_CUDA
    std::string backend = "cuda";
#else
    std::string backend = "cpu";
#endif
    int threads = 32;
    std::string kv = "fp32", kmode = "channel", eval_toks, eval_ref, eval_mode = "write";
    int eval_topk = 1024;
    bool gen = false, dbench = false;
    int synth_ctx = 0;
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
        else if (a == "--kv") kv = next();                 // fp32 (default, the original path) | fp16 (fp16 KV, 64 KiB/token) | int4
        else if (a == "--kv4-kmode") kmode = next();       // channel | token
        else if (a == "--eval") eval_toks = next();        // TOKENS.i32: teacher-forced prefill; ppl, and KL vs a reference run
        else if (a == "--eval-ref") eval_ref = next();     // file written (--eval-mode write) or read (compare)
        else if (a == "--eval-mode") eval_mode = next();
        else if (a == "--eval-topk") eval_topk = std::stoi(next());
        else if (a == "--decode-bench") dbench = true;     // prefill --prompt once, untimed warm-up decode, timed decode of --n tokens
        else if (a == "--synth-ctx") synth_ctx = std::stoi(next());  // decode-bench: decode at this position over UNWRITTEN KV (speed/VRAM only)
        else if (a == "--gen") gen = true;                 // prefill --prompt once, greedy-decode --n tokens, print ids
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
    if (kv != "fp32" && kv != "fp16" && kv != "int4") { std::fprintf(stderr, "--kv fp32|fp16|int4\n"); return 2; }
    opt.kv_mode = kv == "int4" ? 1 : kv == "fp16" ? 2 : 0;
    opt.kv4_kmode = kmode == "token" ? 1 : 0;
    // ---- eval mode state ----
    std::vector<int> etoks;
    struct EvalAcc { double nll = 0, kl = 0, tail = 0, nll_ref = 0; long n = 0, top1 = 0; } ev;
    std::vector<int> ref_ids; std::vector<float> ref_lp, ref_nll;  // compare mode: loaded reference
    FILE* wref = nullptr;
    std::vector<int> ord;
    if (!eval_toks.empty()) {
        etoks = read_i32(eval_toks);
        const int T = (int)etoks.size();
        if (eval_mode == "write") {
            wref = std::fopen(eval_ref.c_str(), "wb");
            if (!wref) { std::fprintf(stderr, "cannot write %s\n", eval_ref.c_str()); return 2; }
        } else {
            FILE* f = std::fopen(eval_ref.c_str(), "rb");
            if (!f) { std::fprintf(stderr, "cannot read %s\n", eval_ref.c_str()); return 2; }
            int hdr[2];
            if (std::fread(hdr, 4, 2, f) != 2 || hdr[0] != T || hdr[1] != eval_topk) { std::fprintf(stderr, "ref header mismatch\n"); return 2; }
            ref_ids.resize((size_t)T * eval_topk); ref_lp.resize((size_t)T * eval_topk); ref_nll.resize(T);
            for (int r = 0; r < T - 1; ++r) {
                if (std::fread(&ref_ids[(size_t)r * eval_topk], 4, eval_topk, f) != (size_t)eval_topk ||
                    std::fread(&ref_lp[(size_t)r * eval_topk], 4, eval_topk, f) != (size_t)eval_topk ||
                    std::fread(&ref_nll[r], 4, 1, f) != 1) { std::fprintf(stderr, "short ref\n"); return 2; }
            }
            std::fclose(f);
        }
        if (wref) { int hdr[2] = {T, eval_topk}; std::fwrite(hdr, 4, 2, wref); }
        opt.row_logits = [&](int pos, const float* lg) {
            if (pos >= (int)etoks.size() - 1) return;
            const int Vn = (int)ord.size();
            float mx = -INFINITY;
            for (int i = 0; i < Vn; ++i) mx = std::max(mx, lg[i]);
            double se = 0;
            for (int i = 0; i < Vn; ++i) se += std::exp((double)lg[i] - mx);
            const double lse = mx + std::log(se);
            const double nll = lse - lg[etoks[pos + 1]];
            ev.nll += nll; ev.n++;
            if (wref) {
                std::iota(ord.begin(), ord.end(), 0);
                std::partial_sort(ord.begin(), ord.begin() + eval_topk, ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });
                std::vector<float> lp(eval_topk);
                for (int k = 0; k < eval_topk; ++k) lp[k] = (float)(lg[ord[k]] - lse);
                std::fwrite(ord.data(), 4, eval_topk, wref); std::fwrite(lp.data(), 4, eval_topk, wref);
                float f = (float)nll; std::fwrite(&f, 4, 1, wref);
            } else {
                const int* ids = &ref_ids[(size_t)pos * eval_topk];
                const float* rlp = &ref_lp[(size_t)pos * eval_topk];
                double kl = 0, ps = 0, qs = 0;
                for (int k = 0; k < eval_topk; ++k) {
                    const double lq = (double)lg[ids[k]] - lse, p = std::exp((double)rlp[k]);
                    kl += p * ((double)rlp[k] - lq); ps += p; qs += std::exp(lq);
                }
                const double pt = std::max(0.0, 1.0 - ps), qt = std::max(1e-12, 1.0 - qs);
                if (pt > 1e-12) kl += pt * std::log(pt / qt);
                ev.kl += kl; ev.tail += pt; ev.nll_ref += ref_nll[pos];
                int am = 0; for (int i = 1; i < Vn; ++i) if (lg[i] > lg[am]) am = i;
                ev.top1 += (am == ids[0]);
            }
        };
    }
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
    ord.assign(eng->config().vocab, 0);
    eng->sync();
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t fr = 0, tot = 0;
#ifdef QV_HAVE_CUDA
    if (!cpu) cudaMemGetInfo(&fr, &tot);
#endif
    std::fprintf(stderr, "loaded in %.1fs: weights %.2f GiB, state+scratch %.2f GiB, device used %.2f / %.2f GiB\n", load_s,
                 eng->device_bytes_weights() / 1073741824.0, eng->device_bytes_state() / 1073741824.0,
                 (tot - fr) / 1073741824.0, tot / 1073741824.0);

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
    if (!etoks.empty()) {
        auto e0 = std::chrono::steady_clock::now();
        eng->prefill(etoks.data(), (int)etoks.size(), 0);
        eng->sync();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
        if (wref) std::fclose(wref);
        std::printf("{\"mode\": \"%s\", \"kv\": \"%s\", \"kmode\": \"%s\", \"tokens\": %zu, \"scored\": %ld, \"ppl\": %.5f, \"mean_nll\": %.6f",
                    eval_mode.c_str(), kv.c_str(), kmode.c_str(), etoks.size(), ev.n, std::exp(ev.nll / ev.n), ev.nll / ev.n);
        if (!wref) std::printf(", \"ref_ppl\": %.5f, \"kl_topk_mean\": %.6f, \"top1_agree\": %.4f, \"ref_tail_mass_mean\": %.6f",
                               std::exp(ev.nll_ref / ev.n), ev.kl / ev.n, (double)ev.top1 / ev.n, ev.tail / ev.n);
        std::printf(", \"seconds\": %.1f}\n", sec);
        return 0;
    }
    if (dbench) {
        using clk = std::chrono::steady_clock;
        std::vector<int> toks = read_i32(prompt);
        auto e0 = clk::now();
        eng->prefill(toks.data(), (int)toks.size(), 0);
        int next = eng->argmax();
        eng->sync();
        const double pre_ms = std::chrono::duration<double, std::milli>(clk::now() - e0).count();
        int pos = (int)toks.size();
        if (synth_ctx > pos) pos = synth_ctx;
        std::vector<int> w, rest;
        const int nw = mtp ? 8 : 1;  // untimed warm-up: captures the CUDA graphs
        eng->decode_greedy(next, pos, nw, w, true);
        eng->sync();
        auto t0 = clk::now();
        eng->decode_greedy(w.back(), pos + nw, n, rest, true);
        eng->sync();
        const double dec_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        size_t fr2 = 0, tot2 = 0;
#ifdef QV_HAVE_CUDA
        cudaMemGetInfo(&fr2, &tot2);
#endif
        std::printf("{\"kv\": \"%s\", \"kmode\": \"%s\", \"mtp_depth\": %d, \"prompt_tokens\": %zu, \"decode_pos\": %d, \"synthetic_kv\": %s, "
                    "\"prefill_ms\": %.1f, \"prefill_tok_s\": %.1f, \"decode_tokens\": %d, \"decode_ms\": %.2f, \"decode_tok_s\": %.2f, "
                    "\"vram_used_gib\": %.3f, \"ids\": [",
                    kv.c_str(), kmode.c_str(), mtp ? mtp_depth : 0, toks.size(), pos + nw, synth_ctx > (int)toks.size() ? "true" : "false",
                    pre_ms, toks.size() / (pre_ms / 1e3), n, dec_ms, n / (dec_ms / 1e3), (tot2 - fr2) / 1073741824.0);
        for (size_t i = 0; i < rest.size(); ++i) std::printf("%s%d", i ? ", " : "", rest[i]);
        std::printf("]");
        qv::MtpStats ms_;
        eng->mtp_stats(ms_);
        if (mtp) std::printf(", \"mtp\": {\"cycles\": %ld, \"accepted\": %ld, \"drafted\": %ld, \"accept_rate\": %.4f}", ms_.cycles, ms_.accepted, ms_.drafted, ms_.drafted ? (double)ms_.accepted / ms_.drafted : 0.0);
        std::printf("}\n");
        return 0;
    }
    if (gen) {  // --prompt a.i32,b.i32,...: for each, reset, prefill, greedy decode n tokens, one JSON line
        std::vector<std::string> files;
        for (size_t st = 0; st <= prompt.size();) {
            size_t e = prompt.find(',', st);
            if (e == std::string::npos) e = prompt.size();
            if (e > st) files.push_back(prompt.substr(st, e - st));
            st = e + 1;
        }
        for (const auto& pf : files) {
            std::vector<int> toks = read_i32(pf);
            eng->reset();
            auto e0 = std::chrono::steady_clock::now();
            eng->prefill(toks.data(), (int)toks.size(), 0);
            int next = eng->argmax();
            eng->sync();
            double pre = std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
            std::vector<int> rest, ids{next};
            eng->decode_greedy(next, (int)toks.size(), n - 1, rest, false);
            ids.insert(ids.end(), rest.begin(), rest.end());
            std::printf("{\"file\": \"%s\", \"prompt_tokens\": %zu, \"prefill_s\": %.2f, \"ids\": [", pf.c_str(), toks.size(), pre);
            for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
            std::printf("]}\n");
            std::fflush(stdout);
        }
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
