// llama_eval: teacher-forced eval of a token file with libllama, protocol-identical to `qv_run --eval`.
//  * tokens are given as int32 ids (no re-tokenisation), so both engines see the same ids
//  * ppl = exp(mean NLL of token[i+1] given tokens[0..i]) over all predicted positions
//  * --mode write  : write the reference file in qv_run's format (int32 T, int32 topk, then per row:
//                    topk int32 ids, topk float32 logprobs, float32 nll)
//  * --mode compare: KL(ref||this) over the ref's top-k ids plus the tail-mass term, top-1 agreement
//                    (the same formula as qv_run --eval-mode compare)
//  * --window W    : split the file into independent windows of W tokens (KV/state reset per window); ppl only
// Build: see benchmarks/compare/scripts/build_llama_eval.sh
#include "llama.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

static std::vector<int> read_i32(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb"); if (!f) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<int> v(n / 4); if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(2); std::fclose(f); return v;
}

int main(int argc, char** argv) {
    std::string model, tokf, mode = "none", ref, tag;
    int ngl = 99, topk = 1024, window = 0, nb = 512, nthreads = 8, probe_ctx = 0; std::string kvt = "f16";
    bool fa = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model = nx(); else if (a == "--tokens") tokf = nx(); else if (a == "--mode") mode = nx();
        else if (a == "--ref") ref = nx(); else if (a == "-ngl") ngl = std::stoi(nx()); else if (a == "--topk") topk = std::stoi(nx());
        else if (a == "--window") window = std::stoi(nx()); else if (a == "-b") nb = std::stoi(nx());
        else if (a == "-t") nthreads = std::stoi(nx()); else if (a == "--no-fa") fa = false; else if (a == "--tag") tag = nx(); else if (a == "--probe-ctx") probe_ctx = std::stoi(nx()); else if (a == "--kv") kvt = nx();
    }
    std::vector<int> toks = read_i32(tokf);
    const int T = (int)toks.size(), W = window > 0 ? window : T;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { std::fprintf(stderr, "load failed\n"); return 2; }
    const llama_vocab* vocab = llama_model_get_vocab(m);
    const int V = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    if (probe_ctx > 0) {  // memory probe: can a context of probe_ctx tokens with this KV type be created and used on this GPU?
        cp.n_ctx = probe_ctx;
        cp.type_k = cp.type_v = kvt == "q8_0" ? GGML_TYPE_Q8_0 : kvt == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
    } else cp.n_ctx = W;
    cp.n_batch = nb; cp.n_ubatch = nb; cp.n_threads = nthreads; cp.n_threads_batch = nthreads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(m, cp);
    if (!ctx) { std::fprintf(stderr, "ctx failed\n"); return 2; }
    if (probe_ctx > 0) {
        llama_batch pb = llama_batch_init(nb, 0, 1);
        int fail = 0;
        for (int p0 = 0; p0 < std::min(T, 2048) && !fail; p0 += nb) {  // a real prefill chunk, then a decode step
            const int cnt = std::min(nb, std::min(T, 2048) - p0); pb.n_tokens = cnt;
            for (int j = 0; j < cnt; ++j) { pb.token[j] = toks[p0 + j]; pb.pos[j] = p0 + j; pb.n_seq_id[j] = 1; pb.seq_id[j][0] = 0; pb.logits[j] = (j == cnt - 1); }
            fail |= llama_decode(ctx, pb) != 0;
        }
        std::printf("{\"engine\": \"llama.cpp\", \"probe_ctx\": %d, \"kv\": \"%s\", \"ok\": %s}\n", probe_ctx, kvt.c_str(), fail ? "false" : "true");
        return fail ? 3 : 0;
    }
    FILE* wf = nullptr; std::vector<int> rid; std::vector<float> rlp, rnll;
    if (mode == "write") { wf = std::fopen(ref.c_str(), "wb"); int h[2] = {T, topk}; std::fwrite(h, 4, 2, wf); }
    if (mode == "compare") {
        FILE* f = std::fopen(ref.c_str(), "rb"); int h[2];
        if (!f || std::fread(h, 4, 2, f) != 2 || h[0] != T || h[1] != topk) { std::fprintf(stderr, "ref header mismatch\n"); return 2; }
        rid.resize((size_t)T * topk); rlp.resize((size_t)T * topk); rnll.resize(T);
        for (int r = 0; r < T - 1; ++r)
            if (std::fread(&rid[(size_t)r * topk], 4, topk, f) != (size_t)topk || std::fread(&rlp[(size_t)r * topk], 4, topk, f) != (size_t)topk ||
                std::fread(&rnll[r], 4, 1, f) != 1) { std::fprintf(stderr, "short ref\n"); return 2; }
        std::fclose(f);
    }
    double nll = 0, kl = 0, tail = 0, nllref = 0; long n = 0, top1 = 0;
    std::vector<int> ord(V);
    llama_batch b = llama_batch_init(nb, 0, 1);
    double t0 = 0;
    for (int w0 = 0; w0 + W <= T || (w0 == 0 && W == T); w0 += W) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p0 = 0; p0 < W; p0 += nb) {
            const int cnt = std::min(nb, W - p0);
            b.n_tokens = cnt;
            for (int j = 0; j < cnt; ++j) { b.token[j] = toks[w0 + p0 + j]; b.pos[j] = p0 + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1; }
            if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 2; }
            for (int j = 0; j < cnt; ++j) {
                const int pos = p0 + j; if (pos >= W - 1) continue;
                const float* lg = llama_get_logits_ith(ctx, j);
                float mx = -INFINITY; for (int i = 0; i < V; ++i) mx = std::max(mx, lg[i]);
                double se = 0; for (int i = 0; i < V; ++i) se += std::exp((double)lg[i] - mx);
                const double lse = mx + std::log(se), x = lse - lg[toks[w0 + pos + 1]];
                nll += x; n++;
                const int row = w0 + pos;
                if (mode == "write") {
                    std::iota(ord.begin(), ord.end(), 0);
                    std::partial_sort(ord.begin(), ord.begin() + topk, ord.end(), [&](int a, int c) { return lg[a] > lg[c]; });
                    std::vector<float> lp(topk); for (int k = 0; k < topk; ++k) lp[k] = (float)(lg[ord[k]] - lse);
                    std::fwrite(ord.data(), 4, topk, wf); std::fwrite(lp.data(), 4, topk, wf); float f = (float)x; std::fwrite(&f, 4, 1, wf);
                } else if (mode == "compare") {
                    const int* ids = &rid[(size_t)row * topk]; const float* rl = &rlp[(size_t)row * topk];
                    double k_ = 0, ps = 0, qs = 0;
                    for (int k = 0; k < topk; ++k) { const double lq = (double)lg[ids[k]] - lse, p = std::exp((double)rl[k]); k_ += p * ((double)rl[k] - lq); ps += p; qs += std::exp(lq); }
                    const double pt = std::max(0.0, 1.0 - ps), qt = std::max(1e-12, 1.0 - qs);
                    if (pt > 1e-12) k_ += pt * std::log(pt / qt);
                    kl += k_; tail += pt; nllref += rnll[row];
                    int am = 0; for (int i = 1; i < V; ++i) if (lg[i] > lg[am]) am = i;
                    top1 += (am == ids[0]);
                }
            }
        }
        if (w0 + W >= T) break;
    }
    if (wf) std::fclose(wf);
    std::printf("{\"engine\": \"llama.cpp\", \"tag\": \"%s\", \"mode\": \"%s\", \"tokens\": %d, \"window\": %d, \"scored\": %ld, \"ppl\": %.5f, \"mean_nll\": %.6f", tag.c_str(), mode.c_str(), T, W, n, std::exp(nll / n), nll / n);
    if (mode == "compare") std::printf(", \"ref_ppl\": %.5f, \"kl_topk_mean\": %.6f, \"top1_agree\": %.4f, \"ref_tail_mass_mean\": %.6f", std::exp(nllref / n), kl / n, (double)top1 / n, tail / n);
    std::printf("}\n");
    (void)t0;
    llama_batch_free(b); llama_free(ctx); llama_model_free(m);
    return 0;
}
