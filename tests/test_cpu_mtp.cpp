// MTP speculative decoding must reproduce plain greedy decoding exactly.
// Usage: test_cpu_mtp MODEL_PREFIX [N_GENERATED=64] [THREADS=32]
// Runs a short prompt (and a repetitive one, which gets many accepted drafts) through plain greedy decode and through
// the MTP path on the same engine (set_mtp_enabled), and requires identical token ids. Skips (exit 0) without a model.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "qv/engine.h"

static std::vector<int> run(qv::Engine& e, const std::vector<int>& prompt, int n, bool mtp) {
    e.reset();
    e.set_mtp_enabled(mtp);
    e.prefill(prompt.data(), (int)prompt.size(), 0);
    std::vector<int> out{e.argmax()}, rest;
    e.decode_greedy(out[0], (int)prompt.size(), n - 1, rest, false);
    out.insert(out.end(), rest.begin(), rest.end());
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2 || !std::ifstream(std::string(argv[1]) + ".qv.json")) { std::printf("test_cpu_mtp: no model, skipped\n"); return 0; }
    const int n = argc > 2 ? std::atoi(argv[2]) : 64;
    qv::EngineOptions opt;
    opt.max_ctx = 1024;
    opt.threads = argc > 3 ? std::atoi(argv[3]) : 32;
    opt.mtp = true;
    auto mf = qv::ModelFile::open(argv[1]);
    auto eng = qv::make_cpu_engine(mf, opt);
    // short chat-like prompt, then a repetitive one
    std::vector<std::vector<int>> prompts{{248045, 846, 198, 9707, 11, 1246, 525, 498, 30, 248046, 198, 248045, 74455, 198}};
    std::vector<int> rep;
    for (int i = 0; i < 24; ++i) for (int t : {1000, 2000, 3000, 4000, 5000}) rep.push_back(t);
    prompts.push_back(rep);
    int fail = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        auto a = run(*eng, prompts[i], n, false);
        auto b = run(*eng, prompts[i], n, true);
        qv::MtpStats st;
        eng->mtp_stats(st);
        bool same = a == b;
        std::printf("prompt %zu (%zu tokens): %d tokens, identical=%d, cycles=%ld accepted=%ld\n", i, prompts[i].size(), n, (int)same,
                    st.cycles, st.accepted);
        if (!same) {
            ++fail;
            for (int k = 0; k < n; ++k) if (a[k] != b[k]) { std::printf("  first divergence at generated token %d: plain=%d mtp=%d\n", k, a[k], b[k]); break; }
        }
    }
    return fail != 0;
}
