// Inkling CUDA MTP (DSpark drafter + batched exact verify) must reproduce plain greedy decoding exactly:
// token ids AND the logits of one extra step after the generation (checks conv-state / KV rollback bit for bit).
// Usage: test_ink_mtp MODEL.qv MTP.qv [N=64] [DEPTHS=7] PROMPT.i32 ...    Needs the GPU (device 0); skips (exit 0) without a model.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "qv/engine.h"

struct Res { std::vector<int> ids; std::vector<float> logits; };

static Res run(qv::Engine& e, const std::vector<int>& prompt, int n, bool mtp) {
    e.reset();
    e.set_mtp_enabled(mtp);
    e.prefill(prompt.data(), (int)prompt.size(), 0);
    Res r;
    r.ids = {e.argmax()};
    std::vector<int> rest;
    e.decode_greedy(r.ids[0], (int)prompt.size(), n - 1, rest, false);
    r.ids.insert(r.ids.end(), rest.begin(), rest.end());
    e.step(r.ids.back(), (int)prompt.size() + n - 1);  // state check: the next logits must also agree
    e.logits(r.logits);
    return r;
}
static std::vector<int> read_i32(const char* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}
int main(int argc, char** argv) {
    if (argc < 5 || !std::ifstream(argv[1])) { std::printf("test_ink_mtp: no model, skipped\n"); return 0; }
    const int n = std::atoi(argv[3]);
    std::vector<int> depths;
    { std::stringstream ss(argv[4]); std::string t; while (std::getline(ss, t, ',')) depths.push_back(std::atoi(t.c_str())); }
    auto mf = qv::ModelFile::open(argv[1]);
    int fail = 0;
    for (int depth : depths) {
        qv::EngineOptions opt;
        opt.max_ctx = std::getenv("QV_TEST_MAXCTX") ? std::atoi(std::getenv("QV_TEST_MAXCTX")) : 8192;
        opt.kv_mode = 1;
        opt.mtp = true;
        opt.mtp_depth = depth;
        opt.mtp_file = argv[2];
        auto eng = qv::make_cuda_engine(mf, opt);
        for (int i = 5; i < argc; ++i) {
            auto prompt = read_i32(argv[i]);
            Res a = run(*eng, prompt, n, false);
            qv::MtpStats s0;
            eng->mtp_stats(s0);
            Res b = run(*eng, prompt, n, true);
            qv::MtpStats st;
            eng->mtp_stats(st);
            const bool same = a.ids == b.ids, lsame = a.logits == b.logits;
            std::printf("depth %d %s (%zu tokens): %d tokens, ids identical=%d, next-step logits identical=%d, cycles=%ld drafted=%ld accepted=%ld (mean accepted/cycle %.2f) plain_steps=%ld\n",
                        depth, argv[i], prompt.size(), n, (int)same, (int)lsame, st.cycles - s0.cycles, st.drafted - s0.drafted, st.accepted - s0.accepted,
                        st.cycles > s0.cycles ? (double)(st.accepted - s0.accepted) / (st.cycles - s0.cycles) : 0.0, st.plain_steps - s0.plain_steps);
            if (!same) for (int k = 0; k < n; ++k) if (a.ids[k] != b.ids[k]) { std::printf("  first divergence at generated token %d: plain=%d mtp=%d\n", k, a.ids[k], b.ids[k]); break; }
            if (!same || !lsame || st.cycles == s0.cycles) ++fail;
        }
    }
    std::printf(fail ? "FAIL\n" : "PASS\n");
    return fail != 0;
}
