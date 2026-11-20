// CUDA MTP speculative decoding must reproduce plain greedy decoding exactly (ids AND final logits bit for bit).
// Usage: test_cuda_mtp MODEL [N_GENERATED=64] [DEPTHS=1,3] [PROMPT.i32 ...]
// For every depth a fresh engine runs each prompt through plain greedy decode and through the MTP path
// (set_mtp_enabled) and requires identical token ids, identical logits of the last processed token (state rollback
// check) and, when MTP was active, at least one verify cycle. Skips (exit 0) without a model. Needs the GPU (device 0).
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (argc < 2 || !std::ifstream(argv[1]) && !std::ifstream(std::string(argv[1]) + ".qv.json")) {
        std::printf("test_cuda_mtp: no model, skipped\n");
        return 0;
    }
    const int n = argc > 2 ? std::atoi(argv[2]) : 64;
    std::vector<int> depths;
    {
        std::stringstream ss(argc > 3 ? argv[3] : "1,3");
        std::string tok;
        while (std::getline(ss, tok, ',')) depths.push_back(std::atoi(tok.c_str()));
    }
    std::vector<std::vector<int>> prompts{{248045, 846, 198, 9707, 11, 1246, 525, 498, 30, 248046, 198, 248045, 74455, 198}};
    std::vector<int> rep;
    for (int i = 0; i < 24; ++i) for (int t : {1000, 2000, 3000, 4000, 5000}) rep.push_back(t);
    prompts.push_back(rep);
    for (int i = 4; i < argc; ++i) prompts.push_back(read_i32(argv[i]));
    auto mf = qv::ModelFile::open(argv[1]);
    int fail = 0;
    for (int depth : depths) {
        qv::EngineOptions opt;
        opt.max_ctx = std::getenv("QV_TEST_MAXCTX") ? std::atoi(std::getenv("QV_TEST_MAXCTX")) : 2048;
        opt.kv_mode = std::getenv("QV_TEST_KV") ? std::atoi(std::getenv("QV_TEST_KV")) : 0;  // 0 fp32, 1 int4, 2 fp16
        opt.kv4_kmode = std::getenv("QV_TEST_KMODE") ? std::atoi(std::getenv("QV_TEST_KMODE")) : 0;
        opt.mtp = true;
        opt.mtp_depth = depth;
        auto eng = qv::make_cuda_engine(mf, opt);
        for (size_t i = 0; i < prompts.size(); ++i) {
            Res a = run(*eng, prompts[i], n, false);
            Res b = run(*eng, prompts[i], n, true);
            qv::MtpStats st;
            eng->mtp_stats(st);
            const bool same = a.ids == b.ids;
            const bool lsame = a.logits == b.logits;
            const bool active = st.cycles > 0;
            std::printf("depth %d prompt %zu (%zu tokens): %d tokens, ids identical=%d, final logits identical=%d, cycles=%ld drafted=%ld accepted=%ld\n",
                        depth, i, prompts[i].size(), n, (int)same, (int)lsame, st.cycles, st.drafted, st.accepted);
            if (!same) {
                for (int k = 0; k < n; ++k) if (a.ids[k] != b.ids[k]) { std::printf("  first divergence at generated token %d: plain=%d mtp=%d\n", k, a.ids[k], b.ids[k]); break; }
            }
            if (!same || !lsame || !active) ++fail;
        }
    }
    std::printf(fail ? "FAIL\n" : "PASS\n");
    return fail != 0;
}
