// Bit-exactness regression for Inkling decode fusions (p26/ink-fuse): per prompt, prefill, then N greedy steps through step() (one graph per
// token) and through decode_greedy(graph). Prints one FNV-1a hash of the full logits row per step plus the id, and the ids of decode_greedy.
// Compare the output of the baseline binary and the new binary with diff: identical output == bit-identical logits and ids.
// Usage: test_ink_exact MODEL.qv N PROMPT.i32 ...   (device 0; skipped without a model)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "qv/engine.h"

static std::vector<int> read_i32(const char* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}
static uint64_t fnv(const std::vector<float>& l) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = (const unsigned char*)l.data();
    for (size_t i = 0; i < l.size() * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
int main(int argc, char** argv) {
    if (argc < 4 || !std::ifstream(argv[1])) { std::printf("test_ink_exact: no model, skipped\n"); return 0; }
    const int n = std::atoi(argv[2]);
    auto mf = qv::ModelFile::open(argv[1]);
    qv::EngineOptions opt;
    opt.max_ctx = std::getenv("QV_TEST_MAXCTX") ? std::atoi(std::getenv("QV_TEST_MAXCTX")) : 8192;
    opt.kv_mode = 1;
    auto eng = qv::make_cuda_engine(mf, opt);
    for (int i = 3; i < argc; ++i) {
        auto prompt = read_i32(argv[i]);
        eng->reset();
        eng->prefill(prompt.data(), (int)prompt.size(), 0);
        std::vector<float> lg;
        eng->logits(lg);
        std::printf("%s prefill %016llx\n", argv[i], (unsigned long long)fnv(lg));
        int tok = eng->argmax();
        const int first = tok;
        for (int k = 0; k < n; ++k) {
            eng->step(tok, (int)prompt.size() + k);
            eng->logits(lg);
            tok = eng->argmax();
            std::printf("step %d logits %016llx next %d\n", k, (unsigned long long)fnv(lg), tok);
        }
        eng->reset();
        eng->prefill(prompt.data(), (int)prompt.size(), 0);
        std::vector<int> ids;
        eng->decode_greedy(first, (int)prompt.size(), n, ids, true);
        std::printf("graph ids:");
        for (int v : ids) std::printf(" %d", v);
        std::printf("\n");
        eng->logits(lg);
        std::printf("graph final logits %016llx\n", (unsigned long long)fnv(lg));
    }
    return 0;
}
