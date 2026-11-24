// Multi-sequence exactness test (CUDA engines): N sequences decoded interleaved on one engine must produce, per sequence, the SAME ids and
// bit-identical logits at every step as the sequence run alone on a single-sequence engine.
//   test_multiseq --model X.qv [--kv fp32|fp16|int4] [--ctx 4096] [--steps 48] [--chunk 512] [--seqs 4]
// Scenario (interleaving = one engine op per sequence per round, like the server's FIFO turn lock):
//   slots 0..2 start together (prefill chunks interleaved with the other slots' decode steps), slot 3 joins late (round 10),
//   slot 2 is ABORTED at step 15 (reset) and re-used with a fifth prompt. Greedy and fixed-seed sampling are both checked.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qv/engine.h"

static std::unique_ptr<qv::Engine> make(const qv::ModelFile& mf, const qv::EngineOptions& o, bool cpu) {
#ifdef QV_HAVE_CPU
    if (cpu) return qv::make_cpu_engine(mf, o);
#endif
    (void)cpu;
    return qv::make_cuda_engine(mf, o);
}

static uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = (const unsigned char*)v.data();
    for (size_t i = 0; i < v.size() * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
struct Run { std::vector<int> ids; std::vector<uint64_t> hs; std::vector<float> last; };

struct Seq {
    int slot = 0, pi = 0;           // engine sequence, prompt index
    std::vector<int> prompt;
    size_t off = 0;                 // prefill progress
    int gen = 0, tok = -1;          // generated count, pending token
    bool sampled = false, done = false;
    std::mt19937_64 rng;
    Run out;
    std::vector<float> lg;
};

static int pick(Seq& s, const std::vector<float>& lg) {
    if (!s.sampled) return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    // temperature 0.8 softmax over the top 40, fixed-seed RNG
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + 40, idx.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    double mx = lg[idx[0]], z = 0, p[40];
    for (int i = 0; i < 40; ++i) { p[i] = std::exp((lg[idx[i]] - mx) / 0.8); z += p[i]; }
    double u = std::uniform_real_distribution<double>(0, z)(s.rng), a = 0;
    for (int i = 0; i < 40; ++i) { a += p[i]; if (u <= a) return idx[i]; }
    return idx[0];
}

// one engine op for sequence s; returns false when the sequence has nothing left to do
static void turn(qv::Engine& e, Seq& s, int chunk, int steps, int V) {
    e.select_seq(s.slot);
    if (s.off < s.prompt.size()) {
        const int n = (int)std::min<size_t>(chunk, s.prompt.size() - s.off);
        e.prefill(s.prompt.data() + s.off, n, (int)s.off);
        s.off += n;
        if (s.off == s.prompt.size()) { e.logits(s.lg); s.out.hs.push_back(fnv(s.lg)); s.tok = pick(s, s.lg); s.out.ids.push_back(s.tok); s.gen = 1; }
        return;
    }
    if (s.gen >= steps) { s.out.last = s.lg; s.done = true; return; }
    e.step(s.tok, (int)(s.prompt.size() + s.gen - 1));
    e.logits(s.lg);
    s.out.hs.push_back(fnv(s.lg));
    s.tok = pick(s, s.lg);
    s.out.ids.push_back(s.tok);
    ++s.gen;
    (void)V;
}

int main(int argc, char** argv) {
    std::string model, kv = "fp32";
    int ctx = 4096, steps = 48, chunk = 512, nseq = 4;
    bool cpu = false, fresh = false;
    double lenscale = 1.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--model") model = nx(); else if (a == "--kv") kv = nx(); else if (a == "--ctx") ctx = std::stoi(nx());
        else if (a == "--steps") steps = std::stoi(nx()); else if (a == "--chunk") chunk = std::stoi(nx()); else if (a == "--seqs") nseq = std::stoi(nx());
        else if (a == "--cpu") cpu = true; else if (a == "--fresh") fresh = true; else if (a == "--lenscale") lenscale = std::stod(nx());
    }
    qv::ModelFile mf = qv::ModelFile::open(model);
    const int V = mf.cfg.vocab;
    const int lens0[5] = {37, 150, 900, 2600, 64};
    int lens[5];
    for (int i = 0; i < 5; ++i) lens[i] = std::max(8, (int)(lens0[i] * lenscale));
    std::vector<std::vector<int>> prompts(5);
    std::mt19937 g(1234);
    for (int i = 0; i < 5; ++i) for (int j = 0; j < lens[i]; ++j) prompts[i].push_back(1000 + (int)(g() % 20000));
    qv::EngineOptions opt;
    opt.threads = 16;
    opt.max_ctx = ctx; opt.prefill_chunk = chunk; opt.kv_mode = kv == "int4" ? 1 : kv == "fp16" ? 2 : 0;
    int bad = 0;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<Run> alone(5);
        {   // reference: single-sequence engine, one prompt at a time
            opt.n_seq = 1;
            auto e = make(mf, opt, cpu);
            for (int p = 0; p < 5; ++p) {
                if (fresh && p) e = make(mf, opt, cpu);   // a brand-new engine per prompt (no leftovers of earlier prompts)
                Seq s; s.prompt = prompts[p]; s.sampled = mode == 1; s.rng.seed(100 + p);
                e->reset();
                while (!s.done) turn(*e, s, chunk, steps, V);
                alone[p] = s.out;
            }
        }
        opt.n_seq = nseq;
        auto e = make(mf, opt, cpu);
        std::printf("mode=%s engine n_seq=%d state %.2f GiB weights %.2f GiB\n", mode ? "sampled" : "greedy", e->n_seq(), e->device_bytes_state() / 1073741824.0, e->device_bytes_weights() / 1073741824.0);
        std::vector<Seq> seqs(5);
        auto init = [&](int idx, int slot, int p) { seqs[idx] = Seq(); seqs[idx].slot = slot; seqs[idx].pi = p; seqs[idx].prompt = prompts[p]; seqs[idx].sampled = mode == 1; seqs[idx].rng.seed(100 + p); };
        init(0, 0, 0); init(1, 1, 1); init(2, 2, 2); init(3, 3, 3);
        for (int sl = 0; sl < nseq; ++sl) { e->select_seq(sl); e->reset(); }
        std::vector<char> active = {1, 1, 1, 0, 0};
        bool aborted = false;
        for (int round = 0; round < 100000; ++round) {
            if (round == 10) active[3] = 1;                                   // late join
            if (!aborted && seqs[2].gen == 15) {                              // abort slot 2 mid-generation, re-use it with prompt 4
                e->select_seq(2); e->reset();
                init(4, 2, 4); active[2] = 0; active[4] = 1; aborted = true;
            }
            bool any = false;
            for (int i = 0; i < 5; ++i) if (active[i] && !seqs[i].done) { turn(*e, seqs[i], chunk, steps, V); any = true; }
            if (!any) break;
        }
        for (int i : {0, 1, 3, 4}) {
            const Run& a = alone[seqs[i].pi];
            const bool ids_ok = a.ids == seqs[i].out.ids, hs_ok = a.hs == seqs[i].out.hs;
            double md = 0;
            for (size_t k = 0; k < a.last.size() && k < seqs[i].out.last.size(); ++k) md = std::max(md, (double)std::fabs(a.last[k] - seqs[i].out.last[k]));
            bool last_ok = a.last.size() == seqs[i].out.last.size() && std::memcmp(a.last.data(), seqs[i].out.last.data(), a.last.size() * 4) == 0;
            std::printf("  prompt %d (len %zu) on slot %d: ids %s, per-step logits %s, final logits %s (max|d|=%g), %zu steps\n", seqs[i].pi, prompts[seqs[i].pi].size(), seqs[i].slot,
                        ids_ok ? "IDENTICAL" : "DIFFER", hs_ok ? "BIT-IDENTICAL" : "DIFFER", last_ok ? "BIT-IDENTICAL" : "DIFFER", md, a.ids.size());
            if (!ids_ok || !hs_ok || !last_ok) ++bad;
        }
    }
    std::printf(bad ? "FAIL (%d)\n" : "PASS\n", bad);
    return bad ? 1 : 0;
}
