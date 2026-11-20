// Public inference API (backend-independent).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "qv/model.h"

namespace qv {

struct EngineOptions {
    int max_ctx = 4096;
    int prefill_chunk = 2048;  // tokens per batched prefill chunk
    int threads = 32;          // CPU backend: worker threads incl. the caller (RAM bandwidth saturates near 32 here)
    bool mtp = false;          // load the MTP head and use it for exact speculative greedy decoding (CPU and CUDA backends)
    int mtp_depth = 1;         // CUDA backend: drafts per verify cycle (1..3); verify processes depth+1 tokens
    bool dump_hidden = false;  // record residual stream after embed + each layer (debug / golden tests)
};

// Counters of the MTP speculative decode path (all zero when MTP is off).
struct MtpStats {
    long cycles = 0;      // verify passes
    long accepted = 0;    // drafts accepted
    long drafted = 0;     // drafts verified (CUDA backend; CPU: == cycles)
    long plain_steps = 0; // single-token steps (tail of the generation)
    double draft_ms = 0, verify_ms = 0, plain_ms = 0, prefill_mtp_ms = 0;
};

class Engine {
   public:
    virtual ~Engine() = default;
    // Process one token at position `pos` (== number of tokens already in the context).
    virtual void step(int token, int pos) = 0;
    // Batched prefill of T tokens at positions [pos0, pos0+T); leaves logits of the last token ready.
    virtual void prefill(const int* tokens, int T, int pos0) = 0;
    // Greedy decode n tokens; first = token at start_pos (argmax of the logits after the context).
    virtual void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool use_graph) = 0;
    virtual int argmax() = 0;                         // greedy next token from the last step's logits
    virtual void logits(std::vector<float>& out) = 0; // copy last logits to host
    virtual void reset() = 0;                         // clear recurrent/KV state
    virtual const std::vector<float>& hidden_dump() const = 0;
    virtual const std::vector<float>& logits_dump() const = 0;  // per-token logits of prefill() when dump_hidden
    virtual void sync() = 0;
    virtual void set_mtp_enabled(bool) {}  // runtime switch (needs EngineOptions::mtp at construction)
    virtual void mtp_stats(MtpStats& s) const { s = MtpStats(); }
    virtual size_t device_bytes_weights() const = 0;
    virtual size_t device_bytes_state() const = 0;
    const Config& config() const { return cfg_; }

   protected:
    Config cfg_;
};

std::unique_ptr<Engine> make_cuda_engine(const ModelFile& mf, const EngineOptions& opt);  // needs the qv_cuda library
std::unique_ptr<Engine> make_cpu_engine(const ModelFile& mf, const EngineOptions& opt);   // needs qv_cpu_engine

}  // namespace qv
