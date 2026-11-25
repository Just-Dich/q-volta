// Scripted engine for HTTP-layer tests (`qv_server --backend mock`): no model weights. After a prompt it replays a fixed token
// script (the tokenisation of --mock-text) and then emits EOS (or loops with --mock-loop). Logits are peaked on the scripted token with a
// few deterministic runners-up, so top-k / top-p / logprobs / seeded sampling behave like on a real model. Honours the
// engine contract (prefill leaves logits of the last row, step(token,pos), reset()).
#pragma once
#include <chrono>
#include <cmath>
#include <thread>

#include "qv/engine.h"

namespace qv {

class MockEngine : public Engine {
   public:
    MockEngine(int vocab, std::vector<int> script, int eos, bool loop, int delay_us, int n_seq = 1) : script_(std::move(script)), eos_(eos), loop_(loop), delay_us_(delay_us), st_(std::max(1, n_seq)) {
        cfg_.vocab = vocab;
        cfg_.eos = eos;
        cfg_.arch = "mock";
        if (script_.empty()) script_.push_back(eos);
    }
    int n_seq() const override { return (int)st_.size(); }
    void select_seq(int i) override { if (i < 0 || i >= (int)st_.size()) throw std::runtime_error("sequence index out of range"); cur_ = i; }
    void step(int, int pos) override { delay(1); st_[cur_].k = pos + 1 - st_[cur_].prompt_end; }
    void prefill(const int*, int T, int pos0) override { delay(T / 8 + 1); st_[cur_].prompt_end = pos0 + T; st_[cur_].k = 0; }
    void prefill_embeds(const int*, const float*, const unsigned char* mask, int T, int pos0) override {
        for (int i = 0; i < T; ++i) n_embeds_ += mask[i] ? 1 : 0;
        prefill(nullptr, T, pos0);
    }
    long n_embeds_ = 0;  // image rows received through prefill_embeds (tests)
    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool) override {
        out.resize(n);
        int tok = first;
        for (int i = 0; i < n; ++i) { step(tok, start_pos + i); tok = argmax(); out[i] = tok; }
    }
    int argmax() override { return scripted(); }
    void logits(std::vector<float>& out) override {
        const int V = cfg_.vocab;
        out.resize(V);
        for (int i = 0; i < V; ++i) out[i] = -6.0f + (float)((uint32_t)(i * 2654435761u) >> 20) / 4096.0f * 2.0f;  // deterministic noise in [-6,-4)
        int t = scripted();
        out[t] = 20.0f;
        const float ru[3] = {18.0f, 17.0f, 16.0f};
        for (int j = 0; j < 3; ++j) { int a = (t + 1 + j * 7919) % V; if (a != t) out[a] = ru[j]; }
    }
    void reset() override { st_[cur_].k = 0; st_[cur_].prompt_end = 0; }
    bool has_state_save() const override { return true; }
    void state_save() override { saved_ = st_[cur_]; }
    void state_restore() override { st_[cur_] = saved_; }
    const std::vector<float>& hidden_dump() const override { return empty_; }
    const std::vector<float>& logits_dump() const override { return empty_; }
    void sync() override {}
    size_t device_bytes_weights() const override { return 0; }
    size_t device_bytes_state() const override { return 0; }

   private:
    std::vector<int> script_;
    int eos_;
    bool loop_;
    int delay_us_;
    struct Seq { int k = 0, prompt_end = 0; };
    std::vector<Seq> st_;
    Seq saved_;
    int cur_ = 0;
    std::vector<float> empty_;
    int scripted() const {
        const int k_ = st_[cur_].k;
        if (k_ < (int)script_.size()) return script_[k_];
        if (loop_) return script_[k_ % script_.size()];
        return eos_;
    }
    void delay(int units) const { if (delay_us_ > 0) std::this_thread::sleep_for(std::chrono::microseconds((long)delay_us_ * units)); }
};

}  // namespace qv

#include "vision/vision.h"
namespace qv {
// Test encoder (`--vision-file mock`): deterministic rows, hidden size 8, no weights.
class MockVisionEncoder : public vision::Encoder {
   public:
    MockVisionEncoder() { cfg_.text_hidden = 8; }
    const char* backend() const override { return "mock"; }
    void encode(const vision::Pixels& p, float* out) override {
        for (int i = 0; i < p.tokens() * 8; ++i) out[i] = 0.01f * (float)(i % 7);
        last_ms = 0.1;
    }
};
}  // namespace qv
