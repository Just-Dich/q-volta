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
    MockEngine(int vocab, std::vector<int> script, int eos, bool loop, int delay_us) : script_(std::move(script)), eos_(eos), loop_(loop), delay_us_(delay_us) {
        cfg_.vocab = vocab;
        cfg_.eos = eos;
        cfg_.arch = "mock";
        if (script_.empty()) script_.push_back(eos);
    }
    void step(int, int pos) override { delay(1); k_ = pos + 1 - prompt_end_; }
    void prefill(const int*, int T, int pos0) override { delay(T / 8 + 1); prompt_end_ = pos0 + T; k_ = 0; }
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
    void reset() override { k_ = 0; prompt_end_ = 0; }
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
    int k_ = 0, prompt_end_ = 0;
    std::vector<float> empty_;
    int scripted() const {
        if (k_ < (int)script_.size()) return script_[k_];
        if (loop_) return script_[k_ % script_.size()];
        return eos_;
    }
    void delay(int units) const { if (delay_us_ > 0) std::this_thread::sleep_for(std::chrono::microseconds((long)delay_us_ * units)); }
};

}  // namespace qv
