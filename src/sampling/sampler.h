// Token sampling on host logits: OpenAI-style penalties, temperature, top-k, top-p, min-p, deterministic seeded RNG.
#pragma once
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace qv {

struct SamplingParams {
    float temperature = 1.0f;       // <= 0 means greedy
    int top_k = 0;                  // 0 = disabled
    float top_p = 1.0f;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;   // OpenAI: logit -= presence * (count > 0)
    float frequency_penalty = 0.0f;  // OpenAI: logit -= frequency * count   (counts over generated tokens)
    float repetition_penalty = 1.0f; // HF/vLLM extension: over prompt + generated tokens
    bool has_seed = false;
    uint64_t seed = 0;

    bool greedy() const { return temperature <= 0.0f; }
    bool has_penalties() const { return presence_penalty != 0 || frequency_penalty != 0 || repetition_penalty != 1.0f; }
    // True when the engine's own argmax() gives the answer (no host logits needed).
    bool engine_argmax_ok() const { return greedy() && !has_penalties(); }
};

struct TokenLogprob {
    int token = -1;
    float logprob = 0;                                  // log-softmax of the raw (unpenalised, temperature 1) logits
    std::vector<std::pair<int, float>> top;             // best alternatives, descending
};

// Hook that constrains sampling (grammar / JSON-schema decoding, see src/grammar/sampler_hook.h).
struct LogitMask {
    virtual ~LogitMask() = default;
    virtual void apply(float* logits, size_t n) = 0;  // set disallowed logits to -inf
    virtual void accept(int tok) = 0;                 // the chosen token was committed
};

class Sampler {
   public:
    // The mask runs after penalties and before temperature/top-k/top-p; greedy decoding on host logits honours it too.
    void set_mask(std::shared_ptr<LogitMask> m) { mask_ = std::move(m); }
    bool has_mask() const { return mask_ != nullptr; }
    Sampler(const SamplingParams& p, const std::vector<int>& prompt_tokens);
    // logits are modified in place (penalties/temperature applied). Does not record the token; call accept().
    int sample(std::vector<float>& logits);
    void accept(int tok);
    // log-probability info from the *raw* logits (call before sample()).
    static TokenLogprob logprob(const std::vector<float>& raw, int chosen, int top_n);

   private:
    SamplingParams p_;
    std::shared_ptr<LogitMask> mask_;
    std::unordered_map<int, int> gen_counts_;
    std::unordered_map<int, char> seen_;  // prompt + generated (repetition penalty)
    uint64_t s_[4];
    std::vector<int> idx_;
    std::vector<float> prob_;
    uint64_t next();
    double uniform();
};

}  // namespace qv
