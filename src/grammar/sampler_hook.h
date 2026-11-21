// Adapters: GrammarState -> sampling hook, and vocabulary construction from qv::Tokenizer.
#pragma once
#include <memory>
#include <vector>

#include "grammar/grammar.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"

namespace qv::grammar {

class GrammarLogitMask final : public LogitMask {
   public:
    explicit GrammarLogitMask(GrammarState st) : st_(std::move(st)) {}
    void apply(float* logits, size_t n) override { st_.mask_logits(logits, n); }
    void accept(int tok) override { st_.accept_token(tok); }
    GrammarState& state() { return st_; }

   private:
    GrammarState st_;
};

// Pieces are the raw token bytes (special tokens: literal text). `eog_ids` are end-of-generation tokens (eos, <|im_end|>, ...).
inline std::shared_ptr<const VocabIndex> vocab_from_tokenizer(const Tokenizer& tok, const std::vector<int>& eog_ids) {
    std::vector<std::string> pieces((size_t)tok.vocab_size());
    std::vector<uint8_t> eog(pieces.size(), 0);
    for (int i = 0; i < tok.vocab_size(); ++i) pieces[(size_t)i] = tok.token_bytes(i);
    for (int e : eog_ids) if (e >= 0 && (size_t)e < eog.size()) eog[(size_t)e] = 1;
    return VocabIndex::build(pieces, eog);
}

}  // namespace qv::grammar
