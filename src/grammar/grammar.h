// GBNF grammar parser + incremental grammar-constrained decoding (llama.cpp llama-grammar.cpp semantics).
//
//   auto g = qv::grammar::Grammar::parse(R"(root ::= "yes" | "no")");           // throws GrammarError
//   auto vocab = VocabIndex::build(pieces, eog_flags);                          // once per tokenizer (trie over token bytes)
//   auto m = std::make_shared<Matcher>(g, vocab);                               // shares per-state mask cache across requests
//   GrammarState st(m);
//   st.mask_logits(logits, n_vocab);  int tok = sample(logits);  st.accept_token(tok);
//
// Semantics follow llama.cpp: a token is allowed iff every code point of its piece advances some grammar stack (a trailing
// partial UTF-8 sequence is allowed if some stack could complete it); EOG tokens are allowed only when the grammar can end;
// tokens with empty pieces are never allowed. Not supported: `<[id]>` token elements.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qv::grammar {

struct GrammarError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Impl;  // compiled grammar + shared state machine cache

class Grammar {
   public:
    Grammar() = default;
    // Parse GBNF. `root` is the start rule. Throws GrammarError (syntax error, undefined rule, left recursion, missing root).
    static Grammar parse(std::string_view gbnf, std::string_view root = "root");
    bool valid() const { return impl_ != nullptr; }
    std::string dump() const;  // normalised GBNF (rule names preserved, generated sub-rules named name_N)
    size_t n_rules() const;
    const std::shared_ptr<Impl>& impl() const { return impl_; }

   private:
    std::shared_ptr<Impl> impl_;
};

struct TokenMask {
    std::vector<uint64_t> bits;
    uint32_t n_allowed = 0;
    size_t n = 0;
    bool allowed(size_t id) const { return id < n && ((bits[id >> 6] >> (id & 63)) & 1); }
};

// Immutable index of a vocabulary: byte trie in pre-order for fast mask computation.
class VocabIndex {
   public:
    // pieces[i] = raw bytes of token i (special tokens: their literal text; "" = never allowed). eog[i] != 0 = end-of-generation token.
    static std::shared_ptr<const VocabIndex> build(const std::vector<std::string>& pieces, const std::vector<uint8_t>& eog);
    size_t size() const { return pieces_.size(); }
    const std::string& piece(size_t id) const { return pieces_[id]; }
    bool is_eog(size_t id) const { return eog_[id] != 0; }

    struct Node {  // pre-order trie node
        uint32_t end;       // index one past the last node of this subtree
        uint32_t tok_begin; // range in sorted_ids of tokens whose piece ends exactly here
        uint32_t tok_cnt;
        uint8_t byte;
    };
    std::vector<Node> nodes;
    std::vector<uint32_t> sorted_ids;
    uint32_t max_depth = 0;
    std::vector<uint32_t> eog_ids;

   private:
    std::vector<std::string> pieces_;
    std::vector<uint8_t> eog_;
};

struct MatcherOptions {
    size_t max_cache_bytes = 256u << 20;  // bound on cached per-state token masks (LRU)
};

// Binds a grammar to a vocabulary; owns the cache of allowed-token masks keyed by grammar state. Thread-safe.
class Matcher {
   public:
    Matcher(Grammar g, std::shared_ptr<const VocabIndex> vocab = nullptr, MatcherOptions opt = {});
    const Grammar& grammar() const { return grammar_; }
    const std::shared_ptr<const VocabIndex>& vocab() const { return vocab_; }
    struct Stats { uint64_t mask_hits = 0, mask_misses = 0, mask_compute_us = 0; size_t states = 0, cached_masks = 0, cache_bytes = 0; };
    Stats stats() const;

   private:
    friend class GrammarState;
    struct Cache;
    Grammar grammar_;
    std::shared_ptr<const VocabIndex> vocab_;
    MatcherOptions opt_;
    std::shared_ptr<Cache> cache_;
};

// Lazy ("triggered") grammars: constraints start only when the output contains a trigger (llama.cpp grammar_lazy for tool_choice=auto).
struct Triggers {
    std::vector<std::string> words;  // substring in the generated text; the grammar consumes from the start of the match
    std::vector<int> tokens;         // token ids that start the constrained part (the token itself is fed to the grammar)
    bool empty() const { return words.empty() && tokens.empty(); }
};

class GrammarState {
   public:
    explicit GrammarState(std::shared_ptr<Matcher> m, Triggers lazy = {});
    explicit GrammarState(const Grammar& g);  // no vocabulary: accept()/is_complete() only

    // Advance by raw bytes (a token piece, partial UTF-8 allowed) or Unicode code points. Returns false (and leaves the state
    // dead) when the grammar rejects the input. While waiting for a lazy trigger the text is buffered and always accepted.
    bool accept(std::string_view piece);
    bool accept(const std::vector<uint32_t>& code_points);
    bool accept_token(int id);  // uses the vocabulary piece; EOG ids require is_complete()

    bool is_complete() const;   // the grammar can end here (EOG would be allowed)
    bool is_dead() const;       // no continuation exists
    bool awaiting_trigger() const { return awaiting_; }

    // Allowed-token mask for the current state (cached across states/requests). Requires a vocabulary.
    std::shared_ptr<const TokenMask> allowed();
    // logits[i] = -inf for every disallowed token (no-op while waiting for a lazy trigger). n_vocab may exceed the vocab size
    // (extra ids are masked out).
    void mask_logits(float* logits, size_t n_vocab);
    bool token_allowed(int id);

    GrammarState fork() const { return *this; }  // cheap copy (state id + partial UTF-8)

   private:
    std::shared_ptr<Matcher> m_;
    std::shared_ptr<Impl> impl_;
    int32_t sid_ = 0;
    uint32_t pv_ = 0;
    int8_t rem_ = 0;
    bool dead_ = false;
    bool awaiting_ = false;
    Triggers trig_;
    std::string trig_buf_;
    std::vector<std::pair<int, std::pair<size_t, size_t>>> trig_tok_pos_;
    bool feed(std::string_view s);
};

}  // namespace qv::grammar
