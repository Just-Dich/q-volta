// Byte-level BPE tokenizer for Qwen3.x tokenizer.json (NFC normaliser + Qwen split regex + GPT-2 byte map).
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace qv {

class Tokenizer {
   public:
    static Tokenizer load(const std::string& tokenizer_json_path);
    // parse_special: recognise added/special tokens (e.g. <|im_start|>) literally in the text.
    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string token_text(int id) const;  // decoded bytes of one token (may be partial UTF-8)
    int token_id(const std::string& special) const;  // -1 if absent
    int vocab_size() const { return (int)id_to_tok_.size(); }

   private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_tok_;
    std::unordered_map<std::string, int> merge_rank_;  // "a b" -> rank
    std::vector<std::pair<std::string, int>> specials_;  // sorted longest-first
    std::string byte_enc_[256];
    std::unordered_map<std::string, unsigned char> byte_dec_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    void bpe(const std::string& word_bytes, std::vector<int>& out) const;
    void encode_ordinary(const std::string& text, std::vector<int>& out) const;
};

}  // namespace qv
