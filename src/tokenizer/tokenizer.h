// Byte-level BPE tokenizer for the Qwen2/3/3.5 family, built from the HF tokenizer.json:
//   NFC normaliser -> added-token split -> hand-written scanner for the Qwen split pattern -> GPT-2 byte map -> BPE merges.
// All methods are const and thread-safe after load().
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qv {

class Tokenizer {
   public:
    static Tokenizer load(const std::string& tokenizer_json_path);
    // Same, from the bytes of a tokenizer.json already in memory (e.g. the blob embedded in a .qv container).
    static Tokenizer from_memory(std::string_view tokenizer_json);

    // parse_special: recognise added tokens (<|im_start|>, <think>, ...) literally in the text.
    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    // Full decode (invalid UTF-8 is replaced by U+FFFD at the end). Special tokens are printed literally.
    std::string decode(const std::vector<int>& ids) const;
    // Raw bytes of one token (may be a partial UTF-8 sequence); added tokens give their literal text.
    std::string token_bytes(int id) const;
    int token_id(const std::string& token_text) const;  // exact vocab / added-token lookup, -1 if absent
    int vocab_size() const { return (int)id_bytes_.size(); }  // max id + 1 (incl. added tokens)
    bool is_added(int id) const { return id >= 0 && id < (int)is_added_.size() && is_added_[id]; }

    // Incremental detokenizer for streaming: bytes of tokens are buffered until they form complete UTF-8
    // sequences; invalid bytes become U+FFFD (maximal-subpart rule, like Python's errors="replace").
    class Decoder {
       public:
        explicit Decoder(const Tokenizer& t) : tok_(&t) {}
        std::string push(int id);   // text that became complete with this token (possibly empty)
        std::string flush();        // emit whatever is buffered (incomplete tail -> U+FFFD)
       private:
        const Tokenizer* tok_;
        std::string buf_;
        std::string drain(bool final_);
    };

    // The split scanner, exposed for tests: code points -> [begin,end) piece ranges.
    static void split_pieces(const std::vector<uint32_t>& cps, std::vector<std::pair<size_t, size_t>>& out);
    static std::string nfc(const std::string& utf8);

   private:
    std::unordered_map<std::string, int> vocab_;  // byte-level-unicode spelling -> id (incl. added)
    std::vector<std::string> id_bytes_;           // id -> raw bytes
    std::vector<char> is_added_;
    std::vector<std::pair<std::string, int>> added_;  // longest first
    struct Merge { int rank; int result; };
    std::unordered_map<uint64_t, Merge> merges_;  // (idA<<32|idB) -> merge
    int byte_id_[256];
    std::string byte_str_[256];  // GPT-2 unicode spelling of each byte

    void bpe(const uint8_t* bytes, size_t n, std::vector<int>& out) const;
    void encode_ordinary(const std::string& text, std::vector<int>& out) const;
};

}  // namespace qv
