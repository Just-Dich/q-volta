#include "tokenizer.h"

#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../core/json.h"

namespace qv {
namespace {

void append_utf8(std::string& s, UChar32 c) {
    char buf[4];
    int32_t i = 0;
    UBool err = false;
    U8_APPEND(buf, i, 4, c, err);
    s.append(buf, i);
}

// Unicode classes used by the Qwen split regex.
bool is_L(UChar32 c) {
    int t = u_charType(c);
    return t == U_UPPERCASE_LETTER || t == U_LOWERCASE_LETTER || t == U_TITLECASE_LETTER || t == U_MODIFIER_LETTER ||
           t == U_OTHER_LETTER;
}
bool is_M(UChar32 c) {
    int t = u_charType(c);
    return t == U_NON_SPACING_MARK || t == U_COMBINING_SPACING_MARK || t == U_ENCLOSING_MARK;
}
bool is_N(UChar32 c) {
    int t = u_charType(c);
    return t == U_DECIMAL_DIGIT_NUMBER || t == U_LETTER_NUMBER || t == U_OTHER_NUMBER;
}
bool is_S(UChar32 c) { return u_isUWhiteSpace(c); }  // regex \s (Unicode White_Space)
bool is_nl(UChar32 c) { return c == '\r' || c == '\n'; }

// Pre-tokenizer: hand-compiled version of
//  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in order at each position (leftmost-first semantics); returns [begin,end) codepoint spans.
std::vector<std::pair<int, int>> split(const std::vector<UChar32>& cp) {
    std::vector<std::pair<int, int>> out;
    const int n = (int)cp.size();
    auto lower = [](UChar32 c) { return (UChar32)u_tolower(c); };
    int i = 0;
    while (i < n) {
        int end = -1;
        // 1. contractions
        if (cp[i] == '\'' && i + 1 < n) {
            UChar32 a = lower(cp[i + 1]);
            UChar32 b = i + 2 < n ? lower(cp[i + 2]) : 0;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3;
            else if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2;
        }
        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (end < 0) {
            int j = i;
            if (!is_nl(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])) {
                // optional prefix char: only taken if letters/marks follow (else backtrack to no prefix)
                if (j + 1 < n && (is_L(cp[j + 1]) || is_M(cp[j + 1]))) ++j;
            }
            int k = j;
            while (k < n && (is_L(cp[k]) || is_M(cp[k]))) ++k;
            if (k > j) end = k;
        }
        // 3. \p{N}
        if (end < 0 && is_N(cp[i])) end = i + 1;
        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (end < 0) {
            auto other = [&](UChar32 c) { return !is_S(c) && !is_L(c) && !is_M(c) && !is_N(c); };
            int j = i;
            if (cp[j] == ' ' && j + 1 < n && other(cp[j + 1])) ++j;
            int k = j;
            while (k < n && other(cp[k])) ++k;
            if (k > j) {
                while (k < n && is_nl(cp[k])) ++k;
                end = k;
            }
        }
        if (end < 0 && is_S(cp[i])) {
            int r = i;
            while (r < n && is_S(cp[r])) ++r;  // maximal whitespace run [i, r)
            // 5. \s*[\r\n]+ : up to the last newline inside the run
            int last_nl = -1;
            for (int k = i; k < r; ++k)
                if (is_nl(cp[k])) last_nl = k;
            if (last_nl >= 0) {
                end = last_nl + 1;
            } else if (r == n) {
                end = r;  // 6. \s+(?!\S) at end of text
            } else if (r - i >= 2) {
                end = r - 1;  // 6. leave the last space to attach to the next token
            } else {
                end = r;  // 7. \s+
            }
        }
        if (end < 0) end = i + 1;  // unreachable for valid input; be safe
        out.emplace_back(i, end);
        i = end;
    }
    return out;
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value root = json::parse(ss.str());
    Tokenizer t;
    const auto& model = root["model"];
    if (model["type"].str() != "BPE") throw std::runtime_error("tokenizer: only BPE supported");
    int max_id = 0;
    for (const auto& [tok, id] : model["vocab"].o) max_id = std::max(max_id, (int)id.i64());
    for (const auto& a : root["added_tokens"].a) max_id = std::max(max_id, (int)a["id"].i64());
    t.id_to_tok_.assign(max_id + 1, "");
    for (const auto& [tok, id] : model["vocab"].o) {
        t.vocab_[tok] = (int)id.i64();
        t.id_to_tok_[(int)id.i64()] = tok;
    }
    int rank = 0;
    for (const auto& m : model["merges"].a) {
        std::string key = m.t == json::Value::T::Str ? m.str() : m[0].str() + " " + m[1].str();
        t.merge_rank_.emplace(std::move(key), rank++);
    }
    for (const auto& a : root["added_tokens"].a) {
        int id = (int)a["id"].i64();
        t.specials_.emplace_back(a["content"].str(), id);
        t.id_to_tok_[id] = a["content"].str();
    }
    std::sort(t.specials_.begin(), t.specials_.end(),
              [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
    // GPT-2 bytes_to_unicode
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int extra = 0;
    for (int b = 0; b < 256; ++b)
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + extra++);
        }
    for (size_t k = 0; k < bs.size(); ++k) {
        std::string s;
        append_utf8(s, cs[k]);
        t.byte_enc_[bs[k]] = s;
        t.byte_dec_[s] = (unsigned char)bs[k];
    }
    return t;
}

int Tokenizer::token_id(const std::string& special) const {
    for (const auto& [s, id] : specials_)
        if (s == special) return id;
    auto it = vocab_.find(special);
    return it == vocab_.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string& word, std::vector<int>& out) const {
    auto hit = cache_.find(word);
    if (hit != cache_.end()) {
        out.insert(out.end(), hit->second.begin(), hit->second.end());
        return;
    }
    // symbols = byte-level encoded characters of the word
    std::vector<std::string> sym;
    for (unsigned char b : word) sym.push_back(byte_enc_[b]);
    while (sym.size() > 1) {
        int best = INT_MAX, bi = -1;
        for (size_t k = 0; k + 1 < sym.size(); ++k) {
            auto it = merge_rank_.find(sym[k] + " " + sym[k + 1]);
            if (it != merge_rank_.end() && it->second < best) { best = it->second; bi = (int)k; }
        }
        if (bi < 0) break;
        // merge every occurrence of this pair, left to right (same as HF BPE)
        const std::string a = sym[bi], b = sym[bi + 1];
        std::vector<std::string> nxt;
        for (size_t k = 0; k < sym.size();) {
            if (k + 1 < sym.size() && sym[k] == a && sym[k + 1] == b) { nxt.push_back(a + b); k += 2; }
            else nxt.push_back(sym[k++]);
        }
        sym.swap(nxt);
    }
    std::vector<int> ids;
    for (const auto& s : sym) {
        auto it = vocab_.find(s);
        if (it == vocab_.end()) throw std::runtime_error("tokenizer: symbol not in vocab");
        ids.push_back(it->second);
    }
    if (word.size() < 64) cache_.emplace(word, ids);
    out.insert(out.end(), ids.begin(), ids.end());
}

void Tokenizer::encode_ordinary(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    UErrorCode st = U_ZERO_ERROR;
    const icu::Normalizer2* nfc = icu::Normalizer2::getNFCInstance(st);
    icu::UnicodeString u = icu::UnicodeString::fromUTF8(text);
    icu::UnicodeString norm = nfc->normalize(u, st);
    if (U_FAILURE(st)) throw std::runtime_error("tokenizer: NFC failed");
    std::vector<UChar32> cp;
    for (int32_t k = 0; k < norm.length();) {
        UChar32 c = norm.char32At(k);
        cp.push_back(c);
        k += U16_LENGTH(c);
    }
    for (auto [b, e] : split(cp)) {
        std::string w;
        for (int k = b; k < e; ++k) append_utf8(w, cp[k]);
        bpe(w, out);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> out;
    if (!parse_special) {
        encode_ordinary(text, out);
        return out;
    }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<') {
            for (const auto& [s, id] : specials_) {
                if (text.compare(i, s.size(), s) == 0) {
                    encode_ordinary(text.substr(start, i - start), out);
                    out.push_back(id);
                    i += s.size();
                    start = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) ++i;
    }
    encode_ordinary(text.substr(start), out);
    return out;
}

std::string Tokenizer::token_text(int id) const {
    if (id < 0 || id >= (int)id_to_tok_.size()) return "";
    const std::string& t = id_to_tok_[id];
    for (const auto& [s, sid] : specials_)
        if (sid == id) return s;
    std::string out;
    icu::UnicodeString u = icu::UnicodeString::fromUTF8(t);
    for (int32_t k = 0; k < u.length();) {
        UChar32 c = u.char32At(k);
        k += U16_LENGTH(c);
        std::string ch;
        append_utf8(ch, c);
        auto it = byte_dec_.find(ch);
        if (it != byte_dec_.end()) out.push_back((char)it->second);
        else out += ch;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += token_text(id);
    return out;
}

}  // namespace qv
