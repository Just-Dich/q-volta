#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "core/json_doc.h"

namespace qv {

#include "tokenizer/unicode_tables.inc"

namespace {

// ------------------------------------------------------------------------------------------------ UTF-8 helpers
void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
}

// Length of the well-formed UTF-8 sequence starting at s[i] (1..4), or 0 if malformed. `avail_prefix` is set to the
// number of leading bytes that form a valid prefix of some sequence (for incomplete-tail detection).
int utf8_seq(const uint8_t* s, size_t n, size_t i, bool& incomplete) {
    incomplete = false;
    uint8_t c = s[i];
    if (c < 0x80) return 1;
    int need;
    uint8_t lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) need = 2;
    else if (c == 0xE0) { need = 3; lo = 0xA0; }
    else if (c == 0xED) { need = 3; hi = 0x9F; }
    else if (c >= 0xE1 && c <= 0xEF) need = 3;
    else if (c == 0xF0) { need = 4; lo = 0x90; }
    else if (c == 0xF4) { need = 4; hi = 0x8F; }
    else if (c >= 0xF1 && c <= 0xF3) need = 4;
    else return 0;
    for (int k = 1; k < need; ++k) {
        if (i + k >= n) { incomplete = true; return 0; }
        uint8_t b = s[i + k];
        uint8_t l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xBF;
        if (b < l || b > h) return 0;
    }
    return need;
}

std::vector<uint32_t> to_codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    const uint8_t* p = (const uint8_t*)s.data();
    size_t n = s.size();
    for (size_t i = 0; i < n;) {
        bool inc;
        int len = utf8_seq(p, n, i, inc);
        if (len == 0) { cps.push_back(0xFFFD); ++i; continue; }
        uint32_t cp = len == 1 ? p[i] : p[i] & (0xFF >> (len + 1));
        for (int k = 1; k < len; ++k) cp = (cp << 6) | (p[i + k] & 63);
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

// ------------------------------------------------------------------------------------------------ unicode classes
template <size_t N>
bool in_ranges(const uint32_t (&t)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
// kCatL etc. are declared with unknown extent in the .inc, so go through pointers + counts.
bool in_ranges_p(const uint32_t (*t)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
(void)sizeof(in_ranges<1>);
bool is_L(uint32_t c) {
    if (c < 0x80) return (c | 32) - 'a' < 26u;
    return in_ranges_p(kCatL, kCatL_n, c);
}
bool is_M(uint32_t c) { return c >= 0x300 && in_ranges_p(kCatM, kCatM_n, c); }
bool is_N(uint32_t c) {
    if (c < 0x80) return c - '0' < 10u;
    return in_ranges_p(kCatN, kCatN_n, c);
}
// Unicode White_Space property (what Rust's regex \s matches).
bool is_space(uint32_t c) {
    if (c < 0x80) return c == ' ' || (c >= 9 && c <= 13);
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

// ------------------------------------------------------------------------------------------------ NFC
int ccc_of(uint32_t c) {
    if (c < 0x300) return 0;
    size_t lo = 0, hi = kCcc_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < kCcc[mid][0]) hi = mid;
        else if (c > kCcc[mid][1]) lo = mid + 1;
        else return (int)kCcc[mid][2];
    }
    return 0;
}
constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7, LCount = 19, VCount = 21, TCount = 28,
                   NCount = VCount * TCount, SCount = LCount * NCount;

void decompose(uint32_t c, std::vector<uint32_t>& out) {
    if (c < 0xC0) { out.push_back(c); return; }
    if (c >= SBase && c < SBase + SCount) {
        uint32_t s = c - SBase;
        out.push_back(LBase + s / NCount);
        out.push_back(VBase + (s % NCount) / TCount);
        if (s % TCount) out.push_back(TBase + s % TCount);
        return;
    }
    size_t lo = 0, hi = kDec_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < kDec[mid][0]) hi = mid;
        else if (c > kDec[mid][0]) lo = mid + 1;
        else { for (uint32_t k = 0; k < kDec[mid][2]; ++k) out.push_back(kDecPool[kDec[mid][1] + k]); return; }
    }
    out.push_back(c);
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount) return a + (b - TBase);
    size_t lo = 0, hi = kComp_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (a < kComp[mid][0] || (a == kComp[mid][0] && b < kComp[mid][1])) hi = mid;
        else if (a > kComp[mid][0] || b > kComp[mid][1]) lo = mid + 1;
        else return kComp[mid][2];
    }
    return 0;
}

}  // namespace

std::string Tokenizer::nfc(const std::string& s) {
    bool ascii = true;
    for (unsigned char c : s) if (c >= 0x80) { ascii = false; break; }
    if (ascii) return s;
    std::vector<uint32_t> cps = to_codepoints(s), d;
    d.reserve(cps.size() + 8);
    for (uint32_t c : cps) decompose(c, d);
    // canonical ordering: stable sort each run of non-starters by combining class
    for (size_t i = 0; i < d.size();) {
        if (ccc_of(d[i]) == 0) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && ccc_of(d[j]) != 0) ++j;
        std::stable_sort(d.begin() + i, d.begin() + j, [](uint32_t x, uint32_t y) { return ccc_of(x) < ccc_of(y); });
        i = j;
    }
    // composition
    std::vector<uint32_t> out;
    out.reserve(d.size());
    long starter = -1;
    int lastcc = 0;
    for (uint32_t c : d) {
        int cc = ccc_of(c);
        if (starter >= 0 && (lastcc < cc || lastcc == 0)) {
            uint32_t comp = compose_pair(out[starter], c);
            if (comp) { out[starter] = comp; continue; }
        }
        if (cc == 0) starter = (long)out.size();
        out.push_back(c);
        lastcc = cc;
    }
    std::string r;
    r.reserve(s.size());
    for (uint32_t c : out) put_utf8(r, c);
    return r;
}

// ------------------------------------------------------------------------------------------------ split scanner
// Hand-written equivalent of the pre-tokenizer pattern
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
// with leftmost-first alternation and greedy backtracking semantics. (Case-insensitive contractions are matched for
// ASCII letters only; the Unicode case-fold variants U+017F / U+212A are not.)
void Tokenizer::split_pieces(const std::vector<uint32_t>& cp, std::vector<std::pair<size_t, size_t>>& out) {
    const size_t n = cp.size();
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    auto is_nl = [](uint32_t c) { return c == '\r' || c == '\n'; };
    size_t i = 0;
    while (i < n) {
        const uint32_t c = cp[i];
        // 1. contractions
        if (c == '\'' && i + 1 < n) {
            uint32_t a = lower(cp[i + 1]);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') { out.emplace_back(i, i + 2); i += 2; continue; }
            if (i + 2 < n) {
                uint32_t b = lower(cp[i + 2]);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { out.emplace_back(i, i + 3); i += 3; continue; }
            }
        }
        // 2. optional single non-letter/number/newline prefix + letters/marks
        {
            size_t j = i;
            if (!is_nl(c) && !is_L(c) && !is_N(c)) j = i + 1;
            if (j < n && (is_L(cp[j]) || is_M(cp[j]))) {
                size_t k = j;
                while (k < n && (is_L(cp[k]) || is_M(cp[k]))) ++k;
                out.emplace_back(i, k); i = k; continue;
            }
            if (j != i && (is_L(c) || is_M(c))) {}  // unreachable: prefix class excludes L; M handled by j==i path below
            if (j == i + 1 && is_M(c) && false) {}
            if (is_M(c)) {  // a lone mark with no following letters: matches via [\p{L}\p{M}]+ without a prefix
                size_t k = i;
                while (k < n && (is_L(cp[k]) || is_M(cp[k]))) ++k;
                out.emplace_back(i, k); i = k; continue;
            }
        }
        // 3. single number
        if (is_N(c)) { out.emplace_back(i, i + 1); ++i; continue; }
        // 4. optional space + punctuation/symbols + trailing newlines
        {
            size_t j = (c == ' ') ? i + 1 : i;
            size_t k = j;
            while (k < n && !is_space(cp[k]) && !is_L(cp[k]) && !is_M(cp[k]) && !is_N(cp[k])) ++k;
            if (k > j) {
                while (k < n && is_nl(cp[k])) ++k;
                out.emplace_back(i, k); i = k; continue;
            }
        }
        // 5-7. whitespace runs (c is whitespace here)
        {
            size_t r = i;
            while (r < n && is_space(cp[r])) ++r;
            size_t last_nl = (size_t)-1;
            for (size_t k = i; k < r; ++k) if (is_nl(cp[k])) last_nl = k;
            if (last_nl != (size_t)-1) { out.emplace_back(i, last_nl + 1); i = last_nl + 1; continue; }
            if (r == n) { out.emplace_back(i, r); i = r; continue; }
            if (r - i >= 2) { out.emplace_back(i, r - 1); i = r - 1; continue; }
            out.emplace_back(i, r); i = r; continue;
        }
    }
}

// ------------------------------------------------------------------------------------------------ BPE
void Tokenizer::bpe(const uint8_t* bytes, size_t n, std::vector<int>& out) const {
    if (n == 0) return;
    if (n == 1) { out.push_back(byte_id_[bytes[0]]); return; }
    thread_local std::vector<int> tok, prv, nxt;
    tok.assign(n, 0);
    prv.resize(n);
    nxt.resize(n);
    for (size_t i = 0; i < n; ++i) { tok[i] = byte_id_[bytes[i]]; prv[i] = (int)i - 1; nxt[i] = i + 1 < n ? (int)i + 1 : -1; }
    struct Ent { int rank, left, right, tl, tr; };
    auto cmp = [](const Ent& a, const Ent& b) { return a.rank != b.rank ? a.rank > b.rank : a.left > b.left; };
    std::priority_queue<Ent, std::vector<Ent>, decltype(cmp)> pq(cmp);
    auto push_pair = [&](int l) {
        int r = nxt[l];
        if (r < 0) return;
        auto it = merges_.find(((uint64_t)(uint32_t)tok[l] << 32) | (uint32_t)tok[r]);
        if (it != merges_.end()) pq.push({it->second.rank, l, r, tok[l], tok[r]});
    };
    for (size_t i = 0; i + 1 < n; ++i) push_pair((int)i);
    while (!pq.empty()) {
        Ent e = pq.top();
        pq.pop();
        if (tok[e.left] < 0 || tok[e.right] < 0 || nxt[e.left] != e.right || tok[e.left] != e.tl || tok[e.right] != e.tr) continue;
        auto it = merges_.find(((uint64_t)(uint32_t)e.tl << 32) | (uint32_t)e.tr);
        tok[e.left] = it->second.result;
        tok[e.right] = -1;  // dead
        nxt[e.left] = nxt[e.right];
        if (nxt[e.right] >= 0) prv[nxt[e.right]] = e.left;
        if (prv[e.left] >= 0) push_pair(prv[e.left]);
        push_pair(e.left);
    }
    for (int i = 0; i >= 0; i = nxt[i]) out.push_back(tok[i]);
}

void Tokenizer::encode_ordinary(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    std::string norm = nfc(text);
    std::vector<uint32_t> cps = to_codepoints(norm);
    std::vector<std::pair<size_t, size_t>> pieces;
    split_pieces(cps, pieces);
    std::string bytes;
    for (auto [b, e] : pieces) {
        bytes.clear();
        for (size_t k = b; k < e; ++k) put_utf8(bytes, cps[k]);
        bpe((const uint8_t*)bytes.data(), bytes.size(), out);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> out;
    out.reserve(text.size() / 3 + 8);
    if (!parse_special || added_.empty()) { encode_ordinary(text, out); return out; }
    size_t seg = 0, i = 0;
    while (i < text.size()) {
        if (text[i] == '<') {
            bool hit = false;
            for (auto& [s, id] : added_) {
                if (text.compare(i, s.size(), s) == 0) {
                    encode_ordinary(text.substr(seg, i - seg), out);
                    out.push_back(id);
                    i += s.size();
                    seg = i;
                    hit = true;
                    break;
                }
            }
            if (hit) continue;
        }
        ++i;
    }
    encode_ordinary(text.substr(seg), out);
    return out;
}

// ------------------------------------------------------------------------------------------------ decode
std::string Tokenizer::token_bytes(int id) const {
    if (id < 0 || id >= (int)id_bytes_.size()) return std::string();
    return id_bytes_[id];
}
int Tokenizer::token_id(const std::string& t) const {
    auto it = vocab_.find(t);
    if (it != vocab_.end()) return it->second;
    for (auto& [s, id] : added_) if (s == t) return id;
    return -1;
}

std::string Tokenizer::Decoder::drain(bool final_) {
    std::string out;
    const uint8_t* p = (const uint8_t*)buf_.data();
    size_t n = buf_.size(), i = 0;
    while (i < n) {
        bool inc;
        int len = utf8_seq(p, n, i, inc);
        if (len) { out.append(buf_, i, len); i += len; continue; }
        if (inc && !final_) break;  // possibly-valid incomplete tail: wait for more bytes
        // maximal subpart of an ill-formed sequence -> one U+FFFD
        size_t k = 1;
        if (p[i] >= 0xC2) {
            // consume continuation bytes that are still a valid prefix
            uint8_t c = p[i];
            uint8_t lo = 0x80, hi = 0xBF;
            if (c == 0xE0) lo = 0xA0; else if (c == 0xED) hi = 0x9F; else if (c == 0xF0) lo = 0x90; else if (c == 0xF4) hi = 0x8F;
            int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
            while ((int)k < need && i + k < n && p[i + k] >= (k == 1 ? lo : 0x80) && p[i + k] <= (k == 1 ? hi : 0xBF)) ++k;
        }
        out += "\xEF\xBF\xBD";
        i += k;
    }
    buf_.erase(0, i);
    return out;
}
std::string Tokenizer::Decoder::push(int id) {
    buf_ += tok_->token_bytes(id);
    return drain(false);
}
std::string Tokenizer::Decoder::flush() { return drain(true); }

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    Decoder d(*this);
    std::string out;
    for (int id : ids) out += d.push(id);
    out += d.flush();
    return out;
}

// ------------------------------------------------------------------------------------------------ load
Tokenizer Tokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    jd::Value doc = jd::parse(ss.str());
    const jd::Value* model = doc.find("model");
    if (!model) throw std::runtime_error("tokenizer: no model section");
    const jd::Value* mtype = model->find("type");
    if (!mtype || mtype->as_string() != "BPE") throw std::runtime_error("tokenizer: only BPE models are supported");
    const jd::Value* vocab = model->find("vocab");
    const jd::Value* merges = model->find("merges");
    if (!vocab || !merges) throw std::runtime_error("tokenizer: vocab/merges missing");

    Tokenizer t;
    // GPT-2 byte <-> unicode map
    std::unordered_map<uint32_t, uint8_t> inv;
    {
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            uint32_t cp = printable ? (uint32_t)b : 256u + n++;
            t.byte_str_[b].clear();
            put_utf8(t.byte_str_[b], cp);
            inv[cp] = (uint8_t)b;
        }
    }
    int max_id = -1;
    for (auto& m : vocab->members()) max_id = std::max(max_id, (int)m.second.as_int());
    const jd::Value* added = doc.find("added_tokens");
    if (added) for (auto& a : added->items()) max_id = std::max(max_id, (int)a.find("id")->as_int());
    t.id_bytes_.assign(max_id + 1, std::string());
    t.is_added_.assign(max_id + 1, 0);
    t.vocab_.reserve(vocab->size() * 2);
    for (auto& m : vocab->members()) {
        int id = (int)m.second.as_int();
        t.vocab_.emplace(m.first, id);
        std::string raw;
        for (uint32_t cp : to_codepoints(m.first)) {
            auto it = inv.find(cp);
            if (it == inv.end()) throw std::runtime_error("tokenizer: vocab entry outside the byte alphabet");
            raw += (char)it->second;
        }
        t.id_bytes_[id] = std::move(raw);
    }
    for (int b = 0; b < 256; ++b) {
        auto it = t.vocab_.find(t.byte_str_[b]);
        if (it == t.vocab_.end()) throw std::runtime_error("tokenizer: byte token missing from vocab");
        t.byte_id_[b] = it->second;
    }
    t.merges_.reserve(merges->size() * 2);
    int rank = 0;
    for (auto& m : merges->items()) {
        std::string a, b;
        if (m.is_array()) { a = m.items().at(0).as_string(); b = m.items().at(1).as_string(); }
        else {
            const std::string& s = m.as_string();
            size_t sp = s.find(' ', 1);  // first char may itself be a space-like byte char; they are never ' ' in byte alphabet
            sp = s.find(' ');
            if (sp == std::string::npos) throw std::runtime_error("tokenizer: bad merge");
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        }
        auto ia = t.vocab_.find(a), ib = t.vocab_.find(b), ir = t.vocab_.find(a + b);
        if (ia == t.vocab_.end() || ib == t.vocab_.end() || ir == t.vocab_.end()) { ++rank; continue; }
        t.merges_.emplace(((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second, Merge{rank, ir->second});
        ++rank;
    }
    if (added)
        for (auto& a : added->items()) {
            int id = (int)a.find("id")->as_int();
            const std::string& content = a.find("content")->as_string();
            t.id_bytes_[id] = content;
            t.is_added_[id] = 1;
            t.added_.emplace_back(content, id);
        }
    std::sort(t.added_.begin(), t.added_.end(), [](auto& x, auto& y) { return x.first.size() > y.first.size(); });
    return t;
}

}  // namespace qv
