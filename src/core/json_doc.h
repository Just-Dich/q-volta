// Small strict JSON document model: parse (RFC 8259, depth-limited, UTF-8 validated) and serialize.
// Objects keep insertion order. Used by the tokenizer loader and the HTTP server.
#pragma once
#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qv::jd {

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Value {
   public:
    enum class Type { Null, Bool, Int, Num, Str, Arr, Obj };
    using Member = std::pair<std::string, Value>;

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : t_(Type::Bool), i_(b) {}
    Value(int v) : t_(Type::Int), i_(v) {}
    Value(long v) : t_(Type::Int), i_(v) {}
    Value(long long v) : t_(Type::Int), i_(v) {}
    Value(unsigned long v) : t_(Type::Int), i_((long long)v) {}
    Value(double v) : t_(Type::Num), d_(v) {}
    Value(const char* s) : t_(Type::Str), s_(s) {}
    Value(std::string s) : t_(Type::Str), s_(std::move(s)) {}
    static Value array() { Value v; v.t_ = Type::Arr; return v; }
    static Value object() { Value v; v.t_ = Type::Obj; return v; }

    Type type() const { return t_; }
    bool is_null() const { return t_ == Type::Null; }
    bool is_bool() const { return t_ == Type::Bool; }
    bool is_int() const { return t_ == Type::Int; }
    bool is_number() const { return t_ == Type::Int || t_ == Type::Num; }
    bool is_string() const { return t_ == Type::Str; }
    bool is_array() const { return t_ == Type::Arr; }
    bool is_object() const { return t_ == Type::Obj; }

    bool as_bool() const { return i_ != 0; }
    long long as_int() const { return t_ == Type::Int ? i_ : (long long)d_; }
    double as_double() const { return t_ == Type::Int ? (double)i_ : d_; }
    const std::string& as_string() const { return s_; }
    const std::vector<Value>& items() const { return a_; }
    std::vector<Value>& items() { return a_; }
    const std::vector<Member>& members() const { return o_; }
    size_t size() const { return t_ == Type::Arr ? a_.size() : o_.size(); }

    const Value* find(std::string_view k) const {
        for (size_t i = o_.size(); i-- > 0;)  // last duplicate wins
            if (o_[i].first == k) return &o_[i].second;
        return nullptr;
    }
    // Append without duplicate check (parser; keeps large objects O(n)).
    Value& append(std::string k, Value v) { o_.emplace_back(std::move(k), std::move(v)); return o_.back().second; }
    // Insert or replace a member (objects) — returns the stored value.
    Value& set(std::string k, Value v) {
        for (auto& m : o_)
            if (m.first == k) { m.second = std::move(v); return m.second; }
        o_.emplace_back(std::move(k), std::move(v));
        return o_.back().second;
    }
    Value& push(Value v) { a_.push_back(std::move(v)); return a_.back(); }

    std::string dump() const { std::string s; dump_to(s); return s; }
    void dump_to(std::string& out) const {
        switch (t_) {
            case Type::Null: out += "null"; break;
            case Type::Bool: out += i_ ? "true" : "false"; break;
            case Type::Int: { char b[24]; auto r = std::to_chars(b, b + 24, i_); out.append(b, r.ptr); break; }
            case Type::Num: {
                if (!std::isfinite(d_)) { out += "null"; break; }
                char b[40];
                auto r = std::to_chars(b, b + 40, d_);
                out.append(b, r.ptr);
                if (std::find_if(b, r.ptr, [](char c) { return c == '.' || c == 'e' || c == 'E'; }) == r.ptr) out += ".0";  // keep floats floats (like nlohmann)
                break;
            }
            case Type::Str: dump_string(out, s_); break;
            case Type::Arr:
                out += '[';
                for (size_t i = 0; i < a_.size(); ++i) { if (i) out += ','; a_[i].dump_to(out); }
                out += ']';
                break;
            case Type::Obj:
                out += '{';
                for (size_t i = 0; i < o_.size(); ++i) {
                    if (i) out += ',';
                    dump_string(out, o_[i].first);
                    out += ':';
                    o_[i].second.dump_to(out);
                }
                out += '}';
                break;
        }
    }
    static void dump_string(std::string& out, std::string_view s) {
        static const char* hex = "0123456789abcdef";
        out += '"';
        for (unsigned char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
                    else out += (char)c;
            }
        }
        out += '"';
    }

   private:
    Type t_ = Type::Null;
    long long i_ = 0;
    double d_ = 0;
    std::string s_;
    std::vector<Value> a_;
    std::vector<Member> o_;
};

class Parser {
   public:
    explicit Parser(std::string_view s) : p_(s.data()), end_(s.data() + s.size()) {}
    Value parse() {
        Value v = value(0);
        ws();
        if (p_ != end_) fail("trailing characters after JSON value");
        return v;
    }

   private:
    const char* p_;
    const char* end_;
    [[noreturn]] void fail(const char* m) { throw ParseError(m); }
    void ws() { while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
    bool lit(const char* w) {
        size_t n = std::char_traits<char>::length(w);
        if ((size_t)(end_ - p_) >= n && std::string_view(p_, n) == w) { p_ += n; return true; }
        return false;
    }
    Value value(int depth) {
        if (depth > 128) fail("nesting too deep");
        ws();
        if (p_ >= end_) fail("unexpected end of input");
        switch (*p_) {
            case '{': {
                ++p_;
                Value v = Value::object();
                ws();
                if (p_ < end_ && *p_ == '}') { ++p_; return v; }
                for (;;) {
                    ws();
                    if (p_ >= end_ || *p_ != '"') fail("expected string key");
                    std::string k = str();
                    ws();
                    if (p_ >= end_ || *p_ != ':') fail("expected ':'");
                    ++p_;
                    v.append(std::move(k), value(depth + 1));
                    ws();
                    if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                    if (p_ < end_ && *p_ == '}') { ++p_; return v; }
                    fail("expected ',' or '}'");
                }
            }
            case '[': {
                ++p_;
                Value v = Value::array();
                ws();
                if (p_ < end_ && *p_ == ']') { ++p_; return v; }
                for (;;) {
                    v.push(value(depth + 1));
                    ws();
                    if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                    if (p_ < end_ && *p_ == ']') { ++p_; return v; }
                    fail("expected ',' or ']'");
                }
            }
            case '"': return Value(str());
            case 't': if (lit("true")) return Value(true); fail("invalid literal");
            case 'f': if (lit("false")) return Value(false); fail("invalid literal");
            case 'n': if (lit("null")) return Value(); fail("invalid literal");
            default: return number();
        }
    }
    Value number() {
        const char* s = p_;
        if (p_ < end_ && *p_ == '-') ++p_;
        if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid number");
        if (*p_ == '0') ++p_;
        else while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        bool real = false;
        if (p_ < end_ && *p_ == '.') {
            real = true;
            ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid number");
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            real = true;
            ++p_;
            if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') fail("invalid number");
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (!real) {
            long long v;
            auto r = std::from_chars(s, p_, v);
            if (r.ec == std::errc() && r.ptr == p_) return Value(v);
        }
        double d;
        auto r = std::from_chars(s, p_, d);
        if (r.ec == std::errc::result_out_of_range) fail("number out of range");
        if (r.ec != std::errc()) fail("invalid number");
        return Value(d);
    }
    static void put_utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    }
    unsigned hex4() {
        if (end_ - p_ < 4) fail("bad \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad \\u escape");
        }
        return v;
    }
    std::string str() {
        ++p_;  // opening quote
        std::string out;
        for (;;) {
            if (p_ >= end_) fail("unterminated string");
            unsigned char c = (unsigned char)*p_;
            if (c == '"') { ++p_; return out; }
            if (c < 0x20) fail("control character in string");
            if (c == '\\') {
                ++p_;
                if (p_ >= end_) fail("unterminated string");
                char e = *p_++;
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        unsigned cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                                const char* save = p_;
                                p_ += 2;
                                unsigned lo = hex4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                else { p_ = save; cp = 0xFFFD; }
                            } else cp = 0xFFFD;
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) cp = 0xFFFD;
                        put_utf8(out, cp);
                        break;
                    }
                    default: fail("invalid escape");
                }
                continue;
            }
            if (c < 0x80) { out += (char)c; ++p_; continue; }
            // validate one UTF-8 sequence
            int n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 ? 2 : 0;
            if (!n || c > 0xF4 || end_ - p_ < n) fail("invalid UTF-8");
            unsigned cp = c & (0xFF >> (n + 1));
            for (int i = 1; i < n; ++i) {
                unsigned char cc = (unsigned char)p_[i];
                if ((cc & 0xC0) != 0x80) fail("invalid UTF-8");
                cp = (cp << 6) | (cc & 63);
            }
            if ((n == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) || (n == 4 && (cp < 0x10000 || cp > 0x10FFFF))) fail("invalid UTF-8");
            out.append(p_, n);
            p_ += n;
        }
    }
};

inline Value parse(std::string_view s) { return Parser(s).parse(); }

}  // namespace qv::jd
