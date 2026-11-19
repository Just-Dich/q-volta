// Minimal JSON reader (objects, arrays, strings, numbers, bools, null) — enough for the .qv.json header.
#pragma once
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qv::json {

struct Value {
    enum class T { Null, Bool, Num, Str, Arr, Obj } t = T::Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Value> a;
    std::map<std::string, Value> o;

    const Value& operator[](const std::string& k) const {
        auto it = o.find(k);
        if (it == o.end()) throw std::runtime_error("json: missing key " + k);
        return it->second;
    }
    const Value& operator[](size_t i) const { return a.at(i); }
    bool has(const std::string& k) const { return o.count(k) != 0; }
    double num() const { return n; }
    long long i64() const { return (long long)n; }
    const std::string& str() const { return s; }
};

class Parser {
   public:
    explicit Parser(const std::string& src) : p_(src.c_str()), end_(src.c_str() + src.size()) {}
    Value parse() {
        Value v = value();
        ws();
        return v;
    }

   private:
    const char* p_;
    const char* end_;
    void ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
    }
    [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
    Value value() {
        ws();
        if (p_ >= end_) fail("eof");
        Value v;
        switch (*p_) {
            case '{': {
                v.t = Value::T::Obj;
                ++p_;
                ws();
                if (*p_ == '}') { ++p_; return v; }
                for (;;) {
                    ws();
                    std::string k = string();
                    ws();
                    if (*p_++ != ':') fail("expected :");
                    v.o.emplace(std::move(k), value());
                    ws();
                    if (*p_ == ',') { ++p_; continue; }
                    if (*p_ == '}') { ++p_; break; }
                    fail("expected , or }");
                }
                return v;
            }
            case '[': {
                v.t = Value::T::Arr;
                ++p_;
                ws();
                if (*p_ == ']') { ++p_; return v; }
                for (;;) {
                    v.a.push_back(value());
                    ws();
                    if (*p_ == ',') { ++p_; continue; }
                    if (*p_ == ']') { ++p_; break; }
                    fail("expected , or ]");
                }
                return v;
            }
            case '"':
                v.t = Value::T::Str;
                v.s = string();
                return v;
            case 't': p_ += 4; v.t = Value::T::Bool; v.b = true; return v;
            case 'f': p_ += 5; v.t = Value::T::Bool; v.b = false; return v;
            case 'n': p_ += 4; return v;
            default: {
                char* e = nullptr;
                v.t = Value::T::Num;
                v.n = std::strtod(p_, &e);
                if (e == p_) fail("bad number");
                p_ = e;
                return v;
            }
        }
    }
    std::string string() {
        if (*p_ != '"') fail("expected string");
        ++p_;
        std::string out;
        while (p_ < end_ && *p_ != '"') {
            if (*p_ == '\\') {
                ++p_;
                char c = *p_++;
                switch (c) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        unsigned cp = std::strtoul(std::string(p_, p_ + 4).c_str(), nullptr, 16);
                        p_ += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && p_[0] == '\\' && p_[1] == 'u') {
                            unsigned lo = std::strtoul(std::string(p_ + 2, p_ + 6).c_str(), nullptr, 16);
                            p_ += 6;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        if (cp < 0x80) out += char(cp);
                        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 63)); }
                        else if (cp < 0x10000) {
                            out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 63)); out += char(0x80 | (cp & 63));
                        } else {
                            out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 63));
                            out += char(0x80 | ((cp >> 6) & 63)); out += char(0x80 | (cp & 63));
                        }
                        break;
                    }
                    default: out += c;
                }
            } else {
                out += *p_++;
            }
        }
        ++p_;
        return out;
    }
};

inline Value parse(const std::string& s) { return Parser(s).parse(); }

}  // namespace qv::json
