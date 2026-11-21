#include "jinja/value.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>

#include "core/json_doc.h"

namespace qv::jinja {

// ---------------------------------------------------------------- construction
Value Value::list(std::vector<Value> v) {
    Value r;
    r.k_ = Kind::List;
    auto d = std::make_shared<ListData>();
    d->v = std::move(v);
    r.o_ = std::move(d);
    return r;
}
Value Value::tuple(std::vector<Value> v) {
    Value r = list(std::move(v));
    r.k_ = Kind::Tuple;
    return r;
}
Value Value::dict() {
    Value r;
    r.k_ = Kind::Dict;
    r.o_ = std::make_shared<DictData>();
    return r;
}
Value Value::ns() {
    Value r = dict();
    r.ns_ = true;
    return r;
}
Value Value::func(std::string name, NativeFn fn) {
    Value r;
    r.k_ = Kind::Func;
    auto d = std::make_shared<FuncData>();
    d->name = std::move(name);
    d->fn = std::move(fn);
    r.o_ = std::move(d);
    return r;
}
Value Value::loop(std::shared_ptr<Obj> st) {
    Value r;
    r.k_ = Kind::Loop;
    r.o_ = std::move(st);
    return r;
}

std::vector<Value>& Value::items() { return static_cast<ListData*>(o_.get())->v; }
const std::vector<Value>& Value::items() const { return static_cast<const ListData*>(o_.get())->v; }
Value::DictData& Value::dict_data() { return *static_cast<DictData*>(o_.get()); }
const Value::DictData& Value::dict_data() const { return *static_cast<const DictData*>(o_.get()); }

int Value::DictData::find_idx(const Value& key) const {
    for (size_t i = 0; i < kv.size(); ++i)
        if (Value::equals(kv[i].first, key)) return (int)i;
    return -1;
}
const Value* Value::find(const std::string& key) const {
    if (k_ != Kind::Dict) return nullptr;
    for (auto& p : dict_data().kv)
        if (p.first.k_ == Kind::Str && p.first.s_ == key) return &p.second;
    return nullptr;
}
void Value::set(const Value& key, Value v) {
    auto& d = dict_data();
    int i = d.find_idx(key);
    if (i >= 0) d.kv[i].second = std::move(v);
    else d.kv.emplace_back(key, std::move(v));
}
void Value::set(const std::string& key, Value v) { set(Value(key), std::move(v)); }

// ---------------------------------------------------------------- utf-8
size_t u8_length(std::string_view s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}
std::vector<uint32_t> u8_decode(std::string_view s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        uint32_t cp;
        int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 31; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 15; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 7; n = 4; }
        else { cp = 0xFFFD; n = 1; }
        for (int k = 1; k < n; ++k) {
            if (i + k < s.size() && (((unsigned char)s[i + k]) & 0xC0) == 0x80) cp = (cp << 6) | (s[i + k] & 63);
            else { cp = 0xFFFD; n = k; break; }
        }
        out.push_back(cp);
        i += n;
    }
    return out;
}
void u8_append(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
}
std::string u8_substr(std::string_view s, int64_t start, int64_t count) {
    // start/count in code points (already clamped by caller)
    size_t i = 0;
    int64_t cp = 0;
    while (i < s.size() && cp < start) { ++i; while (i < s.size() && (((unsigned char)s[i]) & 0xC0) == 0x80) ++i; ++cp; }
    size_t b = i;
    cp = 0;
    while (i < s.size() && cp < count) { ++i; while (i < s.size() && (((unsigned char)s[i]) & 0xC0) == 0x80) ++i; ++cp; }
    return std::string(s.substr(b, i - b));
}

// ---------------------------------------------------------------- numbers / repr
std::string py_float_repr(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
    if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + 64, d, std::chars_format::scientific);
    std::string sci(buf, r.ptr);  // d.ddde[+-]xx
    bool neg = sci[0] == '-';
    if (neg) sci.erase(0, 1);
    size_t e = sci.find('e');
    std::string mant = sci.substr(0, e);
    int exp = atoi(sci.c_str() + e + 1);
    std::string digits;
    for (char c : mant) if (c != '.') digits += c;
    std::string out;
    if (exp < -4 || exp >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    } else if (exp < 0) {
        out = "0." + std::string(-exp - 1, '0') + digits;
    } else {
        if ((int)digits.size() <= exp + 1) {
            out = digits + std::string(exp + 1 - digits.size(), '0') + ".0";
        } else {
            out = digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
        }
    }
    return neg ? "-" + out : out;
}

std::string py_str_repr(std::string_view s) {
    bool has_sq = s.find('\'') != std::string_view::npos, has_dq = s.find('"') != std::string_view::npos;
    char q = (has_sq && !has_dq) ? '"' : '\'';
    std::string o(1, q);
    for (uint32_t cp : u8_decode(s)) {
        if (cp == (uint32_t)q || cp == '\\') { o += '\\'; o += (char)cp; }
        else if (cp == '\n') o += "\\n";
        else if (cp == '\r') o += "\\r";
        else if (cp == '\t') o += "\\t";
        else if (cp < 0x20 || cp == 0x7f || (cp >= 0x80 && cp < 0xa0) || cp == 0xad) {
            char b[16];
            snprintf(b, sizeof b, "\\x%02x", cp);
            o += b;
        } else u8_append(o, cp);
    }
    o += q;
    return o;
}

const char* Value::type_name() const {
    switch (k_) {
        case Kind::Undefined: return "Undefined";
        case Kind::None: return "NoneType";
        case Kind::Bool: return "bool";
        case Kind::Int: return "int";
        case Kind::Float: return "float";
        case Kind::Str: return "str";
        case Kind::List: return "list";
        case Kind::Tuple: return "tuple";
        case Kind::Dict: return ns_ ? "Namespace" : "dict";
        case Kind::Func: return "function";
        case Kind::Loop: return "LoopContext";
    }
    return "?";
}

bool Value::truthy() const {
    switch (k_) {
        case Kind::Undefined: case Kind::None: return false;
        case Kind::Bool: case Kind::Int: return i_ != 0;
        case Kind::Float: return d_ != 0;
        case Kind::Str: return !s_.empty();
        case Kind::List: case Kind::Tuple: return !items().empty();
        case Kind::Dict: return ns_ ? true : !dict_data().kv.empty();
        default: return true;
    }
}

size_t Value::length() const {
    switch (k_) {
        case Kind::Undefined: return 0;
        case Kind::Str: return u8_length(s_);
        case Kind::List: case Kind::Tuple: return items().size();
        case Kind::Dict: return dict_data().kv.size();
        default: throw Error(std::string("object of type '") + type_name() + "' has no len()");
    }
}

std::string Value::repr() const {
    switch (k_) {
        case Kind::Undefined: return "Undefined";
        case Kind::Str: return py_str_repr(s_);
        case Kind::List: case Kind::Tuple: {
            std::string o(1, k_ == Kind::List ? '[' : '(');
            bool first = true;
            for (auto& v : items()) { if (!first) o += ", "; first = false; o += v.repr(); }
            if (k_ == Kind::Tuple && items().size() == 1) o += ",";
            o += k_ == Kind::List ? ']' : ')';
            return o;
        }
        case Kind::Dict: {
            std::string o = ns_ ? "<Namespace {" : "{";
            bool first = true;
            for (auto& p : dict_data().kv) {
                if (!first) o += ", ";
                first = false;
                o += p.first.repr() + ": " + p.second.repr();
            }
            return o + (ns_ ? "}>" : "}");
        }
        case Kind::Func: return "<function " + static_cast<const FuncData*>(o_.get())->name + ">";
        default: return str();
    }
}

std::string Value::str() const {
    switch (k_) {
        case Kind::Undefined: return "";
        case Kind::None: return "None";
        case Kind::Bool: return i_ ? "True" : "False";
        case Kind::Int: return std::to_string(i_);
        case Kind::Float: return py_float_repr(d_);
        case Kind::Str: return s_;
        case Kind::Loop: return "<LoopContext>";
        default: return repr();
    }
}

bool Value::equals(const Value& a, const Value& b) {
    bool an = a.k_ == Kind::Bool || a.k_ == Kind::Int || a.k_ == Kind::Float;
    bool bn = b.k_ == Kind::Bool || b.k_ == Kind::Int || b.k_ == Kind::Float;
    if (an && bn) {
        if (a.k_ == Kind::Float || b.k_ == Kind::Float) return a.as_double() == b.as_double();
        return a.i_ == b.i_;
    }
    if (a.k_ != b.k_) return false;
    switch (a.k_) {
        case Kind::Undefined: case Kind::None: return true;
        case Kind::Str: return a.s_ == b.s_;
        case Kind::List: case Kind::Tuple: {
            auto &x = a.items(), &y = b.items();
            if (x.size() != y.size()) return false;
            for (size_t i = 0; i < x.size(); ++i)
                if (!equals(x[i], y[i])) return false;
            return true;
        }
        case Kind::Dict: {
            if (a.o_ == b.o_) return true;
            const auto& x = a.dict_data().kv;
            const auto& y = b.dict_data();
            if (x.size() != y.kv.size()) return false;
            for (auto& p : x) {
                int i = y.find_idx(p.first);
                if (i < 0 || !equals(p.second, y.kv[i].second)) return false;
            }
            return true;
        }
        default: return a.o_ == b.o_;
    }
}

int Value::compare(const Value& a, const Value& b) {
    bool an = a.k_ == Kind::Bool || a.k_ == Kind::Int || a.k_ == Kind::Float;
    bool bn = b.k_ == Kind::Bool || b.k_ == Kind::Int || b.k_ == Kind::Float;
    if (an && bn) {
        if (a.k_ == Kind::Float || b.k_ == Kind::Float) {
            double x = a.as_double(), y = b.as_double();
            return x < y ? -1 : x > y ? 1 : 0;
        }
        return a.i_ < b.i_ ? -1 : a.i_ > b.i_ ? 1 : 0;
    }
    if (a.k_ == Kind::Str && b.k_ == Kind::Str) {
        int c = a.s_.compare(b.s_);
        return c < 0 ? -1 : c > 0 ? 1 : 0;
    }
    if (a.k_ == b.k_ && a.is_seq()) {
        auto &x = a.items(), &y = b.items();
        for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
            if (!equals(x[i], y[i])) return compare(x[i], y[i]);
        }
        return x.size() < y.size() ? -1 : x.size() > y.size() ? 1 : 0;
    }
    throw Error(std::string("'<' not supported between instances of '") + a.type_name() + "' and '" + b.type_name() + "'");
}

// ---------------------------------------------------------------- json.dumps
namespace {
void json_str(std::string& out, std::string_view s, bool ascii) {
    static const char* hex = "0123456789abcdef";
    out += '"';
    auto esc_u = [&](uint32_t u) {
        out += "\\u";
        for (int sh = 12; sh >= 0; sh -= 4) out += hex[(u >> sh) & 15];
    };
    if (!ascii) {
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
                    if (c < 0x20) esc_u(c);
                    else out += (char)c;
            }
        }
    } else {
        for (uint32_t cp : u8_decode(s)) {
            switch (cp) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (cp < 0x20 || cp > 0x7e) {
                        if (cp == 0x7f) { out += (char)cp; break; }
                        if (cp >= 0x10000) { cp -= 0x10000; esc_u(0xD800 + (cp >> 10)); esc_u(0xDC00 + (cp & 0x3ff)); }
                        else esc_u(cp);
                    } else out += (char)cp;
            }
        }
    }
    out += '"';
}
void dump(std::string& out, const Value& v, const JsonOpts& o, int depth) {
    auto nl = [&](int d) {
        out += '\n';
        out.append((size_t)d * o.indent, ' ');
    };
    switch (v.kind()) {
        case Value::Kind::None: out += "null"; break;
        case Value::Kind::Bool: out += v.as_bool() ? "true" : "false"; break;
        case Value::Kind::Int: out += std::to_string(v.as_int()); break;
        case Value::Kind::Float: {
            double d = v.as_double();
            out += std::isnan(d) ? "NaN" : std::isinf(d) ? (d < 0 ? "-Infinity" : "Infinity") : py_float_repr(d);
            break;
        }
        case Value::Kind::Str: json_str(out, v.as_string(), o.ensure_ascii); break;
        case Value::Kind::List: case Value::Kind::Tuple: {
            auto& it = v.items();
            if (it.empty()) { out += "[]"; break; }
            out += '[';
            for (size_t i = 0; i < it.size(); ++i) {
                if (i) out += o.item_sep;
                if (o.indent >= 0) nl(depth + 1);
                dump(out, it[i], o, depth + 1);
            }
            if (o.indent >= 0) nl(depth);
            out += ']';
            break;
        }
        case Value::Kind::Dict: {
            std::vector<const std::pair<Value, Value>*> kv;
            for (auto& p : v.dict_data().kv) kv.push_back(&p);
            if (kv.empty()) { out += "{}"; break; }
            if (o.sort_keys)
                std::stable_sort(kv.begin(), kv.end(), [](auto* a, auto* b) { return Value::compare(a->first, b->first) < 0; });
            out += '{';
            for (size_t i = 0; i < kv.size(); ++i) {
                if (i) out += o.item_sep;
                if (o.indent >= 0) nl(depth + 1);
                const Value& k = kv[i]->first;
                std::string ks = k.is_string() ? k.as_string() : k.is_none() ? "null" : k.is_bool() ? (k.as_bool() ? "true" : "false") : k.str();
                json_str(out, ks, o.ensure_ascii);
                out += o.key_sep;
                dump(out, kv[i]->second, o, depth + 1);
            }
            if (o.indent >= 0) nl(depth);
            out += '}';
            break;
        }
        default: throw Error(std::string("Object of type ") + v.type_name() + " is not JSON serializable");
    }
}
}  // namespace

std::string to_json_string(const Value& v, const JsonOpts& o) {
    std::string out;
    dump(out, v, o, 0);
    return out;
}

Value from_json(const jd::Value& j) {
    switch (j.type()) {
        case jd::Value::Type::Null: return Value::none();
        case jd::Value::Type::Bool: return Value(j.as_bool());
        case jd::Value::Type::Int: return Value((long long)j.as_int());
        case jd::Value::Type::Num: return Value(j.as_double());
        case jd::Value::Type::Str: return Value(j.as_string());
        case jd::Value::Type::Arr: {
            Value l = Value::list();
            for (auto& x : j.items()) l.push(from_json(x));
            return l;
        }
        case jd::Value::Type::Obj: {
            Value d = Value::dict();
            for (auto& m : j.members()) d.set(m.first, from_json(m.second));
            return d;
        }
    }
    return Value();
}

jd::Value to_json(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Undefined: case Value::Kind::None: return jd::Value();
        case Value::Kind::Bool: return jd::Value(v.as_bool());
        case Value::Kind::Int: return jd::Value((long long)v.as_int());
        case Value::Kind::Float: return jd::Value(v.as_double());
        case Value::Kind::Str: return jd::Value(v.as_string());
        case Value::Kind::List: case Value::Kind::Tuple: {
            jd::Value a = jd::Value::array();
            for (auto& x : v.items()) a.push(to_json(x));
            return a;
        }
        case Value::Kind::Dict: {
            jd::Value o = jd::Value::object();
            for (auto& p : v.dict_data().kv) o.set(p.first.str(), to_json(p.second));
            return o;
        }
        default: throw Error(std::string("cannot convert ") + v.type_name() + " to JSON");
    }
}

}  // namespace qv::jinja
