// Python-compatible builtin filters, tests, str/list/dict methods and operators for the jinja interpreter.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

#include "core/json_doc.h"
#include "jinja/env_impl.h"

namespace qv::jinja {
namespace {

[[noreturn]] void err(const std::string& m) { throw Error(m); }

Value S(std::string s) { return Value(std::move(s)); }
Value Lst(std::vector<Value> v = {}) { return Value::list(std::move(v)); }

// ------------------------------------------------------------------------ unicode helpers
bool is_py_space(uint32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
           c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
uint32_t lower_cp(uint32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if ((c >= 0xC0 && c <= 0xDE && c != 0xD7)) return c + 32;
    if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    if (c >= 0x100 && c <= 0x17F && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17F) {
        bool odd_up = (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E);
        if (odd_up) return (c & 1) ? c + 1 : c;
        return (c & 1) ? c : c + 1;
    }
    return c;
}
uint32_t upper_cp(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') ? c - 32 : c;
    if ((c >= 0xE0 && c <= 0xFE && c != 0xF7)) return c - 32;
    if (c == 0xFF) return 0x178;
    if (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    if (c >= 0x100 && c <= 0x17F && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17F) {
        bool odd_up = (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E);
        if (odd_up) return (c & 1) ? c : c - 1;
        return (c & 1) ? c - 1 : c;
    }
    return c;
}
bool is_alpha_cp(uint32_t c) { return (c < 0x80) ? isalpha((int)c) != 0 : (lower_cp(c) != c || upper_cp(c) != c || (c >= 0xAA && c < 0x2000 && c != 0xD7 && c != 0xF7) || c >= 0x3040); }
std::string map_case(const std::string& s, bool up) {
    std::string o;
    o.reserve(s.size());
    bool ascii = true;
    for (unsigned char c : s) if (c >= 0x80) { ascii = false; break; }
    if (ascii) {
        for (char c : s) o += up ? (char)toupper((unsigned char)c) : (char)tolower((unsigned char)c);
        return o;
    }
    for (uint32_t cp : u8_decode(s)) {
        if (up && cp == 0xDF) { o += "SS"; continue; }
        u8_append(o, up ? upper_cp(cp) : lower_cp(cp));
    }
    return o;
}
std::string py_strip(const std::string& s, const Value* chars, bool left, bool right) {
    auto cps = u8_decode(s);
    std::vector<uint32_t> set;
    bool has = chars && !chars->is_none();
    if (has) set = u8_decode(chars->as_string());
    auto strip_it = [&](uint32_t c) { return has ? std::find(set.begin(), set.end(), c) != set.end() : is_py_space(c); };
    size_t b = 0, e = cps.size();
    if (left) while (b < e && strip_it(cps[b])) ++b;
    if (right) while (e > b && strip_it(cps[e - 1])) --e;
    std::string o;
    for (size_t i = b; i < e; ++i) u8_append(o, cps[i]);
    return o;
}

// ------------------------------------------------------------------------ arg helpers
const Value& need_str(const Value& v, const char* what) {
    if (!v.is_string()) err(std::string(what) + " must be str, not " + v.type_name());
    return v;
}
int64_t opt_int(const Args& a, size_t i, const char* name, int64_t def) {
    const Value* v = a.get(i, name);
    if (!v || v->is_none()) return def;
    if (!v->is_number()) err(std::string("'") + v->type_name() + "' object cannot be interpreted as an integer");
    return v->as_int();
}
bool opt_bool(const Args& a, size_t i, const char* name, bool def) {
    const Value* v = a.get(i, name);
    return v ? v->truthy() : def;
}

std::vector<Value> split_ws(const std::string& s, int64_t maxsplit) {
    std::vector<Value> out;
    auto cps = u8_decode(s);
    size_t i = 0, n = cps.size();
    while (i < n) {
        while (i < n && is_py_space(cps[i])) ++i;
        if (i >= n) break;
        if (maxsplit >= 0 && (int64_t)out.size() == maxsplit) {
            std::string rest;
            for (size_t k = i; k < n; ++k) u8_append(rest, cps[k]);
            size_t e = rest.size();
            out.push_back(S(py_strip(rest, nullptr, false, true)));
            (void)e;
            return out;
        }
        std::string w;
        while (i < n && !is_py_space(cps[i])) u8_append(w, cps[i++]);
        out.push_back(S(w));
    }
    return out;
}
std::vector<Value> split_sep(const std::string& s, const std::string& sep, int64_t maxsplit) {
    if (sep.empty()) err("empty separator");
    std::vector<Value> out;
    size_t pos = 0;
    while (maxsplit < 0 || (int64_t)out.size() < maxsplit) {
        size_t f = s.find(sep, pos);
        if (f == std::string::npos) break;
        out.push_back(S(s.substr(pos, f - pos)));
        pos = f + sep.size();
    }
    out.push_back(S(s.substr(pos)));
    return out;
}
std::vector<Value> rsplit_sep(const std::string& s, const std::string& sep, int64_t maxsplit) {
    if (sep.empty()) err("empty separator");
    std::vector<Value> out;
    size_t end = s.size();
    while (maxsplit < 0 || (int64_t)out.size() < maxsplit) {
        if (end < sep.size()) break;
        size_t f = s.rfind(sep, end - sep.size());
        if (f == std::string::npos) break;
        out.push_back(S(s.substr(f + sep.size(), end - f - sep.size())));
        end = f;
    }
    out.push_back(S(s.substr(0, end)));
    std::reverse(out.begin(), out.end());
    return out;
}
std::vector<Value> splitlines(const std::string& s, bool keepends) {
    std::vector<Value> out;
    auto cps = u8_decode(s);
    std::string cur;
    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t c = cps[i];
        bool brk = c == '\n' || c == '\r' || c == 0x0b || c == 0x0c || (c >= 0x1c && c <= 0x1e) || c == 0x85 || c == 0x2028 || c == 0x2029;
        if (!brk) { u8_append(cur, c); continue; }
        if (keepends) { u8_append(cur, c); if (c == '\r' && i + 1 < cps.size() && cps[i + 1] == '\n') { u8_append(cur, '\n'); ++i; } }
        else if (c == '\r' && i + 1 < cps.size() && cps[i + 1] == '\n') ++i;
        out.push_back(S(cur));
        cur.clear();
    }
    if (!cur.empty()) out.push_back(S(cur));
    return out;
}

int64_t cp_find(const std::string& s, const std::string& sub, int64_t start, int64_t end, bool rev) {
    auto cs = u8_decode(s), ct = u8_decode(sub);
    int64_t n = (int64_t)cs.size();
    if (start < 0) start = std::max<int64_t>(0, start + n);
    if (end < 0) end = std::max<int64_t>(0, end + n);
    end = std::min(end, n);
    int64_t m = (int64_t)ct.size();
    if (rev) {
        for (int64_t i = end - m; i >= start; --i)
            if (std::equal(ct.begin(), ct.end(), cs.begin() + i)) return i;
    } else {
        for (int64_t i = start; i + m <= end; ++i)
            if (std::equal(ct.begin(), ct.end(), cs.begin() + i)) return i;
    }
    return -1;
}
std::string replace_all(const std::string& s, const std::string& from, const std::string& to, int64_t count) {
    if (count == 0) return s;
    std::string o;
    if (from.empty()) {  // python inserts between every code point
        auto cps = u8_decode(s);
        int64_t done = 0;
        for (size_t i = 0; i <= cps.size(); ++i) {
            if (count < 0 || done < count) { o += to; ++done; }
            if (i < cps.size()) u8_append(o, cps[i]);
        }
        return o;
    }
    size_t pos = 0;
    int64_t done = 0;
    while (count < 0 || done < count) {
        size_t f = s.find(from, pos);
        if (f == std::string::npos) break;
        o.append(s, pos, f - pos);
        o += to;
        pos = f + from.size();
        ++done;
    }
    o.append(s, pos, std::string::npos);
    return o;
}

// ------------------------------------------------------------------------ python formatting
struct Spec {
    std::string fill = " ";
    char align = 0, sign = '-', type = 0;
    bool alt = false, zero = false, comma = false;
    int width = -1, prec = -1;
};
Spec parse_spec(const std::string& sp) {
    Spec r;
    auto cps = u8_decode(sp);
    size_t i = 0;
    auto is_al = [](uint32_t c) { return c == '<' || c == '>' || c == '^' || c == '='; };
    if (cps.size() >= 2 && is_al(cps[1])) { r.fill.clear(); u8_append(r.fill, cps[0]); r.align = (char)cps[1]; i = 2; }
    else if (!cps.empty() && is_al(cps[0])) { r.align = (char)cps[0]; i = 1; }
    if (i < cps.size() && (cps[i] == '+' || cps[i] == '-' || cps[i] == ' ')) r.sign = (char)cps[i++];
    if (i < cps.size() && cps[i] == '#') { r.alt = true; ++i; }
    if (i < cps.size() && cps[i] == '0') { r.zero = true; ++i; }
    if (i < cps.size() && isdigit((int)cps[i])) { r.width = 0; while (i < cps.size() && isdigit((int)cps[i])) r.width = r.width * 10 + (int)(cps[i++] - '0'); }
    if (i < cps.size() && (cps[i] == ',' || cps[i] == '_')) { r.comma = true; ++i; }
    if (i < cps.size() && cps[i] == '.') { ++i; r.prec = 0; while (i < cps.size() && isdigit((int)cps[i])) r.prec = r.prec * 10 + (int)(cps[i++] - '0'); }
    if (i < cps.size()) r.type = (char)cps[i];
    return r;
}
std::string group_thousands(const std::string& digits) {
    std::string o;
    int n = (int)digits.size();
    for (int i = 0; i < n; ++i) {
        o += digits[i];
        if ((n - i - 1) % 3 == 0 && i != n - 1) o += ',';
    }
    return o;
}
std::string format_value(const Value& v, const std::string& specs) {
    if (specs.empty()) return v.str();
    Spec sp = parse_spec(specs);
    std::string body;
    bool numeric = false;
    std::string sign;
    if (v.is_string() || (!v.is_number() && !v.is_bool())) {
        body = v.str();
        if (sp.prec >= 0) body = u8_substr(body, 0, sp.prec);
    } else {
        numeric = true;
        char t = sp.type;
        if (v.is_float() || t == 'f' || t == 'e' || t == 'g' || t == 'F' || t == 'E' || t == 'G' || t == '%') {
            double d = v.as_double();
            bool neg = std::signbit(d);
            double ad = std::fabs(d);
            char buf[512];
            if (t == 0) {
                body = sp.prec >= 0 ? [&] { snprintf(buf, sizeof buf, "%.*g", sp.prec ? sp.prec : 1, ad); return std::string(buf); }() : py_float_repr(ad);
            } else if (t == '%') { snprintf(buf, sizeof buf, "%.*f%%", sp.prec < 0 ? 6 : sp.prec, ad * 100); body = buf; }
            else { char f[8] = {'%', '.', '*', t, 0}; snprintf(buf, sizeof buf, f, sp.prec < 0 ? 6 : sp.prec, ad); body = buf; }
            if (sp.comma) { size_t dot = body.find_first_of(".eE"); std::string ip = body.substr(0, dot); body = group_thousands(ip) + (dot == std::string::npos ? "" : body.substr(dot)); }
            sign = neg ? "-" : sp.sign == '+' ? "+" : sp.sign == ' ' ? " " : "";
        } else {
            int64_t n = v.as_int();
            bool neg = n < 0;
            uint64_t un = neg ? (uint64_t)(-(n + 1)) + 1 : (uint64_t)n;
            if (t == 'x' || t == 'X' || t == 'o' || t == 'b') {
                int base = t == 'o' ? 8 : t == 'b' ? 2 : 16;
                if (un == 0) body = "0";
                while (un) { body.insert(body.begin(), "0123456789abcdef"[un % base]); un /= base; }
                if (t == 'X') for (auto& c : body) c = (char)toupper(c);
                if (sp.alt) body = std::string(t == 'o' ? "0o" : t == 'b' ? "0b" : t == 'X' ? "0X" : "0x") + body;
            } else if (t == 'c') { body.clear(); u8_append(body, (uint32_t)n); }
            else { body = std::to_string(un); if (sp.comma) body = group_thousands(body); }
            sign = neg ? "-" : sp.sign == '+' ? "+" : sp.sign == ' ' ? " " : "";
        }
    }
    std::string full = sign + body;
    size_t len = u8_length(full);
    if (sp.width > 0 && (int)len < sp.width) {
        size_t pad = sp.width - len;
        char al = sp.align;
        if (!al) al = numeric ? (sp.zero ? '=' : '>') : '<';
        std::string fill = sp.zero && !sp.align ? "0" : sp.fill;
        auto rep = [&](size_t n) { std::string r; for (size_t i = 0; i < n; ++i) r += fill; return r; };
        if (al == '<') full = full + rep(pad);
        else if (al == '>') full = rep(pad) + full;
        else if (al == '^') full = rep(pad / 2) + full + rep(pad - pad / 2);
        else full = sign + rep(pad) + body;
    }
    return full;
}

Value py_attr_chain(EnvImpl& env, Value v, const std::string& path) {
    size_t i = 0;
    while (i <= path.size()) {
        size_t d = path.find('.', i);
        std::string part = path.substr(i, d == std::string::npos ? std::string::npos : d - i);
        bool digits = !part.empty() && std::all_of(part.begin(), part.end(), [](char c) { return isdigit((unsigned char)c); });
        v = digits ? get_item(env, v, Value((long long)atoll(part.c_str()))) : get_item(env, v, Value(part));
        if (d == std::string::npos) break;
        i = d + 1;
    }
    return v;
}

std::string str_format(EnvImpl& env, const std::string& fmt, const Args& a) {
    std::string out;
    size_t auto_idx = 0;
    for (size_t i = 0; i < fmt.size(); ++i) {
        char c = fmt[i];
        if (c == '{') {
            if (i + 1 < fmt.size() && fmt[i + 1] == '{') { out += '{'; ++i; continue; }
            size_t e = fmt.find('}', i);
            if (e == std::string::npos) err("Single '{' encountered in format string");
            std::string field = fmt.substr(i + 1, e - i - 1);
            i = e;
            std::string spec, conv;
            size_t col = field.find(':');
            if (col != std::string::npos) { spec = field.substr(col + 1); field.resize(col); }
            size_t bang = field.find('!');
            if (bang != std::string::npos) { conv = field.substr(bang + 1); field.resize(bang); }
            std::string head = field, rest;
            size_t dot = field.find_first_of(".[");
            if (dot != std::string::npos) { head = field.substr(0, dot); rest = field.substr(dot); }
            Value v;
            if (head.empty()) { if (auto_idx >= a.pos.size()) err("Replacement index out of range"); v = a.pos[auto_idx++]; }
            else if (isdigit((unsigned char)head[0])) { size_t k = (size_t)atoi(head.c_str()); if (k >= a.pos.size()) err("Replacement index out of range"); v = a.pos[k]; }
            else { const Value* kv = a.kwarg(head.c_str()); if (!kv) err("KeyError: '" + head + "'"); v = *kv; }
            while (!rest.empty()) {
                if (rest[0] == '.') { size_t n = rest.find_first_of(".[", 1); std::string nm = rest.substr(1, n == std::string::npos ? n : n - 1); v = get_attr(env, v, nm); rest = n == std::string::npos ? "" : rest.substr(n); }
                else { size_t n = rest.find(']'); std::string k = rest.substr(1, n - 1); v = isdigit((unsigned char)k[0]) ? get_item(env, v, Value((long long)atoll(k.c_str()))) : get_item(env, v, Value(k)); rest = rest.substr(n + 1); }
            }
            if (conv == "r") v = S(v.repr());
            else if (conv == "s") v = S(v.str());
            out += format_value(v, spec);
        } else if (c == '}') {
            if (i + 1 < fmt.size() && fmt[i + 1] == '}') { out += '}'; ++i; continue; }
            err("Single '}' encountered in format string");
        } else out += c;
    }
    return out;
}

std::string percent_format(const std::string& fmt, const Value& arg) {
    std::vector<Value> args;
    bool is_map = arg.is_dict();
    if (arg.is_tuple()) args = arg.items(); else args.push_back(arg);
    size_t ai = 0;
    std::string out;
    for (size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') { out += fmt[i]; continue; }
        ++i;
        if (i >= fmt.size()) err("incomplete format");
        if (fmt[i] == '%') { out += '%'; continue; }
        Value val;
        bool have = false;
        if (fmt[i] == '(') {
            size_t e = fmt.find(')', i);
            if (e == std::string::npos || !is_map) err("format requires a mapping");
            const Value* f = arg.find(fmt.substr(i + 1, e - i - 1));
            if (!f) err("KeyError in format");
            val = *f; have = true; i = e + 1;
        }
        std::string flags;
        while (i < fmt.size() && strchr("-+ 0#", fmt[i])) flags += fmt[i++];
        std::string width, prec;
        while (i < fmt.size() && isdigit((unsigned char)fmt[i])) width += fmt[i++];
        if (i < fmt.size() && fmt[i] == '.') { ++i; prec = "."; while (i < fmt.size() && isdigit((unsigned char)fmt[i])) prec += fmt[i++]; }
        if (i >= fmt.size()) err("incomplete format");
        char t = fmt[i];
        if (!have) { if (ai >= args.size()) err("not enough arguments for format string"); val = args[ai++]; }
        std::string piece;
        char buf[512];
        if (t == 's' || t == 'r' || t == 'a') {
            piece = t == 's' ? val.str() : val.repr();
            if (prec.size() > 1) piece = u8_substr(piece, 0, atoi(prec.c_str() + 1));
        } else if (t == 'd' || t == 'i' || t == 'u') {
            snprintf(buf, sizeof buf, ("%" + flags + width + "lld").c_str(), (long long)val.as_int());
            out += buf; continue;
        } else if (t == 'x' || t == 'X' || t == 'o') {
            snprintf(buf, sizeof buf, ("%" + flags + width + "ll" + t).c_str(), (long long)val.as_int());
            out += buf; continue;
        } else if (strchr("fFeEgG", t)) {
            snprintf(buf, sizeof buf, ("%" + flags + width + prec + t).c_str(), val.as_double());
            out += buf; continue;
        } else err(std::string("unsupported format character '") + t + "'");
        size_t w = width.empty() ? 0 : (size_t)atoi(width.c_str());
        size_t len = u8_length(piece);
        if (len < w) piece = flags.find('-') != std::string::npos ? piece + std::string(w - len, ' ') : std::string(w - len, ' ') + piece;
        out += piece;
    }
    if (!is_map && ai < args.size() && !arg.is_dict()) err("not all arguments converted during string formatting");
    return out;
}

// ------------------------------------------------------------------------ sequence helpers
double to_double(const Value& v) {
    if (v.is_number() || v.is_bool()) return v.as_double();
    err(std::string("must be real number, not ") + v.type_name());
}

bool is_number_like(const Value& v) { return v.is_int() || v.is_float() || v.is_bool(); }

struct KeyedSort {
    bool case_sensitive;
    Value key_of(EnvImpl& env, const Value& v, const std::string& attr) const {
        Value k = attr.empty() ? v : py_attr_chain(env, v, attr);
        if (!case_sensitive && k.is_string()) return S(map_case(k.as_string(), false));
        return k;
    }
};
std::vector<Value> sorted_values(EnvImpl& env, const Value& it, bool reverse, bool cs, const std::string& attr) {
    std::vector<Value> src = iterate(it);
    KeyedSort ks{cs};
    std::vector<std::pair<Value, Value>> v;
    for (auto& x : src) v.emplace_back(ks.key_of(env, x, attr), x);
    std::stable_sort(v.begin(), v.end(), [&](auto& a, auto& b) { return reverse ? Value::compare(b.first, a.first) < 0 : Value::compare(a.first, b.first) < 0; });
    std::vector<Value> out;
    for (auto& p : v) out.push_back(p.second);
    return out;
}

Value do_tojson(const Value& v, const Args& a) {
    JsonOpts o;
    const Value* ea = a.get(0, "ensure_ascii");
    const Value* ind = a.get(1, "indent");
    const Value* sep = a.get(2, "separators");
    const Value* sk = a.get(3, "sort_keys");
    if (ea) o.ensure_ascii = ea->truthy();
    if (ind && !ind->is_none()) { o.indent = (int)ind->as_int(); o.item_sep = ","; }
    if (sep && !sep->is_none()) {
        if (!sep->is_seq() || sep->items().size() != 2) err("separators must be a 2-tuple");
        o.item_sep = sep->items()[0].str();
        o.key_sep = sep->items()[1].str();
    }
    if (sk) o.sort_keys = sk->truthy();
    return S(to_json_string(v, o));
}

std::string html_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '&': o += "&amp;"; break;
            case '<': o += "&lt;"; break;
            case '>': o += "&gt;"; break;
            case '"': o += "&#34;"; break;
            case '\'': o += "&#39;"; break;
            default: o += c;
        }
    }
    return o;
}

Value make_tuple2(Value a, Value b) { return Value::tuple({std::move(a), std::move(b)}); }

Value do_int(const Value& v, const Args& a) {
    Value def = a.get(0, "default") ? *a.get(0, "default") : Value(0);
    int base = (int)opt_int(a, 1, "base", 10);
    if (v.is_int() || v.is_bool()) return Value((long long)v.as_int());
    if (v.is_float()) {
        double d = v.as_double();
        if (std::isnan(d) || std::isinf(d)) return def;
        return Value((long long)d);
    }
    if (v.is_string()) {
        std::string s = py_strip(v.as_string(), nullptr, true, true);
        std::string t;
        for (char c : s) if (c != '_') t += c;
        char* e = nullptr;
        if (!t.empty()) {
            long long r = strtoll(t.c_str(), &e, base);
            if (*e == 0 && e != t.c_str()) return Value(r);
            bool plain = t.find_first_not_of("0123456789+-.eE") == std::string::npos;
            double d = plain ? strtod(t.c_str(), &e) : 0;
            if (plain && *e == 0 && e != t.c_str() && base == 10 && !std::isnan(d) && !std::isinf(d)) return Value((long long)d);
        }
    }
    return def;
}
Value do_float(const Value& v, const Args& a) {
    Value def = a.get(0, "default") ? *a.get(0, "default") : Value(0.0);
    if (v.is_number() || v.is_bool()) return Value(v.as_double());
    if (v.is_string()) {
        std::string s = py_strip(v.as_string(), nullptr, true, true);
        char* e = nullptr;
        if (!s.empty()) {
            double d = strtod(s.c_str(), &e);
            if (*e == 0 && e != s.c_str()) return Value(d);
        }
    }
    return def;
}

Value slice_impl(const Value& obj, const Value* start, const Value* stop, const Value* step) {
    int64_t st = 1;
    if (step && !step->is_none()) { st = step->as_int(); if (st == 0) err("slice step cannot be zero"); }
    bool is_str = obj.is_string();
    std::vector<uint32_t> cps;
    int64_t n;
    if (is_str) { cps = u8_decode(obj.as_string()); n = (int64_t)cps.size(); } else n = (int64_t)obj.items().size();
    auto norm = [&](const Value* v, int64_t def, int64_t lo, int64_t hi) {
        if (!v || v->is_none()) return def;
        int64_t x = v->as_int();
        if (x < 0) x += n;
        return std::max(lo, std::min(hi, x));
    };
    int64_t b, e;
    if (st > 0) { b = norm(start, 0, 0, n); e = norm(stop, n, 0, n); }
    else { b = norm(start, n - 1, -1, n - 1); e = norm(stop, -1, -1, n - 1); }
    if (is_str) {
        std::string o;
        if (st > 0) for (int64_t i = b; i < e; i += st) u8_append(o, cps[i]);
        else for (int64_t i = b; i > e; i += st) u8_append(o, cps[i]);
        return S(o);
    }
    std::vector<Value> o;
    if (st > 0) for (int64_t i = b; i < e; i += st) o.push_back(obj.items()[i]);
    else for (int64_t i = b; i > e; i += st) o.push_back(obj.items()[i]);
    return obj.is_tuple() ? Value::tuple(o) : Lst(o);
}

}  // namespace

// ============================================================================ public helpers
Value slice_value(const Value& obj, const Value* a, const Value* b, const Value* c) {
    if (!obj.is_string() && !obj.is_seq()) err(std::string("'") + obj.type_name() + "' object is not subscriptable");
    return slice_impl(obj, a, b, c);
}

std::vector<Value> iterate(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Undefined: return {};
        case Value::Kind::List: case Value::Kind::Tuple: return v.items();
        case Value::Kind::Dict: {
            std::vector<Value> k;
            for (auto& p : v.dict_data().kv) k.push_back(p.first);
            return k;
        }
        case Value::Kind::Str: {
            std::vector<Value> k;
            for (uint32_t cp : u8_decode(v.as_string())) { std::string c; u8_append(c, cp); k.push_back(S(c)); }
            return k;
        }
        default: err(std::string("'") + v.type_name() + "' object is not iterable");
    }
}

bool contains(const Value& c, const Value& item) {
    switch (c.kind()) {
        case Value::Kind::Undefined: return false;
        case Value::Kind::Str:
            if (!item.is_string()) err("'in <string>' requires string as left operand, not " + std::string(item.type_name()));
            return c.as_string().find(item.as_string()) != std::string::npos;
        case Value::Kind::List: case Value::Kind::Tuple:
            for (auto& x : c.items()) if (Value::equals(x, item)) return true;
            return false;
        case Value::Kind::Dict: return c.dict_data().find_idx(item) >= 0;
        default: err(std::string("argument of type '") + c.type_name() + "' is not iterable");
    }
}

Value binop(Op op, const Value& a, const Value& b) {
    auto bad = [&](const char* sym) -> Value {
        if (a.is_undefined() || b.is_undefined()) err("'undefined' value used in operation");
        err(std::string("unsupported operand type(s) for ") + sym + ": '" + a.type_name() + "' and '" + b.type_name() + "'");
    };
    switch (op) {
        case Op::Concat: return S(a.str() + b.str());
        case Op::Add:
            if (is_number_like(a) && is_number_like(b)) {
                if (a.is_float() || b.is_float()) return Value(a.as_double() + b.as_double());
                return Value((long long)(a.as_int() + b.as_int()));
            }
            if (a.is_string() && b.is_string()) return S(a.as_string() + b.as_string());
            if (a.is_list() && b.is_list()) { auto v = a.items(); v.insert(v.end(), b.items().begin(), b.items().end()); return Lst(v); }
            if (a.is_tuple() && b.is_tuple()) { auto v = a.items(); v.insert(v.end(), b.items().begin(), b.items().end()); return Value::tuple(v); }
            if (a.is_string() || b.is_string()) {
                if (a.is_undefined() || b.is_undefined()) return bad("+");
                err(a.is_string() ? std::string("can only concatenate str (not \"") + b.type_name() + "\") to str"
                                  : std::string("unsupported operand type(s) for +: '") + a.type_name() + "' and 'str'");
            }
            return bad("+");
        case Op::Sub:
            if (is_number_like(a) && is_number_like(b)) {
                if (a.is_float() || b.is_float()) return Value(a.as_double() - b.as_double());
                return Value((long long)(a.as_int() - b.as_int()));
            }
            return bad("-");
        case Op::Mul: {
            if (is_number_like(a) && is_number_like(b)) {
                if (a.is_float() || b.is_float()) return Value(a.as_double() * b.as_double());
                return Value((long long)(a.as_int() * b.as_int()));
            }
            const Value* seq = (a.is_string() || a.is_seq()) ? &a : (b.is_string() || b.is_seq()) ? &b : nullptr;
            const Value* n = seq == &a ? &b : &a;
            if (seq && (n->is_int() || n->is_bool())) {
                int64_t k = std::max<int64_t>(0, n->as_int());
                if (seq->is_string()) { std::string o; for (int64_t i = 0; i < k; ++i) o += seq->as_string(); return S(o); }
                std::vector<Value> o;
                for (int64_t i = 0; i < k; ++i) o.insert(o.end(), seq->items().begin(), seq->items().end());
                return seq->is_tuple() ? Value::tuple(o) : Lst(o);
            }
            return bad("*");
        }
        case Op::Div:
            if (is_number_like(a) && is_number_like(b)) {
                double d = b.as_double();
                if (d == 0) err("division by zero");
                return Value(a.as_double() / d);
            }
            return bad("/");
        case Op::FloorDiv:
            if (is_number_like(a) && is_number_like(b)) {
                if (a.is_float() || b.is_float()) {
                    double d = b.as_double();
                    if (d == 0) err("float floor division by zero");
                    return Value(std::floor(a.as_double() / d));
                }
                int64_t x = a.as_int(), y = b.as_int();
                if (y == 0) err("integer division or modulo by zero");
                int64_t q = x / y;
                if ((x % y != 0) && ((x < 0) != (y < 0))) --q;
                return Value((long long)q);
            }
            return bad("//");
        case Op::Mod:
            if (a.is_string()) return S(percent_format(a.as_string(), b));
            if (is_number_like(a) && is_number_like(b)) {
                if (a.is_float() || b.is_float()) {
                    double y = b.as_double();
                    if (y == 0) err("float modulo");
                    double r = std::fmod(a.as_double(), y);
                    if (r != 0 && ((r < 0) != (y < 0))) r += y;
                    return Value(r);
                }
                int64_t x = a.as_int(), y = b.as_int();
                if (y == 0) err("integer division or modulo by zero");
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) r += y;
                return Value((long long)r);
            }
            return bad("%");
        case Op::Pow:
            if (is_number_like(a) && is_number_like(b)) {
                if (!a.is_float() && !b.is_float() && b.as_int() >= 0) {
                    int64_t r = 1, base = a.as_int();
                    for (int64_t i = 0; i < b.as_int(); ++i) r *= base;
                    return Value((long long)r);
                }
                return Value(std::pow(a.as_double(), b.as_double()));
            }
            return bad("**");
        case Op::Eq: return Value(Value::equals(a, b));
        case Op::Ne: return Value(!Value::equals(a, b));
        case Op::Lt: return Value(Value::compare(a, b) < 0);
        case Op::Le: return Value(Value::compare(a, b) <= 0);
        case Op::Gt: return Value(Value::compare(a, b) > 0);
        case Op::Ge: return Value(Value::compare(a, b) >= 0);
        case Op::In: return Value(contains(b, a));
        case Op::NotIn: return Value(!contains(b, a));
    }
    return Value();
}

Value apply_filter(EnvImpl& env, const std::string& name, const Value& v, Args& a) {
    auto it = env.filters.find(name);
    if (it == env.filters.end()) err("No filter named '" + name + "'.");
    return it->second(env, v, a);
}
bool apply_test(EnvImpl& env, const std::string& name, const Value& v, Args& a) {
    auto it = env.tests.find(name);
    if (it == env.tests.end()) err("No test named '" + name + "'.");
    return it->second(env, v, a);
}

Value get_item(EnvImpl& env, const Value& obj, const Value& key) {
    switch (obj.kind()) {
        case Value::Kind::Undefined: err("'undefined' value has no items");
        case Value::Kind::Str:
            if (key.is_int() || key.is_bool()) {
                auto cps = u8_decode(obj.as_string());
                int64_t i = key.as_int(), n = (int64_t)cps.size();
                if (i < 0) i += n;
                if (i < 0 || i >= n) return Value();
                std::string c;
                u8_append(c, cps[i]);
                return S(c);
            }
            break;
        case Value::Kind::List: case Value::Kind::Tuple:
            if (key.is_int() || key.is_bool()) {
                int64_t i = key.as_int(), n = (int64_t)obj.items().size();
                if (i < 0) i += n;
                if (i < 0 || i >= n) return Value();
                return obj.items()[i];
            }
            break;
        case Value::Kind::Dict: {
            int i = obj.dict_data().find_idx(key);
            if (i >= 0) return obj.dict_data().kv[i].second;
            break;
        }
        default: break;
    }
    if (key.is_string()) return get_attr(env, obj, key.as_string());
    return Value();
}

// ============================================================================ methods
namespace {
using Method = std::function<Value(EnvImpl&, Value& self, Args& a)>;

const std::map<std::string, Method>& str_methods() {
    static const std::map<std::string, Method> m = {
        {"upper", [](EnvImpl&, Value& s, Args&) { return S(map_case(s.as_string(), true)); }},
        {"lower", [](EnvImpl&, Value& s, Args&) { return S(map_case(s.as_string(), false)); }},
        {"casefold", [](EnvImpl&, Value& s, Args&) { return S(map_case(s.as_string(), false)); }},
        {"strip", [](EnvImpl&, Value& s, Args& a) { return S(py_strip(s.as_string(), a.get(0, "chars"), true, true)); }},
        {"lstrip", [](EnvImpl&, Value& s, Args& a) { return S(py_strip(s.as_string(), a.get(0, "chars"), true, false)); }},
        {"rstrip", [](EnvImpl&, Value& s, Args& a) { return S(py_strip(s.as_string(), a.get(0, "chars"), false, true)); }},
        {"capitalize", [](EnvImpl&, Value& s, Args&) {
             auto cps = u8_decode(s.as_string());
             std::string o;
             for (size_t i = 0; i < cps.size(); ++i) u8_append(o, i == 0 ? upper_cp(cps[i]) : lower_cp(cps[i]));
             return S(o);
         }},
        {"title", [](EnvImpl&, Value& s, Args&) {
             std::string o;
             bool prev_alpha = false;
             for (uint32_t cp : u8_decode(s.as_string())) {
                 bool al = is_alpha_cp(cp);
                 u8_append(o, al ? (prev_alpha ? lower_cp(cp) : upper_cp(cp)) : cp);
                 prev_alpha = al;
             }
             return S(o);
         }},
        {"swapcase", [](EnvImpl&, Value& s, Args&) {
             std::string o;
             for (uint32_t cp : u8_decode(s.as_string())) u8_append(o, lower_cp(cp) != cp ? lower_cp(cp) : upper_cp(cp));
             return S(o);
         }},
        {"split", [](EnvImpl&, Value& s, Args& a) {
             const Value* sep = a.get(0, "sep");
             int64_t mx = opt_int(a, 1, "maxsplit", -1);
             if (!sep || sep->is_none()) {
                 std::string t = s.as_string();
                 return Lst(split_ws(t, mx));
             }
             return Lst(split_sep(s.as_string(), need_str(*sep, "sep").as_string(), mx));
         }},
        {"rsplit", [](EnvImpl&, Value& s, Args& a) {
             const Value* sep = a.get(0, "sep");
             int64_t mx = opt_int(a, 1, "maxsplit", -1);
             if (!sep || sep->is_none()) {
                 auto parts = split_ws(s.as_string(), -1);
                 if (mx >= 0 && (int64_t)parts.size() > mx + 1) {
                     // re-join the leading parts by splitting from the right on whitespace runs
                     auto cps = u8_decode(s.as_string());
                     std::vector<Value> out;
                     int64_t i = (int64_t)cps.size();
                     while ((int64_t)out.size() < mx) {
                         while (i > 0 && is_py_space(cps[i - 1])) --i;
                         if (i == 0) break;
                         int64_t e = i;
                         while (i > 0 && !is_py_space(cps[i - 1])) --i;
                         std::string w;
                         for (int64_t k = i; k < e; ++k) u8_append(w, cps[k]);
                         out.push_back(S(w));
                     }
                     std::string head;
                     for (int64_t k = 0; k < i; ++k) u8_append(head, cps[k]);
                     head = py_strip(head, nullptr, true, true);
                     if (!head.empty()) out.push_back(S(head));
                     std::reverse(out.begin(), out.end());
                     return Lst(out);
                 }
                 return Lst(parts);
             }
             return Lst(rsplit_sep(s.as_string(), need_str(*sep, "sep").as_string(), mx));
         }},
        {"splitlines", [](EnvImpl&, Value& s, Args& a) { return Lst(splitlines(s.as_string(), opt_bool(a, 0, "keepends", false))); }},
        {"startswith", [](EnvImpl&, Value& s, Args& a) {
             const Value* p = a.get(0, "prefix");
             if (!p) err("startswith() takes at least 1 argument (0 given)");
             std::string str = s.as_string();
             if (a.pos.size() > 1) {
                 int64_t st = opt_int(a, 1, "start", 0), en = opt_int(a, 2, "end", (int64_t)u8_length(str));
                 int64_t n = (int64_t)u8_length(str);
                 if (st < 0) st = std::max<int64_t>(0, st + n);
                 if (en < 0) en = std::max<int64_t>(0, en + n);
                 en = std::min(en, n);
                 str = st >= en ? "" : u8_substr(str, st, en - st);
             }
             auto test = [&](const Value& x) {
                 if (!x.is_string()) err("startswith first arg must be str or a tuple of str, not " + std::string(x.type_name()));
                 return str.compare(0, x.as_string().size(), x.as_string()) == 0 && str.size() >= x.as_string().size();
             };
             if (p->is_tuple()) { for (auto& x : p->items()) if (test(x)) return Value(true); return Value(false); }
             return Value(test(*p));
         }},
        {"endswith", [](EnvImpl&, Value& s, Args& a) {
             const Value* p = a.get(0, "suffix");
             if (!p) err("endswith() takes at least 1 argument (0 given)");
             std::string str = s.as_string();
             if (a.pos.size() > 1) {
                 int64_t st = opt_int(a, 1, "start", 0), en = opt_int(a, 2, "end", (int64_t)u8_length(str));
                 int64_t n = (int64_t)u8_length(str);
                 if (st < 0) st = std::max<int64_t>(0, st + n);
                 if (en < 0) en = std::max<int64_t>(0, en + n);
                 en = std::min(en, n);
                 str = st >= en ? "" : u8_substr(str, st, en - st);
             }
             auto test = [&](const Value& x) {
                 if (!x.is_string()) err("endswith first arg must be str or a tuple of str, not " + std::string(x.type_name()));
                 const std::string& suf = x.as_string();
                 return str.size() >= suf.size() && str.compare(str.size() - suf.size(), suf.size(), suf) == 0;
             };
             if (p->is_tuple()) { for (auto& x : p->items()) if (test(x)) return Value(true); return Value(false); }
             return Value(test(*p));
         }},
        {"replace", [](EnvImpl&, Value& s, Args& a) {
             if (a.pos.size() < 2) err("replace() takes at least 2 arguments");
             return S(replace_all(s.as_string(), need_str(a.pos[0], "replace() argument 1").as_string(), need_str(a.pos[1], "replace() argument 2").as_string(),
                                  opt_int(a, 2, "count", -1)));
         }},
        {"find", [](EnvImpl&, Value& s, Args& a) {
             int64_t n = (int64_t)u8_length(s.as_string());
             return Value((long long)cp_find(s.as_string(), need_str(a.pos.at(0), "must").as_string(), opt_int(a, 1, "start", 0), opt_int(a, 2, "end", n), false));
         }},
        {"rfind", [](EnvImpl&, Value& s, Args& a) {
             int64_t n = (int64_t)u8_length(s.as_string());
             return Value((long long)cp_find(s.as_string(), need_str(a.pos.at(0), "must").as_string(), opt_int(a, 1, "start", 0), opt_int(a, 2, "end", n), true));
         }},
        {"index", [](EnvImpl&, Value& s, Args& a) {
             int64_t n = (int64_t)u8_length(s.as_string());
             int64_t r = cp_find(s.as_string(), need_str(a.pos.at(0), "must").as_string(), opt_int(a, 1, "start", 0), opt_int(a, 2, "end", n), false);
             if (r < 0) err("substring not found");
             return Value((long long)r);
         }},
        {"count", [](EnvImpl&, Value& s, Args& a) {
             const std::string& sub = need_str(a.pos.at(0), "must").as_string();
             const std::string& str = s.as_string();
             if (sub.empty()) return Value((long long)u8_length(str) + 1);
             int64_t c = 0;
             for (size_t p = str.find(sub); p != std::string::npos; p = str.find(sub, p + sub.size())) ++c;
             return Value((long long)c);
         }},
        {"join", [](EnvImpl&, Value& s, Args& a) {
             std::string o;
             bool first = true;
             for (auto& x : iterate(a.pos.at(0))) {
                 if (!x.is_string()) err("sequence item: expected str instance, " + std::string(x.type_name()) + " found");
                 if (!first) o += s.as_string();
                 first = false;
                 o += x.as_string();
             }
             return S(o);
         }},
        {"format", [](EnvImpl& env, Value& s, Args& a) { return S(str_format(env, s.as_string(), a)); }},
        {"zfill", [](EnvImpl&, Value& s, Args& a) {
             int64_t w = opt_int(a, 0, "width", 0);
             std::string t = s.as_string();
             int64_t len = (int64_t)u8_length(t);
             if (len >= w) return S(t);
             std::string sign;
             if (!t.empty() && (t[0] == '+' || t[0] == '-')) { sign = t.substr(0, 1); t = t.substr(1); }
             return S(sign + std::string(w - len, '0') + t);
         }},
        {"center", [](EnvImpl&, Value& s, Args& a) {
             int64_t w = opt_int(a, 0, "width", 0);
             std::string fill = a.pos.size() > 1 ? a.pos[1].as_string() : " ";
             int64_t len = (int64_t)u8_length(s.as_string());
             if (len >= w) return S(s.as_string());
             int64_t pad = w - len, left = pad / 2 + (pad & w & 1);
             std::string o;
             for (int64_t i = 0; i < left; ++i) o += fill;
             o += s.as_string();
             for (int64_t i = 0; i < pad - left; ++i) o += fill;
             return S(o);
         }},
        {"ljust", [](EnvImpl&, Value& s, Args& a) {
             int64_t w = opt_int(a, 0, "width", 0);
             std::string fill = a.pos.size() > 1 ? a.pos[1].as_string() : " ";
             int64_t len = (int64_t)u8_length(s.as_string());
             std::string o = s.as_string();
             for (int64_t i = len; i < w; ++i) o += fill;
             return S(o);
         }},
        {"rjust", [](EnvImpl&, Value& s, Args& a) {
             int64_t w = opt_int(a, 0, "width", 0);
             std::string fill = a.pos.size() > 1 ? a.pos[1].as_string() : " ";
             int64_t len = (int64_t)u8_length(s.as_string());
             std::string o;
             for (int64_t i = len; i < w; ++i) o += fill;
             return S(o + s.as_string());
         }},
        {"isdigit", [](EnvImpl&, Value& s, Args&) {
             auto c = u8_decode(s.as_string());
             return Value(!c.empty() && std::all_of(c.begin(), c.end(), [](uint32_t x) { return x >= '0' && x <= '9'; }));
         }},
        {"isnumeric", [](EnvImpl&, Value& s, Args&) {
             auto c = u8_decode(s.as_string());
             return Value(!c.empty() && std::all_of(c.begin(), c.end(), [](uint32_t x) { return x >= '0' && x <= '9'; }));
         }},
        {"isdecimal", [](EnvImpl&, Value& s, Args&) {
             auto c = u8_decode(s.as_string());
             return Value(!c.empty() && std::all_of(c.begin(), c.end(), [](uint32_t x) { return x >= '0' && x <= '9'; }));
         }},
        {"isalpha", [](EnvImpl&, Value& s, Args&) {
             auto c = u8_decode(s.as_string());
             return Value(!c.empty() && std::all_of(c.begin(), c.end(), [](uint32_t x) { return is_alpha_cp(x); }));
         }},
        {"isalnum", [](EnvImpl&, Value& s, Args&) {
             auto c = u8_decode(s.as_string());
             return Value(!c.empty() && std::all_of(c.begin(), c.end(), [](uint32_t x) { return is_alpha_cp(x) || (x >= '0' && x <= '9'); }));
         }},
        {"isspace", [](EnvImpl&, Value& s, Args&) {
             auto c = u8_decode(s.as_string());
             return Value(!c.empty() && std::all_of(c.begin(), c.end(), [](uint32_t x) { return is_py_space(x); }));
         }},
        {"isupper", [](EnvImpl&, Value& s, Args&) {
             bool cased = false;
             for (uint32_t c : u8_decode(s.as_string())) { if (lower_cp(c) != c) cased = true; else if (upper_cp(c) != c) return Value(false); }
             return Value(cased);
         }},
        {"islower", [](EnvImpl&, Value& s, Args&) {
             bool cased = false;
             for (uint32_t c : u8_decode(s.as_string())) { if (upper_cp(c) != c) cased = true; else if (lower_cp(c) != c) return Value(false); }
             return Value(cased);
         }},
        {"partition", [](EnvImpl&, Value& s, Args& a) {
             const std::string& sep = need_str(a.pos.at(0), "must").as_string();
             size_t f = s.as_string().find(sep);
             if (f == std::string::npos) return Value::tuple({s, S(""), S("")});
             return Value::tuple({S(s.as_string().substr(0, f)), S(sep), S(s.as_string().substr(f + sep.size()))});
         }},
        {"rpartition", [](EnvImpl&, Value& s, Args& a) {
             const std::string& sep = need_str(a.pos.at(0), "must").as_string();
             size_t f = s.as_string().rfind(sep);
             if (f == std::string::npos) return Value::tuple({S(""), S(""), s});
             return Value::tuple({S(s.as_string().substr(0, f)), S(sep), S(s.as_string().substr(f + sep.size()))});
         }},
        {"removeprefix", [](EnvImpl&, Value& s, Args& a) {
             const std::string& p = a.pos.at(0).as_string();
             const std::string& t = s.as_string();
             return S(t.compare(0, p.size(), p) == 0 && t.size() >= p.size() ? t.substr(p.size()) : t);
         }},
        {"removesuffix", [](EnvImpl&, Value& s, Args& a) {
             const std::string& p = a.pos.at(0).as_string();
             const std::string& t = s.as_string();
             return S(!p.empty() && t.size() >= p.size() && t.compare(t.size() - p.size(), p.size(), p) == 0 ? t.substr(0, t.size() - p.size()) : t);
         }},
    };
    return m;
}

int64_t norm_index(int64_t i, size_t n) { return i < 0 ? i + (int64_t)n : i; }

const std::map<std::string, Method>& list_methods() {
    static const std::map<std::string, Method> m = {
        {"append", [](EnvImpl&, Value& s, Args& a) { if (s.is_tuple()) err("'tuple' object has no attribute 'append'"); s.items().push_back(a.pos.at(0)); return Value::none(); }},
        {"extend", [](EnvImpl&, Value& s, Args& a) { for (auto& x : iterate(a.pos.at(0))) s.items().push_back(x); return Value::none(); }},
        {"insert", [](EnvImpl&, Value& s, Args& a) {
             auto& v = s.items();
             int64_t i = a.pos.at(0).as_int();
             if (i < 0) i = std::max<int64_t>(0, i + (int64_t)v.size());
             i = std::min<int64_t>(i, (int64_t)v.size());
             v.insert(v.begin() + i, a.pos.at(1));
             return Value::none();
         }},
        {"pop", [](EnvImpl&, Value& s, Args& a) {
             auto& v = s.items();
             if (v.empty()) err("pop from empty list");
             int64_t i = norm_index(opt_int(a, 0, "index", -1), v.size());
             if (i < 0 || i >= (int64_t)v.size()) err("pop index out of range");
             Value r = v[i];
             v.erase(v.begin() + i);
             return r;
         }},
        {"remove", [](EnvImpl&, Value& s, Args& a) {
             auto& v = s.items();
             for (size_t i = 0; i < v.size(); ++i) if (Value::equals(v[i], a.pos.at(0))) { v.erase(v.begin() + i); return Value::none(); }
             err("list.remove(x): x not in list");
         }},
        {"index", [](EnvImpl&, Value& s, Args& a) {
             auto& v = s.items();
             for (size_t i = 0; i < v.size(); ++i) if (Value::equals(v[i], a.pos.at(0))) return Value((long long)i);
             err("value is not in list");
         }},
        {"count", [](EnvImpl&, Value& s, Args& a) {
             int64_t c = 0;
             for (auto& x : s.items()) c += Value::equals(x, a.pos.at(0));
             return Value((long long)c);
         }},
        {"reverse", [](EnvImpl&, Value& s, Args&) { std::reverse(s.items().begin(), s.items().end()); return Value::none(); }},
        {"copy", [](EnvImpl&, Value& s, Args&) { return Lst(s.items()); }},
        {"clear", [](EnvImpl&, Value& s, Args&) { s.items().clear(); return Value::none(); }},
        {"sort", [](EnvImpl& env, Value& s, Args& a) {
             s.items() = sorted_values(env, s, opt_bool(a, 0, "reverse", false), true, "");
             return Value::none();
         }},
    };
    return m;
}

const std::map<std::string, Method>& dict_methods() {
    static const std::map<std::string, Method> m = {
        {"items", [](EnvImpl&, Value& s, Args&) {
             std::vector<Value> o;
             for (auto& p : s.dict_data().kv) o.push_back(make_tuple2(p.first, p.second));
             return Lst(o);
         }},
        {"keys", [](EnvImpl&, Value& s, Args&) { return Lst(iterate(s)); }},
        {"values", [](EnvImpl&, Value& s, Args&) {
             std::vector<Value> o;
             for (auto& p : s.dict_data().kv) o.push_back(p.second);
             return Lst(o);
         }},
        {"get", [](EnvImpl&, Value& s, Args& a) {
             const Value& k = a.pos.at(0);
             int i = s.dict_data().find_idx(k);
             if (i >= 0) return s.dict_data().kv[i].second;
             return a.pos.size() > 1 ? a.pos[1] : Value::none();
         }},
        {"update", [](EnvImpl&, Value& s, Args& a) {
             if (!a.pos.empty()) {
                 const Value& o = a.pos[0];
                 if (o.is_dict()) for (auto& p : o.dict_data().kv) s.set(p.first, p.second);
                 else for (auto& x : iterate(o)) s.set(x.items().at(0), x.items().at(1));
             }
             for (auto& p : a.kw) s.set(p.first, p.second);
             return Value::none();
         }},
        {"pop", [](EnvImpl&, Value& s, Args& a) {
             auto& d = s.dict_data();
             int i = d.find_idx(a.pos.at(0));
             if (i < 0) { if (a.pos.size() > 1) return a.pos[1]; err("KeyError: " + a.pos[0].repr()); }
             Value r = d.kv[i].second;
             d.kv.erase(d.kv.begin() + i);
             return r;
         }},
        {"setdefault", [](EnvImpl&, Value& s, Args& a) {
             auto& d = s.dict_data();
             int i = d.find_idx(a.pos.at(0));
             if (i >= 0) return d.kv[i].second;
             Value dv = a.pos.size() > 1 ? a.pos[1] : Value::none();
             d.kv.emplace_back(a.pos[0], dv);
             return dv;
         }},
        {"copy", [](EnvImpl&, Value& s, Args&) {
             Value d = Value::dict();
             for (auto& p : s.dict_data().kv) d.set(p.first, p.second);
             return d;
         }},
        {"clear", [](EnvImpl&, Value& s, Args&) { s.dict_data().kv.clear(); return Value::none(); }},
    };
    return m;
}
}  // namespace

Value bound_method(EnvImpl& env, const Value& self, const std::string& name) {
    const std::map<std::string, Method>* tbl = nullptr;
    if (self.is_string()) tbl = &str_methods();
    else if (self.is_list()) tbl = &list_methods();
    else if (self.is_tuple()) {
        if (name != "index" && name != "count") return Value();
        tbl = &list_methods();
    } else if (self.is_dict() && !self.is_ns()) tbl = &dict_methods();
    if (!tbl) return Value();
    auto it = tbl->find(name);
    if (it == tbl->end()) return Value();
    Method m = it->second;
    Value s = self;
    EnvImpl* e = &env;
    return Value::func(name, [m, s, e](Args& a) mutable { return m(*e, s, a); });
}

Value call_value(EnvImpl& env, const Value& f, Args& a) {
    if (f.is_func()) return static_cast<const FuncData*>(f.obj())->fn(a);
    if (f.kind() == Value::Kind::Loop) return static_cast<LoopStateBase*>(const_cast<Obj*>(f.obj()))->call(env, a);
    if (f.is_undefined()) err("'undefined' is not callable");
    err(std::string("'") + f.type_name() + "' object is not callable");
}

Value get_attr(EnvImpl& env, const Value& obj, const std::string& name) {
    switch (obj.kind()) {
        case Value::Kind::Undefined: err("'" + name + "': object is undefined");
        case Value::Kind::Dict: {
            if (!obj.is_ns()) {
                Value m = bound_method(env, obj, name);
                if (!m.is_undefined()) return m;
            }
            const Value* v = obj.find(name);
            return v ? *v : Value();
        }
        case Value::Kind::Loop: return static_cast<LoopStateBase*>(const_cast<Obj*>(obj.obj()))->attr(env, name);
        case Value::Kind::Tuple:  // groupby() results behave like namedtuple(grouper, list)
            if (obj.items().size() == 2 && (name == "grouper" || name == "list")) return obj.items()[name == "list" ? 1 : 0];
            return bound_method(env, obj, name);
        case Value::Kind::Str: case Value::Kind::List: return bound_method(env, obj, name);
        default: return Value();
    }
}

// ============================================================================ filters & tests
namespace {

#define FILTER(NAME) env.filters[NAME] = [](EnvImpl & env, const Value& v, Args& a) -> Value
#define TEST(NAME) env.tests[NAME] = [](EnvImpl & env, const Value& v, Args& a) -> bool

std::vector<Value> do_select(EnvImpl& env, const Value& v, Args& a, bool want, bool attr) {
    std::vector<Value> out;
    if (!v.truthy()) return out;  // jinja: `if value:` guard (None/undefined/empty -> nothing)
    size_t first = attr ? 1 : 0;
    std::string attr_name = attr ? a.pos.at(0).str() : "";
    std::string test_name;
    bool have_test = a.pos.size() > first;
    if (have_test) test_name = a.pos[first].str();
    Args targs;
    for (size_t i = first + 1; i < a.pos.size(); ++i) targs.pos.push_back(a.pos[i]);
    for (auto& x : iterate(v)) {
        Value t = attr ? py_attr_chain(env, x, attr_name) : x;
        bool r = have_test ? apply_test(env, test_name, t, targs) : t.truthy();
        if (r == want) out.push_back(x);
    }
    return out;
}

}  // namespace

void register_builtins(EnvImpl& env_) {
    EnvImpl& env0 = env_;
    (void)env0;
    auto& env = env_;

    // ---------------------------------------------------------------- filters
    FILTER("abs") { (void)env; (void)a; if (v.is_float()) return Value(std::fabs(v.as_double())); if (v.is_int() || v.is_bool()) return Value((long long)std::llabs(v.as_int())); err("bad operand type for abs()"); };
    FILTER("attr") { return get_attr(env, v, a.pos.at(0).str()).is_undefined() ? Value() : get_attr(env, v, a.pos.at(0).str()); };
    FILTER("length") { (void)env; (void)a; return Value((long long)v.length()); };
    env.filters["count"] = env.filters["length"];
    FILTER("default") {
        (void)env;
        const Value* d = a.get(0, "default_value");
        const Value* b = a.get(1, "boolean");
        bool boolean = b && b->truthy();
        if (v.is_undefined() || (boolean && !v.truthy())) return d ? *d : S("");
        return v;
    };
    env.filters["d"] = env.filters["default"];
    FILTER("first") { (void)env; (void)a; auto it = iterate(v); if (it.empty()) return Value(); return it.front(); };
    FILTER("last") { (void)env; (void)a; if (v.is_dict()) { auto it = iterate(v); return it.empty() ? Value() : it.back(); } if (v.is_string() || v.is_seq()) { auto it = iterate(v); return it.empty() ? Value() : it.back(); } err("reversed() argument must be a sequence"); };
    FILTER("lower") { (void)env; (void)a; return S(map_case(v.str(), false)); };
    FILTER("upper") { (void)env; (void)a; return S(map_case(v.str(), true)); };
    FILTER("capitalize") { Args e2; Value s = S(v.str()); return bound_method(env, s, "capitalize").is_func() ? static_cast<const FuncData*>(bound_method(env, s, "capitalize").obj())->fn(e2) : s; (void)a; };
    FILTER("title") { Args e2; Value s = S(v.str()); (void)a; return static_cast<const FuncData*>(bound_method(env, s, "title").obj())->fn(e2); };
    FILTER("trim") { (void)env; return S(py_strip(v.str(), a.get(0, "chars"), true, true)); };
    FILTER("string") { (void)env; (void)a; return v.is_string() ? v : S(v.str()); };
    FILTER("safe") { (void)env; (void)a; return v.is_string() ? v : S(v.str()); };
    FILTER("escape") { (void)env; (void)a; return S(html_escape(v.str())); };
    env.filters["e"] = env.filters["escape"];
    env.filters["forceescape"] = env.filters["escape"];
    FILTER("tojson") { (void)env; return do_tojson(v, a); };
    FILTER("from_json") { (void)env; (void)a; return from_json(jd::parse(v.as_string())); };
    FILTER("int") { (void)env; return do_int(v, a); };
    FILTER("float") { (void)env; return do_float(v, a); };
    FILTER("list") { (void)env; (void)a; return Lst(iterate(v)); };
    FILTER("items") { (void)a; if (v.is_undefined() || v.is_none()) return Lst(); if (!v.is_dict()) err("items filter requires a mapping"); Value s = v; Args e2; return static_cast<const FuncData*>(bound_method(env, s, "items").obj())->fn(e2); };
    FILTER("join") {
        (void)env;
        std::string sep = a.get(0, "d") ? a.get(0, "d")->str() : "";
        const Value* at = a.get(1, "attribute");
        std::string o;
        bool first = true;
        for (auto& x : iterate(v)) {
            if (!first) o += sep;
            first = false;
            o += (at && !at->is_none() ? py_attr_chain(env, x, at->str()) : x).str();
        }
        return S(o);
    };
    FILTER("replace") {
        (void)env;
        const Value* cnt = a.get(2, "count");
        return S(replace_all(v.str(), a.pos.at(0).str(), a.pos.at(1).str(), cnt && !cnt->is_none() ? cnt->as_int() : -1));
    };
    FILTER("reverse") {
        (void)env; (void)a;
        if (v.is_string()) { auto c = u8_decode(v.as_string()); std::string o; for (size_t i = c.size(); i-- > 0;) u8_append(o, c[i]); return S(o); }
        auto it = iterate(v);
        std::reverse(it.begin(), it.end());
        return Lst(it);
    };
    FILTER("indent") {
        (void)env;
        const Value* w = a.get(0, "width");
        std::string ind = w ? (w->is_string() ? w->as_string() : std::string((size_t)w->as_int(), ' ')) : "    ";
        bool first = opt_bool(a, 1, "first", false), blank = opt_bool(a, 2, "blank", false);
        std::string s = v.str() + "\n";
        auto lines = splitlines(s, false);
        std::string rv;
        if (blank) {
            for (size_t i = 0; i < lines.size(); ++i) { if (i) rv += "\n" + ind; rv += lines[i].as_string(); }
        } else {
            rv = lines.empty() ? "" : lines[0].as_string();
            for (size_t i = 1; i < lines.size(); ++i) { rv += "\n"; if (!lines[i].as_string().empty()) rv += ind; rv += lines[i].as_string(); }
        }
        if (first) rv = ind + rv;
        return S(rv);
    };
    FILTER("sum") {
        Value acc = a.get(1, "start") ? *a.get(1, "start") : Value(0);
        const Value* at = a.get(0, "attribute");
        for (auto& x : iterate(v)) acc = binop(Op::Add, acc, at && !at->is_none() ? py_attr_chain(env, x, at->str()) : x);
        return acc;
    };
    FILTER("min") {
        bool cs = opt_bool(a, 0, "case_sensitive", false);
        const Value* at = a.get(1, "attribute");
        auto items = iterate(v);
        if (items.empty()) return Value();
        KeyedSort ks{cs};
        std::string an = at && !at->is_none() ? at->str() : "";
        size_t best = 0;
        Value bk = ks.key_of(env, items[0], an);
        for (size_t i = 1; i < items.size(); ++i) { Value k = ks.key_of(env, items[i], an); if (Value::compare(k, bk) < 0) { best = i; bk = k; } }
        return items[best];
    };
    FILTER("max") {
        bool cs = opt_bool(a, 0, "case_sensitive", false);
        const Value* at = a.get(1, "attribute");
        auto items = iterate(v);
        if (items.empty()) return Value();
        KeyedSort ks{cs};
        std::string an = at && !at->is_none() ? at->str() : "";
        size_t best = 0;
        Value bk = ks.key_of(env, items[0], an);
        for (size_t i = 1; i < items.size(); ++i) { Value k = ks.key_of(env, items[i], an); if (Value::compare(k, bk) > 0) { best = i; bk = k; } }
        return items[best];
    };
    FILTER("sort") {
        const Value* at = a.get(2, "attribute");
        return Lst(sorted_values(env, v, opt_bool(a, 0, "reverse", false), opt_bool(a, 1, "case_sensitive", false), at && !at->is_none() ? at->str() : ""));
    };
    FILTER("dictsort") {
        bool cs = opt_bool(a, 0, "case_sensitive", false);
        std::string by = a.get(1, "by") ? a.get(1, "by")->str() : "key";
        bool rev = opt_bool(a, 2, "reverse", false);
        if (by != "key" && by != "value") err("You can only sort by either \"key\" or \"value\"");
        if (!v.is_dict()) err("dictsort requires a mapping");
        std::vector<std::pair<Value, Value>> kv(v.dict_data().kv.begin(), v.dict_data().kv.end());
        size_t pos = by == "key" ? 0 : 1;
        auto key = [&](const std::pair<Value, Value>& p) { const Value& k = pos ? p.second : p.first; return (!cs && k.is_string()) ? S(map_case(k.as_string(), false)) : k; };
        (void)env;
        std::stable_sort(kv.begin(), kv.end(), [&](auto& x, auto& y) { return rev ? Value::compare(key(y), key(x)) < 0 : Value::compare(key(x), key(y)) < 0; });
        std::vector<Value> o;
        for (auto& p : kv) o.push_back(make_tuple2(p.first, p.second));
        return Lst(o);
    };
    FILTER("unique") {
        bool cs = opt_bool(a, 0, "case_sensitive", false);
        const Value* at = a.get(1, "attribute");
        std::string an = at && !at->is_none() ? at->str() : "";
        KeyedSort ks{cs};
        std::vector<Value> seen, out;
        for (auto& x : iterate(v)) {
            Value k = ks.key_of(env, x, an);
            bool dup = false;
            for (auto& s : seen) if (Value::equals(s, k)) { dup = true; break; }
            if (!dup) { seen.push_back(k); out.push_back(x); }
        }
        return Lst(out);
    };
    FILTER("map") {
        std::vector<Value> out;
        if (const Value* at = a.kwarg("attribute")) {
            const Value* def = a.kwarg("default");
            for (auto& x : iterate(v)) {
                Value r = py_attr_chain(env, x, at->str());
                if (r.is_undefined() && def) r = *def;
                out.push_back(r);
            }
            return Lst(out);
        }
        if (a.pos.empty()) err("map requires a filter argument");
        std::string fname = a.pos[0].str();
        for (auto& x : iterate(v)) {
            Args fa;
            fa.pos.assign(a.pos.begin() + 1, a.pos.end());
            fa.kw = a.kw;
            out.push_back(apply_filter(env, fname, x, fa));
        }
        return Lst(out);
    };
    FILTER("select") { return Lst(do_select(env, v, a, true, false)); };
    FILTER("reject") { return Lst(do_select(env, v, a, false, false)); };
    FILTER("selectattr") { return Lst(do_select(env, v, a, true, true)); };
    FILTER("rejectattr") { return Lst(do_select(env, v, a, false, true)); };
    FILTER("batch") {
        (void)env;
        int64_t n = a.pos.at(0).as_int();
        const Value* fill = a.get(1, "fill_with");
        std::vector<Value> out, cur;
        for (auto& x : iterate(v)) {
            cur.push_back(x);
            if ((int64_t)cur.size() == n) { out.push_back(Lst(cur)); cur.clear(); }
        }
        if (!cur.empty()) {
            if (fill) while ((int64_t)cur.size() < n) cur.push_back(*fill);
            out.push_back(Lst(cur));
        }
        return Lst(out);
    };
    FILTER("slice") {
        (void)env;
        int64_t n = a.pos.at(0).as_int();
        auto items = iterate(v);
        const Value* fill = a.get(1, "fill_with");
        int64_t len = (int64_t)items.size(), per = len / n, extra = len % n, off = 0;
        std::vector<Value> out;
        for (int64_t i = 0; i < n; ++i) {
            int64_t cnt = per + (i < extra ? 1 : 0);
            std::vector<Value> part(items.begin() + off, items.begin() + off + cnt);
            off += cnt;
            if (fill && i >= extra && extra) part.push_back(*fill);
            out.push_back(Lst(part));
        }
        return Lst(out);
    };
    FILTER("groupby") {
        std::string attr = a.pos.at(0).str();
        bool cs = opt_bool(a, 2, "case_sensitive", false);
        const Value* def = a.get(1, "default");
        KeyedSort ks{cs};
        auto items = iterate(v);
        std::vector<std::pair<Value, Value>> keyed;
        for (auto& x : items) {
            Value k = py_attr_chain(env, x, attr);
            if (k.is_undefined() && def) k = *def;
            keyed.emplace_back(k, x);
        }
        auto kk = [&](const Value& k) { return (!cs && k.is_string()) ? S(map_case(k.as_string(), false)) : k; };
        std::stable_sort(keyed.begin(), keyed.end(), [&](auto& x, auto& y) { return Value::compare(kk(x.first), kk(y.first)) < 0; });
        (void)ks;
        std::vector<Value> out;
        for (size_t i = 0; i < keyed.size();) {
            size_t j = i;
            std::vector<Value> grp;
            while (j < keyed.size() && Value::equals(kk(keyed[j].first), kk(keyed[i].first))) grp.push_back(keyed[j++].second);
            out.push_back(make_tuple2(keyed[i].first, Lst(grp)));
            i = j;
        }
        return Lst(out);
    };
    FILTER("round") {
        (void)env;
        int64_t prec = opt_int(a, 0, "precision", 0);
        std::string method = a.get(1, "method") ? a.get(1, "method")->str() : "common";
        double x = to_double(v), f = std::pow(10.0, (double)prec);
        if (method == "common") {
            if (v.is_int() && prec >= 0) return v;
            return Value(std::nearbyint(x * f) / f);
        }
        if (method == "ceil") return Value(std::ceil(x * f) / f);
        if (method == "floor") return Value(std::floor(x * f) / f);
        err("method must be common, ceil or floor");
    };
    FILTER("truncate") {
        (void)env;
        int64_t length = opt_int(a, 0, "length", 255);
        bool kill = opt_bool(a, 1, "killwords", false);
        std::string end = a.get(2, "end") ? a.get(2, "end")->str() : "...";
        int64_t leeway = opt_int(a, 3, "leeway", 5);
        std::string s = v.str();
        int64_t n = (int64_t)u8_length(s), elen = (int64_t)u8_length(end);
        if (length < elen) err("expected length >= " + std::to_string(elen) + ", got " + std::to_string(length));
        if (n <= length + leeway) return S(s);
        if (kill) return S(u8_substr(s, 0, length - elen) + end);
        std::string t = u8_substr(s, 0, length - elen);
        size_t sp = t.rfind(' ');
        if (sp != std::string::npos) t = t.substr(0, sp);
        return S(t + end);
    };
    FILTER("center") {
        (void)env;
        int64_t w = opt_int(a, 0, "width", 80);
        std::string s = v.str();
        int64_t len = (int64_t)u8_length(s);
        if (len >= w) return S(s);
        int64_t pad = w - len, left = pad / 2 + (pad & w & 1);
        return S(std::string(left, ' ') + s + std::string(pad - left, ' '));
    };
    FILTER("wordcount") { (void)env; (void)a; return Value((long long)split_ws(v.str(), -1).size()); };
    FILTER("striptags") {
        (void)env; (void)a;
        std::string s = v.str(), o;
        bool in = false;
        for (char c : s) { if (c == '<') in = true; else if (c == '>' && in) in = false; else if (!in) o += c; }
        std::string r;
        for (auto& w : split_ws(o, -1)) { if (!r.empty()) r += " "; r += w.as_string(); }
        return S(r);
    };
    FILTER("urlencode") {
        (void)env; (void)a;
        std::string o, s = v.str();
        for (unsigned char c : s) {
            if (isalnum(c) || strchr("/_.-~", c)) o += (char)c;
            else { char b[4]; snprintf(b, sizeof b, "%%%02X", c); o += b; }
        }
        return S(o);
    };
    FILTER("format") {
        (void)env;
        Value arg = a.pos.size() == 1 ? a.pos[0] : Value::tuple(a.pos);
        return S(percent_format(v.str(), arg));
    };
    FILTER("wordwrap") {
        (void)env;
        int64_t width = opt_int(a, 0, "width", 79);
        std::string s = v.str(), o, line;
        for (auto& w : split_ws(s, -1)) {
            if (!line.empty() && (int64_t)(u8_length(line) + 1 + u8_length(w.as_string())) > width) { o += line + "\n"; line.clear(); }
            line += (line.empty() ? "" : " ") + w.as_string();
        }
        return S(o + line);
    };

    // ---------------------------------------------------------------- tests
    TEST("defined") { (void)env; (void)a; return !v.is_undefined(); };
    TEST("undefined") { (void)env; (void)a; return v.is_undefined(); };
    TEST("none") { (void)env; (void)a; return v.is_none(); };
    TEST("string") { (void)env; (void)a; return v.is_string(); };
    TEST("number") { (void)env; (void)a; return v.is_int() || v.is_float() || v.is_bool(); };
    TEST("integer") { (void)env; (void)a; return v.is_int(); };
    TEST("float") { (void)env; (void)a; return v.is_float(); };
    TEST("boolean") { (void)env; (void)a; return v.is_bool(); };
    TEST("true") { (void)env; (void)a; return v.is_bool() && v.as_bool(); };
    TEST("false") { (void)env; (void)a; return v.is_bool() && !v.as_bool(); };
    TEST("mapping") { (void)env; (void)a; return v.is_dict(); };
    TEST("iterable") { (void)env; (void)a; return v.is_undefined() || v.is_string() || v.is_seq() || v.is_dict() || v.kind() == Value::Kind::Loop; };
    TEST("sequence") { (void)env; (void)a; return v.is_undefined() || v.is_string() || v.is_seq() || v.is_dict(); };
    TEST("callable") { (void)env; (void)a; return v.is_func() || v.kind() == Value::Kind::Loop; };
    TEST("even") { (void)env; (void)a; return v.as_int() % 2 == 0; };
    TEST("odd") { (void)env; (void)a; return v.as_int() % 2 != 0; };
    TEST("divisibleby") { (void)env; return v.as_int() % a.pos.at(0).as_int() == 0; };
    TEST("lower") { (void)env; (void)a; return v.is_string() && map_case(v.as_string(), false) == v.as_string(); };
    TEST("upper") { (void)env; (void)a; return v.is_string() && map_case(v.as_string(), true) == v.as_string(); };
    TEST("eq") { (void)env; return Value::equals(v, a.pos.at(0)); };
    env.tests["equalto"] = env.tests["eq"];
    env.tests["=="] = env.tests["eq"];
    TEST("ne") { (void)env; return !Value::equals(v, a.pos.at(0)); };
    env.tests["!="] = env.tests["ne"];
    TEST("lt") { (void)env; return Value::compare(v, a.pos.at(0)) < 0; };
    env.tests["lessthan"] = env.tests["lt"];
    env.tests["<"] = env.tests["lt"];
    TEST("le") { (void)env; return Value::compare(v, a.pos.at(0)) <= 0; };
    env.tests["<="] = env.tests["le"];
    TEST("gt") { (void)env; return Value::compare(v, a.pos.at(0)) > 0; };
    env.tests["greaterthan"] = env.tests["gt"];
    env.tests[">"] = env.tests["gt"];
    TEST("ge") { (void)env; return Value::compare(v, a.pos.at(0)) >= 0; };
    env.tests[">="] = env.tests["ge"];
    TEST("in") { (void)env; return contains(a.pos.at(0), v); };
    TEST("sameas") {
        (void)env;
        const Value& o = a.pos.at(0);
        if (v.kind() != o.kind()) return false;
        switch (v.kind()) {
            case Value::Kind::Undefined: case Value::Kind::None: return true;
            case Value::Kind::Bool: case Value::Kind::Int: case Value::Kind::Float: case Value::Kind::Str: return Value::equals(v, o);
            default: return v.obj() == o.obj();
        }
    };
    TEST("escaped") { (void)env; (void)a; (void)v; return false; };
    TEST("filter") { (void)a; return env.filters.count(v.str()) != 0; };
    TEST("test") { (void)a; return env.tests.count(v.str()) != 0; };

    // ---------------------------------------------------------------- globals
    env.globals["range"] = Value::func("range", [](Args& a) {
        int64_t start = 0, stop = 0, step = 1;
        if (a.pos.size() == 1) stop = a.pos[0].as_int();
        else if (a.pos.size() >= 2) { start = a.pos[0].as_int(); stop = a.pos[1].as_int(); if (a.pos.size() > 2) step = a.pos[2].as_int(); }
        else err("range expected at least 1 argument, got 0");
        if (step == 0) err("range() arg 3 must not be zero");
        std::vector<Value> o;
        if (step > 0) for (int64_t i = start; i < stop; i += step) o.push_back(Value((long long)i));
        else for (int64_t i = start; i > stop; i += step) o.push_back(Value((long long)i));
        return Lst(o);
    });
    env.globals["dict"] = Value::func("dict", [](Args& a) {
        Value d = Value::dict();
        if (!a.pos.empty()) {
            const Value& o = a.pos[0];
            if (o.is_dict()) for (auto& p : o.dict_data().kv) d.set(p.first, p.second);
            else for (auto& x : iterate(o)) d.set(x.items().at(0), x.items().at(1));
        }
        for (auto& p : a.kw) d.set(p.first, p.second);
        return d;
    });
    env.globals["namespace"] = Value::func("namespace", [](Args& a) {
        Value d = Value::ns();
        if (!a.pos.empty() && a.pos[0].is_dict()) for (auto& p : a.pos[0].dict_data().kv) d.set(p.first, p.second);
        for (auto& p : a.kw) d.set(p.first, p.second);
        return d;
    });
    env.globals["raise_exception"] = Value::func("raise_exception", [](Args& a) -> Value {
        throw RaisedError(a.pos.empty() ? "" : a.pos[0].str());
    });
    EnvImpl* ep = &env_;
    env.globals["strftime_now"] = Value::func("strftime_now", [ep](Args& a) {
        std::time_t t = ep->clock ? ep->clock() : std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[512];
        std::string fmt = a.pos.empty() ? "" : a.pos[0].str();
        size_t n = strftime(buf, sizeof buf, fmt.c_str(), &tm);
        return S(std::string(buf, n));
    });
    env.globals["joiner"] = Value::func("joiner", [](Args& a) {
        std::string sep = a.pos.empty() ? ", " : a.pos[0].str();
        auto used = std::make_shared<bool>(false);
        return Value::func("joiner", [sep, used](Args&) { if (!*used) { *used = true; return S(""); } return S(sep); });
    });
}

}  // namespace qv::jinja
