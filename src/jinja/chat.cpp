#include "jinja/chat.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>

namespace qv::chat {
namespace {

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
bool contains(std::string_view s, std::string_view n) { return s.find(n) != std::string_view::npos; }
bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
std::string_view ltrim(std::string_view s) { while (!s.empty() && is_ws(s.front())) s.remove_prefix(1); return s; }
std::string_view rtrim(std::string_view s) { while (!s.empty() && is_ws(s.back())) s.remove_suffix(1); return s; }
std::string_view trim(std::string_view s) { return ltrim(rtrim(s)); }
bool starts_with(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
bool ends_with(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }

// Length of the longest suffix of s that is a proper prefix of tag (streaming hold-back).
size_t partial_suffix(std::string_view s, std::string_view tag) {
    size_t mx = std::min(s.size(), tag.size() - 1);
    for (size_t n = mx; n > 0; --n)
        if (s.compare(s.size() - n, n, tag, 0, n) == 0) return n;
    return 0;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ rendering
ToolFormat detect_tool_format(std::string_view s) {
    if (contains(s, "<|content_invoke_tool_json|>")) return ToolFormat::Inkling;
    if (contains(s, "<function=") && contains(s, "<parameter=")) return ToolFormat::QwenXml;
    if (contains(s, "<tool_call>")) return ToolFormat::Hermes;
    if (contains(s, "[TOOL_CALLS]")) return ToolFormat::Mistral;
    return ToolFormat::Generic;
}

ChatTemplate ChatTemplate::from_source(std::string source, std::string bos, std::string eos) {
    ChatTemplate t;
    t.src_ = std::move(source);
    t.bos_ = std::move(bos);
    t.eos_ = std::move(eos);
    t.fmt_ = detect_tool_format(t.src_);
    t.supports_tools_ = contains(t.src_, "tools");
    t.thinking_ = contains(t.src_, "<think>") || contains(t.src_, "enable_thinking") || contains(t.src_, "reasoning");
    t.tpl_ = t.env_.compile(t.src_, "chat_template");
    return t;
}
ChatTemplate ChatTemplate::from_file(const std::string& path, std::string bos, std::string eos) {
    return from_source(slurp(path), std::move(bos), std::move(eos));
}
ChatTemplate ChatTemplate::from_tokenizer_config(const std::string& path) {
    jd::Value cfg = jd::parse(slurp(path));
    const jd::Value* ct = cfg.find("chat_template");
    if (!ct) throw std::runtime_error("tokenizer_config.json has no chat_template");
    std::string src;
    if (ct->is_string()) src = ct->as_string();
    else if (ct->is_array()) {
        for (auto& e : ct->items()) {
            const jd::Value* n = e.find("name");
            const jd::Value* t = e.find("template");
            if (t && t->is_string() && (src.empty() || (n && n->is_string() && n->as_string() == "default"))) src = t->as_string();
        }
    }
    if (src.empty()) throw std::runtime_error("no usable chat_template in tokenizer_config.json");
    auto tok = [&](const char* k) -> std::string {
        const jd::Value* v = cfg.find(k);
        if (!v) return "";
        if (v->is_string()) return v->as_string();
        if (v->is_object()) { const jd::Value* c = v->find("content"); if (c && c->is_string()) return c->as_string(); }
        return "";
    };
    return from_source(src, tok("bos_token"), tok("eos_token"));
}

bool ChatTemplate::prompt_forces_thinking(std::string_view prompt, std::string_view tag) { return ends_with(rtrim(prompt), tag); }

std::string ChatTemplate::render(const RenderInputs& in) const {
    using jinja::Value;
    Value ctx = Value::dict();
    bool keep_parts = contains(src_, "image") || contains(src_, "'text'") || contains(src_, "\"text\"") || contains(src_, ".text");
    Value msgs = jinja::from_json(in.messages);
    if (!msgs.is_list()) throw jinja::Error("messages must be an array");
    bool map_developer = !contains(src_, "<|channel|>");
    for (auto& m : msgs.items()) {
        if (!m.is_dict()) continue;
        const Value* role = m.find("role");
        if (map_developer && role && role->is_string() && role->as_string() == "developer") m.set("role", Value("system"));
        const Value* c = m.find("content");
        if (!c || c->is_none()) m.set("content", Value(""));
        else if (c->is_list() && !keep_parts) {
            bool all_text = true;
            std::string joined;
            for (auto& p : c->items()) {
                const Value* t = p.is_dict() ? p.find("text") : nullptr;
                const Value* ty = p.is_dict() ? p.find("type") : nullptr;
                if (p.is_string()) { if (!joined.empty()) joined += "\n"; joined += p.as_string(); }
                else if (t && t->is_string() && (!ty || ty->as_string() == "text")) { if (!joined.empty()) joined += "\n"; joined += t->as_string(); }
                else all_text = false;
            }
            if (all_text) m.set("content", Value(joined));
        }
        const Value* tcs = m.find("tool_calls");
        if (tcs && tcs->is_list()) {
            for (auto& tc : tcs->items()) {
                const Value* fn = tc.is_dict() ? tc.find("function") : nullptr;
                const Value* args = fn && fn->is_dict() ? fn->find("arguments") : nullptr;
                if (args && args->is_string()) {
                    try {
                        Value parsed = jinja::from_json(jd::parse(args->as_string()));
                        const_cast<Value*>(fn)->set("arguments", parsed);
                    } catch (const std::exception&) { /* keep the string */ }
                }
            }
        }
    }
    ctx.set("messages", msgs);
    ctx.set("tools", in.tools.is_null() ? Value::none() : jinja::from_json(in.tools));
    ctx.set("add_generation_prompt", Value(in.add_generation_prompt));
    ctx.set("bos_token", Value(in.bos_token.empty() ? bos_ : in.bos_token));
    ctx.set("eos_token", Value(in.eos_token.empty() ? eos_ : in.eos_token));
    if (in.enable_thinking) ctx.set("enable_thinking", Value(*in.enable_thinking));
    if (!in.reasoning_effort.empty()) ctx.set("reasoning_effort", Value(in.reasoning_effort));
    if (in.extra.is_object())
        for (auto& m : in.extra.members()) ctx.set(m.first, jinja::from_json(m.second));
    if (in.now) {
        std::time_t t = *in.now;
        jinja::Environment e2;  // clock override needs its own environment instance
        e2.set_clock([t] { return t; });
        return e2.compile(src_, "chat_template").render(ctx);
    }
    return tpl_.render(ctx);
}

// ------------------------------------------------------------------------------------------------ misc helpers
ReasoningFormat reasoning_format_from_name(std::string_view n) {
    if (n == "none") return ReasoningFormat::None;
    if (n == "deepseek" || n == "auto" || n.empty()) return ReasoningFormat::Deepseek;
    if (n == "deepseek-legacy") return ReasoningFormat::DeepseekLegacy;
    throw std::invalid_argument("unknown reasoning format: " + std::string(n));
}

bool Message::operator==(const Message& o) const {
    if (content != o.content || reasoning_content != o.reasoning_content || tool_calls.size() != o.tool_calls.size()) return false;
    for (size_t i = 0; i < tool_calls.size(); ++i)
        if (tool_calls[i].name != o.tool_calls[i].name || tool_calls[i].arguments != o.tool_calls[i].arguments) return false;
    return true;
}

std::string gen_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* cs = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string id = "call_";
    for (int i = 0; i < 24; ++i) id += cs[rng() % 62];
    return id;
}

jd::Value message_to_json(const Message& m, bool with_ids) {
    jd::Value o = jd::Value::object();
    o.set("role", "assistant");
    if (m.content.empty() && !m.tool_calls.empty()) o.set("content", jd::Value());
    else o.set("content", m.content);
    if (!m.reasoning_content.empty()) o.set("reasoning_content", m.reasoning_content);
    if (!m.tool_calls.empty()) {
        jd::Value arr = jd::Value::array();
        for (auto& tc : m.tool_calls) {
            jd::Value c = jd::Value::object();
            if (with_ids) c.set("id", tc.id.empty() ? gen_tool_call_id() : tc.id);
            c.set("type", "function");
            jd::Value f = jd::Value::object();
            f.set("name", tc.name);
            f.set("arguments", tc.arguments);
            c.set("function", f);
            arr.push(c);
        }
        o.set("tool_calls", arr);
    }
    return o;
}

size_t scan_json_value(std::string_view s, size_t pos) {
    while (pos < s.size() && is_ws(s[pos])) ++pos;
    if (pos >= s.size()) return std::string_view::npos;
    char c = s[pos];
    if (c == '"') {
        for (size_t i = pos + 1; i < s.size(); ++i) {
            if (s[i] == '\\') ++i;
            else if (s[i] == '"') return i + 1;
        }
        return std::string_view::npos;
    }
    if (c == '{' || c == '[') {
        int depth = 0;
        for (size_t i = pos; i < s.size(); ++i) {
            char d = s[i];
            if (d == '"') {
                size_t e = scan_json_value(s, i);
                if (e == std::string_view::npos) return e;
                i = e - 1;
            } else if (d == '{' || d == '[') ++depth;
            else if (d == '}' || d == ']') { if (--depth == 0) return i + 1; }
        }
        return std::string_view::npos;
    }
    size_t i = pos;
    while (i < s.size() && !is_ws(s[i]) && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    return i < s.size() ? i : std::string_view::npos;  // a bare literal at EOF may still grow
}

// ------------------------------------------------------------------------------------------------ reasoning
void extract_reasoning(std::string_view text, const ParseParams& p, std::string& reasoning, std::string& content) {
    reasoning.clear();
    content.clear();
    if (p.reasoning == ReasoningFormat::None) { content = std::string(text); return; }
    std::string_view rest = text;
    bool in_think = p.thinking_forced_open;
    if (!in_think) {
        std::string_view t = ltrim(text);
        if (starts_with(t, p.think_start)) { in_think = true; rest = t.substr(p.think_start.size()); }
        else if (p.partial && !p.flush && t.size() < p.think_start.size() && starts_with(p.think_start, t)) return;  // may become <think>
        else { content = std::string(text); return; }
    }
    size_t f = rest.find(p.think_end);
    std::string_view r, c;
    bool closed = f != std::string_view::npos;
    if (closed) { r = rest.substr(0, f); c = ltrim(rest.substr(f + p.think_end.size())); }
    else {
        r = rest;
        if (p.partial && !p.flush) r.remove_suffix(partial_suffix(r, p.think_end));
    }
    r = ltrim(r);
    r = rtrim(r);  // while open: trailing whitespace is held back until more text arrives
    reasoning = std::string(r);
    content = std::string(c);
    if (p.reasoning == ReasoningFormat::DeepseekLegacy) {
        content = std::string(text);
    }
}

// ------------------------------------------------------------------------------------------------ tool calls
namespace {

std::string dump_args(const jd::Value& v) { return v.dump(); }

// Normalise a JSON "arguments" member: object/array -> compact dump; string -> itself (OpenAI convention: JSON text).
std::string args_text(const jd::Value& v) {
    if (v.is_string()) {
        try { return jd::parse(v.as_string()).dump(); } catch (const std::exception&) { return v.as_string(); }
    }
    return dump_args(v);
}

const jd::Value* find_tool(const jd::Value* tools, const std::string& name) {
    if (!tools || !tools->is_array()) return nullptr;
    for (auto& t : tools->items()) {
        const jd::Value* fn = t.is_object() ? t.find("function") : nullptr;
        if (!fn) fn = &t;
        const jd::Value* n = fn->is_object() ? fn->find("name") : nullptr;
        if (n && n->is_string() && n->as_string() == name) return fn;
    }
    return nullptr;
}

bool schema_wants_string(const jd::Value* fn, const std::string& key, bool& known) {
    known = false;
    if (!fn) return false;
    const jd::Value* params = fn->find("parameters");
    const jd::Value* props = params && params->is_object() ? params->find("properties") : nullptr;
    const jd::Value* ps = props && props->is_object() ? props->find(key) : nullptr;
    if (!ps || !ps->is_object()) return false;
    known = true;
    const jd::Value* ty = ps->find("type");
    if (!ty) return false;
    if (ty->is_string()) return ty->as_string() == "string";
    if (ty->is_array()) {
        for (auto& x : ty->items()) if (!(x.is_string() && (x.as_string() == "string" || x.as_string() == "null"))) return false;
        return true;
    }
    return false;
}

// "name"+"arguments" object -> ToolCall. partial_args: raw argument text slice from `raw`.
bool call_from_object(const jd::Value& o, ToolCall& out, const char* args_key1 = "arguments", const char* args_key2 = "parameters") {
    if (!o.is_object()) return false;
    const jd::Value* n = o.find("name");
    if (!n || !n->is_string()) return false;
    out.name = n->as_string();
    const jd::Value* a = o.find(args_key1);
    if (!a) a = o.find(args_key2);
    if (!a) a = o.find("args");
    out.arguments = a ? args_text(*a) : "{}";
    return true;
}

// Locate the raw text of a member value inside an (incomplete) JSON object text: returns [b,e) or npos.
bool member_slice(std::string_view obj, std::string_view key, size_t& b, size_t& e, bool& complete) {
    // top-level key search: walk the object, skipping nested values
    size_t i = obj.find('{');
    if (i == std::string_view::npos) return false;
    ++i;
    for (;;) {
        while (i < obj.size() && (is_ws(obj[i]) || obj[i] == ',')) ++i;
        if (i >= obj.size() || obj[i] != '"') return false;
        size_t ke = scan_json_value(obj, i);
        if (ke == std::string_view::npos) return false;
        std::string k;
        try { k = jd::parse(obj.substr(i, ke - i)).as_string(); } catch (const std::exception&) { return false; }
        i = ke;
        while (i < obj.size() && is_ws(obj[i])) ++i;
        if (i >= obj.size() || obj[i] != ':') return false;
        ++i;
        while (i < obj.size() && is_ws(obj[i])) ++i;
        size_t ve = scan_json_value(obj, i);
        if (k == key) {
            b = i;
            complete = ve != std::string_view::npos;
            e = complete ? ve : obj.size();
            return i < obj.size();
        }
        if (ve == std::string_view::npos) return false;
        i = ve;
    }
}

// Parse a Hermes/Inkling style JSON call body (possibly incomplete in partial mode).
bool parse_json_call(std::string_view body, bool partial, const char* args_key, ToolCall& out) {
    body = trim(body);
    size_t e = scan_json_value(body, 0);
    if (e != std::string_view::npos) {
        try {
            jd::Value v = jd::parse(body.substr(0, e));
            if (call_from_object(v, out, args_key, "parameters")) {
                out.complete = true;
                if (partial) {  // raw slice so successive partial parses are prefix-stable
                    size_t b, en;
                    bool comp;
                    for (const char* k : {args_key, "parameters", "args"})
                        if (member_slice(body, k, b, en, comp) && comp) {
                            std::string_view raw = trim(body.substr(b, en - b));
                            if (!raw.empty() && raw[0] != '"') { out.arguments = std::string(raw); }
                            break;
                        }
                }
                return true;
            }
        } catch (const std::exception&) {}
        return false;
    }
    if (!partial) return false;
    // incomplete: need the name (closed string) at least
    size_t b, en;
    bool comp;
    if (!member_slice(body, "name", b, en, comp) || !comp) return false;
    try { out.name = jd::parse(body.substr(b, en - b)).as_string(); } catch (const std::exception&) { return false; }
    out.arguments.clear();
    out.complete = false;
    for (const char* k : {args_key, "parameters", "args"}) {
        if (member_slice(body, k, b, en, comp)) {
            std::string_view raw = body.substr(b, en - b);
            if (!raw.empty() && raw[0] == '"') break;  // string-encoded arguments: wait for completion
            out.arguments = std::string(comp ? trim(raw) : rtrim(raw));
            if (!comp) {  // do not emit a trailing partial scalar/whitespace that may still change
                if (!out.arguments.empty() && out.arguments[0] != '{' && out.arguments[0] != '[') out.arguments.clear();
            }
            break;
        }
    }
    return true;
}

struct ToolScan {
    std::string content;
    std::vector<ToolCall> calls;
};

void append_content(std::string& content, std::string_view s) { content.append(s); }

// <tool_call> ... </tool_call> blocks; `block` parses one body.
template <class BodyFn>
ToolScan scan_tagged(std::string_view text, const ParseParams& p, std::string_view open, std::string_view close, BodyFn body_fn) {
    ToolScan r;
    size_t pos = 0;
    bool any = false;
    std::string pending_between;
    while (pos <= text.size()) {
        size_t o = text.find(open, pos);
        if (o == std::string_view::npos) {
            std::string_view tail = text.substr(pos);
            if (p.partial && !p.flush) {
                tail.remove_suffix(partial_suffix(tail, open));
                if (!any) tail = rtrim(tail);  // whitespace right before a (future) <tool_call> is not content
            }
            if (any) { if (!trim(tail).empty()) { append_content(r.content, r.content.empty() ? tail : tail); } }
            else append_content(r.content, tail);
            break;
        }
        std::string_view before = text.substr(pos, o - pos);
        if (any) { if (!trim(before).empty()) append_content(r.content, before); }
        else append_content(r.content, before);
        any = true;
        size_t bs = o + open.size();
        size_t c = text.find(close, bs);
        std::string_view body = c == std::string_view::npos ? text.substr(bs) : text.substr(bs, c - bs);
        ToolCall tc;
        bool closed = c != std::string_view::npos;
        if (body_fn(body, closed, tc)) {
            r.calls.push_back(std::move(tc));
        } else if (closed || !p.partial) {
            // unparsable block: keep literally as content
            append_content(r.content, text.substr(o, closed ? c + close.size() - o : text.size() - o));
        }
        if (!closed) { pos = text.size() + 1; break; }
        pos = c + close.size();
    }
    // trailing whitespace before the first tool call is not content
    if (any) {
        std::string_view cv = rtrim(r.content);
        r.content.resize(cv.size());
    }
    return r;
}

bool qwen_body(std::string_view body, bool closed, bool partial, const jd::Value* tools, ToolCall& out) {
    size_t f = body.find("<function=");
    if (f == std::string_view::npos) return false;
    size_t ne = body.find('>', f);
    if (ne == std::string_view::npos) return false;
    out.name = std::string(trim(body.substr(f + 10, ne - f - 10)));
    const jd::Value* fn = find_tool(tools, out.name);
    size_t pos = ne + 1;
    std::string args = "{";
    bool first = true;
    bool fn_closed = false;
    for (;;) {
        size_t po = body.find("<parameter=", pos);
        size_t fc = body.find("</function>", pos);
        if (fc != std::string_view::npos && (po == std::string_view::npos || fc < po)) { fn_closed = true; break; }
        if (po == std::string_view::npos) break;
        size_t ke = body.find('>', po);
        if (ke == std::string_view::npos) break;
        std::string key(trim(body.substr(po + 11, ke - po - 11)));
        size_t ve = body.find("</parameter>", ke + 1);
        bool pclosed = ve != std::string_view::npos;
        if (!pclosed) {
            // tolerate a missing </parameter> when the function/tool_call was closed after the value
            size_t fc2 = body.find("</function>", ke + 1);
            if (fc2 != std::string_view::npos) { ve = fc2; pclosed = true; }
            else if (closed || !partial) { ve = body.size(); pclosed = true; }
            else break;  // value still streaming: wait
        }
        std::string val(body.substr(ke + 1, ve - ke - 1));
        if (!val.empty() && val.front() == '\n') val.erase(0, 1);
        if (!val.empty() && val.back() == '\n') val.pop_back();
        bool known = false;
        bool want_str = schema_wants_string(fn, key, known);
        jd::Value jv;
        bool is_json = false;
        if (!want_str) {
            try {
                jv = jd::parse(trim(val));
                is_json = known || jv.is_object() || jv.is_array();
            } catch (const std::exception&) {}
        }
        if (!first) args += ",";
        first = false;
        jd::Value::dump_string(args, key);
        args += ":";
        if (is_json) args += jv.dump();
        else jd::Value::dump_string(args, val);
        pos = pclosed && ve < body.size() && body.compare(ve, 12, "</parameter>") == 0 ? ve + 12 : ve;
    }
    if (fn_closed || (!partial) || closed) { args += "}"; out.complete = true; }
    else out.complete = false;
    out.arguments = args;
    if (partial && !closed && !fn_closed) out.complete = false;
    return true;
}

ToolScan parse_hermes(std::string_view text, const ParseParams& p) {
    return scan_tagged(text, p, "<tool_call>", "</tool_call>", [&](std::string_view body, bool closed, ToolCall& tc) {
        if (!closed && !p.partial) {
            // missing close tag at EOF: accept if the JSON is complete
            return parse_json_call(body, false, "arguments", tc);
        }
        return parse_json_call(body, p.partial, "arguments", tc);
    });
}
ToolScan parse_qwen(std::string_view text, const ParseParams& p) {
    return scan_tagged(text, p, "<tool_call>", "</tool_call>",
                       [&](std::string_view body, bool closed, ToolCall& tc) { return qwen_body(body, closed, p.partial, p.tools, tc); });
}

ToolScan parse_mistral(std::string_view text, const ParseParams& p) {
    ToolScan r;
    size_t o = text.find("[TOOL_CALLS]");
    if (o == std::string_view::npos) {
        std::string_view t = text;
        if (p.partial && !p.flush) t.remove_suffix(partial_suffix(t, "[TOOL_CALLS]"));
        r.content = std::string(t);
        return r;
    }
    r.content = std::string(rtrim(text.substr(0, o)));
    std::string_view rest = ltrim(text.substr(o + 12));
    size_t e = scan_json_value(rest, 0);
    if (e == std::string_view::npos) return r;
    try {
        jd::Value arr = jd::parse(rest.substr(0, e));
        if (arr.is_object()) { jd::Value a2 = jd::Value::array(); a2.push(arr); arr = a2; }
        if (arr.is_array())
            for (auto& x : arr.items()) {
                ToolCall tc;
                if (call_from_object(x, tc)) { const jd::Value* id = x.find("id"); if (id && id->is_string()) tc.id = id->as_string(); r.calls.push_back(tc); }
            }
    } catch (const std::exception&) {
        r.content = std::string(text);
        r.calls.clear();
    }
    return r;
}

ToolScan parse_generic(std::string_view text, const ParseParams& p) {
    ToolScan r;
    std::string_view t = trim(text);
    auto as_content = [&] { r.content = std::string(text); r.calls.clear(); return r; };
    if (!p.tools || !p.tools->is_array() || p.tools->items().empty() || t.empty() || (t[0] != '{' && t[0] != '[')) return as_content();
    size_t e = scan_json_value(t, 0);
    if (e == std::string_view::npos) {
        if (p.partial) { r.content.clear(); return r; }  // might still become a tool-call JSON
        return as_content();
    }
    if (!trim(t.substr(e)).empty()) return as_content();
    try {
        jd::Value v = jd::parse(t.substr(0, e));
        std::vector<const jd::Value*> items;
        if (v.is_object()) {
            if (const jd::Value* tcs = v.find("tool_calls"); tcs && tcs->is_array()) for (auto& x : tcs->items()) items.push_back(&x);
            else if (const jd::Value* tc = v.find("tool_call"); tc && tc->is_object()) items.push_back(tc);
            else if (const jd::Value* resp = v.find("response"); resp && resp->is_string() && v.size() == 1) { r.content = resp->as_string(); return r; }
            else items.push_back(&v);
        } else if (v.is_array()) for (auto& x : v.items()) items.push_back(&x);
        for (auto* x : items) {
            ToolCall tc;
            if (!call_from_object(*x, tc)) return as_content();
            const jd::Value* id = x->find("id");
            if (id && id->is_string()) tc.id = id->as_string();
            r.calls.push_back(tc);
        }
    } catch (const std::exception&) { return as_content(); }
    return r;
}

// ---- Inkling message stream (see tokenizer_config.json "response_template")
Message parse_inkling(std::string_view text, const ParseParams& p) {
    static constexpr std::string_view MM = "<|message_model|>", END = "<|end_message|>", STOP = "<|content_model_end_sampling|>";
    static const std::string_view kinds[] = {"<|content_thinking|>", "<|content_text|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>", "<|content_invoke_tool_json|>"};
    Message m;
    size_t pos = 0;
    bool any_header = false;
    while (pos < text.size()) {
        std::string_view rest = text.substr(pos);
        if (starts_with(rest, STOP)) break;
        // header: optional <|message_model|>name then <|content_X|>
        size_t hdr = pos;
        std::string name;
        if (starts_with(rest, MM)) {
            size_t i = pos + MM.size();
            size_t lt = text.find('<', i);
            if (lt == std::string_view::npos) {
                if (!p.partial) m.content += std::string(rest);
                break;
            }
            name = std::string(text.substr(i, lt - i));
            hdr = lt;
        }
        int kind = -1;
        std::string_view hrest = text.substr(hdr);
        for (int k = 0; k < 5; ++k) if (starts_with(hrest, kinds[k])) { kind = k; break; }
        if (kind < 0) {
            // partial marker at the very end or stray text
            if (p.partial && !p.flush && hrest.size() < 32 && (starts_with(MM, hrest) || starts_with(STOP, hrest) || [&] { for (auto k : kinds) if (starts_with(k, hrest)) return true; return false; }() || starts_with(END, hrest)))
                break;
            if (starts_with(hrest, END)) { pos = hdr + END.size(); continue; }
            // stray text until the next marker
            size_t nx = text.find('<', hdr + 1);
            size_t e = nx == std::string_view::npos ? text.size() : nx;
            if (!any_header && nx == std::string_view::npos) { m.content += std::string(text.substr(pos)); break; }
            m.content += std::string(text.substr(pos, e - pos));
            pos = e;
            continue;
        }
        any_header = true;
        size_t bs = hdr + kinds[kind].size();
        size_t be = text.find(END, bs);
        size_t se = text.find(STOP, bs);
        bool closed = be != std::string_view::npos && (se == std::string_view::npos || be < se);
        std::string_view body;
        size_t next;
        if (closed) { body = text.substr(bs, be - bs); next = be + END.size(); }
        else if (se != std::string_view::npos) { body = text.substr(bs, se - bs); next = se; closed = true; }
        else { body = text.substr(bs); next = text.size(); }
        if (kind == 4) {
            ToolCall tc;
            bool ok = parse_json_call(body, p.partial && !closed, "args", tc);
            if (ok) {
                if (tc.name.empty()) tc.name = name;
                if (!closed && p.partial) tc.complete = false;
                m.tool_calls.push_back(tc);
            } else if (closed || !p.partial) {
                m.content += std::string(body);
            }
        } else {
            std::string_view b = body;
            if (!closed && p.partial && !p.flush) b.remove_suffix(std::max(partial_suffix(b, END), partial_suffix(b, STOP)));
            if (kind == 0) { if (p.reasoning == ReasoningFormat::None) m.content += std::string(MM) + std::string(kinds[0]) + std::string(b) + (closed ? std::string(END) : ""); else m.reasoning_content += std::string(b); }
            else m.content += std::string(b);
        }
        pos = next;
        if (!closed) break;
    }
    return m;
}

}  // namespace

Message parse_output(std::string_view text, const ParseParams& p) {
    if (p.format == ToolFormat::Inkling) return parse_inkling(text, p);
    Message m;
    std::string reasoning, content;
    extract_reasoning(text, p, reasoning, content);
    m.reasoning_content = std::move(reasoning);
    if (!p.parse_tool_calls) { m.content = std::move(content); return m; }
    ParseParams q = p;
    ToolScan s;
    switch (p.format) {
        case ToolFormat::Hermes: s = parse_hermes(content, q); break;
        case ToolFormat::QwenXml: s = parse_qwen(content, q); break;
        case ToolFormat::Mistral: s = parse_mistral(content, q); break;
        default: s = parse_generic(content, q); break;
    }
    // the legacy format keeps the think block inside the content
    if (p.reasoning == ReasoningFormat::DeepseekLegacy && !m.reasoning_content.empty()) {
        std::string_view t = text;
        size_t o = t.find("<tool_call>");
        m.content = o == std::string_view::npos || s.calls.empty() ? std::string(t) : std::string(rtrim(t.substr(0, o)));
    } else m.content = std::move(s.content);
    m.tool_calls = std::move(s.calls);
    return m;
}

Delta diff_messages(const Message& prev, const Message& cur) {
    Delta d;
    auto suffix = [](const std::string& a, const std::string& b) -> std::string {
        return starts_with(b, a) ? b.substr(a.size()) : std::string();
    };
    d.content = suffix(prev.content, cur.content);
    d.reasoning = suffix(prev.reasoning_content, cur.reasoning_content);
    for (size_t i = 0; i < cur.tool_calls.size(); ++i) {
        const ToolCall& c = cur.tool_calls[i];
        ToolCallDelta td;
        td.index = i;
        if (i >= prev.tool_calls.size() || prev.tool_calls[i].name.empty()) {
            td.is_new = true;
            td.name = c.name;
            td.arguments = c.arguments;
            if (!c.name.empty() || !c.arguments.empty()) d.tool_calls.push_back(td);
        } else {
            td.arguments = suffix(prev.tool_calls[i].arguments, c.arguments);
            if (!td.arguments.empty()) d.tool_calls.push_back(td);
        }
    }
    return d;
}

}  // namespace qv::chat
