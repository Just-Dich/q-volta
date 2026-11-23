#include "server/compat_api.h"

#include <ctime>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>

namespace qv::server {
using jd::Value;

namespace {

std::string rnd(int n) {
    static std::mutex m;
    static std::mt19937_64 rng{std::random_device{}()};
    static const char* cs = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::lock_guard<std::mutex> g(m);
    std::string s;
    for (int i = 0; i < n; ++i) s += cs[rng() % 62];
    return s;
}

std::string get_str(const Value& v, const char* k, const std::string& d = "") {
    const Value* x = v.is_object() ? v.find(k) : nullptr;
    return x && x->is_string() ? x->as_string() : d;
}
const Value* get(const Value& v, const char* k) { return v.is_object() ? v.find(k) : nullptr; }
bool is_str(const Value& v, const char* k) { const Value* x = get(v, k); return x && x->is_string(); }
bool is_arr(const Value& v, const char* k) { const Value* x = get(v, k); return x && x->is_array(); }

Value obj() { return Value::object(); }
Value arr() { return Value::array(); }

std::string sse(const char* event, const Value& data) { return std::string("event: ") + event + "\ndata: " + data.dump() + "\n\n"; }

// Edits the cch stamp of an "x-anthropic-billing-header" system prompt (it changes on every request and would defeat prefix caching).
void normalize_billing_header(std::string& s) {
    static const std::string pfx = "x-anthropic-billing-header:";
    if (s.rfind(pfx, 0) != 0) return;
    size_t i = s.find("cch=", pfx.size());
    if (i == std::string::npos) return;
    size_t r = i + 4;
    if (r + 5 < s.size() && s[r + 5] == ';') for (size_t k = 0; k < 5; ++k) s[r + k] = 'f';
}

Value image_part(const Value& source) {
    std::string st = get_str(source, "type");
    Value u = obj();
    if (st == "base64") u.set("url", "data:" + get_str(source, "media_type", "image/jpeg") + ";base64," + get_str(source, "data"));
    else if (st == "url") u.set("url", get_str(source, "url"));
    else return Value();
    Value p = obj();
    p.set("type", "image_url");
    p.set("image_url", std::move(u));
    return p;
}

}  // namespace

// ------------------------------------------------------------------------------------------------- Anthropic -> chat completions
Value anthropic_to_oai(const Value& body) {
    Value out = obj();
    Value msgs = arr();
    if (const Value* sys = get(body, "system"); sys && !sys->is_null()) {
        std::string content;
        if (sys->is_string()) { content = sys->as_string(); normalize_billing_header(content); }
        else if (sys->is_array())
            for (auto& b : sys->items())
                if (get_str(b, "type") == "text") { std::string t = get_str(b, "text"); normalize_billing_header(t); content += t; }
        Value m = obj();
        m.set("role", "system");
        m.set("content", content);
        msgs.push(std::move(m));
    }
    const Value* ms = get(body, "messages");
    if (!ms) throw std::runtime_error("'messages' is required");
    if (ms->is_array()) {
        for (auto& msg : ms->items()) {
            std::string role = get_str(msg, "role");
            const Value* content = get(msg, "content");
            if (!content) { if (role == "assistant") continue; msgs.push(msg); continue; }
            if (!content->is_array()) { msgs.push(msg); continue; }
            Value tool_calls = arr(), converted = arr(), tool_results = arr();
            std::string reasoning;
            bool has_tc = false;
            for (auto& block : content->items()) {
                std::string type = get_str(block, "type");
                if (type == "text") converted.push(block);
                else if (type == "thinking") reasoning += get_str(block, "thinking");
                else if (type == "image") { if (const Value* s = get(block, "source")) { Value p = image_part(*s); if (!p.is_null()) converted.push(std::move(p)); } }
                else if (type == "tool_use") {
                    Value fn = obj();
                    fn.set("name", get_str(block, "name"));
                    const Value* in = get(block, "input");
                    fn.set("arguments", in ? in->dump() : std::string("{}"));
                    Value tc = obj();
                    tc.set("id", get_str(block, "id"));
                    tc.set("type", "function");
                    tc.set("function", std::move(fn));
                    tool_calls.push(std::move(tc));
                    has_tc = true;
                } else if (type == "tool_result") {
                    std::string id = get_str(block, "tool_use_id");
                    const Value* rc = get(block, "content");
                    Value tm = obj();
                    tm.set("role", "tool");
                    tm.set("tool_call_id", id);
                    if (rc && rc->is_string()) tm.set("content", rc->as_string());
                    else if (rc && rc->is_array()) {
                        std::string text;
                        Value parts = arr();
                        bool keep = false;
                        for (auto& c : rc->items()) {
                            std::string ct = get_str(c, "type");
                            if (ct == "text") {
                                text += get_str(c, "text");
                                Value p = obj(); p.set("type", "text"); p.set("text", get_str(c, "text")); parts.push(std::move(p));
                            } else if (ct == "image") {
                                keep = true;
                                if (const Value* s = get(c, "source")) { Value p = image_part(*s); if (!p.is_null()) parts.push(std::move(p)); }
                            } else if (ct == "tool_reference") {
                                keep = true;
                                Value p = obj(); p.set("type", "tool_reference"); p.set("name", get_str(c, "tool_name")); parts.push(std::move(p));
                            }
                        }
                        if (keep) tm.set("content", std::move(parts)); else tm.set("content", text);
                    } else tm.set("content", "");
                    tool_results.push(std::move(tm));
                }
            }
            if (converted.size() > 0 || has_tc || !reasoning.empty()) {
                Value nm = obj();
                nm.set("role", role);
                if (converted.size() > 0) nm.set("content", std::move(converted));
                else nm.set("content", "");
                if (tool_calls.size() > 0) nm.set("tool_calls", std::move(tool_calls));
                if (!reasoning.empty()) nm.set("reasoning_content", reasoning);
                msgs.push(std::move(nm));
            }
            for (auto& tm : tool_results.items()) msgs.push(tm);
        }
    }
    out.set("messages", std::move(msgs));
    if (const Value* tools = get(body, "tools"); tools && tools->is_array()) {
        Value ot = arr();
        for (auto& t : tools->items()) {
            Value fn = obj();
            fn.set("name", get_str(t, "name"));
            fn.set("description", get_str(t, "description"));
            const Value* sch = get(t, "input_schema");
            fn.set("parameters", sch ? *sch : obj());
            Value e = obj();
            e.set("type", "function");
            e.set("function", std::move(fn));
            if (const Value* d = get(t, "defer_loading")) e.set("defer_loading", *d);
            ot.push(std::move(e));
        }
        out.set("tools", std::move(ot));
    }
    if (const Value* tc = get(body, "tool_choice"); tc && tc->is_object()) {
        std::string type = get_str(*tc, "type");
        if (type == "auto") out.set("tool_choice", "auto");
        else if (type == "any" || type == "tool") out.set("tool_choice", "required");
    }
    if (const Value* s = get(body, "stop_sequences")) out.set("stop", *s);
    if (const Value* mt = get(body, "max_tokens")) out.set("max_tokens", *mt); else out.set("max_tokens", 4096);
    for (const char* k : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs", "id_slot"})
        if (const Value* v = get(body, k)) out.set(k, *v);
    if (const Value* th = get(body, "thinking"); th && th->is_object() && get_str(*th, "type") == "enabled") {
        const Value* b = get(*th, "budget_tokens");
        out.set("thinking_budget_tokens", b && b->is_number() ? *b : Value(10000));
    }
    if (const Value* oc = get(body, "output_config"); oc && oc->is_object()) {
        std::string e = get_str(*oc, "effort");
        if (!e.empty()) out.set("reasoning_effort", e);
    }
    if (const Value* ex = get(body, "extra_params"); ex && ex->is_object())
        for (auto& m : ex->members()) out.set(m.first, m.second);
    return out;
}

// ------------------------------------------------------------------------------------------------- Responses -> chat completions
static Value responses_tools(const Value& tools) {
    if (!tools.is_array()) throw std::invalid_argument("'tools' must be an array of objects");
    Value r = arr();
    for (auto& t : tools.items()) {
        if (get_str(t, "type") != "function") continue;  // non-function tools have no chat-completions equivalent
        Value fn = obj();
        for (auto& m : t.members()) if (m.first != "type") fn.set(m.first, m.second);
        if (!get(fn, "strict")) fn.set("strict", true);
        Value e = obj();
        e.set("type", "function");
        e.set("function", std::move(fn));
        r.push(std::move(e));
    }
    return r;
}

Value responses_to_oai(const Value& body) {
    if (!get(body, "input")) throw std::invalid_argument("'input' is required");
    if (!get_str(body, "previous_response_id").empty()) throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    const Value& input = *get(body, "input");
    Value out = obj();
    for (auto& m : body.members()) if (m.first != "input" && m.first != "instructions") out.set(m.first, m.second);
    std::vector<Value> msgs;
    if (get(body, "instructions")) { Value m = obj(); m.set("role", "system"); m.set("content", get_str(body, "instructions")); msgs.push_back(std::move(m)); }
    auto role_of = [](const Value& m) { return get_str(m, "role"); };
    if (input.is_string()) {
        Value m = obj(); m.set("role", "user"); m.set("content", input); msgs.push_back(std::move(m));
    } else if (input.is_array()) {
        for (Value item : input.items()) {
            bool merge_prev = !msgs.empty() && role_of(msgs.back()) == "assistant";
            if (is_str(item, "content")) {
                Value part = obj(); part.set("text", *get(item, "content")); part.set("type", "input_text");
                Value a = arr(); a.push(std::move(part)); item.set("content", std::move(a));
            }
            const std::string role = get_str(item, "role"), type = get_str(item, "type");
            if (is_arr(item, "content") && is_str(item, "role") && (role == "user" || role == "system" || role == "developer")) {
                Value content = arr();
                for (auto& it : get(item, "content")->items()) {
                    std::string t = get_str(it, "type");
                    if (t == "input_text") {
                        if (!get(it, "text")) throw std::invalid_argument("'Input text' requires 'text'");
                        Value p = obj(); p.set("text", *get(it, "text")); p.set("type", "text"); content.push(std::move(p));
                    } else if (t == "input_image") {
                        if (!get(it, "image_url")) throw std::invalid_argument("'image_url' is required");
                        Value u = obj(); u.set("url", *get(it, "image_url"));
                        Value p = obj(); p.set("image_url", std::move(u)); p.set("type", "image_url"); content.push(std::move(p));
                    } else if (t == "input_file") throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    else throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                }
                Value m = obj();
                for (auto& mm : item.members()) if (mm.first != "type" && mm.first != "status" && mm.first != "content") m.set(mm.first, mm.second);
                m.set("content", std::move(content));
                msgs.push_back(std::move(m));
            } else if (is_str(item, "role") && role == "assistant" && type == "message") {
                Value content = arr();
                const Value* c = get(item, "content");
                if (c && c->is_string()) { Value p = obj(); p.set("text", *c); p.set("type", "text"); content.push(std::move(p)); }
                else if (c && c->is_array()) {
                    for (auto& ot : c->items()) {
                        std::string t = get_str(ot, "type");
                        if (t == "output_text" || t == "input_text") {
                            if (!is_str(ot, "text")) throw std::invalid_argument("'Output text' requires 'text'");
                            Value p = obj(); p.set("text", *get(ot, "text")); p.set("type", "text"); content.push(std::move(p));
                        } else if (t == "refusal") {
                            if (!is_str(ot, "refusal")) throw std::invalid_argument("'Refusal' requires 'refusal'");
                            Value p = obj(); p.set("refusal", *get(ot, "refusal")); p.set("type", "refusal"); content.push(std::move(p));
                        } else throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                    }
                }
                if (merge_prev) {
                    Value& prev = msgs.back();
                    if (!is_arr(prev, "content")) prev.set("content", arr());
                    Value& pc = prev.set("content", *get(prev, "content"));
                    for (auto& p : content.items()) pc.push(p);
                } else {
                    Value m = obj();
                    for (auto& mm : item.members()) if (mm.first != "type" && mm.first != "status" && mm.first != "content") m.set(mm.first, mm.second);
                    m.set("content", std::move(content));
                    msgs.push_back(std::move(m));
                }
            } else if (is_str(item, "arguments") && is_str(item, "call_id") && is_str(item, "name") && type == "function_call") {
                Value fn = obj(); fn.set("arguments", *get(item, "arguments")); fn.set("name", *get(item, "name"));
                Value tc = obj(); tc.set("function", std::move(fn)); tc.set("id", *get(item, "call_id")); tc.set("type", "function");
                if (merge_prev) {
                    Value& prev = msgs.back();
                    if (!is_arr(prev, "tool_calls")) prev.set("tool_calls", arr());
                    Value cur = *get(prev, "tool_calls");
                    cur.push(std::move(tc));
                    prev.set("tool_calls", std::move(cur));
                } else {
                    Value a = arr(); a.push(std::move(tc));
                    Value m = obj(); m.set("role", "assistant"); m.set("tool_calls", std::move(a));
                    msgs.push_back(std::move(m));
                }
            } else if (is_str(item, "call_id") && (is_str(item, "output") || is_arr(item, "output")) && type == "function_call_output") {
                Value m = obj();
                const Value& o = *get(item, "output");
                if (o.is_string()) m.set("content", o);
                else {
                    Value outs = arr();
                    for (Value co : o.items()) {
                        if (!get(co, "type")) throw std::invalid_argument("Output of tool call missing 'type' field");
                        std::string t = get_str(co, "type");
                        if (t != "input_text" && t != "input_image") throw std::invalid_argument("Output of tool call should be 'Input text' or 'Input image'");
                        if (t == "input_text") co.set("type", "text");
                        else {
                            if (!get(co, "image_url")) throw std::invalid_argument("'image_url' is required");
                            co.set("type", "image_url");
                            Value u = obj(); u.set("url", *get(co, "image_url"));
                            co.set("image_url", std::move(u));
                        }
                        outs.push(std::move(co));
                    }
                    m.set("content", std::move(outs));
                }
                m.set("role", "tool");
                m.set("tool_call_id", *get(item, "call_id"));
                msgs.push_back(std::move(m));
            } else if (is_arr(item, "summary") && type == "reasoning") {
                const Value* c = get(item, "content");
                if (!c || !c->is_array()) throw std::invalid_argument("item['content'] is not an array");
                if (c->size() == 0) throw std::invalid_argument("item['content'] is empty");
                if (!is_str(c->items()[0], "text")) throw std::invalid_argument("item['content']['text'] is not a string");
                if (merge_prev) msgs.back().set("reasoning_content", *get(c->items()[0], "text"));
                else {
                    Value m = obj(); m.set("role", "assistant"); m.set("content", arr()); m.set("reasoning_content", *get(c->items()[0], "text"));
                    msgs.push_back(std::move(m));
                }
            } else throw std::invalid_argument("Cannot determine type of 'item'");
        }
    } else throw std::invalid_argument("'input' must be a string or array of objects");
    Value ma = arr();
    for (auto& m : msgs) ma.push(std::move(m));
    out.set("messages", std::move(ma));
    if (const Value* tools = get(body, "tools")) {
        Value ct = responses_tools(*tools);
        out.set("tools", Value());
        Value cleaned = obj();
        for (auto& m : out.members()) if (m.first != "tools") cleaned.set(m.first, m.second);
        out = std::move(cleaned);
        if (ct.size() > 0) out.set("tools", std::move(ct));
    }
    if (const Value* mo = get(body, "max_output_tokens")) {
        Value cleaned = obj();
        for (auto& m : out.members()) if (m.first != "max_output_tokens") cleaned.set(m.first, m.second);
        out = std::move(cleaned);
        out.set("max_tokens", *mo);
    }
    if (const Value* r = get(body, "reasoning")) {
        if (const Value* e = get(*r, "effort")) out.set("reasoning_effort", *e);
        Value cleaned = obj();
        for (auto& m : out.members()) if (m.first != "reasoning") cleaned.set(m.first, m.second);
        out = std::move(cleaned);
    }
    if (const Value* tx = get(body, "text")) {
        if (!tx->is_object()) throw std::invalid_argument("'text' must be an object");
        Value cleaned = obj();
        for (auto& m : out.members()) if (m.first != "text") cleaned.set(m.first, m.second);
        out = std::move(cleaned);
        if (const Value* f = get(*tx, "format")) {
            if (!f->is_object()) throw std::invalid_argument("'text.format' must be an object");
            std::string type = get_str(*f, "type");
            if (type == "json_schema") {
                Value js = obj();
                for (auto& m : f->members()) if (m.first != "type") js.set(m.first, m.second);
                Value rf = obj(); rf.set("type", "json_schema"); rf.set("json_schema", std::move(js));
                out.set("response_format", std::move(rf));
            } else if (type == "json_object" || type == "text") out.set("response_format", *f);
            else if (!type.empty()) throw std::invalid_argument("'text.format.type' must be one of 'text', 'json_object', or 'json_schema'");
        }
    }
    if (const Value* ex = get(body, "extra_params")) {
        Value cleaned = obj();
        for (auto& m : out.members()) if (m.first != "extra_params") cleaned.set(m.first, m.second);
        out = std::move(cleaned);
        if (ex->is_object()) for (auto& m : ex->members()) out.set(m.first, m.second);
    }
    return out;
}

// ------------------------------------------------------------------------------------------------- non-streaming results
namespace {
struct Parsed {
    std::string reasoning, content, finish, id, model;
    std::vector<std::tuple<std::string, std::string, std::string>> calls;  // id, name, arguments
    long long prompt = 0, completion = 0, cached = 0;
};
Parsed parse_oai(const Value& r) {
    Parsed p;
    p.id = get_str(r, "id");
    p.model = get_str(r, "model");
    if (const Value* ch = get(r, "choices"); ch && ch->is_array() && ch->size() > 0) {
        const Value& c0 = ch->items()[0];
        p.finish = get_str(c0, "finish_reason");
        if (const Value* m = get(c0, "message")) {
            p.content = get_str(*m, "content");
            p.reasoning = get_str(*m, "reasoning_content");
            if (const Value* tcs = get(*m, "tool_calls"); tcs && tcs->is_array())
                for (auto& tc : tcs->items()) {
                    const Value* fn = get(tc, "function");
                    p.calls.emplace_back(get_str(tc, "id"), fn ? get_str(*fn, "name") : "", fn ? get_str(*fn, "arguments") : "");
                }
        }
    }
    if (const Value* u = get(r, "usage")) {
        if (const Value* x = get(*u, "prompt_tokens")) p.prompt = x->as_int();
        if (const Value* x = get(*u, "completion_tokens")) p.completion = x->as_int();
        if (const Value* d = get(*u, "prompt_tokens_details")) if (const Value* x = get(*d, "cached_tokens")) p.cached = x->as_int();
    }
    return p;
}
const char* anthropic_stop(const std::string& finish) { return finish == "tool_calls" ? "tool_use" : finish == "length" ? "max_tokens" : "end_turn"; }
}  // namespace

Value oai_to_anthropic(const Value& r, const std::string& stopping_word) {
    Parsed p = parse_oai(r);
    Value blocks = arr();
    if (!p.reasoning.empty()) { Value b = obj(); b.set("type", "thinking"); b.set("thinking", p.reasoning); b.set("signature", ""); blocks.push(std::move(b)); }
    if (!p.content.empty()) { Value b = obj(); b.set("type", "text"); b.set("text", p.content); blocks.push(std::move(b)); }
    for (auto& [id, name, args] : p.calls) {
        Value b = obj();
        b.set("type", "tool_use"); b.set("id", id); b.set("name", name);
        Value in;
        try { in = jd::parse(args); } catch (...) { in = obj(); }
        b.set("input", std::move(in));
        blocks.push(std::move(b));
    }
    Value res = obj();
    res.set("id", p.id);
    res.set("type", "message");
    res.set("role", "assistant");
    res.set("content", std::move(blocks));
    res.set("model", p.model);
    res.set("stop_reason", anthropic_stop(p.finish));
    res.set("stop_sequence", stopping_word.empty() ? Value() : Value(stopping_word));
    Value u = obj();
    u.set("cache_read_input_tokens", p.cached);
    u.set("input_tokens", p.prompt - p.cached);
    u.set("output_tokens", p.completion);
    res.set("usage", std::move(u));
    return res;
}

static Value resp_usage(const Parsed& p) {
    Value u = obj();
    u.set("input_tokens", p.prompt);
    u.set("output_tokens", p.completion);
    u.set("total_tokens", p.prompt + p.completion);
    Value d = obj(); d.set("cached_tokens", p.cached);
    u.set("input_tokens_details", std::move(d));
    return u;
}

Value oai_to_responses(const Value& r) {
    Parsed p = parse_oai(r);
    Value out = arr();
    if (!p.reasoning.empty()) {
        Value t = obj(); t.set("text", p.reasoning); t.set("type", "reasoning_text");
        Value c = arr(); c.push(std::move(t));
        Value it = obj();
        it.set("id", "rs_" + rnd(32)); it.set("summary", arr()); it.set("type", "reasoning"); it.set("content", std::move(c)); it.set("encrypted_content", ""); it.set("status", "completed");
        out.push(std::move(it));
    }
    if (!p.content.empty()) {
        Value t = obj(); t.set("type", "output_text"); t.set("annotations", arr()); t.set("logprobs", arr()); t.set("text", p.content);
        Value c = arr(); c.push(std::move(t));
        Value it = obj();
        it.set("content", std::move(c)); it.set("id", "msg_" + rnd(32)); it.set("role", "assistant"); it.set("status", "completed"); it.set("type", "message");
        out.push(std::move(it));
    }
    for (auto& [id, name, args] : p.calls) {
        Value it = obj();
        it.set("id", "fc_" + id); it.set("type", "function_call"); it.set("status", "completed"); it.set("arguments", args); it.set("call_id", "call_" + id); it.set("name", name);
        out.push(std::move(it));
    }
    long long t = (long long)std::time(nullptr);
    Value res = obj();
    res.set("completed_at", t); res.set("created_at", t); res.set("id", "resp_" + rnd(32)); res.set("model", p.model); res.set("object", "response");
    res.set("output", std::move(out)); res.set("status", "completed"); res.set("usage", resp_usage(p));
    return res;
}

// ------------------------------------------------------------------------------------------------- stream translators
namespace {

struct ChunkView {
    std::string reasoning, content, finish;
    struct Tc { long long index; std::string id, name, args; bool has_name; };
    std::vector<Tc> calls;
    bool has_choice = false, has_usage = false;
    long long prompt = 0, completion = 0, cached = 0;
    std::string id, model;
    Value timings;
};
ChunkView view_of(const Value& c) {
    ChunkView v;
    v.id = get_str(c, "id");
    v.model = get_str(c, "model");
    if (const Value* ch = get(c, "choices"); ch && ch->is_array() && ch->size() > 0) {
        v.has_choice = true;
        const Value& c0 = ch->items()[0];
        v.finish = get_str(c0, "finish_reason");
        if (const Value* d = get(c0, "delta")) {
            v.reasoning = get_str(*d, "reasoning_content");
            v.content = get_str(*d, "content");
            if (const Value* tcs = get(*d, "tool_calls"); tcs && tcs->is_array())
                for (auto& tc : tcs->items()) {
                    ChunkView::Tc t{0, get_str(tc, "id"), "", "", false};
                    if (const Value* i = get(tc, "index")) t.index = i->as_int();
                    if (const Value* fn = get(tc, "function")) { t.name = get_str(*fn, "name"); t.args = get_str(*fn, "arguments"); t.has_name = get(*fn, "name") != nullptr; }
                    v.calls.push_back(std::move(t));
                }
        }
    }
    if (const Value* tm = get(c, "timings")) v.timings = *tm;
    if (const Value* u = get(c, "usage"); u && u->is_object()) {
        v.has_usage = true;
        if (const Value* x = get(*u, "prompt_tokens")) v.prompt = x->as_int();
        if (const Value* x = get(*u, "completion_tokens")) v.completion = x->as_int();
        if (const Value* d = get(*u, "prompt_tokens_details")) if (const Value* x = get(*d, "cached_tokens")) v.cached = x->as_int();
    }
    return v;
}

struct AnthropicXlate : StreamXlate {
    int prompt_tokens;
    bool started = false, finished = false;
    int next_block = 0, think_idx = -1, text_idx = -1;
    std::map<long long, int> tool_idx;
    std::string finish_reason;
    long long out_tokens = 0;
    explicit AnthropicXlate(int pt) : prompt_tokens(pt) {}

    std::vector<std::string> feed(const Value& chunk) override {
        std::vector<std::string> ev;
        ChunkView v = view_of(chunk);
        if (!started) {
            started = true;
            Value u = obj(); u.set("cache_read_input_tokens", 0); u.set("input_tokens", prompt_tokens); u.set("output_tokens", 0);
            Value m = obj();
            m.set("id", v.id); m.set("type", "message"); m.set("role", "assistant"); m.set("content", arr()); m.set("model", v.model);
            m.set("stop_reason", Value()); m.set("stop_sequence", Value()); m.set("usage", std::move(u));
            Value d = obj(); d.set("type", "message_start"); d.set("message", std::move(m));
            ev.push_back(sse("message_start", d));
        }
        auto start_block = [&](int idx, Value block) {
            Value d = obj(); d.set("type", "content_block_start"); d.set("index", idx); d.set("content_block", std::move(block));
            ev.push_back(sse("content_block_start", d));
        };
        auto delta = [&](int idx, Value dl) {
            Value d = obj(); d.set("type", "content_block_delta"); d.set("index", idx); d.set("delta", std::move(dl));
            ev.push_back(sse("content_block_delta", d));
        };
        if (!v.reasoning.empty()) {
            if (think_idx < 0) { think_idx = next_block++; Value b = obj(); b.set("type", "thinking"); b.set("thinking", ""); start_block(think_idx, std::move(b)); }
            Value dl = obj(); dl.set("type", "thinking_delta"); dl.set("thinking", v.reasoning); delta(think_idx, std::move(dl));
        }
        if (!v.content.empty()) {
            if (text_idx < 0) { text_idx = next_block++; Value b = obj(); b.set("type", "text"); b.set("text", ""); start_block(text_idx, std::move(b)); }
            Value dl = obj(); dl.set("type", "text_delta"); dl.set("text", v.content); delta(text_idx, std::move(dl));
        }
        for (auto& tc : v.calls) {
            if (tc.has_name && !tc.name.empty() && !tool_idx.count(tc.index)) {
                int idx = next_block++;
                tool_idx[tc.index] = idx;
                Value b = obj(); b.set("type", "tool_use"); b.set("id", tc.id); b.set("name", tc.name);
                start_block(idx, std::move(b));
            }
            if (!tc.args.empty() && tool_idx.count(tc.index)) {
                Value dl = obj(); dl.set("type", "input_json_delta"); dl.set("partial_json", tc.args);
                delta(tool_idx[tc.index], std::move(dl));
            }
        }
        if (!v.finish.empty()) finish_reason = v.finish;
        if (v.has_usage && !v.has_choice) { out_tokens = v.completion; for (auto& s : finish()) ev.push_back(std::move(s)); }
        return ev;
    }

    std::vector<std::string> finish() override {
        std::vector<std::string> ev;
        if (finished || !started) return ev;
        finished = true;
        auto stop = [&](int idx) { Value d = obj(); d.set("type", "content_block_stop"); d.set("index", idx); ev.push_back(sse("content_block_stop", d)); };
        if (think_idx >= 0) {
            Value dl = obj(); dl.set("type", "signature_delta"); dl.set("signature", "");
            Value d = obj(); d.set("type", "content_block_delta"); d.set("index", think_idx); d.set("delta", std::move(dl));
            ev.push_back(sse("content_block_delta", d));
            stop(think_idx);
        }
        if (text_idx >= 0) stop(text_idx);
        for (auto& [k, idx] : tool_idx) stop(idx);
        Value dl = obj(); dl.set("stop_reason", anthropic_stop(finish_reason)); dl.set("stop_sequence", stopping_word.empty() ? Value() : Value(stopping_word));
        Value u = obj(); u.set("output_tokens", out_tokens);
        Value d = obj(); d.set("type", "message_delta"); d.set("delta", std::move(dl)); d.set("usage", std::move(u));
        ev.push_back(sse("message_delta", d));
        Value ms = obj(); ms.set("type", "message_stop");
        ev.push_back(sse("message_stop", ms));
        return ev;
    }
};

struct ResponsesXlate : StreamXlate {
    std::string resp_id = "resp_" + rnd(32), rs_id = "rs_" + rnd(32), msg_id = "msg_" + rnd(32), model;
    bool started = false, finished = false;
    int next_out = 0, think_out = -1, text_out = -1;
    std::string reasoning, content;
    struct Call { int out; std::string id, name, args; };
    std::map<long long, Call> calls;
    ChunkView last;
    Parsed usage;
    Value timings;

    std::vector<std::string> feed(const Value& chunk) override {
        std::vector<std::string> ev;
        ChunkView v = view_of(chunk);
        model = v.model;
        auto resp_obj = [&](const char* status) { Value r = obj(); r.set("id", resp_id); r.set("object", "response"); r.set("status", status); return r; };
        if (!started) {
            started = true;
            Value a = obj(); a.set("type", "response.created"); a.set("response", resp_obj("in_progress")); ev.push_back(sse("response.created", a));
            Value b = obj(); b.set("type", "response.in_progress"); b.set("response", resp_obj("in_progress")); ev.push_back(sse("response.in_progress", b));
        }
        if (!v.reasoning.empty()) {
            if (think_out < 0) {
                think_out = next_out++;
                Value it = obj(); it.set("id", rs_id); it.set("summary", arr()); it.set("type", "reasoning"); it.set("content", arr()); it.set("encrypted_content", ""); it.set("status", "in_progress");
                Value d = obj(); d.set("type", "response.output_item.added"); d.set("output_index", think_out); d.set("item", std::move(it));
                ev.push_back(sse("response.output_item.added", d));
            }
            reasoning += v.reasoning;
            Value d = obj(); d.set("type", "response.reasoning_text.delta"); d.set("output_index", think_out); d.set("content_index", 0); d.set("delta", v.reasoning); d.set("item_id", rs_id);
            ev.push_back(sse("response.reasoning_text.delta", d));
        }
        if (!v.content.empty()) {
            if (text_out < 0) {
                text_out = next_out++;
                Value it = obj(); it.set("content", arr()); it.set("id", msg_id); it.set("role", "assistant"); it.set("status", "in_progress"); it.set("type", "message");
                Value d = obj(); d.set("type", "response.output_item.added"); d.set("output_index", text_out); d.set("item", std::move(it));
                ev.push_back(sse("response.output_item.added", d));
                Value part = obj(); part.set("type", "output_text"); part.set("text", "");
                Value d2 = obj(); d2.set("type", "response.content_part.added"); d2.set("item_id", msg_id); d2.set("output_index", text_out); d2.set("content_index", 0); d2.set("part", std::move(part));
                ev.push_back(sse("response.content_part.added", d2));
            }
            content += v.content;
            Value d = obj(); d.set("type", "response.output_text.delta"); d.set("item_id", msg_id); d.set("output_index", text_out); d.set("content_index", 0); d.set("delta", v.content);
            ev.push_back(sse("response.output_text.delta", d));
        }
        for (auto& tc : v.calls) {
            if (tc.has_name && !tc.name.empty() && !calls.count(tc.index)) {
                Call c{next_out++, tc.id, tc.name, ""};
                calls[tc.index] = c;
                Value it = obj(); it.set("id", "fc_" + tc.id); it.set("arguments", ""); it.set("call_id", "call_" + tc.id); it.set("name", tc.name); it.set("type", "function_call"); it.set("status", "in_progress");
                Value d = obj(); d.set("type", "response.output_item.added"); d.set("output_index", c.out); d.set("item", std::move(it));
                ev.push_back(sse("response.output_item.added", d));
            }
            auto it = calls.find(tc.index);
            if (!tc.args.empty() && it != calls.end()) {
                it->second.args += tc.args;
                Value d = obj(); d.set("type", "response.function_call_arguments.delta"); d.set("delta", tc.args); d.set("item_id", "fc_" + it->second.id); d.set("output_index", it->second.out);
                ev.push_back(sse("response.function_call_arguments.delta", d));
            }
        }
        if (v.has_usage && !v.has_choice) { timings = v.timings; usage.prompt = v.prompt; usage.completion = v.completion; usage.cached = v.cached; for (auto& s : finish()) ev.push_back(std::move(s)); }
        return ev;
    }

    std::vector<std::string> finish() override {
        std::vector<std::string> ev;
        if (finished || !started) return ev;
        finished = true;
        Value output = arr();
        if (think_out >= 0) {
            Value t = obj(); t.set("text", reasoning); t.set("type", "reasoning_text");
            Value c = arr(); c.push(std::move(t));
            Value it = obj(); it.set("id", rs_id); it.set("summary", arr()); it.set("type", "reasoning"); it.set("content", std::move(c)); it.set("encrypted_content", "");
            Value d = obj(); d.set("type", "response.output_item.done"); d.set("output_index", think_out); d.set("item", it);
            ev.push_back(sse("response.output_item.done", d));
            output.push(std::move(it));
        }
        if (text_out >= 0) {
            Value d1 = obj(); d1.set("type", "response.output_text.done"); d1.set("item_id", msg_id); d1.set("output_index", text_out); d1.set("content_index", 0); d1.set("text", content);
            ev.push_back(sse("response.output_text.done", d1));
            Value part = obj(); part.set("type", "output_text"); part.set("annotations", arr()); part.set("logprobs", arr()); part.set("text", content);
            Value d2 = obj(); d2.set("type", "response.content_part.done"); d2.set("item_id", msg_id); d2.set("output_index", text_out); d2.set("content_index", 0); d2.set("part", part);
            ev.push_back(sse("response.content_part.done", d2));
            Value pc = arr(); pc.push(part);
            Value it = obj(); it.set("type", "message"); it.set("status", "completed"); it.set("id", msg_id); it.set("content", std::move(pc)); it.set("role", "assistant");
            Value d3 = obj(); d3.set("type", "response.output_item.done"); d3.set("output_index", text_out); d3.set("item", it);
            ev.push_back(sse("response.output_item.done", d3));
            output.push(std::move(it));
        }
        for (auto& [k, c] : calls) {
            Value it = obj(); it.set("id", "fc_" + c.id); it.set("type", "function_call"); it.set("status", "completed"); it.set("arguments", c.args); it.set("call_id", "call_" + c.id); it.set("name", c.name);
            Value d = obj(); d.set("type", "response.output_item.done"); d.set("output_index", c.out); d.set("item", it);
            ev.push_back(sse("response.output_item.done", d));
            output.push(std::move(it));
        }
        Value r = obj();
        r.set("id", resp_id); r.set("object", "response"); r.set("created_at", (long long)std::time(nullptr)); r.set("status", "completed"); r.set("model", model);
        r.set("output", std::move(output)); r.set("usage", resp_usage(usage));
        Value d = obj(); d.set("type", "response.completed"); d.set("response", std::move(r));
        if (!timings.is_null()) d.set("timings", timings);
        ev.push_back(sse("response.completed", d));
        return ev;
    }
};
}  // namespace

std::unique_ptr<StreamXlate> make_stream_xlate(ApiFmt fmt, int prompt_tokens) {
    if (fmt == ApiFmt::Anthropic) return std::make_unique<AnthropicXlate>(prompt_tokens);
    if (fmt == ApiFmt::Responses) return std::make_unique<ResponsesXlate>();
    return nullptr;
}

}  // namespace qv::server
