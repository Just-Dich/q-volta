// Completion / chat / infill endpoints (native llama-server and OpenAI shapes).
#include <algorithm>
#include <random>

#include "server/app.h"

namespace qv::server {

namespace {

std::string rand_alnum(int n) {
    static std::mutex m;
    static std::mt19937_64 rng{std::random_device{}()};
    static const char* cs = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::lock_guard<std::mutex> g(m);
    std::string s;
    for (int i = 0; i < n; ++i) s += cs[rng() % 62];
    return s;
}

// Splits "<reasoning></think>answer" model output into reasoning_content / content (deepseek reasoning format).
class ReasoningSplitter {
   public:
    ReasoningSplitter(bool in_reasoning, std::string start, std::string end) : in_(in_reasoning), started_(in_reasoning), start_(std::move(start)), end_(std::move(end)) {}
    void feed(const std::string& t, std::string& reasoning, std::string& content, bool final_) {
        std::string buf = hold_ + t;
        hold_.clear();
        for (;;) {
            if (in_) {
                if (lead_r_) {
                    size_t b = buf.find_first_not_of('\n');
                    buf.erase(0, b == std::string::npos ? buf.size() : b);
                    if (!buf.empty()) lead_r_ = false;
                }
                size_t p = buf.find(end_);
                if (p != std::string::npos) {
                    reasoning += buf.substr(0, p);
                    buf.erase(0, p + end_.size());
                    in_ = false;
                    lead_ = true;
                    continue;
                }
                size_t keep = 0;
                if (!final_)
                    for (size_t k = std::min(buf.size(), end_.size() - 1); k > 0; --k)
                        if (buf.compare(buf.size() - k, k, end_, 0, k) == 0) { keep = k; break; }
                reasoning += buf.substr(0, buf.size() - keep);
                hold_ = buf.substr(buf.size() - keep);
                return;
            }
            if (!started_) {
                size_t b = buf.find_first_not_of(" \n\t");
                std::string rest = b == std::string::npos ? "" : buf.substr(b);
                if (rest.compare(0, start_.size(), start_) == 0 && rest.size() >= start_.size()) { buf = rest.substr(start_.size()); in_ = true; started_ = true; lead_r_ = true; continue; }
                if (!final_ && start_.compare(0, rest.size(), rest) == 0) { hold_ = buf; return; }  // could still become the start tag
                started_ = true;
            }
            if (lead_) {
                size_t b = buf.find_first_not_of('\n');
                buf.erase(0, b == std::string::npos ? buf.size() : b);
                if (!buf.empty()) lead_ = false;
            }
            content += buf;
            return;
        }
    }

   private:
    bool in_, started_, lead_ = false, lead_r_ = false;
    std::string start_, end_, hold_;
};

}  // namespace

void App::respond_json(http::Conn& c, int status, const Value& v, const http::Headers& extra) { c.respond(status, "application/json; charset=utf-8", v.dump(), extra); }

static Value apply_response_fields(const Value& rf, Value&& res) {
    if (!rf.is_array() || rf.size() == 0) return std::move(res);
    Value o = Value::object();
    for (auto& f : rf.items()) if (f.is_string()) if (const Value* v = res.find(f.as_string())) o.set(f.as_string(), *v);
    return o;
}

void App::handle_completion(const http::Request& req, http::Conn& c, Kind kind) {
    Value body;
    try { body = jd::parse(req.body); }
    catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object() && kind == Kind::OaiChat) throw_error(400, "'messages' is required");

    // ---- prompts (native /completion accepts an array of prompts: one result per prompt, non-streaming)
    std::vector<std::vector<int>> prompts;
    Task base = parse_task(body, kind);
    if (kind == Kind::OaiChat) prompts.push_back(base.prompt);
    else {
        const Value* p = body.find("prompt");
        if (!p || p->is_null()) throw_error(400, "[json.exception.out_of_range.403] key 'prompt' not found");
        bool multi = p->is_array() && p->size() > 1;
        if (multi) for (auto& x : p->items()) if (!(x.is_string() || x.is_array())) multi = false;
        if (multi) { for (auto& x : p->items()) prompts.push_back(tokenize_prompt_value(x)); }
        else prompts.push_back(tokenize_prompt_value(*p));
        if (!multi && p->is_string()) base.prompt_text = p->as_string();
    }
    if (prompts.size() > 1 && base.stream) throw_error(400, "streaming is not supported with an array of prompts");
    for (auto& pr : prompts) {
        if (pr.empty()) throw_error(400, "the prompt is empty");
        if ((int)pr.size() >= n_ctx) {
            ApiError e{400, "request (" + std::to_string(pr.size()) + " tokens) exceeds the available context size (" + std::to_string(n_ctx) + " tokens), try increasing it", "exceed_context_size_error", Value::object()};
            e.extra.set("n_prompt_tokens", (long long)pr.size());
            e.extra.set("n_ctx", n_ctx);
            throw e;
        }
    }

    const bool chat = kind == Kind::OaiChat, oai = kind != Kind::Native;
    const std::string id = oai ? "chatcmpl-" + rand_alnum(32) : "";
    const long long created = (long long)std::time(nullptr);
    const std::string fp = build_info;
    auto alive = [&] { return !c.peer_closed(); };
    const bool split = chat && base.reasoning_format != "none";

    auto timings_of = [&](const GenResult& r) { return timings_json(r); };
    auto base_chunk = [&](const char* object) {
        Value o2 = Value::object();
        o2.set("choices", Value::array());
        o2.set("created", created);
        o2.set("id", id);
        o2.set("model", model_id);
        o2.set("system_fingerprint", fp);
        o2.set("object", object);
        return o2;
    };
    auto finish_reason_of = [&](const GenResult& r) { return (r.stop_type == "eos" || r.stop_type == "word") ? "stop" : "length"; };
    auto usage_of = [&](int n_prompt, int n_cache, int n_dec) {
        Value u = Value::object();
        u.set("completion_tokens", n_dec);
        u.set("prompt_tokens", n_prompt);
        u.set("total_tokens", n_dec + n_prompt);
        Value d = Value::object();
        d.set("cached_tokens", n_cache);
        u.set("prompt_tokens_details", std::move(d));
        return u;
    };
    auto native_final = [&](const Task& t, const GenResult& r, int n_prompt, bool streamed) {
        Value o2 = Value::object();
        o2.set("index", 0);
        o2.set("content", streamed ? std::string() : r.content);
        o2.set("tokens", streamed || !t.return_tokens ? Value::array() : [&] { Value a = Value::array(); for (int x : r.tokens) a.push(x); return a; }());
        o2.set("id_slot", r.id_slot);
        o2.set("stop", true);
        o2.set("model", model_id);
        o2.set("tokens_predicted", r.n_decoded);
        o2.set("tokens_evaluated", n_prompt);
        o2.set("generation_settings", gen_settings(t, 0));
        o2.set("prompt", t.prompt_text);
        o2.set("has_new_line", r.has_new_line);
        o2.set("truncated", r.truncated);
        o2.set("stop_type", r.stop_type);
        o2.set("stopping_word", r.stopping_word);
        o2.set("tokens_cached", r.tokens_cached);
        o2.set("timings", timings_of(r));
        if (!streamed && !r.probs.empty()) o2.set("completion_probabilities", probs_json(r.probs, t.post_probs));
        return apply_response_fields(t.response_fields, std::move(o2));
    };

    // ======================================================================================= streaming
    if (base.stream) {
        Task t = base;
        t.prompt = prompts[0];
        bool started = false, ok = true;
        auto begin = [&](int) {
            if (started) return;
            started = true;
            ok = c.begin_stream(200, "text/event-stream");
        };
        auto sse = [&](const Value& v) { if (ok) ok = c.write_chunk("data: " + v.dump() + "\n\n"); return ok; };
        GenResult total;
        int tot_dec = 0, tot_cache = 0;
        std::vector<GenResult> results;
        for (int ci = 0; ci < (oai ? base.n_choices : 1) && ok; ++ci) {
            Task tc = t;
            if (ci > 0 && tc.sp.has_seed) tc.sp.seed += ci;
            ReasoningSplitter rs(t.thinking_open, t.think_start, t.think_end);
            bool first = true;
            auto role_chunk = [&] {
                Value o2 = base_chunk("chat.completion.chunk");
                Value d = Value::object();
                d.set("role", "assistant");
                d.set("content", Value());
                Value ch = Value::object();
                ch.set("finish_reason", Value());
                ch.set("index", ci);
                ch.set("delta", std::move(d));
                o2.set("choices", Value::array()).push(std::move(ch));
                return o2;
            };
            auto on_chunk = [&](const Chunk& ck) {
                if (!ok) return false;
                if (kind == Kind::Native || kind == Kind::Infill) {
                    Value o2 = Value::object();
                    o2.set("index", 0);
                    o2.set("content", ck.text);
                    o2.set("tokens", [&] { Value a = Value::array(); for (int x : ck.tokens) a.push(x); return a; }());
                    o2.set("stop", false);
                    o2.set("id_slot", -1);
                    o2.set("tokens_predicted", ck.n_decoded);
                    o2.set("tokens_evaluated", (int)t.prompt.size());
                    if (!ck.probs.empty()) o2.set("completion_probabilities", probs_json(ck.probs, t.post_probs));
                    return sse(o2);
                }
                if (chat) {
                    std::string rr, cc;
                    if (split) rs.feed(ck.text, rr, cc, false); else cc = ck.text;
                    if (first) { first = false; if (!sse(role_chunk())) return false; }
                    if (rr.empty() && cc.empty() && ck.probs.empty()) return true;
                    Value o2 = base_chunk("chat.completion.chunk");
                    Value d = Value::object();
                    if (!rr.empty()) d.set("reasoning_content", rr);
                    if (!cc.empty() || rr.empty()) d.set("content", cc);
                    Value ch = Value::object();
                    ch.set("finish_reason", Value());
                    ch.set("index", ci);
                    ch.set("delta", std::move(d));
                    if (!ck.probs.empty()) { Value lp = Value::object(); lp.set("content", probs_json(ck.probs, t.post_probs)); ch.set("logprobs", std::move(lp)); }
                    o2.set("choices", Value::array()).push(std::move(ch));
                    return sse(o2);
                }
                Value o2 = base_chunk("text_completion");
                Value ch = Value::object();
                ch.set("text", ck.text);
                ch.set("index", ci);
                if (!ck.probs.empty()) { Value lp = Value::object(); lp.set("content", probs_json(ck.probs, t.post_probs)); ch.set("logprobs", std::move(lp)); } else ch.set("logprobs", Value());
                ch.set("finish_reason", Value());
                o2.set("choices", Value::array()).push(std::move(ch));
                return sse(o2);
            };
            GenResult r = generate(tc, alive, on_chunk, begin);
            if (r.aborted) { c.close_after = true; return; }
            begin(0);
            if (chat && first) { first = false; sse(role_chunk()); }
            if (chat && split) {  // flush text held back by the splitter
                std::string rr, cc;
                rs.feed("", rr, cc, true);
                if (!rr.empty() || !cc.empty()) {
                    Value o2 = base_chunk("chat.completion.chunk");
                    Value d = Value::object();
                    if (!rr.empty()) d.set("reasoning_content", rr);
                    if (!cc.empty() || rr.empty()) d.set("content", cc);
                    Value ch = Value::object();
                    ch.set("finish_reason", Value());
                    ch.set("index", ci);
                    ch.set("delta", std::move(d));
                    o2.set("choices", Value::array()).push(std::move(ch));
                    sse(o2);
                }
            }
            tot_dec += r.n_decoded;
            tot_cache = r.n_cache;
            results.push_back(r);
            total = r;
            if (kind == Kind::Native || kind == Kind::Infill) {
                sse(native_final(tc, r, (int)t.prompt.size(), true));
            } else if (chat) {
                Value o2 = base_chunk("chat.completion.chunk");
                Value ch = Value::object();
                ch.set("finish_reason", finish_reason_of(r));
                ch.set("index", ci);
                ch.set("delta", Value::object());
                o2.set("choices", Value::array()).push(std::move(ch));
                bool last = ci + 1 == base.n_choices;
                if (last && !t.include_usage) o2.set("timings", timings_of(r));
                sse(o2);
            } else {
                Value o2 = base_chunk("text_completion");
                Value ch = Value::object();
                ch.set("text", "");
                ch.set("index", ci);
                ch.set("logprobs", Value());
                ch.set("finish_reason", finish_reason_of(r));
                o2.set("choices", Value::array()).push(std::move(ch));
                if (ci + 1 == base.n_choices) { o2.set("usage", usage_of((int)t.prompt.size(), tot_cache, tot_dec)); o2.set("timings", timings_of(r)); }
                sse(o2);
            }
        }
        begin(0);
        if (chat && base.include_usage && ok) {
            Value o2 = base_chunk("chat.completion.chunk");
            o2.set("usage", usage_of((int)t.prompt.size(), tot_cache, tot_dec));
            o2.set("timings", timings_of(total));
            sse(o2);
        }
        if (oai && ok) ok = c.write_chunk("data: [DONE]\n\n");
        if (ok) c.end_chunks();
        else c.close_after = true;
        return;
    }

    // ======================================================================================= non-streaming
    std::vector<Value> results;
    std::vector<GenResult> grs;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        Task t = base;
        t.prompt = prompts[pi];
        if (kind == Kind::Native && prompts.size() > 1) t.prompt_text = tok.decode(prompts[pi]);
        std::vector<GenResult> rs_;
        for (int ci = 0; ci < (oai ? base.n_choices : 1); ++ci) {
            Task tc = t;
            if (ci > 0 && tc.sp.has_seed) tc.sp.seed += ci;
            GenResult r = generate(tc, alive, [](const Chunk&) { return true; }, [](int) {});
            if (r.aborted) { c.close_after = true; return; }
            rs_.push_back(std::move(r));
        }
        if (!oai) { results.push_back(native_final(t, rs_[0], (int)t.prompt.size(), false)); grs.push_back(rs_[0]); continue; }
        // OpenAI shapes
        Value resp = base_chunk(chat ? "chat.completion" : "text_completion");
        Value choices = Value::array();
        int tot_dec = 0;
        for (size_t ci = 0; ci < rs_.size(); ++ci) {
            GenResult& r = rs_[ci];
            tot_dec += r.n_decoded;
            Value ch = Value::object();
            if (chat) {
                std::string reasoning, content = r.content;
                if (split) {
                    ReasoningSplitter rsp(t.thinking_open, t.think_start, t.think_end);
                    content.clear();
                    rsp.feed(r.content, reasoning, content, true);
                    if (base.reasoning_format == "deepseek-legacy") content = (t.thinking_open ? t.think_start + "\n" : "") + r.content;
                }
                Value m = Value::object();
                m.set("role", "assistant");
                m.set("content", content);
                if (!reasoning.empty()) m.set("reasoning_content", reasoning);
                ch.set("finish_reason", finish_reason_of(r));
                ch.set("index", (int)ci);
                ch.set("message", std::move(m));
                if (!r.probs.empty()) { Value lp = Value::object(); lp.set("content", probs_json(r.probs, t.post_probs)); ch.set("logprobs", std::move(lp)); }
            } else {
                ch.set("text", r.content);
                ch.set("index", (int)ci);
                if (!r.probs.empty()) { Value lp = Value::object(); lp.set("content", probs_json(r.probs, t.post_probs)); ch.set("logprobs", std::move(lp)); } else ch.set("logprobs", Value());
                ch.set("finish_reason", finish_reason_of(r));
            }
            choices.push(std::move(ch));
        }
        resp.set("choices", std::move(choices));
        resp.set("usage", usage_of((int)t.prompt.size(), rs_[0].n_cache, tot_dec));
        GenResult tsum = rs_.back();
        resp.set("timings", timings_of(tsum));
        results.push_back(std::move(resp));
        grs.push_back(rs_.back());
    }
    if (results.size() == 1) respond_json(c, 200, results[0]);
    else { Value a = Value::array(); for (auto& r : results) a.push(std::move(r)); respond_json(c, 200, a); }
}

void App::handle_infill(const http::Request& req, http::Conn& c) {
    if (fim_pre < 0 || fim_suf < 0 || fim_mid < 0) {
        std::string err;
        if (fim_pre < 0) err += "prefix token is missing. ";
        if (fim_suf < 0) err += "suffix token is missing. ";
        if (fim_mid < 0) err += "middle token is missing. ";
        throw_error(501, "Infill is not supported by this model: " + err, "not_supported_error");
    }
    Value body;
    try { body = jd::parse(req.body); }
    catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object()) throw_error(400, "request body must be a JSON object");
    if (const Value* p = body.find("prompt"); p && !p->is_string()) throw_error(400, "\"prompt\" must be a string");
    if (!body.find("input_prefix")) throw_error(400, "\"input_prefix\" is required");
    if (!body.find("input_suffix")) throw_error(400, "\"input_suffix\" is required");
    const Value* extra = body.find("input_extra");
    if (extra && !extra->is_array()) throw_error(400, "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}");
    if (extra)
        for (auto& ch : extra->items()) {
            const Value* tx = ch.find("text");
            if (!tx || !tx->is_string()) throw_error(400, "extra_context chunk must contain a \"text\" field with a string value");
            const Value* fn = ch.find("filename");
            if (fn && !fn->is_string()) throw_error(400, "extra_context chunk's \"filename\" field must be a string");
        }
    auto tokz = [&](const std::string& s) { return tok.encode(s, false); };
    auto str_of = [&](const char* k) { const Value* v = body.find(k); return v && v->is_string() ? v->as_string() : std::string(); };
    std::vector<int> prompt_toks = tokz(str_of("prompt")), tp = tokz(str_of("input_prefix")), ts = tokz(str_of("input_suffix")), ex;
    if (fim_rep >= 0) { ex.push_back(fim_rep); auto r = tokz("myproject\n"); ex.insert(ex.end(), r.begin(), r.end()); }
    if (extra)
        for (auto& ch : extra->items()) {
            std::string text = ch.find("text")->as_string(), fname = ch.find("filename") ? ch.find("filename")->as_string() : "tmp";
            if (fim_sep >= 0) { ex.push_back(fim_sep); auto f = tokz(fname + "\n"); ex.insert(ex.end(), f.begin(), f.end()); }
            else { auto f = tokz("\n\n--- snippet ---\n\n"); ex.insert(ex.end(), f.begin(), f.end()); }
            auto tt = tokz(text);
            ex.insert(ex.end(), tt.begin(), tt.end());
        }
    if (fim_sep >= 0) { ex.push_back(fim_sep); auto f = tokz("filename\n"); ex.insert(ex.end(), f.begin(), f.end()); }
    int n_batch = o.batch > 0 ? o.batch : 2048;
    Task t = parse_task(body, Kind::Infill);
    int n_prefix_take = std::min<int>((int)tp.size(), 3 * (n_batch / 4));
    int n_suffix_take = std::min<int>((int)ts.size(), std::max<int>(0, (n_batch / 4) - (2 + (int)prompt_toks.size())));
    int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - n_batch - 2 * t.n_predict), (int)ex.size());
    tp.erase(tp.begin(), tp.begin() + (tp.size() - n_prefix_take));
    ts.resize(n_suffix_take);
    tp.insert(tp.begin(), fim_pre);
    tp.insert(tp.end(), prompt_toks.begin(), prompt_toks.end());
    ts.insert(ts.begin(), fim_suf);
    std::vector<int> in = tp;
    in.insert(in.begin(), ex.end() - n_extra_take, ex.end());
    in.insert(in.end(), ts.begin(), ts.end());
    in.push_back(fim_mid);
    t.prompt = in;
    t.prompt_text = tok.decode(in);
    if ((int)in.size() >= n_ctx) {
        ApiError e{400, "request (" + std::to_string(in.size()) + " tokens) exceeds the available context size (" + std::to_string(n_ctx) + " tokens), try increasing it", "exceed_context_size_error", Value::object()};
        e.extra.set("n_prompt_tokens", (long long)in.size());
        e.extra.set("n_ctx", n_ctx);
        throw e;
    }
    auto alive = [&] { return !c.peer_closed(); };
    if (t.stream) {
        bool ok = true, started = false;
        auto begin = [&](int) { if (!started) { started = true; ok = c.begin_stream(200, "text/event-stream"); } };
        auto sse = [&](const Value& v) { if (ok) ok = c.write_chunk("data: " + v.dump() + "\n\n"); return ok; };
        GenResult r = generate(t, alive, [&](const Chunk& ck) {
            Value o2 = Value::object();
            o2.set("index", 0); o2.set("content", ck.text);
            Value a = Value::array(); for (int x : ck.tokens) a.push(x);
            o2.set("tokens", std::move(a)); o2.set("stop", false); o2.set("id_slot", -1); o2.set("tokens_predicted", ck.n_decoded); o2.set("tokens_evaluated", (int)t.prompt.size());
            return sse(o2);
        }, begin);
        if (r.aborted) { c.close_after = true; return; }
        begin(0);
        Value o2 = Value::object();
        o2.set("index", 0); o2.set("content", ""); o2.set("tokens", Value::array()); o2.set("id_slot", r.id_slot); o2.set("stop", true); o2.set("model", model_id);
        o2.set("tokens_predicted", r.n_decoded); o2.set("tokens_evaluated", (int)t.prompt.size()); o2.set("generation_settings", gen_settings(t, 0)); o2.set("prompt", t.prompt_text);
        o2.set("has_new_line", r.has_new_line); o2.set("truncated", r.truncated); o2.set("stop_type", r.stop_type); o2.set("stopping_word", r.stopping_word);
        o2.set("tokens_cached", r.tokens_cached); o2.set("timings", timings_json(r));
        sse(o2);
        if (ok) c.end_chunks(); else c.close_after = true;
        return;
    }
    GenResult r = generate(t, alive, [](const Chunk&) { return true; }, [](int) {});
    if (r.aborted) { c.close_after = true; return; }
    Value o2 = Value::object();
    o2.set("index", 0); o2.set("content", r.content);
    Value a = Value::array(); if (t.return_tokens) for (int x : r.tokens) a.push(x);
    o2.set("tokens", std::move(a)); o2.set("id_slot", r.id_slot); o2.set("stop", true); o2.set("model", model_id);
    o2.set("tokens_predicted", r.n_decoded); o2.set("tokens_evaluated", (int)t.prompt.size()); o2.set("generation_settings", gen_settings(t, 0)); o2.set("prompt", t.prompt_text);
    o2.set("has_new_line", r.has_new_line); o2.set("truncated", r.truncated); o2.set("stop_type", r.stop_type); o2.set("stopping_word", r.stopping_word);
    o2.set("tokens_cached", r.tokens_cached); o2.set("timings", timings_json(r));
    if (!r.probs.empty()) o2.set("completion_probabilities", probs_json(r.probs, t.post_probs));
    respond_json(c, 200, o2);
}

}  // namespace qv::server
