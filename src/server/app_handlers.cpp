// Completion / chat / infill endpoints (native llama-server and OpenAI shapes).
#include <algorithm>
#include <random>
#include <thread>

#include "server/app.h"
#include "jinja/chat.h"

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

// Chat output post-processing (src/jinja chat parser): reasoning extraction + tool calls, complete and streaming.
chat::ParseParams parse_params_of(const Task& t) {
    chat::ParseParams p;
    p.format = t.tool_fmt;
    p.reasoning = t.reasoning_format == "none" ? chat::ReasoningFormat::None : t.reasoning_format == "deepseek-legacy" ? chat::ReasoningFormat::DeepseekLegacy : chat::ReasoningFormat::Deepseek;
    p.thinking_forced_open = t.thinking_open;
    p.parse_tool_calls = t.parse_tools;
    p.think_start = t.think_start;
    p.think_end = t.think_end;
    p.tools = t.tools.is_array() ? &t.tools : nullptr;
    return p;
}

Value tool_delta_json(const chat::ToolCallDelta& d, const std::string& id) {
    Value tc = Value::object();
    tc.set("index", (long long)d.index);
    if (d.is_new) { tc.set("id", id); tc.set("type", "function"); }
    Value f = Value::object();
    if (d.is_new) f.set("name", d.name);
    f.set("arguments", d.arguments);
    tc.set("function", std::move(f));
    return tc;
}

}  // namespace

void App::respond_json(http::Conn& c, int status, const Value& v, const http::Headers& extra) { c.respond(status, "application/json; charset=utf-8", v.dump(), extra); }

static Value apply_response_fields(const Value& rf, Value&& res) {
    if (!rf.is_array() || rf.size() == 0) return std::move(res);
    Value o = Value::object();
    for (auto& f : rf.items()) if (f.is_string()) if (const Value* v = res.find(f.as_string())) o.set(f.as_string(), *v);
    return o;
}

void App::handle_completion(const http::Request& req, http::Conn& c, Kind kind, ApiFmt fmt, const Value* body_in) {
    Value body;
    if (body_in) body = *body_in;
    else {
        try { body = jd::parse(req.body); }
        catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    }
    if (!body.is_object() && kind == Kind::OaiChat) throw_error(400, "'messages' is required");

    auto alive = [&] { return !c.peer_closed(); };
    std::string req_model;
    if (body.is_object()) if (const Value* mv = body.find("model"); mv && mv->is_string()) req_model = mv->as_string();
    long lease_ticket = acquire_model_lease(req_model, alive);
    if (lease_ticket == -2) { c.close_after = true; return; }
    struct ModelLeaseGuard { App& a; long t; ~ModelLeaseGuard() { if (t >= 0) a.release_model_lease(t); } } lease_guard{*this, lease_ticket};

    // ---- prompts (native /completion accepts an array of prompts: one result per prompt, non-streaming)
    std::vector<std::vector<int>> prompts;
    Task base = parse_task(body, kind);
    if (fmt != ApiFmt::Oai) base.include_usage = true;  // the translators emit their closing events on the usage chunk
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
    struct CtlGuard { App* a; std::string id; ~CtlGuard() { if (!id.empty()) { std::lock_guard<std::mutex> g(a->sm_); a->controls_.erase(id); } } } ctl_guard{this, ""};
    if (chat) { std::lock_guard<std::mutex> g(sm_); controls_[id] = base.reasoning_end; ctl_guard.id = id; }
    const long long created = (long long)std::time(nullptr);
    const std::string fp = build_info;

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
        // X-Conversation-Id opts in to a replay buffer: generation then survives a client disconnect (llama-server spipe)
        StreamSessionPtr session;
        if (const std::string cid = req.header("x-conversation-id"); !cid.empty()) session = stream_create(cid);
        struct Fin { StreamSessionPtr s; ~Fin() { if (s) s->finalize(); } } fin{session};
        bool started = false, sock_ok = true;
        auto last_write = Clock::now();
        auto live = [&] { return session ? !session->cancelled() : sock_ok; };
        std::mutex wmu;  // socket writes come from the generation thread and the ping thread
        auto emit_raw = [&](const std::string& bytes) {
            std::lock_guard<std::mutex> g(wmu);
            if (session) session->append(bytes.data(), bytes.size());
            if (sock_ok) sock_ok = c.write_chunk(bytes);
            last_write = Clock::now();
            return live();
        };
        auto begin = [&](int) {
            if (started) return;
            started = true;
            sock_ok = c.begin_stream(200, "text/event-stream");
            last_write = Clock::now();
        };
        std::unique_ptr<StreamXlate> xl = make_stream_xlate(fmt, (int)t.prompt.size());
        auto sse = [&](const Value& v) {
            if (!xl) return emit_raw("data: " + v.dump() + "\n\n");
            for (auto& ev : xl->feed(v)) emit_raw(ev);
            return live();
        };
        // SSE comment ping after `sse_ping_interval` seconds of silence (undici & co. time out on idle streams)
        const int ping_s = t.sse_ping_interval;
        auto stream_alive = [&] { return session ? !session->cancelled() : !c.peer_closed(); };
        // the ping thread covers long prefill chunks, slow tokens and queue waits (the engine call cannot be interrupted)
        struct Pinger {
            std::thread th; std::mutex m; std::condition_variable cv; bool stop = false;
            ~Pinger() { { std::lock_guard<std::mutex> g(m); stop = true; } cv.notify_all(); if (th.joinable()) th.join(); }
        } pinger;
        if (ping_s > 0) pinger.th = std::thread([&] {
            std::unique_lock<std::mutex> lk(pinger.m);
            while (!pinger.stop) {
                pinger.cv.wait_for(lk, std::chrono::milliseconds(200));
                if (pinger.stop) break;
                double idle;
                bool go;
                { std::lock_guard<std::mutex> g(wmu); idle = std::chrono::duration<double>(Clock::now() - last_write).count(); go = started && sock_ok && idle > ping_s; }
                if (go) emit_raw(":\n\n");
            }
        });
        const bool bool_ok_unused = true; (void)bool_ok_unused;
        GenResult total;
        int tot_dec = 0, tot_cache = 0;
        std::vector<GenResult> results;
        for (int ci = 0; ci < (oai ? base.n_choices : 1) && live(); ++ci) {
            Task tc = t;
            if (ci > 0 && tc.sp.has_seed) tc.sp.seed += ci;
            const chat::ParseParams pp = parse_params_of(tc);
            chat::Message prev_msg;
            std::string acc;
            std::vector<std::string> tc_ids;
            bool any_tool = false;
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
            std::optional<Value> cur_timings;
            auto emit_delta = [&](const chat::Delta& dl, const std::vector<TokProb>& probs) {
                Value o2 = base_chunk("chat.completion.chunk");
                Value d = Value::object();
                if (!dl.reasoning.empty()) d.set("reasoning_content", dl.reasoning);
                if (!dl.content.empty() || (dl.reasoning.empty() && dl.tool_calls.empty())) d.set("content", dl.content);
                if (!dl.tool_calls.empty()) {
                    Value arr = Value::array();
                    for (auto& td : dl.tool_calls) {
                        if (td.is_new) { while (tc_ids.size() <= td.index) tc_ids.push_back(chat::gen_tool_call_id()); any_tool = true; }
                        arr.push(tool_delta_json(td, td.index < tc_ids.size() ? tc_ids[td.index] : std::string()));
                    }
                    d.set("tool_calls", std::move(arr));
                }
                Value ch = Value::object();
                ch.set("finish_reason", Value());
                ch.set("index", ci);
                ch.set("delta", std::move(d));
                if (!probs.empty()) { Value lp = Value::object(); lp.set("content", probs_json(probs, t.post_probs)); ch.set("logprobs", std::move(lp)); }
                o2.set("choices", Value::array()).push(std::move(ch));
                if (cur_timings) o2.set("timings", *cur_timings);
                return sse(o2);
            };
            auto on_chunk = [&](const Chunk& ck) {
                if (!live()) return false;
                std::optional<Value> tm;
                if (tc.timings_per_token) tm = timings_json(ck.n_cache, ck.n_prompt_eval, ck.prompt_ms, ck.n_decoded, ck.predicted_ms);
                cur_timings = tm;
                if (kind == Kind::Native || kind == Kind::Infill) {
                    Value o2 = Value::object();
                    o2.set("index", 0);
                    o2.set("content", ck.text);
                    o2.set("tokens", [&] { Value a = Value::array(); for (int x : ck.tokens) a.push(x); return a; }());
                    o2.set("stop", false);
                    o2.set("id_slot", -1);
                    o2.set("tokens_predicted", ck.n_decoded);
                    o2.set("tokens_evaluated", (int)t.prompt.size());
                    if (tm) o2.set("timings", *tm);
                    if (!ck.probs.empty()) o2.set("completion_probabilities", probs_json(ck.probs, t.post_probs));
                    return sse(o2);
                }
                if (chat) {
                    if (first) { first = false; if (!sse(role_chunk())) return false; }
                    acc += ck.text;
                    chat::ParseParams q = pp; q.partial = true;
                    chat::Message cur = chat::parse_output(acc, q);
                    chat::Delta dl = chat::diff_messages(prev_msg, cur);
                    prev_msg = std::move(cur);
                    if (dl.empty() && ck.probs.empty()) return true;
                    return emit_delta(dl, ck.probs);
                }
                Value o2 = base_chunk("text_completion");
                Value ch = Value::object();
                ch.set("text", ck.text);
                ch.set("index", ci);
                if (!ck.probs.empty()) { Value lp = Value::object(); lp.set("content", probs_json(ck.probs, t.post_probs)); ch.set("logprobs", std::move(lp)); } else ch.set("logprobs", Value());
                ch.set("finish_reason", Value());
                o2.set("choices", Value::array()).push(std::move(ch));
                if (tm) o2.set("timings", *tm);
                return sse(o2);
            };
            if (tc.return_progress) {
                tc.on_progress = [&](int total, int cache, int processed, double ms) {
                    begin(0);
                    Value pr = Value::object();
                    pr.set("total", total); pr.set("cache", cache); pr.set("processed", processed); pr.set("time_ms", ms);
                    Value o2;
                    if (chat) {
                        o2 = base_chunk("chat.completion.chunk");
                        Value d = Value::object();
                        d.set("role", "assistant");
                        d.set("content", Value());
                        Value ch = Value::object();
                        ch.set("finish_reason", Value());
                        ch.set("index", ci);
                        ch.set("delta", std::move(d));
                        o2.set("choices", Value::array()).push(std::move(ch));
                    } else if (kind == Kind::OaiCmpl) {
                        o2 = base_chunk("text_completion");
                        Value ch = Value::object();
                        ch.set("text", ""); ch.set("index", ci); ch.set("logprobs", Value()); ch.set("finish_reason", Value());
                        o2.set("choices", Value::array()).push(std::move(ch));
                    } else {
                        o2 = Value::object();
                        o2.set("index", 0); o2.set("content", ""); o2.set("tokens", Value::array()); o2.set("stop", false); o2.set("id_slot", -1);
                        o2.set("tokens_predicted", 0); o2.set("tokens_evaluated", (int)t.prompt.size());
                    }
                    o2.set("prompt_progress", std::move(pr));
                    sse(o2);
                };
            }
            GenResult r = generate(tc, stream_alive, on_chunk, begin, lease_ticket);
            if (xl) xl->stopping_word = r.stopping_word;
            if (r.aborted) {
                if (session && session->cancelled() && sock_ok) c.end_chunks();  // explicit stop (DELETE /v1/stream): end the stream cleanly
                else c.close_after = true;
                return;
            }
            begin(0);
            if (chat && first) { first = false; sse(role_chunk()); }
            cur_timings.reset();
            if (chat) {  // release text held back by the parser
                chat::ParseParams q = pp; q.partial = true; q.flush = true;
                chat::Message cur = chat::parse_output(acc, q);
                chat::Delta dl = chat::diff_messages(prev_msg, cur);
                prev_msg = std::move(cur);
                if (!dl.empty()) emit_delta(dl, {});
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
                ch.set("finish_reason", !prev_msg.tool_calls.empty() ? "tool_calls" : finish_reason_of(r));
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
        if (chat && base.include_usage && live()) {
            Value o2 = base_chunk("chat.completion.chunk");
            o2.set("usage", usage_of((int)t.prompt.size(), tot_cache, tot_dec));
            o2.set("timings", timings_of(total));
            sse(o2);
        }
        if (xl) { for (auto& ev : xl->finish()) emit_raw(ev); }
        else if (oai && live()) emit_raw("data: [DONE]\n\n");
        if (sock_ok) c.end_chunks();
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
            GenResult r = generate(tc, alive, [](const Chunk&) { return true; }, [](int) {}, lease_ticket);
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
                chat::Message pm = chat::parse_output(r.content, parse_params_of(t));
                Value m = chat::message_to_json(pm, true);
                if (pm.content.empty() && !pm.tool_calls.empty()) m.set("content", std::string());  // llama-server: "" (not null) next to tool_calls
                ch.set("finish_reason", !pm.tool_calls.empty() ? "tool_calls" : finish_reason_of(r));
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
    if (fmt == ApiFmt::Anthropic && results.size() == 1) { respond_json(c, 200, oai_to_anthropic(results[0], grs[0].stopping_word)); return; }
    if (fmt == ApiFmt::Responses && results.size() == 1) { respond_json(c, 200, oai_to_responses(results[0])); return; }
    if (results.size() == 1) respond_json(c, 200, results[0]);
    else { Value a = Value::array(); for (auto& r : results) a.push(std::move(r)); respond_json(c, 200, a); }
}

void App::handle_control(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    auto str = [&](const char* k) { const Value* v = body.is_object() ? body.find(k) : nullptr; return v && v->is_string() ? v->as_string() : std::string(); };
    const std::string id = str("id"), action = str("action");
    if (id.empty()) throw_error(400, "missing completion id");
    if (action != "reasoning_end") throw_error(400, "unknown control action");
    std::shared_ptr<std::atomic<bool>> flag;
    bool known = false;
    { std::lock_guard<std::mutex> g(sm_); auto it = controls_.find(id); if (it != controls_.end()) { known = true; flag = it->second; } }
    Value r = Value::object();
    if (!known) { r.set("success", false); r.set("message", "no active completion for this id"); }
    else if (!flag) { r.set("success", false); r.set("message", "reasoning control not enabled for this completion"); }
    else { flag->store(true); r.set("success", true); }
    respond_json(c, 200, r);
}

void App::handle_responses(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); }
    catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    Value conv;
    try { conv = responses_to_oai(body); } catch (const std::exception& e) { throw_error(400, e.what(), "invalid_request_error"); }
    handle_completion(req, c, Kind::OaiChat, ApiFmt::Responses, &conv);
}

void App::handle_messages(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); }
    catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    Value conv;
    try { conv = anthropic_to_oai(body); } catch (const std::exception& e) { throw_error(400, e.what(), "invalid_request_error"); }
    handle_completion(req, c, Kind::OaiChat, ApiFmt::Anthropic, &conv);
}

void App::handle_count_tokens(const http::Request& req, http::Conn& c, ApiFmt fmt) {
    Value body;
    try { body = jd::parse(req.body); }
    catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    std::string req_model;
    if (body.is_object()) if (const Value* mv = body.find("model"); mv && mv->is_string()) req_model = mv->as_string();
    auto alive = [&] { return !c.peer_closed(); };
    long lease_ticket = acquire_model_lease(req_model, alive);
    if (lease_ticket == -2) { c.close_after = true; return; }
    struct ModelLeaseGuard { App& a; long t; ~ModelLeaseGuard() { if (t >= 0) a.release_model_lease(t); } } lease_guard{*this, lease_ticket};
    if (fmt == ApiFmt::Responses) { try { body = responses_to_oai(body); } catch (const std::exception& e) { throw_error(400, e.what(), "invalid_request_error"); } }
    else if (fmt == ApiFmt::Anthropic) { try { body = anthropic_to_oai(body); } catch (const std::exception& e) { throw_error(400, e.what(), "invalid_request_error"); } }
    Task t = parse_task(body, Kind::OaiChat);
    Value v = Value::object();
    v.set("input_tokens", (long long)t.prompt.size());
    if (fmt != ApiFmt::Anthropic) v.set("object", "response.input_tokens");
    respond_json(c, 200, v);
}

void App::handle_infill(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); }
    catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object()) throw_error(400, "request body must be a JSON object");
    auto alive = [&] { return !c.peer_closed(); };
    std::string req_model;
    if (const Value* mv = body.find("model"); mv && mv->is_string()) req_model = mv->as_string();
    long lease_ticket = acquire_model_lease(req_model, alive);
    if (lease_ticket == -2) { c.close_after = true; return; }
    struct ModelLeaseGuard { App& a; long t; ~ModelLeaseGuard() { if (t >= 0) a.release_model_lease(t); } } lease_guard{*this, lease_ticket};
    if (fim_pre < 0 || fim_suf < 0 || fim_mid < 0) {
        std::string err;
        if (fim_pre < 0) err += "prefix token is missing. ";
        if (fim_suf < 0) err += "suffix token is missing. ";
        if (fim_mid < 0) err += "middle token is missing. ";
        throw_error(501, "Infill is not supported by this model: " + err, "not_supported_error");
    }
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
        }, begin, lease_ticket);
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
    GenResult r = generate(t, alive, [](const Chunk&) { return true; }, [](int) {}, lease_ticket);
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
