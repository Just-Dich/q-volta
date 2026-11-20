// qv_server — OpenAI-compatible HTTP server for Q-Volta (CPU or CUDA engine).
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#ifdef QV_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/json_doc.h"
#include "qv/engine.h"
#include "sampling/sampler.h"
#include "server/http.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

using qv::jd::Value;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    std::string model, backend = "auto", host = "127.0.0.1", api_key, model_name = "qwen3.8-27b", tokenizer, reasoning_format = "none";
    int port = 8080, threads = 0, max_ctx = 4096, prefill_chunk = 0, http_threads = 16, max_queue = 8, default_max_tokens = 1024;
    float def_temperature = 1.0f, def_top_p = 0.95f;
    int def_top_k = 20;
    bool verbose = true;
};

struct ApiError {
    int status;
    std::string message, type, param, code;
};
[[noreturn]] void bad(const std::string& msg, const std::string& param = "", const std::string& code = "") {
    throw ApiError{400, msg, "invalid_request_error", param, code};
}

std::string error_json(const ApiError& e) {
    Value err = Value::object();
    err.set("message", e.message);
    err.set("type", e.type);
    err.set("param", e.param.empty() ? Value() : Value(e.param));
    err.set("code", e.code.empty() ? Value() : Value(e.code));
    Value o = Value::object();
    o.set("error", std::move(err));
    return o.dump();
}

std::string rand_hex(int n) {
    static std::mutex m;
    static std::mt19937_64 rng{std::random_device{}()};
    std::lock_guard<std::mutex> g(m);
    static const char* h = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; ++i) s += h[rng() & 15];
    return s;
}

// ---------------------------------------------------------------------------------------------- request model
struct Job {
    bool chat = false, stream = false, include_usage = false, thinking = true;
    std::vector<int> prompt;
    int max_tokens = 0;
    qv::SamplingParams sp;
    std::vector<std::string> stops;
    int logprobs_top = -1;  // -1 = off; otherwise number of alternatives
};

struct TokLp {
    int token;
    std::string text;
    float logprob;
    std::vector<std::pair<std::string, float>> top;
    std::vector<int> top_ids;
};

struct GenResult {
    std::string finish = "length";
    int completion_tokens = 0, cached = 0;
    double ttft_ms = 0, total_ms = 0;
    bool aborted = false;
};

// Splits "<reasoning></think>answer" output into reasoning_content / content (only with --reasoning-format separate).
class ReasoningSplitter {
   public:
    explicit ReasoningSplitter(bool in_reasoning) : in_(in_reasoning) {}
    void feed(const std::string& t, std::string& reasoning, std::string& content, bool final_) {
        static const std::string tag = "</think>";
        std::string buf = hold_ + t;
        hold_.clear();
        for (;;) {
            if (in_) {
                size_t p = buf.find(tag);
                if (p != std::string::npos) {
                    reasoning += buf.substr(0, p);
                    buf.erase(0, p + tag.size());
                    in_ = false;
                    lead_ = true;
                    continue;
                }
                size_t keep = 0;
                if (!final_)
                    for (size_t k = std::min(buf.size(), tag.size() - 1); k > 0; --k)
                        if (buf.compare(buf.size() - k, k, tag, 0, k) == 0) { keep = k; break; }
                reasoning += buf.substr(0, buf.size() - keep);
                hold_ = buf.substr(buf.size() - keep);
                return;
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
    bool in_, lead_ = false;
    std::string hold_;
};

// ---------------------------------------------------------------------------------------------- application
class App {
   public:
    Options o;
    qv::Tokenizer tok;
    std::unique_ptr<qv::Engine> eng;
    std::mutex eng_mu;
    std::atomic<int> active{0};  // admitted requests (running + waiting)
    std::vector<int> cached;     // tokens currently inside the engine state (logits are valid for exactly this context)
    std::vector<int> stop_ids;
    std::string backend_name;
    int chunk = 256;

    // ---- generation ------------------------------------------------------------------------
    template <class Alive, class Begin, class Delta>
    GenResult generate(const Job& j, Alive alive, Begin begin, Delta on_delta) {
        std::lock_guard<std::mutex> lk(eng_mu);
        GenResult r;
        auto t0 = Clock::now();
        auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
        if (!alive()) { r.aborted = true; return r; }
        const int N = (int)j.prompt.size();
        try {
            size_t start = 0;
            if (!cached.empty() && cached.size() <= j.prompt.size() && std::equal(cached.begin(), cached.end(), j.prompt.begin())) start = cached.size();
            else { eng->reset(); cached.clear(); }
            r.cached = (int)start;
            if (!begin(r.cached)) { r.aborted = true; return r; }
            for (size_t off = start; off < (size_t)N; off += chunk) {
                if (!alive()) { r.aborted = true; r.total_ms = ms(); return r; }
                int n = (int)std::min<size_t>(chunk, N - off);
                eng->prefill(j.prompt.data() + off, n, (int)off);
                cached.insert(cached.end(), j.prompt.begin() + off, j.prompt.begin() + off + n);
            }
            qv::Sampler sampler(j.sp, j.prompt);
            qv::Tokenizer::Decoder dec(tok);
            std::string pending;
            std::vector<float> lg, raw;
            const bool want_lp = j.logprobs_top >= 0;
            auto emit_text = [&](std::string&& text, const TokLp* lp) { return on_delta(text, lp); };
            bool hit_string = false;
            for (int n = 0;;) {
                int t;
                TokLp lp;
                if (j.sp.engine_argmax_ok() && !want_lp) t = eng->argmax();
                else {
                    eng->logits(lg);
                    if (want_lp) raw = lg;
                    t = j.sp.engine_argmax_ok() ? (int)(std::max_element(lg.begin(), lg.end()) - lg.begin()) : sampler.sample(lg);
                    if (want_lp) {
                        auto info = qv::Sampler::logprob(raw, t, j.logprobs_top);
                        lp.token = t;
                        lp.text = tok.decode({t});
                        lp.logprob = info.logprob;
                        for (auto& [id, v] : info.top) { lp.top.emplace_back(tok.decode({id}), v); lp.top_ids.push_back(id); }
                    }
                }
                if (n == 0) r.ttft_ms = ms();
                ++n;
                r.completion_tokens = n;
                if (std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end()) {
                    r.finish = "stop";
                    break;
                }
                sampler.accept(t);
                pending += dec.push(t);
                // stop strings with hold-back of a possible partial match
                size_t hit = std::string::npos;
                for (auto& s : j.stops) hit = std::min(hit, pending.find(s));
                if (hit != std::string::npos) {
                    pending.resize(hit);
                    hit_string = true;
                    r.finish = "stop";
                    if (!pending.empty() || want_lp) { if (!emit_text(std::move(pending), want_lp ? &lp : nullptr)) r.aborted = true; }
                    pending.clear();
                    break;
                }
                size_t keep = 0;
                for (auto& s : j.stops)
                    for (size_t k = std::min(pending.size(), s.size() - 1); k > keep; --k)
                        if (pending.compare(pending.size() - k, k, s, 0, k) == 0) { keep = k; break; }
                std::string out = pending.substr(0, pending.size() - keep);
                pending.erase(0, pending.size() - keep);
                if (!out.empty() || want_lp) {
                    if (!emit_text(std::move(out), want_lp ? &lp : nullptr)) { r.aborted = true; break; }
                }
                if (n >= j.max_tokens) { r.finish = "length"; break; }
                if (!alive()) { r.aborted = true; break; }
                eng->step(t, (int)cached.size());
                cached.push_back(t);
            }
            if (!r.aborted && !hit_string) {  // flush the held-back tail (partial stop-string prefix, incomplete UTF-8)
                pending += dec.flush();
                if (!pending.empty()) emit_text(std::move(pending), nullptr);
            }
        } catch (const std::exception& e) {
            eng->reset();
            cached.clear();
            std::fprintf(stderr, "[qv_server] engine error: %s\n", e.what());
            r.finish = "error";
        }
        r.total_ms = ms();
        return r;
    }

    // ---- request parsing -------------------------------------------------------------------
    static const Value* opt(const Value& b, const char* k) {
        const Value* v = b.find(k);
        return (v && !v->is_null()) ? v : nullptr;
    }
    static double num(const Value& b, const char* k, double lo, double hi, double dflt, bool lo_excl = false) {
        const Value* v = opt(b, k);
        if (!v) return dflt;
        if (!v->is_number()) bad(std::string("'") + k + "' must be a number", k, "invalid_type");
        double d = v->as_double();
        if (d < lo || d > hi || (lo_excl && d == lo)) bad(std::string("'") + k + "' is out of range", k, "invalid_value");
        return d;
    }

    Job parse(const Value& b, bool chat) {
        if (!b.is_object()) bad("request body must be a JSON object");
        Job j;
        j.chat = chat;
        if (const Value* n = opt(b, "n"); n && !(n->is_number() && n->as_int() == 1))
            throw ApiError{400, "Only n=1 is supported by this server.", "invalid_request_error", "n", "unsupported_value"};
        if (const Value* n = opt(b, "best_of"); n && !(n->is_number() && n->as_int() == 1)) bad("best_of is not supported", "best_of", "unsupported_value");
        for (const char* k : {"tools", "functions", "tool_choice", "function_call", "logit_bias", "suffix"}) {
            const Value* v = opt(b, k);
            if (v && !((v->is_array() || v->is_object()) && v->size() == 0) && !(v->is_string() && v->as_string() == "none"))
                bad(std::string("'") + k + "' is not supported by this server", k, "unsupported_value");
        }
        if (const Value* rf = opt(b, "response_format"); rf && rf->is_object()) {
            const Value* t = rf->find("type");
            if (t && t->is_string() && t->as_string() != "text") bad("only response_format type 'text' is supported", "response_format", "unsupported_value");
        }
        if (const Value* e = opt(b, "echo"); e && e->is_bool() && e->as_bool()) bad("echo is not supported", "echo", "unsupported_value");
        if (const Value* s = opt(b, "stream")) {
            if (!s->is_bool()) bad("'stream' must be a boolean", "stream", "invalid_type");
            j.stream = s->as_bool();
        }
        if (const Value* so = opt(b, "stream_options"); so && so->is_object())
            if (const Value* iu = opt(*so, "include_usage")) j.include_usage = iu->is_bool() && iu->as_bool();

        // sampling
        j.sp.temperature = (float)num(b, "temperature", 0, 2, o.def_temperature);
        j.sp.top_p = (float)num(b, "top_p", 0, 1, o.def_top_p, true);
        j.sp.min_p = (float)num(b, "min_p", 0, 1, 0);
        j.sp.presence_penalty = (float)num(b, "presence_penalty", -2, 2, 0);
        j.sp.frequency_penalty = (float)num(b, "frequency_penalty", -2, 2, 0);
        j.sp.repetition_penalty = (float)num(b, "repetition_penalty", 0, 10, 1, true);
        if (const Value* k = opt(b, "top_k")) {
            if (!k->is_number()) bad("'top_k' must be an integer", "top_k", "invalid_type");
            j.sp.top_k = std::max(0, (int)k->as_int());
        } else j.sp.top_k = o.def_top_k;
        if (const Value* s = opt(b, "seed")) {
            if (!s->is_number()) bad("'seed' must be an integer", "seed", "invalid_type");
            j.sp.has_seed = true;
            j.sp.seed = (uint64_t)s->as_int();
        }
        if (const Value* s = opt(b, "stop")) {
            if (s->is_string()) j.stops.push_back(s->as_string());
            else if (s->is_array()) {
                for (auto& x : s->items()) {
                    if (!x.is_string()) bad("'stop' entries must be strings", "stop", "invalid_type");
                    j.stops.push_back(x.as_string());
                }
            } else bad("'stop' must be a string or an array of strings", "stop", "invalid_type");
            if (j.stops.size() > 4) bad("'stop' accepts at most 4 sequences", "stop", "invalid_value");
            j.stops.erase(std::remove(j.stops.begin(), j.stops.end(), std::string()), j.stops.end());
        }
        int mt = -1;
        for (const char* k : {"max_completion_tokens", "max_tokens"}) {
            if (const Value* m = opt(b, k)) {
                if (!m->is_number()) bad(std::string("'") + k + "' must be an integer", k, "invalid_type");
                if (m->as_int() < 1) bad(std::string("'") + k + "' must be >= 1", k, "invalid_value");
                mt = (int)std::min<long long>(m->as_int(), 1 << 30);
                break;
            }
        }
        // logprobs
        if (chat) {
            if (const Value* l = opt(b, "logprobs"); l && l->is_bool() && l->as_bool()) {
                j.logprobs_top = 0;
                if (const Value* t = opt(b, "top_logprobs")) {
                    if (!t->is_number() || t->as_int() < 0 || t->as_int() > 20) bad("'top_logprobs' must be an integer in [0, 20]", "top_logprobs", "invalid_value");
                    j.logprobs_top = (int)t->as_int();
                }
            }
        } else if (const Value* l = opt(b, "logprobs")) {
            if (!l->is_number() || l->as_int() < 0 || l->as_int() > 20) bad("'logprobs' must be an integer in [0, 20]", "logprobs", "invalid_value");
            j.logprobs_top = (int)l->as_int();
        }

        // prompt
        if (chat) {
            const Value* m = opt(b, "messages");
            if (!m) bad("'messages' is required", "messages", "missing_required_parameter");
            qv::ChatTemplateOptions co;
            const Value* kw = opt(b, "chat_template_kwargs");
            auto kwget = [&](const char* k) -> const Value* {
                if (kw && kw->is_object()) if (const Value* v = opt(*kw, k)) return v;
                return opt(b, k);
            };
            if (const Value* v = kwget("enable_thinking")) { if (!v->is_bool()) bad("'enable_thinking' must be a boolean", "enable_thinking", "invalid_type"); co.enable_thinking = v->as_bool(); }
            if (const Value* v = kwget("preserve_thinking")) { if (!v->is_bool()) bad("'preserve_thinking' must be a boolean", "preserve_thinking", "invalid_type"); co.preserve_thinking = v->as_bool(); }
            if (const Value* v = kwget("reasoning_effort")) {
                if (!v->is_string()) bad("'reasoning_effort' must be a string", "reasoning_effort", "invalid_type");
                std::string e = v->as_string();
                if (e == "high") e = "xhigh";            // OpenAI vocabulary -> template vocabulary
                else if (e == "minimal") e = "low";
                else if (e == "none") { co.enable_thinking = false; e = ""; }
                co.reasoning_effort = e;
            }
            j.thinking = !co.enable_thinking.has_value() || *co.enable_thinking;
            std::string text;
            try { text = qv::render_chat(qv::messages_from_json(*m), co); }
            catch (const std::exception& e) { bad(e.what(), "messages", "invalid_value"); }
            j.prompt = tok.encode(text, true);
        } else {
            const Value* p = opt(b, "prompt");
            if (!p) bad("'prompt' is required", "prompt", "missing_required_parameter");
            auto ids_of = [&](const Value& a) {
                std::vector<int> ids;
                for (auto& x : a.items()) {
                    if (!x.is_int() || x.as_int() < 0 || x.as_int() >= tok.vocab_size()) bad("invalid token id in 'prompt'", "prompt", "invalid_value");
                    ids.push_back((int)x.as_int());
                }
                return ids;
            };
            const Value* q = p;
            if (p->is_array() && p->size() > 0 && p->items()[0].is_array()) q = &p->items()[0];
            if (q->is_string()) j.prompt = tok.encode(q->as_string(), true);
            else if (q->is_array() && q->size() > 0 && q->items()[0].is_string()) j.prompt = tok.encode(q->items()[0].as_string(), true);
            else if (q->is_array()) j.prompt = ids_of(*q);
            else bad("'prompt' must be a string, an array of strings or an array of token ids", "prompt", "invalid_type");
        }
        if (j.prompt.empty()) bad("the prompt is empty", chat ? "messages" : "prompt", "invalid_value");
        if ((int)j.prompt.size() >= o.max_ctx)
            throw ApiError{413, "This model's maximum context length is " + std::to_string(o.max_ctx) + " tokens, however the prompt has " + std::to_string(j.prompt.size()) +
                                    " tokens. Please reduce the length of the prompt.",
                           "invalid_request_error", chat ? "messages" : "prompt", "context_length_exceeded"};
        int avail = o.max_ctx - (int)j.prompt.size();
        j.max_tokens = std::min(mt > 0 ? mt : o.default_max_tokens, avail);
        return j;
    }

    // ---- handlers --------------------------------------------------------------------------
    struct Admission {
        App& a;
        bool ok;
        explicit Admission(App& app) : a(app) { ok = a.active.fetch_add(1) < a.o.max_queue + 1; }
        ~Admission() { a.active.fetch_sub(1); }
    };

    static Value usage(const Job& j, const GenResult& r) {
        Value u = Value::object();
        u.set("prompt_tokens", (long long)j.prompt.size());
        u.set("completion_tokens", (long long)r.completion_tokens);
        u.set("total_tokens", (long long)(j.prompt.size() + r.completion_tokens));
        Value d = Value::object();
        d.set("cached_tokens", (long long)r.cached);
        u.set("prompt_tokens_details", std::move(d));
        return u;
    }
    static Value bytes_of(const std::string& s) {
        Value a = Value::array();
        for (unsigned char c : s) a.push((int)c);
        return a;
    }
    static Value chat_lp(const std::vector<TokLp>& v) {
        Value content = Value::array();
        for (auto& t : v) {
            Value e = Value::object();
            e.set("token", t.text);
            e.set("logprob", (double)t.logprob);
            e.set("bytes", bytes_of(t.text));
            Value top = Value::array();
            for (auto& [s, lp] : t.top) {
                Value x = Value::object();
                x.set("token", s);
                x.set("logprob", (double)lp);
                x.set("bytes", bytes_of(s));
                top.push(std::move(x));
            }
            e.set("top_logprobs", std::move(top));
            content.push(std::move(e));
        }
        Value o = Value::object();
        o.set("content", std::move(content));
        o.set("refusal", Value());
        return o;
    }
    static Value completion_lp(const std::vector<TokLp>& v) {
        Value toks = Value::array(), lps = Value::array(), tops = Value::array(), offs = Value::array();
        long long off = 0;
        for (auto& t : v) {
            toks.push(t.text);
            lps.push((double)t.logprob);
            offs.push(off);
            off += (long long)t.text.size();
            Value m = Value::object();
            for (auto& [s, lp] : t.top) m.set(s, (double)lp);
            tops.push(std::move(m));
        }
        Value o = Value::object();
        o.set("tokens", std::move(toks));
        o.set("token_logprobs", std::move(lps));
        o.set("top_logprobs", std::move(tops));
        o.set("text_offset", std::move(offs));
        return o;
    }

    void log_req(const char* kind, const std::string& id, const Job& j, const GenResult& r) {
        if (!o.verbose) return;
        double dec_ms = r.total_ms - r.ttft_ms;
        std::fprintf(stderr, "[qv_server] %s %s prompt=%zu cached=%d gen=%d finish=%s%s ttft=%.0fms decode=%.2f tok/s total=%.0fms\n", kind, id.c_str(),
                     j.prompt.size(), r.cached, r.completion_tokens, r.finish.c_str(), r.aborted ? " (client aborted)" : "", r.ttft_ms,
                     r.completion_tokens > 1 && dec_ms > 0 ? (r.completion_tokens - 1) * 1000.0 / dec_ms : 0.0, r.total_ms);
    }

    void handle_generation(const qv::http::Request& req, qv::http::Conn& c, bool chat) {
        Value body;
        try { body = qv::jd::parse(req.body); }
        catch (const qv::jd::ParseError& e) { throw ApiError{400, std::string("Invalid JSON body: ") + e.what(), "invalid_request_error", "", "invalid_json"}; }
        Job j = parse(body, chat);
        Admission adm(*this);
        if (!adm.ok) throw ApiError{503, "The server is busy (request queue is full). Retry later.", "server_error", "", "server_busy"};

        const std::string id = (chat ? "chatcmpl-" : "cmpl-") + rand_hex(24);
        const long long created = (long long)std::time(nullptr);
        const bool split = chat && o.reasoning_format == "separate";
        ReasoningSplitter rs(split && j.thinking);
        auto alive = [&] { return !c.peer_closed(); };
        const char* kind = chat ? "chat" : "completion";

        auto make_choice_chunk = [&](Value choice) {
            Value o2 = Value::object();
            o2.set("id", id);
            o2.set("object", chat ? "chat.completion.chunk" : "text_completion");
            o2.set("created", created);
            o2.set("model", o.model_name);
            Value ch = Value::array();
            if (!choice.is_null()) ch.push(std::move(choice));
            o2.set("choices", std::move(ch));
            if (j.include_usage) o2.set("usage", Value());
            return o2;
        };
        auto sse = [&](const Value& v) { return c.write_chunk("data: " + v.dump() + "\n\n"); };

        if (j.stream) {
            bool started = false;
            auto begin = [&](int cached_tokens) {
                bool ok = c.begin_stream(200, "text/event-stream", {{"X-QV-Prompt-Cached-Tokens", std::to_string(cached_tokens)}, {"X-QV-Prompt-Tokens", std::to_string(j.prompt.size())}});
                started = true;
                if (ok && chat) {
                    Value d = Value::object();
                    d.set("role", "assistant");
                    d.set("content", "");
                    Value ch = Value::object();
                    ch.set("index", 0);
                    ch.set("delta", std::move(d));
                    ch.set("logprobs", Value());
                    ch.set("finish_reason", Value());
                    ok = sse(make_choice_chunk(std::move(ch)));
                }
                return ok;
            };
            auto send_text = [&](const std::string& reasoning, const std::string& content, const TokLp* lp) {
                if (reasoning.empty() && content.empty() && !lp) return true;
                Value ch = Value::object();
                ch.set("index", 0);
                std::vector<TokLp> one;
                if (lp) one.push_back(*lp);
                if (chat) {
                    Value d = Value::object();
                    if (!reasoning.empty()) d.set("reasoning_content", reasoning);
                    if (!content.empty() || reasoning.empty()) d.set("content", content);
                    ch.set("delta", std::move(d));
                    ch.set("logprobs", lp ? chat_lp(one) : Value());
                    ch.set("finish_reason", Value());
                } else {
                    ch.set("text", content);
                    ch.set("logprobs", lp ? completion_lp(one) : Value());
                    ch.set("finish_reason", Value());
                }
                return sse(make_choice_chunk(std::move(ch)));
            };
            auto on_delta = [&](const std::string& text, const TokLp* lp) {
                std::string r, ct;
                rs.feed(text, r, ct, false);
                return send_text(r, ct, lp);
            };
            GenResult r = generate(j, alive, begin, on_delta);
            if (!started) {  // aborted before the stream began
                log_req(kind, id, j, r);
                c.close_after = true;
                return;
            }
            if (!r.aborted && r.finish != "error") {
                std::string rr, ct;
                rs.feed("", rr, ct, true);
                send_text(rr, ct, nullptr);
                Value ch = Value::object();
                ch.set("index", 0);
                if (chat) ch.set("delta", Value::object());
                else ch.set("text", "");
                ch.set("logprobs", Value());
                ch.set("finish_reason", r.finish);
                bool ok = sse(make_choice_chunk(std::move(ch)));
                if (ok && j.include_usage) {
                    Value u = make_choice_chunk(Value());
                    u.set("usage", usage(j, r));
                    ok = sse(u);
                }
                if (ok) ok = c.write_chunk("data: [DONE]\n\n");
                if (ok) c.end_chunks();
            } else if (r.finish == "error") {
                ApiError e{500, "internal engine error", "server_error", "", ""};
                c.write_chunk("data: " + error_json(e) + "\n\n");
                c.end_chunks();
            }
            log_req(kind, id, j, r);
            c.close_after = true;
            return;
        }

        // ---- non-streaming
        std::string reasoning, content;
        std::vector<TokLp> lps;
        int cached_tokens = 0;
        auto begin = [&](int ct) { cached_tokens = ct; return true; };
        auto on_delta = [&](const std::string& text, const TokLp* lp) {
            rs.feed(text, reasoning, content, false);
            if (lp) lps.push_back(*lp);
            return true;
        };
        GenResult r = generate(j, alive, begin, on_delta);
        log_req(kind, id, j, r);
        if (r.aborted) { c.close_after = true; return; }
        if (r.finish == "error") throw ApiError{500, "internal engine error", "server_error", "", ""};
        rs.feed("", reasoning, content, true);
        Value resp = Value::object();
        resp.set("id", id);
        resp.set("object", chat ? "chat.completion" : "text_completion");
        resp.set("created", created);
        resp.set("model", o.model_name);
        Value choice = Value::object();
        choice.set("index", 0);
        if (chat) {
            Value m = Value::object();
            m.set("role", "assistant");
            m.set("content", content);
            if (split) m.set("reasoning_content", reasoning);
            m.set("refusal", Value());
            choice.set("message", std::move(m));
        } else choice.set("text", content);
        choice.set("logprobs", j.logprobs_top >= 0 ? (chat ? chat_lp(lps) : completion_lp(lps)) : Value());
        choice.set("finish_reason", r.finish);
        Value choices = Value::array();
        choices.push(std::move(choice));
        resp.set("choices", std::move(choices));
        resp.set("usage", usage(j, r));
        c.respond(200, "application/json", resp.dump(), {{"X-QV-Prompt-Cached-Tokens", std::to_string(cached_tokens)}});
    }

    void route(const qv::http::Request& req, qv::http::Conn& c) {
        try {
            if (req.method == "OPTIONS") { c.respond(204, "", ""); return; }
            if (req.path == "/health" || req.path == "/healthz") {
                if (req.method != "GET" && req.method != "HEAD") throw ApiError{405, "method not allowed", "invalid_request_error", "", ""};
                c.respond(200, "application/json", "{\"status\":\"ok\"}");
                return;
            }
            if (req.path.rfind("/v1/", 0) != 0) throw ApiError{404, "Unknown route: " + req.method + " " + req.path, "invalid_request_error", "", "not_found"};
            if (!o.api_key.empty()) {
                std::string a = req.header("authorization");
                if (a != "Bearer " + o.api_key) throw ApiError{401, "Incorrect API key provided.", "invalid_request_error", "", "invalid_api_key"};
            }
            if (req.path == "/v1/models" || req.path.rfind("/v1/models/", 0) == 0) {
                if (req.method != "GET") throw ApiError{405, "method not allowed", "invalid_request_error", "", ""};
                Value m = Value::object();
                m.set("id", o.model_name);
                m.set("object", "model");
                m.set("created", (long long)started_at);
                m.set("owned_by", "q-volta");
                if (req.path == "/v1/models") {
                    Value list = Value::object();
                    list.set("object", "list");
                    Value d = Value::array();
                    d.push(std::move(m));
                    list.set("data", std::move(d));
                    c.respond(200, "application/json", list.dump());
                } else c.respond(200, "application/json", m.dump());
                return;
            }
            if (req.path == "/v1/chat/completions" || req.path == "/v1/completions") {
                if (req.method != "POST") throw ApiError{405, "method not allowed (use POST)", "invalid_request_error", "", ""};
                handle_generation(req, c, req.path == "/v1/chat/completions");
                return;
            }
            throw ApiError{404, "Unknown route: " + req.method + " " + req.path, "invalid_request_error", "", "not_found"};
        } catch (const ApiError& e) {
            if (!c.broken) c.respond(e.status, "application/json", error_json(e));
            if (e.status == 413 || e.status == 503) c.close_after = true;
        }
    }
    long long started_at = (long long)std::time(nullptr);
};

std::atomic<qv::http::Server*> g_server{nullptr};
void on_signal(int) { if (auto* s = g_server.load()) s->stop(); }

void usage_exit() {
    std::fprintf(stderr,
                 "usage: qv_server --model PREFIX [--backend cuda|cpu|auto] [--threads N] [--max-ctx N] [--host 127.0.0.1] [--port 8080]\n"
                 "                 [--api-key KEY] [--model-name NAME] [--prefill-chunk N] [--tokenizer tokenizer.json]\n"
                 "                 [--http-threads N] [--max-queue N] [--default-max-tokens N] [--reasoning-format none|separate]\n"
                 "                 [--temperature T] [--top-p P] [--top-k K] [--quiet]\n");
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); usage_exit(); } return argv[++i]; };
        try {
            if (a == "--model") o.model = next();
            else if (a == "--backend") o.backend = next();
            else if (a == "--threads") o.threads = std::stoi(next());
            else if (a == "--max-ctx") o.max_ctx = std::stoi(next());
            else if (a == "--host") o.host = next();
            else if (a == "--port") o.port = std::stoi(next());
            else if (a == "--api-key") o.api_key = next();
            else if (a == "--model-name") o.model_name = next();
            else if (a == "--prefill-chunk") o.prefill_chunk = std::stoi(next());
            else if (a == "--tokenizer") o.tokenizer = next();
            else if (a == "--http-threads") o.http_threads = std::stoi(next());
            else if (a == "--max-queue") o.max_queue = std::stoi(next());
            else if (a == "--default-max-tokens") o.default_max_tokens = std::stoi(next());
            else if (a == "--reasoning-format") o.reasoning_format = next();
            else if (a == "--temperature") o.def_temperature = std::stof(next());
            else if (a == "--top-p") o.def_top_p = std::stof(next());
            else if (a == "--top-k") o.def_top_k = std::stoi(next());
            else if (a == "--quiet") o.verbose = false;
            else usage_exit();
        } catch (const std::exception&) { std::fprintf(stderr, "bad value for %s\n", a.c_str()); usage_exit(); }
    }
    if (o.model.empty()) usage_exit();
    if (o.reasoning_format != "none" && o.reasoning_format != "separate") usage_exit();
    signal(SIGPIPE, SIG_IGN);

    auto app = std::make_unique<App>();
    app->o = o;
    // tokenizer: explicit flag, else <prefix>.tokenizer.json, else <prefix dir>/../Qwen3.8-27B/tokenizer.json
    std::string tpath = o.tokenizer;
    if (tpath.empty()) {
        std::string dir = o.model.substr(0, o.model.find_last_of('/') == std::string::npos ? 0 : o.model.find_last_of('/'));
        for (std::string cand : {o.model + ".tokenizer.json", dir + "/../Qwen3.8-27B/tokenizer.json", dir + "/tokenizer.json"})
            if (FILE* f = std::fopen(cand.c_str(), "rb")) { std::fclose(f); tpath = cand; break; }
    }
    if (tpath.empty()) { std::fprintf(stderr, "cannot find tokenizer.json; pass --tokenizer\n"); return 2; }
    try { app->tok = qv::Tokenizer::load(tpath); }
    catch (const std::exception& e) { std::fprintf(stderr, "tokenizer: %s\n", e.what()); return 2; }
    for (const char* s : {"<|im_end|>", "<|endoftext|>"}) { int id = app->tok.token_id(s); if (id >= 0) app->stop_ids.push_back(id); }

    // backend selection
    std::string backend = o.backend;
#ifdef QV_HAVE_CUDA
    if (backend == "auto") {
        int n = 0;
        backend = (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? "cuda" : "cpu";
    }
#else
    if (backend == "auto") backend = "cpu";
#endif
    if (backend != "cpu" && backend != "cuda") usage_exit();
    qv::EngineOptions eo;
    eo.max_ctx = o.max_ctx;
    eo.threads = o.threads > 0 ? o.threads : (int)std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
    app->chunk = o.prefill_chunk > 0 ? o.prefill_chunk : (backend == "cpu" ? 256 : 2048);
    eo.prefill_chunk = app->chunk;
    auto t0 = Clock::now();
    try {
        qv::ModelFile mf = qv::ModelFile::open(o.model);
        if (backend == "cpu") {
#ifdef QV_HAVE_CPU
            app->eng = qv::make_cpu_engine(mf, eo);
#else
            std::fprintf(stderr, "this build has no CPU engine\n"); return 2;
#endif
        } else {
#ifdef QV_HAVE_CUDA
            app->eng = qv::make_cuda_engine(mf, eo);
#else
            std::fprintf(stderr, "this build has no CUDA engine (configure with -DQV_CUDA=ON)\n"); return 2;
#endif
        }
        app->eng->sync();
    } catch (const std::exception& e) { std::fprintf(stderr, "model load failed: %s\n", e.what()); return 1; }
    app->backend_name = backend;
    std::fprintf(stderr, "[qv_server] model loaded in %.1fs (backend=%s, threads=%d, max_ctx=%d, tokenizer=%s)\n",
                 std::chrono::duration<double>(Clock::now() - t0).count(), backend.c_str(), eo.threads, o.max_ctx, tpath.c_str());

    App* ap = app.get();
    qv::http::Server srv(o.host, o.port, o.http_threads, [ap](const qv::http::Request& r, qv::http::Conn& c) { ap->route(r, c); });
    std::string err;
    if (!srv.listen_on(err)) { std::fprintf(stderr, "listen %s:%d failed: %s\n", o.host.c_str(), o.port, err.c_str()); return 1; }
    g_server = &srv;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    std::fprintf(stderr, "[qv_server] listening on http://%s:%d (model id \"%s\")\n", o.host.c_str(), srv.port(), o.model_name.c_str());
    srv.run();
    g_server = nullptr;
    std::fprintf(stderr, "[qv_server] shut down\n");
    return 0;
}
