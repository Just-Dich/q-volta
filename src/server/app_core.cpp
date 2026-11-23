// Routing, middleware and the non-generation endpoints (health, props, slots, metrics, models, tokenize, ...).
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "server/app.h"

namespace qv::server {

namespace {
const char* kJson = "application/json; charset=utf-8";

std::string read_all(const std::string& p, bool& ok) {
    std::ifstream f(p, std::ios::binary);
    ok = (bool)f;
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::string mime_of(const std::string& p) {
    auto ends = [&](const char* e) { size_t n = std::strlen(e); return p.size() >= n && p.compare(p.size() - n, n, e) == 0; };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js") || ends(".mjs")) return "text/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".json")) return "application/json; charset=utf-8";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".txt")) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}
}  // namespace

void App::init_slots() {
    n_slots = o.n_parallel < 0 ? 4 : std::max(1, o.n_parallel);
    slots_.assign(n_slots, SlotInfo());
    for (int i = 0; i < n_slots; ++i) slots_[i].id = i;
    m_.t_start = (long long)std::time(nullptr);
}

Value App::model_json() const {
    Value m = Value::object();
    m.set("id", model_id);
    Value al = Value::array();
    for (auto& a : o.alias.empty() ? std::vector<std::string>{model_id} : o.alias) al.push(a);
    m.set("aliases", std::move(al));
    m.set("tags", Value::array());
    m.set("object", "model");
    Value arch = Value::object();
    Value in = Value::array(), out = Value::array();
    in.push("text"); if (venc) in.push("image"); out.push("text");
    arch.set("input_modalities", std::move(in));
    arch.set("output_modalities", std::move(out));
    m.set("architecture", std::move(arch));
    m.set("created", started_at);
    m.set("owned_by", "q-volta");
    Value meta = Value::object();
    meta.set("vocab_type", 2);
    meta.set("n_vocab", tok.vocab_size());
    meta.set("n_ctx", n_ctx);
    meta.set("n_ctx_train", model_arch == "mock" ? n_ctx : 262144);
    meta.set("n_embd", eng ? eng->config().hidden : 0);
    meta.set("n_params", 0);
    meta.set("size", eng ? (long long)eng->device_bytes_weights() : 0ll);
    meta.set("ftype", model_arch == "mock" ? "mock" : "Q-Volta");
    m.set("meta", std::move(meta));
    return m;
}

void App::handle_models(http::Conn& c) {
    Value m = model_json();
    Value models = Value::array();
    Value e = Value::object();
    e.set("name", model_id);
    e.set("model", model_id);
    e.set("modified_at", "");
    e.set("size", "");
    e.set("digest", "");
    e.set("type", "model");
    e.set("description", "");
    Value tags = Value::array();
    tags.push("");
    e.set("tags", std::move(tags));
    Value caps = Value::array();
    caps.push("completion");
    e.set("capabilities", std::move(caps));
    e.set("parameters", "");
    Value det = Value::object();
    det.set("parent_model", "");
    det.set("format", "qv");
    det.set("family", "");
    Value fam = Value::array();
    fam.push("");
    det.set("families", std::move(fam));
    det.set("parameter_size", "");
    det.set("quantization_level", "");
    e.set("details", std::move(det));
    models.push(std::move(e));
    Value r = Value::object();
    r.set("models", std::move(models));
    r.set("object", "list");
    Value data = Value::array();
    data.push(std::move(m));
    r.set("data", std::move(data));
    respond_json(c, 200, r);
}

void App::handle_props(http::Conn& c) {
    Value r = Value::object();
    Task dt;
    parse_sampling(Value::object(), dt);
    dt.n_predict = o.n_predict;
    dt.reasoning_format = o.reasoning_format == "auto" ? "deepseek" : o.reasoning_format;
    Value dg = Value::object();
    dg.set("params", gen_settings(dt, 1));
    dg.set("n_ctx", n_ctx);
    r.set("default_generation_settings", std::move(dg));
    r.set("total_slots", n_slots);
    r.set("model_alias", model_id);
    r.set("model_ftype", model_arch == "mock" ? "mock" : "Q-Volta");
    r.set("model_path", o.model);
    Value mod = Value::object();
    mod.set("vision", venc != nullptr);
    mod.set("video", false);
    mod.set("audio", false);
    r.set("modalities", std::move(mod));
    r.set("media_marker", "<__media_qv__>");
    r.set("endpoint_slots", o.slots);
    r.set("endpoint_props", o.props);
    r.set("endpoint_metrics", o.metrics);
    r.set("ui", !o.path.empty());
    r.set("ui_settings", Value::object());
    r.set("chat_template", tmpl ? tmpl->source() : std::string());
    r.set("chat_template_caps", tmpl ? tmpl->caps() : Value::object());
    r.set("bos_token", bos_id >= 0 ? tok.token_bytes(bos_id) : std::string());
    r.set("eos_token", eos_id >= 0 ? tok.token_bytes(eos_id) : std::string());
    r.set("build_info", build_info);
    r.set("is_sleeping", false);
    r.set("cors_proxy_enabled", false);
    respond_json(c, 200, r);
}

void App::handle_slots(const http::Request& req, http::Conn& c) {
    if (!o.slots) throw_error(501, "This server does not support slots endpoint. Start it with `--slots`", "not_supported_error");
    Value arr = Value::array();
    int idle = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : slots_) {
            Value e = Value::object();
            e.set("id", s.id);
            e.set("n_ctx", n_ctx);
            e.set("speculative", false);
            e.set("is_processing", s.processing);
            if (!s.processing) ++idle;
            if (s.has_task) {
                e.set("id_task", (long long)s.id_task);
                e.set("n_prompt_tokens", s.n_prompt);
                e.set("n_prompt_tokens_processed", s.n_prompt_processed);
                e.set("n_prompt_tokens_cache", s.n_prompt_cache);
                e.set("params", s.params);
                Value nt = Value::object();
                nt.set("has_next_token", s.has_next_token);
                nt.set("has_new_line", s.has_new_line);
                nt.set("n_remain", s.n_remain);
                nt.set("n_decoded", s.n_decoded);
                Value na = Value::array();
                na.push(std::move(nt));
                e.set("next_token", std::move(na));
            }
            arr.push(std::move(e));
        }
    }
    if (!req.param("fail_on_no_slot").empty() && idle == 0) throw_error(503, "no slot available", "unavailable_error");
    respond_json(c, 200, arr);
}

std::string App::slot_path(const std::string& f) const { return o.slot_save_path + (o.slot_save_path.empty() || o.slot_save_path.back() == '/' ? "" : "/") + f; }

void App::handle_slot_action(const http::Request& req, http::Conn& c, int id) {
    if (o.slot_save_path.empty()) throw_error(501, "This server does not support slots action. Start it with `--slot-save-path`", "not_supported_error");
    if (id < 0 || id >= n_slots) throw_error(400, "Invalid slot ID");
    std::string action = req.param("action");
    if (action != "save" && action != "restore" && action != "erase") throw_error(400, "Invalid action");
    auto alive = [&] { return !c.peer_closed(); };
    Value body = Value::object();
    if (action != "erase") {
        try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
        const Value* f = body.is_object() ? body.find("filename") : nullptr;
        if (!f || !f->is_string()) throw_error(400, "[json.exception.out_of_range.403] key 'filename' not found");
        const std::string& fn = f->as_string();
        if (fn.empty() || fn.find('/') != std::string::npos || fn.find('\\') != std::string::npos || fn.find("..") != std::string::npos) throw_error(400, "Invalid filename");
    }
    long ticket = lease_acquire(alive);
    if (ticket < 0) return;
    struct Rel { App& a; long t; ~Rel() { a.lease_release(t); } } rel{*this, ticket};
    auto t0 = Clock::now();
    auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
    Value r = Value::object();
    r.set("id_slot", id);
    if (action == "erase") {
        long n = (long)cached_.size();
        eng->reset();
        cached_.clear();
        r.set("n_erased", n);
    } else if (action == "save") {
        for (int c : cached_) if (c < 0) throw_error(400, "Unable to save slot: it holds image tokens (images are not serialised)");
        const std::string fn = body.find("filename")->as_string();
        FILE* f = std::fopen(slot_path(fn).c_str(), "wb");
        if (!f) throw_error(500, "Unable to save slot", "server_error");
        // Q-Volta slot file: the token sequence of the slot. The recurrent GatedDeltaNet state is not serialised; restore re-evaluates the tokens.
        const char magic[8] = {'Q', 'V', 'S', 'L', 'O', 'T', '1', '\n'};
        int32_t n = (int32_t)cached_.size();
        std::fwrite(magic, 1, 8, f);
        std::fwrite(&n, 4, 1, f);
        std::fwrite(cached_.data(), 4, cached_.size(), f);
        long bytes = std::ftell(f);
        std::fclose(f);
        r.set("filename", fn);
        r.set("n_saved", (long)cached_.size());
        r.set("n_written", bytes);
        Value tm = Value::object();
        tm.set("save_ms", ms());
        r.set("timings", std::move(tm));
    } else {
        const std::string fn = body.find("filename")->as_string();
        FILE* f = std::fopen(slot_path(fn).c_str(), "rb");
        if (!f) throw_error(400, "Unable to restore slot, no available space in KV cache or invalid slot save file");
        char magic[8];
        int32_t n = 0;
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "QVSLOT1\n", 8) != 0 || std::fread(&n, 4, 1, f) != 1 || n < 0 || n >= n_ctx) { std::fclose(f); throw_error(400, "Unable to restore slot, no available space in KV cache or invalid slot save file"); }
        std::vector<int> toks(n);
        if (n && std::fread(toks.data(), 4, n, f) != (size_t)n) { std::fclose(f); throw_error(400, "Unable to restore slot, no available space in KV cache or invalid slot save file"); }
        std::fclose(f);
        eng->reset();
        cached_.clear();
        for (size_t off = 0; off < toks.size(); off += chunk) {
            int k = (int)std::min<size_t>(chunk, toks.size() - off);
            eng->prefill(toks.data() + off, k, (int)off);
            cached_.insert(cached_.end(), toks.begin() + off, toks.begin() + off + k);
        }
        r.set("filename", fn);
        r.set("n_restored", n);
        r.set("n_read", 12 + 4 * (long)n);
        Value tm = Value::object();
        tm.set("restore_ms", ms());
        r.set("timings", std::move(tm));
    }
    respond_json(c, 200, r);
}

void App::handle_metrics(http::Conn& c) {
    if (!o.metrics) throw_error(501, "This server does not support metrics endpoint. Start it with `--metrics`", "not_supported_error");
    std::ostringstream os;
    Metrics m;
    int processing = 0, deferred;
    {
        std::lock_guard<std::mutex> lk(mu_);
        m = m_;
        for (auto& s : slots_) processing += s.processing;
        deferred = waiting_;
        m_.b_prompt_tokens = m_.b_prompt_s = m_.b_pred_tokens = m_.b_pred_s = m_.b_busy_sum = m_.b_decode = 0;  // gauges: window between scrapes
    }
    auto metric = [&](const char* type, const char* name, const char* help, double v) {
        os << "# HELP llamacpp:" << name << " " << help << "\n# TYPE llamacpp:" << name << " " << type << "\nllamacpp:" << name << " ";
        char b[64];
        std::snprintf(b, sizeof b, "%.10g", v);
        os << b << "\n";
    };
    metric("counter", "prompt_tokens_total", "Number of prompt tokens processed, excluding cached tokens", m.prompt_tokens_total);
    metric("counter", "prompt_tokens_cached_total", "Number of prompt tokens reused from the cache", m.prompt_cached_total);
    metric("counter", "prompt_seconds_total", "Total time spent processing prompts", m.prompt_seconds_total);
    metric("counter", "tokens_predicted_total", "Number of generation tokens processed", m.predicted_total);
    metric("counter", "tokens_predicted_seconds_total", "Total time spent generating tokens", m.predicted_seconds_total);
    metric("counter", "n_decode_total", "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding", (double)m.n_decode_total);
    metric("counter", "n_tokens_max", "Largest observed sequence length (prompt + generation)", (double)m.n_tokens_max);
    metric("counter", "spec_decode_num_draft_tokens_total", "Speculative: Total draft tokens generated", 0);
    metric("counter", "spec_decode_num_accepted_tokens_total", "Speculative: Total draft tokens accepted by the target model", 0);
    metric("counter", "spec_decode_num_drafts_total", "Speculative: Total speculative decoding verification steps", 0);
    metric("gauge", "prompt_tokens_seconds", "Average prompt throughput in tokens/s", m.b_prompt_s > 0 ? m.b_prompt_tokens / m.b_prompt_s : 0);
    metric("gauge", "predicted_tokens_seconds", "Average generation throughput in tokens/s", m.b_pred_s > 0 ? m.b_pred_tokens / m.b_pred_s : 0);
    metric("gauge", "requests_processing", "Number of requests processing", processing);
    metric("gauge", "requests_deferred", "Number of requests deferred", deferred);
    metric("gauge", "n_busy_slots_per_decode", "Average number of busy slots per llama_decode() call", m.b_decode > 0 ? m.b_busy_sum / m.b_decode : 0);
    c.respond(200, "text/plain; version=0.0.4", os.str(), {{"Process-Start-Time-Unix", std::to_string(m.t_start)}});
}

void App::handle_tokenize(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object()) throw_error(400, "[json.exception.type_error.302] type must be object, but is " + std::string(body.is_array() ? "array" : "string"));
    auto tn = [](const Value& v) { return v.is_string() ? "string" : v.is_number() ? "number" : v.is_bool() ? "boolean" : v.is_array() ? "array" : "object"; };
    std::string content;
    if (const Value* v = body.find("content"); v && !v->is_null()) {
        if (!v->is_string()) throw_error(400, std::string("[json.exception.type_error.302] type must be string, but is ") + tn(*v));
        content = v->as_string();
    }
    auto flag = [&](const char* k, bool d) { const Value* v = body.find(k); return v && v->is_bool() ? v->as_bool() : d; };
    bool with_pieces = flag("with_pieces", false), parse_special = flag("parse_special", true);
    flag("add_special", false);  // Qwen tokenizers add no BOS/EOS: add_special has no effect
    std::vector<int> ids = tok.encode(content, parse_special);
    Value toks = Value::array();
    for (int id : ids) {
        if (!with_pieces) { toks.push(id); continue; }
        Value e = Value::object();
        e.set("id", id);
        std::string b = tok.token_bytes(id);
        bool valid = true;
        {
            Value probe;
            std::string s = piece_text(id);
            valid = s == b;
        }
        if (valid) e.set("piece", b);
        else { Value a = Value::array(); for (unsigned char ch : b) a.push((int)ch); e.set("piece", std::move(a)); }
        toks.push(std::move(e));
    }
    Value r = Value::object();
    r.set("tokens", std::move(toks));
    respond_json(c, 200, r);
}

void App::handle_detokenize(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object()) throw_error(400, "[json.exception.type_error.302] type must be object, but is " + std::string(body.is_array() ? "array" : "string"));
    std::vector<int> ids;
    if (const Value* v = body.find("tokens"); v && !v->is_null()) {
        if (!v->is_array()) throw_error(400, std::string("[json.exception.type_error.302] type must be array, but is ") + (v->is_string() ? "string" : v->is_number() ? "number" : v->is_bool() ? "boolean" : "object"));
        for (auto& x : v->items()) {
            if (!x.is_number()) throw_error(400, "[json.exception.type_error.302] type must be number, but is string");
            long long id = x.as_int();
            if (id >= 0 && id < tok.vocab_size()) ids.push_back((int)id);
        }
    }
    Value r = Value::object();
    r.set("content", tok.decode(ids));
    respond_json(c, 200, r);
}

void App::handle_apply_template(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    Task t = parse_task(body, Kind::OaiChat);
    Value r = Value::object();
    r.set("prompt", t.prompt_text);
    respond_json(c, 200, r);
}

void App::serve_static(const http::Request& req, http::Conn& c) {
    std::string p = req.path;
    if (p.empty() || p == "/") p = "/index.html";
    if (p.find("..") != std::string::npos) throw_error(404, "File Not Found", "not_found_error");
    bool ok;
    std::string data = read_all(o.path + p, ok);
    if (!ok) throw_error(404, "File Not Found", "not_found_error");
    c.respond(200, mime_of(p), data);
}

void App::route(const http::Request& req, http::Conn& c) {
    // ---- headers common to every response
    std::string origin = req.header("origin");
    c.base_headers = {{"Server", "qv_server"}};
    if (o.cors_credentials && o.cors_origins == "*") c.base_headers.push_back({"Access-Control-Allow-Origin", origin});
    else if (o.cors_origins != "localhost") c.base_headers.push_back({"Access-Control-Allow-Origin", o.cors_origins});
    try {
        if (req.method == "OPTIONS") {
            c.respond(200, "text/html", "", {{"Access-Control-Allow-Credentials", o.cors_credentials ? "true" : "false"}, {"Access-Control-Allow-Methods", o.cors_methods}, {"Access-Control-Allow-Headers", o.cors_headers}});
            return;
        }
        std::string path = req.path;
        if (!o.api_prefix.empty()) {
            if (path.rfind(o.api_prefix, 0) == 0 && (path.size() == o.api_prefix.size() || path[o.api_prefix.size()] == '/')) path = path.substr(o.api_prefix.size());
            else throw_error(404, "File Not Found", "not_found_error");
            if (path.empty()) path = "/";
        }
        // ---- model loading
        if (!ready.load()) throw_error(503, "Loading model", "unavailable_error");
        // ---- API key (public: /health, /v1/health)
        const bool is_public = path == "/health" || path == "/v1/health";
        if (!o.api_keys.empty() && !is_public) {
            std::string k = req.header("authorization");
            if (k.empty()) k = req.header("x-api-key");
            if (k.rfind("Bearer ", 0) == 0) k = k.substr(7);
            if (std::find(o.api_keys.begin(), o.api_keys.end(), k) == o.api_keys.end()) throw_error(401, "Invalid API Key", "authentication_error");
        }
        const std::string& m = req.method;
        const bool get = m == "GET" || m == "HEAD", post = m == "POST";
        if (get && (path == "/health" || path == "/v1/health")) { Value v = Value::object(); v.set("status", "ok"); respond_json(c, 200, v); return; }
        if (get && path == "/metrics") { handle_metrics(c); return; }
        if (get && path == "/props") { handle_props(c); return; }
        if (post && path == "/props") {
            if (!o.props) throw_error(501, "This server does not support changing global properties. Start it with `--props`", "not_supported_error");
            Value v = Value::object(); v.set("success", true); respond_json(c, 200, v); return;
        }
        if (get && (path == "/models" || path == "/v1/models")) { handle_models(c); return; }
        if (post && (path == "/completion" || path == "/completions")) { handle_completion(req, c, Kind::Native); return; }
        if (post && path == "/v1/completions") { handle_completion(req, c, Kind::OaiCmpl); return; }
        if (post && (path == "/chat/completions" || path == "/v1/chat/completions")) { handle_completion(req, c, Kind::OaiChat); return; }
        if (post && path == "/infill") { handle_infill(req, c); return; }
        if (post && path == "/tokenize") { handle_tokenize(req, c); return; }
        if (post && path == "/detokenize") { handle_detokenize(req, c); return; }
        if (post && path == "/apply-template") { handle_apply_template(req, c); return; }
        if (post && (path == "/chat/completions/input_tokens" || path == "/v1/chat/completions/input_tokens")) {
            Value body;
            try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
            Task t = parse_task(body, Kind::OaiChat);
            Value v = Value::object(); v.set("input_tokens", (long long)t.prompt.size()); v.set("object", "response.input_tokens"); respond_json(c, 200, v); return;
        }
        if (get && path == "/slots") { handle_slots(req, c); return; }
        if (post && path.rfind("/slots/", 0) == 0) {
            int id;
            try { id = std::stoi(path.substr(7)); } catch (...) { throw_error(400, "Invalid slot ID"); }
            handle_slot_action(req, c, id);
            return;
        }
        if (get && path == "/lora-adapters") { respond_json(c, 200, Value::array()); return; }
        if (post && path == "/lora-adapters") {
            Value body;
            try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
            if (!body.is_array()) throw_error(400, "Request body must be an array");
            if (body.size() > 0) throw_error(400, "invalid adapter id");
            Value v = Value::object(); v.set("success", true); respond_json(c, 200, v); return;
        }
        if (post && (path == "/embedding" || path == "/embeddings" || path == "/v1/embeddings"))
            throw_error(501, o.embedding ? "Q-Volta engines do not expose pooled embeddings; /embeddings is not implemented" : "This server does not support embeddings. Start it with `--embeddings`", "not_supported_error");
        if (post && (path == "/rerank" || path == "/reranking" || path == "/v1/rerank" || path == "/v1/reranking"))
            throw_error(501, "This server does not support reranking. Start it with `--reranking`", "not_supported_error");
        if (post && (path == "/v1/messages" || path == "/v1/messages/count_tokens" || path == "/v1/responses" || path == "/responses" || path == "/v1/audio/transcriptions" ||
                     path == "/v1/audio/speech" || path == "/tts" || path == "/v1/systemone" || path == "/v1/chat/completions/control"))
            throw_error(501, "Endpoint " + path + " is not implemented by qv_server", "not_supported_error");
        if (get && (!o.path.empty())) { serve_static(req, c); return; }
        throw_error(404, "File Not Found", "not_found_error");
    } catch (const ApiError& e) {
        if (c.streaming && !c.broken) { c.write_chunk("data: " + error_json(e) + "\n\n"); c.end_chunks(); c.close_after = true; }
        else if (!c.broken) {
            std::string body = error_json(e);
            if (e.status == 404) {  // httplib's error handler uses a different key order
                Value err = Value::object();
                err.set("message", e.message); err.set("type", e.type); err.set("code", e.status);
                Value o2 = Value::object(); o2.set("error", std::move(err)); body = o2.dump();
            }
            c.respond(e.status, kJson, body);
        }
    } catch (const std::exception& e) {
        if (c.streaming && !c.broken) { c.write_chunk(std::string("data: ") + error_json(ApiError{500, e.what(), "server_error", Value()}) + "\n\n"); c.end_chunks(); }
        else if (!c.broken) c.respond(500, kJson, error_json(ApiError{500, e.what(), "server_error", Value()}));
        c.close_after = true;
    }
}

}  // namespace qv::server
