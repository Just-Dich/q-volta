// Resumable SSE streams: GET /v1/stream, POST /v1/streams/lookup, DELETE /v1/stream (llama-server server-stream.cpp).
#include "server/app.h"

namespace qv::server {

namespace {
constexpr size_t kMaxBytes = 4 * 1024 * 1024;
constexpr long long kTtlSeconds = 300;
long long now_s() { return (long long)std::time(nullptr); }
}  // namespace

StreamSession::StreamSession(std::string i) : id(std::move(i)), started_at(now_s()) {}

bool StreamSession::append(const char* d, size_t n) {
    if (n == 0) return true;
    {
        std::lock_guard<std::mutex> g(mu_);
        if (done_) return false;
        if (n >= kMaxBytes) { dropped_ += buf_.size() + (n - kMaxBytes); buf_.assign(d + (n - kMaxBytes), kMaxBytes); }
        else {
            if (buf_.size() + n > kMaxBytes) { size_t drop = buf_.size() + n - kMaxBytes; buf_.erase(0, drop); dropped_ += drop; }
            buf_.append(d, n);
        }
    }
    cv_.notify_all();
    return true;
}
void StreamSession::finalize() {
    {
        std::lock_guard<std::mutex> g(mu_);
        if (done_) return;
        done_ = true;
        completed_ = now_s();
    }
    cv_.notify_all();
}
bool StreamSession::done() const { std::lock_guard<std::mutex> g(mu_); return done_; }
size_t StreamSession::total() const { std::lock_guard<std::mutex> g(mu_); return dropped_ + buf_.size(); }
size_t StreamSession::dropped() const { std::lock_guard<std::mutex> g(mu_); return dropped_; }
long long StreamSession::completed_at() const { std::lock_guard<std::mutex> g(mu_); return completed_; }

bool StreamSession::read_from(size_t& offset, const std::function<bool(const char*, size_t)>& sink, const std::function<bool()>& should_stop) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        if (should_stop && should_stop()) return true;
        if (offset < dropped_) return false;
        if (offset < dropped_ + buf_.size()) {
            std::string chunk = buf_.substr(offset - dropped_);
            offset += chunk.size();
            lk.unlock();
            if (!sink(chunk.data(), chunk.size())) return true;
            lk.lock();
            continue;
        }
        if (done_) return true;
        cv_.wait_for(lk, std::chrono::milliseconds(200));
    }
}

StreamSessionPtr App::stream_create(const std::string& id) {
    auto fresh = std::make_shared<StreamSession>(id);
    StreamSessionPtr prev;
    {
        std::lock_guard<std::mutex> g(sm_);
        // drop sessions that finished more than 300 s ago
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            long long c = it->second->completed_at();
            if (c != 0 && c <= now_s() - kTtlSeconds) it = sessions_.erase(it); else ++it;
        }
        auto it = sessions_.find(id);
        if (it != sessions_.end()) { prev = it->second; it->second = fresh; } else sessions_[id] = fresh;
    }
    if (prev) { prev->cancel(); prev->finalize(); }  // one conversation = at most one live session
    return fresh;
}

StreamSessionPtr App::stream_get(const std::string& id) {
    std::lock_guard<std::mutex> g(sm_);
    auto it = sessions_.find(id);
    return it == sessions_.end() ? nullptr : it->second;
}

void App::handle_stream_get(const http::Request& req, http::Conn& c) {
    std::string id = req.param("conv_id");
    if (id.empty()) throw_error(400, "Missing conversation id in path");
    auto s = stream_get(id);
    if (!s) throw_error(404, "Stream not found or expired", "not_found_error");
    size_t from = 0;
    std::string f = req.param("from");
    if (!f.empty()) { try { from = (size_t)std::stoull(f); } catch (...) { throw_error(400, "Invalid 'from' offset"); } }
    if (from < s->dropped()) throw_error(400, "Stream offset lost, please restart");
    if (!c.begin_stream(200, "text/event-stream")) return;
    bool ok = true;
    s->read_from(from, [&](const char* d, size_t n) { ok = c.write_chunk(std::string_view(d, n)); return ok; }, [&] { return c.peer_closed() || s->cancelled(); });
    if (ok) c.end_chunks(); else c.close_after = true;
}

void App::handle_stream_lookup(const http::Request& req, http::Conn& c) {
    std::vector<std::string> ids;
    try {
        Value b = jd::parse(req.body);
        if (const Value* v = b.find("conversation_ids"); v && v->is_array())
            for (auto& x : v->items()) if (x.is_string() && !x.as_string().empty()) ids.push_back(x.as_string());
    } catch (const std::exception& e) {
        Value err = Value::object(), o2 = Value::object();
        err.set("message", std::string("invalid body: ") + e.what());
        err.set("type", "invalid_request_error");
        o2.set("error", std::move(err));
        respond_json(c, 400, o2);
        return;
    }
    Value arr = Value::array();
    std::lock_guard<std::mutex> g(sm_);
    for (auto& rid : ids)
        for (auto& [k, s] : sessions_)
            if (k == rid || k.compare(0, rid.size() + 2, rid + "::") == 0) {
                Value e = Value::object();
                e.set("conversation_id", k);
                e.set("is_done", s->done());
                e.set("total_bytes", (long long)s->total());
                e.set("started_at", s->started_at);
                e.set("completed_at", s->completed_at());
                arr.push(std::move(e));
            }
    respond_json(c, 200, arr);
}

void App::handle_stream_delete(const http::Request& req, http::Conn& c) {
    std::string id = req.param("conv_id");
    if (id.empty()) throw_error(400, "Missing conversation id in path");
    StreamSessionPtr s;
    {
        std::lock_guard<std::mutex> g(sm_);
        auto it = sessions_.find(id);
        if (it != sessions_.end()) { s = it->second; sessions_.erase(it); }
    }
    if (s) { s->cancel(); s->finalize(); }
    c.respond(204, "application/json", "");
}

}  // namespace qv::server
