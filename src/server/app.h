// qv_server application: llama-server compatible routes on top of a single stateful Q-Volta engine.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

#include "core/json_doc.h"
#include "qv/engine.h"
#include "sampling/sampler.h"
#include "server/chat_render.h"
#include "server/http.h"
#include "server/options.h"
#include "tokenizer/tokenizer.h"

namespace qv::server {

using jd::Value;
using Clock = std::chrono::steady_clock;

struct ApiError {
    int status;
    std::string message, type;
    Value extra;  // object merged into the error object (n_prompt_tokens, n_ctx, ...)
};
[[noreturn]] void throw_error(int status, const std::string& msg, const std::string& type = "");
std::string error_json(const ApiError& e);

enum class Kind { Native, OaiCmpl, OaiChat, Infill };

struct TokProb {
    int id;
    std::string text, bytes_text;
    std::vector<int> bytes;
    float logprob, prob;
    std::vector<std::tuple<int, std::string, std::vector<int>, float>> top;  // id, text, bytes, logprob or prob
};

struct Task {
    Kind kind = Kind::Native;
    std::vector<int> prompt;
    std::string prompt_text;
    int n_predict = -1;
    std::vector<std::string> stops;
    bool cache_prompt = true, stream = false, return_tokens = false, timings_per_token = false, post_probs = false, ignore_eos = false;
    int id_slot = -1, n_probs = 0, n_keep = 0, n_choices = 1;
    SamplingParams sp;
    std::string grammar;
    Value json_schema, response_fields, logit_bias_echo, stop_echo;
    // chat
    bool include_usage = false, thinking_open = false;
    std::string reasoning_format = "none";  // none | deepseek | deepseek-legacy (resolved)
    std::string think_start = "<think>", think_end = "</think>";
    std::string model_echo;
    bool logprobs_oai = false;
};

struct SlotInfo {
    int id = 0;
    bool processing = false, has_task = false;
    long id_task = -1;
    int n_prompt = 0, n_prompt_processed = 0, n_prompt_cache = 0, n_decoded = 0, n_remain = 0;
    bool has_next_token = false, has_new_line = false;
    Value params;
};

struct GenResult {
    std::string content;
    std::vector<int> tokens;
    std::vector<TokProb> probs;
    std::string stop_type = "none", stopping_word;  // none | eos | word | limit
    bool truncated = false, aborted = false, has_new_line = false;
    int n_prompt = 0, n_cache = 0, n_prompt_eval = 0, n_decoded = 0, tokens_cached = 0, id_slot = 0;
    double prompt_ms = 0, predicted_ms = 0;
};

struct Chunk {
    std::string text;
    std::vector<int> tokens;
    std::vector<TokProb> probs;
    int n_decoded = 0;
};

class App {
   public:
    Options o;
    Tokenizer tok;
    std::unique_ptr<Engine> eng;
    std::unique_ptr<ChatTemplates> tmpl;
    std::string backend_name, tokenizer_path, model_id, model_template, model_arch;
    std::vector<int> stop_ids;
    int n_ctx = 4096, chunk = 256, n_slots = 4, mtp_chunk = 8;
    std::atomic<bool> ready{false};
    long long started_at = 0;
    std::string build_info = "qv-dev";
    int bos_id = -1, eos_id = -1;
    int fim_pre = -1, fim_suf = -1, fim_mid = -1, fim_rep = -1, fim_sep = -1;

    void init_slots();
    void route(const http::Request& req, http::Conn& c);

   private:
    // ---- engine queue + slots
    std::mutex mu_;
    std::condition_variable cv_;
    long next_ticket_ = 0, serving_ = 0;
    int waiting_ = 0;
    std::set<long> skipped_;
    std::vector<SlotInfo> slots_;
    std::vector<int> cached_;  // tokens inside the engine (logits valid for exactly this context)
    int owner_slot_ = 0;
    long task_counter_ = 0;
    // metrics
    struct Metrics {
        long long t_start = 0;
        double prompt_tokens_total = 0, prompt_cached_total = 0, prompt_seconds_total = 0, predicted_total = 0, predicted_seconds_total = 0;
        long long n_decode_total = 0, n_tokens_max = 0;
        double b_prompt_tokens = 0, b_prompt_s = 0, b_pred_tokens = 0, b_pred_s = 0, b_busy_sum = 0, b_decode = 0;
    } m_;
    std::string slot_path(const std::string& fname) const;

    // ---- helpers
    std::string piece_text(int id) const;
    std::string piece_bytes(int id, std::vector<int>& bytes) const;
    Value gen_settings(const Task& t, int flavor) const;
    Value timings_json(const GenResult& r) const;
    Value probs_json(const std::vector<TokProb>& v, bool post) const;
    Task parse_task(const Value& body, Kind kind);
    void parse_sampling(const Value& body, Task& t);
    std::vector<int> tokenize_prompt_value(const Value& p);
    std::string template_prefix_json(const Value&);

    // ---- generation
    long lease_acquire(const std::function<bool()>& alive);  // FIFO turn on the engine; -1 if the client left while waiting
    void lease_release(long ticket);
    struct Admission;
    GenResult generate(Task& t, const std::function<bool()>& alive, const std::function<bool(const Chunk&)>& on_chunk, const std::function<void(int)>& on_start);
    // ---- handlers
    void handle_completion(const http::Request&, http::Conn&, Kind);
    void handle_infill(const http::Request&, http::Conn&);
    void handle_tokenize(const http::Request&, http::Conn&);
    void handle_detokenize(const http::Request&, http::Conn&);
    void handle_apply_template(const http::Request&, http::Conn&);
    void handle_props(http::Conn&);
    void handle_slots(const http::Request&, http::Conn&);
    void handle_slot_action(const http::Request&, http::Conn&, int id);
    void handle_metrics(http::Conn&);
    void handle_models(http::Conn&);
    void handle_messages(const http::Request&, http::Conn&);
    void respond_json(http::Conn& c, int status, const Value& v, const http::Headers& extra = {});
    void serve_static(const http::Request& req, http::Conn& c);
    Value model_json() const;
};

}  // namespace qv::server
