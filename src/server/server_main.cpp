// qv_server: llama-server compatible HTTP server for Q-Volta engines (CUDA, CPU, or the scripted mock engine for tests).
#include <signal.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#ifdef QV_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "server/app.h"
#include "server/mock_engine.h"

#ifndef QV_GIT_REV
#define QV_GIT_REV "dev"
#endif

using namespace qv::server;
using Clock = std::chrono::steady_clock;

namespace {
std::atomic<qv::http::Server*> g_server{nullptr};
void on_signal(int) { if (auto* s = g_server.load()) s->stop(); }

bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return (bool)f; }

void print_help() {
    std::printf(
        "usage: qv_server -m MODEL [options]\n\n"
        "qv_server is a llama-server compatible HTTP server. Every llama-server flag spelling and LLAMA_ARG_* environment variable\n"
        "of llama.cpp 9d6f04f is accepted; flags without a meaning for Q-Volta are warned about and ignored (see docs/SERVER_COMPAT.md).\n\n"
        "common flags:\n"
        "  -m, --model PATH            Q-Volta model (PREFIX or PREFIX.qv)\n"
        "  -c, --ctx-size N            context size (default 4096; per slot, all slots share one engine)\n"
        "  -t, --threads N             CPU threads (default min(32, cores))\n"
        "  -ngl, --n-gpu-layers N      0 = CPU engine, >0 / auto / all = CUDA engine when a device exists (all-or-nothing)\n"
        "  -b / -ub N                  prefill chunk size (-ub wins)\n"
        "  --host H --port P           default 127.0.0.1:9931 (llama-server default of the reference commit)\n"
        "  --api-key K[,K2] / --api-key-file F, -a/--alias NAME, -np/--parallel N, --cache-prompt/--no-cache-prompt\n"
        "  --jinja/--no-jinja, --chat-template NAME, --chat-template-kwargs JSON, --reasoning-format auto|none|deepseek|deepseek-legacy\n"
        "  --reasoning on|off|auto, --metrics, --slots/--no-slots, --props, --slot-save-path DIR, --path DIR, --no-webui\n"
        "  -n/--predict, --temp, --top-k, --top-p, --min-p, --typical, --repeat-penalty, --repeat-last-n, --presence-penalty,\n"
        "  --frequency-penalty, --mirostat, --xtc-*, --dry-*, --dynatemp-*, -s/--seed, --samplers, --ignore-eos, -l/--logit-bias,\n"
        "  --grammar, --json-schema, -to/--timeout, --cors-*, --api-prefix, --log-disable, -v\n\n"
        "Q-Volta flags:\n"
        "  --backend cuda|cpu|auto|mock   --kv fp32|fp16|int4   --mtp / --mtp-depth N / --mtp-file PATH (Inkling)   --prefill-chunk N   --tokenizer PATH\n"
        "  --max-queue N (503 beyond N waiting requests; default unbounded like llama-server)\n"
        "  mock backend (tests): --mock-text TEXT --mock-loop --mock-delay-us N (env QV_MOCK_LOAD_MS delays loading)\n");
}
}  // namespace

int main(int argc, char** argv) {
    Options o;
    bool exit_now = false;
    int rc = parse_args(argc, argv, o, exit_now);
    if (exit_now) {
        if (rc == -1) print_help();
        else std::printf("version: qv-%s (qv_server, llama-server compatible; reference llama.cpp 9d6f04f)\n", QV_GIT_REV);
        return 0;
    }
    if (rc) return rc;
    for (auto& w : o.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    signal(SIGPIPE, SIG_IGN);

    auto app = std::make_unique<App>();
    app->build_info = std::string("qv-") + QV_GIT_REV;
    if (o.ctx <= 0) o.ctx = 4096;
    app->n_ctx = o.ctx;
    app->o = o;
    app->model_id = !o.alias.empty() ? o.alias[0] : (o.model.empty() ? std::string("qv-mock") : o.model);
    app->started_at = (long long)std::time(nullptr);
    app->init_slots();
    std::string host = o.host;
    if (auto c = host.find(','); c != std::string::npos) host = host.substr(0, c);

    App* ap = app.get();
    const int max_conns = o.http_threads > 0 ? std::max(o.http_threads, ap->n_slots + 2) : 256;
    qv::http::Server srv(host, o.port, max_conns, [ap](const qv::http::Request& r, qv::http::Conn& c) { ap->route(r, c); });
    srv.set_timeout(o.timeout);
    std::string err;
    if (!srv.listen_on(err)) { std::fprintf(stderr, "couldn't bind HTTP server socket, hostname: %s, port: %d (%s)\n", host.c_str(), o.port, err.c_str()); return 1; }
    g_server = &srv;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    std::thread http_thread([&] { srv.run(); });
    std::fprintf(stderr, "[qv_server] listening on http://%s:%d (loading model)\n", host.c_str(), srv.port());

    // ---- model loading (the HTTP port is already up: /health answers 503 "Loading model" until ready)
    auto t0 = Clock::now();
    auto fail = [&](const std::string& m) {
        std::fprintf(stderr, "error: %s\n", m.c_str());
        srv.stop();
        http_thread.join();
        return 1;
    };
    std::string stem = o.model;
    if (stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".qv") == 0 && !file_exists(stem)) stem.resize(stem.size() - 3);
    std::string tpath = o.tokenizer;
    if (tpath.empty() && !o.model.empty()) {
        std::string base = o.model;
        if (base.size() > 3 && base.compare(base.size() - 3, 3, ".qv") == 0) base.resize(base.size() - 3);
        size_t sl = base.find_last_of('/');
        std::string dir = sl == std::string::npos ? "." : base.substr(0, sl);
        for (std::string cand : {base + ".tokenizer.json", dir + "/../Qwen3.8-27B/tokenizer.json", dir + "/tokenizer.json"})
            if (file_exists(cand)) { tpath = cand; break; }
    }
    if (tpath.empty()) return fail("cannot find tokenizer.json; pass --tokenizer");
    try { app->tok = qv::Tokenizer::load(tpath); } catch (const std::exception& e) { return fail(std::string("tokenizer: ") + e.what()); }
    app->tokenizer_path = tpath;
    for (const char* s : {"<|im_end|>", "<|endoftext|>"}) { int id = app->tok.token_id(s); if (id >= 0) app->stop_ids.push_back(id); }
    app->eos_id = app->tok.token_id("<|im_end|>");
    app->bos_id = app->tok.token_id("<|endoftext|>");
    app->fim_pre = app->tok.token_id("<|fim_prefix|>");
    app->fim_suf = app->tok.token_id("<|fim_suffix|>");
    app->fim_mid = app->tok.token_id("<|fim_middle|>");
    app->fim_rep = app->tok.token_id("<|repo_name|>");
    app->fim_sep = app->tok.token_id("<|file_sep|>");
    {
        std::string tp = tpath.substr(0, tpath.find_last_of('/') == std::string::npos ? 0 : tpath.find_last_of('/')) + "/chat_template.jinja";
        std::ifstream f(tp, std::ios::binary);
        if (f) { std::stringstream ss; ss << f.rdbuf(); app->model_template = ss.str(); }
    }
    try {
        std::string warn, over = o.chat_template;
        if (!o.chat_template_file.empty()) {
            std::ifstream f(o.chat_template_file, std::ios::binary);
            if (!f) return fail("failed to open chat template file " + o.chat_template_file);
            std::stringstream ss; ss << f.rdbuf(); over = ss.str();
        }
        app->tmpl = std::make_unique<ChatTemplates>(app->model_template, over, o.jinja, warn);
        if (!warn.empty()) std::fprintf(stderr, "warning: %s\n", warn.c_str());
    } catch (const std::exception& e) { return fail(e.what()); }

    std::string backend = o.backend;
#ifdef QV_HAVE_CUDA
    if (backend == "auto") {
        int n = 0;
        backend = (!(o.ngl_given && o.n_gpu_layers == 0) && cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? "cuda" : "cpu";
    }
#else
    if (backend == "auto") backend = "cpu";
#endif
    if (backend != "cpu" && backend != "cuda" && backend != "mock") return fail("--backend must be cuda, cpu, auto or mock");
    qv::EngineOptions eo;
    eo.max_ctx = o.ctx;
    eo.kv_mode = o.kv_mode;
    eo.kv4_kmode = o.kv4_kmode;
    eo.mtp = o.mtp;
    eo.mtp_depth = o.mtp_depth;
    eo.threads = o.threads > 0 ? o.threads : (int)std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
    app->chunk = o.prefill_chunk > 0 ? o.prefill_chunk : (o.ubatch > 0 ? o.ubatch : (backend == "cuda" ? 2048 : 256));
    eo.prefill_chunk = app->chunk;
    try {
        if (backend == "mock") {
            if (const char* ms = std::getenv("QV_MOCK_LOAD_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(ms)));
            auto script = app->tok.encode(o.mock_text, false);
            app->eng = std::make_unique<qv::MockEngine>(app->tok.vocab_size(), script, app->eos_id, o.mock_loop, o.mock_delay_us);
            app->model_arch = "mock";
        } else {
            if (o.model.empty()) return fail("no model: pass -m MODEL (or set LLAMA_ARG_MODEL)");
            qv::ModelFile mf = qv::ModelFile::open(stem);
            if (eo.mtp && mf.cfg.arch == "inkling2_loop") {  // Inkling MTP drafter: sidecar container, default <model>-mtp.qv; depth 3 unless given (depth 1 is slower than plain)
                std::string base = stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".qv") == 0 ? stem.substr(0, stem.size() - 3) : stem;
                auto exists = [](const std::string& f) { return std::ifstream(f).good(); };
                if (!o.mtp_file.empty()) eo.mtp_file = o.mtp_file;
                else if (exists(base + "-mtp.qv")) eo.mtp_file = base + "-mtp.qv";
                else {  // quant suffix after the last '-' dropped: inkling2-mini-q3a5.qv -> inkling2-mini-mtp.qv
                    const size_t d = base.rfind('-'), sl = base.rfind('/');
                    const std::string alt = d != std::string::npos && (sl == std::string::npos || d > sl) ? base.substr(0, d) + "-mtp.qv" : "";
                    if (!alt.empty() && exists(alt)) eo.mtp_file = alt;
                    else return fail("--mtp: MTP sidecar not found (tried " + base + "-mtp.qv" + (alt.empty() ? "" : " and " + alt) + "); pass --mtp-file PATH");
                }
                app->mtp_chunk = 32;  // a speculative cycle commits up to depth+1 tokens; short chunks end in slow plain tail steps
                if (eo.mtp_depth == 1) eo.mtp_depth = 3;
            }
            if (backend == "cpu") {
#ifdef QV_HAVE_CPU
                app->eng = qv::make_cpu_engine(mf, eo);
#else
                return fail("this build has no CPU engine");
#endif
            } else {
#ifdef QV_HAVE_CUDA
                app->eng = qv::make_cuda_engine(mf, eo);
#else
                return fail("this build has no CUDA engine (configure with -DQV_CUDA=ON)");
#endif
            }
            app->model_arch = app->eng->config().arch;
        }
        app->eng->sync();
    } catch (const std::exception& e) { return fail(std::string("model load failed: ") + e.what()); }
    app->backend_name = backend;
    std::fprintf(stderr, "[qv_server] model loaded in %.1fs (backend=%s, threads=%d, n_ctx=%d, slots=%d, mtp=%d, tokenizer=%s)\n", std::chrono::duration<double>(Clock::now() - t0).count(),
                 backend.c_str(), eo.threads, o.ctx, app->n_slots, (int)o.mtp, tpath.c_str());
    app->ready = true;
    std::fprintf(stderr, "[qv_server] server is listening on http://%s:%d (model id \"%s\")\n", host.c_str(), srv.port(), app->model_id.c_str());
    http_thread.join();
    g_server = nullptr;
    std::fprintf(stderr, "[qv_server] shut down\n");
    return 0;
}
