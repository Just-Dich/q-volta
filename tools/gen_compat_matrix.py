#!/usr/bin/env python3
"""Generate the flag part of docs/SERVER_COMPAT.md from src/server/llama_flags.json (tools/gen_llama_flags.py) + the status rules below.
usage: gen_compat_matrix.py src/server/llama_flags.json docs/SERVER_COMPAT.head.md docs/SERVER_COMPAT.md
(the head file holds the hand-written endpoint / behaviour sections; the generated flag table is appended)"""
import json, re, sys

flags = json.load(open(sys.argv[1]))
head = open(sys.argv[2]).read()
src = open(__file__.replace("tools/gen_compat_matrix.py", "src/server/options.cpp")).read()

I, P, N, U = "implemented", "partial", "n/a", "unsupported"
# canonical name -> (status, note)
S = {
    "model": (I, "Q-Volta model prefix or `.qv` file (not GGUF)"),
    "ctx-size": (I, "0 = 4096 (default). All slots share one engine context: `n_ctx` per slot = the full context (llama-server's unified-KV layout)"),
    "threads": (I, "CPU engine worker threads (default min(32, cores))"),
    "threads-http": (P, "caps concurrent connections (one thread per connection), default 256"),
    "gpu-layers": (P, "all-or-nothing: `0` selects the CPU engine, any other value (incl. `auto`/`all`) the CUDA engine if a device exists; `--backend` overrides"),
    "batch-size": (P, "only used as the infill batch size; prefill chunk comes from `-ub` / `--prefill-chunk`"),
    "ubatch-size": (P, "prefill chunk size (also the granularity at which a client disconnect is noticed during prefill)"),
    "host": (P, "IPv4 address or `localhost`; first entry of a comma list; no IPv6 / UNIX sockets"),
    "port": (I, "default 9931 like the reference commit"),
    "api-key": (I, "comma list; `Authorization: Bearer` or `X-Api-Key`; `/health` and `/v1/health` stay public"),
    "api-key-file": (I, "one key per line, `#` comments"),
    "alias": (I, "first alias is the model id in `/v1/models`, `/props`, responses"),
    "parallel": (P, "N logical slots reported by `/slots`/`/props`; the engine holds ONE sequence, requests are served FIFO (queued, not rejected). True concurrent decoding is a future engine feature"),
    "cont-batching": (N, "accepted; single-sequence engine"),
    "cache-prompt": (I, "prefix reuse of the engine state (see docs/SERVER.md for the recurrent-state limits)"),
    "jinja": (I, "Jinja interpreter (src/jinja) evaluates the model's chat template; `--no-jinja` selects the native Qwen3.8 renderer (no tools)"),
    "chat-template": (P, "`chatml` or a template text (Jinja source); other llama built-in names (llama3, mistral-v7, ...) are refused at startup"),
    "chat-template-file": (I, "Jinja template file"),
    "chat-template-kwargs": (I, "merged under per-request `chat_template_kwargs`; `enable_thinking`, `reasoning_effort`, `preserve_thinking` are honoured by the Qwen3.8 template"),
    "reasoning-format": (I, "`auto` (=deepseek), `none`, `deepseek`, `deepseek-legacy`; legacy spelling `separate` accepted"),
    "reasoning": (P, "`off` -> `enable_thinking=false`; `on`/`auto` -> template default"),
    "reasoning-budget": (P, "only `0` (thinking off) is honoured; positive budgets are not enforced"),
    "reasoning-effort": (I, "passed to the Qwen3.8 template (`low|medium|xhigh`; OpenAI `high` -> `xhigh`, `minimal` -> `low`)"),
    "metrics": (I, "Prometheus endpoint with llama.cpp's metric names"),
    "slots": (I, ""),
    "props": (I, "POST /props accepted (no mutable property exists, same as llama-server)"),
    "embedding": (P, "flag accepted; `/embedding(s)` answer 501 (the engines do not expose pooled hidden states)"),
    "predict": (I, "default `n_predict` (-1 = until EOS / context end)"),
    "temp": (I, ""), "top-k": (I, ""), "top-p": (I, ""), "min-p": (I, ""), "typical": (I, ""), "top-nsigma": (I, ""), "xtc-probability": (I, ""),
    "xtc-threshold": (I, ""), "repeat-penalty": (I, ""), "repeat-last-n": (I, ""), "presence-penalty": (I, ""), "frequency-penalty": (I, ""),
    "dry-multiplier": (P, "DRY uses a simplified suffix matcher (single-token sequence breakers, match length capped at 64)"),
    "dry-base": (P, "see `--dry-multiplier`"), "dry-allowed-length": (P, "see `--dry-multiplier`"), "dry-penalty-last-n": (P, "see `--dry-multiplier`"),
    "dry-sequence-breaker": (P, "see `--dry-multiplier`"),
    "mirostat": (I, "v1 and v2"), "mirostat-lr": (I, ""), "mirostat-ent": (I, ""), "dynatemp-range": (I, ""), "dynatemp-exp": (I, ""),
    "seed": (I, "same mt19937 + uniform_real_distribution draw as llama_sampler_dist"),
    "samplers": (I, "`;`-separated; same sampler names as llama (`penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`)"),
    "ignore-eos": (I, ""), "logit-bias": (I, "`TOKEN_ID(+/-)BIAS`"), "min-keep": (I, ""),
    "grammar": (I, "src/grammar; host-logit sampling"), "grammar-file": (I, ""), "json-schema": (I, "converted to GBNF"), "json-schema-file": (I, ""),
    "timeout": (I, "socket read/write timeout"),
    "path": (I, "serves static files from the directory instead of the embedded UI (index.html default, no gzip, `..` rejected); `/sw.js` then answers the self-removing service worker like llama"),
    "ui": (I, "the llama.cpp web UI (commit 9d6f04f, gzip) is embedded; `--no-webui` makes `/` a 404 like llama, `/props.ui` follows"),
    "ui-config": (I, "JSON object served in `/props.ui_settings`; invalid JSON exits 1 like llama"), "ui-config-file": (I, "same, from a file"),
    "ui-mcp-proxy": (N, "accepted with a warning: `/cors-proxy` always answers 403 `feature_disabled` (no CORS proxy is implemented)"),
    "tools": (N, "accepted with a warning: `/tools` always answers 403 `feature_disabled` (built-in shell/file tools are deliberately not implemented); the UI copes"),
    "log-disable": (I, "also `--quiet`"), "verbose": (P, "raises the verbosity number; per-request log lines are printed at the default level"),
    "log-verbosity": (P, "per-request lines at >= 3"),
    "slot-save-path": (P, "enables `POST /slots/:id?action=save|restore|erase`. File format is Q-Volta's own (token list); `restore` re-evaluates the tokens, it does not load a KV snapshot"),
    "special": (P, "only affects output of control tokens"),
    "cors-origins": (I, ""), "cors-methods": (I, ""), "cors-headers": (I, ""), "cors-credentials": (I, ""), "api-prefix": (I, ""),
    "help": (I, ""), "version": (I, ""),
    "context-shift": (N, "recurrent GatedDeltaNet state cannot be shifted: generation stops at the context limit (`stop_type: limit`, `truncated: true`), like llama-server's default"),
    "lora": (U, "refuses to start: merge adapters at conversion time (`tools/convert.py --lora`)"),
    "lora-scaled": (U, "refuses to start: merge adapters at conversion time"),
    "ssl-key-file": (U, "no TLS support (llama-server built without OpenSSL behaves the same: error)"), "ssl-cert-file": (U, "no TLS support"),
    "hf-repo": (U, "no model download; pass a local Q-Volta model"), "model-url": (U, "no model download"), "docker-repo": (U, "no model download"),
    "hf-file": (N, "no model download"), "hf-token": (N, "no model download"), "offline": (N, ""),
    "cache-list": (N, "no model cache"),
    "sse-ping-interval": (I, "default 30 s; `:\\n\\n` comment after that much silence on a stream (also per request: `sse_ping_interval`)"),
    "sleep-idle-seconds": (I, "engine is destroyed after N idle seconds and rebuilt on the next generation; `/health`, `/props`, `/models`, `/metrics` do not wake it; `/props.is_sleeping`"),
}
RULES = [  # (regex on canonical name, status, note)
    (r"^(cpu-|prio|poll|cpu-mask|cpu-range|cpu-strict|numa)", N, "thread placement / priority is not configurable"),
    (r"^(rope|yarn)", N, "RoPE parameters are fixed by the converted model"),
    (r"^(cache-type|cache-type-k|cache-type-v)", N, "KV dtype is selected with `--kv fp32|fp16|int4`"),
    (r"^(spec|draft|lookup|ngram|model-draft|ctx-size-draft|threads-draft|device-draft|gpu-layers-draft|hf-repo-draft)", N, "draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`)"),
    (r"^(mmproj|image|video|mtmd|media|tts|vocoder|embd-gemma|vision|fim-|gpt-oss|spec-default|mcp|allow-unauth|agent|connect)", N, "multimodal / audio / agent features are not supported"),
    (r"^(override|cpu-moe|n-cpu-moe|split-mode|tensor-split|main-gpu|device|list-devices|rpc|fit|kv-offload|repack|no-host|op-offload)", N, "multi-device / tensor placement does not apply to Q-Volta engines"),
    (r"^(models|router)", N, "router mode is not supported"),
    (r"^(ctx-checkpoints|checkpoint|cache-ram|kv-unified|cache-idle|swa|kv-unified-per-slot|slot-prompt-similarity)", N, "llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache"),
    (r"^(cache-reuse)", N, "KV shifting does not apply to the recurrent state"),
    (r"^(flash-attn|mlock|mmap|no-mmap|direct-io|warmup|perf|escape|check-tensors|swa-full|no-perf|load-mode)", N, "no effect"),
    (r"^(log-|verbosity|log)", N, "logging format flags are accepted and ignored"),
    (r"^(pooling|embd-normalize|rerank|reranking)", N, "embeddings / reranking are not supported"),
    (r"^(adaptive|backend-sampling|sampler-seq|sleep-idle|lora-init|log-prompts|reuse-port|skip-chat|prefill-assistant|reasoning-preserve|keep|lookup)", N, "accepted, no effect"),
    
    (r"^(reverse-prompt|spm-infill|input-prefix)", N, "accepted, no effect"),
]
GRAMMAR = ("accepted; `grammar` / `json_schema` / `response_format` constraints are enforced only once src/grammar (branch p15/jinja) is integrated; until then requests carrying them are answered without the constraint -- see the status line at the top of this file")

def canon(f):
    for n in f["names"]:
        if n.startswith("--") and not n.startswith("--no-"): return n[2:]
    return f["names"][0].lstrip("-")

rows, counts, by_section = [], {I: 0, P: 0, N: 0, U: 0}, {}
for f in flags:
    c = canon(f)
    st, note = S.get(c, (None, None))
    if st is None:
        for rx, s2, n2 in RULES:
            if re.search(rx, c): st, note = s2, n2; break
    if st is None: st, note = N, "accepted for compatibility, no effect in qv_server"
    note = note.replace("STATUS_GRAMMAR", GRAMMAR)
    counts[st] += 1
    spell = ", ".join("`%s`" % n for n in f["names"]) + (" `%s`" % f["meta"] if f["meta"] else "")
    env = f["env"][0] if f["env"] else ""
    by_section.setdefault(f["section"], []).append("| %s | %s | %s | %s |" % (spell.replace("|", "\\|"), ("`%s`" % env) if env else "", st, note.replace("|", "\\|")))
    if st in (I, P) and ('"%s"' % c) not in src and c not in ("gpu-layers", "n-gpu-layers"):
        print("WARNING: %s marked %s but not in options.cpp" % (c, st), file=sys.stderr)

out = head.rstrip() + "\n\n## 3. Command line flags\n\n"
out += "Every flag of `llama-server --help` (llama.cpp 9d6f04f, %d entries, generated into `src/server/llama_flags.inc`) is parsed with its own argument arity, so " % len(flags)
out += "an unsupported flag never swallows or leaks the next argument. Environment variables (`LLAMA_ARG_*`, `LLAMA_API_KEY`) are read for every flag that has one; the command line wins. Boolean env values `1/on/true/enabled` set, `0/off/false/disabled` unset (the `--no-X` spelling).\n\n"
out += "Counts: **%d implemented, %d partial, %d accepted-and-ignored (n/a), %d refuse-to-start (unsupported)**.\n\n" % (counts[I], counts[P], counts[N], counts[U])
out += "Not in llama-server (Q-Volta flags): `--backend cuda|cpu|auto|mock`, `--kv fp32|fp16|int4`, `--kv4-kmode`, `--mtp`, `--mtp-depth N`, `--prefill-chunk N`, `--tokenizer PATH`, `--max-queue N`, " \
       "`--model-name` (= `-a`), `--max-ctx` (= `-c`), `--http-threads` (= `--threads-http`), `--quiet` (= `--log-disable`), `--default-max-tokens` (= `-n`), mock-engine flags `--mock-text/--mock-loop/--mock-delay-us`. Legacy llama spellings `--mlock/--no-mmap/--mmap` are accepted.\n\n"
for sec, ls in by_section.items():
    out += "### %s\n\n| flag | env | status | note |\n|---|---|---|---|\n%s\n\n" % (sec, "\n".join(ls))
open(sys.argv[3], "w").write(out)
print(counts)
