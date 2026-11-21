# qv_server vs llama-server: compatibility matrix

Reference: `llama-server` from llama.cpp commit `9d6f04f` (source in `/workspace/q-volta-models/scratch-cpu-baseline/llama.cpp`, built CPU-only
for differential testing). Goal: a client written for llama-server (curl, OpenAI SDKs, llama.cpp web clients, scripts using `LLAMA_ARG_*`)
works unchanged against `qv_server`. Status values: **implemented** (same behaviour, verified), **partial** (works, with the stated limit),
**n/a** (accepted/answered, no effect for Q-Volta, reason given), **unsupported** (refused explicitly).

Jinja interpreter (`src/jinja`, docs/JINJA.md) and GBNF / JSON-schema grammar sampler (`src/grammar`, docs/GRAMMAR.md) are integrated: chat templates are evaluated by
`qv::chat::ChatTemplate` (the model's `chat_template.jinja`, `--chat-template`, `--chat-template-file`; the native Qwen3.8 renderer is only the fallback if the model's own template
fails to evaluate and no tools were requested), `tools` / `tool_choice` / `parallel_tool_calls` produce OpenAI `tool_calls` (also as streaming deltas), and `grammar`, `json_schema`,
`response_format`, forced `tool_choice` and `--grammar/--json-schema` constrain sampling on host logits (the engine-argmax fast path and MTP lookahead are skipped for constrained requests).

## 1. Endpoints

| endpoint | status | notes |
|---|---|---|
| `GET /health`, `GET /v1/health` | implemented | `{"status":"ok"}`; **503 `Loading model`** (`unavailable_error`) on every route while the model loads (the port is opened first); public even with an API key; `HEAD` works |
| `GET /props` | implemented | same keys as llama-server (`default_generation_settings`, `total_slots`, `model_alias`, `model_path`, `modalities`, `endpoint_*`, `chat_template`, `chat_template_caps`, `bos_token`, `eos_token`, `build_info`, `is_sleeping`, ...). `chat_template` is the model's `chat_template.jinja`, `build_info` is `qv-<git rev>` |
| `POST /props` | implemented | 501 `not_supported_error` unless `--props`, then `{"success":true}` |
| `GET /slots` | implemented | `--parallel N` logical slots, `is_processing`, `id_task`, `params`, `next_token`; `?fail_on_no_slot` -> 503 |
| `POST /slots/:id?action=save\|restore\|erase` | partial | needs `--slot-save-path` (501 otherwise, like llama). Save stores the token list; restore re-evaluates it (no KV/state snapshot in the file; files are not interchangeable with llama.cpp). Filename validation and error messages as llama |
| `GET /metrics` | implemented | needs `--metrics` (501 otherwise); same 15 metric names/HELP/TYPE (`llamacpp:*`), `Process-Start-Time-Unix` header, `text/plain; version=0.0.4`. Gauges are windowed between scrapes like llama |
| `GET /models`, `GET /v1/models` | implemented | llama's merged shape (`models[]` Ollama-style + `data[]` OpenAI-style). `meta.n_params` is 0 (not read from the container) |
| `POST /completion`, `/completions` | implemented | see section 2 |
| `POST /v1/completions` | implemented | llama's shape: `chatcmpl-` ids, `logprobs: {"content":[...]}` (not the legacy OpenAI arrays), `timings` |
| `POST /chat/completions`, `/v1/chat/completions` | implemented | multi-part text content, `max_tokens`/`max_completion_tokens`, `stream_options.include_usage`, `logprobs`/`top_logprobs`, `n` (sequential generations, not concurrent), `reasoning_format`, `chat_template_kwargs`, `reasoning_effort`, all llama sampling extras, **tools / `tool_choice` (auto = lazy grammar, `required`/function = grammar) / `parallel_tool_calls`**, tool-call history messages, `tool_calls` streaming deltas, `response_format` (`json_object`, `json_schema`), `grammar`, `json_schema`. Image/audio parts: error 500 like llama without mmproj |
| `POST /infill` | implemented | llama's FIM prompt layout (repo/file separators, 3:1 prefix:suffix budget); uses the Qwen `<|fim_*|>` tokens, 501 with llama's message if the vocabulary has none |
| `POST /tokenize` | implemented | `with_pieces`, `parse_special`; `add_special` accepted (Qwen tokenizers add no BOS/EOS so it has no effect) |
| `POST /detokenize` | implemented | |
| `POST /apply-template` | implemented | same renderer as chat completions, including `tools` and tool-call history |
| `POST /chat/completions/input_tokens`, `/v1/...` | implemented | `{"input_tokens":N,"object":"response.input_tokens"}` |
| `GET /lora-adapters`, `POST /lora-adapters` | partial | no runtime LoRA: GET returns `[]`, POST of `[]` returns `{"success":true}`, non-empty -> 400 (adapters are merged at conversion time) |
| `POST /embedding`, `/embeddings`, `/v1/embeddings` | n/a | 501 `not_supported_error` with llama's message (the engines do not expose a pooled hidden state) |
| `POST /rerank`, `/reranking`, `/v1/rerank(ing)` | n/a | 501 like a llama-server started without `--reranking` |
| `POST /v1/messages`, `/v1/messages/count_tokens` (Anthropic) | not implemented | 501 `not_supported_error` |
| `POST /v1/responses`, `/responses`, `/v1/responses/input_tokens` | not implemented | 501 |
| `/v1/audio/*`, `/tts`, `/v1/realtime`, `/v1/systemone`, `/v1/chat/completions/control` | n/a | 501 / no WebSocket |
| `/v1/stream`, `/v1/streams/lookup`, `/cors-proxy`, `/tools`, `/models/load` ... (router / MCP / agent features) | n/a | 404 |
| web UI (`/`) | n/a | none embedded; `--path DIR` serves static files |
| `OPTIONS *` | implemented | `200 text/html`, empty body, `Access-Control-Allow-Origin/Credentials/Methods/Headers` as llama |

Cross-cutting behaviour (all verified against the real llama-server in `tests/test_server_compat.py`):

* **Errors** `{"error":{"code":<http status>,"message":...,"type":...}}`; types `invalid_request_error` (400), `authentication_error` (401, `Invalid API Key`),
  `not_found_error` (404 `File Not Found`, key order as llama), `server_error` (500), `not_supported_error` (501), `unavailable_error` (503),
  `exceed_context_size_error` (400, with `n_prompt_tokens` and `n_ctx`). Error responses carry `Connection: close`. Type errors on request fields use
  llama's nlohmann wording (`Field 'temperature': [json.exception.type_error.302] type must be number, but is string`); JSON parse errors have the same
  code/type but our own parser's text after `parse error:`.
* **Headers**: `Server: qv_server` (llama sends `llama.cpp`: deliberately not impersonated), `Access-Control-Allow-Origin` (echoes `Origin` with `--cors-credentials`, as llama),
  `Content-Type: application/json; charset=utf-8`, `Keep-Alive: timeout=5, max=100`, `X-Accel-Buffering: no` + `Transfer-Encoding: chunked` for SSE. HEAD answers without body. Chunked request bodies are accepted.
* **SSE**: `data: <json>\n\n` framing, OpenAI streams end with `data: [DONE]`, native `/completion` streams end with the final result object (no `[DONE]`), as llama. Chat streams: role chunk with `content:null`, content / `reasoning_content` deltas, finish chunk with `delta:{}`, optional `usage` chunk with `choices:[]`, `timings` on the last chunk.
  Errors that occur after the headers were sent are emitted as a `data: {"error":...}` event. No SSE keep-alive pings.
* **Queueing / slots**: see section 4.

## 2. `/completion` and sampling parameters

Accepted request fields: `prompt` (string, token array, mixed strings+tokens, `{prompt_string}`, array of prompts -> array response, non-stream only), `n_predict`
(`max_tokens`), `stream`, `stop`, `cache_prompt`, `id_slot`, `return_tokens`, `timings_per_token`, `post_sampling_probs`, `n_probs`, `n_keep`, `response_fields`,
`temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `repeat_penalty` (`repetition_penalty`), `repeat_last_n`, `presence_penalty`,
`frequency_penalty`, `dry_*`, `dynatemp_range`, `dynatemp_exponent`, `mirostat` (1, 2), `mirostat_tau`, `mirostat_eta`, `seed`, `ignore_eos`, `logit_bias` (`[[id|"text", bias|false]]` or `{"id": bias}`),
`samplers`, `min_keep`, `grammar`, `json_schema` (enforced; `grammar` wins over `json_schema` on the native endpoints, the chat endpoint answers 500 `Cannot use both json_schema and grammar` like llama). Unknown fields are ignored (as llama).
Response fields: `index, content, tokens, id_slot, stop, model, tokens_predicted, tokens_evaluated, generation_settings, prompt, has_new_line, truncated, stop_type (eos|word|limit|none), stopping_word, tokens_cached, timings{cache_n, prompt_n, prompt_ms, ...}, completion_probabilities`.
Defaults are llama-server's (temperature 0.8, top_k 40, top_p 0.95, min_p 0.05, `n_predict` -1), **not** the Qwen generation_config values the old server used; set `--temp/--top-k/--top-p` (or `LLAMA_ARG_*`) for model-recommended sampling.
The sampler chain, sampler order and the final draw follow `common_sampler` / `llama_sampler_dist` (mt19937 + `uniform_real_distribution<double>`); identical probabilities + identical seed give identical draws, but the logits themselves come from a different model/runtime, so seeded texts are not comparable across servers.
Context end: like llama-server without `--context-shift`, generation stops with `stop_type: limit`, `truncated: true` when `n_tokens + 1 >= n_ctx`; a prompt with `n_tokens >= n_ctx` is a 400 `exceed_context_size_error`.
Control tokens (`<|...|>`) are not printed unless `--special`.

## 4. Slots, parallelism, prompt cache

`--parallel N` (default 4, llama's auto) creates N logical slots. The CPU and CUDA engines hold **one** sequence state, so generation is serialized:
the running request is the only slot with `is_processing: true`; other requests wait in a FIFO (`llamacpp:requests_deferred` counts them) and are served in arrival order;
nothing is rejected unless `--max-queue N` is given (then 503 `unavailable_error`, a Q-Volta addition; llama-server itself queues without bound). A request whose client disconnects while queued or running is dropped and
the engine keeps exactly the tokens it processed. `id_slot` is honoured for reporting only (`id_slot` in the response, `/slots` entry that records the task); all slots share one prefix cache.
Prefix reuse (`cache_prompt`, default on): the new prompt must start with the exact token sequence in the engine (recurrent state cannot be rewound), then only the suffix is evaluated; `timings.cache_n`,
`usage.prompt_tokens_details.cached_tokens` and `/metrics` `prompt_tokens_cached_total` report it. A repeated identical prompt re-evaluates the whole prompt, because the state has advanced past it by the generated tokens (llama.cpp rewinds its KV cache; a state snapshot after prefill would be the engine-level fix).
True concurrent multi-slot decoding (batched sequences) needs per-slot recurrent/KV state in the engines and is a later engine-level feature.

## 5. Known differences (intentional or unavoidable)

| area | difference | reason |
|---|---|---|
| `Server` header | `qv_server` instead of `llama.cpp` | not impersonating another product |
| Thinking | `reasoning_content` appears for the Qwen3.8 template (its generation prompt opens `<think>`); llama + Qwen2.5 template has no thinking | model/template, same `reasoning_format` semantics (`auto`=deepseek) |
| Tool-call wire format | the Qwen3.8 template makes the model emit `<tool_call><function=..><parameter=..>` (QwenXml); llama-server + Qwen2.5 uses Hermes JSON. The OpenAI `tool_calls` output has the same shape; the model-specific text format follows the loaded model's template | model/template |
| Grammar speed | allowed-token masks are cached per grammar state; first use of a new state costs one trie walk over the 248K-token vocabulary (docs/GRAMMAR.md) | different implementation |
| `--reasoning-budget N>0`, grammar for `<think>` text | not enforced | |
| Slot concurrency | one decoding sequence | engine design, see section 4 |
| `/slots` save/restore | token-list files, restore = re-prefill | no KV snapshot API in the engines |
| `meta.n_params`, `model_ftype` | 0 / `Q-Volta` | not recorded in the `.qv` header |
| Anthropic / Responses / audio / realtime / router / MCP APIs | 501 or 404 | out of scope |
| Embeddings, reranking, LoRA at runtime, vision | 501 / 400 | not supported by the engines (LoRA is merged at conversion time) |
| `n > 1` | run sequentially (re-prefills per choice, cost x n) | one sequence |
| JSON parse error text | `[json.exception.parse_error.101] parse error: <our parser message>` | different parser |

## 3. Command line flags

Every flag of `llama-server --help` (llama.cpp 9d6f04f, 266 entries, generated into `src/server/llama_flags.inc`) is parsed with its own argument arity, so an unsupported flag never swallows or leaks the next argument. Environment variables (`LLAMA_ARG_*`, `LLAMA_API_KEY`) are read for every flag that has one; the command line wins. Boolean env values `1/on/true/enabled` set, `0/off/false/disabled` unset (the `--no-X` spelling).

Counts: **51 implemented, 20 partial, 188 accepted-and-ignored (n/a), 7 refuse-to-start (unsupported)**.

Not in llama-server (Q-Volta flags): `--backend cuda|cpu|auto|mock`, `--kv fp32|fp16|int4`, `--kv4-kmode`, `--mtp`, `--mtp-depth N`, `--prefill-chunk N`, `--tokenizer PATH`, `--max-queue N`, `--model-name` (= `-a`), `--max-ctx` (= `-c`), `--http-threads` (= `--threads-http`), `--quiet` (= `--log-disable`), `--default-max-tokens` (= `-n`), mock-engine flags `--mock-text/--mock-loop/--mock-delay-us`. Legacy llama spellings `--mlock/--no-mmap/--mmap` are accepted.

### common params

| flag | env | status | note |
|---|---|---|---|
| `-h`, `--help`, `--usage` |  | implemented |  |
| `--version` |  | implemented |  |
| `-cl`, `--cache-list` |  | n/a | no model cache |
| `--completion-bash` |  | n/a | accepted for compatibility, no effect in qv_server |
| `-t`, `--threads` `N` | `LLAMA_ARG_THREADS` | implemented | CPU engine worker threads (default min(32, cores)) |
| `-tb`, `--threads-batch` `N` |  | n/a | accepted for compatibility, no effect in qv_server |
| `-C`, `--cpu-mask` `M` |  | n/a | thread placement / priority is not configurable |
| `-Cr`, `--cpu-range` `lo-hi` |  | n/a | thread placement / priority is not configurable |
| `--cpu-strict` `<0\|1>` |  | n/a | thread placement / priority is not configurable |
| `--prio` `N` |  | n/a | thread placement / priority is not configurable |
| `--poll` `<0...100>` |  | n/a | thread placement / priority is not configurable |
| `-Cb`, `--cpu-mask-batch` `M` |  | n/a | thread placement / priority is not configurable |
| `-Crb`, `--cpu-range-batch` `lo-hi` |  | n/a | thread placement / priority is not configurable |
| `--cpu-strict-batch` `<0\|1>` |  | n/a | thread placement / priority is not configurable |
| `--prio-batch` `N` |  | n/a | thread placement / priority is not configurable |
| `--poll-batch` `<0\|1>` |  | n/a | thread placement / priority is not configurable |
| `-c`, `--ctx-size` `N` | `LLAMA_ARG_CTX_SIZE` | implemented | 0 = 4096 (default). All slots share one engine context: `n_ctx` per slot = the full context (llama-server's unified-KV layout) |
| `-n`, `--predict`, `--n-predict` `N` | `LLAMA_ARG_N_PREDICT` | implemented | default `n_predict` (-1 = until EOS / context end) |
| `-b`, `--batch-size` `N` | `LLAMA_ARG_BATCH` | partial | only used as the infill batch size; prefill chunk comes from `-ub` / `--prefill-chunk` |
| `-ub`, `--ubatch-size` `N` | `LLAMA_ARG_UBATCH` | partial | prefill chunk size (also the granularity at which a client disconnect is noticed during prefill) |
| `--keep` `N` |  | n/a | accepted, no effect |
| `--swa-full` | `LLAMA_ARG_SWA_FULL` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `-fa`, `--flash-attn` `[on\|off\|auto]` | `LLAMA_ARG_FLASH_ATTN` | n/a | no effect |
| `--perf`, `--no-perf` | `LLAMA_ARG_PERF` | n/a | no effect |
| `-e`, `--escape`, `--no-escape` |  | n/a | no effect |
| `--rope-scaling` `{none,linear,yarn}` | `LLAMA_ARG_ROPE_SCALING_TYPE` | n/a | RoPE parameters are fixed by the converted model |
| `--rope-scale` `N` | `LLAMA_ARG_ROPE_SCALE` | n/a | RoPE parameters are fixed by the converted model |
| `--rope-freq-base` `N` | `LLAMA_ARG_ROPE_FREQ_BASE` | n/a | RoPE parameters are fixed by the converted model |
| `--rope-freq-scale` `N` | `LLAMA_ARG_ROPE_FREQ_SCALE` | n/a | RoPE parameters are fixed by the converted model |
| `--yarn-orig-ctx` `N` | `LLAMA_ARG_YARN_ORIG_CTX` | n/a | RoPE parameters are fixed by the converted model |
| `--yarn-ext-factor` `N` | `LLAMA_ARG_YARN_EXT_FACTOR` | n/a | RoPE parameters are fixed by the converted model |
| `--yarn-attn-factor` `N` | `LLAMA_ARG_YARN_ATTN_FACTOR` | n/a | RoPE parameters are fixed by the converted model |
| `--yarn-beta-slow` `N` | `LLAMA_ARG_YARN_BETA_SLOW` | n/a | RoPE parameters are fixed by the converted model |
| `--yarn-beta-fast` `N` | `LLAMA_ARG_YARN_BETA_FAST` | n/a | RoPE parameters are fixed by the converted model |
| `-kvo`, `--kv-offload`, `-nkvo`, `--no-kv-offload` | `LLAMA_ARG_KV_OFFLOAD` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--repack`, `-nr`, `--no-repack` | `LLAMA_ARG_REPACK` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--no-host` | `LLAMA_ARG_NO_HOST` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-ctk`, `--cache-type-k` `TYPE` | `LLAMA_ARG_CACHE_TYPE_K` | n/a | KV dtype is selected with `--kv fp32\|fp16\|int4` |
| `-ctv`, `--cache-type-v` `TYPE` | `LLAMA_ARG_CACHE_TYPE_V` | n/a | KV dtype is selected with `--kv fp32\|fp16\|int4` |
| `--rpc` `SERVERS` | `LLAMA_ARG_RPC` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-lm`, `--load-mode` `MODE` | `LLAMA_ARG_LOAD_MODE` | n/a | no effect |
| `-lzm`, `--lazy-mode` `MODE` | `LLAMA_ARG_LAZY_MODE` | n/a | accepted for compatibility, no effect in qv_server |
| `--numa` `TYPE` | `LLAMA_ARG_NUMA` | n/a | thread placement / priority is not configurable |
| `-dev`, `--device` `<dev1,dev2,..>` | `LLAMA_ARG_DEVICE` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--list-devices` |  | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-ot`, `--override-tensor` `<tensor name pattern>=<buffer type>,...` | `LLAMA_ARG_OVERRIDE_TENSOR` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-cmoe`, `--cpu-moe` | `LLAMA_ARG_CPU_MOE` | n/a | thread placement / priority is not configurable |
| `-ncmoe`, `--n-cpu-moe` `N` | `LLAMA_ARG_N_CPU_MOE` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--moe-cache-mib` `N` | `LLAMA_ARG_MOE_CACHE_MIB` | n/a | accepted for compatibility, no effect in qv_server |
| `--moe-cache-cpu-share` `F` | `LLAMA_ARG_MOE_CACHE_CPU_SHARE` | n/a | accepted for compatibility, no effect in qv_server |
| `-ncffn`, `--n-cpu-ffn` `N` | `LLAMA_ARG_N_CPU_FFN` | n/a | accepted for compatibility, no effect in qv_server |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` `N` | `LLAMA_ARG_N_GPU_LAYERS` | partial | all-or-nothing: `0` selects the CPU engine, any other value (incl. `auto`/`all`) the CUDA engine if a device exists; `--backend` overrides |
| `-sm`, `--split-mode` `{none,layer,row,tensor}` | `LLAMA_ARG_SPLIT_MODE` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-ts`, `--tensor-split` `N0,N1,N2,...` | `LLAMA_ARG_TENSOR_SPLIT` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-mg`, `--main-gpu` `INDEX` | `LLAMA_ARG_MAIN_GPU` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-fit`, `--fit` `[on\|off]` | `LLAMA_ARG_FIT` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-fitt`, `--fit-target` `MiB0,MiB1,MiB2,...` | `LLAMA_ARG_FIT_TARGET` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `-fitc`, `--fit-ctx` `N` | `LLAMA_ARG_FIT_CTX` | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--check-tensors` |  | n/a | no effect |
| `--override-kv` `KEY=TYPE:VALUE,...` |  | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--op-offload`, `--no-op-offload` |  | n/a | multi-device / tensor placement does not apply to Q-Volta engines |
| `--lora` `FNAME` |  | unsupported | refuses to start: merge adapters at conversion time (`tools/convert.py --lora`) |
| `--lora-scaled` `FNAME:SCALE,...` |  | unsupported | refuses to start: merge adapters at conversion time |
| `--control-vector` `FNAME` |  | n/a | accepted for compatibility, no effect in qv_server |
| `--control-vector-scaled` `FNAME:SCALE,...` |  | n/a | accepted for compatibility, no effect in qv_server |
| `--control-vector-layer-range` `START END` |  | n/a | accepted for compatibility, no effect in qv_server |
| `-m`, `--model` `FNAME` | `LLAMA_ARG_MODEL` | implemented | Q-Volta model prefix or `.qv` file (not GGUF) |
| `--ple`, `--shard` `FNAME,...` | `LLAMA_ARG_SHARD` | n/a | accepted for compatibility, no effect in qv_server |
| `-mu`, `--model-url` `MODEL_URL` | `LLAMA_ARG_MODEL_URL` | unsupported | no model download |
| `-dr`, `--docker-repo` `[<repo>/]<model>[:quant]` | `LLAMA_ARG_DOCKER_REPO` | unsupported | no model download |
| `-hf`, `-hfr`, `--hf-repo` `<user>/<model>[:quant]` | `LLAMA_ARG_HF_REPO` | unsupported | no model download; pass a local Q-Volta model |
| `-hff`, `--hf-file` `FILE` | `LLAMA_ARG_HF_FILE` | n/a | no model download |
| `-hft`, `--hf-token` `TOKEN` |  | n/a | no model download |
| `--log-disable` |  | implemented | also `--quiet` |
| `--log-file` `FNAME` | `LLAMA_ARG_LOG_FILE` | n/a | logging format flags are accepted and ignored |
| `--log-jsonl`, `--no-log-jsonl` | `LLAMA_ARG_LOG_JSONL` | n/a | logging format flags are accepted and ignored |
| `--log-colors` `[on\|off\|auto]` | `LLAMA_ARG_LOG_COLORS` | n/a | logging format flags are accepted and ignored |
| `-v`, `--verbose`, `--log-verbose` |  | partial | raises the verbosity number; per-request log lines are printed at the default level |
| `--offline` | `LLAMA_ARG_OFFLINE` | n/a |  |
| `-lv`, `--verbosity`, `--log-verbosity` `N` | `LLAMA_ARG_LOG_VERBOSITY` | n/a | logging format flags are accepted and ignored |
| `--log-prefix`, `--no-log-prefix` | `LLAMA_ARG_LOG_PREFIX` | n/a | logging format flags are accepted and ignored |
| `--log-timestamps`, `--no-log-timestamps` | `LLAMA_ARG_LOG_TIMESTAMPS` | n/a | logging format flags are accepted and ignored |
| `--spec-draft-type-k`, `-ctkd`, `--cache-type-k-draft` `TYPE` | `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-type-v`, `-ctvd`, `--cache-type-v-draft` `TYPE` | `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |

### sampling params

| flag | env | status | note |
|---|---|---|---|
| `--samplers` `SAMPLERS` |  | implemented | `;`-separated; same sampler names as llama (`penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`) |
| `-s`, `--seed` `SEED` |  | implemented | same mt19937 + uniform_real_distribution draw as llama_sampler_dist |
| `--sampler-seq`, `--sampling-seq` `SEQUENCE` |  | n/a | accepted, no effect |
| `--ignore-eos` |  | implemented |  |
| `--temp`, `--temperature` `N` | `LLAMA_ARG_TEMPERATURE` | implemented |  |
| `--top-k` `N` | `LLAMA_ARG_TOP_K` | implemented |  |
| `--top-p` `N` | `LLAMA_ARG_TOP_P` | implemented |  |
| `--min-p` `N` | `LLAMA_ARG_MIN_P` | implemented |  |
| `--top-nsigma`, `--top-n-sigma` `N` |  | implemented |  |
| `--xtc-probability` `N` |  | implemented |  |
| `--xtc-threshold` `N` |  | implemented |  |
| `--typical`, `--typical-p` `N` |  | implemented |  |
| `--repeat-last-n` `N` |  | implemented |  |
| `--repeat-penalty` `N` | `LLAMA_ARG_REPEAT_PENALTY` | implemented |  |
| `--presence-penalty` `N` | `LLAMA_ARG_PRESENCE_PENALTY` | implemented |  |
| `--frequency-penalty` `N` | `LLAMA_ARG_FREQUENCY_PENALTY` | implemented |  |
| `--dry-multiplier` `N` |  | partial | DRY uses a simplified suffix matcher (single-token sequence breakers, match length capped at 64) |
| `--dry-base` `N` |  | partial | see `--dry-multiplier` |
| `--dry-allowed-length` `N` |  | partial | see `--dry-multiplier` |
| `--dry-penalty-last-n` `N` |  | partial | see `--dry-multiplier` |
| `--dry-sequence-breaker` `STRING` |  | partial | see `--dry-multiplier` |
| `--adaptive-target` `N` |  | n/a | accepted, no effect |
| `--adaptive-decay` `N` |  | n/a | accepted, no effect |
| `--dynatemp-range` `N` |  | implemented |  |
| `--dynatemp-exp` `N` |  | implemented |  |
| `--mirostat` `N` |  | implemented | v1 and v2 |
| `--mirostat-lr` `N` |  | implemented |  |
| `--mirostat-ent` `N` |  | implemented |  |
| `-l`, `--logit-bias` `TOKEN_ID(+/-)BIAS` |  | implemented | `TOKEN_ID(+/-)BIAS` |
| `--grammar` `GRAMMAR` |  | implemented | src/grammar; host-logit sampling |
| `--grammar-file` `FNAME` |  | implemented |  |
| `-j`, `--json-schema` `SCHEMA` |  | implemented | converted to GBNF |
| `-jf`, `--json-schema-file` `FILE` |  | implemented |  |
| `-bs`, `--backend-sampling` | `LLAMA_ARG_BACKEND_SAMPLING` | n/a | accepted, no effect |

### speculative params

| flag | env | status | note |
|---|---|---|---|
| `--spec-draft-hf`, `-hfd`, `-hfrd`, `--hf-repo-draft` `<user>/<model>[:quant]` | `LLAMA_ARG_SPEC_DRAFT_HF_REPO` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-threads`, `-td`, `--threads-draft` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-threads-batch`, `-tbd`, `--threads-batch-draft` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-cpu-mask`, `-Cd`, `--cpu-mask-draft` `M` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-cpu-range`, `-Crd`, `--cpu-range-draft` `lo-hi` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-cpu-strict`, `--cpu-strict-draft` `<0\|1>` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-prio`, `--prio-draft` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-poll`, `--poll-draft` `<0\|1>` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-cpu-mask-batch`, `-Cbd`, `--cpu-mask-batch-draft` `M` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-cpu-strict-batch`, `--cpu-strict-batch-draft` `<0\|1>` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-prio-batch`, `--prio-batch-draft` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-poll-batch`, `--poll-batch-draft` `<0\|1>` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-override-tensor`, `-otd`, `--override-tensor-draft` `<tensor name pattern>=<buffer type>,...` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-cpu-moe`, `-cmoed`, `--cpu-moe-draft` | `LLAMA_ARG_SPEC_DRAFT_CPU_MOE` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-n-cpu-moe`, `--spec-draft-ncmoe`, `-ncmoed`, `--n-cpu-moe-draft` `N` | `LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-n-max` `N` | `LLAMA_ARG_SPEC_DRAFT_N_MAX` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-n-min` `N` | `LLAMA_ARG_SPEC_DRAFT_N_MIN` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-n-ubatch` `N` | `LLAMA_ARG_SPEC_DRAFT_N_UBATCH` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-synth-len` `L` | `LLAMA_ARG_SPEC_SYNTH_LEN` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-synth-rates` `P0,P1,...` | `LLAMA_ARG_SPEC_SYNTH_RATES` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-p-split`, `--draft-p-split` `P` | `LLAMA_ARG_SPEC_DRAFT_P_SPLIT` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-p-min`, `--draft-p-min` `P` | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-backend-sampling`, `--no-spec-draft-backend-sampling` | `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-sampling` `{greedy,probabilistic}` | `LLAMA_ARG_SPEC_DRAFT_SAMPLING` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-device`, `-devd`, `--device-draft` `<dev1,dev2,..>` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-ngl`, `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft` `N` | `LLAMA_ARG_N_GPU_LAYERS_DRAFT` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-draft-model`, `-md`, `--model-draft` `FNAME` | `LLAMA_ARG_SPEC_DRAFT_MODEL` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-type` `none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,draft-dflash2,draft-lilicorr,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache` | `LLAMA_ARG_SPEC_TYPE` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-mod-n-min` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-mod-n-max` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-mod-n-match` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-simple-size-n` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-simple-size-m` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-simple-min-hits` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-map-k-size-n` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-map-k-size-m` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-map-k-min-hits` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-map-k4v-size-n` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-map-k4v-size-m` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-map-k4v-min-hits` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--draft`, `--draft-n`, `--draft-max` `N` | `LLAMA_ARG_DRAFT_MAX` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--draft-min`, `--draft-n-min` `N` | `LLAMA_ARG_DRAFT_MIN` | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-size-n` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-size-m` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--spec-ngram-min-hits` `N` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |

### example-specific params

| flag | env | status | note |
|---|---|---|---|
| `-lcs`, `--lookup-cache-static` `FNAME` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `-lcd`, `--lookup-cache-dynamic` `FNAME` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |
| `--kv-unified-per-slot` `N` | `LLAMA_ARG_KV_UNIFIED_PER_SLOT` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `-ctxcp`, `--ctx-checkpoints`, `--swa-checkpoints` `N` | `LLAMA_ARG_CTX_CHECKPOINTS` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `-cms`, `--checkpoint-min-step` `N` | `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `-cram`, `--cache-ram` `N` | `LLAMA_ARG_CACHE_RAM` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `-kvu`, `--kv-unified`, `-no-kvu`, `--no-kv-unified` | `LLAMA_ARG_KV_UNIFIED` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `--cache-idle-slots`, `--no-cache-idle-slots` | `LLAMA_ARG_CACHE_IDLE_SLOTS` | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `--context-shift`, `--no-context-shift` | `LLAMA_ARG_CONTEXT_SHIFT` | n/a | recurrent GatedDeltaNet state cannot be shifted: generation stops at the context limit (`stop_type: limit`, `truncated: true`), like llama-server's default |
| `-r`, `--reverse-prompt` `PROMPT` |  | n/a | accepted, no effect |
| `-sp`, `--special` |  | partial | only affects output of control tokens |
| `--warmup`, `--no-warmup` |  | n/a | no effect |
| `--spm-infill` |  | n/a | accepted, no effect |
| `--pooling` `{none,mean,cls,last,rank}` | `LLAMA_ARG_POOLING` | n/a | embeddings / reranking are not supported |
| `-np`, `--parallel` `N` | `LLAMA_ARG_N_PARALLEL` | partial | N logical slots reported by `/slots`/`/props`; the engine holds ONE sequence, requests are served FIFO (queued, not rejected). True concurrent decoding is a future engine feature |
| `-cb`, `--cont-batching`, `-nocb`, `--no-cont-batching` | `LLAMA_ARG_CONT_BATCHING` | n/a | accepted; single-sequence engine |
| `-mm`, `--mmproj` `FILE` | `LLAMA_ARG_MMPROJ` | n/a | multimodal / audio / agent features are not supported |
| `-mmu`, `--mmproj-url` `URL` | `LLAMA_ARG_MMPROJ_URL` | n/a | multimodal / audio / agent features are not supported |
| `--mmproj-auto`, `--no-mmproj`, `--no-mmproj-auto` | `LLAMA_ARG_MMPROJ_AUTO` | n/a | multimodal / audio / agent features are not supported |
| `--mmproj-offload`, `--no-mmproj-offload` | `LLAMA_ARG_MMPROJ_OFFLOAD` | n/a | multimodal / audio / agent features are not supported |
| `-mmdev`, `--mmproj-device` `DEVICE` |  | n/a | multimodal / audio / agent features are not supported |
| `--image-min-tokens` `N` | `LLAMA_ARG_IMAGE_MIN_TOKENS` | n/a | multimodal / audio / agent features are not supported |
| `--image-max-tokens` `N` | `LLAMA_ARG_IMAGE_MAX_TOKENS` | n/a | multimodal / audio / agent features are not supported |
| `--mtmd-batch-max-tokens` `N` | `LLAMA_ARG_MTMD_BATCH_MAX_TOKENS` | n/a | multimodal / audio / agent features are not supported |
| `--video-fps` `N` | `LLAMA_ARG_VIDEO_FPS` | n/a | multimodal / audio / agent features are not supported |
| `--video-timestamp-interval` `N` | `LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL` | n/a | multimodal / audio / agent features are not supported |
| `--video-ffmpeg-dir` `DIR` | `LLAMA_ARG_VIDEO_FFMPEG_DIR` | n/a | multimodal / audio / agent features are not supported |
| `-a`, `--alias` `STRING` | `LLAMA_ARG_ALIAS` | implemented | first alias is the model id in `/v1/models`, `/props`, responses |
| `--tags` `STRING` | `LLAMA_ARG_TAGS` | n/a | accepted for compatibility, no effect in qv_server |
| `--embd-normalize` `N` |  | n/a | embeddings / reranking are not supported |
| `--host` `HOST` | `LLAMA_ARG_HOST` | partial | IPv4 address or `localhost`; first entry of a comma list; no IPv6 / UNIX sockets |
| `--port` `PORT` | `LLAMA_ARG_PORT` | implemented | default 9931 like the reference commit |
| `--reuse-port` | `LLAMA_ARG_REUSE_PORT` | n/a | accepted, no effect |
| `--path` `PATH` | `LLAMA_ARG_STATIC_PATH` | partial | serves static files from the directory; there is no embedded web UI |
| `--cors-origins` `ORIGINS` | `LLAMA_ARG_CORS_ORIGINS` | implemented |  |
| `--cors-methods` `METHODS` | `LLAMA_ARG_CORS_METHODS` | implemented |  |
| `--cors-headers` `HEADERS` | `LLAMA_ARG_CORS_HEADERS` | implemented |  |
| `--cors-credentials`, `--no-cors-credentials` | `LLAMA_ARG_CORS_CREDENTIALS` | implemented |  |
| `--api-prefix` `PREFIX` | `LLAMA_ARG_API_PREFIX` | implemented |  |
| `--ui-config`, `--webui-config` `JSON` | `LLAMA_ARG_UI_CONFIG` | n/a | accepted for compatibility, no effect in qv_server |
| `--ui-config-file`, `--webui-config-file` `PATH` | `LLAMA_ARG_UI_CONFIG_FILE` | n/a | accepted for compatibility, no effect in qv_server |
| `--ui-mcp-proxy`, `--webui-mcp-proxy`, `--no-ui-mcp-proxy`, `--no-webui-mcp-proxy` | `LLAMA_ARG_UI_MCP_PROXY` | n/a | multimodal / audio / agent features are not supported |
| `--tools` `TOOL1,TOOL2,...` | `LLAMA_ARG_TOOLS` | n/a | multimodal / audio / agent features are not supported |
| `--tools-runtime` `OPTION` | `LLAMA_ARG_TOOLS_RUNTIME` | n/a | multimodal / audio / agent features are not supported |
| `--allow-unauthenticated-tools` | `LLAMA_ARG_ALLOW_UNAUTHENTICATED_TOOLS` | n/a | multimodal / audio / agent features are not supported |
| `--mcp-servers-config` `PATH` | `LLAMA_ARG_MCP_SERVERS_CONFIG` | n/a | multimodal / audio / agent features are not supported |
| `--mcp-servers-json` `JSON` | `LLAMA_ARG_MCP_SERVERS_JSON` | n/a | multimodal / audio / agent features are not supported |
| `-ag`, `--agent`, `-no-ag`, `--no-agent` | `LLAMA_ARG_AGENT` | n/a | multimodal / audio / agent features are not supported |
| `--ui`, `--webui`, `--no-ui`, `--no-webui` | `LLAMA_ARG_UI` | partial | no embedded web UI exists; `--no-webui` is a no-op |
| `--connect` | `LLAMA_ARG_CONNECT` | n/a | multimodal / audio / agent features are not supported |
| `--connect-code` `CODE` | `LLAMA_ARG_CONNECT_CODE` | n/a | multimodal / audio / agent features are not supported |
| `--embedding`, `--embeddings` | `LLAMA_ARG_EMBEDDINGS` | partial | flag accepted; `/embedding(s)` answer 501 (the engines do not expose pooled hidden states) |
| `--rerank`, `--reranking` | `LLAMA_ARG_RERANKING` | n/a | embeddings / reranking are not supported |
| `--api-key` `KEY` | `LLAMA_API_KEY` | implemented | comma list; `Authorization: Bearer` or `X-Api-Key`; `/health` and `/v1/health` stay public |
| `--api-key-file` `FNAME` | `LLAMA_ARG_API_KEY_FILE` | implemented | one key per line, `#` comments |
| `--ssl-key-file` `FNAME` | `LLAMA_ARG_SSL_KEY_FILE` | unsupported | no TLS support (llama-server built without OpenSSL behaves the same: error) |
| `--ssl-cert-file` `FNAME` | `LLAMA_ARG_SSL_CERT_FILE` | unsupported | no TLS support |
| `--chat-template-kwargs` `STRING` | `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` | implemented | merged under per-request `chat_template_kwargs`; `enable_thinking`, `reasoning_effort`, `preserve_thinking` are honoured by the Qwen3.8 template |
| `-to`, `--timeout` `N` | `LLAMA_ARG_TIMEOUT` | implemented | socket read/write timeout |
| `--sse-ping-interval` `N` | `LLAMA_ARG_SSE_PING_INTERVAL` | n/a | SSE comment pings are not sent |
| `--threads-http` `N` | `LLAMA_ARG_THREADS_HTTP` | partial | caps concurrent connections (one thread per connection), default 256 |
| `--cache-prompt`, `--no-cache-prompt` | `LLAMA_ARG_CACHE_PROMPT` | implemented | prefix reuse of the engine state (see docs/SERVER.md for the recurrent-state limits) |
| `--cache-reuse` `N` | `LLAMA_ARG_CACHE_REUSE` | n/a | KV shifting does not apply to the recurrent state |
| `--metrics` | `LLAMA_ARG_ENDPOINT_METRICS` | implemented | Prometheus endpoint with llama.cpp's metric names |
| `--props` | `LLAMA_ARG_ENDPOINT_PROPS` | implemented | POST /props accepted (no mutable property exists, same as llama-server) |
| `--slots`, `--no-slots` | `LLAMA_ARG_ENDPOINT_SLOTS` | implemented |  |
| `--slot-save-path` `PATH` | `LLAMA_ARG_SLOT_SAVE_PATH` | partial | enables `POST /slots/:id?action=save\|restore\|erase`. File format is Q-Volta's own (token list); `restore` re-evaluates the tokens, it does not load a KV snapshot |
| `--media-path` `PATH` |  | n/a | multimodal / audio / agent features are not supported |
| `--media-allow-private-urls` | `LLAMA_ARG_MEDIA_ALLOW_PRIVATE_URLS` | n/a | multimodal / audio / agent features are not supported |
| `--models-dir` `PATH` | `LLAMA_ARG_MODELS_DIR` | n/a | router mode is not supported |
| `--models-preset` `PATH` | `LLAMA_ARG_MODELS_PRESET` | n/a | router mode is not supported |
| `--models-max` `N` | `LLAMA_ARG_MODELS_MAX` | n/a | router mode is not supported |
| `--models-memory-margin` `N` | `LLAMA_ARG_MODELS_MEMORY_MARGIN` | n/a | router mode is not supported |
| `--models-autoload`, `--no-models-autoload` | `LLAMA_ARG_MODELS_AUTOLOAD` | n/a | router mode is not supported |
| `--jinja`, `--no-jinja` | `LLAMA_ARG_JINJA` | implemented | Jinja interpreter (src/jinja) evaluates the model's chat template; `--no-jinja` selects the native Qwen3.8 renderer (no tools) |
| `--reasoning-format` `FORMAT` | `LLAMA_ARG_THINK` | implemented | `auto` (=deepseek), `none`, `deepseek`, `deepseek-legacy`; legacy spelling `separate` accepted |
| `-rea`, `--reasoning` `[on\|off\|auto]` | `LLAMA_ARG_REASONING` | partial | `off` -> `enable_thinking=false`; `on`/`auto` -> template default |
| `--reasoning-effort` `LEVEL` | `LLAMA_ARG_REASONING_EFFORT` | implemented | passed to the Qwen3.8 template (`low\|medium\|xhigh`; OpenAI `high` -> `xhigh`, `minimal` -> `low`) |
| `--reasoning-budget` `N` | `LLAMA_ARG_THINK_BUDGET` | partial | only `0` (thinking off) is honoured; positive budgets are not enforced |
| `--reasoning-budget-message` `MESSAGE` | `LLAMA_ARG_THINK_BUDGET_MESSAGE` | n/a | accepted for compatibility, no effect in qv_server |
| `--reasoning-preserve`, `--no-reasoning-preserve` | `LLAMA_ARG_REASONING_PRESERVE` | n/a | accepted, no effect |
| `--chat-template` `JINJA_TEMPLATE` | `LLAMA_ARG_CHAT_TEMPLATE` | partial | `chatml` or a template text (Jinja source); other llama built-in names (llama3, mistral-v7, ...) are refused at startup |
| `--chat-template-file` `JINJA_TEMPLATE_FILE` | `LLAMA_ARG_CHAT_TEMPLATE_FILE` | implemented | Jinja template file |
| `--skip-chat-parsing`, `--no-skip-chat-parsing` | `LLAMA_ARG_SKIP_CHAT_PARSING` | n/a | accepted, no effect |
| `--prefill-assistant`, `--no-prefill-assistant` | `LLAMA_ARG_PREFILL_ASSISTANT` | n/a | accepted, no effect |
| `-sps`, `--slot-prompt-similarity` `SIMILARITY` |  | n/a | llama.cpp KV/checkpoint management; Q-Volta keeps one prompt-prefix cache |
| `--lora-init-without-apply` |  | n/a | accepted, no effect |
| `--sleep-idle-seconds` `SECONDS` |  | n/a | accepted, no effect |
| `--log-prompts-dir` `PATH` |  | n/a | logging format flags are accepted and ignored |
| `--tts-lang` `FNAME` |  | n/a | multimodal / audio / agent features are not supported |
| `--tts-voice` `VOICE` |  | n/a | multimodal / audio / agent features are not supported |
| `--embd-gemma-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--fim-qwen-1.5b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--fim-qwen-3b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--fim-qwen-7b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--fim-qwen-7b-spec` |  | n/a | multimodal / audio / agent features are not supported |
| `--fim-qwen-14b-spec` |  | n/a | multimodal / audio / agent features are not supported |
| `--fim-qwen-30b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--gpt-oss-20b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--gpt-oss-120b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--vision-gemma-4b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--vision-gemma-12b-default` |  | n/a | multimodal / audio / agent features are not supported |
| `--spec-default` |  | n/a | draft-model speculative decoding is not supported (Q-Volta uses MTP: `--mtp`) |

