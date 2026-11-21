# qv_server vs llama-server: compatibility matrix

Reference: `llama-server` from llama.cpp commit `9d6f04f` (source in `/workspace/q-volta-models/scratch-cpu-baseline/llama.cpp`, built CPU-only
for differential testing). Goal: a client written for llama-server (curl, OpenAI SDKs, llama.cpp web clients, scripts using `LLAMA_ARG_*`)
works unchanged against `qv_server`. Status values: **implemented** (same behaviour, verified), **partial** (works, with the stated limit),
**n/a** (accepted/answered, no effect for Q-Volta, reason given), **unsupported** (refused explicitly).

Integration status of the two sibling libraries (branch `p15/jinja`): the Jinja interpreter (`src/jinja`) and the GBNF / JSON-schema
constrained sampler (`src/grammar`) had **not landed on `autoresearch/q-volta-gen1`** when this was written, so:
`tools` / `tool_choice`, custom chat templates, `grammar`, `json_schema` and `response_format: json_object|json_schema` are **not enforced yet**
(tools and custom templates are rejected with a 400/startup error; grammar-type constraints are parsed but ignored, which is stated here rather than hidden).
The interfaces are in place: `src/server/chat_render.h` (template front end) and `qv::LogitsConstraint` in `src/sampling/sampler.h` (grammar hook).

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
| `POST /chat/completions`, `/v1/chat/completions` | partial | multi-part text content, `max_tokens`/`max_completion_tokens`, `stream_options.include_usage`, `logprobs`/`top_logprobs`, `n` (sequential generations, not concurrent), `reasoning_format`, `chat_template_kwargs`, `reasoning_effort`, all llama sampling extras. **Not yet**: tools / tool_choice / parallel_tool_calls (need src/jinja), `response_format` json constraints (need src/grammar), image/audio parts (error 500 like llama without mmproj) |
| `POST /infill` | implemented | llama's FIM prompt layout (repo/file separators, 3:1 prefix:suffix budget); uses the Qwen `<|fim_*|>` tokens, 501 with llama's message if the vocabulary has none |
| `POST /tokenize` | implemented | `with_pieces`, `parse_special`; `add_special` accepted (Qwen tokenizers add no BOS/EOS so it has no effect) |
| `POST /detokenize` | implemented | |
| `POST /apply-template` | implemented | same renderer as chat completions |
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
`samplers`, `min_keep`, `grammar`, `json_schema` (parsed, see status above). Unknown fields are ignored (as llama).
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
| Tool calls, grammar, JSON mode | not enforced yet | waiting for src/jinja and src/grammar to land (see top) |
| Slot concurrency | one decoding sequence | engine design, see section 4 |
| `/slots` save/restore | token-list files, restore = re-prefill | no KV snapshot API in the engines |
| `meta.n_params`, `model_ftype` | 0 / `Q-Volta` | not recorded in the `.qv` header |
| Anthropic / Responses / audio / realtime / router / MCP APIs | 501 or 404 | out of scope |
| Embeddings, reranking, LoRA at runtime, vision | 501 / 400 | not supported by the engines (LoRA is merged at conversion time) |
| `n > 1` | run sequentially (re-prefills per choice, cost x n) | one sequence |
| JSON parse error text | `[json.exception.parse_error.101] parse error: <our parser message>` | different parser |
