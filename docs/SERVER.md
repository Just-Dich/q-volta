# qv_server: llama-server compatible HTTP server

`qv_server` serves a Q-Volta model over HTTP/1.1 with the **same API, flags and environment variables as llama.cpp's `llama-server`**
(reference commit 9d6f04f): native `/completion`, `/infill`, OpenAI `/v1/chat/completions` and `/v1/completions`, `/tokenize`, `/detokenize`,
`/apply-template`, `/props`, `/slots`, `/metrics`, `/health` (503 while loading), `/v1/models`, SSE streaming. Every endpoint and flag with its
status (implemented / partial / n/a / unsupported) is in [SERVER_COMPAT.md](SERVER_COMPAT.md). Plain C++20 on POSIX sockets, no third-party dependencies.

Chat templates run through the Jinja interpreter (`src/jinja`): the model's own `chat_template.jinja`, `--chat-template chatml|TEXT`, `--chat-template-file`. Tools (`tools`, `tool_choice`,
`parallel_tool_calls`, tool-call history) produce OpenAI `tool_calls`, also as streaming deltas; `grammar`, `json_schema`, `response_format` and forced tool choices are enforced by the GBNF sampler (`src/grammar`)
on host logits. Anthropic `/v1/messages` and OpenAI `/v1/responses` are translated onto the chat path. Still unavailable: embeddings, rerank, audio, runtime LoRA, router mode, `/tools`, `/cors-proxy`. Generation is serialized: `--parallel N` reports N logical slots, requests queue FIFO.

## Web UI

`GET /` serves llama.cpp's own web UI (SvelteKit, commit 9d6f04f), embedded in the binary: no runtime file dependency. The built assets live gzip-compressed in `src/server/webui/gzip/` (provenance, licence and rebuild script: `src/server/webui/README.md`, `scripts/vendor_webui.sh`, THIRD_PARTY.md) and `cmake/qv_embed_webui.cmake` turns them into a generated C++ source at build time.
Behaviour matches llama-server: gzip only (415 without `Accept-Encoding: gzip`), `ETag`/304, `Cache-Control`, COEP/COOP on the shell, UI routes public with `--api-key` and available while the model loads. `--no-webui` -> 404 (and a self-removing service worker for old browsers), `--path DIR` replaces the UI by a directory, `--ui-config JSON` / `--ui-config-file` feed `/props.ui_settings`, `--api-prefix` is honoured.
Endpoints the UI needs and that were added for it: `/v1/streams/lookup`, `/v1/stream` (resumable generation via `X-Conversation-Id`), `/v1/chat/completions/control`, `return_progress` / `timings_per_token` / `sse_ping_interval` request fields, `/tools` and `/cors-proxy` (403 `feature_disabled`, as llama-server by default).
Screenshots from headless Chromium: `docs/img/webui_*.png`; the browser check is `tests/webui_browser_check.py` (run by `test_webui_in_browser` when Playwright is installed).

## Sleep, streams, queueing

* `--sleep-idle-seconds N`: after N idle seconds the engine is destroyed (weights and KV freed) and rebuilt on the next generation (`is_sleeping` in `/props`; `/health`, `/props`, `/models`, `/metrics` do not wake it).
* Streams with `X-Conversation-Id` survive client disconnects; see SERVER_COMPAT.md.
* `--parallel N` is honest queueing: one engine sequence, FIFO, no concurrent decoding (SERVER_COMPAT.md section 4).
* Logging is a single line per request in Q-Volta's own format, not llama's.

## Build

    cmake -S . -B build-cpu -G Ninja -DQV_CUDA=OFF && ninja -C build-cpu -j8 qv_server qv_tok
    cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && ninja -C build qv_server qv_tok     # with the CUDA engine

## Run

    ./build-cpu/qv_server -m /workspace/q-volta-models/qv/qwen38-27b-e0 -t 32 -c 4096 --port 8080 --metrics -a qwen3.8-27b

Any llama-server flag spelling works (`-c`, `-ngl`, `-np`, `-ub`, `--temp`, `--top-k`, `--api-key`, `--chat-template-kwargs`, ...) and so do the `LLAMA_ARG_*` variables
(command line wins). Flags with no meaning for Q-Volta print a warning and are ignored; an unknown flag is an error (exit 1), as in llama-server.
Defaults are llama-server's: port **9931**, temperature 0.8, top_k 40, top_p 0.95, min_p 0.05, 4 slots. Use `--temp/--top-k/--top-p` for the Qwen-recommended values.

Q-Volta flags: `--backend cuda|cpu|auto|mock` (`-ngl 0` also selects CPU), `--kv fp32|fp16|int4`, `--mtp` / `--mtp-depth N` (exact greedy MTP lookahead for temperature-0 requests),
`--prefill-chunk N`, `--tokenizer PATH` (optional when the .qv embeds a tokenizer.json; overrides it), `--no-embedded-defaults`, `--max-queue N` (503 beyond N waiting; default unbounded like llama), legacy `--max-ctx`, `--model-name`, `--http-threads`, `--quiet`.
Vision (Inkling image input, docs/INKLING_VISION.md): `--vision-file PATH|none` (default: `<model>-vision.qv` / `<family>-vision.qv` next to the model), `--image-root DIR` (local image files, default the working directory, `""` = off),
`--allow-remote-images` (http(s) `image_url`, SSRF-guarded, needs `curl`; off by default), `--image-max-side N` (4096), `--image-token STR`. `/props` reports `modalities.vision`.
The `mock` backend (scripted tokens, `--mock-text`) is for tests and needs no weights.

Thinking: the Qwen3.8 template opens `<think>`; with `--reasoning-format auto` (= deepseek) the thinking text is returned as `reasoning_content`.
Disable with `"chat_template_kwargs":{"enable_thinking":false}` or `--reasoning off`.

## Tools, grammar, JSON mode

    curl -s localhost:9931/v1/chat/completions -d '{"messages":[{"role":"user","content":"Weather in Paris?"}],
      "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
      "chat_template_kwargs":{"enable_thinking":false}}'
    curl -s localhost:9931/completion -d '{"prompt":"Answer: ","grammar":"root ::= \"yes\" | \"no\""}'
    curl -s localhost:9931/v1/chat/completions -d '{"messages":[...],"response_format":{"type":"json_object"}}'

`tool_choice` auto uses a lazy grammar (unconstrained text until the tool-call marker appears), `required` / a named function constrain the whole answer. Constrained requests skip the engine argmax and MTP fast paths.

## Slots and queueing

`--parallel N` creates N logical slots (default 4). The engine holds one sequence, so one request decodes at a time; others wait FIFO (`/slots` shows one `is_processing`, `/metrics` `requests_deferred`).
Prefix reuse (`cache_prompt`): if the new prompt starts with the tokens already in the engine only the suffix is evaluated (`timings.cache_n`, `usage.prompt_tokens_details.cached_tokens`, `tokens_cached`).
The recurrent state cannot be rewound, so an identical repeated prompt is re-evaluated. Context end stops generation (`stop_type: limit`, `truncated: true`). True multi-slot batched decoding is a later engine-level feature.

## Tokenizer and chat template fidelity

* `qv_tok` (CLI): `qv_tok tokenizer.json encode|decode|chat` reads JSON lines on stdin.
* `tests/test_tokenizer.py build-cpu/qv_tok`: 509 strings (English, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Thai, emoji, code with tabs/CRLF,
  numbers, repeated spaces, special tokens in text, random multi-script strings, NFC edge cases) compared with
  the `tokenizers` library loading `tokenizer.json` verbatim: **509/509 exact**; decode (300 id sequences incl. random ids that split
  UTF-8 sequences): **300/300 exact**.
  Against `transformers.AutoTokenizer` (5.17.0) 498/509 match; all 11 differences are strings containing combining marks:
  transformers' Qwen2Tokenizer rebuilds the pre-tokenizer with a pattern in which `\p{M}` never matches, whereas
  `tokenizer.json` (the pattern the model ships with) does match marks. This server follows `tokenizer.json`.
* `tests/test_chat_template.py build-cpu/qv_tok`: **31/31** conversations match `apply_chat_template` (26 render identically, 5 are
  error cases where both raise): system/user/assistant/tool roles, enable_thinking true/false, reasoning_effort, preserve_thinking,
  reasoning_content, multi-part content, whitespace trimming, no-generation-prompt, invalid role/ordering.
  Not covered: `tools` definitions and assistant `tool_calls` rendering.
* Streaming detokenization buffers incomplete UTF-8 and replaces invalid bytes with U+FFFD (maximal-subpart rule).

## Tests

    python3 -I tests/test_tokenizer.py build-cpu/qv_tok
    python3 -I tests/test_chat_template.py build-cpu/qv_tok
    python3 -m pytest tests/test_server_compat.py -q                 # mock engine; + structural diff vs llama-server if /tmp/llama_srv_build is present
    python3 -I tests/test_server.py --server build-cpu/qv_server     # real model on CPU, ~60 s (hold flock /tmp/qv_bench.lock)

`tests/test_server_compat.py` runs a stand-alone suite (endpoint shapes, SSE framing, errors, queue/slots, CLI/env) and sends the same requests to llama-server (Qwen2.5-0.5B GGUF) and to qv_server
`--backend mock`, comparing status codes, content types, header names, JSON key sets/types and SSE event shapes (documented exceptions whitelisted in the file). Results are in SERVER_COMPAT.md / PROGRESS.md.

## curl examples

    curl -s localhost:8080/v1/models
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "messages":[{"role":"user","content":"Say hi"}], "max_tokens":16, "temperature":0,
      "chat_template_kwargs":{"enable_thinking":false}}'
    curl -sN localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "messages":[{"role":"user","content":"Write a haiku about rain."}], "max_tokens":40, "stream":true,
      "stream_options":{"include_usage":true}, "chat_template_kwargs":{"enable_thinking":false}}'
    curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
      -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0.8,"seed":7,"stop":["\n"]}'
    curl -s localhost:8080/v1/chat/completions -H 'Authorization: Bearer KEY' ...     # with --api-key KEY

Python SDK: `openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")` (not exercised here, package not installed).

## Measured latency (CPU backend, e0 model, 32 threads, max-ctx 1024)

Measured with `/tmp`-side script over real HTTP/SSE, non-thinking chat, greedy, 16 generated tokens. **The machine was not idle:** other jobs
were running (load average 8 before the run, 12-17 during it, including this server's own 32 threads), so these numbers are
pessimistic versus the standalone engine figures (~3.9 tok/s decode, ~20 tok/s prefill).

| request | prompt tokens | time to first token | streaming decode |
|---|---|---|---|
| short chat ("Write a haiku about rain.") | 19 | 0.86 s | 2.73 tok/s |
| ~120-token prompt | 136 | 4.58 s | 2.71 tok/s |
| same, `qwen38-27b-pi-e0` model | 19 / 136 | 0.83 s / 4.56 s | 2.82 / 2.81 tok/s |

Model load: 1.3-2 s with a warm page cache (the CPU engine maps the weights).
No CUDA numbers: the GPU was not used.

## Self-contained .qv (embedded tokenizer / template / generation defaults)

`qv_server --model x.qv` works with no `--tokenizer` and no side files when the container carries the blobs (docs/QV_FORMAT.md; add them with
`tools/qv_embed.py`). Precedence: `--tokenizer` > embedded > sidecar search > error ("no tokenizer ..."); `--chat-template[-file]` > embedded template > template next to
the tokenizer. The embedded generation_config.json supplies the default temperature/top_k/top_p only where the flag was not given (llama-server defaults otherwise;
`--no-embedded-defaults` disables) and its eos ids join the stop set. /props has `qv_*` keys and /v1/models `data[0].qv` has the meta, only for files with blobs/meta.
Tests: `python3 -m pytest tests/test_qv_embed.py` (mock backend, synthetic .qv) and `build/test_blobs`.
