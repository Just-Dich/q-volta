# qv_server — OpenAI-compatible HTTP server

`qv_server` serves a Q-Volta model over HTTP/1.1 with the OpenAI `chat/completions` and `completions` schemas
(non-streaming and SSE streaming). It is plain C++20 on POSIX sockets: no third-party dependencies
(JSON, tokenizer, chat template and sampling are in `src/core/json_doc.h`, `src/tokenizer/`, `src/sampling/`).
One generation runs at a time (the engine is stateful); other requests wait in a bounded queue.

## Build

CPU only (no CUDA toolkit needed):

    cmake -S . -B build-cpu -G Ninja -DQV_CUDA=OFF && ninja -C build-cpu -j8 qv_server qv_tok

With the CUDA engine (`--backend cuda` selectable; `--backend auto` picks CUDA only if a device exists):

    cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && ninja -C build qv_server qv_tok

Both configurations were compiled. Only the CPU backend was run for this work (CUDA_VISIBLE_DEVICES empty; in the
CUDA build, `--backend auto` with no visible device fell back to `cpu`). The CUDA path in the server is untested at runtime.

## Run

    CUDA_VISIBLE_DEVICES= ./build-cpu/qv_server --model /workspace/q-volta-models/qv/qwen38-27b-e0 \
        --backend cpu --threads 32 --max-ctx 4096 --port 8080

| flag | meaning (default) |
|---|---|
| `--model PREFIX` | model prefix (`PREFIX.qv.json/.qv.bin`), required; `qwen38-27b-pi-e0` works the same way |
| `--backend cuda\|cpu\|auto` | `auto`: cuda if built and a device exists, else cpu |
| `--threads N` | CPU worker threads (min(32, cores)) |
| `--max-ctx N` | context window / KV size (4096) |
| `--prefill-chunk N` | tokens per prefill call; also the granularity at which a client disconnect is noticed during prefill (cpu 256, cuda 2048) |
| `--host H --port P` | `127.0.0.1:8080` |
| `--api-key KEY` | require `Authorization: Bearer KEY` on `/v1/*` (`/health` stays open) |
| `--model-name NAME` | id reported in `/v1/models` and responses (`qwen3.8-27b`); any `model` in requests is accepted |
| `--tokenizer PATH` | tokenizer.json (default: `PREFIX.tokenizer.json`, else `<prefix dir>/../Qwen3.8-27B/tokenizer.json`) |
| `--http-threads N` | connection worker threads (16) |
| `--max-queue N` | requests allowed to wait behind the running one; more get `503` (8) |
| `--default-max-tokens N` | used when the request has no max_tokens (1024) |
| `--temperature/--top-p/--top-k` | defaults when a request omits them (1.0 / 0.95 / 20, from the model's generation_config.json) |
| `--reasoning-format none\|separate` | `none` (default): raw model text, thinking text and `</think>` stay in `content`. `separate`: text before `</think>` goes to `reasoning_content`, the rest to `content` |
| `--quiet` | no per-request log line |

## Endpoints

* `GET /health` -> `{"status":"ok"}`
* `GET /v1/models`, `GET /v1/models/{id}`
* `POST /v1/chat/completions`, `POST /v1/completions` (`stream` true/false)
* `OPTIONS *` -> 204 with CORS headers (`Access-Control-Allow-Origin: *`)
* Errors use `{"error":{"message","type","param","code"}}`: 400 (bad JSON / params), 401 (bad key), 404 (route), 405,
  413 (prompt >= max-ctx, code `context_length_exceeded`), 503 (queue full), 500.

Debug headers on non-stream and stream responses: `X-QV-Prompt-Cached-Tokens` (prompt tokens reused from the engine
state), `X-QV-Prompt-Tokens`. `usage.prompt_tokens_details.cached_tokens` carries the same number.

### Supported request parameters

`model` (ignored), `messages` / `prompt` (string, list of strings -> first, token-id list, list of token-id lists -> first),
`max_tokens` / `max_completion_tokens` (clamped to the remaining context), `temperature` (0 = greedy, <= 2), `top_p`, `top_k`,
`min_p`, `seed` (deterministic: same seed + same request -> same text), `stop` (string or up to 4 strings),
`presence_penalty`, `frequency_penalty` (OpenAI semantics, over generated tokens), `repetition_penalty` (extension; prompt + generated),
`stream`, `stream_options.include_usage`, `n` (only 1), `logprobs`/`top_logprobs`, `chat_template_kwargs`
(`enable_thinking`, `reasoning_effort` xhigh|medium|low, `preserve_thinking`; the same keys are also accepted at top level;
OpenAI `reasoning_effort` high -> xhigh, minimal -> low, none -> thinking off).

Behaviour notes:

* **Thinking mode**: the Qwen3.8 template thinks by default (generation prompt ends with `<think>\n`, and a reasoning-effort
  system line is injected). Pass `"chat_template_kwargs":{"enable_thinking":false}` for direct answers. Output with thinking
  includes the reasoning text and `</think>` in `content` unless `--reasoning-format separate`.
* **Finish reasons**: `stop` (an EOS token, `<|im_end|>` or `<|endoftext|>`, or a stop string) / `length`. Usage
  `completion_tokens` counts every sampled token including the terminating EOS token.
* **Defaults**: when a request omits sampling parameters the server uses temperature 1.0, top_p 0.95, top_k 20.
  Greedy requests without penalties use the engine's `argmax()` fast path; everything else copies host logits.
* **logprobs**: chat `logprobs:true` + `top_logprobs` (0-20) and completions `logprobs: N` return the log-softmax of the raw
  (pre-penalty, temperature-1) logits for each generated token with the top-N alternatives. The EOS token is not listed.
  Costs a full-vocabulary pass per token on the host.
* **Prefix reuse**: the model is recurrent (GatedDeltaNet), so the context cannot be rewound. The server keeps the token
  sequence that is inside the engine; if a new prompt starts with exactly that sequence only the new suffix is prefilled,
  otherwise the engine is reset and the whole prompt is prefilled. A chat that continues the previous turn reuses the cache
  when the re-rendered history tokenizes to the same ids as what was generated (true for the tested non-thinking turn). In thinking mode the
  template re-renders earlier assistant turns as `<think>\n\n</think>\n\n...` (or with the client-sent `reasoning_content`),
  which differs from what was generated, so those continuations are prefilled in full.
* **Disconnects**: the socket is polled between tokens and between prefill chunks; on a closed peer the generation stops, the
  engine keeps exactly the tokens it has processed, and the next request proceeds (tested).
* Prompt tokenization parses special-token text (`<|im_end|>` etc.) inside message content as special tokens, like HF does.

### NOT supported (rejected with 400 or ignored as noted)

* tools / function calling / `tool_choice` / `functions` (400); assistant `tool_calls` in history (400). `tool`-role messages are rendered.
* `n > 1` and `best_of > 1` (400), `echo`, `suffix`, `logit_bias`, non-text `response_format` (400).
* Vision / audio / video content parts (400); text parts only.
* `logprobs` beyond what is listed above (no per-token `bytes` merging across multi-byte characters, no prompt logprobs).
* Chunked request bodies (411), HTTPS, multiple concurrent generations (requests are serialized), batching.
* Case-insensitive contractions match ASCII only (`'S`, `'RE`); the Unicode case-fold forms U+017F / U+212A are not matched.

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
    python3 -I tests/test_server.py --server build-cpu/qv_server     # starts servers on free ports; ~60 s on CPU

`test_server.py` result: **45/45 checks passed** (CPU backend, e0 model). The `openai` python package is not installed here, so the
optional SDK checks were skipped.

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

Model load: 1.3-2 s with a warm page cache (the CPU engine maps the weights). HTTP/JSON/tokenizer overhead is negligible next to
the engine (tokenizing a prompt is microseconds-to-milliseconds). No CUDA numbers: the GPU was not used.
