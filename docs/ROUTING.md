# Model Routing in `qv_server` (`--models-preset`)

Router mode allows a single `qv_server` process to serve multiple models by automatically unloading the active model from VRAM/RAM and loading the requested preset based on the `"model"` field in incoming API requests.

---

## 1. Startup

Pass the INI presets file via CLI or the `LLAMA_ARG_MODELS_PRESET` environment variable:

```bash
./qv_server --models-preset /models/presets.ini --backend cuda --host 0.0.0.0 --port 8080
```

Example for `docker-compose.yml`:

```yaml
command: ["--models-preset", "/models/presets.ini", "--backend", "cuda"]
```

### Startup Behaviour & Tokenizer Auto-Discovery
1. `presets.ini` is parsed on startup. The file must define at least one named model section with `model = ...`.
2. Setting `tokenizer` in `presets.ini` is **optional**: for each model, the server automatically uses the `tokenizer.json` embedded in the `.qv` file, or looks for `tokenizer.json` (and `chat_template.jinja`) in the same directory as the model's `.qv` file (isolated from any container-level `--tokenizer` argument).
3. By default, **no model is loaded into memory at startup** (all presets start with status `"unloaded"` until the first request or an explicit `POST /models/load` call).
4. To preload a specific model at startup, set `preload = true` (or `load-on-startup = true`) in its section, or pass `-m <preset_id>` on the command line. If the preloaded `.qv` file is missing on disk, the server logs a warning and stays ready to serve the remaining presets.

---

## 2. `presets.ini` Format

* Comments: lines starting with `;` or `#`.
* `[*]` section: global default settings applied to every model section.
* Named `[model-id]` sections: per-model configuration (`model-id` is exposed in `GET /v1/models` and matched against `"model"` in requests).
* Quoted values (`"..."` or `'...'`) are automatically unquoted.
* Key names are case-insensitive and `_` is normalized to `-`.

### Option Precedence
1. Global CLI arguments / `LLAMA_ARG_*` environment variables
2. `[*]` section in `presets.ini`
3. `[model-id]` section in `presets.ini`
4. Embedded `generation_config.json` inside the `.qv` file (never overrides flags explicitly set via CLI or preset)
5. Per-request parameters in the HTTP JSON payload

### Example `presets.ini`

```ini
[*]
backend = cuda
kv-dtype = int4
prefill-chunk = 1024
default-reasoning-effort = medium
default-max-tokens = 8192
max-pending-requests = 32
max-concurrency = 4

[inkling-2-mini]
model = /models/q-project/Inkling-2-Mini-QVolta/inkling2-mini-q3a5.qv
max-context = 32768
default-reasoning-effort = low
vision = true

[qwen3.8-27b]
model = /models/q-project/Qwen3.8-27B-QVolta/qwen38-27b-e1.qv
spec = mtp
draft-tokens = 3
max-context = 131072
vision = false
```

---

## 3. Supported Keys & Aliases in `presets.ini`

### Preset Compatibility Aliases (`llama-server` / `ninfer`)

| Key in `presets.ini` | `qv_server` Equivalent | Notes |
|---|---|---|
| `max-context`, `kv-capacity` | `--ctx-size` | Context size (`n_ctx`) |
| `kv-dtype` | `--kv` | Supports `fp32`, `fp16`, `int4`. Other values (e.g. `int8`, `rk4v4`) log a warning and fall back to `int4` |
| `spec` | `--mtp` / `--no-mtp` | `mtp` enables MTP; `none`/`off`/`false` disables it; others (e.g. `dspark`) warn and are ignored |
| `draft-tokens` | `--mtp-depth` | MTP draft lookahead depth |
| `default-reasoning-effort` | `--reasoning-effort` | Default reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh` depending on template) |
| `default-max-tokens` | `--predict` (`n_predict`) | Default maximum generated tokens |
| `default-thinking-budget` | `--reasoning-budget` | `0` disables thinking in the chat template |
| `max-pending-requests` | `--max-queue` | Maximum waiting requests in queue (excess returns HTTP `503`) |
| `max-concurrency` | `--parallel` | Number of logical slots |
| `vision` | `--vision-file` | `false`/`0`/`off` disables vision (`vision-file = none`); `true`/`1`/`on` enables `*-vision.qv` auto-discovery |
| `preload`, `load-on-startup` | — | `true` preloads this preset at server startup (default `false`) |

### Standard `qv_server` / `llama-server` Keys
Inside `[*]` and `[model-id]`, all supported server flags work without the `--` prefix:
* **Model & Engine:** `model`, `tokenizer`, `backend`, `threads`, `n-gpu-layers` (`gpu-layers`), `ctx-size`, `prefill-chunk`, `ubatch-size`, `batch-size`, `kv`, `kv4-kmode`, `mtp`, `no-mtp`, `mtp-depth`, `mtp-file`, `no-embedded-defaults`, `alias`.
* **Vision:** `vision-file`, `allow-remote-images`, `image-root`, `image-max-side`, `image-token`.
* **Templates & Reasoning:** `jinja`, `chat-template`, `chat-template-file`, `chat-template-kwargs`, `reasoning`, `reasoning-format`, `reasoning-budget`, `reasoning-effort`, `special`.
* **Sampling & Constraints:** `predict`, `temp`, `top-k`, `top-p`, `min-p`, `typical`, `top-nsigma`, `xtc-probability`, `xtc-threshold`, `repeat-penalty`, `repeat-last-n`, `presence-penalty`, `frequency-penalty`, `dry-*`, `mirostat*`, `dynatemp-*`, `seed`, `samplers`, `ignore-eos`, `logit-bias`, `grammar`, `grammar-file`, `json-schema`, `json-schema-file`, `cache-prompt`, `sleep-idle-seconds`.

### Ignored / Unsupported Keys
* **Server-wide options** (`host`, `port`, `models-preset`, `models-max`) inside INI sections warn and are ignored — pass them on the CLI or via environment variables.
* **Third-party engine keys** with no Q-Volta equivalent (`reasoning-loop`, `thinking-budget-message`, `kv-lease-growth`, `concurrent-prefill`, `decode-rounds-per-prefill`, `expert-residency`, `expert-cache-mib`, `gdn-state-fp16`, `vision-residency`, etc.) warn once and are ignored rather than failing startup.

---

## 4. Router HTTP API

### Automatic Model Switching via `"model"`
Model switching triggers automatically when the JSON body specifies a different `model-id` (or one of the preset's `alias` values):
* `POST /v1/chat/completions`, `POST /chat/completions`
* `POST /v1/completions`, `POST /completions`, `POST /completion`
* `POST /v1/messages` (Anthropic API)
* `POST /v1/responses`, `POST /responses` (OpenAI Responses API)
* `POST /infill`
* `POST /apply-template`, `POST /tokenize`, `POST /detokenize`
* `POST /v1/chat/completions/input_tokens`, `POST /v1/responses/input_tokens`, `POST /v1/messages/count_tokens`

**Behaviour:**
* If `"model"` is omitted or empty, the currently active model is used (or the first preset in the file if no model is loaded yet).
* If `"model"` does not match any preset or alias, the server returns HTTP **`404`** with error type `model_not_found`.

### Model Management & Status
* **`GET /v1/models`** and **`GET /models`**
  Returns all presets from `presets.ini`. Each entry includes `status.value` (`"loaded"` for the active in-memory preset, `"unloaded"` otherwise).
* **`POST /models/load`** (and `/v1/models/load`)
  Explicitly loads the specified preset into memory:
  ```json
  {"model": "qwen3.8-27b"}
  ```
  Response: `{"success": true, "model": "qwen3.8-27b"}`.
* **`POST /models/unload`** (and `/v1/models/unload`)
  Unloads the active model and frees VRAM/RAM:
  ```json
  {}
  ```
  *(or `{"model": "qwen3.8-27b"}` to unload only if that model is currently loaded).*

---

## 5. Operational Notes

1. **Single Active Engine in Memory**
   `qv_server` keeps one active engine in VRAM/RAM at a time. On a model switch:
   * the server waits for the in-flight request to finish;
   * unloads the previous engine, tokenizer, chat template, vision encoder, and resets KV/prompt caches;
   * loads the target preset and re-initializes logical slots for its `parallel` (`max-concurrency`) setting.
2. **Request Serialization & Client Aborts**
   Model switching happens under an exclusive engine lease (`acquire_model_lease`). If a waiting client disconnects before acquiring the lease, its request is cancelled without triggering an unnecessary model reload.
3. **Prompt Cache Reset on Switch**
   KV and prefix caches are not preserved across model switches; switching back to a previously used model will re-prefill its prompt from scratch.
4. **Non-Blocking Monitoring**
   `GET /health`, `GET /v1/models`, and `GET /props` are thread-safe and do not block while a model engine is being loaded.
5. **Idle Sleep Compatibility (`--sleep-idle-seconds`)**
   When idle sleep is configured, the active engine unloads after the idle timeout (`"unloaded"`) and automatically reloads on the next request.
