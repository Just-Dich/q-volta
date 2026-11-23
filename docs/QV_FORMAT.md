# The .qv container (format_version 2)

One file holds weights, tokenizer, chat template, generation defaults and metadata, like GGUF:
`qv_server --model x.qv` needs no `--tokenizer` and no side files. Writers: `tools/convert.py --emit-qv`, `tools/convert_inkling.py`,
`tools/qv_pack.py`, `tools/qv_embed.py` (adds blobs to an existing file). Reader: `ModelFile::open` (`include/qv/model.h`).

## Layout

```
offset 0   "QVOLTA1\0"            8 bytes magic
       8   u64 json_len           little endian
      16   u64 data_off           start of the data area; multiple of 2 MiB
      24   JSON index             json_len bytes, UTF-8, compact
           zero padding           up to data_off
data_off   data area              tensors (256-byte aligned), then blobs (64-byte aligned)
```

All offsets inside the index are relative to `data_off`. The runtime rebases them to absolute file offsets on load.

## JSON index

```
{ "format": "qvolta-v1",            // unchanged tag; an old runtime still loads a v2 file
  "format_version": 2,              // absent = 1 (no blobs / meta)
  "config": {...},                  // model config; "arch": "qwen3_5" (default) | "inkling2_loop" | "inkling2_mtp"; "quant": {...}
  "tensors": { name: entry },
  "blobs":   { name: {"off": n, "nbytes": n, "sha256": hex} },   // optional
  "meta":    { ... } }                                           // optional
```

### Tensor kinds (`entry.kind`)

| kind | fields | meaning |
|---|---|---|
| `q` | shape, bits 2..8, group 64/128, off_q/nbytes_q, off_sm/nbytes_sm, optional meta="s8z8" + off_rs/nbytes_rs | bit-plane packed asymmetric group quantization (tools/qv_format.py docstring); shape [E,N,K] = stacked experts |
| `f32`, `f16` | shape, off, nbytes | dense; `lora_f16` pairs are also kind f16 |
| `nvfp4`, `mxfp8` | shape, bits, group, off_q, off_sm, optional off_rs | checkpoint-native passthrough experts (Inkling) |

### Blobs

Named byte strings stored after the last tensor, each 64-byte aligned, stored raw (no compression). Known names:

| name | content |
|---|---|
| `tokenizer.json` | HF tokenizer.json verbatim (12.8 MB Qwen, 26 MB Inkling) |
| `chat_template.jinja` | the Jinja chat template |
| `generation_config.json` | HF generation_config.json verbatim |
| `special_tokens_map.json` | verbatim, when the model ships one |
| `tokenizer_config.json` | subset only (bos/eos/pad/unk token, add_bos_token, model_max_length, ...) |

Unknown blob names are kept by the tools and ignored by the runtime. `sha256` is informational (qv_embed verifies it).

### meta (free-form object)

Written by the tools: `arch`, `qv_format_version`, `context_length`, `quantization` (summary computed from the tensor table),
`converter_git` (set by converters) or `embed_tool_git` (set by qv_embed), `eos_token_ids`, `bos_token_id`, `pad_token_id`, `add_bos`.
Via `--meta k=v` or `--from-dir`: `model_name`, `base_model`, `license`, `source_repo`, `sidecars` (e.g. `{"mtp": "x-mtp.qv", "vision": "x-vision.qv"}`).
The runtime reads `context_length` (/v1/models `n_ctx_train`) and exposes the whole object in /v1/models `data[0].qv.meta`.

## Compatibility and versioning

* v1 files (no `format_version`, blobs, meta) load unchanged; the server then needs `--tokenizer` or a sidecar tokenizer.json as before.
* v2 files load in v1 runtimes: the extra keys are ignored and tensor offsets did not move.
* The tensor area of a file is never changed by qv_embed (verified by sha256 of the data area before/after).
* Bump `format_version` only for changes that old readers cannot ignore; adding blob names or meta keys needs no bump.
* A blob or tensor that extends past the end of the file makes the load fail ("truncated model").

## Read it with 20 lines of Python

```python
import json, struct
def read_qv(path):
    f = open(path, "rb")
    assert f.read(8) == b"QVOLTA1\0"
    jl, data_off = struct.unpack("<QQ", f.read(16))
    index = json.loads(f.read(jl))
    def blob(name):
        e = index["blobs"][name]
        f.seek(data_off + e["off"])
        return f.read(e["nbytes"])
    return index, data_off, blob

index, data_off, blob = read_qv("model.qv")
print(index.get("format_version", 1), index.get("meta", {}).get("model_name"), len(index["tensors"]), "tensors")
tok = json.loads(blob("tokenizer.json"))           # the HF tokenizer
print(blob("chat_template.jinja").decode()[:80])
```

## Tools

```
python3 tools/qv_embed.py model.qv --from-dir HF_DIR -o model-st.qv     # or --tokenizer/--chat-template/--generation-config/--meta k=v
python3 tools/qv_embed.py model.qv --from-dir HF_DIR -o model.qv        # in place: temp file + verify + atomic rename
python3 tools/qv_embed.py --list model-st.qv
python3 tools/qv_embed.py --extract tokenizer.json model-st.qv -o tokenizer.json
./build/qv_tok model-st.qv encode        # tokenizer CLI straight from the container
```

qv_embed streams the tensor area in 32 MiB chunks (no big RAM), checks free disk first (the new file is a full copy, so peak extra
disk = one copy of the model), re-reads the output and checks the tensor-area sha256 and every blob sha256 before the rename.
`--drop NAME` removes a blob. Stripping blobs does not reproduce the original file byte for byte (the index is re-serialized
compactly), only the tensor area is identical; keep the original if you need the exact bytes.

## Runtime precedence (qv_server)

* tokenizer: `--tokenizer` > embedded `tokenizer.json` > sidecar search (`X.tokenizer.json`, `dir/tokenizer.json`) > error "no tokenizer".
* chat template: `--chat-template` / `--chat-template-file` > embedded `chat_template.jinja` > `chat_template.jinja` next to the tokenizer.
* stop ids: `<|im_end|>`, `<|endoftext|>` as before, plus `eos_token_id` of the embedded generation_config.json.
* sampling defaults: embedded `temperature`, `top_k`, `top_p` replace the llama-server defaults (0.8 / 40 / 0.95) only for flags not given on
  the command line or environment; requests still override. `--no-embedded-defaults` turns this off. The applied values are logged at start and
  shown in /props `qv_embedded_defaults`. Plain v1 files behave exactly as before.
* /props gains `qv_chat_template_source` (override|embedded|sidecar|none), `qv_tokenizer`, `qv_embedded_blobs`, `qv_embedded_defaults`;
  /v1/models `data[0].qv` has `meta`, `blobs`, `tokenizer`, `chat_template`. These keys appear only for files with blobs/meta.
* `qv_run` / `qv_ink` take token-id files and have no tokenizer, so nothing changes there; `qv_tok` accepts a `.qv` in place of tokenizer.json.
