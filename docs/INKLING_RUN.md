# Running Inkling-2-Mini in Q-Volta

Model file: `inkling2-mini-q3a5.qv` (9.89 GiB; experts 3-bit, attention 5-bit, embeddings 4-bit; produced by `tools/convert_inkling.py --profile q3a5`, see INKLING_FORMAT.md).
Tokenizer and chat template come from the original repository (`thinkingmachines/Inkling-2-Mini`: `tokenizer.json`, `chat_template.jinja`).

```bash
# OpenAI / llama-server compatible server on the V100
CUDA_VISIBLE_DEVICES=0 ./build/qv_server --model inkling2-mini-q3a5.qv --tokenizer Inkling-2-Mini/tokenizer.json \
    --backend cuda --kv int4 --max-ctx 131072 --port 8080
# CPU backend (AVX2, 32 threads)
./build/qv_server --model inkling2-mini-q3a5.qv --tokenizer Inkling-2-Mini/tokenizer.json --backend cpu --threads 32
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Measured through the server (Tesla V100, int4 KV, 2026-11-22, single requests, one run each)
| request | result |
|---|---|
| 31-token prompt, 256 new tokens | decode **120.3 tok/s** |
| 16,045-token prompt (default prefill chunk), 32 new tokens | prefill **1,924 tok/s**, decode 110.7 tok/s |
| load time | 28.7 s |

Engine-level numbers (CUDA graph, synthetic KV at long positions, `docs/INKLING_ENGINE.md`): decode 122 tok/s at ctx 25, 101 at 32K, 76 at 128K, 57 at 256K; prefill 2,764 tok/s at 4K with chunk 4096 (`--prefill-chunk 4096`; not measured through the server); largest context that fits with int4 KV about 580K tokens.
The server tokenizer was checked against the Hugging Face `tokenizers` library on a mixed English/Russian/code string: identical ids.

## Not available / caveats
- **No MTP (speculative) decoding for Inkling yet**: the draft-head graph is not specified by the Hugging Face reference (see INKLING_SPEC.md section 14); expect ~120 tok/s plain, not the 2-3x that Qwen gets from MTP.
- Decode is latency-bound (many short kernels, about a third of the HBM bandwidth bound): kernel fusion is the next speedup, not shown here.
- Quality: Q3A5 is a 3-bit expert quantization; on held-out text its NLL is 1.4643 vs 1.4479 for the fp32 reference (README text 1.7620 vs 1.7561), no tool-calling or long-context quality evaluation was run. The engine matches the Hugging Face implementation on 4 golden prompts (bf16-level noise, 31-32 of 32 tokens, `docs/INKLING_SPEC.md` section 15); long-context (16K+) whole-model parity is untested.
- The model thinks by default: the reasoning appears in `content` unless `--reasoning-format` / the template's thinking switch is used.
- Server limits as for Qwen: one sequence at a time (requests queue), `/embeddings` unsupported.
