# Q-Volta

An original C++20/CUDA inference engine for **Qwen3.8-27B** on a single **NVIDIA Tesla V100-SXM2-16GB** (sm_70), built GPU-first.
A native CPU backend (x86 AVX2, later ARM/NEON) is planned and deliberately **not started** until the GPU stage is complete.

**Status: early research code, Stage A (GPU) in progress. Not a release.** All numbers are measured on one V100 machine; see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) for methodology, caveats and the comparison with NInfer, and
[docs/PROGRESS.md](docs/PROGRESS.md) for the log.

## What works (tested)
- Quantizer/converter (`tools/convert.py`): streams BF16 weights, writes the Q-Volta bit-plane format (asymmetric group-64, 2-8 bit), optional PEFT LoRA merge (`--lora`).
- Full-model CUDA forward: step (decode, int8-activation dp4a GEMV at 81-99% of HBM peak, CUDA-graph) and batched prefill (dequant slab + cuBLAS HGEMM, in-kernel GDN recurrence, causal attention). Golden tests against a simulated-quantization reference pass on real weights.
- Measured on V100: plain decode 52 tok/s (E0 quantization, 12.6 GiB); cold prefill 1013 / 1205 / 1143 tok/s at 512 / 2048 / 4088 tokens (NInfer on the same GPU: 837 / 1010 / 1013).

## Known limitations (read before comparing)
- **Quality is behind the strongest baseline**: E0 is plain round-to-nearest (+12% wikitext perplexity vs BF16); the comparison engine is +0.3%. Part of our decode lead is simply fewer bytes read. Improving quantization is the next big item.
- No MTP/speculative decoding yet; no server/CLI; KV cache is fp32 (about 20K tokens fit); long-context (128K/256K) is untested.
- `src/tokenizer/` is unfinished and untested.
- The pi LoRA adapter's effect is partly blurred by 3-bit rounding; runtime LoRA is planned.

## Build and test
```
cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && ninja -C build
./build/test_dequant && ./build/test_gdn_seq && ./build/test_attn_seq
python3 -I tests/gen_gemv_cases.py build/gemv_cases && ./build/test_gemv build/gemv_cases q8
```
Full-model golden and benchmarks need the converted model (not included): see `CLAUDE.md`.

## Third-party
Written from scratch. Other engines (NInfer, llama.cpp) were read for design ideas and used only as black-box benchmark baselines; no code copied.
Model weights are not part of this repository and carry their own licences. License for this code: to be chosen by the maintainers.
