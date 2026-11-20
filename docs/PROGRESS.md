# Q-Volta — Progress log (newest last)
- 2026-11-19: environment audit (V100 free, CUDA 12.9; CPU = Xeon E5-2683 v4, AVX2 only, no AVX-512). Prior sessions: engine skeleton, dp4a GEMV at 81-99% of HBM peak, CUDA-graph decode.
- 2026-11-19: real E0 model converted (12.63 GiB); golden PASS on real weights; decode 51.1-52.2 tok/s.
- 2026-11-19: requirements added: pi LoRA adapter (converter --lora, merge before quant), ARM/Snapdragon design constraint for Stage B (no ARM tooling here), prefill must beat competitors, 256K context.
- 2026-11-20: B0 baseline: NInfer re-run on this V100 (plain 45.3, MTP-3 106.5 code / 69.0 prose). Bytes-normalized comparison shows most of our raw decode lead is fewer bytes.
- 2026-11-20: adapter analysis: RTN noise >> adapter shift; runtime LoRA planned.
- 2026-11-20: research fan-out (decode kernels, MTP, prefill, quantization quality) -> PLAN.md (hypotheses with mechanism/expected/falsification).
- 2026-11-20: P1 batched prefill: dequant_rows_f16 (751 GB/s), gdn_seq (2.15 us/token/layer), attn_seq (8.4 TFLOP/s), cuBLAS slab GEMM; golden PASS; prefill 51 -> 1013 tok/s @512 (NInfer 837), 917 @4088 (NInfer 1013).
- 2026-11-20: prefill chunk 512 -> 2048 (fewer dequant passes): 1205 / 1143 tok/s @2048/4088 vs NInfer 1010/1013 (+19%/+13%); multi-chunk golden PASS.
- 2026-11-20: split-K (flash-decoding) decode attention wired in (13x faster kernel at 32K ctx, 8x at 4K; kernel test vs old kernel rel err <=6.4e-6, q8 bit-identical). End-to-end decode after a 4088-token prompt: 36 -> 50.2 tok/s; short context 52.3 (unchanged). Scratch scores[24][max_ctx] replaced by fixed 1.9 MiB.
- 2026-11-20: CPU backend v1 (AVX2 GEMV + ops + engine): 3.93 tok/s decode @32 threads (~49-54 GB/s of RAM, bandwidth-bound), golden PASS (thin margin). GPU work paused at user request.
- 2026-11-20: CPU batched prefill (gemm_q, 256-token chunks): ~21-24 tok/s @32 threads vs 3.2 (step loop) under heavy host load (lower bound); golden PASS on prefill path (top1 0.9524, KL 5.1e-3).
Next: tensor-core attention + dequant/GEMM overlap (prefill at long prompts), runtime LoRA, MTP, KV quantization + split-K decode attention, quantization quality (clip search, bit allocation, GPTQ), NIAH/ppl at 32K-256K.
