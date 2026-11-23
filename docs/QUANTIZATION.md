# Quantization (p10/quant, 2026-11-21)

Format: per-group asymmetric (scale, min), bit-plane packed (tools/qv_format.py). New in this branch:
* group 128 per tensor (kernels: CUDA all paths; CPU engine expands to 64-groups at load), tests in test_gemv / test_gemv_multi / test_dequant cover g64, g128 and meta8 (bit-exact where they were bit-exact before).
* meta8 metadata: 8-bit scale + 8-bit zero point per group + fp16 row scale = 0.125 bit/weight at g128 (fp16 scale+min is 0.25). Weight-MSE change on real tensors 0.02-0.07%.
* tools/qv_quant.py: RTN, clip search (per-group MSE shrink grid), GPTQ (Hessian from calibration, static groups, params from error-compensated weights).
* tools/convert.py: `--method rtn|gptq --group --clip --meta8 --plan --calib --emit-qv --sens --head-sweep --lora-section`; resumable in slices (`tools/quant_slices.sh`, takes /tmp/qv_bench.lock per slice). `tools/alloc_bits.py` = knapsack over a sensitivity table. `tools/make_calib.py` = calibration set (16 wikitext-train, 8 code, 8 chat windows of 1024). `tools/qv_remtp.py` re-quantizes only the MTP head (not run: see below).
* Optional LoRA section in the container (kind "lora_f16", names `lora.<weight>.A/.B`, config.lora): written by `--lora-section`, ignored by the loader; LoRA runtime NOT implemented.

## Results (same eval for all: BF16 ppl 8.468, 4088 wikitext-2 test tokens; KL/top-1 vs BF16 logits)
| profile | file GiB | bpw | ppl | KL | top-1 | note |
|---|---:|---:|---:|---:|---:|---|
| BF16 | 50.3 | 16 | 8.468 | 0 | 100 | |
| E0 RTN g64 (mlp3/gdn4/attn4/e4/h6) | 12.63 | 4.03 | 9.508 (+12.3%) | 0.1224 | 84.6 | reproduced exactly |
| E0 bits, RTN g128 | 11.83 | 3.77 | 9.523 (+12.5%) | 0.1469 | 83.6 | lever 1 alone: -0.79 GiB, KL +20% |
| E0 bits, RTN g64 + clip | 12.63 | 4.03 | 9.470 (+11.8%) | 0.1152 | 85.5 | clip search alone |
| E0 bits, GPTQ g64 + clip | 12.63 | 4.03 | 8.984 (+6.1%) | 0.0796 | 88.1 | GPTQ halves the loss |
| E0 bits, GPTQ g128 + clip | 11.83 | 3.77 | 9.082 (+7.3%) | 0.0891 | 87.6 | GPTQ g128 beats RTN g64 at -0.8 GiB |
| e1 = all 3-bit, e4, h5, GPTQ g128 + clip + meta8 | 10.44 | 3.32 | 9.278 (+9.6%) | 0.1122 | 85.2 | RECOMMENDED size profile |
| e1a = knapsack (rel-error proxy, 2-bit allowed) | 10.28 | 3.27 | 9.847 (+16.3%) | 0.1777 | 82.1 | falsified |
Decode (qv_run --graph, shared GPU, one run): E0 48.7, e1 55.4 tok/s (bandwidth-proportional; e1 output coherent). 
Head (final hidden of the e1 run, wikitext, KL): 4-bit RTN m8 0.1252 / 5-bit 0.1140 / 6-bit 0.1122 / 4-bit GPTQ 0.1149 (head GiB 0.61/0.76/0.91/0.61): GPTQ 4-bit head = RTN 5-bit at 0.15 GiB less (not applied in e1; would give ~10.29 GiB).

## Honest conclusions
* <=10.4 GiB at +5% ppl is NOT reached. Reaching 10.4 GiB needs ~3 raw bits on every tensor; best measured there is +9.6% ppl (KL 0.112), i.e. e1 is 17% smaller than E0 with slightly better quality (E0 +12.3%, KL 0.122) but worse than E0-bits GPTQ (+6.1%, 12.63 GiB). Pareto points: (12.63, +6.1%), (11.83, +7.3%), (10.44, +9.6%). With meta8 the 11.83 GiB profile would be ~11.4 GiB (not built).
* H1 g128: size -0.79 GiB; RTN KL +20%, GPTQ KL +12% -> confirmed as a size lever, costs quality. H3 GPTQ: confirmed (KL -35%, ppl +12.3% -> +6.1%). Clip search: small alone (KL -6%). meta8: no measurable loss, -0.39 GiB.
* H2 sensitivity allocation: only a RTN-clip proxy table was measured (tools/alloc_bits.py, 256 units). The proxy plan with 2-bit tensors was worse than uniform 3-bit at similar size: falsified in this form (needs calibrated class weights; output-error/residual normalisation also unreliable across layer types). Uniform-3 + sensible early-layer upgrades not tried.
* H4 embed: only 4-bit tested (kept). Head: see above. H5 entropy coding: not tried. H6 MTP bits / acceptance: NOT measured (qv_remtp.py written, untested; e1 quantizes MTP at 3-bit RTN clip, acceptance with it unmeasured).
* Agentic set (17x512, secondary): ppl numbers are noisy and KL is tail-dominated (p99 > 15 for E0); e1 is worse than E0 there (KL 1.34 vs 1.13, top-1 74.9 vs 79.4) and GPTQ g128 (1.25) was not better than RTN g64+clip (0.95). Calibration has no agentic text; treat e1 with caution for tool calling.
* Not done: pi LoRA build with e1, CPU engine run on e1 (CPU kernels unchanged, expansion at load; ctest cpu tests pass), prefill re-measure, 256K fit check (e1 weights 10.44 GiB + 0.65 scratch measured 11.4 GiB device).

## Self-contained files (p23/embed)

The .qv now optionally embeds tokenizer.json, chat_template.jinja, generation_config.json and a meta object (quantization summary, base model, license, eos ids, context length); see docs/QV_FORMAT.md.
`tools/convert.py --emit-qv` and `tools/convert_inkling.py` embed the files found in the model dir by default (`--no-embed` to skip); the tensor bytes and therefore all quality numbers above are unchanged.
