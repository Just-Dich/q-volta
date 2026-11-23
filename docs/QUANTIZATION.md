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

## p21/quant2 (2026-11-23): quality at equal bytes, e1b1 measured, e0b1 not run
Motivation: benchmarks/compare shows llama.cpp ahead at ~equal size. Note the compare table's e0 is RTN; GPTQ at E0 bits (above: 12.63 GiB, ppl 8.984, KL 0.080 vs BF16) was never put through the compare protocol, so most of the e0 gap is "RTN vs GPTQ". Targets (ppl 8.9 / KL-Q8 0.055 at <=12.7 GiB; 9.0 / 0.075 at 10.4 GiB) are NOT yet reached or measured.

Proxy loop (tools/qv_ablate.py, tools/ablate_slices.sh; data benchmarks/results/quant2_2026-11-23.json): layer-local output error on a HELD-OUT calibration text (different seed; wikitext-train/stdlib code/chat, never the eval text), layers 2/15/30/43/55/62, 3-bit g128 meta8, geometric mean over layers, a = qkv/z/q/k/v, o = out/o_proj, m = gate/up, d = down:
| variant | a | o | m | d |
|---|---:|---:|---:|---:|
| RTN + clip | 0.0096 | 0.0265 | 0.0144 | 0.0240 |
| GPTQ (original sweep), 128 seqs | 0.0045 | 0.0158 | 0.0092 | 0.0209 |
| + act-order, static groups (damp .01) | 0.0042 | 0.0148 | 0.0092 | 0.0213 |
| + damp 0.1 | 0.0042 | 0.0143 | 0.0090 | 0.0192 |
| + damp 0.3 | 0.0045 | 0.0145 | 0.0090 | 0.0185 |
Findings: (1) GPTQ on down_proj overfits the calibration Hessian (17408^2 from 131k tokens): train error 0.005, held-out 0.013-0.02, barely better than RTN; heavy damping (0.3) or diag shrinkage gives -12%. (2) act-order with static groups: -7% (a), -6% (o), neutral on m/d. (3) calibration 32 -> 128 sequences: -12..-25% held-out error (a/o/m). (4) 4-bit = 0.26x the error of 3-bit on every tensor and layer; g64 = 0.87x of g128 for +0.125 bit/weight, i.e. g64 is the cheaper first spend of bytes (1.04 E3 per bit vs 0.74 E3 per bit for 3->4); 5-bit 0.07x. (5) Scale/zero refit under the full Hessian: 0.98x (noise); the block-diagonal refit is harmful (ignores GPTQ error compensation). Proxy error is not model KL: it must be confirmed by full runs.
Tools added: `qv_quant.gptq_static` (act-order + static groups + Hessian shrinkage), `refit_full`, `effective_weights` (GPTAQ-style asymmetric target W C H^-1; implemented, smoke-tested only at 2 layers, runs out of GPU memory at 256 calibration sequences, NOT evaluated), `convert.py --gptq-mode ao --damp-map a=.01,o=.1,m=.1,d=.3 --calib calib256.npy --calib-bf16 1 --gptaq --refit --percdamp`, `tools/make_plan.py` + `tools/plan_size.py` (class plans and size estimates, calibrated to e1 10.44 GiB), `make_calib.py` seed env (calib128/256/heldout32 in /workspace/q-volta-models/calib).
Recipes started (full run ~2-3.5 h each on the shared GPU, restartable per 6 layers; NOT finished at the time of writing):
* e1b1 (10.40 GiB, MEASURED, below): `tools/make_plan.py plan_e1b1.json 'gu=3:128,down=3:64,qkvz=3:128,o=3:64'`; `tools/quant_slices.sh LOG 6 --out qv/qwen38-27b-e1b1 --eval-dir EV --eval-text2 eval/agentic_synth.txt --method gptq --gptq-mode ao --damp-map a=0.01,o=0.1,m=0.1,d=0.3 --group 128 --clip 1 --meta8 1 --calib calib256.npy --calib-bf16 1 --calib-batch 4 --head-method gptq --emit-qv --plan plan_e1b1.json --bits mlp_gu=3,mlp_down=3,gdn=3,attn=3,embed=4,head=4,mtp=3`. 
* e0b1 (~12.3 GiB est.): same flags, plan `gu=3:64,down=4:64,qkvz=4:64,o=4:64`, `--bits mlp_gu=3,mlp_down=4,gdn=4,attn=4,embed=4,head=5,mtp=3`. Not started (disk).
Not tried: rotations, QAT-lite, entropy coding, vector quantization (needs a new kernel), per-layer KL-guided allocation beyond the class plans above.

### e1b1 measured (2026-11-23)
| model | GiB | ppl 8x512 | ppl 4088 | KL to Q8_0 | top-1 vs Q8_0 | KL to BF16 (convert eval) | agentic KL (BF16) | decode plain / MTP-3 code / prose |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e1 | 10.44 | 9.278 | 6.281 | 0.0923 | 87.1% | 0.1122 | 1.34 | 57.6 / 138.5 / 88.4 |
| e1b1 | 10.40 | 9.112 (+7.6%) | 6.102 | 0.0750 | 89.3% | 0.0897 (ppl 9.110) | 1.370 | 57.5 / 137.2 / 92.2 |
| llama.cpp IQ3_XXS+im (10.89 GiB) | 10.89 | 8.992 | 6.102 | 0.0751 | 89.0% | n/a | n/a | 38.7 |
e1b1 meets the KL target (<=0.075) and matches IQ3_XXS on ppl 4088 and KL, with 4.5% fewer bytes and 1.5x its decode speed; it misses the ppl 8x512 target (<=9.0: 9.112) and does not improve the agentic KL (1.370 vs e1 1.34, E0 1.13), so the agentic caveat for 3-bit profiles stands. Speed: no regression (plain -0.2%, MTP-3 code -1%, prose +4%). Overfitting: calibration (256 windows of wikitext-train/stdlib code/chat) is disjoint from eval text; the recipe settings (damping, act-order, g64 classes) were chosen on a separate held-out calibration shard, not on the eval windows; each was checked once on the eval set, so residual selection risk is small but nonzero. Not run: e0b1 (~12.3 GiB), GPTQ E0 bits through the compare protocol, GPTAQ, pi-adapter rebuild with this recipe.
