# Inkling-2-Mini vision (image input)

Status: implemented (sidecar converter, CPU + CUDA encoders, engine `prefill_embeds`, server `image_url`). Validated against the Hugging Face
PR #51050 vision tower (numbers in section 9). Everything below is tagged by source:
**[PR file:line]** = the checked-out PR source `scratch-hfparity/tf/src/transformers/` (`models/inkling2/modeling_inkling2.py` = `I2`,
`models/inkling2/image_processing_inkling2.py` = `IP`, `models/inkling2/processing_inkling2.py` = `PR`, `models/inkling2_loop/modeling_inkling2_loop.py` = `I2L`,
`vision_utils.py` = `VU`, `conversion_mapping.py` = `CM`), **[cfg]** = `config.json` / `processor_config.json`, **[tok]** = `tokenizer.json` / `chat_template.jinja`,
**[ckpt]** = safetensors header, **[README]** = model card, **[assumption]** = not stated by any of those.

## 1. Pipeline in one paragraph
Image -> RGB (alpha composited onto a checkerboard) -> **no resize**: pad right/bottom to a multiple of 40 px -> cut into 40x40 tiles, one tile = one text token
-> each tile is 2x2 native 20x20 patches -> ViT (23 layers, 2-D axial RoPE, learned position table bilinearly resampled) with one class token per image
-> the 4 patch states of a tile are concatenated (4096) -> MLP 4096 -> 4096 -> 2304 -> one 2304-vector per tile. The text sequence contains one
placeholder token per tile; the placeholder rows of the text residual stream are **replaced** by these vectors. The text decoder is otherwise unchanged
(causal, same positions, no special image attention).

## 2. Preprocessing
| step | rule | source |
|---|---|---|
| RGB | PIL: if the image has an alpha channel (`has_transparency_data`: RGBA/LA/PA or `info["transparency"]`) it is converted as RGBA, else RGB | IP:51-53 |
| alpha | `out = (rgb*a + bg*(255-a) + 127) // 255` (int32) with `bg = 204` where `(y//8 + x//8)` is even, else `153` (8 px checkerboard) | IP:59-68 |
| resize | **none**. `size = {height: 40, width: 40}` is the tile size, not a resize target | IP:41, [cfg] processor_config.json |
| grid | `rows = ceil(H/40)`, `cols = ceil(W/40)` | IP:87 |
| pad | bottom/right, fill value `-1.0` on the 0..255 scale (before rescale/normalise), so padded pixels become `(-1/255 - 0.5)/0.5 = -1.0078` | IP:89-90, `tvF.pad` padding=(0,0,pw,ph) (`image_processing_backends.py:219`) |
| rescale | `x * 0.00392156862745098` | [cfg] |
| normalise | mean 0.5, std 0.5 per channel -> range [-1, 1] | [cfg], IP:36-37 |
| layout | `(C, rows, 40, cols, 40) -> (rows, cols, 40, 40, C)`, flattened to `[rows*cols][40][40][3]` (HWC inside a tile, tiles in raster order) | IP:92-93 |

The README says "each image dimension should be between 40 px and 4096 px" **[README]**; the PR does not enforce it (smaller images are padded to one tile, larger ones
just produce more tokens). Q-Volta rejects sides above 4096 (`--image-max-side`) before decoding the pixels; smaller images are padded like the PR.
Token count: `tokens = ceil(H/40) * ceil(W/40)` (`[1, rows, cols]` is `image_grid_thw`, `PR:36-37`; 224x224 -> 36, 512x512 -> 169, 1024x1024 -> 676, 4096x4096 -> 10404).
`temporal_patch_size: 2` in the config is parsed but **unused** for images (grid t = 1) [IP:94].

## 3. Patches and position embeddings
* Patch = 20x20x3 (`native_patch_size`), `adapter_merge_size = 2` -> a 40x40 tile holds 2x2 patches [cfg, I2:1211-1217].
* A tile `[40][40][3]` is reshaped to `(2, 20, 2, 20, 3)` and transposed to `(sub_row, sub_col, y, x, c)`; patch vector index `(y*20 + x)*3 + c` (1200 values), sub-patches in
  order (0,0),(0,1),(1,0),(1,1) [I2:1229-1230]. Patches are laid out **token-major** (4 consecutive sub-patches per tile).
* `patch_embedding`: Linear 1200 -> 1024, **no bias** [I2:1217, ckpt `patch_embedding.weight [1024,1200]`].
* Position embedding: learned table `[32*32][1024]` (`posemb_grid_size = 32`) **resampled bilinearly** (`align_corners=False`, no antialiasing, border clamp) to the image's native
  patch grid `H_p = 2*rows`, `W_p = 2*cols`: `src = (i + 0.5) * 32 / size - 0.5`, taps `floor(src)` and `floor(src)+1` clamped to [0, 31], hat weights; separable in rows and cols
  [I2:1233-1243, VU:191-262]. For grids smaller than 32 it is plain downsampling (no antialiasing), for larger grids upsampling.
* Class token: `class_embedding + cls_position_embedding` (no table position), one per image, placed **first** in that image's sequence [I2:1247-1250].
* Then `pre_layernorm` (LayerNorm 1024, eps 1e-5, affine) on all rows including the class token [I2:1588].

## 4. ViT forward
Per layer (23) [I2:1181-1208]: `x = x + lambda_1 * Attn(LN_1(x))`; `x = x + lambda_2 * MLP(LN_2(x))`.
* LayerNorm: standard (mean/var, biased var), eps `1e-5` [cfg], affine weight + bias.
* Layer scale `lambda_1/2`: per-channel vectors (checkpoint `ls_1.gamma`, `ls_2.gamma`, init 0.1) [I2:1196-1198, CM].
* Attention: 16 heads x 64, **separate q/k/v/out Linear with bias**, **bidirectional**, softmax scale `1/sqrt(64)`, attention restricted to one image (cu_seqlens per image, the class
  token participates) [I2:1060-1160]. No windowing.
* **2-D axial RoPE** on q and k, full head dim, theta 10000 **[assumption: `rope_parameters` default `rope_theta=10000` of the PR config class; it is not in config.json]** [I2:983-1057]:
  `inv_freq_j = 10000^(-2j/32)`, `j = 0..15`; angle vectors for width (x) and height (y) positions are concatenated **width first**, each frequency repeated for an interleaved
  channel pair: channels `0..31` rotate with the width index, `32..63` with the height index, pair `(x1,x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)`. Positions are 1-indexed
  `(row+1, col+1)` in native-patch units; the class token is at `(0,0)` [I2:1594-1596 `position_ids + 1`].
* MLP: `fc1 1024->4096 (bias)`, **exact GELU** (`hidden_act = "gelu"`, erf), `fc2 4096->1024 (bias)` [I2:1166-1178].
* No final LayerNorm: the merger reads the raw residual stream of the last layer.

## 5. Adapter / projector
Patch states (class token dropped) are regrouped `[tokens, 4*1024]` (the 4 sub-patches of a tile in order), `linear_1 (4096x4096, no bias) -> GELU -> linear_2 (2304x4096, no bias)`
[I2:1253-1267, 1601-1603; checkpoint `proj_in`, `proj_out`, `projector_bias=false`]. Output 2304 = text hidden size, **one row per tile = one text token**.

## 6. Placing the tokens in the text sequence
* Tokenizer: `<|content_image|>` = id 200005 and the placeholder `<|unused_200054|>` = **200054** [tok]; `image_token_id = 200054` [I2 config default; cfg processor_config `image_token`].
* `chat_template.jinja` lines 84-88: parts of type `image`, `input_image` or `image_url` (user and tool messages only, otherwise an exception) emit
  `<rtok><|content_image|><|unused_200054|><|end_message|>` - **one** placeholder per image, the image data itself is not read by the template [tok].
  Text parts emit `<rtok><|content_text|>...<|end_message|>`.
* The processor expands each placeholder to `image_grid_thw.prod()` copies (`rows*cols`) [PR:36-37]:
  `<rtok><|content_image|>` + N x `<|unused_200054|>` + `<|end_message|>`.
* The OpenAI content part is `{"type":"image_url","image_url":{"url":...}}`; the template accepts it by its `type` only.

## 7. How the text model consumes the embeddings
[I2L:1684-1694 and I2:1735] `inputs_embeds = embed_tokens(input_ids)` (this includes `embed_norm`, `Inkling2NormedEmbedding` I2:1316-1322), then
`inputs_embeds.masked_scatter(input_ids == image_token_id, image_embeds)`: the vision rows **replace the post-`embed_norm` rows**, in order, and are NOT normalised again.
After that nothing is image-specific: `position_ids = arange` (consecutive; Inkling has no RoPE, only relative-distance bias on distances in tokens), the causal / sliding masks are
the usual ones (**no bidirectional attention among image tokens**), the conv states and attention-residual blocks see the injected rows exactly like token rows (block 0 = the embedding row)
[I2:1348-1445]. Consequence for the engines: an image row r at position p replaces the hidden state AND the block-0 entry of the attention-residual stack at p.

## 8. Sidecar container `inkling2-mini-vision.qv`
`tools/convert_inkling_vision.py MODEL_DIR OUT.qv [--quant fp16|int8]`. The text `.qv` contains no vision weights. Envelope identical to the text `.qv`
(`"QVOLTA1\0" | u64 json_len | u64 data_off | json | pad | data`, tensors 256-byte aligned), format tag `"qvolta-vision-v1"`; header `config` holds the tower hyper-parameters
(hidden, layers, heads, mlp ratio, eps, patch 20, merge 2, grid 32, projector dims, rope theta, image token id, min/max side). Tensor names are the checkpoint names (`model.visual.*`).
* `fp16` (default, **0.639 GB**): matrices and the position table fp16 (bf16 -> fp16 is exact for |w| in [6.1e-5, 65504]; measured: 0 overflows, 332 of 317,898,752 values flushed to zero), vectors (LN, biases, gamma, class tokens) fp32.
* `int8` (**0.323 GB**): big matrices int8 with one fp32 scale per row, expanded to fp16 at load. Measured error is 2-4 % relative L2 on the embeddings, the same order as Hugging Face's own bf16 run (1.3-5 %), so it is **not** the default.
* Lookup (`qv_server`, only for arch `inkling2_loop`): `--vision-file PATH`; else `<model>-vision.qv` (without `.qv`), else `<family>-vision.qv` (`inkling2-mini-q3a5.qv` -> `inkling2-mini-vision.qv`) next to the model; `--vision-file none` disables it. No vision file = text-only server (`/props` `vision:false`).
  The llama-server `--mmproj` flag is parsed and ignored (our sidecar is the vision file).

## 9. Implementation and validation
Code: `src/vision/` (`vision.h` API; `image.cpp` stb + libwebp decode + preprocessing + `image_url` sources; `vision_cpu.cpp`; `vision_cuda.cu`; `vision_file.cpp`), `third_party/stb/stb_image.h`, `third_party/libwebp/` (decode-only subset, BSD-3).
* **CUDA encoder** (V100): fp16 weights on the device (0.64 GB, resident), fp32 residual stream, cuBLAS tensor-op GEMMs (fp16 in, fp32 accumulate; out-proj/fc2 write fp32), LayerNorm / residual / layer-scale fused, bias+GELU kernel,
  bidirectional two-pass WMMA flash attention (head 64; pass 1 row max and sum, pass 2 P*V), RoPE tables on the device. fp16 overflow check (`QV_VIS_CHECK=1`): the largest |residual| after any layer is **49.3** (limit 65504) on the test images; the largest
  fp16 intermediate is the fc1 output (not separately instrumented, no non-finite values seen).
* **CPU encoder**: fp32 activations, fp16-stored weights packed in 16-column panels, AVX2/FMA 6x16 micro-kernel, OpenMP; correct first, attention is a simple 4-query-blocked dot-product kernel.
* Golden data (`tests/golden/inkling_vision/`, 4.7 MB): 6 images (`tools/hfparity/vision_imgs.py`: gradient 200x120, rendered "HELLO" 320x120, red square on blue 224x224, JPEG 240x160, RGBA 120x80, noise 33x47).
  `tools/hfparity/vision_ref.py` runs the **HF `Inkling2ImageProcessor` and `Inkling2VisionModel`** from PR #51050 (processor in a user-level process with the PR on `PYTHONPATH`; tower loaded with only the 0.64 GB vision weights, fp32 and bf16) and stores
  `pixel_values`, the embeddings after `pre_layernorm`, the last hidden state and the final embeddings (`embeds` fp32 = golden, `embeds_bf16` = HF default dtype).
* Results (`benchmarks/results/inkling_vision_20261123.json`, `python3 -I tools/vision_check.py build/qv_vision_run VISION.qv [--cuda]`), 6 images:
  * Pixels (C++ stb decode + our preprocessing vs the HF processor): max |diff| 1.2e-7 (one float32 ulp, order of the rescale/normalise operations) on all PNGs, i.e. bit-level equal up to rounding; the JPEG differs by at most one 8-bit level (0.0078; decoder difference, section 11).
  * Embeddings vs the fp32 HF tower (relative L2 per image, cosine >= 0.99999): **CPU 2e-6..1e-5** (PNG), **CUDA fp16 5.6e-4..4.1e-3**; JPEG 3.9e-3 on both (decoder). The HF **bf16** tower (its default dtype) differs from the HF fp32 tower by 1.3-5.0 %, so our CUDA path is 12-90x closer to fp32 than HF's own default run.
  * int8 sidecar: 1.7-4.0 % (cosine 0.9992-0.9999): not negligible, not the default.
  * End to end (`tools/inkling_vision_golden.py`, `tools/vision_e2e_check.py`, `qv_vision_e2e`): the fp32 reference text model with the HF embeddings injected answers the red-square prompt correctly
    ("a blue background with a red square centered"). The Q3A5 engines consuming the same embeddings through `prefill_embeds`: last-position top-1 equal, KL(ref||engine) 2.0e-10 (CUDA) / 2.4e-10 (CPU), top-20 max |dlogit| 3.38 / 3.33 (above the 2.1 text gate: the engines run 3-bit weights, the gate was set for the HF bf16 run),
    greedy ids equal to the reference for the first **9** (CUDA) / **31** (CPU) of 32 tokens. The CUDA divergence at step 9 is a logit-gap-1.14 flip (28.72 vs 27.58) and the continuation is a paraphrase with the same content
    ("...blue background with red square centered. Answer concise"). Text-only control on the same CUDA engine: 32/32 on three goldens, so the shorter agreement is image-prompt/quantisation noise, not a text-path regression; it was not investigated further.

## 10. Server (`/v1/chat/completions`)
* Content parts: `image_url` (`{"url": ...}` or a bare string), `input_image`, `image`. Placeholders are expanded as in section 6; `usage.prompt_tokens` counts the expanded rows.
* Sources: `data:image/...;base64,` URLs and files below `--image-root DIR` (default: the working directory; `""` disables; plain paths and `file://`, realpath-checked, `..` and symlinks leaving the root are refused) are always on. **http(s) is off** (`400 ... --allow-remote-images`);
  with `--allow-remote-images` the server runs `curl` (no TLS library is linked) with: scheme/credential checks, resolution of the host and **refusal of every private, loopback, link-local, CGNAT, multicast, ULA address** (the connection is pinned to the checked address with `--resolve`, so DNS rebinding cannot swap it),
  no redirects, `--max-filesize` 32 MiB, 20 s limit. Tested: 127.0.0.1, 169.254.169.254, `[::1]`, `user:pw@` URLs are refused; a public image URL fetches and decodes.
* Limits: encoded size 32 MiB, each side <= 4096 px (`--image-max-side`) checked from the header before pixel allocation; the decode/preprocess happens before the response starts, so a bad image is a clean `400 {"error":{"message":"image: ..."}}`.
  A server without vision answers `500 image input is not supported - hint: if this is unexpected, you may need to provide the mmproj` (llama-server wording).
* Formats: PNG, JPEG (baseline/progressive), GIF (first frame), BMP, TGA, PSD composite, PNM, PIC via stb_image; **WebP** (lossy VP8, lossless VP8L, alpha, animated: first frame composited on the canvas) via the vendored libwebp 1.5.0 decode-only subset (BSD-3). WebP output is bit-identical to Pillow/libwebp (`tests/test_image_formats.py`). 16-bit PNG is reduced to 8 bit (stb, `>>8`). **Not supported: HEIC/HEIF, AVIF (patents, size), TIFF (no small permissive decoder), JPEG XL, JPEG 2000, ICO, SVG, PDF**; they are recognised from the magic bytes and rejected with `400 image: unsupported image format 'HEIC' (supported: PNG, JPEG, GIF, BMP, WebP, PNM, TGA); convert the image first`.
* Other image errors (all `400 invalid_request_error`, message prefix `image:`): non-image payloads such as an HTML/JSON error page give `image data is not an image (starts with '<html>...')`; unknown binary data `unrecognised or corrupt image data (starts with bytes 00 01 ...)`; bad base64 `invalid base64 in image data URL (unexpected byte 0x40 at offset N)`, `... (truncated: N characters is not a valid length)` or `image data URL has an empty base64 payload`; truncated/corrupt WebP `WebP image is truncated` / `cannot decode WebP image (corrupt or truncated)`.
* The prompt cache: image rows appear in the cached token list as content-hashed negative pseudo ids, so an identical image in a follow-up turn keeps its prefix cache and a different image can never reuse stale KV. Slot save/restore refuses a slot that holds images.
* `/props`: `modalities.vision = true` and `/v1/models` lists `image` in `input_modalities` when a vision encoder is loaded.
* Encoding runs inside the engine lease (the GPU is shared), counted in `timings.prompt_ms`.

## 11. Limits / known differences
* JPEG decoding: stb_image and PIL/libjpeg-turbo differ by at most one 8-bit level per pixel (max |pixel diff| 0.0078 = 1/127.5 on the test JPEG); embedding effect 0.4 % relative L2 (below the bf16 noise of 1.3-5 %).
* Large images: CUDA attention is quadratic (2048x2048 = 2704 tokens = 10816 patches: 2.1 s); 4096x4096 (10404 tokens) needs about 1.1 GB of transient device memory and was not benchmarked. The vision weights (0.64 GB) sit next to the 9.9 GiB text weights on the 16 GB card.
* One text-model difference is outside vision: nothing here changes the text path; Q3A5 is a 3-bit quantisation, so image answers carry that degradation (section 9 / PERFORMANCE.md).
* `temporal_patch_size`, video and multi-frame input are not implemented (the PR has no video path for this model either).
* `qv_ink` / `qv_run` have no image option; image prompts go through `qv_server` (and `qv_vision_e2e` for validation).
