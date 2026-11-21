# Inkling-2-Mini (`inkling2_loop`) - exact text-only inference spec

Status: H-I1 (Gen 7). Written 2026-11-21 from (a) the checkpoint `thinkingmachines/Inkling-2-Mini` @ `ed40e7da`
(config, `hf_quant_config.json`, safetensors headers, tokenizer, chat template), (b) the Hugging Face modeling code of
the open PR huggingface/transformers#51050 "Add Inkling2 Mini" @ `a10ebc3c` (`models/inkling2_loop/modeling_inkling2_loop.py`,
`conversion_mapping.py`, `core_model_loading.py`), (c) our plain-torch reimplementation `tools/inkling_ref.py`.

IMPORTANT facts about the reference situation (read first):

* The installed transformers 5.17 has only the OLD `inkling` model (the first Inkling); PyPI 5.23.0 (newest) has `inkling2`
  (= "Inkling 2 Small", NO loop, no `controllers`, no `middle_cycle_*`). `inkling2_loop` exists only in the open PR #51050.
  Nothing in the installed environment can run this checkpoint.
* The PR code was READ, but not executed: running the downloaded unmerged PR tree was refused by the permission system of
  this session. Therefore **no numeric comparison against HF-executed logits exists** (see "Validation status").
  Everything below is transcribed from the code text; every claim is tagged `[HF]` (read in PR code), `[CKPT]` (read from
  checkpoint/config files), `[README]`, or `[ours]` (our implementation / measurement).
* The MTP head has NO reference code in HF (the PR ignores `model.mtp.*`); only tensor names/shapes and config are known.

## 1. Hyper-parameters `[CKPT config.json]`

| name | value |
|---|---|
| hidden_size H | 2304 |
| layers (weights) | 24; layers 6..17 (`core_start=6`, `core_end=18`) run `num_passes=3` times -> 48 layer applications |
| vocab (padded / real) | 201024 / 200058 (`unpadded_vocab_size`; logits sliced to the first 200058 rows) |
| heads | 24 q, 8 kv, head_dim 128 (GQA group 3, kv head = q head // 3), same for local and global layers |
| global (full-attention) layers | 5, 11, 17, 23 (`layer_types[i] = "hybrid" if (i+1)%6==0`; `local_layer_ids` is the complement) |
| local layers | the other 20: sliding window 1024 |
| d_rel / rel_extent | 16 / 1024 (both layer kinds; local: rel_extent = window = 1024) |
| qk_l2_clip_threshold | 22.627416997969522 (= 2*sqrt(128)) |
| rel_l2_clip_threshold | 8.0 |
| log_scaling_n_floor / alpha | 8192 / 0.1 (global layers only) |
| rms_norm_eps | 1e-6 (all norms, also the unweighted ones) |
| sconv_kernel_size | 4 |
| attention residual block size `n_layers_per_attn_res_block` | 6 |
| experts | 128 routed, top-8, 0 shared; expert FF 1152; `route_scale` 8.0; sigmoid scoring; `use_gate_bias` |
| mlp_gate_softcap / mlp_up_softcap | 4.0 / 25.0 |
| final_logit_softcapping | 30.0 |
| loop `gate_rank` | 64; `is_varloop` false (the separate research checkpoint `TML-d24-varloop-research` allows other pass counts) |
| max position | 1,048,576 |
| eos | 200006 (`<|content_model_end_sampling|>`); pad 199999 (generation_config) / `<|endoftext|>` |

No RoPE and no learned absolute position embedding anywhere in the text model: positional information comes only from
(i) the causal/sliding mask, (ii) the relative-logit bias (section 5), (iii) the short convolutions (causal, so they see
order). Log scaling (global layers) uses the absolute position index.

## 2. Notation and dtype flow

`rms_w(x,w) = w * x * rsqrt(mean(x^2, -1) + 1e-6)` (fp32 internally, result cast back to the module dtype).
`rms_u(x) = x * rsqrt(mean(x^2, -1) + 1e-6)` (no gamma, last dim).
`softcap(x,c) = c * tanh(x/c)`.
HF runs the model in bf16 with these fp32 islands `[HF _keep_in_fp32_modules_strict + code]`: router (weight, logits, sigmoid,
bias, top-k, weights), all attention-residual / message-attention queries and softmaxes, both message-gate output
projections (`forget_gate_proj`, `message_gate_proj`), all short convolutions (static `*_sconv.conv1d` and the post
attn/mlp convolutions; dynamic conv computed in fp32), l2-clips and tau scaling (computed in fp32, cast back to bf16),
expert activation (fp32), expert-output accumulation (fp32), sigmoid gate of the attention output, the residual stream
inside a layer (`residual = hidden.float()`), final logits (`.float()`, softcap in fp32). Matmuls with bf16 weights
(q/k/v/r/o/gate/sink projections, expert matmuls, message-gate `down_proj`, lm_head) are bf16 x bf16.
Our reference (`tools/inkling_ref.py`) computes everything in fp32 from the bf16 weights (optionally rounding to bf16 at
module outputs, `emu_bf16`).

## 3. Embedding, stack, head

```
x   = embed[ids]                      # [n, 2304], embed.weight [201024, 2304] bf16
h   = rms_w(x, embed_norm.weight)     # "NormedEmbedding": the embedding is RMS-normalised with gamma
blocks = []                           # list of residual "block states" [n,2304] each (attention-residual memory)
pre  : for L in 0..5   : h, blocks = layer(L, pass 0)
pass0: for L in 6..17  : h, blocks = layer(L, pass 0)
for p in 1,2:                         # extra passes restart from the PRE-loop state, not from the previous pass output
    prev_blocks = blocks_after_previous_pass[npre+1:]   # see 8
    prev_h      = h_after_previous_pass
    h, blocks   = pre_loop_h, pre_loop_blocks
    for L in 6..17: msgs = message_attn(prev_blocks, prev_h, query=loop_idx); h, blocks = layer(L, pass p, messages=msgs)
post : for L in 18..23 : h, blocks = layer(L, pass 0)
hf   = final_input(blocks, h)         # block attention residual with a single query = model.llm.attn_res.output_query
hf   = rms_w(hf, model.llm.norm.weight)
logits = softcap(hf @ unembed[:200058].T (fp32), 30.0)      # NOT tied to embed; no division by any mup multiplier
```
`blocks` update rule (model level, after each layer application): `if L % 6 == 0: blocks.append(layer_input_h)` (the hidden
state that ENTERED the layer, before the layer ran). Block boundaries in 0..23 are layers 0, 6, 12, 18.
The prefill and each decode step run the full 48 applications for every token (no early exit; `is_varloop=false`).

## 4. Block attention residual ("attention residual")

`BlockAttentionResidual(queries Q x H)` given `blocks [n,B,H]` (B>=0) and the current partial state `p [n,H]`:
```
S  = rms_u(concat(blocks, p[:,None,:], dim=1).float())          # [n, B+1, H]  (no gamma)
a_q = softmax_over_B+1( (S @ query_q) * H**-0.5 )                # per token, per query (H = 2304)
out_q = sum_b a_q[b] * S[b]                                      # value = the SAME normalised states
```
It returns one `[n,H]` vector per query. Queries are fp32 parameters, no bias/projection. Which parameter feeds which input
(`attn_res.*` indices):

* attention input of layer L (L >= 1): `model.llm.attn_res.attn_queries.{L-1}` (index is shifted by one: 23 tensors for
  layers 1..23). Local layer: shape `[2304]` (1 query). Global layer (5,11,17,23): shape `[3,2304]` = (query, key, value)
  queries -> three different mixes feed the q, k and v paths. Layer 0 has no attention input mixing (h = embedding).
* MLP input of layer L: `model.llm.attn_res.mlp_queries.{L}` `[2304]` (24 tensors).
* output: `model.llm.attn_res.output_query` `[2304]`.

Layer-internal residual rule (this replaces a classic residual stream): with `res = h_in` (layer input),
```
a = attn_branch(...)                         # after post-attention conv (6)
if L % 6 == 0:  blocks_l = blocks + [h_in];  residual = a                 # boundary: residual stream RESETS to the attention output
else:           residual = a + h_in.float()                               # normal residual add
m_in = BlockAttnRes(blocks_l, residual, mlp_queries[L])[0]               # MLP input = learned mix of block states + partial
y = mlp_branch(...)
h_out = (residual + y).to(bf16)
```
Note at a boundary layer the previous residual stream lives only inside `blocks` (as `h_in`) and is reachable via the
attention-residual softmax; the hidden stream continues from the attention output.

## 5. Attention (both kinds)

Common (per layer L, input streams `hq` (query path), `hk`, `hv`; all `[n,2304]` bf16 after their input norm, see 7):
```
q  = hq @ wq_du.T                          # [n,3072] -> [n,24,128]
kp = hk @ wk_dv.T ; vp = hv @ wv_dv.T      # [n,1024]
r  = hq @ wr_du.T  -> [n,24,16]            # relative states (wr_du [384,2304])
gate = hq @ gate_attn.T                    # [n,3072]
sink = hq @ attn_sink_proj.T               # [n,24]   one extra softmax logit per head, value 0
q  = l2clip( q * q_l2_clip.scale[24,128] , 22.627 )            # per head vector, fp32: x * min(1, c/(||x||+1e-6))
k  = l2clip( sconv_k(kp) , 22.627 )                             # no learned scale, no k-norm gamma
v  = rms_u( sconv_v(vp) per head vector of 128 )                # "value norm without gamma"
rel= l2clip( r * rel_l2_clip.scale[24,16] , 8.0 )
local layer : nothing else.
global layer: tau = 1 + 0.1 * ln( max((pos+1)/8192, 1) )   (pos = absolute 0-based position of the QUERY token)
              q = q * tau ;  rel = rel * tau              # tau == 1 for pos+1 <= 8192
rl[h,i,d]  = rel[i,h,:] @ rel_logits_proj.proj  [16,1024]   # bias profile per token/head, d = backward distance 0..1023
bias[h,i,j]= rl[h,i,i-j] if 0 <= i-j < 1024 else 0
scores[h,i,j] = (q[i,h].k[j,h//3]) * 128**-0.5 + bias[h,i,j]
mask: j <= i (global) ; i-1024 < j <= i  i.e. 0 <= i-j <= 1023 (local)
p = softmax over [scores_masked..., sink[i,h]] ; drop the sink column ; o[i,h] = sum_j p[j] v[j,h//3]
o = (o.reshape(n,3072) * sigmoid(gate)) @ wo_ud.T          # sigmoid output gate, no bias
```
Differences local vs global:

| | local (20 layers) | global (4 layers: 5,11,17,23) |
|---|---|---|
| key/value inputs | same stream `hk=hv=hq` (one `attn_norm`) | three separate mixed streams (3 attn-res queries) + three norms `q/k/v_input_norm` |
| k/v short conv | static depthwise conv (k=4), weights `k_sconv.weight[1024,1,4]`, scale 1/sqrt(5) | DYNAMIC conv: taps = static `[4,1024]` + per-token `weight_proj[32,2304] @ hk` shared within a kv head (4 taps x 8 heads), scale 1/sqrt(9)=1/3 |
| mask | sliding window 1024 | full causal |
| rel bias | distance < 1024 | distance < 1024 (so beyond 1024 tokens only the content term remains) |
| q / rel scaling | none | log scaling tau (above), fp32, applied after the clip |
| KV cache | last 1024 tokens suffice (per pass slot) | all tokens |
Sink logit, o-gate, l2 clips, value norm, `1/sqrt(128)` scale are identical. Weights shapes are identical in both kinds.

Important consequences: softmax scale is `1/sqrt(128)` (NOT 1/128 as in the old `inkling`); q is bounded by the clip (|q|<=22.6),
k by the clip, so |score content| <= 22.6*22.6/11.3 = 45.25.

## 6. Short convolutions (`sconv`) and their state

All depthwise, causal, kernel 4. Four per layer: `k_sconv`, `v_sconv` (inside attention), `attn_sconv`, `mlp_sconv` (on the
output of the attention / MoE branch, `[n,2304]`, weight `[2304,1,4]`).
Static: `y[t] = sum_{j=0..3} w[c,0,j] * x[t-3+j]` (so `w[...,3]` multiplies the CURRENT token, `w[...,0]` the token 3 back;
this is `F.conv1d` with left padding 3, as in HF), then `out = (y + x) / sqrt(5)`.
Dynamic (global k/v): per kv-head g (8) and tap lag l (0 = current token):
`taps[t,g,c,l] = static_w[l, g*128+c] + (weight_proj @ hk[t])[g*4 + l]`; `y[t,c] = sum_l taps * x[t-l,c]`;
`out = (y + x)/sqrt(2*4+1) = (y+x)/3`. `x` = the projected (pre-conv) k or v; `hk`/`hv` = the normalised k/v input stream.
State: the last 3 PRE-conv inputs per (layer application, conv) - zeros before the sequence start. Computation fp32.
Per pass: the conv state, the KV cache of attention layers and the post-attn/post-MLP conv state are SEPARATE for each of the
3 passes of a loop layer `[HF: cache_offset appends extra cache layers per pass]`. Slots per token: 24 + 12*2 = 48 layer
applications, each with its own 4 conv states. Counted: local layers 0-4, 18-22 (10) run once, local layers 6-10, 12-16 (10) run
3 times -> 40 local attention slots (window 1024 => constant 40 x 1024 x 2 x 1024 x 2 B = 160 MiB fp16); global layers 5 and 23
run once, 11 and 17 run 3 times -> 8 global KV slots = 8 x 2 x 1024 x 2 B = 32 KiB per token (fp16): 8 GiB @256K, 32 GiB @1M.

## 7. Layer-input norms and the loop messages

Non-loop layers (and pass 0 of loop layers): local: `hq=hk=hv = rms_w(attn_in, attn_norm.weight)`;
global: `hq = rms_w(q_in, q_input_norm.weight)`, likewise k, v (`q_in,k_in,v_in` = the three attention-residual outputs).
Passes 1 and 2 of loop layers receive `msgs [n,4,H]` = (query, key, value, mlp) messages and a gate per stream:
```
gate(h, msg) = (1 - tanh(F(g))) * h + tanh(G(g)) * msg ,  g = down( concat(rms_u(h), rms_u(msg)) ) [64]
  down = controllers.<s>.down_weight [64,4608] (bf16 matmul), F = forget_weight [2304,64], G = gain_weight [2304,64]
streams: s in {attn -> q path, k -> key path, v -> value path, mlp -> mlp path}   (controllers.attn/k/v/mlp)
local loop layer: hq = attn_norm(gate_q(h_in, msg_q)), hk = attn_norm(gate_k(h_in, msg_k)), hv = attn_norm(gate_v(h_in, msg_v))
global loop layer: hq = q_input_norm(gate_q(q_in, msg_q)) ... (gates applied to the three attention-residual outputs, then norm)
mlp: m_in = gate_mlp(m_in, msg_mlp)  (applied after the MLP attention-residual mix, before post_attention_layernorm)
```
Messages (computed per loop layer index li = L-6, once per pass p>=1, from the PREVIOUS pass):
`msgs = BlockAttnRes_{4 queries}(prev_blocks, prev_h)` with queries rows `(query, key, value, mlp)` taken from the two checkpoint
tensors `middle_cycle_cross_pass_queries [24,2304]` = per loop layer `[q, mlp]` interleaved (`reshape(12,2,2304)`) and
`middle_cycle_cross_pass_kv_queries [24,2304]` = per loop layer `[k, v]`; merged order `[query,key,value,mlp]`
(`Inkling2LoopMergeMessageQueries`). `prev_blocks` = the block states appended during the previous pass AFTER the first one,
i.e. only the input state of layer 12 of that pass (layers 6 and 12 are boundaries; the one appended by layer 6 is skipped:
`block_states[..., npre+1:, :]`, npre = 1 = the embedding block); `prev_h` = output of layer 17 of the previous pass.
Pass 1 uses pass 0, pass 2 uses pass 1. The messages do not depend on the loop layer's own previous activations.

## 8. MoE (all 24 layers; no dense layers, no shared experts)

```
x  = rms_w(m_in, mlp_norm.weight)                                  # = post_attention_layernorm
lg = x.float() @ mlp.gate.weight.T            # [n,128] (gate.weight [128,2304] bf16 stored, fp32 compute)
s  = sigmoid(lg)
idx= topk(s + mlp.gate.bias[128] (fp32), 8)   # bias only for SELECTION
w  = s.gather(idx) ; w = w / (sum(w) + 1e-20) * 8.0              # route_scale, weights from the un-biased scores
expert(e): gu = x @ W13[e].T  (W13 [2304,2304]: rows 2i = gate_i, rows 2i+1 = up_i  <- INTERLEAVED in the checkpoint)
           g,u = gu[:, 0::2], gu[:, 1::2]  (== HF de-interleave then chunk(2))
           a = softcap(g,4) * sigmoid(g) * softcap(u,25)           # NOTE: sigmoid sees the raw gate, only the linear factor is capped
           y = a @ W2[e].T  (W2 [2304,1152])
out = sum_k w_k * y_{idx_k}  (accumulated in fp32)  ->  post_mlp_convolution (static conv, 1/sqrt(5))
```
The `global_scale` of the old `inkling` model does not exist here. The router runs fp32 on bf16-rounded input in HF.
Active expert weights per token per layer application: 8 * (2304*2304 + 1152*2304) = 63.7 M params.

## 9. Per-layer application op list (local layer, pass 0)

```
[attn_in = attnres(blocks, h; attn_queries[L-1])]   (L>=1)
x   = rms_w(attn_in, attn_norm)
a   = attention_local(x)                         # 5
a   = sconv(a; attn_sconv)                       # 6, (conv+a)/sqrt5
boundary? blocks+=[h]; res=a  else res=a+h
m   = attnres(blocks, res; mlp_queries[L])[0]
x2  = rms_w(m, mlp_norm); y = moe(x2); y = sconv(y; mlp_sconv)
h'  = res + y
```
Global layer: three attention-residual outputs, three norms, attention with separate key/value streams, otherwise identical.
Loop layers pass>=1: message gates as in 7.

## 10. Tokenizer, chat template, generation `[CKPT]`

* `tokenizer.json` (HF fast, 200058 tokens incl. 60 added) + `tiktoken/tokenizer.model` (base64 BPE ranks, tiktoken format).
  transformers tokenizer class `PreTrainedTokenizerFast`, no BOS added, `model_max_length` 1048576.
* Special ids: `<|message_user|>` 200000, `<|message_system|>` 200002, `<|content_text|>` 200004, `<|end_message|>` 200010,
  `<|content_model_end_sampling|>` 200006 (EOS), `<|endoftext|>` 199999 (pad), image token `<|unused_200054|>` 200054.
  (ids of `<|message_model|>`, `<|content_thinking|>`: see section 12.)
* Chat template (`chat_template.jinja`), text-only: `<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>`
  (default `reasoning_effort` 0.9 injected before the first non-system message; map none 0, minimal .1, low .2, medium .7,
  high .9, xhigh/max .99), then per message `<|message_{user|model|system|tool}|><|content_text|>{text}<|end_message|>`;
  assistant turns end with `<|content_model_end_sampling|>`. `add_generation_prompt` adds NOTHING: the model itself emits
  `<|message_model|>` then typically `<|content_thinking|>...<|end_message|><|message_model|><|content_text|>answer<|end_message|><|content_model_end_sampling|>`.
  Tools: `<|message_system|>tool_declare<|content_xml|>{sorted compact json}<|end_message|>`; calls
  `<|message_model|>{name}<|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>`.
* `generation_config.json`: `do_sample=true, temperature=1.0, eos_token_id=200006, pad_token_id=199999` (no top-k/top-p).

## 11. Checkpoint format `[CKPT]`

One file `model-00001-of-00001.safetensors` (19,526,019,168 B, 1244 tensors; offset of data 148,416) + `mtp.safetensors`
(635,801,416 B, 54 tensors) + index json. Names (L = layer 0..23 unless stated), shapes, dtypes:

| group | tensor | shape | dtype |
|---|---|---|---|
| embed | `model.llm.embed.weight` | [201024,2304] | BF16 |
| | `model.llm.embed_norm.weight` | [2304] | BF16 |
| unembed | `model.llm.unembed.weight` (rows >= 200058 are padding, dropped) | [201024,2304] | BF16 (fwd output fp32) |
| final norm | `model.llm.norm.weight` | [2304] | BF16 |
| attn-res queries | `model.llm.attn_res.attn_queries.{0..22}` (-> layer idx+1); local `[2304]`, idx 4,10,16,22 (layers 5,11,17,23) `[3,2304]` | | F32 |
| | `model.llm.attn_res.mlp_queries.{0..23}`, `attn_res.output_query` | [2304] | F32 |
| loop messages | `model.llm.middle_cycle_cross_pass_queries`, `..._kv_queries` | [24,2304] | F32 |
| attention (all L) | `layers.L.attn.wq_du` | [3072,2304] | BF16 |
| | `wk_dv`, `wv_dv` | [1024,2304] | BF16 |
| | `wr_du` | [384,2304] | BF16 |
| | `wo_ud` | [2304,3072] | BF16 |
| | `gate_attn` | [3072,2304] | BF16 |
| | `attn_sink_proj.weight` | [24,2304] | BF16 |
| | `q_l2_clip.scale` | [24,128] | BF16 |
| | `rel_l2_clip.scale` | [24,16] | BF16 |
| | `rel_logits_proj.proj` | [16,1024] | BF16 |
| local-layer convs (20 layers) | `attn.k_sconv.weight`, `attn.v_sconv.weight` | [1024,1,4] | BF16 |
| global-layer convs (5,11,17,23) | `attn.k_sconv.static_w`, `attn.v_sconv.static_w` | [4,1024] | BF16 |
| | `attn.k_sconv.weight_proj`, `attn.v_sconv.weight_proj` | [32,2304] | BF16 |
| norms | local: `attn_norm.weight`; global: `q_input_norm`, `k_input_norm`, `v_input_norm`; all: `mlp_norm.weight` | [2304] | BF16 |
| post convs | `attn_sconv.weight`, `mlp_sconv.weight` | [2304,1,4] | BF16 |
| router | `mlp.gate.weight` | [128,2304] | BF16 |
| | `mlp.gate.bias` | [128] | F32 |
| loop gates (L=6..17) | `controllers.{attn,k,v,mlp}.down_weight` | [64,4608] | BF16 |
| | `controllers.*.forget_weight`, `gain_weight` | [2304,64] | BF16 |
| experts L=0..19 (NVFP4) | `mlp.experts.w13_weight` | [128,2304,1152] | U8 (2 x e2m1 per byte) |
| | `w13_weight_scale` | [128,2304,144] | F8_E4M3 |
| | `w2_weight` | [128,2304,576] | U8 |
| | `w2_weight_scale` | [128,2304,72] | F8_E4M3 |
| | `w13_weight_amax`, `w2_weight_amax` | [128] | F32 |
| | `w13_weight_outer_scale_denominator`, `w2_...` | [1] | F32 |
| experts L=20..23 (MXFP8) | `w13_weight` | [128,2304,2304] | F8_E4M3 |
| | `w13_weight_scale` | [128,2304,72] | U8 (e8m0) |
| | `w2_weight` | [128,2304,1152] | F8_E4M3 |
| | `w2_weight_scale` | [128,2304,36] | U8 (e8m0) |
| vision (ignored) | `model.visual.*` (23-layer ViT, proj_in/out) | | BF16 |

`w13_weight` rows: 2304 = 2 x 1152 = gate/up INTERLEAVED (even rows gate, odd rows up); columns = hidden (K). `w2_weight`
rows = hidden (2304), columns = expert FF (1152). All matmuls are `x @ W.T` (nn.Linear layout [out,in]).

Dequantisation `[HF Inkling2LoopDequantizeExperts]` (K = reduction dim, blocks along K):
* NVFP4: byte `b` holds elements `2i` (low nibble `b & 15`) and `2i+1` (high nibble `b >> 4`). Nibble code -> value table
  `[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]` (sign bit = 8). One F8_E4M3 scale per 16 consecutive elements along K
  and one fp32 factor per expert: `W[e,r,k] = e2m1(code) * float(scale[e,r,k//16]) * (amax[e] / outer_scale_denominator)`.
  (The `hf_quant_config` calls this `weight_scale_format: amax_denominator`, `weight_transform: nvfp4_nearest`.)
* MXFP8: `W[e,r,k] = float(e4m3[e,r,k]) * 2^(scale_u8[e,r,k//32] - 127)`.
* Then for `w13` the row de-interleave `W = cat(W[0::2], W[1::2])` (HF `Interleave(dim=1)`), giving `[gate(1152); up(1152)]`.
* The engine-side native format also quantises ACTIVATIONS for expert matmuls (`required_activation_quantization`:
  NVFP4 layers: per-token amax, block 16, e2m1 + e4m3 scale, rule `fouroversix_mae`; MXFP8 layers: block 32 e4m3 + ue8m0).
  HF (and our reference) is weight-only dequant with bf16/fp32 activations; this activation quantisation is not specified
  further in the files we have (open question for a bit-exact match of the vendor runtime).
* Layers 0..19 experts NVFP4, layers 20..23 experts MXFP8 (last 4 layers kept at higher precision). Everything else bf16/fp32:
  `hf_quant_config` lists the fp32 group (attn_res queries, gate.bias, middle_cycle_*), bf16 group (all other text + vision) and
  the fp32-output group (`unembed`).
Name mapping to HF module names (for the code): `wq_du->q_proj.linear, wk_dv->k_proj.linear, wv_dv->v_proj.linear,
wr_du->relative_position_bias.linear, rel_logits_proj->relative_position_bias.proj, q_l2_clip->q_proj.norm,
rel_l2_clip->relative_position_bias.norm, wo_ud->o_proj.linear, gate_attn->o_proj.gate_proj, attn_sink_proj->sink_proj,
attn_sconv->post_attention_convolution, mlp_sconv->post_mlp_convolution, controllers.{attn,k,v,mlp}->{q,k,v,mlp}_message_gate
(down_weight->down_proj, gain_weight->message_gate_proj, forget_weight->forget_gate_proj)`.


## 12. Special token ids `[ours, tokenizer]`

`<|endoftext|>` 199999 (pad), `<|message_user|>` 200000, `<|message_model|>` 200001, `<|message_system|>` 200002,
`<|message_tool|>` 200003, `<|content_text|>` 200004, `<|content_image|>` 200005, `<|content_model_end_sampling|>` 200006 (EOS),
`<|content_thinking|>` 200008, `<|end_message|>` 200010, `<|content_tool_error|>` 200022, `<|content_xml|>` 200024,
`<|begin_of_text|>` 200028, `<|content_invoke_tool_json|>` 200049, `<|content_invoke_tool_text|>` 200057.
BPE: byte-level, 199998 base ranks + specials, o200k-style split regex (see `tokenizer.json` `pre_tokenizer`), no BOS.

## 13. Sizes and per-token cost `[ours, from the safetensors headers]`

Stored bytes (file 18.185 GiB incl. 0.59 GiB vision): experts NVFP4 (layers 0-19) 10.68 GiB, experts MXFP8 (layers 20-23)
3.92 GiB, embed + unembed (bf16) 1.73 GiB, attention/norm/conv/attn-res 1.22 GiB, router 0.013, loop controllers 0.04.
Text-only resident: ~17.6 GiB as stored (bf16 dense). Per layer application dense bf16 weights: 76.0-80.5 MiB
(q 3072x2304, k/v 1024x2304, r 384x2304, o 2304x3072, gate 3072x2304, ...). One token touches, per layer application:
8 experts = 34.2 MiB (NVFP4 incl. block scales, 4.5 bit/weight) or 62.6 MiB (MXFP8, 8.25 bit/weight).
Bytes read per decoded token (batch 1, all 48 applications, nothing cached across passes), AS STORED:
dense 3.70 GiB (bf16) + experts 1.71 GiB (44 NVFP4 applications x 34.2 MiB = 1.47 GiB, 4 MXFP8 x 62.6 MiB = 0.24 GiB)
= 5.41 GiB (+0.86 GiB if the full unembed matrix is streamed; ~0.4 GiB with a row-subset/sampled head) ~ 6.3 GiB.
=> at stored precision the HBM-bound ceiling on a 890 GB/s V100 is ~130 tok/s (6.3 GiB / 890 GB/s = 7.6 ms); the dense bf16 attention/projection weights
(not the experts) are 68 % of the per-token bytes, so 4-bit dense weights are required to approach the ~250 tok/s in the plan.
(Looped layers are read three times per token; experts selected may differ per pass.)
Parameter count check: experts 3.06 B active per token (48 x 63.7 M) + dense 1.8 B + unembed 0.46 B = 5.3 B (README: 5.4 B).

## 14. MTP head (`mtp.safetensors`, 54 tensors, all BF16) - what is and is not known

`[CKPT mtp_config]`: hidden 2304, 3 layers, 24 q / 8 kv heads, head_dim 128, dense SwiGLU intermediate 9216, `rope_theta` 1e6,
`local_layer_ids [0,2]` (layers 0 and 2 sliding window 1024, layer 1 global), `block_size 8`, `markov_rank 256`,
`markov_head_type "rnn_hidden"`, `target_layer_ids [5,23,41,48]`.
Tensors: `fuse_weight [4,4]`; `mask_embed [2304]`; `norm.weight`; per layer i in 0..2: `input_layernorm`, `post_attention_layernorm`,
`dflare_norm`, `self_attn.{q_proj [3072,2304], k_proj,v_proj [1024,2304], o_proj [2304,3072], q_norm, k_norm [128]}` plus a SECOND
set for context: `k_proj_ctx, v_proj_ctx [1024,2304], k_norm_ctx [128]`, `mlp.{gate,up}_proj [9216,2304], down_proj [2304,9216]`;
`markov_head.{markov_w1 [201024,256], rnn_proj [768,5120] (+bias 768), hidden_out [2304,256] (+bias 2304), dflare_norm [2304]}`.
Inferences (NOT confirmed by any code we could read; HF PR ignores `model.mtp.*`, no vLLM/SGLang source was reachable):
* "dflare"/`mask_embed`/`block_size 8` indicate a block-parallel drafter: 8 masked positions (all `mask_embed`) are drafted in one
  forward, attending (via the `*_ctx` K/V projections, which look like cross-attention to projected target hidden states) to the
  target model's hidden states; `fuse_weight [4,4]` plausibly mixes the 4 target hidden states from `target_layer_ids`.
* `target_layer_ids [5,23,41,48]` read as 0-based indices into the 48 unrolled layer applications: 5 = layer 5 (pass 0),
  23 = layer 11 (pass 1), 41 = layer 17 (pass 2) - all three are GLOBAL layers - and 48 = the final (post-last-layer) state.
  This reading is consistent (three global layers + the end) but is an inference.
* `markov_head` with rank 256 and `rnn_proj [768, 5120]` (768 = 3 x 256, i.e. a GRU-style 3-gate recurrence; 5120 = 2 x 2304 + 2 x 256
  or similar) conditions each drafted position on the previously drafted token (`markov_w1` = a 201024 x 256 token embedding, the
  output logits likely via the main unembedding after `hidden_out`).
The exact graph (RoPE layout, how fuse_weight/rnn_proj consume inputs, acceptance rule) is an OPEN QUESTION; engine work on H-I4
must obtain reference code (vLLM/SGLang `inkling2_loop` + MTP, or the Thinking Machines tinker-cookbook) before implementing it.

## 15. Validation status (honest)

H-I1 as stated ("reproduce HF logits, KL < 1e-3") is **NOT verified**: HF cannot be executed here (see top). What exists instead:

1. Own implementation `tools/inkling_ref.py` (fp32 compute on the dequantised checkpoint, CPU) transcribed op by op from the HF PR text.
2. Behavioural evidence that the transcription is right (all in `tests/golden/inkling`, `benchmarks/results/inkling_ablation_20261121.json`):
   * greedy generations are fluent and on-task for all 4 prompts, in the shape documented in the HF doc
     (`<|message_model|><|content_thinking|>The user asks ...`), including a Chinese/French prompt (the model quotes it exactly) and a
     4096-token prompt where it identifies the question about "each token is routed to 8" (long-context retrieval through the
     4 global layers works);
   * next-token NLL on 511 plain-text tokens: reference 1.756 nats/token; every single-op ablation is worse: static-conv tap order
     reversed 6.57, gate/up swapped 3.42, no output sigmoid gate 12.82, no attention sink 2.39, no relative bias 5.50,
     softmax scale 1/128 instead of 1/sqrt(128) 2.89, no residual reset at block boundaries 8.08, loop messages removed (== a
     single pass) 2.36; dynamic k/v conv taps removed 1.780 (only +0.02 at 511 tokens; weak sensitivity, so the dynamic-conv transcription is the least tightly validated op);
   * dequantisation sanity: NVFP4 (layers 0-19) and MXFP8 (layers 20-23) experts dequantise to the same std (w13 0.0208 = 1/sqrt(2304),
     w2 0.0294 = 1/sqrt(1152)) and all values finite;
   * state handling: one-shot prefill vs two-chunk prefill vs prefill + per-token decode agree to KL 5.5e-14 / 2.5e-13
     (max |dlogit| 2e-5), i.e. the per-pass conv/KV slots are consistent;
   * bf16-flow emulation (`emu_bf16`, activations rounded to bf16 at module outputs) vs the fp32 golden: top-1 identical on the 3
     short prompts, max |dlogit| 0.16 / 0.63 / 0.79 (KL 1e-11 only because the first-token distribution is extremely peaked:
     top-1 logit ~27 vs 7). So an HF-bf16 run is expected to differ from our fp32 golden by up to ~0.8 logit at some positions;
     use top-20 logit tolerance rather than KL on peaked positions.
3. What is still unverified against HF: exact bf16 rounding points, the router tie behaviour, `message_gate` rounding, and that the PR
   code equals the Thinking Machines original (the PR states it is a port; the checkpoint loads through its conversion mapping).
   First action for a follow-up with permission to run the PR: run `Inkling2LoopForCausalLM` (transformers @ a10ebc3, dtype bf16)
   on `tests/golden/inkling/*.npz` ids and compare logits_last / hid_* with `tools/inkling_selfcheck.compare`.

Reference speed (CPU, 64 threads shared, fp32): prefill 17-41 s for 25-37 tokens (first call incl. lazy expert dequant),
285 s for 4096 tokens (14 tok/s); decode 1.4-2.5 s/token. Memory: lazily dequantised bf16 experts, cache cap 26-40 GiB.

## 16. Open questions

1. HF parity (above). 2. MTP graph (section 14). 3. Activation quantisation of the vendor runtime for expert matmuls (NVFP4 per-token
amax / `fouroversix_mae`, MXFP8) - HF ignores it; how much quality does it cost / is it required for the intended accuracy?
4. Router tie-breaking and `topk(sorted=False)` ordering (irrelevant for the sum, relevant for bit-exactness of fp32 accumulation order).
5. `is_varloop` / research checkpoint: other pass counts are possible for speed trading; the main checkpoint was trained at 3.
6. Whether the sliding-layer KV truly needs only 1024 entries including the 3-token conv history (conv state is separate and tiny).
7. Long context beyond 8192: `tau` log scaling only affects global layers; the rel bias covers only 1024 back, so global-layer
   attention beyond that is pure content (+sink). Not exercised by the 4K golden (tau == 1 there); need an 16K+ golden to cover tau.
