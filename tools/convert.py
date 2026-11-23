#!/usr/bin/env python3
"""Stream Qwen3.8-27B BF16 tensors from the HF Hub with HTTP range requests (never storing the full checkpoint),
quantize them into the Q-Volta format, and in the same pass run two reference forwards layer by layer on the GPU:
  * ref : original BF16 weights (fp32 math)          -> quality baseline
  * qsim: weights dequantized from the packed format  -> golden values the CUDA engine must reproduce
Outputs: <out>.qv.json/.qv.bin, <eval_dir>/ (tokens, per-layer activations of seq 0, logits, metrics.json)
"""
import argparse, json, os, re, subprocess, sys, threading, time, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_format as F
import qv_quant as Q

REPO = "Qwen/Qwen3.8-27B"
REV = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
LM = "model.language_model."

# ---------------------------------------------------------------- remote tensor access
pool = ThreadPoolExecutor(12)


def curl_range(fname, a, b, tries=6):
    for t in range(tries):
        r = subprocess.run(["curl", "-sfL", "-m", "600", "-r", f"{a}-{b}", BASE + fname], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == b - a + 1:
            return r.stdout
        time.sleep(2 + 3 * t)
    raise RuntimeError(f"range fetch failed {fname} {a}-{b} rc={r.returncode} got={len(r.stdout)}")


class Remote:
    def __init__(self, local_dir):
        idx = json.load(open(os.path.join(local_dir, "model.safetensors.index.json")))
        self.wmap = idx["weight_map"]
        self.headers = {}
        self.lock = threading.Lock()
        self.local_dir = local_dir if all(os.path.exists(os.path.join(local_dir, f)) for f in set(self.wmap.values())) else None
        print("tensor source:", "local shards" if self.local_dir else "HTTP range requests", flush=True)

    def header(self, shard):
        with self.lock:
            if shard in self.headers:
                return self.headers[shard]
        n = int.from_bytes(curl_range(shard, 0, 7), "little")
        h = json.loads(curl_range(shard, 8, 8 + n - 1))
        with self.lock:
            self.headers[shard] = (8 + n, h)
        return self.headers[shard]

    # ---- optional LoRA merge (PEFT semantics: W' = W + (alpha/r) * B @ A, rounded to bf16 like merge_and_unload)
    lora = None  # (dir-loaded dict name->tensor, scale)

    def load_lora(self, d):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(d, "adapter_config.json")))
        assert cfg["peft_type"] == "LORA" and not cfg.get("use_dora") and not cfg.get("use_rslora")
        assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern") and cfg.get("bias") == "none"
        sd = load_file(os.path.join(d, "adapter_model.safetensors"))
        pre = "base_model.model."
        self.lora = ({k[len(pre):]: v for k, v in sd.items()}, cfg["lora_alpha"] / cfg["r"])
        self.lora_used = set()
        print(f"LoRA: {len(sd)//2} modules, r={cfg['r']} alpha={cfg['lora_alpha']} scale={self.lora[1]}", flush=True)

    def fetch(self, name) -> torch.Tensor:
        t = self.fetch_base(name)
        if self.lora and name.endswith(".weight"):
            sd, scale = self.lora
            a = name[:-len("weight")] + "lora_A.weight"
            if a in sd:
                A = sd[a].cuda().float()
                B = sd[name[:-len("weight")] + "lora_B.weight"].cuda().float()
                assert B.shape[0] == t.shape[0] and A.shape[1] == t.shape[1], (name, A.shape, B.shape, t.shape)
                t = (t.cuda().float() + scale * (B @ A)).to(torch.bfloat16).cpu()
                self.lora_used.add(a)
        return t

    def fetch_base(self, name) -> torch.Tensor:
        shard = self.wmap[name]
        if self.local_dir:  # local shard present (downloaded by scripts/download_bf16.sh)
            from safetensors import safe_open
            with safe_open(os.path.join(self.local_dir, shard), framework="pt") as fh:
                return fh.get_tensor(name)
        base, h = self.header(shard)
        m = h[name]
        a, b = m["data_offsets"]
        a += base
        b += base
        step = 64 << 20
        parts = [(s, min(s + step, b) - 1) for s in range(a, b, step)]
        bufs = list(pool.map(lambda p: curl_range(shard, p[0], p[1]), parts))
        raw = b"".join(bufs)
        assert m["dtype"] == "BF16", (name, m["dtype"])
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(m["shape"])
        return t


# ---------------------------------------------------------------- bit plan
def plan_bits(name, bp):
    if re.search(r"(in_proj_a|in_proj_b)\.weight$", name):
        return "f16"
    if name.endswith("embed_tokens.weight"):
        return bp["embed"]
    if name == "lm_head.weight":
        return bp["head"]
    if name.startswith("mtp.") and "mtp" in bp and (name == "mtp.fc.weight" or name.endswith("proj.weight")) \
            and not re.search(r"(in_proj_a|in_proj_b)\.weight$", name):
        return bp["mtp"]  # MTP head: own plan entry, else same classes as the main model
    if re.search(r"mlp\.(gate|up)_proj\.weight$", name):
        return bp["mlp_gu"]
    if re.search(r"mlp\.down_proj\.weight$", name):
        return bp["mlp_down"]
    if re.search(r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$", name):
        return bp["gdn"]
    if re.search(r"self_attn\.(q|k|v|o)_proj\.weight$", name):
        return bp["attn"]
    if name == "mtp.fc.weight":
        return bp["attn"]
    return "f32"


def tensor_cfg(name, bp, plan, group):
    """-> (bits | 'f32' | 'f16', group). plan: {full tensor name: bits | [bits, group]} overrides the class defaults."""
    b = plan_bits(name, bp)
    g = group
    if b in ("f32", "f16"):
        return b, g
    v = plan.get(name)
    if v is not None:
        b, g = (v[0], v[1]) if isinstance(v, list) else (v, g)
    return b, g


def hess_key(name):
    if re.search(r"(linear_attn\.in_proj_(qkv|z)|self_attn\.(q|k|v)_proj)\.weight$", name):
        return "a"
    if re.search(r"(linear_attn\.out_proj|self_attn\.o_proj)\.weight$", name):
        return "o"
    if re.search(r"mlp\.(gate|up)_proj\.weight$", name):
        return "m"
    if re.search(r"mlp\.down_proj\.weight$", name):
        return "d"
    return None


def layer_names(i, ltype):
    p = f"{LM}layers.{i}."
    n = [p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
         p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"]
    if ltype == "linear_attention":
        n += [p + "linear_attn." + s for s in ("A_log", "conv1d.weight", "dt_bias", "in_proj_a.weight",
                                                 "in_proj_b.weight", "in_proj_qkv.weight", "in_proj_z.weight",
                                                 "norm.weight", "out_proj.weight")]
    else:
        n += [p + "self_attn." + s for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                               "q_norm.weight", "k_norm.weight")]
    return n


def store_dense(W, name, t, bits, dev):
    """Fold zero-centred RMSNorm (1+w) and A -> -exp(A_log) into stored values (documented in docs/FORMAT)."""
    v = t.float()
    if re.search(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm)\.weight$|language_model\.norm\.weight$"
                 r"|mtp\.(norm|pre_fc_norm_embedding|pre_fc_norm_hidden)\.weight$", name):
        stored = 1.0 + v
    elif name.endswith("A_log"):
        stored = -torch.exp(v)
    elif name.endswith("conv1d.weight"):
        stored = v.reshape(v.shape[0], v.shape[-1])
    else:
        stored = v
    nb = W.add_dense(name, stored, bits)
    if bits == "f16":
        return v.half().float().to(dev), nb  # qsim must see the same rounding as the engine
    return v.to(dev), nb


def quant_rows(t, bits, group, dev, method, Hu=None, clip=False, colw=None, rows=4096, meta8=False, H=None, opt=None, damp=None):
    """t [N,K] bf16 (cpu or dev) -> (packed int32, sm, rs) quantized in row chunks (GPTQ chunks share Hu).
    sm = fp16 [N,K/g,2] (scale,min) or, with meta8, uint16 [N,K/g] (+ rs fp16 [N] row scales)."""
    packed, sm, rss = [], [], []
    for r0 in range(0, t.shape[0], rows):
        Wd = t[r0:r0 + rows].to(dev).float()
        if method == "gptq" and H is not None and opt is not None and opt.gptq_mode != "orig":
            q, s, m, rs = Q.gptq_static(Wd, H, bits, group, clip, damp if damp is not None else opt.percdamp, actorder=opt.gptq_mode == "ao", meta8=meta8)
        elif method == "gptq" and Hu is not None:
            q, s, m, rs = Q.gptq(Wd, Hu, bits, group, clip=clip, colw=colw, meta8=meta8)
        else:
            q, s, m, rs = Q.rtn(Wd, bits, group, clip=clip, colw=colw, meta8=meta8)
        if method == "gptq" and H is not None and opt is not None and opt.refit:
            s, m = Q.refit_full(Wd, q, s, m, rs, H, bits, group, passes=opt.refit)
        packed.append(F.pack(q, bits))
        sm.append(Q.pack_meta(s, m, rs))
        rss.append(rs)
        del q, Wd
    return torch.cat(packed), torch.cat(sm), (torch.cat(rss) if meta8 else None)


class QuantRows:
    """Packed matrix kept on GPU; dequantizes row ranges on demand (embed / lm_head are 1.27B params)."""

    def __init__(self, packed, sm, bits, group, rs=None):
        self.packed, self.sm, self.bits, self.group, self.rs = packed, sm, bits, group, rs

    def rows(self, idx):
        return F.dequant(self.packed[idx], self.sm[idx], self.bits, self.group, None if self.rs is None else self.rs[idx])


class HAcc:
    def __init__(self, K, dev):
        self.H = torch.zeros(K, K, device=dev)
        self.n = 0

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        self.H.addmm_(x.T, x)
        self.n += x.shape[0]

    def mean(self):
        self.H.div_(max(self.n, 1))  # in place (the 17408^2 Hessian is 1.2 GB)
        self.n = 1
        return self.H


def read_tokens(tok, path, seqlen, nmax=None):
    ids = tok.encode(open(path).read()).ids
    n = len(ids) // seqlen
    if nmax:
        n = min(n, nmax)
    return torch.tensor(ids[:n * seqlen], dtype=torch.long).view(n, seqlen)


def save_state(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/workspace/q-volta-models/Qwen3.8-27B")
    ap.add_argument("--out", default="/workspace/q-volta-models/qv/qwen38-27b-e0")
    ap.add_argument("--eval-dir", default="/workspace/q-volta-models/qv/eval-e0")
    ap.add_argument("--eval-text", default="/workspace/q-volta-models/eval/wikitext2_test.txt")
    ap.add_argument("--eval-text2", default=None, help="second eval set (e.g. agentic_synth.txt), 512-token windows")
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--dbg-tokens", type=int, default=64)
    ap.add_argument("--lora", default=None, help="PEFT LoRA adapter dir to merge into the BF16 weights before quantization")
    ap.add_argument("--lora-section", default=None, help="PEFT LoRA dir stored UNMERGED (fp16 A/B) as an optional container section")
    ap.add_argument("--bits", default="mlp_gu=3,mlp_down=3,gdn=4,attn=4,embed=4,head=6")
    ap.add_argument("--group", type=int, default=64, help="default quantization group (64 | 128)")
    ap.add_argument("--method", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--meta8", type=int, default=0, help="8-bit scale + 8-bit zero point per group + fp16 row scale (0.125 bit/weight at g128)")
    ap.add_argument("--clip", type=int, default=0, help="per-group clip-range search")
    ap.add_argument("--embed-clip", type=int, default=0)
    ap.add_argument("--head-sweep", default=None, help="extra lm_head variants evaluated on the final hidden states, e.g. 4:128:rtn,5:128:rtn,4:128:gptq")
    ap.add_argument("--head-method", default=None, choices=[None, "rtn", "gptq"])
    ap.add_argument("--plan", default=None, help="JSON {tensor name: bits | [bits, group]} per-tensor overrides")
    ap.add_argument("--calib", default=None, help="npy int32 [n, L] calibration tokens (required for gptq / --sens)")
    ap.add_argument("--ncalib", type=int, default=0)
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--emit-qv", action="store_true", help="pack into a single .qv container and delete the .json/.bin pair")
    ap.add_argument("--no-embed", action="store_true", help="with --emit-qv: do not embed tokenizer.json / chat_template.jinja / generation_config.json from --hf-dir (default: embed what exists)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="evaluation only: do not write the model file")
    ap.add_argument("--max-layers", type=int, default=0, help="stop after this many layers in this invocation (state saved)")
    ap.add_argument("--sens", default=None, help="sensitivity mode: write per-tensor GPTQ errors to this json; no model written")
    ap.add_argument("--sens-bits", default="2,3,4,5,6")
    ap.add_argument("--sens-groups", default="128")
    ap.add_argument("--sens-method", default="rtn", choices=["rtn", "gptq"], help="quantizer used for the sensitivity table (rtn = RTN + clip search, ~10x cheaper)")
    ap.add_argument("--percdamp", type=float, default=0.01, help="GPTQ Hessian damping (fraction of mean diag)")
    ap.add_argument("--damp-map", default="", help="per Hessian-key damping, e.g. a=0.01,o=0.1,m=0.1,d=0.3 (keys: a qkv/z/q/k/v, o out/o_proj, m gate/up, d down); default --percdamp")
    ap.add_argument("--gptq-mode", default="orig", choices=["orig", "static", "ao"],
                    help="orig: group params from error-compensated weights (original sweep order); static: params from original weights; ao: static + act-order")
    ap.add_argument("--refit", type=int, default=0, help="passes of full-Hessian least-squares refit of (scale, min) after GPTQ")
    ap.add_argument("--gptaq", type=int, default=0, help="asymmetric calibration (GPTQv2/GPTAQ style): quantize W_eff = W Cxy H^-1, Cxy = E[x_fp x_q^T]")
    ap.add_argument("--calib-bf16", type=int, default=0, help="store the calibration residual streams in bf16 (2x less memory; allows 256 sequences)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    bp = {k: int(v) for k, v in (kv.split("=") for kv in args.bits.split(","))}
    dmap = {k: float(v) for k, v in (kv.split("=") for kv in args.damp_map.split(",") if kv)}
    plan = json.load(open(args.plan)) if args.plan else {}
    sens = args.sens is not None
    use_h = args.method == "gptq" or sens
    head_method = args.head_method or args.method
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
    from tokenizers import Tokenizer

    full_cfg = json.load(open(os.path.join(args.hf_dir, "config.json")))
    tcfg_d = full_cfg["text_config"]
    tcfg = Qwen3_5TextConfig(**tcfg_d)
    tcfg._attn_implementation = "eager"
    L = tcfg.num_hidden_layers
    ltypes = tcfg.layer_types

    # eval sets: wikitext-2 test (held out), optional agentic corpus. Both use fixed 512-token windows.
    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
    ids = tok.encode(open(args.eval_text).read()).ids
    need = args.nseq * args.seqlen
    assert len(ids) >= need
    sets = {"wiki": torch.tensor(ids[:need], dtype=torch.long).view(args.nseq, args.seqlen)}
    if args.eval_text2:
        sets["agentic"] = read_tokens(tok, args.eval_text2, args.seqlen)
    toks = torch.cat(list(sets.values()))
    set_rng, o = {}, 0
    for k, v in sets.items():
        set_rng[k] = (o, o + len(v))
        o += len(v)
    nseq = len(toks)
    np.save(os.path.join(args.eval_dir, "tokens.npy"), sets["wiki"].numpy().astype(np.int32))
    calib = None
    if use_h:
        calib = torch.from_numpy(np.load(args.calib)).long()
        if args.ncalib:
            calib = calib[:args.ncalib]
        print(f"calibration: {tuple(calib.shape)} tokens", flush=True)
    Sc = calib.shape[1] if calib is not None else 0

    R = Remote(args.hf_dir)
    if args.lora:
        R.load_lora(args.lora)
    state_path = args.out + ".state.pt"
    st = None
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        print(f"resuming at layer {st['next']}", flush=True)
    W = None
    if not sens:
        W = F.Writer(args.out, resume=None if st is None else (st["w_off"], st["w_tensors"]), null=args.no_write)
    total = {"bytes": 0 if st is None else st["bytes"]}
    stats = {} if st is None else st["stats"]
    t0 = time.time()

    S = args.seqlen
    rot = MQ.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)

    def mk_pe(seq):
        pos = torch.arange(seq, device=dev)[None, None, :].expand(3, 1, -1)
        z = torch.zeros(1, seq, tcfg.hidden_size, device=dev)
        pe = rot(z, pos)
        return (pe[0].float(), pe[1].float()), torch.full((seq, seq), torch.finfo(torch.float32).min, device=dev).triu(1)[None, None]

    pe_e, causal_e = mk_pe(S)
    if use_h:
        pe_c, causal_c = mk_pe(Sc)
    dbg = args.dbg_tokens
    EB = 13  # eval batch

    def run_layer(layer, h, pe, causal, ltype, bs):
        mask = causal if ltype == "full_attention" else None
        out = torch.empty_like(h)  # same dtype as the input stream (calibration streams may be stored in bf16)
        for b in range(0, h.shape[0], bs):
            out[b:b + bs] = layer(h[b:b + bs].float(), position_embeddings=pe, attention_mask=mask)
        return out

    if st is None:
        emb = R.fetch(LM + "embed_tokens.weight")
        ename = LM + "embed_tokens.weight"
        eb, eg = tensor_cfg(ename, bp, plan, args.group)
        if sens:
            emb_q = None
        else:
            pk, sm, rs_ = quant_rows(emb, eb, eg, dev, "rtn", clip=bool(args.embed_clip), meta8=bool(args.meta8))
            total["bytes"] += W.add_quant(ename, pk, sm, eb, emb.shape, eg, rs_)
            emb_q = QuantRows(pk, sm, eb, eg, rs_)
        flat = toks.view(-1)
        h_ref = emb[flat].float().to(dev).view(nseq, S, -1)
        h_q = h_ref.clone() if sens else emb_q.rows(flat.to(dev)).view(nseq, S, -1).clone()
        h_cal = None
        h_fp = None
        if use_h:
            cf = calib.view(-1)
            cdt = torch.bfloat16 if args.calib_bf16 else torch.float32  # the BF16 model itself carries a bf16 residual stream
            h_cal = torch.empty(len(calib), Sc, emb.shape[1], dtype=cdt, device=dev)
            if args.gptaq:
                h_fp = torch.empty_like(h_cal)
            for c0 in range(0, len(calib), 16):  # chunked: the fp32 rows of 256 sequences would be 5 GB
                ci = cf.view(len(calib), Sc)[c0:c0 + 16].reshape(-1)
                x = (emb[ci].float().to(dev) if sens else emb_q.rows(ci.to(dev))).view(-1, Sc, emb.shape[1])
                h_cal[c0:c0 + 16] = x.to(cdt)
                if args.gptaq:
                    h_fp[c0:c0 + 16] = emb[ci].float().to(dev).view(-1, Sc, emb.shape[1]).to(cdt)
        del emb_q, emb
        torch.cuda.empty_cache()
        dbg_ref = [h_ref[0, :dbg].cpu().clone()]
        dbg_q = [h_q[0, :dbg].cpu().clone()]
        nxt = 0
        print(f"[{time.time()-t0:6.0f}s] embed done  gpu_alloc={torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    else:
        h_ref, h_q, h_cal = st["h_ref"].to(dev), st["h_q"].to(dev), None if st["h_cal"] is None else st["h_cal"].to(dev)
        h_fp = None if st.get("h_fp") is None else st["h_fp"].to(dev)
        dbg_ref, dbg_q, nxt = st["dbg_ref"], st["dbg_q"], st["next"]
        del st

    q_layers = queue.Queue(maxsize=2)
    last = L if not args.max_layers else min(L, nxt + args.max_layers)

    def prefetcher():
        for i in range(nxt, last):
            q_layers.put((i, {n: R.fetch(n) for n in layer_names(i, ltypes[i])}))

    threading.Thread(target=prefetcher, daemon=True).start()

    for _ in range(nxt, last):
        i, tensors = q_layers.get()
        p = f"{LM}layers.{i}."
        sd_ref = {}
        for n, t in tensors.items():
            k = n[len(p):]
            sd_ref[k] = t.float().to(dev)
            if k.endswith("conv1d.weight"):
                sd_ref[k] = sd_ref[k].view(-1, 1, t.shape[-1])
        layer_ref = MQ.Qwen3_5DecoderLayer(tcfg, i).to(dev).float()
        layer_ref.load_state_dict(sd_ref, strict=True)
        del sd_ref
        mixer = layer_ref.linear_attn if ltypes[i] == "linear_attention" else layer_ref.self_attn
        # ---- Hessians (inputs of the already-quantized stream, BF16 weights of this layer) + ref eval stream
        Hs, hooks, Cs = {}, [], {}
        resid_e = None
        with torch.no_grad():
            if use_h:
                resid_e = float(sum((h_cal[c0:c0 + 16].float() ** 2).sum(-1).sum() for c0 in range(0, len(h_cal), 16)) / (h_cal.shape[0] * h_cal.shape[1]))
                mods = {"a": mixer.in_proj_qkv if ltypes[i] == "linear_attention" else mixer.q_proj,
                        "o": mixer.out_proj if ltypes[i] == "linear_attention" else mixer.o_proj,
                        "m": layer_ref.mlp.gate_proj, "d": layer_ref.mlp.down_proj}
                cap = {}
                for key, m in mods.items():
                    acc = HAcc(m.in_features, dev)
                    Hs[key] = acc
                    if args.gptaq:
                        Cs[key] = torch.zeros(m.in_features, m.in_features, device=dev)
                        hooks.append(m.register_forward_pre_hook(lambda mod, a, key=key: cap.__setitem__(key, a[0].reshape(-1, a[0].shape[-1]).float())))
                    else:
                        hooks.append(m.register_forward_pre_hook(lambda mod, a, acc=acc: acc.add(a[0])))
                if args.gptaq:
                    mask_c = causal_c if ltypes[i] == "full_attention" else None
                    for b in range(0, h_cal.shape[0], args.calib_batch):
                        layer_ref(h_cal[b:b + args.calib_batch].float(), position_embeddings=pe_c, attention_mask=mask_c)
                        xq = dict(cap)
                        h_fp[b:b + args.calib_batch] = layer_ref(h_fp[b:b + args.calib_batch].float(), position_embeddings=pe_c,
                                                                 attention_mask=mask_c).to(h_fp.dtype)  # fp stream advances in place
                        for key in mods:
                            Hs[key].add(xq[key])
                            Cs[key].addmm_(cap[key].T, xq[key])
                        cap.clear()
                    h_cal_ref_out = None
                    for key in Cs:
                        Cs[key].div_(Hs[key].n)
                else:
                    h_cal_ref_out = run_layer(layer_ref, h_cal, pe_c, causal_c, ltypes[i], args.calib_batch)
                if not sens:
                    del h_cal_ref_out
                for hk in hooks:
                    hk.remove()
            h_ref = run_layer(layer_ref, h_ref, pe_e, causal_e, ltypes[i], EB)
        del layer_ref, hooks
        torch.cuda.empty_cache()
        # ---- quantize this layer's tensors
        dq_sd, lstat = {}, {}
        Hu_cache = {}
        groups = {}  # (hess key, bits, group) -> [names]
        for n, t in tensors.items():
            k = n[len(p):]
            bits, grp = tensor_cfg(n, bp, plan, args.group)
            if bits in ("f32", "f16"):
                dq, nb = store_dense(W, n, t, bits, dev) if not sens else (t.float().to(dev), 0)
                if k.endswith("conv1d.weight"):
                    dq = dq.view(-1, 1, t.shape[-1])
                total["bytes"] += nb
                dq_sd[k] = dq
            else:
                groups.setdefault((hess_key(n), bits, grp), []).append(n)
        last_use = {k[0]: j for j, k in enumerate(groups)}
        for gi, ((hk, bits, grp), names) in enumerate(groups.items()):
            Hm = Hs[hk].mean() if (use_h and hk) else None
            colw = torch.diag(Hm).clone() if Hm is not None else None
            Hu = None
            if args.method == "gptq" and Hm is not None and (args.gptq_mode == "orig" or sens):
                if hk not in Hu_cache:
                    Hu_cache[hk] = Q.hinv_chol(Hm, dmap.get(hk, args.percdamp))
                Hu = Hu_cache[hk]
            cat = torch.cat([tensors[n] for n in names]) if len(names) > 1 else tensors[names[0]]
            if args.gptaq and hk in Cs and not sens:
                cat = Q.effective_weights(cat.to(dev).float(), Hm, Cs[hk], dmap.get(hk, args.percdamp))
            if sens:
                Wd = cat.to(dev).float()
                if args.sens_method == "gptq" and hk not in Hu_cache:
                    Hu_cache[hk] = Q.hinv_chol(Hm)
                for sb in [int(x) for x in args.sens_bits.split(",")]:
                    for sg in [int(x) for x in args.sens_groups.split(",")]:
                        if Wd.shape[1] % sg:
                            continue
                        if args.sens_method == "gptq":
                            q_, s_, m_, _ = Q.gptq(Wd, Hu_cache[hk], sb, sg, clip=True, colw=colw)
                        else:
                            q_, s_, m_, _ = Q.rtn(Wd, sb, sg, clip=True, colw=colw)
                        r0 = 0
                        # per-tensor split of the concatenated matrix
                        for n in names:
                            rr = tensors[n].shape[0]
                            Dn = Wd[r0:r0 + rr] - Q.dequant_g(q_[r0:r0 + rr], s_[r0:r0 + rr], m_[r0:r0 + rr], sg)
                            e_n = ((Dn @ Hm) * Dn).sum().item()
                            en_n = ((Wd[r0:r0 + rr] @ Hm) * Wd[r0:r0 + rr]).sum().item()
                            lstat.setdefault(n, {})[f"b{sb}g{sg}"] = [e_n, en_n]
                            r0 += rr
                        del q_, s_, m_
                del Wd
                if last_use[hk] == gi:
                    Hs[hk] = None
                    Hu_cache.pop(hk, None)
                del Hm, colw
                torch.cuda.empty_cache()
                continue
            pk, sm, rs_ = quant_rows(cat, bits, grp, dev, args.method, Hu, bool(args.clip), colw, rows=1 << 30,
                                     meta8=bool(args.meta8), H=Hm, opt=args, damp=dmap.get(hk))
            r0 = 0
            for n in names:
                rr = tensors[n].shape[0]
                pkn, smn = pk[r0:r0 + rr], sm[r0:r0 + rr]
                rsn = None if rs_ is None else rs_[r0:r0 + rr]
                total["bytes"] += W.add_quant(n, pkn, smn, bits, tensors[n].shape, grp, rsn)
                dq = F.dequant(pkn, smn, bits, grp, rsn)
                if use_h:  # layer-local output error on the calibration Hessian
                    Wo = tensors[n].to(dev).float()
                    e, en = Q.out_err(Wo, dq, Hm)
                    lstat[n] = {"err": e, "energy": en, "bits": bits, "group": grp}
                    del Wo
                dq_sd[n[len(p):]] = dq
                r0 += rr
            del pk, sm, rs_, Hm, colw, Hu
            if last_use[hk] == gi:
                Hs[hk] = None
                Hu_cache.pop(hk, None)
            torch.cuda.empty_cache()
        del Hs, Hu_cache, Cs
        torch.cuda.empty_cache()
        stats.update(lstat)
        if resid_e is not None:
            stats[f"resid.{i}"] = resid_e
        # ---- propagate the quantized stream
        if sens:
            h_q = h_ref
            h_cal = h_cal_ref_out
        else:
            layer_q = MQ.Qwen3_5DecoderLayer(tcfg, i).to(dev).float()
            layer_q.load_state_dict(dq_sd, strict=True)
            del dq_sd
            with torch.no_grad():
                h_q = run_layer(layer_q, h_q, pe_e, causal_e, ltypes[i], EB)
                if use_h:
                    h_cal = run_layer(layer_q, h_cal, pe_c, causal_c, ltypes[i], args.calib_batch)
            del layer_q
        del tensors
        torch.cuda.empty_cache()
        dbg_ref.append(h_ref[0, :dbg].cpu().clone())
        dbg_q.append(h_q[0, :dbg].cpu().clone())
        rel = ((h_q - h_ref).norm() / h_ref.norm()).item()
        print(f"[{time.time()-t0:6.0f}s] layer {i:2d} {ltypes[i][:6]} rel_err={rel:.4f} "
              f"file={total['bytes']/2**30:.2f} GiB", flush=True)
        nxt = i + 1

    if nxt < L:  # slice finished: save state and exit (driver relaunches with --resume)
        save_state(state_path, dict(next=nxt, h_ref=h_ref.cpu(), h_q=h_q.cpu(), h_cal=None if h_cal is None else h_cal.cpu(),
                                    h_fp=None if h_fp is None else h_fp.cpu(),
                                    dbg_ref=dbg_ref, dbg_q=dbg_q, bytes=total["bytes"], stats=stats,
                                    w_off=None if W is None else W.flush_state()[0],
                                    w_tensors=None if W is None else W.tensors))
        print(f"[{time.time()-t0:6.0f}s] SLICE DONE next={nxt}", flush=True)
        return

    # final norm + lm_head
    nw = R.fetch(LM + "norm.weight")
    if not sens:
        total["bytes"] += store_dense(W, LM + "norm.weight", nw, "f32", dev)[1]
    norm = MQ.Qwen3_5RMSNorm(tcfg.hidden_size, eps=tcfg.rms_norm_eps).to(dev)
    norm.weight.data = nw.float().to(dev)
    with torch.no_grad():
        f_ref = norm(h_ref).view(-1, h_ref.shape[-1])
        f_q = norm(h_q).view(-1, h_q.shape[-1])
        f_cal = torch.cat([norm(h_cal[c0:c0 + 16].float()).to(h_cal.dtype) for c0 in range(0, len(h_cal), 16)]).view(-1, h_cal.shape[-1]) if h_cal is not None else None
    head = R.fetch("lm_head.weight")
    hb, hg = tensor_cfg("lm_head.weight", bp, plan, args.group)
    if sens:
        # lm_head sensitivity: errors of KL-relevant logits for each candidate bits (RTN+clip, GPTQ on a row subset)
        head_dev = head.to(dev).float()
        Hh = HAcc(head.shape[1], dev)
        [Hh.add(f_cal[c0:c0 + 32768].float()) for c0 in range(0, len(f_cal), 32768)]
        Hm = Hh.mean()
        sub = head_dev[:32768]
        for sb in [int(x) for x in args.sens_bits.split(",")]:
            for sg in [int(x) for x in args.sens_groups.split(",")]:
                q_, s_, m_, _ = Q.rtn(sub, sb, sg, clip=True, colw=torch.diag(Hm))
                e, en = Q.out_err(sub, Q.dequant_g(q_, s_, m_, sg), Hm)
                stats.setdefault("lm_head.weight", {})[f"b{sb}g{sg}"] = [e, en]
        json.dump(stats, open(args.sens, "w"), indent=0)
        print(f"sens written {args.sens}", flush=True)
        return
    hclip = bool(args.clip)
    Hu = None
    if head_method == "gptq":
        Hh = HAcc(head.shape[1], dev)
        [Hh.add(f_cal[c0:c0 + 32768].float()) for c0 in range(0, len(f_cal), 32768)]
        Hm = Hh.mean()
        Hu = Q.hinv_chol(Hm)
    pk, sm, rs_ = quant_rows(head, hb, hg, dev, head_method, Hu, hclip, torch.diag(Hm).clone() if Hu is not None else None,
                             rows=32768, meta8=bool(args.meta8))
    total["bytes"] += W.add_quant("lm_head.weight", pk, sm, hb, head.shape, hg, rs_)
    head_q = QuantRows(pk, sm, hb, hg, rs_)
    head_bf = head.to(dev)
    del head
    print(f"[{time.time()-t0:6.0f}s] head quantized", flush=True)

    # metrics per eval set: next-token NLL for ref/q, KL(ref||q), top-1 agreement
    tgt = toks.to(dev)[:, 1:].reshape(-1)
    V = head_bf.shape[0]
    acc = {k: dict(nll_r=0.0, nll_q=0.0, kl=0.0, agree=0, cnt=0, kls=[]) for k in sets}
    set_of = {}
    for k, (a, b) in set_rng.items():
        for s_ in range(a, b):
            set_of[s_] = k
    dbg_logits = None
    with torch.no_grad():
        for s in range(nseq):
            rows = slice(s * S, s * S + S - 1)
            lr = torch.empty(S - 1, V, device=dev)
            lq = torch.empty(S - 1, V, device=dev)
            for v0 in range(0, V, 32768):
                v1 = min(V, v0 + 32768)
                lr[:, v0:v1] = f_ref[rows] @ head_bf[v0:v1].float().T
                lq[:, v0:v1] = f_q[rows] @ head_q.rows(slice(v0, v1)).T
            if s == 0:
                dbg_logits = lq[:dbg].cpu().clone()
                dbg_logits_ref = lr[:dbg].cpu().clone()
            pr = torch.log_softmax(lr, -1)
            pq = torch.log_softmax(lq, -1)
            t = tgt[s * (S - 1):(s + 1) * (S - 1)]
            a = acc[set_of[s]]
            a["nll_r"] += -pr.gather(1, t[:, None]).sum().item()
            a["nll_q"] += -pq.gather(1, t[:, None]).sum().item()
            klv = (pr.exp() * (pr - pq)).sum(-1)
            a["kl"] += klv.sum().item()
            a["kls"].append(klv.cpu())
            a["agree"] += (lr.argmax(-1) == lq.argmax(-1)).sum().item()
            a["cnt"] += S - 1
            del lr, lq, pr, pq
    sweep = {}
    if args.head_sweep:  # lever 4: lm_head bits/group/method evaluated on this run's final hidden states (wikitext set)
        lo, hi = set_rng["wiki"]
        Hu_h = colw_h = None
        for spec in args.head_sweep.split(","):
            hb_, hg_, hm_ = spec.split(":")[:3]
            hb_, hg_ = int(hb_), int(hg_)
            if hm_ == "gptq" and Hu_h is None:
                Hh = HAcc(head_bf.shape[1], dev)
                [Hh.add(f_cal[c0:c0 + 32768].float()) for c0 in range(0, len(f_cal), 32768)]
                Hm_h = Hh.mean()
                colw_h = torch.diag(Hm_h).clone()
                Hu_h = Q.hinv_chol(Hm_h)
                del Hh, Hm_h
            m8_ = len(spec.split(":")) > 3 and spec.split(":")[3] == "m8"
            pk_, sm_, rs_h = quant_rows(head_bf, hb_, hg_, dev, hm_, Hu_h if hm_ == "gptq" else None, True,
                                        colw_h if hm_ == "gptq" else None, rows=32768, meta8=m8_)
            hq_ = QuantRows(pk_, sm_, hb_, hg_, rs_h)
            nll_q = kl_s = 0.0
            agree = cnt = 0
            with torch.no_grad():
                for s_ in range(lo, hi):
                    rows = slice(s_ * S, s_ * S + S - 1)
                    lr = torch.empty(S - 1, V, device=dev)
                    lq = torch.empty(S - 1, V, device=dev)
                    for v0 in range(0, V, 32768):
                        v1 = min(V, v0 + 32768)
                        lr[:, v0:v1] = f_ref[rows] @ head_bf[v0:v1].float().T
                        lq[:, v0:v1] = f_q[rows] @ hq_.rows(slice(v0, v1)).T
                    pr, pq = torch.log_softmax(lr, -1), torch.log_softmax(lq, -1)
                    t = tgt[s_ * (S - 1):(s_ + 1) * (S - 1)]
                    nll_q += -pq.gather(1, t[:, None]).sum().item()
                    kl_s += (pr.exp() * (pr - pq)).sum().item()
                    agree += (lr.argmax(-1) == lq.argmax(-1)).sum().item()
                    cnt += S - 1
                    del lr, lq, pr, pq
            sweep[spec] = dict(ppl_q=float(np.exp(nll_q / cnt)), kl_ref_q=kl_s / cnt, top1_agree=agree / cnt,
                               head_gib=(pk_.numel() * 4 + sm_.numel() * (2 if m8_ else 2) + (0 if rs_h is None else rs_h.numel() * 2)) / 2**30)
            print("head-sweep", spec, sweep[spec], flush=True)
            del hq_, pk_, sm_
    del head_q, head_bf
    torch.cuda.empty_cache()
    metrics = dict(bits=bp, group=args.group, method=args.method, clip=args.clip, plan=bool(plan),
                   final_rel_err=((h_q - h_ref).norm() / h_ref.norm()).item())
    for k, a in acc.items():
        kl_all = torch.cat(a["kls"])
        metrics[k] = dict(tokens=a["cnt"], ppl_ref=float(np.exp(a["nll_r"] / a["cnt"])), ppl_q=float(np.exp(a["nll_q"] / a["cnt"])),
                          kl_ref_q=a["kl"] / a["cnt"], kl_median=kl_all.median().item(), kl_p99=kl_all.quantile(0.99).item(),
                          top1_agree=a["agree"] / a["cnt"])
    if sweep:
        metrics['head_sweep'] = sweep
    # legacy flat keys (wikitext) kept for tools that read metrics.json
    metrics.update(tokens=metrics["wiki"]["tokens"], ppl_ref=metrics["wiki"]["ppl_ref"], ppl_q=metrics["wiki"]["ppl_q"],
                   kl_ref_q=metrics["wiki"]["kl_ref_q"], top1_agree=metrics["wiki"]["top1_agree"])
    print(json.dumps(metrics, indent=1), flush=True)

    # MTP tensors (RTN + clip: the MTP head is 0.3B params; GPTQ Hessians for it are not collected)
    for n in sorted(k for k in R.wmap if k.startswith("mtp.")):
        b_, g_ = tensor_cfg(n, bp, plan, args.group)
        t = R.fetch(n)
        if b_ in ("f32", "f16"):
            total["bytes"] += store_dense(W, n, t, b_, dev)[1]
        else:
            pk, sm, rs_ = quant_rows(t, b_, g_, dev, "rtn", clip=bool(args.clip), meta8=bool(args.meta8))
            total["bytes"] += W.add_quant(n, pk, sm, b_, t.shape, g_, rs_)
    print(f"[{time.time()-t0:6.0f}s] mtp done", flush=True)

    if args.lora_section:
        total["bytes"] += add_lora_section(W, args.lora_section)

    cfg_out = dict(tcfg_d)
    cfg_out["source"] = dict(repo=REPO, revision=REV)
    cfg_out["bit_plan"] = bp
    cfg_out["quant"] = dict(method=args.method, group=args.group, clip=args.clip, meta8=args.meta8, plan=args.plan, calib=args.calib)
    cfg_out["folded"] = "zero-centred RMSNorm weights stored as (1+w); A_log stored as -exp(A_log); conv1d [C,4]"
    W.close(cfg_out)
    torch.save(dict(ref=torch.stack(dbg_ref), q=torch.stack(dbg_q), logits_q=dbg_logits,
                    logits_ref=dbg_logits_ref), os.path.join(args.eval_dir, "golden_seq0.pt"))
    if args.lora:  # every adapter module must have been merged exactly once; otherwise names did not match
        n_a = sum(k.endswith("lora_A.weight") for k in R.lora[0])
        assert len(R.lora_used) == n_a, f"LoRA merged {len(R.lora_used)}/{n_a} modules"
        metrics["lora"] = {"dir": args.lora, "modules": n_a, "scale": R.lora[1]}
        print(f"LoRA merged into {n_a}/{n_a} modules", flush=True)
    metrics["file_gib"] = total["bytes"] / 2**30
    json.dump(metrics, open(os.path.join(args.eval_dir, "metrics.json"), "w"), indent=1)
    if os.path.exists(state_path):
        os.remove(state_path)
    if args.emit_qv and not args.no_write:
        import qv_pack, qv_blobs
        blobs = meta = None
        if not args.no_embed:  # self-contained .qv: tokenizer + chat template + generation defaults + metadata (docs/QV_FORMAT.md)
            blobs, meta = qv_blobs.collect_hf_dir(args.hf_dir)
            if args.lora and os.path.isfile(os.path.join(args.lora, "chat_template.jinja")):  # an adapter's own template wins
                blobs["chat_template.jinja"] = open(os.path.join(args.lora, "chat_template.jinja"), "rb").read()
                meta["chat_template_from"] = "lora:" + os.path.basename(os.path.normpath(args.lora))
            meta.update(model_name=os.path.basename(os.path.normpath(args.out)), base_model=os.path.basename(os.path.normpath(args.hf_dir)),
                        quantization=f"bits={args.bits} group={args.group} method={args.method}" + (" meta8" if args.meta8 else "") + (" lora-merged" if args.lora else ""))
            if not blobs:
                print(f"note: no tokenizer/template files found in {args.hf_dir}; the .qv is not self-contained", flush=True)
        qv_pack.pack(args.out, args.out + ".qv", delete_src=True, blobs=blobs, meta=meta)
    print(f"[{time.time()-t0:6.0f}s] DONE total {total['bytes']/2**30:.2f} GiB", flush=True)


def add_lora_section(W, d):
    """Optional container section: the PEFT LoRA adapter stored unmerged as fp16 A/B tensors, one pair per target.
    Names: lora.<base weight name>.A [r,K], lora.<base weight name>.B [N,r]; scale = alpha/r goes into config.lora."""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    nb = 0
    for k, v in sd.items():
        k = k[len("base_model.model."):]
        base = re.sub(r"\.lora_[AB]\.weight$", ".weight", k)
        ab = "A" if ".lora_A." in k else "B"
        nb += W.add_dense(f"lora.{base}.{ab}", v, "f16", kind="lora_f16")
    W.extra_config["lora"] = dict(r=cfg["r"], alpha=cfg["lora_alpha"], scale=cfg["lora_alpha"] / cfg["r"], merged=False,
                                  source=os.path.basename(d.rstrip("/")))
    return nb


if __name__ == "__main__":
    main()
