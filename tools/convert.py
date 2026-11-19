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


# ---------------------------------------------------------------- bit plan (E0)
def plan_bits(name, bp):
    if re.search(r"(in_proj_a|in_proj_b)\.weight$", name):
        return "f16"
    if name.endswith("embed_tokens.weight"):
        return bp["embed"]
    if name == "lm_head.weight":
        return bp["head"]
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


def store(W, name, t, bits, dev, lazy=False):
    """Write tensor `t` (bf16 cpu) to the file; returns (fp32 dequantized tensor on dev for qsim, bytes)."""
    if bits in ("f32", "f16"):
        # Fold zero-centred RMSNorm (1+w) and A -> -exp(A_log) into stored values (documented in docs/FORMAT).
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
    packed, sm = [], []
    for r0 in range(0, t.shape[0], 4096):
        Wd = t[r0:r0 + 4096].to(dev).float()
        q, s, m = F.quantize_rtn(Wd, bits)
        packed.append(F.pack(q, bits))
        sm.append(torch.stack([s, m], -1))
        del q, Wd
    packed = torch.cat(packed)
    sm = torch.cat(sm)
    nb = W.add_quant(name, packed, sm, bits, t.shape)
    if lazy:
        return QuantRows(packed, sm, bits), nb
    return F.dequant(packed, sm, bits), nb


class QuantRows:
    """Packed matrix kept on GPU; dequantizes row ranges on demand (embed / lm_head are 1.27B params)."""

    def __init__(self, packed, sm, bits):
        self.packed, self.sm, self.bits = packed, sm, bits

    def rows(self, idx):
        return F.dequant(self.packed[idx], self.sm[idx], self.bits)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/workspace/q-volta-models/Qwen3.8-27B")
    ap.add_argument("--out", default="/workspace/q-volta-models/qv/qwen38-27b-e0")
    ap.add_argument("--eval-dir", default="/workspace/q-volta-models/qv/eval-e0")
    ap.add_argument("--eval-text", default="/workspace/q-volta-models/eval/wikitext2_test.txt")
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--dbg-tokens", type=int, default=64)
    ap.add_argument("--lora", default=None, help="PEFT LoRA adapter dir to merge into the BF16 weights before quantization")
    ap.add_argument("--bits", default="mlp_gu=3,mlp_down=3,gdn=4,attn=4,embed=4,head=6")
    args = ap.parse_args()
    bp = {k: int(v) for k, v in (kv.split("=") for kv in args.bits.split(","))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    dev = "cuda:0"
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

    # eval tokens (held-out wikitext-2 test)
    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
    ids = tok.encode(open(args.eval_text).read()).ids
    need = args.nseq * args.seqlen
    assert len(ids) >= need
    toks = torch.tensor(ids[:need], dtype=torch.long).view(args.nseq, args.seqlen)
    np.save(os.path.join(args.eval_dir, "tokens.npy"), toks.numpy().astype(np.int32))

    R = Remote(args.hf_dir)
    if args.lora:
        R.load_lora(args.lora)
    W = F.Writer(args.out)
    total = {"bytes": 0}

    def put(name, t, bits, lazy=False):
        d, nb = store(W, name, t, bits, dev, lazy)
        total["bytes"] += nb
        return d

    t0 = time.time()
    # embeddings
    emb = R.fetch(LM + "embed_tokens.weight")
    emb_q = put(LM + "embed_tokens.weight", emb, plan_bits(LM + "embed_tokens.weight", bp), lazy=True)
    flat = toks.view(-1)
    h_ref = emb[flat].float().to(dev).view(args.nseq, args.seqlen, -1)
    h_q = emb_q.rows(flat.to(dev)).view(args.nseq, args.seqlen, -1).clone()
    del emb_q, emb
    torch.cuda.empty_cache()
    print(f"[{time.time()-t0:6.0f}s] embed done", flush=True)

    rot = MQ.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
    pos = torch.arange(args.seqlen, device=dev)[None, None, :].expand(3, args.nseq, -1)
    pe = rot(h_ref, pos)
    pe = (pe[0].float(), pe[1].float())
    S = args.seqlen
    causal = torch.full((S, S), torch.finfo(torch.float32).min, device=dev).triu(1)[None, None]

    dbg = args.dbg_tokens
    dbg_ref = [h_ref[0, :dbg].cpu().clone()]
    dbg_q = [h_q[0, :dbg].cpu().clone()]

    # prefetch layers in a background thread
    q_layers = queue.Queue(maxsize=2)

    def prefetcher():
        for i in range(L):
            names = layer_names(i, ltypes[i])
            q_layers.put((i, {n: R.fetch(n) for n in names}))

    threading.Thread(target=prefetcher, daemon=True).start()

    for _ in range(L):
        i, tensors = q_layers.get()
        p = f"{LM}layers.{i}."
        layer_ref = MQ.Qwen3_5DecoderLayer(tcfg, i).to(dev).float()
        layer_q = MQ.Qwen3_5DecoderLayer(tcfg, i).to(dev).float()
        sd_ref, sd_q = {}, {}
        for n, t in tensors.items():
            k = n[len(p):]
            sd_ref[k] = t.float().to(dev)
            if k.endswith("conv1d.weight"):
                sd_ref[k] = sd_ref[k].view(-1, 1, t.shape[-1])
            dq = put(n, t, plan_bits(n, bp))
            if k.endswith("conv1d.weight"):
                dq = dq.view(-1, 1, t.shape[-1])
            sd_q[k] = dq
        layer_ref.load_state_dict(sd_ref, strict=True)
        layer_q.load_state_dict(sd_q, strict=True)
        del sd_ref, sd_q, tensors
        mask = causal if ltypes[i] == "full_attention" else None
        with torch.no_grad():
            h_ref = layer_ref(h_ref, position_embeddings=pe, attention_mask=mask)
            h_q = layer_q(h_q, position_embeddings=pe, attention_mask=mask)
        del layer_ref, layer_q
        torch.cuda.empty_cache()
        dbg_ref.append(h_ref[0, :dbg].cpu().clone())
        dbg_q.append(h_q[0, :dbg].cpu().clone())
        rel = ((h_q - h_ref).norm() / h_ref.norm()).item()
        print(f"[{time.time()-t0:6.0f}s] layer {i:2d} {ltypes[i][:6]} rel_err={rel:.4f} "
              f"file={total['bytes']/2**30:.2f} GiB", flush=True)

    # final norm + lm_head
    nw = R.fetch(LM + "norm.weight")
    put(LM + "norm.weight", nw, "f32")
    norm = MQ.Qwen3_5RMSNorm(tcfg.hidden_size, eps=tcfg.rms_norm_eps).to(dev)
    norm.weight.data = nw.float().to(dev)
    with torch.no_grad():
        f_ref = norm(h_ref).view(-1, h_ref.shape[-1])
        f_q = norm(h_q).view(-1, h_q.shape[-1])
    head = R.fetch("lm_head.weight")
    head_q = put("lm_head.weight", head, plan_bits("lm_head.weight", bp), lazy=True)
    head_bf = head.to(dev)
    del head
    print(f"[{time.time()-t0:6.0f}s] head quantized", flush=True)

    # metrics: next-token NLL for ref/q, KL(ref||q), top-1 agreement
    tgt = toks.to(dev)[:, 1:].reshape(-1)
    nll_r, nll_q, kl, agree, cnt = 0.0, 0.0, 0.0, 0, 0
    V = head_bf.shape[0]
    dbg_logits = None
    with torch.no_grad():
        for s in range(args.nseq):
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
            nll_r += -pr.gather(1, t[:, None]).sum().item()
            nll_q += -pq.gather(1, t[:, None]).sum().item()
            kl += (pr.exp() * (pr - pq)).sum().item()
            agree += (lr.argmax(-1) == lq.argmax(-1)).sum().item()
            cnt += S - 1
            del lr, lq, pr, pq
    del head_q, head_bf
    torch.cuda.empty_cache()
    metrics = dict(bits=bp, tokens=cnt, ppl_ref=float(np.exp(nll_r / cnt)), ppl_q=float(np.exp(nll_q / cnt)),
                   kl_ref_q=kl / cnt, top1_agree=agree / cnt, final_rel_err=((h_q - h_ref).norm() / h_ref.norm()).item())
    print(json.dumps(metrics, indent=1), flush=True)

    # MTP tensors (quantized, used in a later milestone)
    for n in sorted(k for k in R.wmap if k.startswith("mtp.")):
        put(n, R.fetch(n), plan_bits(n, bp) if "proj" in n or n == "mtp.fc.weight" else "f32")
    print(f"[{time.time()-t0:6.0f}s] mtp done", flush=True)

    cfg_out = dict(tcfg_d)
    cfg_out["source"] = dict(repo=REPO, revision=REV)
    cfg_out["bit_plan"] = bp
    cfg_out["folded"] = "zero-centred RMSNorm weights stored as (1+w); A_log stored as -exp(A_log); conv1d [C,4]"
    W.close(cfg_out)
    torch.save(dict(ref=torch.stack(dbg_ref), q=torch.stack(dbg_q), logits_q=dbg_logits,
                    logits_ref=dbg_logits_ref), os.path.join(args.eval_dir, "golden_seq0.pt"))
    if args.lora:  # every adapter module must have been merged exactly once; otherwise names did not match
        n_a = sum(k.endswith("lora_A.weight") for k in R.lora[0])
        assert len(R.lora_used) == n_a, f"LoRA merged {len(R.lora_used)}/{n_a} modules"
        metrics["lora"] = {"dir": args.lora, "modules": n_a, "scale": R.lora[1]}
        print(f"LoRA merged into {n_a}/{n_a} modules", flush=True)
    json.dump(metrics, open(os.path.join(args.eval_dir, "metrics.json"), "w"), indent=1)
    print(f"[{time.time()-t0:6.0f}s] DONE total {total['bytes']/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
