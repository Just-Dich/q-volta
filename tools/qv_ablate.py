#!/usr/bin/env python3
"""Proxy ablation loop for GPTQ variants (p21/quant2): BF16 stream over train+held-out calibration sequences, and on sampled layers
compare quantizer variants by the layer-local output error  tr(D H_test D^T) / tr(W H_test W^T)  with H_test from HELD-OUT sequences
(never seen by the quantizer).  usage: qv_ablate.py OUT.json --layers 2,15,30,43,55,62 [--bits 3 --group 128 --meta8 1]"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as C
import qv_quant as Q

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--layers", default="2,15,30,43,55,62")
ap.add_argument("--bits", type=int, default=3)
ap.add_argument("--group", type=int, default=128)
ap.add_argument("--meta8", type=int, default=1)
ap.add_argument("--variants", default="rtn,g128,g128d10,g128d30,ao128,ao128d10,ao128d30,ao128d100,ao128s30,ao128s50")
ap.add_argument("--train", default="/workspace/q-volta-models/calib/calib128.npy")
ap.add_argument("--test", default="/workspace/q-volta-models/calib/heldout32.npy")
ap.add_argument("--budget", type=float, default=540, help="stop after this many seconds (state saved, rerun to resume)")
ap.add_argument("--state", default="/workspace/q-volta-models/scratch-quant2/ablate_state.pt")
ap.add_argument("--hf-dir", default="/workspace/q-volta-models/Qwen3.8-27B")
args = ap.parse_args()
dev = "cuda:0"
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as MQ
import json as J
tcfg_d = J.load(open(os.path.join(args.hf_dir, "config.json")))["text_config"]
tcfg = Qwen3_5TextConfig(**tcfg_d); tcfg._attn_implementation = "eager"
ltypes = tcfg.layer_types
R = C.Remote(args.hf_dir)
tr = torch.from_numpy(np.load(args.train)).long(); te = torch.from_numpy(np.load(args.test)).long()
ntr = len(tr)
toks = torch.cat([tr, te]); S = toks.shape[1]
rot = MQ.Qwen3_5TextRotaryEmbedding(tcfg).to(dev)
pos = torch.arange(S, device=dev)[None, None, :].expand(3, 1, -1)
pe = rot(torch.zeros(1, S, tcfg.hidden_size, device=dev), pos); pe = (pe[0].float(), pe[1].float())
causal = torch.full((S, S), torch.finfo(torch.float32).min, device=dev).triu(1)[None, None]
os.makedirs(os.path.dirname(args.state), exist_ok=True)
start = 0
res = {}
if os.path.exists(args.state):
    st = torch.load(args.state, map_location="cpu"); h = st["h"].to(dev); start = st["next"]; res = st["res"]; del st
    print("resume at layer", start, flush=True)
else:
    emb = R.fetch(C.LM + "embed_tokens.weight")
    h = emb[toks.view(-1)].float().to(dev).view(len(toks), S, -1); del emb
sample = [int(x) for x in args.layers.split(",")]
bits, group, m8 = args.bits, args.group, bool(args.meta8)
t0 = time.time()

def qspec(name, W, Hs):
    """generic spec  mode[:d=damp][:s=shrink][:b=bits][:g=group][:m8=0|1][:r=refit passes]  mode in rtn|gptq|static|ao"""
    H32, H128 = Hs
    parts = name.split(":")
    o = dict(d=0.01, s=0.0, b=bits, g=group, m8=int(m8), r=0)
    for kv in parts[1:]:
        k, v = kv.split("=")
        o[k] = float(v) if k in ("d", "s") else int(v)
    H = H128
    if parts[0] == "rtn":
        q, sc, m, rs = Q.rtn(W, o["b"], o["g"], clip=True, colw=torch.diagonal(H).clone(), meta8=bool(o["m8"]))
    elif parts[0] == "gptq":
        q, sc, m, rs = Q.gptq(W, Q.hinv_chol(H, o["d"]), o["b"], o["g"], clip=True, colw=torch.diagonal(H).clone(), meta8=bool(o["m8"]))
    else:
        q, sc, m, rs = Q.gptq_static(W, H, o["b"], o["g"], True, o["d"], actorder=parts[0] == "ao", meta8=bool(o["m8"]), shrink=o["s"])
    if o["r"]:
        sc, m = Q.refit_full(W, q, sc, m, rs, H, o["b"], o["g"], passes=o["r"])
    return Q.dequant_g(q, sc, m, o["g"], o["b"], rs)


def qvariant(name, W, Hs):
    if ":" in name:
        return qspec(name, W, Hs)
    H32, H128 = Hs
    if name == "rtn":
        q, s, m, rs = Q.rtn(W, bits, group, clip=True, colw=torch.diagonal(H128).clone(), meta8=m8)
    elif name in ("g32", "g128", "g128d5", "g128d10", "g128d30", "g128r"):
        H = H32 if name == "g32" else H128
        damp = {"g128d5": 0.05, "g128d10": 0.1, "g128d30": 0.3}.get(name, 0.01)
        q, s, m, rs = Q.gptq(W, Q.hinv_chol(H, damp), bits, group, clip=True, colw=torch.diagonal(H).clone(), meta8=m8)
        if name == "g128r":
            s, m = Q.refit_full(W, q, s, m, rs, H, bits, group)
    elif name in ("s128", "ao128", "ao128r", "ao32", "ao128d5", "ao128d10", "ao128d30", "ao128d100", "ao128s30", "ao128s50"):
        H = H32 if name == "ao32" else H128
        damp = {"ao128d5": 0.05, "ao128d10": 0.1, "ao128d30": 0.3, "ao128d100": 1.0}.get(name, 0.01)
        shr = {"ao128s30": 0.3, "ao128s50": 0.5}.get(name, 0.0)
        q, s, m, rs = Q.gptq_static(W, H, bits, group, True, damp, actorder=name != "s128", meta8=m8, shrink=shr)
        if name == "ao128r":
            s, m = Q.refit_full(W, q, s, m, rs, H, bits, group)
    else:
        raise ValueError(name)
    return Q.dequant_g(q, s, m, group, bits, rs)

with torch.no_grad():
    for i in range(start, max(sample) + 1):
        if time.time() - t0 > args.budget:
            torch.save(dict(h=h.cpu(), next=i, res=res), args.state + ".tmp"); os.replace(args.state + ".tmp", args.state)
            print("BUDGET stop at layer", i, flush=True); sys.exit(0)
        names = C.layer_names(i, ltypes[i])
        tensors = {n: R.fetch(n) for n in names} if True else None
        p = f"{C.LM}layers.{i}."
        sd = {}
        for n, t in tensors.items():
            k = n[len(p):]; sd[k] = t.float().to(dev)
            if k.endswith("conv1d.weight"): sd[k] = sd[k].view(-1, 1, t.shape[-1])
        layer = MQ.Qwen3_5DecoderLayer(tcfg, i).to(dev).float(); layer.load_state_dict(sd, strict=True); del sd
        lin = ltypes[i] == "linear_attention"
        mixer = layer.linear_attn if lin else layer.self_attn
        mods = {"a": mixer.in_proj_qkv if lin else mixer.q_proj, "o": mixer.out_proj if lin else mixer.o_proj,
                "m": layer.mlp.gate_proj, "d": layer.mlp.down_proj}
        issample = i in sample
        accs, hooks = {}, []
        if issample:
            for k, m in mods.items():
                accs[k] = {"32": C.HAcc(m.in_features if k != "d" else 1, dev), "128": C.HAcc(m.in_features, dev), "te": C.HAcc(m.in_features, dev)}
            cur = {"b": 0}
            def mk(k):
                def hook(mod, a):
                    x = a[0]; b0 = cur["b"]; n = x.shape[0]
                    for j in range(n):
                        si = b0 + j
                        if si < ntr:
                            accs[k]["128"].add(x[j])
                            if si % 4 == 0 and k != "d": accs[k]["32"].add(x[j])
                        else:
                            accs[k]["te"].add(x[j])
                return hook
            hooks = [m.register_forward_pre_hook(mk(k)) for k, m in mods.items()]
        mask = causal if not lin else None
        for b in range(0, len(h), 8):
            if issample: cur["b"] = b
            h[b:b + 8] = layer(h[b:b + 8], position_embeddings=pe, attention_mask=mask)
        for hk in hooks: hk.remove()
        if issample:
            h = h.cpu(); torch.cuda.empty_cache()
            Wm = {"a": [n for n in names if n.endswith(("in_proj_qkv.weight", "q_proj.weight"))][0],
                  "o": [n for n in names if n.endswith(("out_proj.weight", "o_proj.weight"))][0],
                  "m": p + "mlp.gate_proj.weight", "d": p + "mlp.down_proj.weight"}
            for k, wn in Wm.items():
                W = tensors[wn].to(dev).float()
                H128, Hte = accs[k]["128"].mean(), accs[k]["te"].mean()
                H32 = H128 if k == "d" else accs[k]["32"].mean()
                en = ((W @ Hte) * W).sum().item()
                for v in args.variants.split(","):
                    Wq = qvariant(v, W, (H32, H128))
                    e = (((W - Wq) @ Hte) * (W - Wq)).sum().item()
                    etr = (((W - Wq) @ H128) * (W - Wq)).sum().item() / ((W @ H128) * W).sum().item()
                    res.setdefault(str(i), {}).setdefault(k, {})[v] = dict(rel_test=e / en, rel_train128=etr)
                    print(f"[{time.time()-t0:5.0f}s] L{i} {k} {v:8s} test {e/en:.5f} train {etr:.5f}", flush=True)
                    del Wq
                del W, H32, H128, Hte
                torch.cuda.empty_cache()
            del accs
            h = h.to(dev)
            json.dump(dict(bits=bits, group=group, meta8=m8, res=res), open(args.out, "w"), indent=1)
        del layer, tensors; torch.cuda.empty_cache()
        print(f"[{time.time()-t0:5.0f}s] layer {i} done", flush=True)
    json.dump(dict(bits=bits, group=group, meta8=m8, res=res), open(args.out, "w"), indent=1)
    print("ALLDONE", flush=True)
