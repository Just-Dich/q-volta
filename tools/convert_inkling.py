#!/usr/bin/env python3 -I
"""Inkling-2-Mini -> Q-Volta .qv converter (docs/INKLING_FORMAT.md). Shares tools/qv_quant.py / tools/qv_format.py with
tools/convert.py; written straight into a single .qv container (tools/qv_format.ContainerWriter).

Profiles (--profile, each field can be overridden):
  native : experts copied bit-exactly as checkpoint-native NVFP4 (layers 0-19) / MXFP8 (20-23) tensors (kinds "nvfp4"/"mxfp8"),
           dense fp16/fp32. ~17.4 GiB: only useful with --layers (test files) on this 19 GB-free disk.
  q4     : experts 4-bit g128 meta8, GPTQ (+clip) from the calibration activations; dense matrices 4-bit g64 meta8 RTN+clip
  q3     : experts 3-bit g128 meta8 GPTQ; dense as q4
Experts are requantised from the exact dequantisation (rounded to bf16 as tools/inkling_ref.py does). Stacked layout: tensor
`model.llm.layers.L.mlp.experts.w13_weight` [128,2304,2304] (rows interleaved gate/up as in the checkpoint) and `w2_weight`
[128,2304,1152]. Hessians: per-expert X_e^T X_e from the calibration dump of tools/inkling_calib.py (routed tokens only),
shrunk toward the layer-pooled Hessian: H = (n_e H_e + c H_pool) / (n_e + c) (--prior c, c=0: pure per-expert).

  python3 -I tools/convert_inkling.py --profile q4 --out /path/x.qv --calib DIR [--layers 0:24] [--max-seconds 480]
Resumable: the run stops after --max-seconds (checked between layers) and continues from <out>.state.json on the next call.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_format as F
import qv_quant as Q
import inkling_ckpt as K

PROFILES = {
    "native": dict(expert="native", dense_bits=0),
    "q4": dict(expert="q", expert_bits=4, expert_group=128, method="gptq", dense_bits=4, dense_group=64),
    "q3": dict(expert="q", expert_bits=3, expert_group=128, method="gptq", dense_bits=4, dense_group=64),
}
QDENSE = re.compile(r"\.attn\.(wq_du|wk_dv|wv_dv|wo_ud|gate_attn)\.weight$")
F16_MAT = re.compile(r"\.attn\.(wr_du|attn_sink_proj)\.weight$|\.mlp\.gate\.weight$|controllers\.|k_sconv\.weight_proj$|v_sconv\.weight_proj$")
FP16_MIN_NORMAL = 6.103515625e-05


def dense_kind(name, shape):
    """'qrows' (embed/unembed), 'q' (big attention matrices), 'f16' (small matrices) or 'f32' (vectors, convs, queries)."""
    if re.search(r"\.(embed|unembed)\.weight$", name):
        return "qrows"
    if QDENSE.search(name):
        return "q"
    if F16_MAT.search(name):
        return "f16"
    return "f32"


def fp16_report(t):
    """range statistics of a tensor if stored as fp16 (the source is bf16/fp32)."""
    a = t.float()
    nz = a != 0
    absa = a.abs()
    h = a.half().float()
    sub = nz & (absa < FP16_MIN_NORMAL)
    return dict(max_abs=float(absa.max()), n_overflow=int((absa > 65504).sum()), n_subnormal=int(sub.sum()), n=a.numel(),
                rel_err=float((h - a).norm() / (a.norm() + 1e-30)))


def quant_dense(W, bits, group, dev, rows=16384):
    """W [N,K] -> (packed, sm, rs, stats) RTN+clip, meta8, in row chunks; stats = relative RMS weight error."""
    pk, sm, rs, err2, en2 = [], [], [], 0.0, 0.0
    for r0 in range(0, W.shape[0], rows):
        Wd = W[r0:r0 + rows].to(dev).float()
        q, S, M, r = Q.rtn(Wd, bits, group, clip=True, meta8=True)
        dq = Q.dequant_g(q, S, M, group, bits, r)
        err2 += float(((dq - Wd) ** 2).sum()); en2 += float((Wd ** 2).sum())
        pk.append(F.pack(q, bits).cpu()); sm.append(Q.pack_meta(S, M, r).cpu()); rs.append(r.cpu())
    return torch.cat(pk), torch.cat(sm), torch.cat(rs), dict(rel_rms=(err2 / en2) ** 0.5)


def load_calib(calib_dir, names, L):
    """(X fp16 [Nt,2304], I int8 [Nt,8]) of layer L concatenated over the given sequences and passes."""
    xs, ids = [], []
    for nm in names:
        z = np.load(os.path.join(calib_dir, f"seq_{nm}.npz"))
        for p in range(3):
            if f"x_L{L}_p{p}" in z.files:
                xs.append(z[f"x_L{L}_p{p}"]); ids.append(z[f"i_L{L}_p{p}"])
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ids))


def bf16r(t):
    return t.to(torch.bfloat16).float()


def routed_rows(I, e0, e1):
    """per expert e in [e0,e1): index tensor of token rows routed to e"""
    return [torch.nonzero((I == e).any(1)).flatten() for e in range(e0, e1)]


def shrink(Hs, n, Hp, c):
    """Hs [B,K,K] = per-expert X^T X / n (mean second moment), n [B], Hp pooled mean; -> shrunk Hessians."""
    if c == float("inf"):
        return Hp[None].expand_as(Hs).clone()
    n = n.float()[:, None, None]
    return (n * Hs + c * Hp[None]) / (n + c).clamp_min(1e-9)


def quant_layer_experts(f, L, X, I, a, dev, log, val=None):
    """-> dict name -> (packed [E,N,C,bits] int32 cpu, sm cpu, rs cpu), stats dict."""
    bits, grp, method, c = a.expert_bits, a.expert_group, a.method, a.prior
    E = K.NE
    Eb = a.expert_batch
    Xg = X.to(dev).float()
    Nt = Xg.shape[0]
    Hp13 = (Xg.T @ Xg) / Nt if method == "gptq" else None
    rows = routed_rows(I.to(dev), 0, E)
    cnt = torch.tensor([len(r) for r in rows])
    out = {}
    st = dict(n_tokens=Nt, count=cnt.tolist())
    out13 = [[], [], []]
    H2s, w2err = [], []
    e13 = dict(wnum=0.0, wden=0.0, onum=0.0, oden=0.0, vnum=0.0, vden=0.0)
    Xv = val[0].to(dev).float() if val else None
    rowsv = routed_rows(val[1].to(dev), 0, E) if val else None
    Av = {}  # held-out expert activations (exact w13) for the w2 held-out error
    t0 = time.time()
    # ---- pass A: w13 (needs only X) and the per-expert activations for the w2 Hessians
    for e0 in range(0, E, Eb):
        e1 = min(E, e0 + Eb)
        w13, w2 = K.dequant_experts(f, L, e0, e1, dev)
        w13 = bf16r(w13); del w2
        B = e1 - e0
        Hm = torch.zeros(B, K.H, K.H, device=dev)
        H2 = torch.zeros(B, K.FF, K.FF, device=dev)
        for b in range(B):
            r = rows[e0 + b]
            if len(r) == 0:
                continue
            xe = Xg[r]
            Hm[b] = (xe.T @ xe) / len(r)
            gu = xe @ w13[b].T
            act = K.moe_act(gu[:, 0::2], gu[:, 1::2])
            H2[b] = (act.T @ act) / len(r)
        n = cnt[e0:e1].to(dev)
        H2s.append((H2.cpu(), n.cpu()))
        if method == "gptq":
            Hs = shrink(Hm, n, Hp13, c)
            colw = torch.diagonal(Hs, dim1=-2, dim2=-1).clone()
            Hu = Q.hinv_chol_batched(Hs)
            q, S, M, rs = Q.gptq_batched(w13, Hu, bits, grp, clip=True, colw=colw, meta8=True)
            del Hu, Hs
        else:
            q, S, M, rs = Q.rtn_batched(w13, bits, grp, clip=a.clip, meta8=True)
        dq = torch.stack([Q.dequant_g(q[b], S[b], M[b], grp, bits, rs[b]) for b in range(B)])
        D = dq - w13
        e13["wnum"] += float((D ** 2).sum()); e13["wden"] += float((w13 ** 2).sum())
        if val:  # held-out (validation sequence) relative output error ||(W-Wq) x||^2 / ||W x||^2
            for b in range(B):
                rv = rowsv[e0 + b]
                if len(rv):
                    xv = Xv[rv]
                    e13["vnum"] += float(((xv @ D[b].T) ** 2).sum()); e13["vden"] += float(((xv @ w13[b].T) ** 2).sum())
                    gu = xv @ w13[b].T
                    Av[e0 + b] = K.moe_act(gu[:, 0::2], gu[:, 1::2])
        for b in range(B):  # output error on the calibration tokens of this expert (exact Hessian of the routed tokens)
            if cnt[e0 + b] > 0:
                e13["onum"] += float(((D[b] @ Hm[b]) * D[b]).sum()) * int(cnt[e0 + b])
                e13["oden"] += float(((w13[b] @ Hm[b]) * w13[b]).sum()) * int(cnt[e0 + b])
        out13[0].append(F.pack(q.view(B * 2 * K.FF, K.H).contiguous(), bits).view(B, 2 * K.FF, K.H // 32, bits).cpu())
        out13[1].append(Q.pack_meta(S, M, rs).cpu()); out13[2].append(rs.cpu())
        del w13, q, S, M, rs, dq, D, Hm, H2
        torch.cuda.empty_cache()
    out["w13_weight"] = tuple(torch.cat(x) for x in out13)
    # ---- pass B: w2 with Hessians of the activations
    Hp2 = None
    if method == "gptq":
        tot = sum(int(n.sum()) for _, n in H2s)
        Hp2 = sum((H * n.float()[:, None, None]).sum(0) for H, n in H2s).to(dev) / max(tot, 1)
    out2 = [[], [], []]
    e2 = dict(wnum=0.0, wden=0.0, onum=0.0, oden=0.0, vnum=0.0, vden=0.0)
    for bi, e0 in enumerate(range(0, E, Eb)):
        e1 = min(E, e0 + Eb)
        _, w2 = K.dequant_experts(f, L, e0, e1, dev)
        w2 = bf16r(w2)
        B = e1 - e0
        Hm, n = H2s[bi][0].to(dev), H2s[bi][1].to(dev)
        if method == "gptq":
            Hs = shrink(Hm, n, Hp2, c)
            colw = torch.diagonal(Hs, dim1=-2, dim2=-1).clone()
            Hu = Q.hinv_chol_batched(Hs)
            q, S, M, rs = Q.gptq_batched(w2, Hu, bits, grp, clip=True, colw=colw, meta8=True)
            del Hu, Hs
        else:
            q, S, M, rs = Q.rtn_batched(w2, bits, grp, clip=a.clip, meta8=True)
        dq = torch.stack([Q.dequant_g(q[b], S[b], M[b], grp, bits, rs[b]) for b in range(B)])
        D = dq - w2
        e2["wnum"] += float((D ** 2).sum()); e2["wden"] += float((w2 ** 2).sum())
        for b in range(B):
            if (e0 + b) in Av:
                e2["vnum"] += float(((Av[e0 + b] @ D[b].T) ** 2).sum()); e2["vden"] += float(((Av[e0 + b] @ w2[b].T) ** 2).sum())
            if n[b] > 0:
                e2["onum"] += float(((D[b] @ Hm[b]) * D[b]).sum()) * int(n[b])
                e2["oden"] += float(((w2[b] @ Hm[b]) * w2[b]).sum()) * int(n[b])
        out2[0].append(F.pack(q.view(B * K.H, K.FF).contiguous(), bits).view(B, K.H, K.FF // 32, bits).cpu())
        out2[1].append(Q.pack_meta(S, M, rs).cpu()); out2[2].append(rs.cpu())
        del w2, q, S, M, rs, dq, D, Hm
        torch.cuda.empty_cache()
    out["w2_weight"] = tuple(torch.cat(x) for x in out2)
    st["w13_rel_rms"] = (e13["wnum"] / e13["wden"]) ** 0.5
    st["w2_rel_rms"] = (e2["wnum"] / e2["wden"]) ** 0.5
    st["w13_out_rel"] = (e13["onum"] / max(e13["oden"], 1e-30)) ** 0.5
    st["w2_out_rel"] = (e2["onum"] / max(e2["oden"], 1e-30)) ** 0.5
    if val:
        st["w13_val_rel"] = (e13["vnum"] / max(e13["vden"], 1e-30)) ** 0.5
        st["w2_val_rel"] = (e2["vnum"] / max(e2["vden"], 1e-30)) ** 0.5
    st["seconds"] = time.time() - t0
    return out, st


def native_layer_experts(f, L, w):
    """checkpoint-native passthrough: nvfp4 (L<20) / mxfp8 (L>=20), bit-exact copies."""
    kind = K.expert_kind(L)
    nb = 0
    for nm in ("w13_weight", "w2_weight"):
        p = f"model.llm.layers.{L}.mlp.experts.{nm}"
        codes = f.get_tensor(p).numpy()
        sc = f.get_tensor(p + "_scale").view(torch.uint8).numpy()
        if kind == "nvfp4":
            g = (f.get_tensor(p + "_amax").float() / f.get_tensor(p + "_outer_scale_denominator").float()).numpy().astype(np.float32)
            shape = [codes.shape[0], codes.shape[1], codes.shape[2] * 2]
            nb += w.add_raw(p, "nvfp4", shape, dict(q=codes, sm=sc, rs=g), group=16, bits=4)
        else:
            nb += w.add_raw(p, "mxfp8", list(codes.shape), dict(q=codes.view(np.uint8) if codes.dtype != np.uint8 else codes, sm=sc), group=32, bits=8)
    return nb


def inkling_config(model_dir, args, prof):
    c = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    lc = c["loop_config"]
    glob = [i for i in range(c["num_hidden_layers"]) if i not in c["local_layer_ids"]]
    return {
        "arch": "inkling2_loop",
        "hidden_size": c["hidden_size"], "num_hidden_layers": c["num_hidden_layers"], "vocab_size": c["vocab_size"],
        "num_attention_heads": c["num_attention_heads"], "num_key_value_heads": c["num_key_value_heads"],
        "head_dim": c["head_dim"], "rms_norm_eps": c["rms_norm_eps"], "eos_token_id": 200006,
        "inkling": {
            "unpadded_vocab": c["unpadded_vocab_size"], "loop_start": lc["core_start"], "loop_end": lc["core_end"],
            "loop_passes": lc["num_passes"], "loop_gate_rank": lc["gate_rank"],
            "global_layers": glob, "local_layers": c["local_layer_ids"], "window": c["sliding_window_size"],
            "d_rel": c["d_rel"], "rel_extent": c["rel_extent"], "qk_l2_clip": c["qk_l2_clip_threshold"],
            "rel_l2_clip": c["rel_l2_clip_threshold"], "log_scaling_n_floor": c["log_scaling_n_floor"],
            "log_scaling_alpha": c["log_scaling_alpha"], "attn_res_block": c["n_layers_per_attn_res_block"],
            "sconv_kernel": c["sconv_kernel_size"], "n_experts": c["n_routed_experts"], "topk": c["num_experts_per_tok"],
            "n_shared_experts": c["n_shared_experts"], "expert_ff": c["intermediate_size"], "route_scale": c["route_scale"],
            "gate_softcap": c["mlp_gate_softcap"], "up_softcap": c["mlp_up_softcap"],
            "final_logit_softcap": c["final_logit_softcapping"], "w13_layout": "interleaved",
            "max_position": c["model_max_length"],
        },
        "quant": {"profile": args.profile, "expert": prof.get("expert"), "expert_bits": prof.get("expert_bits"),
                  "expert_group": prof.get("expert_group"), "method": prof.get("method"), "prior": args.prior,
                  "dense_bits": prof.get("dense_bits"), "dense_group": prof.get("dense_group"), "meta": "s8z8",
                  "calib": args.calib_seqs, "layers": [args.l0, args.l1]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=K.MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="q4", choices=list(PROFILES))
    ap.add_argument("--calib", default="/workspace/q-volta-models/scratch-inkling/calib")
    ap.add_argument("--calib-seqs", default="wiki0,wiki1,code0,chat0")
    ap.add_argument("--layers", default="0:24", help="a:b layer range of expert tensors (test files); dense tensors of those layers only")
    ap.add_argument("--expert-bits", type=int); ap.add_argument("--expert-group", type=int)
    ap.add_argument("--method", choices=["rtn", "gptq"]); ap.add_argument("--dense-bits", type=int); ap.add_argument("--dense-group", type=int)
    ap.add_argument("--prior", type=float, default=1024.0, help="Hessian shrinkage prior sample count c (inf = pooled only)")
    ap.add_argument("--clip", action="store_true", default=True)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--no-vocab-trim", action="store_true")
    a = ap.parse_args()
    prof = dict(PROFILES[a.profile])
    for k in ("expert_bits", "expert_group", "method", "dense_bits", "dense_group"):
        if getattr(a, k) is not None:
            prof[k] = getattr(a, k)
    for k, v in prof.items():
        if k in ("expert_bits", "expert_group", "method", "dense_group"):
            setattr(a, k, v)
    a.l0, a.l1 = (int(x) for x in a.layers.split(":"))
    dev = "cuda"
    f = K.open_ckpt(a.model)
    state_path = a.out + ".state.json"
    t_start = time.time()
    if os.path.exists(state_path) and os.path.exists(a.out):
        st = json.load(open(state_path))
        w = F.ContainerWriter(a.out, resume=(st["off"], st["tensors"]))
    else:
        st = dict(done_dense=False, done_layers=[], stats={}, dense_report={}, dense_stats={}, off=0, tensors={})
        w = F.ContainerWriter(a.out)

    def save():
        w.f.flush()
        st.update(off=w.off, tensors=w.tensors)
        json.dump(st, open(state_path + ".tmp", "w")); os.replace(state_path + ".tmp", state_path)

    names = [k for k in f.keys() if k.startswith("model.llm.") and "mlp.experts" not in k]
    def in_range(n):
        m = re.search(r"layers\.(\d+)\.", n)
        return m is None or a.l0 <= int(m.group(1)) < a.l1
    names = sorted(n for n in names if in_range(n))
    uv = 200058
    if not st["done_dense"]:
        for n in names:
            t = f.get_tensor(n)
            kd = dense_kind(n, t.shape)
            if kd in ("qrows", "q") and prof["dense_bits"] == 0:
                kd = "f16"
            if re.search(r"\.(embed|unembed)\.weight$", n) and not a.no_vocab_trim:
                t = t[:uv]
            if kd in ("qrows", "q"):
                pk, sm, rs, s = quant_dense(t, prof["dense_bits"], prof["dense_group"], dev)
                w.add_quant(n, pk, sm, prof["dense_bits"], list(t.shape), prof["dense_group"], rs)
                st["dense_stats"][n] = s
            elif kd == "f16":
                st["dense_report"][n] = fp16_report(t)
                w.add_dense(n, t, "f16")
            else:
                if t.dim() == 3:  # depthwise conv kernels [D,1,K] -> [D,K]
                    t = t.reshape(t.shape[0], t.shape[-1])
                w.add_dense(n, t, "f32")
            if kd == "f16" or kd == "f32":
                st["dense_report"].setdefault(n, fp16_report(t)); st["dense_report"][n]["stored"] = kd
        st["done_dense"] = True
        save()
        print(f"dense done {time.time() - t_start:.0f}s, {w.off / 2**30:.2f} GiB", flush=True)
    for L in range(a.l0, a.l1):
        if L in st["done_layers"]:
            continue
        if time.time() - t_start > a.max_seconds:
            print("time budget reached, resume later", flush=True)
            save()
            return 3
        if prof["expert"] == "native":
            native_layer_experts(f, L, w)
            sts = {}
        else:
            X, I = load_calib(a.calib, a.calib_seqs.split(","), L)
            val = load_calib(a.calib, ["val0"], L) if os.path.exists(os.path.join(a.calib, "seq_val0.npz")) else None
            res, sts = quant_layer_experts(f, L, X, I, a, dev, print, val)
            for nm, (pk, sm, rs) in res.items():
                shp = [K.NE, 2 * K.FF, K.H] if nm == "w13_weight" else [K.NE, K.H, K.FF]
                w.add_quant(f"model.llm.layers.{L}.mlp.experts.{nm}", pk, sm, a.expert_bits, shp, a.expert_group, rs)
            del res
        st["stats"][str(L)] = sts
        st["done_layers"].append(L)
        save()
        print(f"layer {L}: {json.dumps({k: v for k, v in sts.items() if k != 'count'})} total {w.off / 2**30:.2f} GiB t={time.time() - t_start:.0f}s", flush=True)
    cfg = inkling_config(a.model, a, prof)
    w.close(cfg)
    json.dump(dict(stats=st["stats"], dense_stats=st["dense_stats"], dense_report=st["dense_report"]), open(a.out + ".report.json", "w"), indent=1)
    os.remove(state_path)
    print(f"FINISHED {a.out}: {os.path.getsize(a.out) / 2**30:.3f} GiB ({os.path.getsize(a.out)} B)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
