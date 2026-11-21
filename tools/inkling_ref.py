#!/usr/bin/env python3 -I
"""Inkling-2-Mini (inkling2_loop) text-only reference implementation in plain torch (CPU, fp32 compute).

Written from the reading of the (unmerged) HF PR huggingface/transformers#51050 (modeling_inkling2_loop.py,
conversion_mapping.py, core_model_loading.py at a10ebc3) and the checkpoint itself; see docs/INKLING_SPEC.md.
Weights are read straight from the NVFP4/MXFP8 checkpoint (lazy per-expert dequant to bf16, then fp32 matmul).
`emu_bf16=True` additionally rounds activations to bf16 at the module outputs where the HF bf16 model has bf16 tensors
(an approximation of the HF dtype flow, used only to measure bf16 sensitivity).

Usage as a library: m = Inkling(model_dir); logits = m.forward(ids, cache)  (cache = m.new_cache()).
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class Cfg:
    H = 2304; NL = 24; NH = 24; NKV = 8; HD = 128; DREL = 16; REXT = 1024; WIN = 1024
    EPS = 1e-6; K = 4; NE = 128; TOPK = 8; FF = 1152; ROUTE = 8.0
    LOG_N = 8192; LOG_A = 0.1; QKCLIP = 22.627416997969522; RELCLIP = 8.0
    BLK = 6; GATE_CAP = 4.0; UP_CAP = 25.0; LOGIT_CAP = 30.0; VOCAB = 200058
    LS, LE, NP, GR = 6, 18, 3, 64
    GLOBAL = (5, 11, 17, 23)


C = Cfg()
ABL = set()  # ablation switches used by tools/inkling_ablate.py (never set in normal runs)


def rb(x, emu):
    return x.to(torch.bfloat16).to(torch.float32) if emu else x


def rms_w(x, w, emu=False, eps=C.EPS):
    return rb(w.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)), emu)


def rms_u(x, eps=C.EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


class Store:
    def __init__(self, d):
        self.d = d
        self.f = safe_open(os.path.join(d, "model-00001-of-00001.safetensors"), "pt")
        self.cache = {}

    def get(self, name, dtype=torch.float32):
        k = (name, dtype)
        if k not in self.cache:
            self.cache[k] = self.f.get_tensor(name).to(dtype)
        return self.cache[k]

    def raw(self, name):
        return self.f.get_tensor(name)


class Inkling:
    def __init__(self, d, emu_bf16=False, expert_cache_gb=40.0, verbose=True):
        self.s = Store(d)
        self.emu = emu_bf16
        self.verbose = verbose
        self.ecache = {}
        self.ecache_bytes = 0
        self.ecache_limit = int(expert_cache_gb * 2**30)
        self.P = {}  # weight cache (fp32) for non-expert tensors
        self.trace = None  # dict -> filled by forward when not None
        self.sl = {}  # per-tensor safetensors slice handles
        self.last_k = 1  # number of final positions whose logits forward() returns when all_logits=False

    # ---------------- weights -----------------
    def w(self, name):
        if name not in self.P:
            self.P[name] = self.s.f.get_tensor(name).float()
        return self.P[name]

    def lw(self, L, n):
        return self.w(f"model.llm.layers.{L}.{n}")

    def expert(self, L, e):
        """Dequantized (w13 [2*FF,H] rows = [gate;up] after de-interleave, w2 [H,FF]) as bf16."""
        key = (L, e)
        if key in self.ecache:
            return self.ecache[key]
        f = self.s.f
        p = f"model.llm.layers.{L}.mlp.experts."
        out = []
        for nm in ("w13_weight", "w2_weight"):
            wt = f.get_slice(p + nm)[e:e + 1][0]
            sc = f.get_slice(p + nm + "_scale")[e:e + 1][0]
            R = wt.shape[0]
            if L < 20:  # NVFP4
                amax = f.get_slice(p + nm + "_amax")[e:e + 1][0].float()
                den = f.get_tensor(p + nm + "_outer_scale_denominator").float()
                wl = wt.long()
                vals = torch.stack([E2M1[wl & 0xF], E2M1[wl >> 4]], -1)  # [R, C/2, 2]; low nibble = even element
                C_ = wt.shape[1] * 2
                v = vals.reshape(R, C_ // 16, 16) * sc.float()[..., None] * (amax / den)
                v = v.reshape(R, C_)
            else:  # MXFP8 (e4m3 values, e8m0 scale per 32 along K)
                C_ = wt.shape[1]
                v = wt.float().reshape(R, C_ // 32, 32) * torch.exp2(sc.float() - 127.0)[..., None]
                v = v.reshape(R, C_)
            if nm == "w13_weight":
                v = torch.cat([v[0::2], v[1::2]], 0)  # even rows = gate, odd rows = up
            out.append(v.to(torch.bfloat16))
        self.ecache[key] = tuple(out)
        self.ecache_bytes += sum(t.numel() * 2 for t in out)
        if self.ecache_bytes > self.ecache_limit:  # evict oldest non-loop entries
            for k in list(self.ecache.keys()):
                if not (C.LS <= k[0] < C.LE):
                    t = self.ecache.pop(k)
                    self.ecache_bytes -= sum(x.numel() * 2 for x in t)
                    if self.ecache_bytes < 0.9 * self.ecache_limit:
                        break
        return self.ecache[key]

    # ---------------- cache -----------------
    def new_cache(self):
        return {"n": 0, "slots": {}}

    def slot(self, cache, L, p):
        return cache["slots"].setdefault((L, p), {"k": None, "v": None, "conv": {}})

    # ---------------- ops -----------------
    def conv_static(self, x, wconv, st, idx, scale_k):
        """x [n,D] fp32 ; wconv [D,1,K]; returns (conv(x)+x)/sqrt(K+1) ; st = slot dict (conv state per idx)."""
        n, D = x.shape
        prev = st["conv"].get(idx)
        if prev is None:
            prev = torch.zeros(C.K - 1, D)
        ctx = torch.cat([prev, x], 0)  # [K-1+n, D]
        st["conv"][idx] = ctx[-(C.K - 1):].clone()
        win = ctx.unfold(0, C.K, 1)  # [n, D, K]  ; win[...,j] = x[t-(K-1)+j]
        wk = wconv[:, 0, :].flip(-1) if 'conv_flip' in ABL else wconv[:, 0, :]
        out = (win * wk[None]).sum(-1)
        return (out + x) / math.sqrt(scale_k)

    def conv_dynamic(self, x, dyn_in, static_w, wproj, st, idx):
        n, D = x.shape
        nh = C.NKV
        dw = (dyn_in @ wproj.T).view(n, nh, 1, C.K)
        if 'no_dyn' in ABL:
            dw = dw * 0
        w = static_w.T.reshape(nh, D // nh, C.K)[None] + dw  # [n, nh, D/nh, K]; tap index = lag (0 = current token)
        prev = st["conv"].get(idx)
        if prev is None:
            prev = torch.zeros(C.K - 1, D)
        ctx = torch.cat([prev, x], 0)
        st["conv"][idx] = ctx[-(C.K - 1):].clone()
        win = ctx.unfold(0, C.K, 1).flip(-1)  # win[..., lag] = token `lag` steps back
        win = win.reshape(n, nh, D // nh, C.K)
        out = (win * w).sum(-1).reshape(n, D)
        return (out + x) / math.sqrt(2 * C.K + 1)

    def block_attn_res(self, blocks, partial, queries):
        """blocks [n,B,H], partial [n,H]; queries [Q,H] -> list of Q tensors [n,H] (value = normalized stack)."""
        st = torch.cat([blocks, partial[:, None, :]], 1)
        st = rms_u(st, 1e-6)
        sc = torch.einsum("qh,nbh->nqb", queries, st) * (C.H ** -0.5)
        p = torch.softmax(sc, -1)
        o = torch.einsum("nqb,nbh->nqh", p, st)
        return [o[:, i] for i in range(queries.shape[0])]

    def message_gate(self, L, which, h, msg):
        pre = f"model.llm.layers.{L}.controllers.{which}."
        g = torch.cat([rms_u(h), rms_u(msg)], -1)
        g = rb(rb(g, self.emu) @ self.w(pre + "down_weight").T, self.emu)  # HF: gate input cast to bf16 for down_proj
        fgt = torch.tanh(g @ self.w(pre + "forget_weight").T)
        mg = torch.tanh(g @ self.w(pre + "gain_weight").T)
        return (1 - fgt) * h + mg * msg

    def attention(self, L, p, cache, hq, hk, hv, pos):
        emu = self.emu
        glob = L in C.GLOBAL
        n = hq.shape[0]
        slot = self.slot(cache, L, p)
        a = f"model.llm.layers.{L}.attn."
        q = rb(hq @ self.w(a + "wq_du.weight").T, emu).view(n, C.NH, C.HD)
        kp = rb(hk @ self.w(a + "wk_dv.weight").T, emu)
        vp = rb(hv @ self.w(a + "wv_dv.weight").T, emu)
        if glob:
            ks = rb(self.conv_dynamic(kp, hk, self.w(a + "k_sconv.static_w"), self.w(a + "k_sconv.weight_proj"), slot, 0), emu)
            vs = rb(self.conv_dynamic(vp, hv, self.w(a + "v_sconv.static_w"), self.w(a + "v_sconv.weight_proj"), slot, 1), emu)
        else:
            ks = rb(self.conv_static(kp, self.w(a + "k_sconv.weight"), slot, 0, C.K + 1), emu)
            vs = rb(self.conv_static(vp, self.w(a + "v_sconv.weight"), slot, 1, C.K + 1), emu)
        rel = rb(hq @ self.w(a + "wr_du.weight").T, emu).view(n, C.NH, C.DREL)
        gate = hq @ self.w(a + "gate_attn.weight").T
        sink = hq @ self.w(a + "attn_sink_proj.weight").T  # [n,24]
        # q: scaled l2 clip (+ log scale)
        q = q * self.w(a + "q_l2_clip.scale")[None]
        q = q * (C.QKCLIP / (q.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
        q = rb(q, emu)
        rel = rel * self.w(a + "rel_l2_clip.scale")[None]
        rel = rel * (C.RELCLIP / (rel.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
        rel = rb(rel, emu)
        if glob and 'no_tau' not in ABL:
            tau = 1.0 + C.LOG_A * torch.log(((pos + 1).float() / C.LOG_N).clamp(min=1.0))
            q = rb(q * tau[:, None, None], emu)
            rel = rb(rel * tau[:, None, None], emu)
        k = ks.view(n, C.NKV, C.HD)
        k = rb(k * (C.QKCLIP / (k.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0), emu)
        v = vs.view(n, C.NKV, C.HD)
        v = rb(rms_u(v), emu)
        # kv cache
        slot["k"] = k if slot["k"] is None else torch.cat([slot["k"], k], 0)
        slot["v"] = v if slot["v"] is None else torch.cat([slot["v"], v], 0)
        K_, V_ = slot["k"], slot["v"]
        T = K_.shape[0]
        kpos = torch.arange(T)
        relw = rb(rel.permute(1, 0, 2) @ self.w(a + "rel_logits_proj.proj"), emu)  # [24,n,1024]
        scale = (1.0 / C.HD) if 'scale128' in ABL else C.HD ** -0.5
        outs = []
        QC = 512  # query chunking only bounds peak memory (rows are independent)
        for i0 in range(0, n, QC):
            i1 = min(n, i0 + QC)
            nc = i1 - i0
            dist = pos[i0:i1, None] - kpos[None, :]  # [nc,T]
            valid = dist >= 0
            if not glob:
                valid &= dist < C.WIN
            idx = dist.clamp(0, C.REXT - 1)[None].expand(C.NH, -1, -1)
            bias = relw[:, i0:i1].gather(-1, idx).masked_fill(((dist < 0) | (dist >= C.REXT))[None], 0.0)
            qh = q[i0:i1].view(nc, C.NKV, 3, C.HD).permute(1, 2, 0, 3).reshape(C.NKV, 3 * nc, C.HD)
            s = (qh @ K_.permute(1, 2, 0)).view(C.NKV, 3, nc, T).reshape(C.NH, nc, T) * scale
            s = s if 'no_rel' in ABL else s + bias
            s = s.masked_fill(~valid[None], float("-inf"))
            sk = sink[i0:i1].T[:, :, None] if 'no_sink' not in ABL else torch.full((C.NH, nc, 1), -1e30)
            s = torch.cat([s, sk], -1)
            w_ = torch.softmax(s, -1)[..., :T]
            oc = (w_.view(C.NKV, 3 * nc, T) @ V_.permute(1, 0, 2)).view(C.NKV, 3, nc, C.HD).permute(2, 0, 1, 3).reshape(nc, C.NH * C.HD)
            outs.append(oc)
        o = torch.cat(outs, 0)
        o = rb(o, emu)
        o = o if 'no_gate' in ABL else o * torch.sigmoid(gate)
        return rb(o @ self.w(a + "wo_ud.weight").T, emu)

    def moe(self, L, x, tr=None):
        emu = self.emu
        n = x.shape[0]
        logits = x @ self.lw(L, "mlp.gate.weight").T
        sc = torch.sigmoid(logits)
        bias = self.lw(L, "mlp.gate.bias")
        idx = torch.topk(sc + bias, C.TOPK, dim=-1, sorted=False).indices
        wts = sc.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * C.ROUTE
        if tr is not None:
            tr["router_idx"] = idx.clone()
            tr["router_w"] = wts.clone()
        out = torch.zeros(n, C.H)
        xb = rb(x, emu)
        flat = idx.reshape(-1)
        for e in torch.unique(flat).tolist():
            tok, slotpos = torch.where(idx == e)
            w13, w2 = self.expert(L, e)
            gu = rb(xb[tok] @ w13.float().T, emu)
            g, u = (gu[:, :C.FF], gu[:, C.FF:]) if 'swap_gu' not in ABL else (gu[:, C.FF:], gu[:, :C.FF])
            act = softcap(g, C.GATE_CAP) * torch.sigmoid(g) * softcap(u, C.UP_CAP)
            act = rb(act, emu)
            y = rb(act @ w2.float().T, emu)
            out.index_add_(0, tok, y * wts[tok, slotpos][:, None])
        return rb(out, emu)

    def layer(self, L, p, h, blocks, cache, pos, messages=None, tr=None):
        """One decoder-layer application. h [n,H] fp32, blocks [n,B,H]. Returns new h (blocks updated by caller)."""
        emu = self.emu
        glob = L in C.GLOBAL
        pre = f"model.llm.layers.{L}."
        res_blk = h
        boundary = (L % C.BLK == 0)
        slot = self.slot(cache, L, p)
        if glob:
            if L > 0:
                qs = self.w(f"model.llm.attn_res.attn_queries.{L - 1}")
                hq, hk, hv = self.block_attn_res(blocks, h, qs)
            else:
                hq = hk = hv = h
            if messages is not None:
                hq = self.message_gate(L, "attn", hq, messages[:, 0])
                hk = self.message_gate(L, "k", hk, messages[:, 1])
                hv = self.message_gate(L, "v", hv, messages[:, 2])
            hq = rms_w(hq, self.lw(L, "q_input_norm.weight"), emu)
            hk = rms_w(hk, self.lw(L, "k_input_norm.weight"), emu)
            hv = rms_w(hv, self.lw(L, "v_input_norm.weight"), emu)
        else:
            if L > 0:
                h_in = self.block_attn_res(blocks, h, self.w(f"model.llm.attn_res.attn_queries.{L - 1}")[None])[0]
            else:
                h_in = h
            if messages is None:
                hq = hk = hv = rms_w(h_in, self.lw(L, "attn_norm.weight"), emu)
            else:
                nw = self.lw(L, "attn_norm.weight")
                hq = rms_w(self.message_gate(L, "attn", h_in, messages[:, 0]), nw, emu)
                hk = rms_w(self.message_gate(L, "k", h_in, messages[:, 1]), nw, emu)
                hv = rms_w(self.message_gate(L, "v", h_in, messages[:, 2]), nw, emu)
        a_out = self.attention(L, p, cache, hq, hk, hv, pos)
        a_out = rb(self.conv_static(a_out, self.lw(L, "attn_sconv.weight"), slot, 2, C.K + 1), emu)
        a_out = a_out.float()
        if boundary:
            blocks = torch.cat([blocks, res_blk[:, None, :]], 1)
            residual = hmid = a_out if 'no_reset' not in ABL else res_blk + a_out
        else:
            residual = hmid = res_blk + a_out
        hm = self.block_attn_res(blocks, hmid, self.w(f"model.llm.attn_res.mlp_queries.{L}")[None])[0]
        if messages is not None:
            hm = self.message_gate(L, "mlp", hm, messages[:, 3])
        xm = rms_w(hm, self.lw(L, "mlp_norm.weight"), emu)
        if tr is not None:
            tr["mlp_in"] = xm.clone()
        m_out = self.moe(L, xm, tr)
        m_out = rb(self.conv_static(m_out, self.lw(L, "mlp_sconv.weight"), slot, 3, C.K + 1), emu)
        return rb(residual + m_out, emu)

    # ---------------- model -----------------
    def forward(self, ids, cache, want_trace=False, all_logits=False):
        t0 = time.time()
        emu = self.emu
        n = len(ids)
        pos = torch.arange(cache["n"], cache["n"] + n)
        emb = self.s.f.get_slice("model.llm.embed.weight")
        x = torch.stack([emb[int(i):int(i) + 1][0] for i in ids]).float()
        h = rms_w(x, self.w("model.llm.embed_norm.weight"), emu)
        trace = {"emb": h.clone()} if want_trace else None
        blocks = h.new_zeros(n, 0, C.H)
        app = 0

        def run(L, p, h, blocks, messages=None):
            nonlocal app
            tr = {} if want_trace else None
            new_h = self.layer(L, p, h, blocks, cache, pos, messages, tr)
            if L % C.BLK == 0:
                blocks = torch.cat([blocks, h[:, None, :]], 1)
            if want_trace:
                trace[f"app{app:02d}_L{L}_p{p}"] = new_h.clone()
                trace[f"app{app:02d}_L{L}_p{p}_router"] = tr["router_idx"].clone()
            app += 1
            return new_h, blocks

        for L in range(C.LS):
            h, blocks = run(L, 0, h, blocks)
        pre_h, pre_blocks = h, blocks
        npre = pre_blocks.shape[1]
        for L in range(C.LS, C.LE):
            h, blocks = run(L, 0, h, blocks)
        for pss in range(1, C.NP):
            prev_blocks = blocks[:, npre + 1:, :]
            prev_h = h
            h, blocks = pre_h, pre_blocks
            q = self.w("model.llm.middle_cycle_cross_pass_queries").view(C.LE - C.LS, 2, C.H)
            kvq = self.w("model.llm.middle_cycle_cross_pass_kv_queries").view(C.LE - C.LS, 2, C.H)
            for li, L in enumerate(range(C.LS, C.LE)):
                mq = torch.stack([q[li, 0], kvq[li, 0], kvq[li, 1], q[li, 1]])  # [query, key, value, mlp]
                msgs = torch.stack(self.block_attn_res(prev_blocks, prev_h, mq), 1)  # [n,4,H]
                h, blocks = run(L, pss, h, blocks, None if 'no_msgs' in ABL else msgs)
        for L in range(C.LE, C.NL):
            h, blocks = run(L, 0, h, blocks)
        hf = self.block_attn_res(blocks, h, self.w("model.llm.attn_res.output_query")[None])[0]
        hf = rms_w(hf, self.w("model.llm.norm.weight"), emu)
        if want_trace:
            trace["final_norm"] = hf.clone()
        cache["n"] += n
        hs = hf if all_logits else hf[-self.last_k:]
        logits = self.unembed(hs)
        if self.verbose:
            print(f"  forward n={n} {time.time() - t0:.1f}s (expert cache {self.ecache_bytes / 2**30:.1f} GiB)", flush=True)
        return (logits, trace) if want_trace else logits

    def unembed(self, hs):
        if "unembed" not in self.P:
            self.P["unembed"] = self.s.f.get_slice("model.llm.unembed.weight")[0:C.VOCAB].float()
        lg = hs @ self.P["unembed"].T
        return softcap(lg, C.LOGIT_CAP)
