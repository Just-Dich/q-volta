#!/usr/bin/env python3 -I
"""Torch reference of the Inkling-2-Mini MTP ("DSpark") drafter, written from vLLM's
vllm/models/inkling2/nvidia/dspark.py (+ ops/dspark.py, speculator.py); see docs/INKLING_MTP.md.
fp32 compute from the bf16 weights.

  python3 -I tools/inkling_mtp_ref.py --feats f_code4k.npz [--stride 8] [--variant NAME]

Evaluates draft acceptance for anchors p: anchor token x_p (the token sampled from the target at p-1, not yet run through the
target), context = target features at positions 0..p-1; draft d_1..d_7 predicts tokens p+1..p+7. Reference for acceptance:
target argmax chain `argmax[p..p+6]` of the saved sequence (exact greedy only inside the generated continuation; inside a
teacher-forced text it is the target's prediction GIVEN THE TEXT prefix, see the doc).
"""
import argparse, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MD = "/workspace/q-volta-models/Inkling-2-Mini"
H, NH, NKV, HD, EPS, BLOCK, RANK = 2304, 24, 8, 128, 1e-6, 8, 256
LOCAL = (0, 2)
WIN = 1024
CAP = 30.0
VOCAB = 200058


def rms(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y * w if w is not None else y


def rope_tables(npos, theta=1e6):
    inv = theta ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = torch.arange(npos, dtype=torch.float64)[:, None] * inv[None]
    return ang.cos().float(), ang.sin().float()  # [npos, 64]


def rope(x, cos, sin):
    # x [..., 128], cos/sin broadcastable [..., 64]; neox half-split
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class Drafter:
    def __init__(self, md=MD, variant=None):
        f = safe_open(os.path.join(md, "mtp.safetensors"), "pt")
        self.W = {k: f.get_tensor(k).float() for k in f.keys()}
        self.s = safe_open(os.path.join(md, "model-00001-of-00001.safetensors"), "pt")
        self.embed_norm = self.s.get_tensor("model.llm.embed_norm.weight").float()
        self.unembed = self.s.get_slice("model.llm.unembed.weight")[0:VOCAB].float()  # [V,H]
        self.embed = self.s.get_slice("model.llm.embed.weight")
        self.alpha = torch.softmax(self.W["fuse_weight"], -1)  # [4 rows (3 layers + markov), 4 taps]
        self.variant = variant or ""
        self.cos, self.sin = rope_tables(1 << 17)
        mh = "markov_head."
        self.w1 = self.W[mh + "markov_w1.weight"]
        self.rnn_w, self.rnn_b = self.W[mh + "rnn_proj.weight"], self.W[mh + "rnn_proj.bias"]
        self.ho_w, self.ho_b = self.W[mh + "hidden_out.weight"], self.W[mh + "hidden_out.bias"]

    def mix(self, feat, row, normw):
        # feat [n,4,H] -> rmsnorm(sum_t alpha[row,t]*feat[:,t]) * normw
        mixed = (feat * self.alpha[row][None, :, None]).sum(1)
        return rms(mixed, normw)

    def ctx_kv(self, feat):
        """context K/V per draft layer for all positions: list of (k [n,8,128] roped, v [n,8,128])."""
        n = feat.shape[0]
        out = []
        pos = torch.arange(n)
        for i in range(3):
            P = f"layers.{i}."
            c = self.mix(feat, i, self.W[P + "dflare_norm.weight"])
            k = (c @ self.W[P + "self_attn.k_proj_ctx.weight"].T).view(n, NKV, HD)
            v = (c @ self.W[P + "self_attn.v_proj_ctx.weight"].T).view(n, NKV, HD)
            k = rms(k, self.W[P + "self_attn.k_norm_ctx.weight"])
            k = rope(k, self.cos[pos][:, None], self.sin[pos][:, None])
            out.append((k, v))
        return out

    def draft(self, ckv, feat, ids, anchors):
        """anchors: list of positions p (>=1); ids[p] is the anchor token; context = positions < p.
        Returns tokens [B,7] (greedy) and base-logit argmax [B,7] (without markov bias)."""
        B = len(anchors)
        n = feat.shape[0]
        p = torch.tensor(anchors)
        emb = torch.stack([self.embed[int(ids[a]):int(ids[a]) + 1][0] for a in anchors]).float()
        x = torch.empty(B, BLOCK, H)
        x[:] = self.W["mask_embed"]
        x[:, 0] = rms(emb, self.embed_norm)
        qpos = p[:, None] + torch.arange(BLOCK)[None]  # [B,8]
        cos, sin = self.cos[qpos][:, :, None], self.sin[qpos][:, :, None]
        kpos = torch.arange(n)[None, :]  # context key positions
        for i in range(3):
            P = f"layers.{i}."
            res = x
            xn = rms(x, self.W[P + "input_layernorm.weight"])
            q = (xn @ self.W[P + "self_attn.q_proj.weight"].T).view(B, BLOCK, NH, HD)
            k = (xn @ self.W[P + "self_attn.k_proj.weight"].T).view(B, BLOCK, NKV, HD)
            v = (xn @ self.W[P + "self_attn.v_proj.weight"].T).view(B, BLOCK, NKV, HD)
            q = rope(rms(q, self.W[P + "self_attn.q_norm.weight"]), cos, sin)
            k = rope(rms(k, self.W[P + "self_attn.k_norm.weight"]), cos, sin)
            kc, vc = ckv[i]
            # scores vs context keys
            g = NH // NKV
            kcx = kc.repeat_interleave(g, 1)  # [n,24,128]
            vcx = vc.repeat_interleave(g, 1)
            kb = k.repeat_interleave(g, 2)  # [B,8,24,128]
            vb = v.repeat_interleave(g, 2)
            sc_c = torch.einsum("bqhd,khd->bqhk", q, kcx) * HD ** -0.5  # [B,8,24,n]
            ok = kpos < p[:, None]
            if i in LOCAL:
                ok = ok & (kpos >= p[:, None] - WIN)
            sc_c = sc_c.masked_fill(~ok[:, None, None, :], float("-inf"))
            sc_b = torch.einsum("bqhd,bkhd->bqhk", q, kb) * HD ** -0.5  # [B,8,24,8]
            sc = torch.cat([sc_c, sc_b], -1).softmax(-1)
            o = torch.einsum("bqhk,khd->bqhd", sc[..., :n], vcx) + torch.einsum("bqhk,bkhd->bqhd", sc[..., n:], vb)
            x = res + o.reshape(B, BLOCK, NH * HD) @ self.W[P + "self_attn.o_proj.weight"].T
            res = x
            xn = rms(x, self.W[P + "post_attention_layernorm.weight"])
            mlp = (F.silu(xn @ self.W[P + "mlp.gate_proj.weight"].T) * (xn @ self.W[P + "mlp.up_proj.weight"].T)) \
                @ self.W[P + "mlp.down_proj.weight"].T
            x = res + mlp
        hid = rms(x, self.W["norm.weight"])  # [B,8,H]
        hs = hid[:, 1:]  # masks at p+1..p+7 predict tokens p+1..p+7
        base = hs @ self.unembed.T  # [B,7,V]
        # markov head
        f_rnn = self.mix(feat[p - 1], 3, self.W["markov_head.dflare_norm.weight"])  # features of last context position
        if "no_frnn" in self.variant:
            f_rnn = torch.zeros_like(f_rnn)
        wa, wb = self.rnn_w[:, :2 * RANK], self.rnn_w[:, 2 * RANK:]
        ctxterm = torch.cat([hs, f_rnn[:, None].expand(-1, 7, -1)], -1) @ wb.T + self.rnn_b  # [B,7,768]
        state = torch.zeros(B, RANK)
        prev = torch.tensor([int(ids[a]) for a in anchors])
        toks, btoks = [], []
        for j in range(7):
            raw = torch.cat([state, self.w1[prev]], -1) @ wa.T + ctxterm[:, j]
            g_, c_, o_ = raw.chunk(3, -1)
            gate = torch.sigmoid(g_)
            state = gate * state + (1 - gate) * torch.tanh(c_)
            latent = torch.tanh(o_)
            bias = (latent @ self.ho_w.T + self.ho_b) @ self.unembed.T  # [B,V]
            lg = base[:, j] + bias
            if "nomarkov" in self.variant:
                lg = base[:, j]
            lg = CAP * torch.tanh(lg / CAP)
            prev = lg.argmax(-1)
            toks.append(prev)
            btoks.append(base[:, j].argmax(-1))
        return torch.stack(toks, 1), torch.stack(btoks, 1)


def evaluate(d, npz, stride=8, start=16, batch=8, text_ref=False):
    z = np.load(npz)
    feat = torch.from_numpy(z["feat"])
    ids = z["ids"]
    am = z["argmax"]
    n = len(ids)
    ckv = d.ctx_kv(feat)
    anchors = list(range(start, n - 8, stride))
    acc_len = []
    pos_hit = np.zeros(7)
    pos_cnt = np.zeros(7)
    txt_hit = np.zeros(7)
    for b in range(0, len(anchors), batch):
        A = anchors[b:b + batch]
        toks, btoks = d.draft(ckv, feat, ids, A)
        for r, a in enumerate(A):
            # target chain: argmax[a..a+6] = target's greedy for tokens a+1..a+7 (given ids prefix)
            ref = am[a:a + 7]
            hit = (toks[r].numpy() == ref)
            L = 0
            while L < 7 and hit[L]:
                L += 1
            acc_len.append(L)
            for j in range(7):
                if j <= L:  # position j counted only if all earlier ones accepted (conditional acceptance)
                    pos_cnt[j] += 1
                    pos_hit[j] += hit[j]
            txt_hit += (toks[r].numpy() == ids[a + 1:a + 8])
    acc_len = np.array(acc_len)
    return dict(n_anchors=len(anchors), mean_accept=float(acc_len.mean()), dist=np.bincount(acc_len, minlength=8).tolist(),
                cond_pos=(pos_hit / np.maximum(pos_cnt, 1)).round(3).tolist(),
                pos1_vs_target=float((acc_len >= 1).mean()), text_agree=(txt_hit / len(anchors)).round(3).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--start", type=int, default=16)
    ap.add_argument("--variant", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = Drafter(variant=a.variant)
    out = {}
    for f in a.feats:
        r = evaluate(d, f, a.stride, a.start)
        out[os.path.basename(f)] = r
        print(os.path.basename(f), r, flush=True)
    if a.json:
        import json
        json.dump(out, open(a.json, "w"), indent=1)
