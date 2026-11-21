#!/usr/bin/env python3 -I
"""Calibration run of the Inkling-2-Mini torch reference (tools/inkling_ref.py, CPU, fp32 compute on the exact dequantized
checkpoint) for the Q-Volta converter.

For each calibration sequence it records, per layer APPLICATION (L, pass):
  * x   = the normalised MoE input (post mlp_norm, the exact input of every expert w13), fp16 [n, 2304]
  * idx = the routed expert ids int8 [n, 8]
and activation-range statistics (max |value|) of: layer output h (residual stream), attention branch output, expert gate/up
pre-activations, expert activation, expert output, MoE output. The converter derives per-expert Hessians offline:
H13_e = X_e^T X_e (X_e = rows of x routed to e) and H2_e from the expert activations recomputed with the exact w13.
Output dir: <out>/seq_<name>.npz (+ stats_<name>.json). Sequences are independent (own KV cache).

  python3 -I tools/inkling_calib.py --out DIR [--only wiki0,code0] [--threads 48]
Sets: calibration = wiki0, wiki1, code0, chat0 (2048 tokens each = 8192); held-out validation = val0 (wiki+code, 2048).
The golden prompts / README text are NOT used (they are the evaluation set).
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_ref as R
from inkling_ref import Inkling, C
import inkling_ckpt as K

SPECIAL = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"


def build_sets(model_dir, L=2048):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    wt = open("/workspace/q-volta-models/calib/wt_train.txt").read()
    wids = tok.encode(wt, add_special_tokens=False).ids
    rng = random.Random(4321)

    def wiki(n):
        s = rng.randrange(0, len(wids) - n)
        return wids[s:s + n]

    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    rng.shuffle(files)
    code = []
    for f in files:
        try:
            code += tok.encode(open(f, errors="ignore").read(), add_special_tokens=False).ids
        except Exception:
            continue
        if len(code) > 40000:
            break

    def codew(n):
        s = rng.randrange(0, len(code) - n)
        return code[s:s + n]

    paras = [p.strip() for p in wt.split("\n") if len(p.strip()) > 300 and not p.strip().startswith("=")]

    def chat(n):
        txt = SPECIAL
        while True:
            a, b = rng.choice(paras), rng.choice(paras)
            txt += (f"<|message_user|><|content_text|>Summarize or continue this passage:\n{a[:700]}<|end_message|>"
                    f"<|message_model|><|content_text|>{b[:900]}<|end_message|><|content_model_end_sampling|>")
            t = tok.encode(txt, add_special_tokens=False).ids
            if len(t) >= n:
                return t[:n]

    S = {"wiki0": wiki(L), "wiki1": wiki(L), "code0": codew(L), "chat0": chat(L), "val0": wiki(L // 2) + codew(L // 2)}
    return S


class CalibInkling(Inkling):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rec = {}
        self.stats = {}
        self.cur = None

    def _st(self):
        return self.stats.setdefault(f"L{self.cur[0]}_p{self.cur[1]}", {})

    def _mx(self, key, t):
        s = self._st()
        s[key] = max(s.get(key, 0.0), float(t.abs().max()))

    def layer(self, L, p, h, blocks, cache, pos, messages=None, tr=None):
        self.cur = (L, p)
        out = super().layer(L, p, h, blocks, cache, pos, messages, tr)
        self._mx("h_out", out)
        return out

    def attention(self, L, p, cache, hq, hk, hv, pos):
        o = super().attention(L, p, cache, hq, hk, hv, pos)
        self._mx("attn_out", o)
        self._mx("attn_in", hq)
        return o

    def moe(self, L, x, tr=None):
        n = x.shape[0]
        logits = x @ self.lw(L, "mlp.gate.weight").T
        sc = torch.sigmoid(logits)
        idx = torch.topk(sc + self.lw(L, "mlp.gate.bias"), C.TOPK, dim=-1, sorted=False).indices
        wts = sc.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * C.ROUTE
        if tr is not None:
            tr["router_idx"] = idx.clone()
        key = f"L{L}_p{self.cur[1]}"
        self.rec["x_" + key] = x.clamp(-65000, 65000).to(torch.float16).numpy()
        self.rec["i_" + key] = idx.to(torch.int8).numpy()
        self._mx("mlp_in", x)
        self._mx("router_logit", logits)
        out = torch.zeros(n, C.H)
        flat = idx.reshape(-1)
        for e in torch.unique(flat).tolist():
            tok, sp = torch.where(idx == e)
            w13, w2 = self.expert(L, e)
            gu = x[tok] @ w13.float().T
            g, u = gu[:, :C.FF], gu[:, C.FF:]
            act = R.softcap(g, C.GATE_CAP) * torch.sigmoid(g) * R.softcap(u, C.UP_CAP)
            y = act @ w2.float().T
            self._mx("gu", gu); self._mx("act", act); self._mx("expert_out", y)
            out.index_add_(0, tok, y * wts[tok, sp][:, None])
        self._mx("moe_out", out)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=K.MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--cache-gb", type=float, default=30.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    S = build_sets(a.model)
    m = CalibInkling(a.model, expert_cache_gb=a.cache_gb, verbose=False)
    for name, ids in S.items():
        if a.only and name not in a.only.split(","):
            continue
        if os.path.exists(os.path.join(a.out, f"seq_{name}.npz")):
            print("skip", name); continue
        t0 = time.time()
        m.rec, m.stats = {}, {}
        with torch.no_grad():
            m.forward(ids, m.new_cache())
        np.savez(os.path.join(a.out, f"seq_{name}.npz"), ids=np.array(ids, dtype=np.int32), **m.rec)
        json.dump(m.stats, open(os.path.join(a.out, f"stats_{name}.json"), "w"), indent=1)
        print(f"{name}: {len(ids)} tokens, {len(m.rec) // 2} apps, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
