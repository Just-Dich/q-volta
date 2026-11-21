#!/usr/bin/env python3 -I
"""Quality of a quantised Inkling-2-Mini: the torch reference (tools/inkling_ref.py, fp32 compute) with the weights read back
from a converted .qv (real file, real unpack/dequant path) or simulated on the fly (RTN), against tests/golden/inkling.

  python3 -I tools/inkling_qeval.py --qv FILE.qv --tasks p1,p2,p3,nll --out res.json
  python3 -I tools/inkling_qeval.py --sim 5:64:rtn:clip --tasks p1,p2,p3,p4,nll --out res.json     # simulated experts, dense = checkpoint
  python3 -I tools/inkling_qeval.py --tasks nll,p1                                                   # the reference itself (sanity)
Metrics per golden prompt (teacher forced on the golden greedy ids, i.e. ids + gen_ids[:31] in one prefill):
  top1 = agreement of argmax with the 32 golden greedy ids; top20_max/top20_mean = |dlogit| on the golden top-20 ids per step;
  overlap20 = mean |top20(q) & top20(golden)|; kl0 = full-vocab KL(golden || q) at the first step; klT20 = mean over the 32 steps of
  the KL over the golden top-20 ids renormalised (the golden file stores only the top-20 values). NLL: next-token NLL of the first
  512 README tokens ("## 3. Model Properties" onwards, as tools/inkling_ablate.py).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_ref as R
from inkling_ref import Inkling, C
import inkling_ckpt as K
import qv_format as F
import qv_quant as Q


class Slicer:
    def __init__(self, t): self.t = t
    def __getitem__(self, i): return self.t[i]


class Proxy:
    def __init__(self, f, over): self.f, self.over = f, over
    def get_slice(self, n): return Slicer(self.over[n]) if n in self.over else self.f.get_slice(n)
    def get_tensor(self, n): return self.f.get_tensor(n)


class QInkling(Inkling):
    def __init__(self, d, qv=None, sim=None, sim_dense=None, dense_sel="embed,unembed,attn", **kw):
        super().__init__(d, **kw)
        self.dsel = set(dense_sel.split(","))
        self.sim_dense = sim_dense  # (bits, group): RTN+clip meta8 on the dense matrices of convert_inkling (embed, unembed, q/k/v/o/gate_attn)
        self.rd = F.Reader(qv) if qv else None
        self.sim = sim  # (bits, group, clip)
        self.qcache = {}
        if self.sim_dense and "embed" in self.dsel:
            e = self.s.f.get_slice("model.llm.embed.weight")[0:C.VOCAB].float()
            self.s.f = Proxy(self.s.f, {"model.llm.embed.weight": self._sq(e)})
        if self.rd:
            self.s.f = Proxy(self.s.f, {"model.llm.embed.weight": self._dense("model.llm.embed.weight")})

    def _dense(self, name):
        t = self.rd.tensors[name]
        if t["kind"] == "q":
            return self.rd.dequant(name)
        x = self.rd.dense(name).float()
        if name.endswith("sconv.weight") and x.dim() == 2:
            x = x[:, None, :]
        return x

    def w(self, name):
        if self.rd and name in self.rd.tensors:
            if name not in self.P:
                self.P[name] = self._dense(name)
            return self.P[name]
        if self.sim_dense and name not in self.P and "attn" in self.dsel and __import__("convert_inkling").QDENSE.search(name):
            self.P[name] = self._sq(super().w(name))
            return self.P[name]
        return super().w(name)

    def _sq(self, W):
        b, g = self.sim_dense
        q, S, M, rs = Q.rtn(W.cuda(), b, g, clip=True, meta8=True)
        return Q.dequant_g(q, S, M, g, b, rs).cpu()

    def unembed(self, hs):
        if self.rd:
            if "unembed" not in self.P:
                self.P["unembed"] = self._dense("model.llm.unembed.weight")[:C.VOCAB]
            return R.softcap(hs @ self.P["unembed"].T, C.LOGIT_CAP)
        if self.sim_dense and "unembed" in self.dsel:
            if "unembed" not in self.P:
                self.P["unembed"] = self._sq(self.s.f.get_slice("model.llm.unembed.weight")[0:C.VOCAB].float())
            return R.softcap(hs @ self.P["unembed"].T, C.LOGIT_CAP)
        return super().unembed(hs)

    def expert(self, L, e):
        if not self.rd and not self.sim:
            return super().expert(L, e)
        key = (L, e)
        if key in self.ecache:
            return self.ecache[key]
        if self.rd:
            p = f"model.llm.layers.{L}.mlp.experts."
            w13 = K.deinterleave(self.rd.dequant(p + "w13_weight", e)).half()
            w2 = self.rd.dequant(p + "w2_weight", e).half()
        else:
            b, g, clip = self.sim
            w13, w2 = Inkling.expert(self, L, e)
            self.ecache.pop(key); self.ecache_bytes -= (w13.numel() + w2.numel()) * 2
            out = []
            for W in (w13.float(), w2.float()):
                q, S, M, rs = Q.rtn(W.cuda(), b, g, clip=clip, meta8=True)  # GPU: CPU RTN+clip is ~50x slower
                out.append(Q.dequant_g(q, S, M, g, b, rs).half().cpu())
            w13, w2 = out
        res = self.ecache[key] = (w13, w2)
        self.ecache_bytes += (w13.numel() + w2.numel()) * 2
        if self.ecache_bytes > self.ecache_limit:
            for k in list(self.ecache.keys()):
                if not (C.LS <= k[0] < C.LE):
                    t = self.ecache.pop(k); self.ecache_bytes -= sum(x.numel() * 2 for x in t)
                    if self.ecache_bytes < 0.9 * self.ecache_limit:
                        break
        return res


def kl(a, b):
    la, lb = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
    return float((la.exp() * (la - lb)).sum())


def eval_prompt(m, golden_dir, name):
    g = np.load(os.path.join(golden_dir, name + ".npz"))
    ids = [int(x) for x in g["ids"]]
    gen = [int(x) for x in g["gen_ids"]]
    m.last_k = len(gen)
    t0 = time.time()
    with torch.no_grad():
        lg = m.forward(ids + gen[:-1], m.new_cache())  # [32, V]
    gi, gv = g["gen_top20_ids"], g["gen_top20_val"]
    top1 = int((lg.argmax(-1).numpy() == np.array(gen)).sum())
    d, ov, kt = [], [], []
    for s in range(len(gen)):
        qv = lg[s, gi[s]].numpy()
        d.append(np.abs(qv - gv[s]))
        ov.append(len(set(torch.topk(lg[s], 20).indices.tolist()) & set(gi[s].tolist())))
        kt.append(kl(torch.tensor(gv[s]), torch.tensor(qv)))
    d = np.stack(d)
    return dict(n_prompt=len(ids), top1=top1, of=len(gen), top20_max=float(d.max()), top20_mean=float(d.mean()),
                overlap20=float(np.mean(ov)), kl0=kl(torch.tensor(g["logits_last"]), lg[0]), klT20=float(np.mean(kt)),
                first_mismatch=next((i for i in range(len(gen)) if int(lg[i].argmax()) != gen[i]), None), seconds=round(time.time() - t0))


def eval_nll(m, d=K.MODEL, n=512):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    txt = open(os.path.join(d, "README.md"), encoding="utf-8").read()
    txt = txt[txt.index("## 3. Model Properties"):]
    ids = tok.encode(txt, add_special_tokens=False).ids[:n]
    with torch.no_grad():
        lg = m.forward(ids, m.new_cache(), all_logits=True)
    lp = torch.log_softmax(lg[:-1].float(), -1)
    return dict(nll=float(-lp.gather(1, torch.tensor(ids[1:])[:, None]).mean()), n=len(ids) - 1)


def eval_val(m, calib="/workspace/q-volta-models/scratch-inkling/calib"):
    """NLL on the held-out calibration-dump sequence val0 (1024 wikitext-train + 1024 python-source tokens; never used for calibration)."""
    ids = [int(x) for x in np.load(os.path.join(calib, "seq_val0.npz"))["ids"]]
    with torch.no_grad():
        lg = m.forward(ids, m.new_cache(), all_logits=True)
    lp = torch.log_softmax(lg[:-1].float(), -1)
    nl = -lp.gather(1, torch.tensor(ids[1:])[:, None]).squeeze(1)
    return dict(nll=float(nl.mean()), nll_wiki=float(nl[:1023].mean()), nll_code=float(nl[1024:].mean()), n=len(ids) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=K.MODEL)
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "inkling"))
    ap.add_argument("--qv"); ap.add_argument("--sim", help="bits:group:rtn[:clip]"); ap.add_argument("--dense-sel", default="embed,unembed,attn"); ap.add_argument("--sim-dense", help="bits:group, RTN+clip meta8 on embed/unembed/q/k/v/o/gate_attn")
    ap.add_argument("--tasks", default="p1,p2,p3,nll")
    ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--threads", type=int, default=48); ap.add_argument("--cache-gb", type=float, default=30.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sim = None
    if a.sim:
        x = a.sim.split(":"); sim = (int(x[0]), int(x[1]), "clip" in x[3:])
    m = QInkling(a.model, qv=a.qv, sim=sim, dense_sel=a.dense_sel, sim_dense=tuple(int(x) for x in a.sim_dense.split(':')) if a.sim_dense else None, expert_cache_gb=a.cache_gb, verbose=False)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    names = {"p1": "p1_en", "p2": "p2_code", "p3": "p3_multi", "p4": "p4_long4k"}
    for t in a.tasks.split(","):
        r = eval_nll(m) if t == "nll" else eval_val(m) if t == "val" else eval_prompt(m, a.golden, names[t])
        res[t] = r
        print(a.label or a.qv or a.sim or "ref", t, json.dumps(r), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
