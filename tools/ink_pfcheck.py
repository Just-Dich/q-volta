#!/usr/bin/env python3 -I
"""python3 -I tools/ink_pfcheck.py OUT.bin GOLDEN.npz : compare qv_ink --pf-gen output (last logits + greedy ids) to a golden."""
import sys, numpy as np
g = np.load(sys.argv[2]); V = len(g['logits_last'])
b = np.fromfile(sys.argv[1], dtype='<f4'); lg = b[:V]; ids = np.frombuffer(b[V:].tobytes(), '<i4')
ref = g['logits_last'].astype(np.float64); x = lg.astype(np.float64)
def lsm(a): a = a - a.max(); return a - np.log(np.exp(a).sum())
kl = float((np.exp(lsm(ref)) * (lsm(ref) - lsm(x))).sum())
n = min(len(ids), len(g['gen_ids']))
print(dict(kl_last=kl, top1=int(ref.argmax()) == int(x.argmax()), max_abs=float(np.abs(ref - x).max()),
           top20_mean_abs=float(np.abs(x[g['gen_top20_ids'][0]] - g['gen_top20_val'][0]).mean()),
           gen_match=int(sum(ids[i] == g['gen_ids'][i] for i in range(n))), of=n,
           prefix=int(next((i for i in range(n) if ids[i] != g['gen_ids'][i]), n))))
