#!/usr/bin/env python3 -I
"""Metrics of the teacher-forced sparse-attention probe (qv_run_cpu --tail T --sparse-list ...): per config KL(dense||sparse),
top-1 agreement and NLL of the true next tokens, over the T logit rows produced by sparse single-token steps (row 0 = dense prefill).
usage: sparse_logits_eval.py PROMPT.i32 LOGITS_PREFIX T OUT.json cfg0;cfg1;...   (cfg0 = 'off' = dense reference)"""
import sys, json, numpy as np
prompt, pre, T, out, cfgs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5].split(';')
toks = np.fromfile(prompt, dtype=np.int32)
def lsm(x): x = x.astype(np.float64); x -= x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
load = lambda i: np.fromfile(f'{pre}.{i}', dtype=np.float32).reshape(T + 1, -1)
D = load(0); LD = lsm(D); PD = np.exp(LD); nxt = toks[-T:]
res = {}
for i, c in enumerate(cfgs):
    A = load(i); LA = lsm(A); r = slice(1, T + 1)
    kl = (PD[r] * (LD[r] - LA[r])).sum(1)
    nd = float(-LD[np.arange(1, T), nxt[1:T]].mean()); ns = float(-LA[np.arange(1, T), nxt[1:T]].mean())
    res[c] = {'kl_mean': float(kl.mean()), 'kl_max': float(kl.max()), 'top1_agree': float((D[r].argmax(1) == A[r].argmax(1)).mean()),
              'nll_dense': nd, 'nll_sparse': ns, 'ppl_delta_pct': float(100 * (np.exp(ns - nd) - 1)), 'rows': T,
              'last_row_argmax': int(A[T].argmax()), 'last_row_pmax': float(np.exp(LA[T]).max())}
json.dump(res, open(out, 'w'), indent=1)
for c, v in res.items(): print(f"{c:18s} KL {v['kl_mean']:.5f} top1 {v['top1_agree']:.3f} ppl {v['ppl_delta_pct']:+.2f}%")
