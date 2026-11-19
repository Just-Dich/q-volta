#!/usr/bin/env python3
"""Full-model correctness: run qv_run --golden on the first T eval tokens and compare, per layer and for the logits,
against the converter's simulated-quantization reference (qsim, same dequantized weights, fp32 torch) and against
the BF16 reference. Usage: check_golden.py MODEL_PREFIX EVAL_DIR QV_RUN [T] [--env K=V ...]"""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch

prefix, eval_dir, qv_run = sys.argv[1:4]
T = int(sys.argv[4]) if len(sys.argv) > 4 else 64
cfg = json.load(open(prefix + ".qv.json"))["config"]
H, V, L = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
toks = np.load(os.path.join(eval_dir, "tokens.npy"))[0, :T].astype(np.int32)
g = torch.load(os.path.join(eval_dir, "golden_seq0.pt"))

with tempfile.TemporaryDirectory() as td:
    tf, of = os.path.join(td, "t.i32"), os.path.join(td, "out.bin")
    toks.tofile(tf)
    r = subprocess.run([qv_run, "--model", prefix, "--golden", tf, "--out", of], capture_output=True, text=True)
    print(r.stderr.strip())
    if r.returncode:
        sys.exit("qv_run failed")
    raw = np.fromfile(of, dtype=np.float32)
logits = torch.from_numpy(raw[: T * V].reshape(T, V).copy())
hid = torch.from_numpy(raw[T * V:].reshape(T, L + 1, H).copy()).transpose(0, 1)  # [L+1, T, H]

ref, q = g["ref"][:, :T], g["q"][:, :T]


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


worst = 0.0
print("layer  rel(engine,qsim)  rel(qsim,ref)")
for i in range(L + 1):
    e = rel(hid[i], q[i])
    worst = max(worst, e)
    if i < 5 or i % 8 == 0 or i == L:
        print(f"{i:5d}  {e:14.2e}  {rel(q[i], ref[i]):12.2e}")

lq = g["logits_q"][:T - 1]  # golden logits predict tokens 1..T-1 (positions 0..T-2)
le = logits[:T - 1]
lr = g["logits_ref"][:T - 1]
pe, pq, pr = (torch.log_softmax(x.double(), -1) for x in (le, lq, lr))
kl_eq = (pq.exp() * (pq - pe)).sum(-1).mean().item()
kl_re = (pr.exp() * (pr - pe)).sum(-1).mean().item()
kl_rq = (pr.exp() * (pr - pq)).sum(-1).mean().item()
agree_q = (le.argmax(-1) == lq.argmax(-1)).float().mean().item()
agree_r = (le.argmax(-1) == lr.argmax(-1)).float().mean().item()
tgt = torch.from_numpy(toks[1:]).long()
nll = lambda p: -p.gather(1, tgt[:, None]).mean().item()
res = dict(T=T, worst_layer_rel=worst, final_logits_rel=rel(le, lq), kl_qsim_engine=kl_eq, kl_ref_engine=kl_re,
           kl_ref_qsim=kl_rq, top1_agree_qsim=agree_q, top1_agree_ref=agree_r,
           nll_engine=nll(pe), nll_qsim=nll(pq), nll_ref=nll(pr))
print(json.dumps(res, indent=1))
# Gate: the engine must reproduce qsim far more closely than qsim differs from the BF16 reference.
ok = res["kl_qsim_engine"] < 0.1 * max(res["kl_ref_qsim"], 1e-6) and res["top1_agree_qsim"] >= 0.95
print("GOLDEN", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
