#!/usr/bin/env python3 -I
"""Sensitivity / sanity study for tools/inkling_ref.py: next-token NLL of 511 plain-text tokens (README body) under
the reference and under single-op ablations. A correct transcription should give the lowest NLL; a wrong convention
(conv tap order, gate/up order, ...) should visibly hurt. Not a proof of HF parity (no HF run available)."""
import os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_ref as R
from inkling_ref import Inkling, C
from transformers import AutoTokenizer

d = "/workspace/q-volta-models/Inkling-2-Mini"
torch.set_num_threads(int(os.environ.get("THREADS", "48")))
tok = AutoTokenizer.from_pretrained(d)
txt = open(os.path.join(d, "README.md"), encoding="utf-8").read()
txt = txt[txt.index("## 3. Model Properties"):]
ids = tok.encode(txt, add_special_tokens=False)[:int(os.environ.get("NTOK", "512"))]
m = Inkling(d, verbose=False)
variants = os.environ.get("VARIANTS", "ref,conv_flip,swap_gu,no_gate,no_sink,no_rel,scale128,no_reset,no_msgs,one_pass,no_dyn").split(",")
res = {}
for v in variants:
    R.ABL.clear()
    C.NP = 3
    if v == "one_pass":
        C.NP = 1
    elif v != "ref":
        R.ABL.add(v)
    for k in list(m.ecache):  # expert cache is variant independent; keep
        break
    t = time.time()
    cache = m.new_cache()
    with torch.no_grad():
        lg = m.forward(ids, cache, all_logits=True)
    lp = torch.log_softmax(lg[:-1].float(), -1)
    nll = -lp.gather(1, torch.tensor(ids[1:])[:, None]).mean().item()
    res[v] = nll
    print(f"{v:10s} NLL {nll:.4f} nats/token (n={len(ids)-1}) {time.time()-t:.0f}s", flush=True)
json.dump(res, open(os.environ.get("OUT", "ablate.json"), "w"), indent=1)
