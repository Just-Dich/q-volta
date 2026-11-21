#!/usr/bin/env python3 -I
"""Self-consistency checks for tools/inkling_ref.py (state handling) + bf16-flow sensitivity against the goldens.
  python3 -I tools/inkling_selfcheck.py [--golden tests/golden/inkling]
1. one-shot prefill vs chunked prefill vs prefill+token-by-token decode (conv states, KV caches, per-pass slots): last logits.
2. bf16-flow emulation (emu_bf16=True) vs the fp32 golden: KL(golden || emu), top-1 agreement (short prompts only).
`kl(a,b)` / `compare(npz, logits)` are the helpers an engine test should use against tests/golden/inkling/*.npz."""
import argparse, os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling_ref import Inkling


def kl(a, b):
    """KL(softmax(a) || softmax(b)) in nats for logit vectors."""
    a = torch.as_tensor(a, dtype=torch.float64); b = torch.as_tensor(b, dtype=torch.float64)
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return float((la.exp() * (la - lb)).sum())


def compare(npz_path, logits):
    g = np.load(npz_path)
    ref = g["logits_last"]
    return {"kl": kl(ref, logits), "top1_match": int(np.argmax(ref)) == int(np.argmax(np.asarray(logits))),
            "max_abs": float(np.abs(ref - np.asarray(logits)).max())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="tests/golden/inkling")
    ap.add_argument("--model", default="/workspace/q-volta-models/Inkling-2-Mini")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--emu", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = {}
    if not a.emu:
        m = Inkling(a.model, verbose=False)
        g = np.load(os.path.join(a.golden, "p1_en.npz"))
        ids = [int(x) for x in g["ids"]]
        gen = [int(x) for x in g["gen_ids"]]
        full = ids + gen[:5]
        with torch.no_grad():
            c1 = m.new_cache(); l_full = m.forward(full, c1)[0]
            c2 = m.new_cache(); m.forward(full[:11], c2); l_chunk = m.forward(full[11:], c2)[0]
            c3 = m.new_cache(); l = m.forward(full[:12], c3)[0]
            for t in full[12:]:
                l = m.forward([t], c3)[0]
        out["full_vs_chunked"] = {"kl": kl(l_full, l_chunk), "max_abs": float((l_full - l_chunk).abs().max())}
        out["full_vs_stepwise"] = {"kl": kl(l_full, l), "max_abs": float((l_full - l).abs().max())}
        # the prompt-only prefill must reproduce the golden
        with torch.no_grad():
            c4 = m.new_cache(); l_p = m.forward(ids, c4)[0]
        out["rerun_vs_golden"] = compare(os.path.join(a.golden, "p1_en.npz"), l_p.numpy())
        # cached decode step vs golden top-20 of the first generated token (row 1)
        print(json.dumps(out, indent=1))
    else:
        m = Inkling(a.model, emu_bf16=True, verbose=False)
        for nm in ("p1_en", "p2_code", "p3_multi"):
            p = os.path.join(a.golden, nm + ".npz")
            ids = [int(x) for x in np.load(p)["ids"]]
            with torch.no_grad():
                l = m.forward(ids, m.new_cache())[0]
            out[nm] = compare(p, l.numpy())
            print(nm, out[nm], flush=True)
        print(json.dumps(out, indent=1))
