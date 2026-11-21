#!/usr/bin/env python3 -I
"""Quantised-reference goldens for the CPU/GPU Inkling engines: tools/inkling_ref.py run with the weights read back from a
converted .qv (tools/inkling_qeval.QInkling, real unpack/dequant path), same npz layout as tests/golden/inkling (prompt ids come
from those files), so engine-vs-qref checks are tight (same weights, only activation quantisation / summation order differ).

  python3 -I tools/inkling_golden_q.py --qv FILE.qv --out tests/golden/inkling_q3a5 --only p1,p2,p3 [--threads 24] [--gen 32]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling_qeval import QInkling
import inkling_ckpt as K

ap = argparse.ArgumentParser()
ap.add_argument("--qv", required=True)
ap.add_argument("--model", default=K.MODEL)
ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "inkling"))
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="p1,p2,p3")
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--gen", type=int, default=32)
ap.add_argument("--cache-gb", type=float, default=30.0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
names = {"p1": "p1_en", "p2": "p2_code", "p3": "p3_multi", "p4": "p4_long4k"}
m = QInkling(a.model, qv=a.qv, expert_cache_gb=a.cache_gb, verbose=False)
for t in a.only.split(","):
    name = names[t]
    ids = [int(x) for x in np.load(os.path.join(a.src, name + ".npz"))["ids"]]
    n = len(ids)
    cache = m.new_cache()
    t0 = time.time()
    with torch.no_grad():
        logits, tr = m.forward(ids, cache, want_trace=True)
    tp = time.time() - t0
    out = {"ids": np.array(ids, dtype=np.int32), "logits_last": logits[0].numpy().astype(np.float32)}
    posl = [0, n // 2, n - 1]
    out["hidden_pos"] = np.array(posl)
    for k, v in tr.items():
        if k.endswith("_router"):
            out["router_" + k[:-7]] = v.numpy().astype(np.int8)
        else:
            out["hid_" + k] = v[posl].numpy().astype(np.float32)
    top_ids, top_val, gen = [], [], []
    lg = logits[0]
    t1 = time.time()
    for s in range(a.gen):
        tv, ti = torch.topk(lg, 20)
        top_ids.append(ti.numpy()); top_val.append(tv.numpy())
        gen.append(int(ti[0]))
        if s == a.gen - 1:
            break
        with torch.no_grad():
            lg = m.forward([gen[-1]], cache)[0]
    out["gen_ids"] = np.array(gen, dtype=np.int32)
    out["gen_top20_ids"] = np.stack(top_ids).astype(np.int32)
    out["gen_top20_val"] = np.stack(top_val).astype(np.float32)
    np.savez_compressed(os.path.join(a.out, name + ".npz"), **out)
    print(f"{name}: n={n} prefill {tp:.1f}s decode {time.time() - t1:.1f}s gen={gen}", flush=True)
