#!/usr/bin/env python3 -I
"""Compare a qv_ink_cpu --dump directory with an Inkling golden npz (tests/golden/inkling = fp32 reference on the checkpoint,
tests/golden/inkling_q3a5 = the same reference with the Q3A5 .qv weights read back).

  python3 -I tests/check_inkling_cpu.py DUMPDIR tests/golden/inkling_q3a5/p1_en.npz [--json out.json]
Reports: per-application relative hidden error at the golden positions (max / mean / worst app), router id-set mismatches
(token, application pairs whose top-8 expert SET differs), prefill logits top-1 / top-20 |dlogit| / KL over the golden top-20,
and agreement of the 32 greedy tokens (first mismatch) with the golden's own greedy sequence.
"""
import json, sys
import numpy as np

d, npz = sys.argv[1], sys.argv[2]
g = np.load(npz)
ids = g["ids"]; n = len(ids); H = 2304
hid = np.fromfile(f"{d}/hidden.f32", np.float32).reshape(50, n, H)
rt = np.fromfile(f"{d}/router.i8", np.int8).reshape(48, n, 8)
pos = list(g["hidden_pos"])
res = {"n_prompt": int(n)}
keys = sorted(k for k in g.files if k.startswith("hid_app"))
rel = []
for a, k in enumerate(keys):
    ref = g[k]
    mine = hid[1 + a][pos]
    rel.append(float(np.linalg.norm(mine - ref) / np.linalg.norm(ref)))
res["emb_rel"] = float(np.linalg.norm(hid[0][pos] - g["hid_emb"]) / np.linalg.norm(g["hid_emb"]))
res["final_norm_rel"] = float(np.linalg.norm(hid[49][pos] - g["hid_final_norm"]) / np.linalg.norm(g["hid_final_norm"]))
res["hidden_rel_max"] = max(rel); res["hidden_rel_mean"] = float(np.mean(rel)); res["hidden_rel_worst_app"] = int(np.argmax(rel))
res["hidden_rel_per_app"] = [round(r, 5) for r in rel]
rk = sorted(k for k in g.files if k.startswith("router_app"))
mis = 0; tot = 0; mis_app = []
for a, k in enumerate(rk):
    ref = g[k]
    m = sum(set(rt[a][t].tolist()) != set(ref[t].tolist()) for t in range(n))
    mis += m; tot += n; mis_app.append(int(m))
res["router_set_mismatch"] = mis; res["router_total"] = tot; res["router_mismatch_per_app"] = mis_app
lg = np.fromfile(f"{d}/logits_last.f32", np.float32)
gl = g["logits_last"]
gi, gv = g["gen_top20_ids"][0], g["gen_top20_val"][0]
res["prefill_top1_equal"] = bool(lg.argmax() == gl.argmax())
dl = np.abs(lg[gi] - gv)
res["prefill_top20_absd_max"] = float(dl.max()); res["prefill_top20_absd_mean"] = float(dl.mean())
def kl(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    la = a - np.log(np.exp(a - a.max()).sum()) - a.max(); lb = b - np.log(np.exp(b - b.max()).sum()) - b.max()
    return float((np.exp(la) * (la - lb)).sum())
res["prefill_kl_top20"] = kl(gv, lg[gi])
res["prefill_kl_full"] = kl(gl, lg)
try:
    mine = np.fromfile(f"{d}/gen_ids.i32", np.int32)
    ref = g["gen_ids"][:len(mine)]
    res["gen_equal"] = int((mine == ref).sum()); res["gen_n"] = int(len(mine))
    res["gen_first_mismatch"] = next((i for i in range(len(mine)) if mine[i] != ref[i]), None)
    t20 = np.fromfile(f"{d}/gen_top20_ids.i32", np.int32).reshape(-1, 20)
    res["gen_top20_overlap_mean_teacherfree"] = float(np.mean([len(set(t20[i]) & set(g["gen_top20_ids"][i])) for i in range(len(mine))]))
    if res["gen_first_mismatch"] is None or res["gen_first_mismatch"] > 0:
        k = res["gen_first_mismatch"] if res["gen_first_mismatch"] is not None else len(mine)
        res["gen_step_top20_absd_max_before_divergence"] = float(max(np.abs(np.fromfile(f"{d}/gen_top20_val.f32", np.float32).reshape(-1, 20)[i] - g["gen_top20_val"][i]).max() for i in range(k)))
except FileNotFoundError:
    pass
print(json.dumps({k: v for k, v in res.items() if not k.endswith("per_app")}, indent=1))
if len(sys.argv) > 4 and sys.argv[3] == "--json":
    json.dump(res, open(sys.argv[4], "w"), indent=1)
