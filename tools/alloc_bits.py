#!/usr/bin/env python3
"""Sensitivity-based per-tensor bit allocation (multiple-choice knapsack).

Input : sens json written by `convert.py --sens` (per tensor: GPTQ+clip output error  e = E||(W-Wq)x||^2 and output
        energy en = E||Wx||^2 on a calibration Hessian, for every candidate bits/group; `resid.<layer>` = E||h||^2 of the
        residual stream entering that layer).
Unit  : tensors that share an input Hessian and are fused in the engine (gate+up, qkv+z, q+k+v, o, down) get one bit width.
Loss  : norm=resid -> sum_t e_t / E||h_layer||^2   (absolute output error relative to the residual stream it is added to)
        norm=rel   -> sum_t e_t / en_t              (relative error of the tensor output)
Solve : exact DP over bytes (1 MiB bins) for the min total loss under a byte budget for the allocated tensors.
Output: plan json {tensor name: [bits, group]} for convert.py --plan.
usage : alloc_bits.py SENS.json BUDGET_GIB OUT_PLAN.json [--group 128] [--norm resid] [--bits 2,3,4,5,6] [--min-bits "regex=b"]
"""
import argparse, json, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sens"); ap.add_argument("budget_gib", type=float); ap.add_argument("out")
ap.add_argument("--group", type=int, default=128)
ap.add_argument("--norm", default="resid", choices=["resid", "rel"])
ap.add_argument("--bits", default="2,3,4,5,6")
ap.add_argument("--sm-bits", type=float, default=32.0, help="bits per (scale,min) pair (32 = fp16+fp16)")
ap.add_argument("--fixed-gib", type=float, default=0.0, help="bytes of everything not allocated here (embed, head, norms, MTP...)")
ap.add_argument("--force", action="append", default=[], help="regex=bits forced for matching units (e.g. 'layers\\.0\\.=5')")
ap.add_argument("--shape-json", default="/workspace/q-volta-models/Qwen3.8-27B/model.safetensors.index.json")
a = ap.parse_args()

S = json.load(open(a.sens))
cands = [int(b) for b in a.bits.split(",")]
G = a.group
NUMEL = {"mlp.gate_proj": 17408 * 5120, "mlp.up_proj": 17408 * 5120, "mlp.down_proj": 5120 * 17408,
         "in_proj_qkv": 10240 * 5120, "in_proj_z": 6144 * 5120, "linear_attn.out_proj": 5120 * 6144,
         "q_proj": 12288 * 5120, "k_proj": 1024 * 5120, "v_proj": 1024 * 5120, "o_proj": 5120 * 6144}


def numel(n):
    for k, v in NUMEL.items():
        if k in n:
            return v
    raise KeyError(n)


def unit_of(n):
    m = re.match(r"(model\.language_model\.layers\.(\d+)\.)(.*)", n)
    if not m:
        return None
    layer, rest = int(m.group(2)), m.group(3)
    key = "m_gu" if re.search(r"mlp\.(gate|up)_proj", rest) else "m_d" if "down_proj" in rest else \
        "o" if re.search(r"out_proj|o_proj", rest) else "a"
    return (layer, key)


units = {}
for n, v in S.items():
    if not isinstance(v, dict) or n.startswith("resid") or n == "lm_head.weight":
        continue
    u = unit_of(n)
    units.setdefault(u, []).append(n)

opts = []  # per unit: list of (bytes, loss, bits)
names = []
for u, ts in sorted(units.items()):
    row = []
    for b in cands:
        key = f"b{b}g{G}"
        if not all(key in S[t] for t in ts):
            continue
        by = sum(numel(t) * (b + a.sm_bits / G) / 8 for t in ts)
        if a.norm == "resid":
            loss = sum(S[t][key][0] for t in ts) / S[f"resid.{u[0]}"]
        else:
            loss = sum(S[t][key][0] / S[t][key][1] for t in ts)
        row.append((by, loss, b))
    for rx in a.force:
        pat, fb = rx.rsplit("=", 1)
        if any(re.search(pat, t) for t in ts):
            row = [r for r in row if r[2] == int(fb)] or row
    opts.append(row)
    names.append(u)

MB = 1 << 20
budget = int((a.budget_gib - a.fixed_gib) * (1 << 30) / MB)
minb = sum(min(r[0] for r in row) for row in opts) / MB
maxb = sum(max(r[0] for r in row) for row in opts) / MB
print(f"{len(opts)} units; byte range {minb/1024:.2f}..{maxb/1024:.2f} GiB, budget {budget/1024:.2f} GiB", file=sys.stderr)
if budget < minb:
    sys.exit(f'infeasible: budget {budget/1024:.2f} GiB < minimum {minb/1024:.2f} GiB')
INF = 1e30
dp = np.full(budget + 1, INF)
dp[0] = 0.0
choice = []
for row in opts:
    nd = np.full(budget + 1, INF)
    ch = np.full(budget + 1, -1, dtype=np.int8)
    for j, (by, loss, b) in enumerate(row):
        w = int(np.ceil(by / MB))
        if w > budget:
            continue
        cand = np.full(budget + 1, INF)
        cand[w:] = dp[:budget + 1 - w] + loss
        m = cand < nd
        nd[m] = cand[m]
        ch[m] = j
    dp = nd
    choice.append(ch)
best = int(np.argmin(dp))
tot = dp[best]
plan, used = {}, 0
w = best
for i in range(len(opts) - 1, -1, -1):
    j = int(choice[i][w])
    by, loss, b = opts[i][j]
    for t in units[names[i]]:
        plan[t] = [b, G]
    w -= int(np.ceil(by / MB))
    used += by
hist = {}
for t, (b, g) in plan.items():
    hist[b] = hist.get(b, 0) + numel(t)
print(f"total loss {tot:.4e}; allocated {used/2**30:.3f} GiB; params by bits: " +
      ", ".join(f"{b}b {n/1e9:.2f}B" for b, n in sorted(hist.items())), file=sys.stderr)
json.dump(plan, open(a.out, "w"), indent=0)
