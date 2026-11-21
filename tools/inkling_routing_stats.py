#!/usr/bin/env python3 -I
"""Routing statistics (H-I5 expert load skew, H-I8 expert overlap) from (a) the calibration dump of tools/inkling_calib.py and
(b) the router_* arrays of tests/golden/inkling/*.npz.  Per layer application: load = routed (token, slot) count per expert.
Reported per layer: max/mean load, Gini, share of slots held by the top 16 (12.5 %) experts, #experts below 25 % of the mean,
#experts never used; overall: same, averaged. Overlap: mean |set(t) & set(t+1)| (consecutive tokens) and, for loop layers,
|set_pass0 & set_pass1| for the same token.   python3 -I tools/inkling_routing_stats.py --calib DIR --golden DIR --out stats.json"""
import argparse, glob, json, os
import numpy as np


def gini(x):
    x = np.sort(x.astype(np.float64)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def load_stats(idx, E=128):
    c = np.bincount(idx.reshape(-1).astype(np.int64), minlength=E).astype(np.float64)
    m = c.mean()
    s = np.sort(c)[::-1]
    return dict(max_over_mean=float(c.max() / m), min_over_mean=float(c.min() / m), gini=gini(c), top16_share=float(s[:16].sum() / c.sum()),
                n_below_quarter=int((c < 0.25 * m).sum()), n_unused=int((c == 0).sum()), tokens=int(idx.shape[0]))


def overlap(idx):
    a, b = idx[:-1], idx[1:]
    return float(np.mean([len(set(x) & set(y)) for x, y in zip(a.tolist(), b.tolist())]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib"); ap.add_argument("--golden"); ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", default="wiki0,wiki1,code0,chat0")
    a = ap.parse_args()
    out = {}
    if a.calib:
        per = {}
        for nm in a.seqs.split(","):
            z = np.load(os.path.join(a.calib, f"seq_{nm}.npz"))
            for k in z.files:
                if k.startswith("i_"):
                    per.setdefault(k[2:], []).append(z[k])
        layers = {}
        for k, v in per.items():
            layers[k] = np.concatenate(v)
        out["calib_layers"] = {k: load_stats(v) for k, v in sorted(layers.items(), key=lambda kv: (int(kv[0].split("_")[0][1:]), kv[0]))}
        # pooled over passes per layer (what the weights see)
        pooled = {}
        for k, v in layers.items():
            pooled.setdefault(int(k.split("_")[0][1:]), []).append(v)
        out["calib_pooled"] = {str(L): load_stats(np.concatenate(v)) for L, v in sorted(pooled.items())}
        ov = [overlap(v) for k, v in layers.items()]
        out["consecutive_token_overlap_mean"] = float(np.mean(ov))
        # pass-to-pass overlap on the same token (loop layers), per sequence
        po = []
        for nm in a.seqs.split(","):
            z = np.load(os.path.join(a.calib, f"seq_{nm}.npz"))
            for L in range(6, 18):
                i0, i1 = z[f"i_L{L}_p0"], z[f"i_L{L}_p1"]
                po.append(np.mean([len(set(x) & set(y)) for x, y in zip(i0.tolist(), i1.tolist())]))
        out["pass0_vs_pass1_same_token_overlap_mean"] = float(np.mean(po))
        v = list(out["calib_pooled"].values())
        out["calib_summary"] = {k: float(np.mean([x[k] for x in v])) for k in ("max_over_mean", "gini", "top16_share", "n_below_quarter", "n_unused")}
        out["calib_summary"]["worst_max_over_mean"] = float(max(x["max_over_mean"] for x in v))
    if a.golden:
        g = {}
        for p in sorted(glob.glob(os.path.join(a.golden, "p*.npz"))):
            z = np.load(p)
            ids = {}
            for k in z.files:
                if k.startswith("router_"):
                    L = int(k.split("_L")[1].split("_")[0]); ids.setdefault(L, []).append(z[k])
            st = [load_stats(np.concatenate(v)) for L, v in sorted(ids.items())]
            g[os.path.basename(p)] = {k: float(np.mean([x[k] for x in st])) for k in ("max_over_mean", "gini", "top16_share", "n_below_quarter", "n_unused")}
        out["golden"] = g
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("calib_layers", "calib_pooled")}, indent=1))


if __name__ == "__main__":
    main()
