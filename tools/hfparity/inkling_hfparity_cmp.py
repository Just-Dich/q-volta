#!/usr/bin/env python3 -I
"""Compare HF (transformers PR #51050, bf16, CPU) outputs against tests/golden/inkling (our fp32 reference).
HF outputs come from the isolated scratch run (scratch-hfparity/hf_run.py -> out/hf_<prompt>.npz).
  python3 -I tools/inkling_hfparity_cmp.py HF_DIR [prompts...]  -> prints JSON (also written to HF_DIR/parity.json)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden", "inkling")


def kl(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    la = a - a.max(); la = la - np.log(np.exp(la).sum())
    lb = b - b.max(); lb = lb - np.log(np.exp(lb).sum())
    return float((np.exp(la) * (la - lb)).sum())


def main():
    d = sys.argv[1]
    names = sys.argv[2:] or ["p1_en", "p2_code", "p3_multi", "p4_long4k"]
    res = {}
    for nm in names:
        g = np.load(os.path.join(G, nm + ".npz")); h = np.load(os.path.join(d, "hf_" + nm + ".npz"))
        r = {}
        ref, hf = g["logits_last"], h["logits_last"]
        t20 = np.argsort(-ref)[:20]
        r["last_top1_match"] = bool(ref.argmax() == hf.argmax())
        r["last_top20_set_equal"] = bool(set(t20.tolist()) == set(np.argsort(-hf)[:20].tolist()))
        r["last_top20_max_abs"] = float(np.abs(ref[t20] - hf[t20]).max())
        r["last_top20_mean_abs"] = float(np.abs(ref[t20] - hf[t20]).mean())
        r["last_full_max_abs"] = float(np.abs(ref - hf).max())
        r["last_kl_ref_to_hf"] = kl(ref, hf)
        gen = g["gen_ids"]
        r["gen_ids_match"] = f"{int((h['hf_argmax'] == gen).sum())}/{len(gen)}"
        r["gen_first_mismatch"] = next((i for i in range(len(gen)) if h["hf_argmax"][i] != gen[i]), None)
        errs, ov = [], []
        for i in range(len(gen)):
            gi, gv = g["gen_top20_ids"][i], g["gen_top20_val"][i]
            hm = dict(zip(h["top20_ids"][i].tolist(), h["top20_val"][i].tolist()))
            com = [(v, hm[k]) for k, v in zip(gi.tolist(), gv.tolist()) if k in hm]
            ov.append(len(com))
            errs.append(max(abs(a - b) for a, b in com))
        r["step_top20_max_abs_over_steps"] = float(max(errs))
        r["step_top20_max_abs_median"] = float(np.median(errs))
        r["step_top20_min_overlap"] = int(min(ov))
        # hidden states (3 positions) per layer application
        names_app = [k for k in g.files if k.startswith("hid_app")]
        rel = []
        for i, k in enumerate(names_app):
            a, b = g[k], h["app"][i]
            rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(a)))
        r["app_rel_err_max"] = float(max(rel)); r["app_rel_err_first"] = rel[0]
        r["app_rel_err_last"] = rel[-1]; r["app_rel_err"] = [round(x, 5) for x in rel]
        r["emb_rel_err"] = float(np.linalg.norm(g["hid_emb"] - h["emb"]) / np.linalg.norm(g["hid_emb"]))
        r["final_norm_rel_err"] = float(np.linalg.norm(g["hid_final_norm"] - h["fin"]) / np.linalg.norm(g["hid_final_norm"]))
        # routers: per app, fraction of tokens whose expert SET equals
        rk = [k for k in g.files if k.startswith("router_app")]
        fr = []
        for i, k in enumerate(rk):
            a = np.sort(g[k].astype(int), -1); b = np.sort(h["router"][i].astype(int), -1)
            fr.append(float((a == b).all(-1).mean()))
        r["router_set_match_min"] = float(min(fr)); r["router_set_match_mean"] = float(np.mean(fr))
        res[nm] = r
        print(nm, json.dumps({k: v for k, v in r.items() if k != "app_rel_err"}), flush=True)
    json.dump(res, open(os.path.join(d, "parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
