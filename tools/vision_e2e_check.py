#!/usr/bin/env python3 -I
"""usage: vision_e2e_check.py BIN MODEL.qv NAME [--cpu] [--emb-file f32 (override: e.g. our encoder output)] [--json out]
Runs qv_vision_e2e on tests/golden/inkling_vision/e2e_<NAME>.npz inputs (HF-tower embeddings injected) and compares with the fp32 reference run."""
import sys, os, json, subprocess, tempfile, numpy as np
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "inkling_vision")
def kl(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    la = a - a.max(); la -= np.log(np.exp(la).sum()); lb = b - b.max(); lb -= np.log(np.exp(lb).sum())
    return float((np.exp(la) * (la - lb)).sum())
def main():
    a = sys.argv; binp, model, name = a[1], a[2], a[3]
    img = a[a.index("--img") + 1] if "--img" in a else name
    g = np.load(f"{G}/e2e_{name}.npz")
    with tempfile.TemporaryDirectory() as td:
        g["ids"].astype("<i4").tofile(td + "/ids"); g["mask"].astype("u1").tofile(td + "/mask")
        emb = np.load(f"{G}/{img}.npz")["embeds"].astype("<f4")
        if "--emb-file" in a: emb = np.fromfile(a[a.index("--emb-file") + 1], "<f4")
        emb.tofile(td + "/emb")
        cmd = [binp, "--model", model, "--ids", td + "/ids", "--mask", td + "/mask", "--emb", td + "/emb", "--logits-out", td + "/lg", "--gen", "32"] + (["--cpu"] if "--cpu" in a else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode: print(r.stderr); sys.exit(1)
        out = json.loads(r.stdout.strip().splitlines()[-1]); lg = np.fromfile(td + "/lg", "<f4")
    ref = g["logits_last"]; gen = np.array(out["gen"]); rg = g["gen_ids"]
    n = min(len(gen), len(rg)); pre = int(next((i for i in range(n) if gen[i] != rg[i]), n))
    gi, gv = g["gen_top20_ids"][0], g["gen_top20_val"][0]
    res = dict(prompt_tokens=out["tokens"], prefill_ms=out["prefill_ms"], top1_last=bool(lg[:len(ref)].argmax() == ref.argmax()),
               kl_last=kl(ref, lg[:len(ref)]), top20_max_abs=float(np.abs(lg[gi] - gv).max()), gen_match_prefix=pre, gen_match_total=int((gen[:n] == rg[:n]).sum()), gen_n=n,
               gen_ids=gen.tolist())
    print(json.dumps(res))
    if "--json" in a: json.dump(res, open(a[a.index("--json") + 1], "w"), indent=1)
main()
