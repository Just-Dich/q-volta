#!/usr/bin/env python3 -I
"""Compare qv_vision_run output with the HF goldens in tests/golden/inkling_vision.
usage: vision_check.py BIN VISION.qv [--cuda] [--json out.json] [--names a,b] [--tol 0.02]
Per image: pixel_values max/mean abs error vs the HF processor (C++ stb decode + our preprocess), embedding rel-L2 / max abs / cosine vs the fp32 tower,
and the same numbers for the HF bf16 tower (noise floor of HF's own default dtype). Embeddings are computed from the HF pixel_values-equivalent
C++ preprocessing of the PNG/JPEG file (the end-to-end path)."""
import sys, os, json, subprocess, tempfile, numpy as np
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "inkling_vision")

def main():
    a = sys.argv; binp, vf = a[1], a[2]
    cuda = "--cuda" in a
    names = a[a.index("--names") + 1].split(",") if "--names" in a else None
    res = {}
    for f in sorted(os.listdir(G)):
        if not f.endswith(".npz") or f.startswith("e2e_"): continue
        n = f[:-4]
        if names and n not in names: continue
        img = next(os.path.join(G, n + e) for e in (".png", ".jpg") if os.path.exists(os.path.join(G, n + e)))
        z = np.load(os.path.join(G, f))
        with tempfile.TemporaryDirectory() as td:
            cmd = [binp, "--vision", vf, "--image", img, "--out", td + "/e.f32", "--pix", td + "/p.f32"] + (["--cuda"] if cuda else [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode: print(r.stderr); sys.exit(1)
            emb = np.fromfile(td + "/e.f32", "<f4").reshape(-1, 2304)
            pix = np.fromfile(td + "/p.f32", "<f4").reshape(z["pixel_values"].shape)
        ref, bf = z["embeds"], z["embeds_bf16"]
        pd = np.abs(pix - z["pixel_values"])
        rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
        cos = float((emb * ref).sum() / (np.linalg.norm(emb) * np.linalg.norm(ref)))
        res[n] = dict(tokens=int(ref.shape[0]), pix_max_abs=float(pd.max()), pix_mean_abs=float(pd.mean()), pix_exact_frac=float((pd == 0).mean()),
                      emb_rel_l2=rel(emb, ref), emb_max_abs=float(np.abs(emb - ref).max()), emb_cos=cos,
                      hf_bf16_rel_l2=rel(bf, ref), emb_vs_bf16_rel_l2=rel(emb, bf), ref_max_abs=float(np.abs(ref).max()))
        print(n, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[n].items()}))
    if "--json" in a: json.dump(res, open(a[a.index("--json") + 1], "w"), indent=1)

main()
