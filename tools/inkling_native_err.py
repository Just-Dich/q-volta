#!/usr/bin/env python3 -I
"""Profile (a) 'native': how lossless is re-packing the checkpoint-native NVFP4 / MXFP8 expert weights into the uniform asymmetric
bit-plane format? Weight-space relative RMS error ||Wq - W||_F / ||W||_F of RTN (+clip) at several bits/groups on sampled experts.
  python3 -I tools/inkling_native_err.py --layers 0,10,19,20,23 --experts 8 --out native_err.json"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_quant as Q
import inkling_ckpt as K

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0,10,19,20,23"); ap.add_argument("--experts", type=int, default=8); ap.add_argument("--out", required=True)
a = ap.parse_args()
f = K.open_ckpt()
cfgs = [(3, 128), (4, 128), (4, 64), (4, 32), (5, 64), (5, 32), (5, 16), (6, 64), (8, 64), (8, 32)]
res = {}
for L in (int(x) for x in a.layers.split(",")):
    w13, w2 = K.dequant_experts(f, L, 0, a.experts, "cuda")
    r = {}
    for name, M_ in (("w13", w13), ("w2", w2)):
        M_ = M_.to(torch.bfloat16).float().reshape(-1, M_.shape[-1])
        for b, g in cfgs:
            if M_.shape[1] % g:
                continue
            q, S, M, rs = Q.rtn(M_, b, g, clip=True, meta8=True)
            dq = Q.dequant_g(q, S, M, g, b, rs)
            r[f"{name}_b{b}_g{g}"] = float((dq - M_).norm() / M_.norm())
    res[f"L{L}_{K.expert_kind(L)}"] = r
    print(L, K.expert_kind(L), {k: round(v, 5) for k, v in r.items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
