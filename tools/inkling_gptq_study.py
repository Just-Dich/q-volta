#!/usr/bin/env python3 -I
"""Expert-quantiser study on a few layers: held-out (val0) relative output error ||(W-Wq)x||/||Wx|| of w13 / w2 for RTN, RTN+clip and
GPTQ with Hessian shrinkage prior c, at given bits / group. Reuses convert_inkling.quant_layer_experts (nothing is written).
  python3 -I tools/inkling_gptq_study.py --layers 3,14 --bits 3 --group 128 --variants rtn,gptq:0,gptq:256,gptq:inf --out study.json"""
import argparse, json, os, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_inkling as CI
import inkling_ckpt as K

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="3,14"); ap.add_argument("--bits", type=int, default=3); ap.add_argument("--group", type=int, default=128)
ap.add_argument("--variants", default="rtn,gptq:0,gptq:256,gptq:inf"); ap.add_argument("--out", required=True)
ap.add_argument("--calib", default="/workspace/q-volta-models/scratch-inkling/calib"); ap.add_argument("--seqs", default="wiki0,wiki1,code0,chat0")
a = ap.parse_args()
f = K.open_ckpt()
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for L in (int(x) for x in a.layers.split(",")):
    X, I = CI.load_calib(a.calib, a.seqs.split(","), L)
    val = CI.load_calib(a.calib, ["val0"], L)
    for v in a.variants.split(","):
        key = f"L{L}_b{a.bits}_g{a.group}_{v}"
        if key in res:
            continue
        m, _, c = v.partition(":")
        args = types.SimpleNamespace(expert_bits=a.bits, expert_group=a.group, method=m, prior=float("inf") if c == "inf" else float(c or 0), clip=True, expert_batch=8)
        _, st = CI.quant_layer_experts(f, L, X, I, args, "cuda", print, val)
        st.pop("count")
        res[key] = st
        print(key, json.dumps(st), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
