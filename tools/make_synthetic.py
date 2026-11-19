#!/usr/bin/env python3
"""Write a synthetic Q-Volta model with the exact Qwen3.8-27B tensor shapes and bit plan but random weight bits.
For PERFORMANCE work only (kernel timing does not depend on weight values) — never for correctness/quality."""
import argparse, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_format as F
from convert import plan_bits, layer_names, LM

ap = argparse.ArgumentParser()
ap.add_argument("--hf-dir", default="/workspace/q-volta-models/Qwen3.8-27B")
ap.add_argument("--out", default="/workspace/q-volta-models/qv/synthetic-e0")
ap.add_argument("--bits", default="mlp_gu=3,mlp_down=3,gdn=4,attn=4,embed=4,head=6")
a = ap.parse_args()
bp = {k: int(v) for k, v in (kv.split("=") for kv in a.bits.split(","))}
cfg = json.load(open(os.path.join(a.hf_dir, "config.json")))["text_config"]
idx = json.load(open(os.path.join(a.hf_dir, "model.safetensors.index.json")))["weight_map"]
# shapes from the real safetensors headers of the downloaded shards are not needed: derive from config
H, I, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
hk, hv, dk, dv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
conv = 2 * hk * dk + hv * dv


def shape(n):
    s = n.split(".")[-2] if n.endswith("weight") else n.split(".")[-1]
    table = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I), "in_proj_qkv": (conv, H),
             "in_proj_z": (hv * dv, H), "out_proj": (H, hv * dv), "in_proj_a": (hv, H), "in_proj_b": (hv, H),
             "q_proj": (2 * nh * hd, H), "k_proj": (nkv * hd, H), "v_proj": (nkv * hd, H), "o_proj": (H, nh * hd),
             "input_layernorm": (H,), "post_attention_layernorm": (H,), "q_norm": (hd,), "k_norm": (hd,),
             "norm": (dv,), "A_log": (hv,), "dt_bias": (hv,), "conv1d": (conv, 4)}
    return table[s]


rng = np.random.default_rng(0)
W = F.Writer(a.out)


def put(name, shp, bits):
    if bits in ("f32", "f16"):
        val = {"A_log": -1.0, "dt_bias": 0.0}.get(name.split(".")[-1], 1.0)
        if name.endswith("conv1d.weight"):
            t = torch.full(shp, 0.25)
        elif bits == "f16":
            t = torch.from_numpy(rng.standard_normal(shp, dtype=np.float32) * 0.01)
        else:
            t = torch.full(shp, val)
        W.add_dense(name, t, bits)
        return
    N, K = shp
    pk = torch.from_numpy(rng.integers(-2**31, 2**31 - 1, size=(N, K // 32, bits), dtype=np.int64).astype(np.int32))
    s = 0.02 / ((1 << bits) - 1)
    sm = torch.tensor([s, -0.01], dtype=torch.float16).expand(N, K // 64, 2).contiguous()
    W.add_quant(name, pk, sm, bits, shp)


put(LM + "embed_tokens.weight", (V, H), bp["embed"])
for i, lt in enumerate(cfg["layer_types"]):
    for n in layer_names(i, lt):
        put(n, shape(n), plan_bits(n, bp))
put(LM + "norm.weight", (H,), "f32")
put("lm_head.weight", (V, H), bp["head"])
out_cfg = dict(cfg, synthetic=True, bit_plan=bp)
W.close(out_cfg)
print("wrote", a.out, os.path.getsize(a.out + ".qv.bin") / 2**30, "GiB")
