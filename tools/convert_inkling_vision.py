#!/usr/bin/env python3 -I
"""Inkling-2-Mini vision tower -> SIDECAR container <model>-vision.qv (docs/INKLING_VISION.md section 8).

The text model's .qv does not contain the vision tower (0.64 GB of bf16 in the checkpoint under `model.visual.*`). This writes it as a
separate container with the same envelope as the text .qv ("QVOLTA1\\0" | u64 json_len | u64 data_off | json | pad | data) and
format tag "qvolta-vision-v1". Tensor names are the checkpoint names. Dtypes:
  --quant fp16 (default): matrices and embedding tables fp16 (bf16 -> fp16 is exact for |w| >= 6e-5 and < 65504; the converter reports
                          any value that would overflow/flush), small vectors (LayerNorm, biases, layer-scale gamma, class tokens) fp32.
  --quant int8:           the large [N,K] matrices (q/k/v/out, fc1/fc2, proj_in/out, patch_embedding) as int8 with one fp32 scale per row
                          (kind "i8": nbytes = N*K int8 + N*4 scale bytes in a second blob); the runtime expands to fp16 at load.
usage: convert_inkling_vision.py MODEL_DIR OUT.qv [--quant fp16|int8]
"""
import json, os, struct, sys
import numpy as np
import torch
from safetensors import safe_open

ALIGN = 256
DATA_ALIGN = 2 << 20


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    quant = "fp16"
    if "--quant" in sys.argv:
        quant = sys.argv[sys.argv.index("--quant") + 1]
        a = [x for x in a if x != quant]
    model, out = a[0], a[1]
    cfg = json.load(open(os.path.join(model, "config.json")))
    vc = cfg["vision_config"]
    f = safe_open(os.path.join(model, "model-00001-of-00001.safetensors"), "pt")
    keys = sorted(k for k in f.keys() if k.startswith("model.visual."))
    tensors, blobs, off = {}, [], 0
    stats = {"overflow": 0, "flush": 0, "n": 0}

    def add(name, arr, kind, extra=None):
        nonlocal off
        b = arr.tobytes()
        pad = (-len(b)) % ALIGN
        ent = {"kind": kind, "shape": list(arr.shape), "off": off, "nbytes": len(b)}
        if extra: ent.update(extra)
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
        return ent

    for k in keys:
        t = f.get_tensor(k).float()
        big = t.ndim == 2 and t.shape[0] >= 256 and t.shape[1] >= 256 and "position_embedding" not in k
        pos = "position_embedding" in k
        if t.ndim == 1 or k.endswith("class_embedding") or k.endswith("cls_position_embedding"):
            tensors[k] = add(k, t.numpy().astype("<f4"), "f32")
        elif quant == "int8" and big:
            s = t.abs().amax(1).clamp(min=1e-12) / 127.0
            q = torch.round(t / s[:, None]).clamp(-127, 127).to(torch.int8)
            ent = add(k, q.numpy(), "i8")
            ent["scale"] = add(k + ".scale", s.numpy().astype("<f4"), "f32")
            tensors[k] = ent
        else:
            h = t.to(torch.float16)
            stats["overflow"] += int((~torch.isfinite(h)).sum())
            stats["flush"] += int(((h == 0) & (t != 0)).sum()); stats["n"] += t.numel()
            tensors[k] = add(k, h.numpy().astype("<f2"), "f16")
    hdr = {"format": "qvolta-vision-v1", "config": {
        "arch": "inkling2_vision", "hidden_size": vc["hidden_size"], "num_hidden_layers": vc["num_hidden_layers"],
        "num_attention_heads": vc["num_attention_heads"], "mlp_ratio": vc["mlp_ratio"], "layer_norm_eps": vc["layer_norm_eps"],
        "patch_size": vc["native_patch_size"], "merge_size": vc["adapter_merge_size"], "posemb_grid_size": vc["posemb_grid_size"],
        "projection_hidden_dim": vc["projection_hidden_dim"], "text_hidden_size": vc["text_hidden_size"],
        "image_token_id": 200054, "rope_theta": 10000.0, "layer_scale_init": 0.1, "hidden_act": "gelu", "quant": quant,
        "image_mean": 0.5, "image_std": 0.5, "min_side": 40, "max_side": 4096},
        "tensors": tensors}
    js = json.dumps(hdr).encode()
    h0 = 24 + len(js)
    data_off = (h0 + DATA_ALIGN - 1) // DATA_ALIGN * DATA_ALIGN
    with open(out + ".tmp", "wb") as o:
        o.write(b"QVOLTA1\0" + struct.pack("<QQ", len(js), data_off))
        o.write(js); o.write(b"\0" * (data_off - h0))
        for b in blobs: o.write(b)
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {os.path.getsize(out)/1e9:.3f} GB, {len(tensors)} tensors, quant={quant}, fp16 overflow={stats['overflow']}, "
          f"flushed-to-zero={stats['flush']} of {stats['n']}")


main()
