"""Inkling-2-Mini checkpoint access shared by the converter, the calibration run and the evaluators.

Everything here is CPU/GPU torch on the HF safetensors checkpoint (no transformers needed).
  open_ckpt(dir)                      -> safe_open handle of model-00001-of-00001.safetensors
  dequant_experts(f, L, e0, e1, dev)  -> (W13 [E,2304,2304] fp32 with rows INTERLEAVED (even = gate, odd = up) exactly as stored,
                                          W2  [E,2304,1152] fp32)   exact dequantisation of NVFP4 (L<20) / MXFP8 (L>=20)
  deinterleave(W13)                   -> rows [gate(1152); up(1152)] (what tools/inkling_ref.py uses)
Semantics: docs/INKLING_SPEC.md section 11.
"""
import os
import torch
from safetensors import safe_open

MODEL = "/workspace/q-volta-models/Inkling-2-Mini"
NL, NE, H, FF = 24, 128, 2304, 1152
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def open_ckpt(d=MODEL):
    return safe_open(os.path.join(d, "model-00001-of-00001.safetensors"), "pt")


def expert_kind(L):
    return "nvfp4" if L < 20 else "mxfp8"


def _deq(f, L, nm, e0, e1, dev):
    p = f"model.llm.layers.{L}.mlp.experts.{nm}"
    wt = f.get_slice(p)[e0:e1].to(dev)
    sc = f.get_slice(p + "_scale")[e0:e1].to(dev)
    E, R = wt.shape[0], wt.shape[1]
    if L < 20:
        amax = f.get_slice(p + "_amax")[e0:e1].to(dev).float()
        den = f.get_tensor(p + "_outer_scale_denominator").to(dev).float()
        lut = E2M1.to(dev)
        wl = wt.long()
        vals = torch.stack([lut[wl & 0xF], lut[wl >> 4]], -1)  # [E,R,C/2,2], low nibble = even element along K
        K = wt.shape[2] * 2
        v = vals.reshape(E, R, K // 16, 16) * sc.float()[..., None] * (amax / den)[:, None, None, None]
        return v.reshape(E, R, K)
    K = wt.shape[2]
    v = wt.float().reshape(E, R, K // 32, 32) * torch.exp2(sc.float() - 127.0)[..., None]
    return v.reshape(E, R, K)


def dequant_experts(f, L, e0, e1, dev="cpu"):
    return _deq(f, L, "w13_weight", e0, e1, dev), _deq(f, L, "w2_weight", e0, e1, dev)


def deinterleave(w13):
    """[..., 2*FF, K] interleaved (even rows gate, odd rows up) -> [gate(FF); up(FF)] along dim -2."""
    return torch.cat([w13[..., 0::2, :], w13[..., 1::2, :]], dim=-2)


def interleave(w13):
    """inverse of deinterleave"""
    g, u = w13[..., :FF, :], w13[..., FF:, :]
    return torch.stack([g, u], dim=-2).reshape(*w13.shape[:-2], 2 * FF, w13.shape[-1])


def moe_act(gu_g, gu_u):
    """expert activation from the (gate, up) pre-activations (spec 8)"""
    return torch.tanh(gu_g / 4.0) * 4.0 * torch.sigmoid(gu_g) * (torch.tanh(gu_u / 25.0) * 25.0)
