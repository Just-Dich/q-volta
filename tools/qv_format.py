"""Q-Volta weight format (v1) — shared by the converter, tests and reference tools.

Files:  <name>.qv.json  (header: config + tensor table)   <name>.qv.bin (256-byte aligned blobs)

Quantized matrix W[N, K] (row = output feature, K = input dim, K % 64 == 0):
  * asymmetric per-group (G = 64 along K):  w = scale * q + min,   q in [0, 2^bits)
  * qweight: uint32[N][K/32][bits]  — each 32-weight chunk occupies exactly `bits` words, as a stack of
    bit-planes of width 4, 2 or 1 (PLANES[bits]). In a plane of width w the chunk uses w words; word j, field f
    (bits [w*f, w*f+w)) holds weight index i = j + w*f of the chunk. Plane value = (q >> shift) & (2^w - 1).
  * sm: float16[N][K/G][2]  = (scale, min)
"""
import numpy as np
import torch

GROUP = 64
CHUNK = 32
# bits -> list of (plane_width, shift)
PLANES = {
    2: [(2, 0)],
    3: [(2, 0), (1, 2)],
    4: [(4, 0)],
    5: [(4, 0), (1, 4)],
    6: [(4, 0), (2, 4)],
    8: [(4, 0), (4, 4)],
}


def quantize_rtn(W: torch.Tensor, bits: int, group: int = GROUP):
    """W fp32 [N,K] (cuda) -> q uint8 [N,K], scale fp16 [N,K/G], mn fp16 [N,K/G]."""
    N, K = W.shape
    assert K % group == 0
    Wg = W.view(N, K // group, group)
    mn = Wg.amin(-1)
    mx = Wg.amax(-1)
    qmax = (1 << bits) - 1
    scale = ((mx - mn) / qmax)
    s16 = scale.half()
    m16 = mn.half()
    sf = s16.float()
    sf = torch.where(sf > 0, sf, torch.ones_like(sf))
    q = torch.round((Wg - m16.float()[..., None]) / sf[..., None]).clamp_(0, qmax).to(torch.uint8)
    s16 = torch.where(s16.float() > 0, s16, torch.ones_like(s16))
    return q.view(N, K), s16, m16


def pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """q uint8 [N,K] -> int32 [N, K/32, bits] (bit pattern = uint32)."""
    N, K = q.shape
    qc = q.view(N, K // CHUNK, CHUNK).to(torch.int64)
    words = []
    for (w, shift) in PLANES[bits]:
        mask = (1 << w) - 1
        field = (qc >> shift) & mask  # [N, C, 32]
        nf = 32 // w
        for j in range(w):
            idx = torch.arange(nf, device=q.device) * w + j  # weight index for field f
            vals = field[:, :, idx]  # [N, C, nf]
            sh = (torch.arange(nf, device=q.device) * w).to(torch.int64)
            words.append((vals << sh).sum(-1))  # [N, C] < 2^32
    out = torch.stack(words, -1)  # [N, C, bits]
    out = torch.where(out >= 2**31, out - 2**32, out)
    return out.to(torch.int32).contiguous()


def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """int32 [N, C, bits] -> uint8 [N, C*32]."""
    N, C, B = packed.shape
    assert B == bits
    p = packed.to(torch.int64) & 0xFFFFFFFF
    q = torch.zeros(N, C, CHUNK, dtype=torch.int64, device=packed.device)
    wi = 0
    for (w, shift) in PLANES[bits]:
        mask = (1 << w) - 1
        nf = 32 // w
        for j in range(w):
            word = p[:, :, wi]
            wi += 1
            for f in range(nf):
                q[:, :, j + w * f] |= ((word >> (w * f)) & mask) << shift
    return q.view(N, C * CHUNK).to(torch.uint8)


def dequant(packed: torch.Tensor, sm: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    """-> fp32 [N, K] exactly as the engine is meant to reconstruct it."""
    q = unpack(packed, bits).float()
    N, K = q.shape
    s = sm[..., 0].float()
    m = sm[..., 1].float()
    return (q.view(N, K // group, group) * s[..., None] + m[..., None]).view(N, K)


class Writer:
    ALIGN = 256

    def __init__(self, path_prefix: str):
        self.prefix = path_prefix
        self.f = open(path_prefix + ".qv.bin", "wb")
        self.off = 0
        self.tensors = {}

    def _blob(self, arr: np.ndarray) -> int:
        pad = (-self.off) % self.ALIGN
        if pad:
            self.f.write(b"\0" * pad)
            self.off += pad
        start = self.off
        b = np.ascontiguousarray(arr).tobytes()
        self.f.write(b)
        self.off += len(b)
        return start

    def add_quant(self, name, packed: torch.Tensor, sm: torch.Tensor, bits, shape):
        pk = packed.cpu().numpy()
        smn = sm.cpu().numpy().astype(np.float16)
        oq = self._blob(pk)
        os_ = self._blob(smn)
        self.tensors[name] = dict(kind="q", bits=bits, group=GROUP, shape=list(shape),
                                  off_q=oq, nbytes_q=pk.nbytes, off_sm=os_, nbytes_sm=smn.nbytes)
        return pk.nbytes + smn.nbytes

    def add_dense(self, name, t: torch.Tensor, dtype: str):
        npdt = {"f32": np.float32, "f16": np.float16}[dtype]
        a = t.detach().float().cpu().numpy().astype(npdt)
        o = self._blob(a)
        self.tensors[name] = dict(kind=dtype, shape=list(a.shape), off=o, nbytes=a.nbytes)
        return a.nbytes

    def close(self, config: dict):
        import json
        self.f.close()
        with open(self.prefix + ".qv.json", "w") as fh:
            json.dump(dict(format="qvolta-v1", config=config, tensors=self.tensors), fh, indent=1)
