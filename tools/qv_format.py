"""Q-Volta weight format (v1) — shared by the converter, tests and reference tools.

Files:  <name>.qv.json  (header: config + tensor table)   <name>.qv.bin (256-byte aligned blobs)

Container (.qv, tools/qv_pack.py): "QVOLTA1\0" | u64 json_len | u64 data_off | json | pad | data. The JSON has
  config (model config + "quant": {method, group, clip, plan, calib}) and tensors{name: entry}. Entry kinds: "q" (below),
  "f32", "f16" (dense) and the OPTIONAL SECTION "lora_f16": an unmerged PEFT LoRA adapter stored as fp16 pairs
  lora.<base weight name>.A [r,K] and lora.<base weight name>.B [N,r]; config.lora = {r, alpha, scale, merged: false,
  source}. Runtimes that do not know the section ignore it (tensors are looked up by name; sizes are in the table), so a
  file with a LoRA section loads unchanged. The LoRA runtime (W x + scale * B (A x)) is NOT implemented yet.
  Per-tensor "group" (64 or 128) and "bits" (2..8) are in each "q" entry, so bit allocation may differ per tensor.

Quantized matrix W[N, K] (row = output feature, K = input dim, K % 64 == 0):
  * asymmetric per-group (G = 64 or 128 along K, per tensor):  w = scale * q + min,   q in [0, 2^bits)
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


def dequant(packed: torch.Tensor, sm: torch.Tensor, bits: int, group: int = GROUP, rs: torch.Tensor = None) -> torch.Tensor:
    """-> fp32 [N, K] exactly as the engine is meant to reconstruct it.
    sm fp16 [N,G,2] (scale, min), or with rs (fp16 [N] row scale) the meta8 encoding sm uint16 [N,G] = s8 | z8 << 8."""
    q = unpack(packed, bits).float()
    N, K = q.shape
    if rs is None:
        s = sm[..., 0].float()
        m = sm[..., 1].float()
    else:
        import qv_quant
        s, m = qv_quant.meta8_decode((sm.to(torch.int32) & 255), (sm.to(torch.int32) >> 8) & 255, rs, bits)
    return (q.view(N, K // group, group) * s[..., None] + m[..., None]).view(N, K)


class Writer:
    ALIGN = 256

    def __init__(self, path_prefix: str, resume=None, null=False):
        """resume = (offset, tensors dict) of a previous slice: the .bin is reopened and truncated to that offset."""
        self.prefix = path_prefix
        self.extra_config = {}
        self.null = null  # sizes/offsets are tracked but nothing is written (evaluation-only runs)
        if null:
            self.f = None
            self.off, self.tensors = (0, {}) if resume is None else resume
        elif resume is None:
            self.f = open(path_prefix + ".qv.bin", "wb")
            self.off = 0
            self.tensors = {}
        else:
            self.f = open(path_prefix + ".qv.bin", "r+b")
            self.off, self.tensors = resume
            self.f.truncate(self.off)
            self.f.seek(self.off)

    def flush_state(self):
        if self.f:
            self.f.flush()
        return (self.off,)

    def _blob(self, arr: np.ndarray) -> int:
        pad = (-self.off) % self.ALIGN
        if pad:
            if self.f:
                self.f.write(b"\0" * pad)
            self.off += pad
        start = self.off
        if self.f:
            self.f.write(np.ascontiguousarray(arr).tobytes())
        self.off += arr.nbytes
        return start

    def add_quant(self, name, packed: torch.Tensor, sm: torch.Tensor, bits, shape, group=GROUP, rs=None):
        pk = packed.cpu().numpy()
        smn = sm.cpu().numpy().astype(np.float16 if rs is None else np.uint16)
        oq = self._blob(pk)
        os_ = self._blob(smn)
        ent = dict(kind="q", bits=bits, group=group, shape=list(shape),
                   off_q=oq, nbytes_q=pk.nbytes, off_sm=os_, nbytes_sm=smn.nbytes)
        nb = pk.nbytes + smn.nbytes
        if rs is not None:  # meta8: uint16 [N][G] (s8 | z8<<8) + fp16 row scales [N]
            rsn = rs.cpu().numpy().astype(np.float16)
            ent.update(meta="s8z8", off_rs=self._blob(rsn), nbytes_rs=rsn.nbytes)
            nb += rsn.nbytes
        self.tensors[name] = ent
        return nb

    def add_dense(self, name, t: torch.Tensor, dtype: str, kind=None):
        npdt = {"f32": np.float32, "f16": np.float16}[dtype]
        a = t.detach().float().cpu().numpy().astype(npdt)
        o = self._blob(a)
        self.tensors[name] = dict(kind=kind or dtype, shape=list(a.shape), off=o, nbytes=a.nbytes)
        return a.nbytes

    def close(self, config: dict):
        import json
        if self.null:
            return
        self.f.close()
        with open(self.prefix + ".qv.json", "w") as fh:
            config = dict(config, **self.extra_config)
            json.dump(dict(format="qvolta-v1", config=config, tensors=self.tensors), fh, indent=1)


# ---------------------------------------------------------------------------------------------------------------------
# Single-file container written directly (no .json/.bin pair, so a 13 GB model never exists twice on disk) and a reader.
# Added for Inkling (docs/INKLING_FORMAT.md). Stacked expert tensors: a "q" entry whose shape is [E, N, K]; qweight is
# uint32[E][N][K/32][bits], sm is [E][N][K/G] (fp16 pairs or meta8 uint16), rs is fp16 [E][N]; every expert's slice is a
# multiple of 256 bytes, so expert e of a tensor starts at off + e * (nbytes / E).
# ---------------------------------------------------------------------------------------------------------------------
class ContainerWriter(Writer):
    HDR = 8 << 20  # header region (magic, lengths, json); data section starts here (2 MiB aligned)

    def __init__(self, path, resume=None):
        self.prefix = path
        self.path = path
        self.extra_config = {}
        self.null = False
        if resume is None:
            self.f = open(path, "wb")
            self.f.seek(self.HDR)
            self.off, self.tensors = 0, {}
        else:
            self.f = open(path, "r+b")
            self.off, self.tensors = resume
            self.f.truncate(self.HDR + self.off)
            self.f.seek(self.HDR + self.off)

    def add_raw(self, name, kind, shape, parts: dict, **extra):
        """Passthrough (checkpoint-native) tensor, e.g. kind 'nvfp4': parts = {q: uint8 codes, sm: e4m3 scales (uint8),
        rs: fp32 per-expert global factor}; they are stored in off_q/off_sm/off_rs. Returns bytes written."""
        ent = dict(kind=kind, shape=list(shape), **extra)
        nb = 0
        for key, off_name, nb_name in (("q", "off_q", "nbytes_q"), ("sm", "off_sm", "nbytes_sm"), ("rs", "off_rs", "nbytes_rs")):
            if key in parts:
                a = np.ascontiguousarray(parts[key])
                ent[off_name] = self._blob(a)
                ent[nb_name] = a.nbytes
                nb += a.nbytes
        self.tensors[name] = ent
        return nb

    def close(self, config: dict, blobs=None, meta=None):
        """blobs: optional {name: bytes} (tokenizer.json, chat_template.jinja, ...) appended after the tensors (docs/QV_FORMAT.md);
        meta: optional dict for the index "meta" object. Without either the file is a plain format_version 1 container."""
        import json, struct
        import qv_blobs
        config = dict(config, **self.extra_config)
        idx = dict(format="qvolta-v1", config=config, tensors=self.tensors)
        if blobs or meta:
            tend = qv_blobs.tensor_end(idx)
            self.f.truncate(self.HDR + tend)
            idx = qv_blobs.merge_index(idx, qv_blobs.append_blobs(self.f, self.HDR, tend, blobs or {}), qv_blobs.base_meta(idx, meta))
        js = json.dumps(idx, indent=None, separators=(",", ":")).encode()
        assert 24 + len(js) <= self.HDR, "header region too small"
        self.f.flush()
        self.f.seek(0)
        self.f.write(b"QVOLTA1\0" + struct.pack("<QQ", len(js), self.HDR) + js)
        self.f.close()


class Reader:
    """Read-only view of a .qv container (numpy memmap). Tensor offsets in the JSON are relative to the data section."""

    def __init__(self, path):
        import json, struct
        self.path = path
        with open(path, "rb") as fh:
            magic = fh.read(8)
            assert magic == b"QVOLTA1\0", "not a .qv container"
            jl, self.data_off = struct.unpack("<QQ", fh.read(16))
            hdr = json.loads(fh.read(jl))
        self.config, self.tensors = hdr["config"], hdr["tensors"]
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def _arr(self, off, nbytes, dtype, shape):
        a = self.mm[self.data_off + off: self.data_off + off + nbytes]
        return a.view(dtype).reshape(shape)

    def dense(self, name):
        t = self.tensors[name]
        dt = {"f32": np.float32, "f16": np.float16}[t["kind"]]
        return torch.from_numpy(np.array(self._arr(t["off"], t["nbytes"], dt, t["shape"])))

    def quant(self, name, e=None):
        """-> (packed int32 [..N, K/32, bits], sm, rs|None, bits, group). For a stacked tensor pass the expert index e (or None for all)."""
        t = self.tensors[name]
        assert t["kind"] == "q"
        bits, group, shape = t["bits"], t["group"], t["shape"]
        E = shape[0] if len(shape) == 3 else None
        N, K = shape[-2:] if E else shape
        meta8 = t.get("meta") == "s8z8"
        pk = self._arr(t["off_q"], t["nbytes_q"], np.int32, ([E] if E else []) + [N, K // 32, bits])
        sm = self._arr(t["off_sm"], t["nbytes_sm"], np.uint16 if meta8 else np.float16,
                       ([E] if E else []) + ([N, K // group] if meta8 else [N, K // group, 2]))
        rs = self._arr(t["off_rs"], t["nbytes_rs"], np.float16, ([E] if E else []) + [N]) if meta8 else None
        if E and e is not None:
            pk, sm, rs = pk[e], sm[e], (rs[e] if rs is not None else None)
        pk, sm = torch.from_numpy(np.array(pk)), torch.from_numpy(np.array(sm).view(np.uint16 if meta8 else np.float16))
        if meta8:
            sm = sm.view(torch.int16) if False else sm.to(torch.int32)
        return pk, sm, (torch.from_numpy(np.array(rs)) if rs is not None else None), bits, group

    def dequant(self, name, e=None, dev="cpu"):
        """Exact engine reconstruction (fp32) of a quant tensor (one expert if e is given)."""
        pk, sm, rs, bits, group = self.quant(name, e)
        pk, sm = pk.to(dev), sm.to(dev)
        rs = None if rs is None else rs.to(dev)
        return dequant(pk, sm, bits, group, rs)
