#!/usr/bin/env python3
"""Make a copy of a .qv container whose MTP head (mtp.* tensors, the last blobs of the data section) is re-quantized at
other bits / group.  Used to measure MTP draft acceptance vs MTP bits without re-running the whole conversion.
usage: qv_remtp.py BASE.qv OUT.qv BITS [GROUP=64] [--clip]"""
import json, os, struct, sys, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_format as F
import convert as C

args = [a for a in sys.argv[1:] if not a.startswith("--")]
base, out, bits = args[0], args[1], int(args[2])
group = int(args[3]) if len(args) > 3 else 64
clip = "--clip" in sys.argv
with open(base, "rb") as f:
    magic = f.read(8)
    jl, doff = struct.unpack("<QQ", f.read(16))
    hdr = json.loads(f.read(jl))
tens = hdr["tensors"]
mtp = {n: t for n, t in tens.items() if n.startswith("mtp.")}


def tail_off(t):
    return t["off"] if "off" in t else t["off_q"]


mo = min(tail_off(t) for t in mtp.values())
assert all(tail_off(t) < mo for n, t in tens.items() if n not in mtp), "mtp tensors are not the tail of the data section"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
R = C.Remote("/workspace/q-volta-models/Qwen3.8-27B")
shutil.copyfile(base, out)
with open(out, "r+b") as f:
    f.truncate(doff + mo)
    f.seek(doff + mo)
    off = mo

    def blob(a):
        global off
        pad = (-off) % 256
        f.write(b"\0" * pad)
        off += pad
        s = off
        f.write(np.ascontiguousarray(a).tobytes())
        off += a.nbytes
        return s

    for n in sorted(mtp):
        t = R.fetch(n)
        b, g = C.tensor_cfg(n, {"mlp_gu": bits, "mlp_down": bits, "gdn": bits, "attn": bits, "embed": 4, "head": 6, "mtp": bits}, {}, group)
        if b in ("f32", "f16"):
            old = tens[n]
            # dense tensors are re-emitted unchanged: copy from the base file
            with open(base, "rb") as bf:
                bf.seek(doff + old["off"])
                raw = bf.read(old["nbytes"])
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            tens[n]["off"] = off
            f.write(raw)
            off += len(raw)
        else:
            pk, sm = C.quant_rows(t, b, g, dev, "rtn", clip=clip)
            pkn, smn = pk.cpu().numpy(), sm.cpu().numpy().astype(np.float16)
            oq = blob(pkn)
            os_ = blob(smn)
            tens[n] = dict(kind="q", bits=b, group=g, shape=list(t.shape), off_q=oq, nbytes_q=pkn.nbytes, off_sm=os_, nbytes_sm=smn.nbytes)
    js = json.dumps(hdr, indent=1).encode()
    assert 24 + len(js) <= doff, "header outgrew its padding"
    f.seek(0)
    f.write(magic + struct.pack("<QQ", len(js), doff) + js + b"\0" * max(0, jl - len(js)))
print(f"wrote {out}: {os.path.getsize(out)} bytes (MTP at {bits} bit, group {group})")
