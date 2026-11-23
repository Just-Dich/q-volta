#!/usr/bin/env python3
"""Merge a legacy <prefix>.qv.json + <prefix>.qv.bin pair into one self-contained <prefix>.qv file.

Layout: "QVOLTA1\\0" | u64 json_len | u64 data_off | json | zero pad | data (data_off aligned to 2 MiB).
Tensor offsets in the JSON stay relative to the data section (the loader rebases them).
usage: qv_pack.py <prefix> [out.qv] [--from-dir HF_DIR]   (--from-dir embeds tokenizer / chat template / generation config, see tools/qv_embed.py)
"""
import os, struct, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALIGN = 2 << 20

def pack(prefix, out=None, delete_src=False, blobs=None, meta=None):
    """blobs: optional {name: bytes} embedded after the tensor data (tokenizer.json, chat_template.jinja, generation_config.json, ...);
    meta: optional dict for the index "meta" object (docs/QV_FORMAT.md)."""
    out = out or prefix + ".qv"
    js = open(prefix + ".qv.json", "rb").read()
    tend = os.path.getsize(prefix + ".qv.bin")
    if blobs or meta:
        import json, qv_blobs
        idx = json.loads(js)
        offs, _ = qv_blobs.layout(tend, {n: len(b) for n, b in (blobs or {}).items()})
        import hashlib
        entries = {n: dict(off=offs[n], nbytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n, b in (blobs or {}).items()}
        js = qv_blobs.dump_index(qv_blobs.merge_index(idx, entries, qv_blobs.base_meta(idx, meta)))
    hdr = 24 + len(js)
    data_off = (hdr + ALIGN - 1) // ALIGN * ALIGN
    tmp = out + ".tmp"
    with open(tmp, "wb") as o:
        o.write(b"QVOLTA1\0" + struct.pack("<QQ", len(js), data_off))
        o.write(js)
        o.write(b"\0" * (data_off - hdr))
        with open(prefix + ".qv.bin", "rb") as b:
            shutil.copyfileobj(b, o, 64 << 20)
        for n, bl in (blobs or {}).items():
            o.write(b"\0" * (data_off + offs[n] - o.tell()))
            o.write(bl)
    os.replace(tmp, out)
    print(f"wrote {out}: {os.path.getsize(out)} bytes, data_off={data_off}")
    if delete_src:
        os.remove(prefix + ".qv.json")
        os.remove(prefix + ".qv.bin")


def main():
    args = [a for a in sys.argv[1:]]
    blobs = meta = None
    if "--from-dir" in args:
        i = args.index("--from-dir")
        import qv_blobs
        blobs, meta = qv_blobs.collect_hf_dir(args[i + 1])
        del args[i:i + 2]
    pack(args[0], args[1] if len(args) > 1 else None, blobs=blobs, meta=meta)

if __name__ == "__main__":
    main()
