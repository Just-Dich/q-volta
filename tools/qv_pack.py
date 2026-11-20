#!/usr/bin/env python3
"""Merge a legacy <prefix>.qv.json + <prefix>.qv.bin pair into one self-contained <prefix>.qv file.

Layout: "QVOLTA1\\0" | u64 json_len | u64 data_off | json | zero pad | data (data_off aligned to 2 MiB).
Tensor offsets in the JSON stay relative to the data section (the loader rebases them).
usage: qv_pack.py <prefix> [out.qv]
"""
import os, struct, sys, shutil

ALIGN = 2 << 20

def pack(prefix, out=None, delete_src=False):
    out = out or prefix + ".qv"
    js = open(prefix + ".qv.json", "rb").read()
    hdr = 24 + len(js)
    data_off = (hdr + ALIGN - 1) // ALIGN * ALIGN
    tmp = out + ".tmp"
    with open(tmp, "wb") as o:
        o.write(b"QVOLTA1\0" + struct.pack("<QQ", len(js), data_off))
        o.write(js)
        o.write(b"\0" * (data_off - hdr))
        with open(prefix + ".qv.bin", "rb") as b:
            shutil.copyfileobj(b, o, 64 << 20)
    os.replace(tmp, out)
    print(f"wrote {out}: {os.path.getsize(out)} bytes, data_off={data_off}")
    if delete_src:
        os.remove(prefix + ".qv.json")
        os.remove(prefix + ".qv.bin")


def main():
    pack(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

if __name__ == "__main__":
    main()
