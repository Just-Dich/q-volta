#!/usr/bin/env python3
"""Estimated .qv size (GiB) of a per-class bit/group plan (meta8). classes: gu, down, qkvz (gdn in_proj_qkv/z + attn q/k/v), o (gdn out_proj + attn o_proj), embed, head, mtp.
usage: plan_size.py 'gu=3:64,down=4:128,qkvz=3:64,o=4:128,embed=4:128,head=5:128,mtp=3:128'"""
import sys
P = dict(gu=64 * 2 * 17408 * 5120, down=64 * 5120 * 17408, qkvz=48 * (10240 + 6144) * 5120 + 16 * (12288 + 2048) * 5120,
         o=48 * 5120 * 6144 + 16 * 5120 * 6144, embed=248320 * 5120, head=248320 * 5120,
         mtp=(10240 * 5120 + 12288 * 5120 + 2 * 1024 * 5120 + 5120 * 6144 + 3 * 5120 * 17408))
DENSE = 0.012 + 0.05  # norms/conv/in_proj_a,b + container overhead observed (e1: est 10.39 -> actual 10.44)
def size(spec):
    t = 0
    for kv in spec.split(","):
        k, v = kv.split("=")
        b, g = (int(x) for x in v.split(":"))
        t += P[k] * (b + 16 / g) / 8
    return t / 2**30 + DENSE
if __name__ == "__main__":
    print(f"{size(sys.argv[1]):.2f} GiB")
