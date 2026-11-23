#!/usr/bin/env python3
"""Calibration tokens for GPTQ / sensitivity: [N, 1024] int32.  Mix: wikitext-2 TRAIN windows (the eval is the test split),
source code (python stdlib + C/C++ headers), and chat-template turns built from wikitext-train paragraphs.
usage: make_calib.py OUT.npy [n_wiki=16 n_code=8 n_chat=8]"""
import glob, random, sys
import numpy as np
from tokenizers import Tokenizer

out = sys.argv[1]
nw, nc, nt = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (16, 8, 8)
L = 1024
tok = Tokenizer.from_file("/workspace/q-volta-models/Qwen3.8-27B/tokenizer.json")
import os
rng = random.Random(int(os.environ.get("QV_CALIB_SEED", "1234")))
wt = open("/workspace/q-volta-models/calib/wt_train.txt").read()
ids = tok.encode(wt).ids
rows = []
for _ in range(nw):
    s = rng.randrange(0, len(ids) - L)
    rows.append(ids[s:s + L])
files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
rng.shuffle(files)
code, n = [], 0
for f in files:
    try:
        code += tok.encode(open(f, errors="ignore").read()).ids
    except Exception:
        continue
    if len(code) > nc * L * 4:
        break
for _ in range(nc):
    s = rng.randrange(0, len(code) - L)
    rows.append(code[s:s + L])
paras = [p.strip() for p in wt.split("\n") if len(p.strip()) > 300 and not p.strip().startswith("=")]
for _ in range(nt):
    txt = ""
    while True:
        a = rng.choice(paras)
        b = rng.choice(paras)
        txt += f"<|im_start|>user\nSummarize or continue this passage:\n{a[:700]}<|im_end|>\n<|im_start|>assistant\n{b[:900]}<|im_end|>\n"
        t = tok.encode(txt, add_special_tokens=False).ids
        if len(t) >= L:
            rows.append(t[:L])
            break
arr = np.array(rows, dtype=np.int32)
np.save(out, arr)
print(arr.shape, "wiki", nw, "code", nc, "chat", nt)
