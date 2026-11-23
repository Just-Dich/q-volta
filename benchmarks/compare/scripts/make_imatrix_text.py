#!/usr/bin/env python3
"""Decode the SAME calibration windows used for Q-Volta's GPTQ (calib32.npy: 32 x 1024 tokens = 16 wikitext-train,
8 code, 8 chat windows; tools/make_calib.py) back to text for llama-imatrix (which re-tokenises: boundaries shift slightly)."""
import sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/q-volta-models/Qwen3.8-27B")
c = np.load("/workspace/q-volta-models/calib/calib32.npy")
print(c.shape, file=sys.stderr)
open(sys.argv[1], "w").write("\n".join(tok.decode(r.tolist()) for r in c))
