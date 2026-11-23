#!/usr/bin/env python3
"""Largest context that fits the 16 GB V100 (device 0, nothing else resident), by bisection on allocation + one real prefill chunk + decode.
llama.cpp: llama_eval --probe-ctx N --kv f16|q8_0|q4_0 (n_ctx=N, flash-attn, ngl 99, ubatch 512).
Q-Volta : qv_run --max-ctx N --kv int4 --decode-bench --prompt P512 --synth-ctx N-8 --n 4 (decode over unwritten KV: speed/VRAM only).
usage: maxctx.py llama MODEL.gguf KV [lo hi] | maxctx.py qv MODEL.qv KV [lo hi]   (run under flock)"""
import json, os, subprocess, sys
eng, model, kv = sys.argv[1:4]
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (4096, 400000)
E = os.environ; DATA = E["DATA"]
def ok(n):
    if eng == "llama":
        cmd = [f"{E['LB']}/llama_eval", "-m", model, "--tokens", f"{DATA}/P2048.i32", "--probe-ctx", str(n), "--kv", kv, "-ngl", "99"]
    else:
        cmd = [E["QV_BIN"], "--model", model, "--max-ctx", str(n), "--kv", kv, "--decode-bench", "--prompt", f"{DATA}/P512.i32", "--synth-ctx", str(n - 8), "--n", "4", "--prefill-chunk", "512"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**E, "CUDA_VISIBLE_DEVICES": "0"})
    return r.returncode == 0 and bool(r.stdout.strip())
assert ok(lo), "lower bound does not fit"
while hi - lo > 1024:
    mid = (lo + hi) // 2 // 1024 * 1024
    if ok(mid): lo = mid
    else: hi = mid
print(json.dumps({"engine": eng, "model": os.path.basename(model), "kv": kv, "max_ctx_fit": lo, "resolution": 1024}))
