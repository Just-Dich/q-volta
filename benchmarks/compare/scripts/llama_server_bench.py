#!/usr/bin/env python3
"""Start llama-server (optionally with MTP speculation), send the benchmark prompts as TOKEN IDS (greedy, 128 new tokens),
print one JSON line per request with the server's own timings. Run under `flock /tmp/qv_bench.lock` by speed.py.
usage: llama_server_bench.py MODEL.gguf SPEC(none|mtp) NDRAFT KV(f16|q8_0|q4_0) REPS prompt1.i32,prompt2.i32,..."""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
model, spec, nd, kv, reps, prompts = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6].split(",")
LB = os.environ["LB"]; port = 18080 + os.getpid() % 500
cmd = [f"{LB}/llama-server", "-m", model, "-ngl", "99", "-fa", "on", "-c", "4608", "--port", str(port), "--parallel", "1",
       "-ctk", kv, "-ctv", kv, "--no-warmup", "--no-webui", "--cache-ram", "0"]
if spec == "mtp":
    cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(nd)]
log = open(os.environ.get("SERVER_LOG", "/tmp/llama_server_bench.log"), "w")
p = subprocess.Popen(cmd, stdout=log, stderr=log)
def post(path, obj):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception:
            if p.poll() is not None: sys.exit("server died, see log")
            time.sleep(1)
    for pf in prompts:
        ids = np.fromfile(pf, dtype=np.int32).tolist()
        for rep in range(reps + 1):  # first request = warm-up (not reported)
            o = post("/completion", {"prompt": ids, "n_predict": 128, "temperature": 0, "top_k": 1, "cache_prompt": False,
                                     "ignore_eos": True, "n_probs": 0})
            t = o["timings"]
            if rep == 0: continue
            print(json.dumps({"engine": "llama.cpp-server", "model": os.path.basename(model), "spec": spec, "ndraft": nd, "kv": kv,
                              "prompt": os.path.basename(pf), "prompt_n": t["prompt_n"], "prompt_tok_s": t["prompt_per_second"],
                              "predicted_n": t["predicted_n"], "decode_tok_s": t["predicted_per_second"],
                              "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}), flush=True)
finally:
    p.terminate(); p.wait()
