#!/usr/bin/env python3
"""Baseline: NInfer (ninfer-all build, Apache-2.0) serving the WaveCut IQ3_S checkpoint on the V100.
Replicates the pre-existing /workspace/results/qwen27-{base,mtp3} launch args (only the binary path differs).
Same two chat prompts as benchmarks/make_prompts.py, greedy, 128 new tokens, thinking disabled; 1 warmup + 3 runs.
Writes benchmarks/baselines/ninfer-<label>.json"""
import json, os, signal, subprocess, sys, time, urllib.request
label = sys.argv[1]            # base | mtp3
BIN = "/workspace/ninfer-all/build/apps/ninfer-serve"
MODEL = "/workspace/q-volta-models/ninfer-iq3s/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
PORT = 8011
args = [BIN, MODEL, "--model-id", "qwen3.8-27b", "--device", "0", "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", "16384", "--kv-capacity", "16384", "--max-concurrency", "1", "--kv-dtype", "rk8v4",
        "--gdn-state-fp16", "--host-cache-mib", "512", "--prefill-chunk", "512"]
args += ["--spec", "none"] if label == "base" else ["--spec", "mtp", "--draft-tokens", "3"]
PROMPTS = {
 "code": "Write a Python function binary_search for a sorted list. Add a short usage example. Code only.",
 "prose": "Explain in simple words why the seasons change on Earth. Mention the axial tilt and the difference between hemispheres.",
}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
log = open(f"/workspace/q-volta/logs/ninfer-{label}.log", "w")
srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception:
            if srv.poll() is not None: sys.exit("ninfer exited, see logs/ninfer-%s.log" % label)
            time.sleep(1)
    def run(prompt):
        payload = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 128,
                   "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic(); first = None; parts = []; timings = None
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.startswith(b"data: "): continue
                raw = line[6:].strip()
                if raw == b"[DONE]": break
                d = json.loads(raw)
                timings = d.get("timings", timings)
                for c in d.get("choices", []):
                    t = c.get("delta", {}).get("content", "")
                    if t and first is None: first = time.monotonic() - t0
                    parts.append(t)
        return dict(ttft=first, elapsed=time.monotonic() - t0, timings=timings, text="".join(parts))
    rows = []
    for kind, p in PROMPTS.items():
        run(p)  # warmup
        for rep in range(3):
            r = run(p); r.update(kind=kind, rep=rep); rows.append(r)
            print(kind, rep, round(r["timings"]["predicted_per_second"], 2), "tok/s decode,",
                  round(r["timings"]["prompt_per_second"], 1), "tok/s prefill", flush=True)
    smi = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = dict(label=label, engine="ninfer-all@90268532 (sm_70 build)", model=os.path.basename(MODEL), argv=args,
               nvidia_smi_while_loaded=smi, rows=rows)
    os.makedirs("/workspace/q-volta/benchmarks/baselines", exist_ok=True)
    json.dump(out, open(f"/workspace/q-volta/benchmarks/baselines/ninfer-{label}.json", "w"), ensure_ascii=False, indent=1)
    print(smi)
finally:
    os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=60)
