#!/usr/bin/env python3
"""One speed block: a llama.cpp GGUF vs a Q-Volta .qv, 3 rounds, interleaved (each round runs every cell once), every
command under `flock /tmp/qv_bench.lock`, V100 only. Appends JSON lines to raw/speed_<LABEL>.jsonl.
usage: speed.py LABEL MODEL.gguf MODEL.qv [rounds=3] [--no-qv] [--no-llama] [--llama-mtp]
Cells: llama-bench pp512/pp2048/pp4096 + tg128 (depth 0) and tg128 at depth 512/2048/4088 (f16 KV, flash-attn on, ngl 99);
       qv_run --bench --prefill --graph --n 129 on code/prose/P512/P2048/P4088, plain and --mtp depth 2/3;
       llama-server (plain; --spec-type draft-mtp depth 2/3 with --llama-mtp) on the same token-id prompts, 128 new tokens."""
import json, os, subprocess, sys, time
LABEL, GGUF, QV = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]
ROUNDS = int(next((a for a in args if a.isdigit()), 3))
E = os.environ; LB, LOCK, DATA, OUT, QVB = E["LB"], E["GPU_LOCK"], E["DATA"], E["OUT"], E["QV_BIN"]
PROMPTS = {"code": f"{DATA}/code.i32", "prose": f"{DATA}/prose.i32", "wiki512": f"{DATA}/P512.i32", "wiki2048": f"{DATA}/P2048.i32", "wiki4088": f"{DATA}/P4088.i32"}
outf = open(f"{OUT}/speed_{LABEL}.jsonl", "a")
def emit(d):
    d["label"] = LABEL; d["t"] = time.strftime("%F %T"); outf.write(json.dumps(d) + "\n"); outf.flush()
def clocks():
    r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()
def locked(cmd, **kw):
    return subprocess.run(([] if E.get("QV_HELD") else ["flock", LOCK]) + cmd, capture_output=True, text=True, env={**E, "CUDA_VISIBLE_DEVICES": "0"}, **kw)
def llama_bench(rnd):
    for tag, a in (("pp+tg", ["-p", "512,2048,4096", "-n", "128", "-d", "0"]), ("tg_depth", ["-p", "0", "-n", "128", "-d", "512,2048,4088"])):
        r = locked([f"{LB}/llama-bench", "-m", GGUF, "-ngl", "99", "-fa", "1", "-r", "1", "-o", "json"] + a)
        try:
            rows = json.loads(r.stdout)
        except Exception:
            emit({"engine": "llama.cpp", "cell": tag, "round": rnd, "error": (r.stdout + r.stderr)[-500:]}); continue
        for x in rows:
            emit({"engine": "llama.cpp", "cell": tag, "round": rnd, "n_prompt": x["n_prompt"], "n_gen": x["n_gen"], "n_depth": x.get("n_depth", 0),
                  "tok_s": x["avg_ts"], "clocks": clocks(), "model_size": x.get("model_size")})
def qv(rnd, mtp):
    for pn, pf in PROMPTS.items():
        cmd = [QVB, "--model", QV, "--bench", "--prefill", "--graph", "--n", "129", "--max-ctx", "4608", "--prompt", pf] + (["--mtp", "--mtp-depth", str(mtp)] if mtp else [])
        r = locked(cmd)
        try:
            d = json.loads(r.stdout.strip().splitlines()[-1]); d.pop("ids", None)
        except Exception:
            emit({"engine": "q-volta", "cell": f"qv_{pn}", "mtp": mtp, "round": rnd, "error": (r.stdout + r.stderr)[-500:]}); continue
        emit({"engine": "q-volta", "cell": f"qv_{pn}", "prompt": pn, **d, "mtp_depth": mtp, "round": rnd, "clocks": clocks()})
def server(rnd, spec, nd):
    r = locked(["python3", "-I", f"{os.path.dirname(os.path.abspath(__file__))}/llama_server_bench.py", GGUF, spec, str(nd), "f16", "1", ",".join(PROMPTS.values())])
    for ln in r.stdout.splitlines():
        try: emit({"cell": "server", "round": rnd, **json.loads(ln)})
        except Exception: pass
    if not r.stdout.strip(): emit({"engine": "llama.cpp-server", "cell": "server", "spec": spec, "ndraft": nd, "round": rnd, "error": r.stderr[-500:]})
for rnd in range(1, ROUNDS + 1):
    if "--no-llama" not in args:
        llama_bench(rnd); server(rnd, "none", 0)
        if "--llama-mtp" in args:
            for nd in (2, 3): server(rnd, "mtp", nd)
    if "--no-qv" not in args:
        for m in (0, 2, 3): qv(rnd, m)
