#!/usr/bin/env python3
"""Concurrent-request benchmark for qv_server (streaming /completion, greedy, ignore_eos).
  bench_concurrent.py --port 8100 --conc 1,2,4 --prompt-tokens 100 --n 128 --runs 3 [--long 30000 --short 100]
Per run: C simultaneous requests with DIFFERENT prompts (cache off). Reports aggregate tok/s (tokens / wall), per-request tok/s
(decoded tokens / (last - first token time)), TTFT. --long L: one client sends an L-token prompt (n_predict 1) while a short request
arrives 1 s later; reports the short request's TTFT."""
import argparse, http.client, json, random, statistics, threading, time

WORDS = "the quick brown fox jumps over lazy dog while seven small birds sing under bright green trees near an old stone bridge".split()

def make_prompt(ntok, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(max(1, ntok // 3)))  # ~3 tokens per item (measured)

def stream(port, prompt, n, out, key, ignore_eos=True):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "stream": True, "cache_prompt": False, "ignore_eos": ignore_eos})
    t0 = time.time()
    c.request("POST", "/completion", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    first = last = None; nt = 0; info = {}
    buf = b""
    while True:
        d = r.read1(65536)
        if not d: break
        buf += d
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            if not ev.startswith(b"data: "): continue
            j = json.loads(ev[6:])
            if j.get("content") or j.get("tokens"):
                now = time.time()
                if first is None: first = now
                last = now
            if j.get("stop"): info = j
    out[key] = dict(ttft=(first or time.time()) - t0, t_first=first, t_last=last, n=info.get("tokens_predicted", 0), n_prompt=info.get("tokens_evaluated", 0), t0=t0, t_end=time.time())

def run(port, conc, ptok, n, seed0):
    res = {}
    ths = [threading.Thread(target=stream, args=(port, make_prompt(ptok, seed0 + i), n, res, i)) for i in range(conc)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    toks = sum(r["n"] for r in res.values())
    per = [(r["n"] - 1) / (r["t_last"] - r["t_first"]) for r in res.values() if r["t_last"] and r["t_last"] > r["t_first"]]
    # aggregate decode rate: tokens decoded while all clients were decoding (overlap window), plus end-to-end
    return dict(conc=conc, wall=wall, agg_e2e=toks / wall, per_req=statistics.mean(per), ttft=statistics.mean(r["ttft"] for r in res.values()), n_prompt=res[0]["n_prompt"],
                agg_decode=sum(per))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100); ap.add_argument("--conc", default="1,2,4"); ap.add_argument("--prompt-tokens", type=int, default=100)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--runs", type=int, default=3); ap.add_argument("--long", type=int, default=0); ap.add_argument("--short", type=int, default=100)
    a = ap.parse_args()
    if a.long:
        for rep in range(a.runs):
            res = {}
            tl = threading.Thread(target=stream, args=(a.port, make_prompt(a.long, 7 + rep), 1, res, "long"))
            t0 = time.time(); tl.start(); time.sleep(1.0)
            ts = threading.Thread(target=stream, args=(a.port, make_prompt(a.short, 99 + rep), 16, res, "short")); ts.start(); ts.join(); tl.join()
            print(json.dumps(dict(run=rep, long_prompt_tokens=res["long"]["n_prompt"], long_total_s=round(res["long"]["t_end"] - res["long"]["t0"], 2), short_ttft_s=round(res["short"]["ttft"], 3),
                                  short_total_s=round(res["short"]["t_end"] - res["short"]["t0"], 2))), flush=True)
    else:
        warm = {}; stream(a.port, make_prompt(a.prompt_tokens, 1), 8, warm, 0)
        for c in [int(x) for x in a.conc.split(",")]:
            for rep in range(a.runs):
                r = run(a.port, c, a.prompt_tokens, a.n, 1000 * c + 10 * rep)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
