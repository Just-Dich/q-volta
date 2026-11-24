#!/usr/bin/env python3
"""Real-model concurrency smoke test: 4 different chat requests (temperature 0) sent simultaneously to a qv_server with --parallel >= 4 must give
exactly the same text/tokens as the same requests sent one after the other, and all must finish with finish_reason stop|length.
  python3 tests/test_server_concurrent_real.py [--port 8100] [--max-tokens 96]
(start the server first, e.g. qv_server --model inkling2-mini-q3a5.qv --backend cuda --kv int4 --max-ctx 32768 --parallel 4)"""
import argparse, http.client, json, sys, threading, time

Q = ["Write a haiku about the sea.", "What is 17 * 23? Answer briefly.", "Name three prime numbers greater than 50.", "Explain in two sentences what a hash table is."]

def chat(port, q, n):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0, "stream": True, "cache_prompt": False}),
              {"Content-Type": "application/json"})
    r = c.getresponse(); content = reasoning = ""; fin = None; t0 = time.time(); first = None; chunks = 0
    buf = b""
    while True:
        d = r.read1(65536)
        if not d: break
        buf += d
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            if not ev.startswith(b"data: ") or b"[DONE]" in ev: continue
            j = json.loads(ev[6:])
            for ch in j.get("choices", []):
                dl = ch.get("delta", {})
                if dl.get("content"): content += dl["content"]; first = first or time.time()
                if dl.get("reasoning_content"): reasoning += dl["reasoning_content"]; first = first or time.time()
                if ch.get("finish_reason"): fin = ch["finish_reason"]
                chunks += 1
    return dict(content=content, reasoning=reasoning, finish=fin, ttft=(first or time.time()) - t0, chunks=chunks)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8100); ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()
    seq = [chat(a.port, q, a.max_tokens) for q in Q]
    res = [None] * len(Q)
    def w(i): res[i] = chat(a.port, Q[i], a.max_tokens)
    ths = [threading.Thread(target=w, args=(i,)) for i in range(len(Q))]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    ok = True
    for i, q in enumerate(Q):
        same = seq[i]["content"] == res[i]["content"] and seq[i]["reasoning"] == res[i]["reasoning"]
        fin_ok = res[i]["finish"] in ("stop", "length")
        ok &= same and fin_ok and (res[i]["content"] or res[i]["reasoning"]) != ""
        print(f"[{i}] identical_to_sequential={same} finish={res[i]['finish']} ttft={res[i]['ttft']:.2f}s  {(res[i]['content'] or res[i]['reasoning'])[:90]!r}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
