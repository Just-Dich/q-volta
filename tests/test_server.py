#!/usr/bin/env python3
"""End-to-end test of qv_server on the CPU backend (stdlib only; uses the `openai` package too when installed).
usage: python3 -I tests/test_server.py --server BUILD/qv_server [--model PREFIX] [--threads 32]
Starts the server on free ports (GPU hidden via CUDA_VISIBLE_DEVICES=), uses tiny max_tokens (CPU decode ~3-4 tok/s)."""
import argparse, http.client, json, os, socket, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", default="/workspace/q-volta-models/qv/qwen38-27b-e0")
ap.add_argument("--threads", default="32")
ap.add_argument("--max-ctx", default="512")
args = ap.parse_args()
QV_TOK = os.path.join(os.path.dirname(os.path.abspath(args.server)), "qv_tok")
TOKJSON = "/workspace/q-volta-models/Qwen3.8-27B/tokenizer.json"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (("  -- " + str(detail)) if (detail and not cond) else ""), flush=True)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def start(extra, logname):
    port = free_port()
    log = open(logname, "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.Popen([args.server, "--model", args.model, "--backend", "cpu", "--port", str(port), "--max-ctx", args.max_ctx,
                          "--threads", args.threads] + extra, stderr=log, stdout=log, env=env)
    t0 = time.time()
    while time.time() - t0 < 180:
        if p.poll() is not None: raise SystemExit("server died, see " + logname)
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2); c.request("GET", "/health")
            if c.getresponse().status == 200: return p, port, time.time() - t0
        except OSError: time.sleep(0.5)
    raise SystemExit("server did not come up")


def req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    h = dict(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, (str, bytes)) else json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    b = r.read()
    ok_json = b and r.getheader("content-type", "").startswith("application/json")
    return r.status, dict((k.lower(), v) for k, v in r.getheaders()), (json.loads(b) if ok_json else b)


def sse(port, path, body):
    """returns (status, events, t_first_content, t_total, headers); events are parsed JSON objects or '[DONE]'"""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    t0 = time.time()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    hdrs = dict((k.lower(), v) for k, v in r.getheaders())
    ev, tfirst, buf = [], None, b""
    while True:
        ch = r.read1(65536)
        if not ch: break
        buf += ch
        while b"\n\n" in buf:
            blk, buf = buf.split(b"\n\n", 1)
            assert blk.startswith(b"data: "), blk
            payload = blk[6:].decode()
            if payload == "[DONE]":
                ev.append("[DONE]"); continue
            o = json.loads(payload)
            if tfirst is None and o.get("choices") and (o["choices"][0].get("delta", {}).get("content") or o["choices"][0].get("text")):
                tfirst = time.time() - t0
            ev.append(o)
    return r.status, ev, tfirst, time.time() - t0, hdrs


def tok_count(text_prompt=None, chat_body=None):
    if chat_body is not None:
        out = subprocess.run([QV_TOK, TOKJSON, "chat"], input=json.dumps(chat_body) + "\n", capture_output=True, text=True, check=True).stdout
        text_prompt = json.loads(out)["text"]
    out = subprocess.run([QV_TOK, TOKJSON, "encode"], input=json.dumps(text_prompt) + "\n", capture_output=True, text=True, check=True).stdout
    return len(json.loads(out))


NOTHINK = {"enable_thinking": False}
proc, port, load_s = start([], tempfile.gettempdir() + "/qv_server_test.log")
print(f"server up on :{port} after {load_s:.1f}s; load average {os.getloadavg()}")
proc2 = None
try:
    st, _, b = req(port, "GET", "/health"); check("health 200 {status:ok}", st == 200 and b == {"status": "ok"}, b)
    st, h, b = req(port, "GET", "/v1/models"); check("models list", st == 200 and b["object"] == "list" and b["data"][0]["id"] == "qwen3.8-27b" and b["data"][0]["object"] == "model", b)
    st, h, b = req(port, "OPTIONS", "/v1/chat/completions"); check("CORS preflight 204 + headers", st == 204 and h.get("access-control-allow-origin") == "*" and "authorization" in h.get("access-control-allow-headers", ""), (st, h))
    st, h, b = req(port, "GET", "/nope"); check("unknown route 404 openai error", st == 404 and "error" in b and set(b["error"]) >= {"message", "type", "param", "code"}, b)

    # completions, non-stream
    st, h, b = req(port, "POST", "/v1/completions", {"model": "whatever", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    c_text = b["choices"][0]["text"]
    check("completions non-stream schema", st == 200 and b["object"] == "text_completion" and b["id"].startswith("cmpl-") and isinstance(b["created"], int)
          and b["choices"][0]["finish_reason"] in ("stop", "length") and set(b["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens"}, b)
    check("completions max_tokens=8 -> length, 8 tokens", b["choices"][0]["finish_reason"] == "length" and b["usage"]["completion_tokens"] == 8, b["usage"])
    check("completions prompt_tokens == tokenizer count", b["usage"]["prompt_tokens"] == tok_count("The capital of France is"), b["usage"])
    check("usage total = prompt + completion", b["usage"]["total_tokens"] == b["usage"]["prompt_tokens"] + b["usage"]["completion_tokens"])
    check("greedy output mentions Paris", "Paris" in c_text, c_text)
    st, _, b2 = req(port, "POST", "/v1/completions", {"prompt": ["The capital of France is", "ignored"], "max_tokens": 4, "temperature": 0})
    check("completions prompt list-of-strings uses first", st == 200 and c_text.startswith(b2["choices"][0]["text"]) and b2["usage"]["completion_tokens"] == 4, b2)
    ids = json.loads(subprocess.run([QV_TOK, TOKJSON, "encode"], input=json.dumps("The capital of France is") + "\n", capture_output=True, text=True).stdout)
    st, _, b3 = req(port, "POST", "/v1/completions", {"prompt": ids, "max_tokens": 4, "temperature": 0})
    check("completions prompt as token ids", st == 200 and b3["usage"]["prompt_tokens"] == len(ids) and b3["choices"][0]["text"] == b2["choices"][0]["text"], b3)

    # chat non-stream
    chat1 = {"messages": [{"role": "user", "content": "Reply with exactly one word: yes"}], "max_tokens": 16, "temperature": 0, "chat_template_kwargs": NOTHINK}
    st, h1, b = req(port, "POST", "/v1/chat/completions", chat1)
    m = b["choices"][0]["message"]
    check("chat non-stream schema", st == 200 and b["object"] == "chat.completion" and b["id"].startswith("chatcmpl-") and m["role"] == "assistant" and isinstance(m["content"], str)
          and b["choices"][0]["finish_reason"] in ("stop", "length"), b)
    check("chat prompt_tokens == tokenizer(chat template) count", b["usage"]["prompt_tokens"] == tok_count(chat_body={"messages": chat1["messages"], "enable_thinking": False}), b["usage"])
    check("first chat request has 0 cached tokens", h1.get("x-qv-prompt-cached-tokens") == "0", h1)
    a1 = m["content"]; p1 = b["usage"]["prompt_tokens"]
    print("   chat answer:", repr(a1), "finish:", b["choices"][0]["finish_reason"])

    # prefix reuse: continuation of the same conversation
    chat2 = {"messages": chat1["messages"] + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Now say no"}], "max_tokens": 4, "temperature": 0, "chat_template_kwargs": NOTHINK}
    st, h2, b = req(port, "POST", "/v1/chat/completions", chat2)
    cached = int(h2.get("x-qv-prompt-cached-tokens", "-1"))
    check("continuation reuses prefix (X-QV-Prompt-Cached-Tokens >= first prompt length)", st == 200 and cached >= p1, (cached, p1))
    check("usage.prompt_tokens_details.cached_tokens matches header", b["usage"]["prompt_tokens_details"]["cached_tokens"] == cached)
    st, h3, b = req(port, "POST", "/v1/chat/completions", chat1)
    check("unrelated/shorter prompt resets (cached=0)", h3.get("x-qv-prompt-cached-tokens") == "0", h3)

    # chat stream
    st, ev, tfirst, ttot, sh = sse(port, "/v1/chat/completions", dict(chat1, stream=True, max_tokens=8, stream_options={"include_usage": True}))
    chunks = [e for e in ev if e != "[DONE]"]
    d0 = chunks[0]["choices"][0]["delta"]
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in chunks if e["choices"] and e["choices"][0].get("delta"))
    fin = [e for e in chunks if e["choices"] and e["choices"][0].get("finish_reason")]
    usage_chunks = [e for e in chunks if e.get("usage")]
    check("stream: status 200 SSE", st == 200 and sh.get("content-type", "").startswith("text/event-stream"), sh)
    check("stream: first delta is role=assistant", d0.get("role") == "assistant" and chunks[0]["object"] == "chat.completion.chunk", chunks[0])
    check("stream: content deltas present", len(content) > 0 and sum(1 for e in chunks if e["choices"] and e["choices"][0]["delta"].get("content")) >= 2, content)
    check("stream: one finish chunk (length) with empty delta", len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "length" and fin[0]["choices"][0]["delta"] == {}, fin)
    check("stream: usage chunk (include_usage) has empty choices", len(usage_chunks) == 1 and usage_chunks[0]["choices"] == [] and usage_chunks[0]["usage"]["completion_tokens"] == 8, usage_chunks)
    check("stream: terminated by [DONE]", ev[-1] == "[DONE]")
    st, _, nb = req(port, "POST", "/v1/chat/completions", dict(chat1, max_tokens=8))
    check("stream content == non-stream content (greedy)", nb["choices"][0]["message"]["content"] == content, (content, nb["choices"][0]["message"]["content"]))
    print(f"METRIC chat stream: time to first content chunk {tfirst:.2f}s (prompt {nb['usage']['prompt_tokens']} tok), 8 tokens in {ttot:.2f}s -> {7 / (ttot - tfirst):.2f} tok/s; loadavg {os.getloadavg()[0]:.1f}")

    # completion stream
    st, ev, tf, tt, sh = sse(port, "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 5, "temperature": 0, "stream": True})
    ch = [e for e in ev if e != "[DONE]"]
    txt = "".join(e["choices"][0]["text"] for e in ch)
    check("completions stream: text deltas, finish_reason, [DONE]", c_text.startswith(txt) and ch[-1]["choices"][0]["finish_reason"] == "length" and ev[-1] == "[DONE]", ev[-3:])

    # stop sequence
    k = max(1, len(c_text) // 2)
    stop = c_text[k:k + 2] if c_text[k:k + 2].strip() else c_text[k:k + 3]
    st, _, b = req(port, "POST", "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, "stop": [stop]})
    exp = c_text[:c_text.find(stop)]
    check("stop sequence truncates text and finish_reason=stop", st == 200 and b["choices"][0]["text"] == exp and b["choices"][0]["finish_reason"] == "stop", (stop, exp, b["choices"][0]))
    st, ev, *_ = sse(port, "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, "stop": stop, "stream": True})
    sx = "".join(e["choices"][0]["text"] for e in ev if e != "[DONE]")
    check("stop sequence in stream mode (stop given as string)", sx == exp, (sx, exp))

    # seed determinism
    sb = {"prompt": "Once upon a time", "max_tokens": 6, "temperature": 1.0, "top_p": 0.95, "seed": 1234}
    _, _, s1 = req(port, "POST", "/v1/completions", sb); _, _, s2 = req(port, "POST", "/v1/completions", sb)
    check("same seed -> identical text at temperature 1.0", s1["choices"][0]["text"] == s2["choices"][0]["text"], (s1["choices"][0]["text"], s2["choices"][0]["text"]))
    _, _, s3 = req(port, "POST", "/v1/completions", dict(sb, seed=99, max_tokens=10, temperature=2.0, top_p=1.0, top_k=0))
    print("   seed 1234:", repr(s1["choices"][0]["text"]), " seed 99 (T=2):", repr(s3["choices"][0]["text"]))
    _, _, lp = req(port, "POST", "/v1/chat/completions", dict(chat1, max_tokens=2, logprobs=True, top_logprobs=2))
    lc = lp["choices"][0]["logprobs"]["content"]
    check("chat logprobs (sampled token + top-2 alternatives)", len(lc) == 2 and lc[0]["logprob"] <= 0 and len(lc[0]["top_logprobs"]) == 2 and lc[0]["top_logprobs"][0]["token"] == lc[0]["token"], lc)
    _, _, pp = req(port, "POST", "/v1/completions", {"prompt": "aaa aaa aaa aaa", "max_tokens": 3, "temperature": 0, "frequency_penalty": 1.0, "presence_penalty": 0.5})
    check("penalties accepted (host-logits path)", "choices" in pp, pp)

    # errors
    st, _, b = req(port, "POST", "/v1/chat/completions", "{not json"); check("bad JSON -> 400 invalid_json", st == 400 and b["error"]["code"] == "invalid_json" and b["error"]["type"] == "invalid_request_error", b)
    st, _, b = req(port, "POST", "/v1/chat/completions", dict(chat1, n=2)); check("n=2 -> 400 (param n)", st == 400 and b["error"]["param"] == "n", b)
    st, _, b = req(port, "POST", "/v1/chat/completions", {"max_tokens": 4}); check("missing messages -> 400", st == 400 and b["error"]["param"] == "messages", b)
    st, _, b = req(port, "POST", "/v1/chat/completions", dict(chat1, temperature=5)); check("temperature out of range -> 400", st == 400 and b["error"]["param"] == "temperature", b)
    st, _, b = req(port, "POST", "/v1/chat/completions", dict(chat1, tools=[{"type": "function", "function": {"name": "f"}}])); check("tools -> 400 unsupported", st == 400 and b["error"]["param"] == "tools", b)
    st, _, b = req(port, "POST", "/v1/completions", {"prompt": [100] * 600, "max_tokens": 2}); check("prompt longer than max_ctx -> 413", st == 413 and b["error"]["code"] == "context_length_exceeded", (st, b))
    st, _, b = req(port, "POST", "/v1/completions", {"prompt": [100] * (int(args.max_ctx) - 3), "max_tokens": 50, "temperature": 0})
    check("max_tokens clamped to remaining context", st == 200 and b["usage"]["completion_tokens"] <= 3 and b["choices"][0]["finish_reason"] == "length", b)
    st, _, b = req(port, "GET", "/v1/chat/completions"); check("wrong method -> 405", st == 405, st)

    # client disconnect mid-stream
    s = socket.create_connection(("127.0.0.1", port), timeout=60)
    body = json.dumps({"messages": [{"role": "user", "content": "Count from 1 to 100."}], "max_tokens": 100, "temperature": 0, "stream": True, "chat_template_kwargs": NOTHINK})
    s.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)).encode())
    got = b""
    while b"content" not in got.split(b"\r\n\r\n", 1)[-1][20:] and len(got) < 100000: got += s.recv(4096)
    s.close()
    t0 = time.time()
    st, _, b = req(port, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 2, "temperature": 0})
    waited = time.time() - t0
    check("disconnect aborts generation: next request served promptly and engine consistent", st == 200 and waited < 15 and len(b["choices"][0]["text"]) > 0, waited)
    time.sleep(0.2)
    logtxt = open(tempfile.gettempdir() + "/qv_server_test.log").read()
    check("server log records the client abort", "client aborted" in logtxt, logtxt[-600:])
    st, _, b = req(port, "POST", "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    check("after abort, greedy output identical to earlier result (state consistent)", b["choices"][0]["text"] == c_text, (b["choices"][0]["text"], c_text))

    # api key
    proc2, port2, _ = start(["--api-key", "sekret", "--threads", "4"], tempfile.gettempdir() + "/qv_server_test2.log")
    st, _, b = req(port2, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 1}); check("no key -> 401", st == 401 and b["error"]["code"] == "invalid_api_key", b)
    st, _, b = req(port2, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 1}, {"Authorization": "Bearer wrong"}); check("wrong key -> 401", st == 401, b)
    st, _, b = req(port2, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 1, "temperature": 0}, {"Authorization": "Bearer sekret"}); check("right key -> 200", st == 200, b)
    st, _, b = req(port2, "GET", "/health"); check("/health open without key", st == 200)

    # openai client (optional)
    try:
        import openai
        cl = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x")
        r = cl.chat.completions.create(model="qwen3.8-27b", messages=chat1["messages"], max_tokens=4, temperature=0, extra_body={"chat_template_kwargs": NOTHINK})
        check("openai client chat", r.choices[0].message.content is not None)
        n = 0
        for chunk in cl.chat.completions.create(model="x", messages=chat1["messages"], max_tokens=4, stream=True, temperature=0, extra_body={"chat_template_kwargs": NOTHINK}): n += 1
        check("openai client stream", n >= 3)
    except ImportError:
        print("SKIP openai python package not installed")
finally:
    for p in (proc, proc2):
        if p and p.poll() is None: p.terminate()
    for p in (proc, proc2):
        if p:
            try: p.wait(15)
            except Exception: p.kill()
bad = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (": FAILED " + ", ".join(bad) if bad else ""))
sys.exit(1 if bad else 0)
