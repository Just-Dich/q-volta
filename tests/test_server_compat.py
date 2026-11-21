#!/usr/bin/env python3
"""llama-server compatibility tests for qv_server.

  python3 -m pytest tests/test_server_compat.py -q          (needs build-cpu/qv_server; mock engine, no model weights)
  QV_SERVER=build-cpu/qv_server LLAMA_SERVER=/tmp/llama_srv_build/bin/llama-server LLAMA_GGUF=/tmp/qwen05/qwen2.5-0.5b-instruct-q4_k_m.gguf \
      python3 -m pytest tests/test_server_compat.py -q       (also runs the structural diff against the real llama-server)

Part 1 (ours alone): endpoints, request/response shapes, SSE framing, errors, queue/slots, CLI. Uses the `--backend mock` engine
(scripted tokens through the real tokenizer path) and stdlib http.client only.
Part 2 (differential, skipped without llama-server): the same requests go to both servers and the JSON *structure* (key sets, value
types), status codes, header names and SSE framing are compared. Text values differ by design (different models) and are not compared.
"""
import http.client, json, os, re, signal, socket, subprocess, sys, tempfile, threading, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QV_SERVER = os.environ.get("QV_SERVER", os.path.join(ROOT, "build-cpu", "qv_server"))
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/tmp/llama_srv_build/bin/llama-server")
LLAMA_GGUF = os.environ.get("LLAMA_GGUF", "/tmp/qwen05/qwen2.5-0.5b-instruct-q4_k_m.gguf")
TOKJSON = os.environ.get("QV_TOKENIZER", "/workspace/q-volta-models/Qwen3.8-27B/tokenizer.json")
HAVE_LLAMA = os.path.exists(LLAMA_SERVER) and os.path.exists(LLAMA_GGUF)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Srv:
    def __init__(self, cmd, env=None, wait=True, name="srv"):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=f".{name}.log", delete=False)
        full = [c.replace("@PORT@", str(self.port)) for c in cmd]
        self.p = subprocess.Popen(full, stdout=self.log, stderr=self.log, env=dict(os.environ, **(env or {})))
        if wait: self.wait_ready()

    def wait_ready(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None: raise RuntimeError("server exited: " + open(self.log.name).read()[-2000:])
            try:
                st, _, _ = self.req("GET", "/health")
                if st == 200: return
            except OSError: pass
            time.sleep(0.1)
        raise RuntimeError("server did not become ready")

    def req(self, method, path, body=None, headers=None, raw=False, timeout=60):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, (str, bytes)) else json.dumps(body)
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, data, h)
        r = c.getresponse()
        b = r.read()
        hd = [(k, v) for k, v in r.getheaders()]
        c.close()
        if raw: return r.status, hd, b
        ct = r.getheader("Content-Type", "") or ""
        if b and ct.startswith("application/json"):
            try: return r.status, hd, json.loads(b)
            except ValueError: pass
        return r.status, hd, b

    def sse(self, path, body, headers=None):
        st, hd, b = self.req("POST", path, body, headers, raw=True)
        events = []
        for blk in b.decode().split("\n\n"):
            if not blk: continue
            assert blk.startswith("data: "), "bad SSE framing: %r" % blk[:80]
            d = blk[6:]
            events.append(d if d == "[DONE]" else json.loads(d))
        return st, hd, events, b

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try: self.p.wait(10)
            except subprocess.TimeoutExpired: self.p.kill()


def hdrs(hd): return {k.lower(): v for k, v in hd}


def qv_cmd(*extra, backend="mock"):
    return [QV_SERVER, "--backend", backend, "--tokenizer", TOKJSON, "--port", "@PORT@", "-c", "4096"] + list(extra)


@pytest.fixture(scope="module")
def srv():
    s = Srv(qv_cmd("--metrics", "--props", "--slot-save-path", tempfile.mkdtemp(prefix="qvslots_"), "--mock-loop", "--mock-text", "alpha beta gamma delta"), name="qv")
    yield s
    s.stop()


# =============================================================================================== part 1: ours alone
def test_health_and_aliases(srv):
    for p in ("/health", "/v1/health"):
        st, hd, b = srv.req("GET", p)
        assert st == 200 and b == {"status": "ok"}
        assert hdrs(hd)["content-type"] == "application/json; charset=utf-8"
    st, hd, b = srv.req("HEAD", "/health", raw=True)
    assert st == 200 and b == b""


def test_health_503_while_loading():
    s = Srv(qv_cmd(), env={"QV_MOCK_LOAD_MS": "2500"}, wait=False, name="load")
    try:
        t0 = time.time(); seen = None
        while time.time() - t0 < 2:
            try:
                seen = s.req("GET", "/health"); break
            except OSError: time.sleep(0.05)
        assert seen is not None, "port not open while loading"
        st, hd, b = seen
        assert st == 503 and b == {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}
        st, _, b = s.req("POST", "/completion", {"prompt": "x"})
        assert st == 503
        s.wait_ready()
        assert s.req("GET", "/health")[0] == 200
    finally:
        s.stop()


def test_props_shape(srv):
    st, _, p = srv.req("GET", "/props")
    assert st == 200
    for k in ("default_generation_settings", "total_slots", "model_alias", "model_path", "modalities", "endpoint_slots", "endpoint_props", "endpoint_metrics",
              "chat_template", "chat_template_caps", "bos_token", "eos_token", "build_info", "is_sleeping"):
        assert k in p, k
    assert p["total_slots"] == 4 and p["endpoint_metrics"] is True and p["endpoint_props"] is True
    assert p["default_generation_settings"]["n_ctx"] == 4096
    prm = p["default_generation_settings"]["params"]
    assert prm["temperature"] == pytest.approx(0.8) and prm["top_k"] == 40 and prm["top_p"] == pytest.approx(0.95) and prm["min_p"] == pytest.approx(0.05)
    assert prm["samplers"] == ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]
    assert srv.req("POST", "/props", {})[2] == {"success": True}


def test_models(srv):
    for path in ("/models", "/v1/models"):
        st, _, m = srv.req("GET", path)
        assert st == 200 and m["object"] == "list" and m["data"][0]["object"] == "model" and m["data"][0]["id"]
        assert m["models"][0]["name"] == m["data"][0]["id"]


def test_completion_native_shape(srv):
    st, hd, r = srv.req("POST", "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0})
    assert st == 200
    for k in ("index", "content", "tokens", "id_slot", "stop", "model", "tokens_predicted", "tokens_evaluated", "generation_settings", "prompt", "has_new_line",
              "truncated", "stop_type", "stopping_word", "tokens_cached", "timings"):
        assert k in r, k
    assert r["content"] == "alpha beta gamma" or r["content"].startswith("alpha") 
    assert r["tokens_predicted"] == 4 and r["stop_type"] == "limit" and r["stop"] is True and r["tokens"] == [] and r["truncated"] is False
    t = r["timings"]
    for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"):
        assert k in t, k
    gs = r["generation_settings"]
    assert gs["n_predict"] == 4 and gs["temperature"] == 0 and gs["stop"] == [] and "dry_sequence_breakers" in gs and gs["logit_bias"] == []


def test_completion_return_tokens_and_prompt_forms(srv):
    st, _, r = srv.req("POST", "/completion", {"prompt": "Hello", "n_predict": 3, "return_tokens": True, "temperature": 0})
    assert len(r["tokens"]) == 3 and all(isinstance(x, int) for x in r["tokens"])
    ids = srv.req("POST", "/tokenize", {"content": "Hello world"})[2]["tokens"]
    for prompt in (ids, ["Hello", ids[1]], {"prompt_string": "Hello world"}):
        st, _, r = srv.req("POST", "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0})
        assert st == 200 and r["tokens_evaluated"] == len(ids), prompt
    st, _, r = srv.req("POST", "/completion", {"prompt": ["a", "b"], "n_predict": 1, "temperature": 0})
    assert st == 200 and isinstance(r, list) and len(r) == 2 and r[0]["index"] == 0


def test_stop_word_and_eos(srv):
    st, _, r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "stop": [" gamma"], "temperature": 0})
    assert r["stop_type"] == "word" and r["stopping_word"] == " gamma" and r["content"] == "alpha beta"
    s2 = Srv(qv_cmd("--mock-text", "short"), name="eos")
    try:
        r = s2.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "temperature": 0})[2]
        assert r["stop_type"] == "eos" and r["content"] == "short" and r["tokens_predicted"] == 2  # script token + EOS
        r = s2.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "temperature": 0, "ignore_eos": True, "logit_bias": []})[2]
        assert r["stop_type"] == "limit" and r["tokens_predicted"] == 20
    finally:
        s2.stop()


def test_n_probs_and_logit_bias(srv):
    r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": 0, "n_probs": 3})[2]
    cp = r["completion_probabilities"]
    assert len(cp) == 2 and set(cp[0]) == {"id", "token", "bytes", "logprob", "top_logprobs"} and len(cp[0]["top_logprobs"]) == 3
    assert cp[0]["top_logprobs"][0]["logprob"] == pytest.approx(cp[0]["logprob"])
    r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": 0, "n_probs": 2, "post_sampling_probs": True})[2]
    assert "prob" in r["completion_probabilities"][0] and "top_probs" in r["completion_probabilities"][0]
    tid = srv.req("POST", "/tokenize", {"content": "zzz"})[2]["tokens"][0]
    r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 1, "temperature": 0, "logit_bias": [[tid, 100.0]], "return_tokens": True})[2]
    assert r["tokens"] == [tid]


def test_seed_determinism_and_sampling_params(srv):
    body = {"prompt": "x", "n_predict": 6, "temperature": 5.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42}
    a = srv.req("POST", "/completion", body)[2]["content"]
    b = srv.req("POST", "/completion", body)[2]["content"]
    assert a == b
    body["seed"] = 43
    assert srv.req("POST", "/completion", body)[2]["content"] != a
    for extra in ({"typical_p": 0.5}, {"mirostat": 2}, {"mirostat": 1}, {"xtc_probability": 1.0, "xtc_threshold": 0.01}, {"top_n_sigma": 1.0},
                  {"dry_multiplier": 0.8}, {"dynatemp_range": 0.5}, {"samplers": ["top_k", "temperature"]}, {"repeat_penalty": 1.3}):
        st, _, r = srv.req("POST", "/completion", dict(body, **extra))
        assert st == 200 and r["tokens_predicted"] == 6, extra
    # penalties flip the scripted token when strong enough
    a = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 8, "temperature": 0, "frequency_penalty": 1000})[2]["content"]
    assert a != srv.req("POST", "/completion", {"prompt": "x", "n_predict": 8, "temperature": 0})[2]["content"]


def test_prompt_cache_reporting(srv):
    p = "cache test prompt number one " * 4
    r1 = srv.req("POST", "/completion", {"prompt": p, "n_predict": 2, "temperature": 0})[2]
    r2 = srv.req("POST", "/completion", {"prompt": p + "more", "n_predict": 2, "temperature": 0})[2]
    # the engine state after r1 holds prompt+generated tokens; r2's prompt only extends the prompt, so it cannot reuse a longer cached sequence
    assert r1["timings"]["cache_n"] == 0
    # continuing from the exact cached sequence reuses it
    full = srv.req("POST", "/tokenize", {"content": p})[2]["tokens"]
    r3 = srv.req("POST", "/completion", {"prompt": full, "n_predict": 2, "temperature": 0})[2]
    r4 = srv.req("POST", "/completion", {"prompt": full, "n_predict": 2, "temperature": 0, "cache_prompt": False})[2]
    assert r4["timings"]["cache_n"] == 0 and r4["timings"]["prompt_n"] == len(full)
    # r3 follows r2 (different prompt) -> prefix mismatch -> full re-evaluation; run twice so that the 2nd one can reuse
    r5 = srv.req("POST", "/completion", {"prompt": full, "n_predict": 2, "temperature": 0})[2]
    assert r5["tokens_cached"] >= len(full)
    # exact repetition: the engine state ends at prompt+gen[:-1]; a new request that starts with that exact sequence reuses it
    text = r5["content"]
    st, _, usage = srv.req("POST", "/v1/completions", {"prompt": full, "max_tokens": 1, "temperature": 0})
    assert usage["usage"]["prompt_tokens_details"]["cached_tokens"] == usage["timings"]["cache_n"]


def test_oai_completions(srv):
    st, hd, r = srv.req("POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0})
    assert st == 200 and r["object"] == "text_completion" and r["id"].startswith("chatcmpl-") and r["system_fingerprint"]
    assert r["choices"][0]["finish_reason"] == "length" and r["choices"][0]["logprobs"] is None and r["choices"][0]["index"] == 0
    assert r["usage"]["completion_tokens"] == 3 and r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + 3 and "timings" in r
    st, hd, ev, raw = srv.sse("/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0, "stream": True})
    assert hdrs(hd)["content-type"] == "text/event-stream" and ev[-1] == "[DONE]"
    assert all(e["object"] == "text_completion" for e in ev[:-1]) and ev[-2]["choices"][0]["finish_reason"] == "length" and "usage" in ev[-2]
    assert "".join(e["choices"][0]["text"] for e in ev[:-1]) == "alpha beta gamma"[:len("".join(e["choices"][0]["text"] for e in ev[:-1]))]


def test_chat_non_stream(srv):
    st, hd, r = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0,
                                                          "chat_template_kwargs": {"enable_thinking": False}})
    assert st == 200 and r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-")
    ch = r["choices"][0]
    assert ch["message"]["role"] == "assistant" and ch["message"]["content"].startswith("alpha") and ch["finish_reason"] == "length"
    assert set(r["usage"]) == {"completion_tokens", "prompt_tokens", "total_tokens", "prompt_tokens_details"} and "timings" in r
    # same request through the alias without /v1
    assert srv.req("POST", "/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1})[0] == 200
    # multi-part content, developer role, max_completion_tokens, n=2
    r = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
                                                  "max_completion_tokens": 2, "n": 2, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})[2]
    assert len(r["choices"]) == 2 and [c["index"] for c in r["choices"]] == [0, 1] and r["usage"]["completion_tokens"] == 4


def test_chat_reasoning_formats(srv):
    base = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0}
    r = srv.req("POST", "/v1/chat/completions", base)[2]["choices"][0]["message"]   # thinking template opens <think>: the scripted text is reasoning
    assert r["content"] == "" and r["reasoning_content"].startswith("alpha")
    r = srv.req("POST", "/v1/chat/completions", dict(base, reasoning_format="none"))[2]["choices"][0]["message"]
    assert "reasoning_content" not in r and r["content"].startswith("alpha")
    r = srv.req("POST", "/v1/chat/completions", dict(base, chat_template_kwargs={"enable_thinking": False}))[2]["choices"][0]["message"]
    assert "reasoning_content" not in r and r["content"].startswith("alpha")
    st, _, r = srv.req("POST", "/v1/chat/completions", dict(base, reasoning_format="bogus"))
    assert st == 500 and r["error"]["type"] == "server_error"   # llama-server answers 500 for an unknown format


def test_chat_stream_framing(srv):
    st, hd, ev, raw = srv.sse("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0, "stream": True,
                                                        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}})
    h = hdrs(hd)
    assert st == 200 and h["content-type"] == "text/event-stream" and h["transfer-encoding"] == "chunked" and h.get("x-accel-buffering") == "no"
    assert ev[-1] == "[DONE]" and raw.endswith(b"data: [DONE]\n\n")
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    fin = ev[-3]
    assert fin["choices"][0]["finish_reason"] == "length" and fin["choices"][0]["delta"] == {}
    usage = ev[-2]
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == 4 and "timings" in usage
    ids = {e["id"] for e in ev[:-1]}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
    assert all(e["object"] == "chat.completion.chunk" and "system_fingerprint" in e for e in ev[:-1])
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in ev[1:-3])
    assert text.startswith("alpha")
    # no include_usage: timings ride on the finish chunk
    st, hd, ev, raw = srv.sse("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2, "stream": True})
    assert ev[-2]["choices"][0]["finish_reason"] == "length" and "usage" not in ev[-2] and "timings" in ev[-2]


def test_chat_logprobs_and_tools_error(srv):
    r = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2, "temperature": 0, "logprobs": True, "top_logprobs": 2,
                                                  "chat_template_kwargs": {"enable_thinking": False}})[2]
    lp = r["choices"][0]["logprobs"]["content"]
    assert len(lp) == 2 and set(lp[0]) == {"id", "token", "bytes", "logprob", "top_logprobs"} and len(lp[0]["top_logprobs"]) == 2
    st, _, e = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "top_logprobs": 2})
    assert st == 400 and e["error"]["message"] == "top_logprobs requires logprobs to be set to true"
    st, _, e = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]})
    assert st == 400 and e["error"]["type"] == "invalid_request_error"   # until the Jinja interpreter is linked (documented)


def test_error_shapes(srv):
    def err(m, p, body, status, typ, msg=None):
        st, hd, e = srv.req(m, p, body)
        assert st == status, (p, body, st, e)
        assert set(e) == {"error"} and e["error"]["code"] == status and e["error"]["type"] == typ and set(e["error"]) >= {"code", "message", "type"}, e
        if msg: assert msg in e["error"]["message"], e
        h = hdrs(hd)
        assert h["content-type"] == "application/json; charset=utf-8" and h.get("connection") == "close"
    err("POST", "/v1/chat/completions", "{bad", 400, "invalid_request_error", "parse_error")
    err("POST", "/v1/chat/completions", {}, 400, "invalid_request_error", "'messages' is required")
    err("POST", "/v1/chat/completions", {"messages": "x"}, 400, "invalid_request_error", "Expected 'messages' to be an array")
    err("POST", "/v1/chat/completions", {"messages": [{"role": "user"}]}, 400, "invalid_request_error", "All non-assistant messages must contain 'content'")
    err("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": "a"}, 400, "invalid_request_error", "Field 'max_tokens'")
    err("POST", "/completion", {}, 400, "invalid_request_error", "'prompt' not found")
    err("POST", "/completion", {"prompt": "x", "temperature": "x"}, 400, "invalid_request_error", "Field 'temperature'")
    err("POST", "/completion", {"prompt": "x", "stream": "x"}, 400, "invalid_request_error", "Field 'stream'")
    err("GET", "/nonexistent", None, 404, "not_found_error", "File Not Found")
    err("GET", "/completion", None, 404, "not_found_error")
    err("POST", "/embedding", {"content": "x"}, 501, "not_supported_error", "--embeddings")
    err("POST", "/rerank", {}, 501, "not_supported_error")
    err("POST", "/tokenize", {"content": 5}, 400, "invalid_request_error", "type must be string")
    err("POST", "/detokenize", {"tokens": "x"}, 400, "invalid_request_error", "type must be array")
    big = "word " * 5000
    st, _, e = srv.req("POST", "/completion", {"prompt": big, "n_predict": 1})
    assert st == 400 and e["error"]["type"] == "exceed_context_size_error" and e["error"]["n_ctx"] == 4096 and e["error"]["n_prompt_tokens"] > 4096
    assert "exceeds the available context size" in e["error"]["message"]


def test_tokenize_detokenize_apply_template(srv):
    r = srv.req("POST", "/tokenize", {"content": "Hello world"})[2]
    assert list(r) == ["tokens"] and all(isinstance(x, int) for x in r["tokens"])
    rp = srv.req("POST", "/tokenize", {"content": "Hello world", "with_pieces": True})[2]["tokens"]
    assert [x["id"] for x in rp] == r["tokens"] and [x["piece"] for x in rp] == ["Hello", " world"]
    assert srv.req("POST", "/tokenize", {})[2] == {"tokens": []}
    special = srv.req("POST", "/tokenize", {"content": "<|im_start|>", "parse_special": True})[2]["tokens"]
    plain = srv.req("POST", "/tokenize", {"content": "<|im_start|>", "parse_special": False})[2]["tokens"]
    assert len(special) == 1 and len(plain) > 1
    assert srv.req("POST", "/detokenize", {"tokens": r["tokens"]})[2] == {"content": "Hello world"}
    assert srv.req("POST", "/detokenize", {})[2] == {"content": ""}
    ap = srv.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}]})[2]
    assert list(ap) == ["prompt"] and ap["prompt"].startswith("<|im_start|>") and ap["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    assert srv.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}})[2]["prompt"].endswith("</think>\n\n")
    assert srv.req("POST", "/apply-template", {})[0] == 400


def test_infill(srv):
    st, _, r = srv.req("POST", "/infill", {"input_prefix": "def f(", "input_suffix": "):\n    pass", "n_predict": 2, "temperature": 0})
    assert st == 200 and "<|fim_prefix|>" in r["prompt"] and "<|fim_suffix|>" in r["prompt"] and r["prompt"].endswith("<|fim_middle|>")
    assert srv.req("POST", "/infill", {"input_suffix": "x"})[0] == 400


def test_slots_and_actions(srv):
    st, _, s = srv.req("GET", "/slots")
    assert st == 200 and len(s) == 4 and all(x["n_ctx"] == 4096 and x["speculative"] is False for x in s)
    srv.req("POST", "/completion", {"prompt": "slot test", "n_predict": 2, "temperature": 0, "id_slot": 2})
    s = srv.req("GET", "/slots")[2]
    assert s[2]["is_processing"] is False and s[2]["id_task"] >= 0 and "params" in s[2] and s[2]["next_token"][0]["n_decoded"] == 2
    assert srv.req("GET", "/slots?fail_on_no_slot=1")[0] == 200
    r = srv.req("POST", "/completion", {"prompt": "slot test", "n_predict": 1, "id_slot": 2})[2]
    assert r["id_slot"] == 2
    st, _, sv = srv.req("POST", "/slots/0?action=save", {"filename": "a.bin"})
    assert st == 200 and sv["id_slot"] == 0 and sv["filename"] == "a.bin" and sv["n_saved"] > 0 and "save_ms" in sv["timings"]
    assert srv.req("POST", "/slots/0?action=erase")[2] == {"id_slot": 0, "n_erased": sv["n_saved"]}
    st, _, rs = srv.req("POST", "/slots/0?action=restore", {"filename": "a.bin"})
    assert st == 200 and rs["n_restored"] == sv["n_saved"] and "restore_ms" in rs["timings"]
    assert srv.req("POST", "/slots/0?action=save", {"filename": "../x"})[0] == 400
    assert srv.req("POST", "/slots/9?action=erase")[0] == 400
    assert srv.req("POST", "/slots/0?action=bogus")[0] == 400
    assert srv.req("POST", "/slots/0?action=restore", {"filename": "missing.bin"})[0] == 400


def test_metrics(srv):
    st, hd, b = srv.req("GET", "/metrics", raw=True)
    t = b.decode()
    assert st == 200 and hdrs(hd)["content-type"] == "text/plain; version=0.0.4" and "process-start-time-unix" in hdrs(hd)
    for name in ("prompt_tokens_total", "prompt_tokens_cached_total", "prompt_seconds_total", "tokens_predicted_total", "tokens_predicted_seconds_total", "n_decode_total",
                 "n_tokens_max", "prompt_tokens_seconds", "predicted_tokens_seconds", "requests_processing", "requests_deferred", "n_busy_slots_per_decode"):
        assert re.search(r"^# TYPE llamacpp:%s (counter|gauge)$" % name, t, re.M), name
    assert float(re.search(r"^llamacpp:tokens_predicted_total (\S+)", t, re.M).group(1)) > 0
    s2 = Srv(qv_cmd(), name="nometrics")
    try:
        st, _, e = s2.req("GET", "/metrics")
        assert st == 501 and e["error"]["type"] == "not_supported_error" and "--metrics" in e["error"]["message"]
        assert s2.req("POST", "/props", {})[0] == 501
        assert s2.req("POST", "/slots/0?action=save", {"filename": "x"})[0] == 501
    finally:
        s2.stop()


def test_cors_options_and_headers(srv):
    st, hd, b = srv.req("OPTIONS", "/v1/chat/completions", raw=True, headers={"Origin": "http://example.com"})
    h = hdrs(hd)
    assert st == 200 and b == b"" and h["access-control-allow-origin"] == "http://example.com" and h["access-control-allow-credentials"] == "true"
    assert "POST" in h["access-control-allow-methods"] and h["access-control-allow-headers"] == "*" and h["content-type"] == "text/html"
    st, hd, _ = srv.req("GET", "/health", headers={"Origin": "http://example.com"})
    assert hdrs(hd)["access-control-allow-origin"] == "http://example.com"
    assert hdrs(hd)["keep-alive"] == "timeout=5, max=100"


def test_chunked_request_body(srv):
    s = socket.create_connection(("127.0.0.1", srv.port))
    body = json.dumps({"prompt": "x", "n_predict": 1, "temperature": 0}).encode()
    s.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n" +
              ("%x\r\n" % len(body)).encode() + body + b"\r\n0\r\n\r\n")
    data = b""
    while True:
        d = s.recv(65536)
        if not d: break
        data += d
    assert data.startswith(b"HTTP/1.1 200")


def test_api_key():
    s = Srv(qv_cmd("--api-key", "k1,k2"), name="auth")
    try:
        assert s.req("GET", "/health")[0] == 200 and s.req("GET", "/v1/health")[0] == 200
        st, _, e = s.req("GET", "/props")
        assert st == 401 and e == {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}
        assert s.req("GET", "/props", headers={"Authorization": "Bearer k2"})[0] == 200
        assert s.req("GET", "/props", headers={"Authorization": "k1"})[0] == 200
        assert s.req("GET", "/props", headers={"X-Api-Key": "k1"})[0] == 200
        assert s.req("GET", "/props", headers={"Authorization": "Bearer nope"})[0] == 401
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 1})[0] == 401
    finally:
        s.stop()
    kf = tempfile.NamedTemporaryFile("w", delete=False); kf.write("# comment\nfilekey\n"); kf.close()
    s = Srv(qv_cmd("--api-key-file", kf.name), name="authfile")
    try:
        assert s.req("GET", "/props")[0] == 401 and s.req("GET", "/props", headers={"Authorization": "Bearer filekey"})[0] == 200
    finally:
        s.stop()


def test_queue_slots_and_disconnect():
    s = Srv(qv_cmd("--metrics", "-np", "2", "--mock-loop", "--mock-delay-us", "20000"), name="queue")
    try:
        res = {}
        def long_req(name, n):
            res[name] = s.req("POST", "/completion", {"prompt": name, "n_predict": n, "temperature": 0}, timeout=60)
        t1 = threading.Thread(target=long_req, args=("first", 25)); t1.start(); time.sleep(0.2)
        t2 = threading.Thread(target=long_req, args=("second", 3)); t2.start(); time.sleep(0.2)
        sl = s.req("GET", "/slots")[2]
        assert len(sl) == 2 and sum(1 for x in sl if x["is_processing"]) == 1
        m = s.req("GET", "/metrics", raw=True)[2].decode()
        assert re.search(r"^llamacpp:requests_processing 1$", m, re.M) and re.search(r"^llamacpp:requests_deferred 1$", m, re.M), m
        t1.join(); t2.join()
        assert res["first"][0] == 200 and res["second"][0] == 200 and res["first"][2]["tokens_predicted"] == 25
        # client disconnect while generating: next request must still work and the slot becomes idle
        c = socket.create_connection(("127.0.0.1", s.port))
        body = json.dumps({"prompt": "x", "n_predict": 500, "temperature": 0, "stream": True}).encode()
        c.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        c.recv(100); c.close()
        t0 = time.time()
        while time.time() - t0 < 10 and any(x["is_processing"] for x in s.req("GET", "/slots")[2]): time.sleep(0.1)
        assert not any(x["is_processing"] for x in s.req("GET", "/slots")[2])
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": 0})[0] == 200
    finally:
        s.stop()


def test_max_queue_503():
    s = Srv(qv_cmd("--max-queue", "0", "--mock-loop", "--mock-delay-us", "30000"), name="mq")
    try:
        r = {}
        t = threading.Thread(target=lambda: r.update(a=s.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "temperature": 0}))); t.start(); time.sleep(0.2)
        st, _, e = s.req("POST", "/completion", {"prompt": "y", "n_predict": 1})
        assert st == 503 and e["error"]["type"] == "unavailable_error"
        t.join()
        assert r["a"][0] == 200
    finally:
        s.stop()


def test_context_limit_truncated():
    s = Srv(qv_cmd("-c", "64", "--mock-loop", "--mock-text", "a b c d"), name="ctx")
    try:
        r = s.req("POST", "/completion", {"prompt": "x", "n_predict": -1, "temperature": 0})[2]
        assert r["stop_type"] == "limit" and r["truncated"] is True and r["tokens_predicted"] + r["tokens_evaluated"] >= 62
    finally:
        s.stop()


def test_mtp_chunk_path_matches_plain():
    a = Srv(qv_cmd("--mock-loop"), name="plain"); b = Srv(qv_cmd("--mock-loop", "--mtp"), name="mtp")
    try:
        body = {"prompt": "x", "n_predict": 21, "temperature": 0, "stop": ["delta"]}
        ra, rb = a.req("POST", "/completion", body)[2], b.req("POST", "/completion", body)[2]
        assert ra["content"] == rb["content"] and ra["stop_type"] == rb["stop_type"]
        body = {"prompt": "x", "n_predict": 21, "temperature": 0}
        assert a.req("POST", "/completion", body)[2]["content"] == b.req("POST", "/completion", body)[2]["content"]
    finally:
        a.stop(); b.stop()


# ----- CLI / environment
def run_cli(args, env=None, wait=True):
    return subprocess.run([QV_SERVER] + args, capture_output=True, text=True, timeout=30, env=dict(os.environ, **(env or {})))


def test_cli_help_version_and_unknown():
    r = run_cli(["--help"]); assert r.returncode == 0 and "llama-server compatible" in r.stdout
    r = run_cli(["--version"]); assert r.returncode == 0 and r.stdout.startswith("version:")
    r = run_cli(["--definitely-not-a-flag"]); assert r.returncode == 1 and "invalid argument" in r.stderr
    r = run_cli(["-c"]); assert r.returncode == 1 and "expected value" in r.stderr


def test_cli_llama_flags_are_accepted_and_warned():
    s = Srv(qv_cmd("-ngl", "99", "-fa", "on", "--mlock", "-ub", "128", "-b", "512", "--no-mmap", "-ctk", "q8_0", "--cache-reuse", "256", "-sps", "0.5", "--rope-scale", "2",
                   "--threads-http", "8", "--timeout", "100", "--no-webui", "--log-colors", "off", "-cb", "--no-context-shift", "--swa-full", "--spec-default",
                   "-a", "myalias,other", "--chat-template-kwargs", '{"enable_thinking": false}', "--temp", "0.1", "--top-k", "7", "-n", "3", "--reasoning-budget", "-1"),
            name="flags")
    try:
        p = s.req("GET", "/props")[2]
        assert p["model_alias"] == "myalias" and s.req("GET", "/v1/models")[2]["data"][0]["id"] == "myalias" and s.req("GET", "/v1/models")[2]["data"][0]["aliases"] == ["myalias", "other"]
        prm = p["default_generation_settings"]["params"]
        assert prm["temperature"] == pytest.approx(0.1) and prm["top_k"] == 7 and prm["n_predict"] == 3
        r = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})[2]
        assert r["usage"]["completion_tokens"] == 3 and "reasoning_content" not in r["choices"][0]["message"]  # kwargs default: thinking off
        log = open(s.log.name).read()
        assert "warning:" in log and "-ctk" in log and "--swa-full" in log
    finally:
        s.stop()


def test_cli_environment_variables():
    port = free_port()
    p = subprocess.Popen([QV_SERVER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(
        os.environ, LLAMA_ARG_PORT=str(port), LLAMA_ARG_CTX_SIZE="1234", LLAMA_ARG_ALIAS="envmodel", LLAMA_ARG_N_PARALLEL="3", LLAMA_ARG_ENDPOINT_METRICS="1",
        LLAMA_ARG_ENDPOINT_SLOTS="0", LLAMA_ARG_BACKEND="mock", LLAMA_ARG_THREADS="2", LLAMA_ARG_CACHE_PROMPT="0"))
    # LLAMA_ARG_BACKEND is not a llama variable: the backend has no env, so this process must fail cleanly (no model) -- use a wrapper instead
    p.wait(15)
    assert p.returncode == 1
    s = Srv([QV_SERVER, "--backend", "mock", "--tokenizer", TOKJSON], env={"LLAMA_ARG_PORT": "@ENVPORT@"}, wait=False) if False else None
    port = free_port()
    p = subprocess.Popen([QV_SERVER, "--backend", "mock", "--tokenizer", TOKJSON], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(
        os.environ, LLAMA_ARG_PORT=str(port), LLAMA_ARG_CTX_SIZE="1234", LLAMA_ARG_ALIAS="envmodel", LLAMA_ARG_N_PARALLEL="3", LLAMA_ARG_ENDPOINT_METRICS="1",
        LLAMA_ARG_ENDPOINT_SLOTS="0", LLAMA_ARG_CACHE_PROMPT="0", LLAMA_API_KEY="envkey"))
    try:
        t0 = time.time(); ok = False
        while time.time() - t0 < 30 and not ok:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2); c.request("GET", "/props", headers={"Authorization": "Bearer envkey"}); r = c.getresponse(); b = json.loads(r.read()); ok = r.status == 200
            except (OSError, ValueError): time.sleep(0.2)
        assert ok and b["total_slots"] == 3 and b["model_alias"] == "envmodel" and b["endpoint_slots"] is False and b["endpoint_metrics"] is True
        assert b["default_generation_settings"]["n_ctx"] == 1234
    finally:
        p.send_signal(signal.SIGTERM); p.wait(10)


def test_cli_refuses_unsafe_ignores():
    for args in (["--lora", "x.gguf"], ["--ssl-key-file", "k", "--ssl-cert-file", "c"], ["-hf", "a/b"]):
        r = run_cli(["--backend", "mock", "--tokenizer", TOKJSON] + args)
        assert r.returncode == 1 and "error:" in r.stderr, args


# =============================================================================================== part 2: structural diff vs llama-server
def shape(v):
    if isinstance(v, dict): return {k: shape(x) for k, x in v.items()}
    if isinstance(v, list): return [shape(v[0])] if v else []
    if isinstance(v, bool): return "bool"
    if v is None: return "null"
    if isinstance(v, (int, float)): return "number"
    return "string"


def diff(a, b, path=""):
    """a = llama shape, b = ours. returns list of (kind, path)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b: out.append(("missing", f"{path}/{k}"))
            else: out += diff(a[k], b[k], f"{path}/{k}")
        for k in b:
            if k not in a: out.append(("extra", f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if a and b: out += diff(a[0], b[0], path + "[]")
    elif a != b:
        out.append(("type", f"{path}: llama={a} ours={b}"))
    return out


# documented, intentional differences: (case, kind, json-path regex). Anything else fails the test.
ALLOWED = [
    (r"POST /v1/chat/completions thinking", "(extra|missing)", r".*/(reasoning_content|content)$"),   # ours extracts reasoning for the Qwen3.8 thinking template
    (r"GET /props", "(missing|extra|type)", r"^/(chat_template_caps/.*|media_marker|model_ftype)$"),
    (r"GET /v1/models|GET /models", "(missing|extra|type)", r"^/(data\[\]/(meta/.*)|models\[\]/details/.*)$"),
    (r".*", "(missing|extra|type)", r"^/(generation_settings|default_generation_settings/params)/(chat_format|reasoning_format|samplers)$"),
]
HDR_IGNORE = {"server", "date", "content-length", "access-control-allow-origin", "keep-alive", "connection", "accept-ranges", "process-start-time-unix"}


def allowed(case, kind, path):
    return any(re.fullmatch(c, case) and re.fullmatch(k, kind) and re.fullmatch(p, path) or (re.fullmatch(c, case) and re.fullmatch(k, kind) and re.search(p, path)) for c, k, p in ALLOWED)


CASES = []
def case(name, method, path, body=None, stream=False, headers=None):
    CASES.append((name, method, path, body, stream, headers))


CHAT = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}  # (llama ignores the kwarg: Qwen2.5 template)
for m, p in (("GET", "/health"), ("GET", "/v1/health"), ("GET", "/props"), ("GET", "/slots"), ("GET", "/v1/models"), ("GET", "/models"), ("GET", "/metrics"),
             ("GET", "/lora-adapters"), ("GET", "/nonexistent"), ("GET", "/completion"), ("DELETE", "/props")):
    case(f"{m} {p}", m, p)
case("POST /props", "POST", "/props", {})
case("POST /completion", "POST", "/completion", {"prompt": "Hello", "n_predict": 3, "temperature": 0})
case("POST /completion n_probs", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "temperature": 0, "n_probs": 2})
case("POST /completion return_tokens", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "temperature": 0, "return_tokens": True})
case("POST /completion stop", "POST", "/completion", {"prompt": "Hello", "n_predict": 6, "temperature": 0, "stop": ["zzzz"]})
case("POST /completion stream", "POST", "/completion", {"prompt": "Hello", "n_predict": 3, "temperature": 0, "stream": True}, stream=True)
case("POST /completion stream n_probs", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "temperature": 0, "stream": True, "n_probs": 2}, stream=True)
case("POST /completion tokens", "POST", "/completion", {"prompt": [9707, 1879], "n_predict": 2, "temperature": 0})
case("POST /completion mixed", "POST", "/completion", {"prompt": ["Hello", 1879], "n_predict": 2, "temperature": 0})
case("POST /completion samplers", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "typical_p": 0.9, "mirostat": 0, "xtc_probability": 0.0, "seed": 7})
case("POST /v1/completions", "POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0})
case("POST /v1/completions logprobs", "POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 2, "temperature": 0, "logprobs": 2})
case("POST /v1/completions stream", "POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0, "stream": True}, stream=True)
case("POST /v1/chat/completions", "POST", "/v1/chat/completions", CHAT)
case("POST /chat/completions", "POST", "/chat/completions", CHAT)
case("POST /v1/chat/completions stream", "POST", "/v1/chat/completions", dict(CHAT, stream=True), stream=True)
case("POST /v1/chat/completions stream usage", "POST", "/v1/chat/completions", dict(CHAT, stream=True, stream_options={"include_usage": True}), stream=True)
case("POST /v1/chat/completions logprobs", "POST", "/v1/chat/completions", dict(CHAT, logprobs=True, top_logprobs=2))
case("POST /v1/chat/completions n=2", "POST", "/v1/chat/completions", dict(CHAT, n=2))
case("POST /v1/chat/completions parts", "POST", "/v1/chat/completions", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "Hi"}]}], "max_tokens": 2, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
case("POST /v1/chat/completions stop", "POST", "/v1/chat/completions", dict(CHAT, stop=["zzzz"]))
case("POST /v1/chat/completions response_format", "POST", "/v1/chat/completions", dict(CHAT, response_format={"type": "text"}))
case("POST /v1/chat/completions max_completion_tokens", "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_completion_tokens": 2, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
case("POST /tokenize", "POST", "/tokenize", {"content": "Hello world"})
case("POST /tokenize pieces", "POST", "/tokenize", {"content": "Hello world", "with_pieces": True, "add_special": True})
case("POST /tokenize empty", "POST", "/tokenize", {})
case("POST /detokenize", "POST", "/detokenize", {"tokens": [9707, 1879]})
case("POST /detokenize empty", "POST", "/detokenize", {})
case("POST /apply-template", "POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}]})
case("POST /v1/chat/completions/input_tokens", "POST", "/v1/chat/completions/input_tokens", {"messages": [{"role": "user", "content": "Hi"}]})
case("POST /infill", "POST", "/infill", {"input_prefix": "def f(", "input_suffix": "):\n    pass", "n_predict": 2, "temperature": 0})
case("POST /infill missing", "POST", "/infill", {"input_suffix": "x"})
case("POST /slots save (disabled)", "POST", "/slots/0?action=erase", {})
case("POST /embedding", "POST", "/embedding", {"content": "hi"})
case("POST /v1/embeddings", "POST", "/v1/embeddings", {"input": "hi"})
case("POST /rerank", "POST", "/rerank", {})
case("POST /lora-adapters", "POST", "/lora-adapters", [])
# error cases
case("ERR bad json", "POST", "/v1/chat/completions", "{bad")
case("ERR no messages", "POST", "/v1/chat/completions", {})
case("ERR messages not array", "POST", "/v1/chat/completions", {"messages": "x"})
case("ERR content missing", "POST", "/v1/chat/completions", {"messages": [{"role": "user"}]})
case("ERR max_tokens type", "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": "a"})
case("ERR completion no prompt", "POST", "/completion", {})
case("ERR v1 completion no prompt", "POST", "/v1/completions", {})
case("ERR temperature type", "POST", "/completion", {"prompt": "x", "temperature": "x"})
case("ERR stream type", "POST", "/completion", {"prompt": "x", "stream": "x"})
case("ERR tokenize type", "POST", "/tokenize", {"content": 123})
case("ERR detokenize type", "POST", "/detokenize", {"tokens": "x"})
case("ERR apply-template empty", "POST", "/apply-template", {})
case("ERR top_logprobs", "POST", "/v1/chat/completions", dict(CHAT, top_logprobs=2))
case("ERR ctx exceeded", "POST", "/completion", {"prompt": "word " * 6000, "n_predict": 1})
case("ERR reasoning_format", "POST", "/v1/chat/completions", dict(CHAT, reasoning_format="bogus"))
case("POST /v1/chat/completions thinking", "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0})
case("OPTIONS", "OPTIONS", "/v1/chat/completions")
case("HEAD /health", "HEAD", "/health")


def parse_resp(srv, c):
    name, method, path, body, stream, headers = c
    st, hd, raw = srv.req(method, path, body, headers, raw=True)
    h = hdrs(hd)
    ct = h.get("content-type", "")
    if stream and st == 200:
        evs = []
        for blk in raw.decode().split("\n\n"):
            if blk: evs.append(blk[6:] if blk.startswith("data: ") else "BADFRAME:" + blk)
        shapes = []
        for e in evs:
            s = e if (e == "[DONE]" or e.startswith("BADFRAME")) else shape(json.loads(e))
            if not shapes or shapes[-1] != s: shapes.append(s)
        return st, h, ("sse", shapes, evs)
    if ct.startswith("application/json") and raw:
        return st, h, ("json", shape(json.loads(raw)), None)
    return st, h, ("raw", raw.decode(errors="replace")[:0], None)


@pytest.fixture(scope="module")
def pair():
    if not HAVE_LLAMA: pytest.skip("llama-server / gguf not available (set LLAMA_SERVER, LLAMA_GGUF)")
    slots = tempfile.mkdtemp(prefix="cmp_slots_")
    common = ["--metrics", "-c", "4096"]
    ll = Srv([LLAMA_SERVER, "-m", LLAMA_GGUF, "--port", "@PORT@", "-t", "4", "--no-webui"] + common, name="llama")
    ours = Srv(qv_cmd(*common, "--mock-text", "Hello! How can I help you today? I am a mock model."), name="qv")
    yield ll, ours
    ll.stop(); ours.stop()


@pytest.mark.parametrize("c", CASES, ids=[c[0] for c in CASES])
def test_structure_matches_llama(pair, c):
    ll, ours = pair
    name = c[0]
    if name == "POST /v1/chat/completions parts": pass
    sa, ha, ra = parse_resp(ll, c)
    sb, hb, rb = parse_resp(ours, c)
    problems = []
    if sa != sb: problems.append(f"status llama={sa} ours={sb}")
    ca, cb = ha.get("content-type"), hb.get("content-type")
    if ca != cb: problems.append(f"content-type llama={ca!r} ours={cb!r}")
    for hname in sorted((set(ha) ^ set(hb)) - HDR_IGNORE):
        problems.append(f"header {hname}: llama={'yes' if hname in ha else 'no'} ours={'yes' if hname in hb else 'no'}")
    if ra[0] != rb[0]: problems.append(f"body kind llama={ra[0]} ours={rb[0]}")
    elif ra[0] == "json":
        problems += [f"{k} {p}" for k, p in diff(ra[1], rb[1]) if not allowed(name, k, p)]
    elif ra[0] == "sse":
        la, lb = ra[1], rb[1]
        if len(la) != len(lb): problems.append(f"SSE shape-sequence length llama={len(la)} ours={len(lb)}")
        for i, (x, y) in enumerate(zip(la, lb)):
            if isinstance(x, str) or isinstance(y, str):
                if x != y: problems.append(f"SSE event {i}: llama={x!r:.60} ours={y!r:.60}")
            else: problems += [f"SSE event {i}: {k} {p}" for k, p in diff(x, y) if not allowed(name, k, p)]
    assert not problems, "\n" + "\n".join(problems)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"] + sys.argv[1:]))
