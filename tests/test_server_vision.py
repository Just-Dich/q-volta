#!/usr/bin/env python3
"""Image input (`image_url` content parts) of qv_server with the mock engine + mock vision encoder (no weights needed).
  QV_SERVER=build/qv_server python3 -m pytest tests/test_server_vision.py -q
The real encoder is covered by tools/vision_check.py / tools/vision_e2e_check.py (docs/INKLING_VISION.md)."""
import base64, json, os, struct, sys, tempfile, zlib
import pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QV_SERVER", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "qv_server"))
import test_server_compat as T

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}"
            "{% if p.type == 'text' %}{{ p.text }}{% elif p.type == 'image_url' %}<|vision_start|><|image_pad|><|vision_end|>{% endif %}{% endfor %}{% endif %}"
            "<|im_end|>\n{% endfor %}<|im_start|>assistant\n")


def png(w, h, rgba=False):
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    ch = 4 if rgba else 3
    raw = b"".join(b"\0" + bytes([(x * 7 + y * 3) & 255 for x in range(w) for _ in range(ch)]) for y in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if rgba else 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def durl(b, mt="image/png"): return f"data:{mt};base64," + base64.b64encode(b).decode()


def msg(url, text="what is this?"):
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}], "max_tokens": 4}


@pytest.fixture(scope="module")
def tmpl():
    f = tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False); f.write(TEMPLATE); f.close(); return f.name


@pytest.fixture(scope="module")
def root():
    d = tempfile.mkdtemp(prefix="qvimg_"); open(d + "/a.png", "wb").write(png(80, 80)); return d


@pytest.fixture(scope="module")
def srv(tmpl, root):
    s = T.Srv(T.qv_cmd("--props", "--vision-file", "mock", "--image-token", "<|image_pad|>", "--chat-template-file", tmpl, "--image-root", root, "--image-max-side", "200",
                       "--mock-text", "ok"), name="vis")
    yield s
    s.stop()


def test_props_and_models(srv):
    st, _, b = srv.req("GET", "/props")
    assert st == 200 and b["modalities"]["vision"] is True
    st, _, b = srv.req("GET", "/v1/models")
    assert "image" in b["data"][0]["architecture"]["input_modalities"] if "architecture" in b["data"][0] else True


def test_token_accounting(srv):
    # 100x60 -> ceil(60/40)*ceil(100/40) = 2*3 = 6 rows; 40x40 -> 1 row; the difference is exactly 5 prompt tokens
    st, _, a = srv.req("POST", "/v1/chat/completions", msg(durl(png(100, 60))))
    st2, _, b = srv.req("POST", "/v1/chat/completions", msg(durl(png(40, 40))))
    assert st == 200 and st2 == 200, (a, b)
    assert a["usage"]["prompt_tokens"] - b["usage"]["prompt_tokens"] == 5
    assert a["choices"][0]["message"]["content"]


def test_rgba_and_odd_size(srv):
    st, _, a = srv.req("POST", "/v1/chat/completions", msg(durl(png(33, 47, rgba=True))))
    st2, _, b = srv.req("POST", "/v1/chat/completions", msg(durl(png(40, 80))))
    assert st == 200 and a["usage"]["prompt_tokens"] == b["usage"]["prompt_tokens"]  # 33x47 -> 2 rows x 1 col == 40x80


def test_stream(srv):
    st, _, ev, _ = srv.sse("/v1/chat/completions", dict(msg(durl(png(80, 80))), stream=True, stream_options={"include_usage": True}))
    assert st == 200 and ev[-1] == "[DONE]"
    usage = [e for e in ev if isinstance(e, dict) and e.get("usage")]
    assert usage and usage[-1]["usage"]["prompt_tokens"] > 4


def test_two_images_and_cache(srv):
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": durl(png(40, 40))}}, {"type": "image_url", "image_url": {"url": durl(png(80, 40))}},
                                                       {"type": "text", "text": "compare"}]}], "max_tokens": 2}
    st, _, a = srv.req("POST", "/v1/chat/completions", body)
    assert st == 200
    st, _, b = srv.req("POST", "/v1/chat/completions", body)  # identical request: prompt cache may be reused, result must not fail
    assert st == 200 and b["usage"]["prompt_tokens"] == a["usage"]["prompt_tokens"]


def test_errors(srv, root):
    st, _, b = srv.req("POST", "/v1/chat/completions", msg(durl(png(300, 40))))
    assert st == 400 and "at most 200" in b["error"]["message"]
    st, _, b = srv.req("POST", "/v1/chat/completions", msg("data:image/png;base64,AAAA"))
    assert st == 400 and "image" in b["error"]["message"]
    st, _, b = srv.req("POST", "/v1/chat/completions", msg("data:image/png;base64,@@@@"))
    assert st == 400
    st, _, b = srv.req("POST", "/v1/chat/completions", msg("https://example.com/a.png"))
    assert st == 400 and "--allow-remote-images" in b["error"]["message"]
    st, _, b = srv.req("POST", "/v1/chat/completions", msg("/etc/passwd"))
    assert st == 400 and "outside" in b["error"]["message"]
    st, _, b = srv.req("POST", "/v1/chat/completions", msg(root + "/../../etc/passwd"))
    assert st == 400
    st, _, b = srv.req("POST", "/v1/chat/completions", msg(root + "/a.png"))
    assert st == 200
    st, _, b = srv.req("POST", "/v1/chat/completions", msg("file://" + root + "/a.png"))
    assert st == 200
    st, _, b = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]})
    assert st == 400


def test_remote_private_refused(tmpl, root):
    s = T.Srv(T.qv_cmd("--vision-file", "mock", "--image-token", "<|image_pad|>", "--chat-template-file", tmpl, "--allow-remote-images", "--mock-text", "ok"), name="vis2")
    try:
        for u in ("http://127.0.0.1:1/a.png", "http://169.254.169.254/latest/meta-data", "http://[::1]/a.png", "http://user:pw@example.com/a.png"):
            st, _, b = s.req("POST", "/v1/chat/completions", msg(u))
            assert st == 400, (u, b)
    finally:
        s.stop()


def test_no_vision_server(tmpl):
    s = T.Srv(T.qv_cmd("--props", "--chat-template-file", tmpl, "--mock-text", "ok"), name="novis")
    try:
        st, _, b = s.req("GET", "/props")
        assert b["modalities"]["vision"] is False
        st, _, b = s.req("POST", "/v1/chat/completions", msg(durl(png(40, 40))))
        assert st == 500 and "image input is not supported" in b["error"]["message"]
    finally:
        s.stop()
