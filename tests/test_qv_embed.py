#!/usr/bin/env python3
"""Self-contained .qv (tokenizer / chat template / generation config / meta embedded in the container), docs/QV_FORMAT.md.

  python3 -m pytest tests/test_qv_embed.py -q        (needs build/qv_server + build/qv_tok; no model weights: a synthetic .qv with one tensor)

Part 1: tools/qv_embed.py round trip (tensor area byte-identical, blobs, meta, list/extract, replace/drop, in place, errors).
Part 2: qv_tok and qv_server (mock backend) with NO --tokenizer: /tokenize equals the --tokenizer path, embedded template in /props and
        rendering, embedded stop ids / sampling defaults (CLI flags win; --no-embedded-defaults), /v1/models metadata, clear error without a tokenizer.
"""
import hashlib, json, os, struct, subprocess, sys, tempfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import qv_blobs
import qv_embed
from test_server_compat import Srv

BUILD = os.environ.get("QV_BUILD", os.path.join(ROOT, "build"))
QV_SERVER = os.environ.get("QV_SERVER", os.path.join(BUILD, "qv_server"))
QV_TOK = os.environ.get("QV_TOK", os.path.join(BUILD, "qv_tok"))
TOKJSON = os.environ.get("QV_TOKENIZER", "/workspace/q-volta-models/Qwen3.8-27B/tokenizer.json")
TEMPLATE = os.environ.get("QV_TEMPLATE", "/workspace/q-volta-models/Qwen3.8-27B/chat_template.jinja")
pytestmark = pytest.mark.skipif(not os.path.exists(TOKJSON) or not os.path.exists(QV_SERVER), reason="needs a tokenizer.json and build/qv_server")

GEN = {"bos_token_id": 248044, "eos_token_id": [248046, 248044], "pad_token_id": 248044, "temperature": 0.6, "top_k": 7, "top_p": 0.9, "do_sample": True}
TINY_TEMPLATE = "{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}"


def make_qv(path, tensor_bytes=b"\x01\x02\x03\x04" * 8):
    """Minimal version-1 container: header + one f32 tensor, like tools/qv_pack.py output."""
    idx = {"format": "qvolta-v1", "config": {"arch": "inkling2_mtp", "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 8, "num_attention_heads": 1,
                                             "num_key_value_heads": 1, "head_dim": 8, "rms_norm_eps": 1e-6, "eos_token_id": 248046},
           "tensors": {"t": {"kind": "f32", "shape": [8], "off": 0, "nbytes": len(tensor_bytes)}}}
    js = json.dumps(idx).encode()
    with open(path, "wb") as f:
        f.write(b"QVOLTA1\0" + struct.pack("<QQ", len(js), 2 << 20) + js)
        f.write(b"\0" * ((2 << 20) - 24 - len(js)))
        f.write(tensor_bytes)
    return path


def tensor_hash(path):
    idx, off, _ = qv_blobs.read_header(path)
    return qv_blobs.sha256_file_range(path, off, qv_blobs.tensor_end(idx))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("embed")
    (d / "generation_config.json").write_text(json.dumps(GEN))
    (d / "chat_template.jinja").write_text(TINY_TEMPLATE)
    (d / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": False, "eos_token": "<|im_end|>", "model_max_length": 262144, "added_tokens_decoder": {"1": {}}}))
    return d


@pytest.fixture(scope="module")
def st(files):
    src = make_qv(str(files / "src.qv"))
    out = str(files / "out.qv")
    subprocess.check_call([sys.executable, "-I", os.path.join(ROOT, "tools", "qv_embed.py"), src, "--tokenizer", TOKJSON, "--chat-template", str(files / "chat_template.jinja"),
                           "--generation-config", str(files / "generation_config.json"), "--tokenizer-config", str(files / "tokenizer_config.json"),
                           "--meta", "model_name=unit-test", "--meta", "license=apache-2.0", "-o", out])
    return src, out


# ------------------------------------------------------------------------------------------------ part 1: the tool
def test_roundtrip(files, st):
    src, out = st
    assert tensor_hash(src) == tensor_hash(out)  # tensor area byte-identical
    idx, off, _ = qv_blobs.read_header(out)
    assert idx["format"] == "qvolta-v1" and idx["format_version"] == 2
    assert set(idx["blobs"]) == {"tokenizer.json", "chat_template.jinja", "generation_config.json", "tokenizer_config.json"}
    assert off % (2 << 20) == 0 and all(e["off"] % 64 == 0 for e in idx["blobs"].values())
    raw = open(out, "rb").read()
    for n, e in idx["blobs"].items():
        assert hashlib.sha256(raw[off + e["off"]: off + e["off"] + e["nbytes"]]).hexdigest() == e["sha256"]
    assert raw[off + idx["blobs"]["tokenizer.json"]["off"]:][:idx["blobs"]["tokenizer.json"]["nbytes"]] == open(TOKJSON, "rb").read()
    m = idx["meta"]
    assert m["model_name"] == "unit-test" and m["license"] == "apache-2.0" and m["qv_format_version"] == 2 and m["arch"] == "inkling2_mtp"
    assert m["eos_token_ids"] == [248046, 248044] and m["bos_token_id"] == 248044 and m["pad_token_id"] == 248044 and m["add_bos"] is False
    assert m["context_length"] == 262144
    # the original file is untouched and still a valid (version 1) container
    assert "blobs" not in qv_blobs.read_header(src)[0]


def test_list_extract(st, tmp_path, capsys):
    _, out = st
    qv_embed.cmd_list(out)
    txt = capsys.readouterr().out
    assert "tokenizer.json" in txt and "chat_template.jinja" in txt and "unit-test" in txt
    p = tmp_path / "t.json"
    qv_embed.cmd_extract(out, "tokenizer.json", str(p))
    assert p.read_bytes() == open(TOKJSON, "rb").read()
    with pytest.raises(SystemExit):
        qv_embed.cmd_extract(out, "nope", None)


def test_replace_drop_keep(st, tmp_path):
    _, out = st
    out2 = str(tmp_path / "o2.qv")
    qv_embed.embed(out, out2, {"chat_template.jinja": b"NEW"}, {"base_model": "x/y"}, drop=["tokenizer_config.json"], log=lambda *a: None)
    idx, off, _ = qv_blobs.read_header(out2)
    assert set(idx["blobs"]) == {"tokenizer.json", "chat_template.jinja", "generation_config.json"}
    assert idx["meta"]["base_model"] == "x/y" and idx["meta"]["model_name"] == "unit-test"
    assert tensor_hash(out) == tensor_hash(out2)
    qv_embed.cmd_extract(out2, "chat_template.jinja", str(tmp_path / "c.jinja"))
    assert (tmp_path / "c.jinja").read_bytes() == b"NEW"
    assert qv_blobs.sha256_file_range(out2, off + idx["blobs"]["tokenizer.json"]["off"], idx["blobs"]["tokenizer.json"]["nbytes"]) == idx["blobs"]["tokenizer.json"]["sha256"]


def test_in_place_and_failure_leaves_original(files, tmp_path):
    p = make_qv(str(tmp_path / "ip.qv"))
    before = tensor_hash(p)
    qv_embed.embed(p, p, {"chat_template.jinja": TINY_TEMPLATE.encode()}, None, log=lambda *a: None)
    assert tensor_hash(p) == before and "chat_template.jinja" in qv_blobs.read_header(p)[0]["blobs"]
    assert not os.path.exists(p + ".embed.tmp")
    # a truncated source is refused and neither the source nor a temp file is left changed
    t = str(tmp_path / "trunc.qv")
    open(t, "wb").write(open(p, "rb").read()[: (2 << 20) + 8])
    with pytest.raises(ValueError):
        qv_embed.embed(t, t, {"x": b"y"}, None, log=lambda *a: None)
    assert not os.path.exists(t + ".embed.tmp")


def test_disk_check(tmp_path, monkeypatch):
    p = make_qv(str(tmp_path / "d.qv"))
    import shutil, collections
    monkeypatch.setattr(shutil, "disk_usage", lambda _: collections.namedtuple("u", "total used free")(10, 10, 1000))
    with pytest.raises(OSError, match="not enough disk space"):
        qv_embed.embed(p, str(tmp_path / "e.qv"), {"x": b"y"}, None, log=lambda *a: None)


def test_pack_with_blobs(tmp_path):
    """qv_pack.pack (used by tools/convert.py --emit-qv) writes the same schema."""
    import qv_pack
    pre = str(tmp_path / "m")
    idx = {"format": "qvolta-v1", "config": {"arch": "qwen3_5"}, "tensors": {"t": {"kind": "f32", "shape": [8], "off": 0, "nbytes": 32}}}
    json.dump(idx, open(pre + ".qv.json", "w")); open(pre + ".qv.bin", "wb").write(b"\x07" * 32)
    qv_pack.pack(pre, pre + ".qv", blobs={"chat_template.jinja": b"hello"}, meta={"model_name": "p"})
    i, off, _ = qv_blobs.read_header(pre + ".qv")
    e = i["blobs"]["chat_template.jinja"]
    assert e["off"] == 64 and open(pre + ".qv", "rb").read()[off + 64: off + 69] == b"hello" and i["meta"]["model_name"] == "p"


# ------------------------------------------------------------------------------------------------ part 2: runtime
def run_tok(model, mode, lines):
    return subprocess.run([QV_TOK, model, mode], input="\n".join(json.dumps(x) for x in lines) + "\n", capture_output=True, text=True, check=True).stdout


def test_qv_tok_from_container(st):
    _, out = st
    texts = ["Hello, world!", "<|im_start|>user\nпривет 世界<|im_end|>", "def f(x):\n\treturn x**2"]
    assert run_tok(out, "encode", texts) == run_tok(TOKJSON, "encode", texts)
    r = subprocess.run([QV_TOK, st[0], "encode"], input='"x"\n', capture_output=True, text=True)
    assert r.returncode != 0 and "no embedded tokenizer" in r.stderr


def srv_cmd(model, *extra):
    return [QV_SERVER, "--backend", "mock", "--model", model, "--port", "@PORT@", "-c", "4096"] + list(extra)


def test_server_no_tokenizer_flag(st):
    _, out = st
    ref = Srv([QV_SERVER, "--backend", "mock", "--tokenizer", TOKJSON, "--port", "@PORT@", "-c", "4096"], name="ref")
    s = Srv(srv_cmd(out, "--mock-text", "hi there"), name="emb")
    try:
        texts = ["Hello, world!", "<|im_start|>user\nпривет 世界<|im_end|>", "tab\tsep  two spaces"]
        for t in texts:
            a = ref.req("POST", "/tokenize", {"content": t})[2]
            b = s.req("POST", "/tokenize", {"content": t})[2]
            assert a["tokens"] == b["tokens"] and len(b["tokens"]) > 0
        # chat completion works with the embedded tokenizer; the embedded template is what renders the prompt
        st_, _, body = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hey"}], "max_tokens": 16})
        assert st_ == 200 and body["choices"][0]["message"]["content"] == "hi there"
        assert body["choices"][0]["finish_reason"] == "stop"
        st_, _, body = s.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "hey"}]})
        assert body["prompt"] == "[user]hey\n[assistant]"
        props = s.req("GET", "/props")[2]
        assert props["chat_template"] == TINY_TEMPLATE and props["qv_chat_template_source"] == "embedded" and props["qv_tokenizer"] == "embedded"
        assert set(props["qv_embedded_blobs"]) >= {"tokenizer.json", "chat_template.jinja", "generation_config.json"}
        mj = s.req("GET", "/v1/models")[2]["data"][0]
        assert mj["qv"]["meta"]["model_name"] == "unit-test" and mj["qv"]["tokenizer"] == "embedded" and mj["meta"]["n_ctx_train"] == 262144
        # sampling defaults come from generation_config (no CLI flag given), eos ids from the same file
        p = props["default_generation_settings"]["params"]
        assert (p["temperature"], p["top_k"], p["top_p"]) == pytest.approx((0.6, 7, 0.9), rel=1e-6)
        # the llama-server defaults stay where the model file has no opinion (min_p)
        assert p["min_p"] == pytest.approx(0.05)
    finally:
        s.stop(); ref.stop()


def test_cli_wins_and_opt_out(st):
    _, out = st
    s = Srv(srv_cmd(out, "--temp", "0.2", "--top-k", "3", "--chat-template", "X{{ messages[0]['content'] }}", "--jinja"), name="cli")
    try:
        props = s.req("GET", "/props")[2]
        p = props["default_generation_settings"]["params"]
        assert (p["temperature"], p["top_k"], p["top_p"]) == pytest.approx((0.2, 3, 0.9), rel=1e-6)  # explicit flags win, top_p still from the file
        assert props["qv_chat_template_source"] == "override"
        assert s.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "q"}]})[2]["prompt"] == "Xq"
    finally:
        s.stop()
    s = Srv(srv_cmd(out, "--no-embedded-defaults"), name="optout")
    try:
        p = s.req("GET", "/props")[2]["default_generation_settings"]["params"]
        assert (p["temperature"], p["top_k"], p["top_p"]) == pytest.approx((0.8, 40, 0.95), rel=1e-6)  # llama-server defaults
    finally:
        s.stop()


def test_explicit_tokenizer_overrides(st, files):
    _, out = st
    s = Srv(srv_cmd(out, "--tokenizer", TOKJSON), name="ovr")
    try:
        props = s.req("GET", "/props")[2]
        assert props["qv_tokenizer"] == TOKJSON
        assert s.req("POST", "/tokenize", {"content": "abc"})[2]["tokens"]
    finally:
        s.stop()


def test_old_file_needs_tokenizer(st, tmp_path):
    src, _ = st
    r = subprocess.run([QV_SERVER, "--backend", "mock", "--model", src, "--port", "0"], capture_output=True, text=True, timeout=30,
                       cwd=str(tmp_path))
    assert r.returncode != 0 and "no tokenizer" in r.stderr and "--tokenizer" in r.stderr
    s = Srv(srv_cmd(src, "--tokenizer", TOKJSON), name="old")  # back-compat: version-1 file + side tokenizer
    try:
        props = s.req("GET", "/props")[2]
        assert "qv_embedded_blobs" not in props  # plain files keep the llama-server schema
        assert "qv" not in s.req("GET", "/v1/models")[2]["data"][0]
    finally:
        s.stop()
