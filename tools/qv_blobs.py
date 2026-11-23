#!/usr/bin/env python3
"""Shared helpers for the optional blobs / meta section of the .qv container (docs/QV_FORMAT.md).

A blob is a named byte string stored in the data area after the last tensor, 64-byte aligned; the JSON index lists
  "blobs": {name: {"off": <relative to data start>, "nbytes": n, "sha256": hex}}   and   "meta": {...}   and   "format_version": 2.
Readers that do not know the keys ignore them, so a v2 file loads in an old runtime (it just does not see the blobs).
"""
import hashlib, json, os, subprocess

BLOB_ALIGN = 64
DATA_ALIGN = 2 << 20
FORMAT_VERSION = 2
# files of an HF model directory that become blobs (name in the container == file name)
HF_FILES = ("tokenizer.json", "chat_template.jinja", "generation_config.json", "special_tokens_map.json")
# tokenizer_config.json is stored as a subset (the full file carries the added_tokens_decoder table, already in tokenizer.json)
TOKCFG_KEYS = ("tokenizer_class", "bos_token", "eos_token", "pad_token", "unk_token", "add_bos_token", "add_eos_token",
               "model_max_length", "clean_up_tokenization_spaces", "split_special_tokens")


def align(x, a):
    return (x + a - 1) // a * a


def tensor_end(index):
    """First byte (relative to the data section) after the last tensor byte."""
    end = 0
    for t in index["tensors"].values():
        if "off" in t:
            end = max(end, t["off"] + t["nbytes"])
        for o, n in (("off_q", "nbytes_q"), ("off_sm", "nbytes_sm"), ("off_rs", "nbytes_rs")):
            if o in t:
                end = max(end, t[o] + t[n])
    return end


def read_header(path):
    """-> (index dict, data_off, json_len). Raises ValueError for a file that is not a .qv container."""
    import struct
    with open(path, "rb") as f:
        if f.read(8) != b"QVOLTA1\0":
            raise ValueError(f"{path}: not a .qv container")
        jl, data_off = struct.unpack("<QQ", f.read(16))
        index = json.loads(f.read(jl))
    return index, data_off, jl


def dump_index(index):
    return json.dumps(index, separators=(",", ":")).encode()


def sha256_file_range(path, start, nbytes, chunk=32 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = nbytes
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise IOError(f"{path}: short read")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def git_hash(path=None):
    try:
        return subprocess.check_output(["git", "-C", path or os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _token_id(tok, s):
    """id of the token whose text is s, from the added_tokens list or the BPE vocab of a parsed tokenizer.json."""
    if s is None:
        return None
    if isinstance(s, dict):
        s = s.get("content")
    for a in tok.get("added_tokens", []):
        if a.get("content") == s:
            return a["id"]
    return tok.get("model", {}).get("vocab", {}).get(s)


def tokenizer_config_subset(raw):
    cfg = json.loads(raw)
    out = {}
    for k in TOKCFG_KEYS:
        if k in cfg:
            v = cfg[k]
            out[k] = v.get("content") if isinstance(v, dict) and "content" in v else v
    return out


def derive_meta(blobs, hf_config=None):
    """Token-id metadata that runtimes read without parsing the blobs: bos/eos/pad ids, add_bos. blobs: name -> bytes."""
    m = {}
    gen = json.loads(blobs["generation_config.json"]) if "generation_config.json" in blobs else {}
    tcfg = json.loads(blobs["tokenizer_config.json"]) if "tokenizer_config.json" in blobs else {}
    smap = json.loads(blobs["special_tokens_map.json"]) if "special_tokens_map.json" in blobs else {}
    tok = json.loads(blobs["tokenizer.json"]) if "tokenizer.json" in blobs else {}
    eos = gen.get("eos_token_id")
    if eos is None:
        eos = _token_id(tok, smap.get("eos_token", tcfg.get("eos_token")))
    if eos is not None:
        m["eos_token_ids"] = list(eos) if isinstance(eos, list) else [eos]
    for key, src in (("bos_token_id", "bos_token"), ("pad_token_id", "pad_token")):
        v = gen.get(key)
        if v is None:
            v = _token_id(tok, smap.get(src, tcfg.get(src)))
        if v is not None:
            m[key] = v
    if "add_bos_token" in tcfg:
        m["add_bos"] = bool(tcfg["add_bos_token"])
    elif "bos_token_id" in m:
        m["add_bos"] = False
    ctx = tcfg.get("model_max_length")
    if hf_config:
        tc = hf_config.get("text_config", hf_config)
        ctx = tc.get("max_position_embeddings", ctx)
    if isinstance(ctx, int) and ctx < 1 << 40:
        m["context_length"] = ctx
    return m


def parse_front_matter(readme_text):
    """license / base_model from a HF README.md YAML front matter (flat scalar keys only)."""
    out = {}
    if readme_text.startswith("---"):
        for ln in readme_text.split("---", 2)[1].splitlines():
            if ":" in ln and not ln.startswith((" ", "-")):
                k, v = ln.split(":", 1)
                if k.strip() in ("license", "base_model") and v.strip():
                    out[k.strip()] = v.strip()
    return out


def collect_hf_dir(d, include_readme_meta=True):
    """Blobs + derived meta from a HF model directory. Returns (blobs: name -> bytes, meta: dict). Missing files are skipped."""
    blobs = {}
    for n in HF_FILES:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            blobs[n] = open(p, "rb").read()
    p = os.path.join(d, "tokenizer_config.json")
    if os.path.isfile(p):
        blobs["tokenizer_config.json"] = json.dumps(tokenizer_config_subset(open(p, "rb").read()), indent=1).encode()
    hf_config = None
    if os.path.isfile(os.path.join(d, "config.json")):
        hf_config = json.load(open(os.path.join(d, "config.json")))
    meta = derive_meta(blobs, hf_config)
    if include_readme_meta and os.path.isfile(os.path.join(d, "README.md")):
        meta.update(parse_front_matter(open(os.path.join(d, "README.md"), encoding="utf-8", errors="replace").read()))
    return blobs, meta


def parse_kv(items):
    """--meta k=v: v is parsed as JSON when possible (numbers, true/false, lists), else kept as a string."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def describe_quant(index):
    """Auto recipe summary from the tensor table, e.g. "q 3b x358, 4b x1, 5b x1; groups 128; meta8; gptq clip" (what is in the file, not what was intended)."""
    import collections
    bits, groups, meta8 = collections.Counter(), set(), False
    for t in index["tensors"].values():
        if t.get("kind") == "q":
            bits[t["bits"]] += 1
            groups.add(t["group"])
            meta8 |= t.get("meta") == "s8z8"
        elif t.get("kind") in ("nvfp4", "mxfp8"):
            bits[t["kind"]] += 1
    if not bits:
        return None
    parts = [f"{b}b x{n}" if isinstance(b, int) else f"{b} x{n}" for b, n in sorted(bits.items(), key=lambda kv: str(kv[0]))]
    out = "q " + ", ".join(parts) + "; groups " + "/".join(map(str, sorted(groups))) + ("; meta8" if meta8 else "")
    cq = index.get("config", {}).get("quant")
    if isinstance(cq, dict):
        out += "; " + " ".join(f"{k}={v}" for k, v in cq.items() if k in ("method", "profile", "clip") and v is not None)
    return out


def base_meta(index, extra=None):
    """Meta fields every writer sets: arch, qv_format_version, converter_git (+ context length from the config when known)."""
    cfg = index.get("config", {})
    m = {"arch": cfg.get("arch", "qwen3_5"), "qv_format_version": FORMAT_VERSION}
    g = git_hash()
    if g:
        m["converter_git"] = g
    ink = cfg.get("inkling")
    if ink and ink.get("max_position"):
        m["context_length"] = ink["max_position"]
    elif isinstance(cfg.get("max_position_embeddings"), int):
        m["context_length"] = cfg["max_position_embeddings"]
    q = describe_quant(index)
    if q:
        m["quantization"] = q
    m.update(extra or {})
    return m


def merge_index(index, blob_entries, meta):
    """Return a copy of the index with format_version, blobs and meta set (existing meta keys are overridden by `meta`)."""
    idx = dict(index)
    idx["format_version"] = FORMAT_VERSION
    idx["blobs"] = blob_entries
    m = dict(index.get("meta", {}))
    m.update(meta)
    m["qv_format_version"] = FORMAT_VERSION
    idx["meta"] = m
    return idx


def layout(start, sizes):
    """Assign 64-byte aligned offsets (relative to the data section) from `start` to blobs {name: nbytes} -> ({name: off}, end)."""
    offs, pos = {}, start
    for n, sz in sizes.items():
        pos = align(pos, BLOB_ALIGN)
        offs[n] = pos
        pos += sz
    return offs, pos


def append_blobs(f, data_off, tend, blobs):
    """Write blobs (name -> bytes) right after the tensor area of an open container file; returns the index "blobs" dict."""
    offs, _ = layout(tend, {n: len(b) for n, b in blobs.items()})
    entries = {}
    for n, b in blobs.items():
        f.seek(data_off + offs[n])
        f.write(b)
        entries[n] = {"off": offs[n], "nbytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return entries
