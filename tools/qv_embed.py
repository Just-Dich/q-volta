#!/usr/bin/env python3
"""Embed (or list / extract) tokenizer, chat template, generation config and metadata in a .qv container WITHOUT re-quantizing.

  qv_embed.py model.qv --tokenizer tokenizer.json --chat-template chat_template.jinja --generation-config generation_config.json \\
              [--special-tokens-map f] [--tokenizer-config f] [--from-dir HF_DIR] [--meta k=v ...] [--drop NAME] -o out.qv
  qv_embed.py model.qv --from-dir /path/to/hf_model_dir -o out.qv      # picks up every known file of the directory
  qv_embed.py model.qv --from-dir DIR -o model.qv                       # in place: temp file in the same dir, verified, atomic rename
  qv_embed.py --list file.qv                                            # blobs + meta
  qv_embed.py --extract NAME file.qv [-o FILE]                          # dump one blob (stdout when no -o)

The tensor area is copied in chunks (never held in RAM) and hashed on the way; the output is re-read and verified (tensor-area
sha256, every blob sha256, index sanity) before it is renamed to its final name. Peak extra disk = one copy of the file
(checked with statvfs first). Format: docs/QV_FORMAT.md.
"""
import argparse, hashlib, json, os, shutil, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_blobs as B

CHUNK = 32 << 20
FILE_ARGS = (("tokenizer", "tokenizer.json"), ("chat_template", "chat_template.jinja"), ("generation_config", "generation_config.json"),
             ("special_tokens_map", "special_tokens_map.json"))


def _copy_hash(src, start, nbytes, dst):
    """Copy [start, start+nbytes) of the file object src to dst in chunks; returns the sha256 of the bytes."""
    h = hashlib.sha256()
    src.seek(start)
    left = nbytes
    buf = bytearray(CHUNK)
    mv = memoryview(buf)
    while left:
        n = src.readinto(mv[:min(CHUNK, left)])
        if not n:
            raise IOError("short read from source")
        h.update(mv[:n])
        dst.write(mv[:n])
        left -= n
    return h.hexdigest()


def embed(src, dst, new_blobs, meta=None, drop=(), verify=True, log=print):
    """Rewrite src -> dst with `new_blobs` (name -> bytes) added / replaced, `drop` names removed and `meta` merged.
    Tensor bytes are copied verbatim. dst may equal src (in place, via a temp file + atomic rename after verification)."""
    index, data_off, jl = B.read_header(src)
    size = os.path.getsize(src)
    tend = B.tensor_end(index)
    if data_off + tend > size:
        raise ValueError(f"{src}: truncated (tensors end at {data_off + tend}, file has {size} bytes)")
    old_blobs = index.get("blobs", {})
    keep = {n: e for n, e in old_blobs.items() if n not in new_blobs and n not in drop}
    for n, e in keep.items():
        if data_off + e["off"] + e["nbytes"] > size:
            raise ValueError(f"{src}: blob {n} extends past the end of the file")
    # layout: tensors unchanged, then kept blobs, then new blobs (64-byte aligned)
    sizes = {n: e["nbytes"] for n, e in keep.items()}
    sizes.update({n: len(b) for n, b in new_blobs.items()})
    offs, end = B.layout(tend, sizes)
    entries = {n: dict(off=offs[n], nbytes=sizes[n], sha256=keep[n].get("sha256")) if n in keep else
               dict(off=offs[n], nbytes=sizes[n], sha256=hashlib.sha256(new_blobs[n]).hexdigest()) for n in sizes}
    m = B.base_meta(index)
    g = m.pop("converter_git", None)  # here the hash is that of THIS tool; a converter-written converter_git (in the old meta) is kept
    if g:
        m["embed_tool_git"] = g
    m.update(index.get("meta", {}))
    m.update(meta or {})
    new_index = B.merge_index({k: v for k, v in index.items() if k not in ("blobs", "meta")}, entries, m)
    js = B.dump_index(new_index)
    hdr = 24 + len(js)
    new_data_off = data_off if hdr <= data_off else B.align(hdr, B.DATA_ALIGN)
    final = os.path.abspath(dst)
    tmp = final + ".embed.tmp"
    # ---- disk check: the new file is a full copy
    out_size = new_data_off + end
    free = shutil.disk_usage(os.path.dirname(final)).free
    if os.path.exists(final) and final != os.path.abspath(src):
        free += os.path.getsize(final)
    if free < out_size + (1 << 30):
        raise OSError(f"not enough disk space in {os.path.dirname(final)}: need {out_size / 2**30:.2f} GiB + 1 GiB slack, free {free / 2**30:.2f} GiB")
    log(f"writing {tmp}: {out_size / 2**30:.3f} GiB (tensor area {tend / 2**30:.3f} GiB unchanged, blobs: {', '.join(sizes) or 'none'}; free {free / 2**30:.1f} GiB)")
    try:
        with open(src, "rb") as fs, open(tmp, "wb") as fd:
            fd.write(b"QVOLTA1\0" + struct.pack("<QQ", len(js), new_data_off) + js)
            fd.write(b"\0" * (new_data_off - hdr))
            tensor_hash = _copy_hash(fs, data_off, tend, fd)
            for n in sizes:
                fd.write(b"\0" * (new_data_off + offs[n] - fd.tell()))
                if n in new_blobs:
                    fd.write(new_blobs[n])
                else:
                    h = _copy_hash(fs, data_off + old_blobs[n]["off"], old_blobs[n]["nbytes"], fd)
                    entries[n]["sha256"] = entries[n]["sha256"] or h
            assert fd.tell() == new_data_off + end
        if verify:
            verify_file(tmp, tensor_hash, tend, new_blobs, entries, log)
        os.replace(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log(f"wrote {final}: {os.path.getsize(final)} bytes; tensor-area sha256 {tensor_hash[:16]}... identical")
    return tensor_hash


def verify_file(path, tensor_hash, tend, new_blobs, entries, log=print):
    index, data_off, jl = B.read_header(path)
    if B.tensor_end(index) != tend:
        raise AssertionError("tensor area size changed")
    size = os.path.getsize(path)
    if data_off % B.DATA_ALIGN:
        raise AssertionError("data start is not 2 MiB aligned")
    got = B.sha256_file_range(path, data_off, tend)
    if got != tensor_hash:
        raise AssertionError(f"tensor area differs after the copy: {got} != {tensor_hash}")
    for n, e in index.get("blobs", {}).items():
        if e["off"] % B.BLOB_ALIGN or data_off + e["off"] + e["nbytes"] > size:
            raise AssertionError(f"bad blob entry {n}")
        h = B.sha256_file_range(path, data_off + e["off"], e["nbytes"])
        if h != e["sha256"] or (n in new_blobs and h != hashlib.sha256(new_blobs[n]).hexdigest()):
            raise AssertionError(f"blob {n} does not verify")
    log(f"verified {path}: tensor area + {len(index.get('blobs', {}))} blob(s) sha256 ok")


def blobs_from_args(a):
    blobs, meta = {}, {}
    if a.from_dir:
        blobs, meta = B.collect_hf_dir(a.from_dir)
    for attr, name in FILE_ARGS:
        p = getattr(a, attr)
        if p:
            blobs[name] = open(p, "rb").read()
    if a.tokenizer_config:
        blobs["tokenizer_config.json"] = json.dumps(B.tokenizer_config_subset(open(a.tokenizer_config, "rb").read()), indent=1).encode()
    if blobs:  # token ids are re-derived from what is actually embedded (explicit files win over --from-dir)
        derived = B.derive_meta(blobs)
        if a.from_dir and os.path.isfile(os.path.join(a.from_dir, "config.json")):
            derived.update({k: v for k, v in B.derive_meta(blobs, json.load(open(os.path.join(a.from_dir, "config.json")))).items() if k == "context_length"})
        meta.update(derived)
    meta.update(B.parse_kv(a.meta))
    return blobs, meta


def cmd_list(path):
    index, data_off, jl = B.read_header(path)
    print(f"{path}: {os.path.getsize(path)} bytes, format_version {index.get('format_version', 1)}, data_off {data_off}, "
          f"index {jl} bytes, {len(index['tensors'])} tensors, tensor area {B.tensor_end(index)} bytes")
    for n, e in index.get("blobs", {}).items():
        print(f"  blob {n:28s} off {data_off + e['off']:>14d}  {e['nbytes']:>10d} bytes  sha256 {e.get('sha256', '')[:16]}")
    if not index.get("blobs"):
        print("  (no blobs)")
    print("  meta " + json.dumps(index.get("meta", {}), indent=2).replace("\n", "\n  "))


def cmd_extract(path, name, out):
    index, data_off, _ = B.read_header(path)
    e = index.get("blobs", {}).get(name)
    if e is None:
        sys.exit(f"{path}: no blob named {name} (have: {', '.join(index.get('blobs', {})) or 'none'})")
    with open(path, "rb") as f:
        f.seek(data_off + e["off"])
        data = f.read(e["nbytes"])
    if out and out != "-":
        open(out, "wb").write(data)
        print(f"wrote {out} ({len(data)} bytes)", file=sys.stderr)
    else:
        sys.stdout.buffer.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qv", help="input .qv")
    ap.add_argument("-o", "--out", help="output .qv (same path as the input = in place via temp file + rename)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extract", metavar="NAME")
    ap.add_argument("--from-dir", help="HF model directory: tokenizer.json, chat_template.jinja, generation_config.json, special_tokens_map.json, "
                                       "tokenizer_config.json (subset), config.json / README.md (meta)")
    ap.add_argument("--tokenizer"); ap.add_argument("--chat-template"); ap.add_argument("--generation-config")
    ap.add_argument("--special-tokens-map"); ap.add_argument("--tokenizer-config")
    ap.add_argument("--meta", action="append", default=[], metavar="K=V", help="meta entry (V parsed as JSON when possible); repeatable")
    ap.add_argument("--drop", action="append", default=[], metavar="NAME", help="remove a blob")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    if a.list:
        return cmd_list(a.qv)
    if a.extract:
        return cmd_extract(a.qv, a.extract, a.out)
    blobs, meta = blobs_from_args(a)
    if not blobs and not meta and not a.drop:
        ap.error("nothing to embed: pass --tokenizer/--chat-template/--generation-config/--from-dir/--meta")
    if not a.out:
        ap.error("-o OUT is required (use the input path for in-place)")
    embed(a.qv, a.out, blobs, meta, a.drop, verify=not a.no_verify)


if __name__ == "__main__":
    main()
