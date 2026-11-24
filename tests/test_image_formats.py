#!/usr/bin/env python3
"""Image format decoding (WebP via vendored libwebp, stb formats) and image error messages.
  QV_BUILD=build QV_SERVER=build/qv_server python3 -I -m pytest -q tests/test_image_formats.py
Pixels are compared with Pillow (WebP lossy/lossless decoding is deterministic by spec) through the qv_image_decode tool; error texts
are checked through the tool and through the server (mock vision encoder)."""
import base64, io, os, subprocess, sys, tempfile
import pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QV_SERVER", os.path.join(ROOT, "build", "qv_server"))
BUILD = os.environ.get("QV_BUILD") or os.path.dirname(os.environ["QV_SERVER"])
TOOL = os.path.join(BUILD, "qv_image_decode")
PIL = pytest.importorskip("PIL")
from PIL import Image
import test_server_vision as V
import test_server_compat as T

if not os.path.exists(TOOL): pytest.skip("qv_image_decode not built", allow_module_level=True)


def pattern(w, h, mode):
    im = Image.new("RGBA", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 5 + y) & 255, (x * 2 + y * 7) & 255, ((x * y) >> 2) & 255, 255 if mode != "RGBA" else (x * 9 + y * 3) & 255)
    if mode == "L": return im.convert("L")
    if mode == "RGB": return im.convert("RGB")
    return im


def run(data, max_side=None):
    d = tempfile.mkdtemp(prefix="qvfmt_")
    open(d + "/in", "wb").write(data)
    cmd = [TOOL, d + "/in", d + "/out"] + ([str(max_side)] if max_side else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: return None, r.stderr.strip()
    w, h = map(int, r.stdout.split())
    return (w, h, open(d + "/out", "rb").read()), None


def expected_rgb(im):
    """Pillow RGBA/RGB -> RGB with the HF checkerboard composite (same integer formula as image.cpp)."""
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGBA"); w, h = im.size; px = im.load(); out = bytearray()
        for y in range(h):
            for x in range(w):
                r, g, b, a = px[x, y]; bg = 204 if ((y // 8) + (x // 8)) % 2 == 0 else 153
                out += bytes((c * a + bg * (255 - a) + 127) // 255 for c in (r, g, b))
        return bytes(out)
    return im.convert("RGB").tobytes()


def enc(im, **kw):
    b = io.BytesIO(); im.save(b, **kw); return b.getvalue()


SIZES = [(1, 1), (16, 16), (40, 40), (33, 47), (80, 41), (100, 60), (17, 93), (129, 130)]


@pytest.mark.parametrize("w,h", SIZES)
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
@pytest.mark.parametrize("kind", ["lossy", "lossless"])
def test_webp_exact(w, h, mode, kind):
    src = pattern(w, h, mode)
    data = enc(src, format="WEBP", lossless=(kind == "lossless"), quality=80, method=2)
    got, err = run(data)
    assert err is None, err
    ref = Image.open(io.BytesIO(data)); ref.load()
    assert (got[0], got[1]) == (w, h)
    assert got[2] == expected_rgb(ref), f"{kind} {mode} {w}x{h} differs from Pillow"
    if kind == "lossless": assert ref.convert("RGBA").tobytes() == src.convert("RGBA").tobytes() or mode == "RGBA"  # sanity of the fixture


def test_webp_lossy_alpha_and_exact_lossless_roundtrip():
    src = pattern(50, 37, "RGBA")
    for lossless in (False, True):
        data = enc(src, format="WEBP", lossless=lossless, quality=90, exact=True)
        got, err = run(data); assert err is None, err
        ref = Image.open(io.BytesIO(data)); ref.load()
        assert got[2] == expected_rgb(ref)
    assert got[2] == expected_rgb(src)  # lossless + exact: bit-identical to the source


def test_webp_animated_first_frame():
    frames = [pattern(40, 24, "RGB"), pattern(40, 24, "RGB").transpose(Image.FLIP_LEFT_RIGHT)]
    data = enc(frames[0], format="WEBP", save_all=True, append_images=frames[1:], lossless=True, duration=100)
    got, err = run(data); assert err is None, err
    ref = Image.open(io.BytesIO(data)); ref.seek(0); ref.load()
    assert got[2] == expected_rgb(ref) and got[2] == expected_rgb(frames[0])


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "BMP", "TIFF_ABSENT"])
def test_other_formats_still_work(fmt):
    if fmt == "TIFF_ABSENT":
        got, err = run(enc(pattern(20, 20, "RGB"), format="TIFF"))
        assert got is None and "unsupported image format 'TIFF'" in err
        return
    got, err = run(enc(pattern(37, 21, "RGB"), format=fmt))
    assert err is None and got[:2] == (37, 21)


def test_webp_errors():
    data = enc(pattern(64, 64, "RGB"), format="WEBP", lossless=False)
    got, err = run(data[:len(data) // 2])
    assert got is None and "WebP" in err and ("truncated" in err or "corrupt" in err)
    got, err = run(data[:30])
    assert got is None and "WebP" in err
    got, err = run(data, max_side=32)
    assert got is None and "at most 32" in err
    bad = bytearray(data); bad[12:16] = b"VP8Z"
    got, err = run(bytes(bad)); assert got is None and "WebP" in err


def ftyp(brand): return b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x00\x00" + brand + b"mif1" + b"\0" * 32


@pytest.mark.parametrize("data,msg", [
    (ftyp(b"heic"), "unsupported image format 'HEIC' (supported: PNG, JPEG, GIF, BMP, WebP, PNM, TGA); convert the image first"),
    (ftyp(b"avif"), "unsupported image format 'AVIF' (supported: PNG, JPEG, GIF, BMP, WebP, PNM, TGA); convert the image first"),
    (b"%PDF-1.7\n....", "unsupported image format 'PDF'"),
    (b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', "unsupported image format 'SVG'"),
    (b"<!DOCTYPE html><html><body>404 Not Found</body></html>", "image data is not an image (starts with '<!DOCTYPE html><html><body>404 Not Foun"),
    (b'{"error": {"message": "unauthorized"}}', "image data is not an image (starts with '{\"error\": {\"message\": \"unauthorized\"}}')"),
    (b"\x00\x01\x02\x03\x04\x05", "unrecognised or corrupt image data (starts with bytes 00 01 02 03 04 05"),
])
def test_error_texts_tool(data, msg):
    got, err = run(data)
    assert got is None and msg in err, err


@pytest.fixture(scope="module")
def srv():
    f = tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False); f.write(V.TEMPLATE); f.close()
    s = T.Srv(T.qv_cmd("--vision-file", "mock", "--image-token", "<|image_pad|>", "--chat-template-file", f.name, "--mock-text", "ok"), name="fmt")
    yield s
    s.stop()


def post(srv, url):
    st, _, b = srv.req("POST", "/v1/chat/completions", V.msg(url))
    return st, b


def test_server_webp_ok(srv):
    for kw in (dict(lossless=True), dict(lossless=False, quality=75)):
        st, b = post(srv, V.durl(enc(pattern(100, 60, "RGBA"), format="WEBP", **kw), "image/webp"))
        assert st == 200, b
    st2, b2 = post(srv, V.durl(V.png(40, 40)))
    st, b = post(srv, V.durl(enc(pattern(100, 60, "RGB"), format="WEBP"), "image/webp"))
    assert b["usage"]["prompt_tokens"] - b2["usage"]["prompt_tokens"] == 5  # 100x60 -> 6 rows vs 1


def test_server_errors(srv):
    st, b = post(srv, V.durl(ftyp(b"heic"), "image/heic"))
    assert st == 400 and b["error"]["type"] == "invalid_request_error"
    assert b["error"]["message"] == "image: unsupported image format 'HEIC' (supported: PNG, JPEG, GIF, BMP, WebP, PNM, TGA); convert the image first"
    st, b = post(srv, V.durl(b"<html>Bad gateway</html>"))
    assert st == 400 and b["error"]["message"].startswith("image: image data is not an image (starts with '<html>Bad gateway</html>')")
    st, b = post(srv, V.durl(enc(pattern(20, 20, "RGB"), format="TIFF"), "image/tiff"))
    assert st == 400 and "unsupported image format 'TIFF'" in b["error"]["message"]
    st, b = post(srv, "data:image/png;base64,@@@@")
    assert st == 400 and "invalid base64" in b["error"]["message"] and "0x40" in b["error"]["message"]
    st, b = post(srv, "data:image/png;base64,iVBOR")
    assert st == 400 and b["error"]["message"].startswith("image:") and "truncated" in b["error"]["message"]
    st, b = post(srv, "data:image/png;base64,")
    assert st == 400 and "empty base64" in b["error"]["message"]
