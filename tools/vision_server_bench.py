#!/usr/bin/env python3 -I
"""Start qv_server (CUDA, Inkling + vision sidecar), run image requests, print/record results. Run under `flock /tmp/qv_bench.lock`.
usage: vision_server_bench.py BUILD/qv_server MODEL.qv TOKENIZER.json OUT.json [--backend cuda|cpu]"""
import base64, http.client, json, os, signal, socket, struct, subprocess, sys, time, zlib
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "inkling_vision")


def post(port, body, timeout=1800):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    b = json.loads(r.read())
    return r.status, b, time.time() - t0


def synth_png(w, h):
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    rows = b"".join(b"\0" + bytes(((x * 3 + y * 5 + c * 40) // 4 + (x * y) % 17) & 255 for x in range(w) for c in range(3)) for y in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 1)) + chunk(b"IEND", b"")


def main():
    binp, model, tok, out = sys.argv[1:5]
    backend = sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv else "cuda"
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    log = open("/tmp/vision_srv_%s.log" % backend, "w")
    p = subprocess.Popen([binp, "--model", model, "--tokenizer", tok, "--backend", backend, "--kv", "int4", "--max-ctx", "16384", "--port", str(port), "--props", "--image-root", G],
                         stdout=log, stderr=log)
    res = {"backend": backend}
    try:
        for _ in range(900):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2); c.request("GET", "/health")
                if c.getresponse().status == 200: break
            except OSError: pass
            time.sleep(1)
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5); c.request("GET", "/props")
        res["props_vision"] = json.loads(c.getresponse().read())["modalities"]["vision"]

        def ask(img, q, n):
            url = "data:image/png;base64," + base64.b64encode(open(f"{G}/{img}", "rb").read()).decode()
            return post(port, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}],
                               "max_tokens": n, "temperature": 0})
        ask("redsquare_224x224.png", "hi", 1)  # warm-up
        for img, q in [("redsquare_224x224.png", "What shape and colors do you see in this image? Answer briefly."),
                       ("hello_320x120.png", "What text is written in this image?"),
                       ("gradient_200x120.png", "Describe this image."),
                       ("redsquare_224x224.png", "Describe this image in one sentence.")]:
            st, b, dt = ask(img, q, 320 if backend == "cuda" else 48)
            ch = b["choices"][0]["message"]
            res.setdefault("answers", []).append(dict(image=img, question=q, status=st, content=ch.get("content"), reasoning=ch.get("reasoning_content"),
                                                      usage=b.get("usage"), timings=b.get("timings"), wall_s=round(dt, 2)))
            print(json.dumps(res["answers"][-1], ensure_ascii=False)[:1800], flush=True)
        sizes = [(224, 224), (512, 512), (1024, 1024)] if backend == "cuda" else [(224, 224), (512, 512)]
        for w, h in sizes:
            url = "data:image/png;base64," + base64.b64encode(synth_png(w, h)).decode()
            ts = []
            for rep in range(3):
                st, b, dt = post(port, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": f"Describe. {rep}"}]}],
                                        "max_tokens": 1, "temperature": 0, "cache_prompt": False})
                ts.append(dict(wall_s=round(dt, 3), prompt_tokens=b["usage"]["prompt_tokens"], prompt_ms=(b.get("timings") or {}).get("prompt_ms")))
            res.setdefault("ttft", {})[f"{w}x{h}"] = ts
            print(w, h, ts, flush=True)
        st, b, dt = post(port, {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 1, "cache_prompt": False})
        res["text_only_ttft"] = dict(wall_s=round(dt, 3), prompt_tokens=b["usage"]["prompt_tokens"], prompt_ms=(b.get("timings") or {}).get("prompt_ms"))
    finally:
        p.send_signal(signal.SIGTERM)
        try: p.wait(20)
        except Exception: p.kill()
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


main()
