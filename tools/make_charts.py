#!/usr/bin/env python3
"""Render README bar charts (static SVG) from benchmarks/results/*.json. No third-party dependencies.
Only measured numbers from the JSON files are drawn; missing entries are skipped."""
import json, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
COL = {"Q-Volta": "#2563eb", "Q-Volta + MTP": "#16a34a", "NInfer": "#9ca3af", "NInfer + MTP": "#6b7280",
       "llama.cpp": "#9ca3af", "llama.cpp Q4_K_M": "#9ca3af"}

def chart(path, title, groups, unit, note=""):
    """groups: [(group label, [(series, value)])]"""
    W, H, L, T, B = 720, 360, 70, 56, 70
    vmax = max(v for _, s in groups for _, v in s) * 1.15
    n = sum(len(s) for _, s in groups)
    gw = (W - L - 20) / len(groups)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#fff"/>',
         f'<text x="{L}" y="26" font-size="15" font-weight="600" fill="#111">{title}</text>',
         f'<text x="{L}" y="44" fill="#555">{unit}{(" - " + note) if note else ""}</text>']
    for i in range(5):
        y = T + (H - T - B) * (1 - i / 4)
        o.append(f'<line x1="{L}" x2="{W-10}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{vmax*i/4:.0f}</text>')
    seen = []
    for gi, (g, series) in enumerate(groups):
        bw = min(46, (gw - 16) / len(series))
        x0 = L + gi * gw + (gw - bw * len(series)) / 2
        for si, (s, v) in enumerate(series):
            h = (H - T - B) * v / vmax
            x = x0 + si * bw
            o.append(f'<rect x="{x:.1f}" y="{H-B-h:.1f}" width="{bw-4:.1f}" height="{h:.1f}" rx="2" fill="{COL.get(s, "#f59e0b")}"/>')
            o.append(f'<text x="{x+(bw-4)/2:.1f}" y="{H-B-h-5:.1f}" text-anchor="middle" fill="#111">{v:.1f}</text>')
            if s not in seen: seen.append(s)
        o.append(f'<text x="{L+gi*gw+gw/2:.1f}" y="{H-B+18}" text-anchor="middle" fill="#333">{g}</text>')
    lx = L
    for s in seen:
        o.append(f'<rect x="{lx}" y="{H-24}" width="12" height="12" rx="2" fill="{COL.get(s, "#f59e0b")}"/><text x="{lx+17}" y="{H-14}" fill="#333">{s}</text>')
        lx += 30 + 7 * len(s)
    o.append("</svg>")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, path), "w").write("\n".join(o))
    print("wrote", path)

def load(pat):
    f = sorted(glob.glob(os.path.join(ROOT, "benchmarks", "results", pat)))
    return json.load(open(f[-1])) if f else None

cpu = load("cpu_*.json")
if cpu:
    q, l = cpu["qvolta"], cpu["llamacpp"]
    chart("cpu_prefill.svg", "CPU prefill (Xeon E5-2683 v4, 32 threads)",
          [(f"{n} tokens", [("Q-Volta", q["prefill_tok_s"][n]), ("llama.cpp", l["Q4_K_M"]["pp" + n])]) for n in ("512", "2048")],
          "tokens/s, higher is better", "llama.cpp Q4_K_M; shared host, single runs")
    chart("cpu_decode.svg", "CPU decode (greedy, 32 threads)",
          [(f"ctx {m['ctx']}", [("Q-Volta", m["plain"]), ("Q-Volta + MTP", m["mtp"])]) for m in q["mtp"]] +
          [("llama.cpp", [("llama.cpp", l["Q4_K_M"]["decode"])])],
          "tokens/s, higher is better", "MTP output identical to plain greedy")

gpu = load("gpu_*.json")
if gpu:
    S = gpu["summary"]
    ref = gpu["ninfer_reference_not_rerun_today"]
    rows = []
    for key, label in (("code", "code chat (ctx 37)"), ("prose", "prose chat (ctx 47)"), ("P512", "wikitext ctx 512"),
                       ("P2048", "wikitext ctx 2048"), ("P4088", "wikitext ctx 4088")):
        if key + "/plain" not in S: continue
        best = max((S[f"{key}/mtp{d}"]["mean"] for d in (1, 2, 3) if f"{key}/mtp{d}" in S))
        ser = [("Q-Volta", S[key + "/plain"]["mean"]), ("Q-Volta + MTP", best)]
        if key in ref["plain_tok_s"]:
            ser = [("NInfer", ref["plain_tok_s"][key])] + ser + [("NInfer + MTP", ref["mtp3_tok_s"][key])]
        rows.append((label, ser))
    chart("gpu_decode.svg", "GPU decode (Tesla V100-16GB, greedy)", rows, "tokens/s, higher is better",
          "MTP = best depth 1-3, output identical to plain")
    pr = gpu.get("ninfer_prefill_reference_not_rerun_today")
    base = {}
    for r in gpu["baseline_plain_pre_mtp_build"]["runs"]:
        base.setdefault(str(r["prompt_tokens"]), []).append(r["prefill_tok_s"])
    if pr and base:
        chart("gpu_prefill.svg", "GPU prefill (Tesla V100-16GB, cold)",
              [(f"{n} tokens", [("NInfer", pr["tok_s"][n]), ("Q-Volta", sum(base[n]) / len(base[n]))]) for n in ("512", "2048", "4088") if n in base],
              "tokens/s, higher is better")
