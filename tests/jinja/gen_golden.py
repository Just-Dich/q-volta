#!/usr/bin/env python3
"""Regenerate tests/jinja/golden/*.json (expected renders from Python jinja2) for tests/jinja/templates/*.jinja."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import diff_oracle as d

here = os.path.dirname(os.path.abspath(__file__))
env = d.make_env()
for f in sorted(glob.glob(os.path.join(here, "templates", "*.jinja"))):
    tpl = env.from_string(open(f, encoding="utf-8").read())
    out = []
    for name, ctx in d.cases():
        r = d.render_py(tpl, ctx)
        e = {"name": name, "ctx": ctx}
        if r[0] == "ok": e["out"] = r[1]
        else: e["error_kind"] = r[1]; e["error"] = r[2] if r[1] == "raise" else ""
        out.append(e)
    p = os.path.join(here, "golden", os.path.basename(f)[:-6] + ".json")
    json.dump({"now": d.NOW, "cases": out}, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(p, os.path.getsize(p))
