#!/usr/bin/env python3
"""Drives the real web UI (the Q-Volta-themed llama.cpp web UI embedded in qv_server) in headless Chromium. Needs the `playwright` package and a Chromium build:

    python3 -m venv /tmp/pwenv && /tmp/pwenv/bin/pip install playwright && PLAYWRIGHT_BROWSERS_PATH=/tmp/pwbrowsers /tmp/pwenv/bin/playwright install chromium
    PLAYWRIGHT_BROWSERS_PATH=/tmp/pwbrowsers /tmp/pwenv/bin/python tests/webui_browser_check.py http://127.0.0.1:9931/ OUTDIR [--stop] [--tool] [--prompt TEXT] [--wait SECONDS]

Prints one JSON object: final page text, every API request with its HTTP status, page errors; saves screenshots webui_*.png into OUTDIR.
Called by tests/test_server_compat.py::test_webui_in_browser (skipped when playwright is absent).
"""
import json, sys
from playwright.sync_api import sync_playwright

url, outdir = sys.argv[1], sys.argv[2]
flags = sys.argv[3:]
prompt = flags[flags.index("--prompt") + 1] if "--prompt" in flags else "Say hi and give me a short list."
wait_s = float(flags[flags.index("--wait") + 1]) if "--wait" in flags else 6.0
res = {"requests": [], "page_errors": [], "console_errors": []}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 800})
    pg.on("response", lambda r: res["requests"].append([r.request.method, r.url.split("/", 3)[3], r.status]) if "/_app/" not in r.url and not r.url.endswith((".png", ".ico", ".svg", ".woff2")) else None)
    pg.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
    pg.on("console", lambda m: res["console_errors"].append(m.text[:200]) if m.type == "error" else None)
    pg.goto(url); pg.wait_for_timeout(2500)
    pg.screenshot(path=f"{outdir}/webui_home.png")
    pg.locator(".cm-content").first.click(); pg.keyboard.type(prompt); pg.keyboard.press("Enter")
    pg.wait_for_timeout(1800)
    pg.screenshot(path=f"{outdir}/webui_streaming.png")
    if "--stop" in flags:
        pg.locator("button").last.click()  # the stop button (square icon while processing the prompt, "Stop" while reasoning)
        pg.wait_for_timeout(1500)
    elif "--tool" in flags:
        pg.wait_for_timeout(3000)
        pg.get_by_role("button", name="Allow once").first.click()
        pg.wait_for_timeout(wait_s * 1000)
    else:
        pg.wait_for_timeout(wait_s * 1000)
    pg.screenshot(path=f"{outdir}/webui_chat.png")
    res["text"] = pg.inner_text("body")[:3000]
    pg.locator("button[aria-label=\"Expand navigation\"]").first.click(); pg.wait_for_timeout(700)
    pg.get_by_text("Settings", exact=True).first.click(); pg.wait_for_timeout(1000)
    pg.screenshot(path=f"{outdir}/webui_settings.png")
    res["settings_text"] = pg.inner_text("body")[:12000]
    b.close()
print(json.dumps(res))
