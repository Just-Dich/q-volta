#!/usr/bin/env python3
"""Takes the docs/img/webui_*.png screenshots of the built-in web UI against a mock-backend qv_server.

    PLAYWRIGHT_BROWSERS_PATH=/tmp/pwbrowsers /tmp/pwenv/bin/python scripts/webui_screenshots.py QV_SERVER TOKENIZER_JSON OUTDIR [--path DIR]

Starts its own qv_server (mock backend, canned reply with reasoning, a list and a code block); --path serves an unpacked UI build from disk.
"""
import os, socket, subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright

qv, tok, out = sys.argv[1:4]
extra = sys.argv[sys.argv.index("--path"):sys.argv.index("--path") + 2] if "--path" in sys.argv else []
REPLY = ("The user greets me and wants a short list, so a list and a tiny code sample will do.\n</think>\n\n"
         "Hello! I am **Q-Volta**, running locally. Here is a list:\n\n1. one\n2. two\n3. three\n\n"
         "And a snippet:\n\n```python\ndef fib(n: int) -> int:\n    # iterative Fibonacci\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\nUse `inline code` too.\n")

def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def start(text, delay="20000", *more):
    p = port()
    proc = subprocess.Popen([qv, "--backend", "mock", "--tokenizer", tok, "--port", str(p), "-c", "32768", "--mock-text", text, "--mock-delay-us", delay, "-a", "Qwen3.8-27B-Q4", "--temp", "0", "--top-k", "1", *extra, *more],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try: urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=1); break
        except Exception: time.sleep(0.1)
    return proc, f"http://127.0.0.1:{p}/"

def send(pg, text):
    pg.locator(".cm-content").first.click(); pg.keyboard.type(text); pg.keyboard.press("Enter")

def theme(pg, mode):
    pg.evaluate("m => { const c = JSON.parse(localStorage.getItem('LlamaUi.config') || '{}'); c.theme = m; localStorage.setItem('LlamaUi.config', JSON.stringify(c)); }", mode)

proc, url = start(REPLY, "8000")
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 800}); pg.goto(url); pg.wait_for_timeout(2500)
        pg.screenshot(path=f"{out}/webui_home.png")
        send(pg, "Say hi and give me a short list."); pg.wait_for_timeout(900)
        pg.screenshot(path=f"{out}/webui_streaming.png"); pg.wait_for_timeout(5000)
        pg.get_by_text("Reasoning", exact=True).first.click(); pg.wait_for_timeout(600)
        pg.screenshot(path=f"{out}/webui_reasoning.png")
        pg.get_by_text("Reasoning", exact=True).first.click(); pg.wait_for_timeout(400)
        pg.screenshot(path=f"{out}/webui_chat.png")
        pg.locator("pre").first.screenshot(path=f"{out}/webui_code.png")
        pg.locator("button[aria-label=\"Expand navigation\"]").first.click(); pg.wait_for_timeout(700)
        pg.screenshot(path=f"{out}/webui_sidebar.png")
        pg.get_by_text("Settings", exact=True).first.click(); pg.wait_for_timeout(1000)
        pg.screenshot(path=f"{out}/webui_settings.png")
        # light theme, same chat
        theme(pg, "light"); pg.goto(url); pg.wait_for_timeout(2500)
        pg.screenshot(path=f"{out}/webui_light_home.png")
        send(pg, "Say hi and give me a short list."); pg.wait_for_timeout(6500)
        pg.screenshot(path=f"{out}/webui_light_chat.png")
        # mobile 390px, dark
        theme(pg, "dark")
        m = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True).new_page(); m.goto(url); m.wait_for_timeout(2500)
        m.screenshot(path=f"{out}/webui_mobile_home.png")
        m.locator(".cm-content").first.click(); m.keyboard.type("Say hi and give me a short list."); m.locator("button[type=submit]").last.click(); m.wait_for_timeout(6500)
        m.screenshot(path=f"{out}/webui_mobile_chat.png")
        b.close()
    # browser-side tool call awaiting permission, and the API-key screen
    for text, more, name, act in (("I will check the time.\n</think>\n\n<tool_call>\n<function=get_datetime>\n</function>\n</tool_call>", (), "webui_tool.png", "tool"),
                                  ("hi", ("--api-key", "secret"), "webui_apikey.png", "key")):
        pr, u = start(text, "20000", *more)
        try:
            with sync_playwright() as p:
                b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 800}); pg.goto(u); pg.wait_for_timeout(2500)
                if act == "tool": send(pg, "what time is it?"); pg.wait_for_timeout(5000)
                pg.screenshot(path=f"{out}/{name}"); b.close()
        finally:
            pr.terminate()
finally:
    proc.terminate()
