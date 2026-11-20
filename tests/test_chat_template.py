#!/usr/bin/env python3
"""Compare the C++ chat template (qv_tok chat) with HF apply_chat_template (jinja) on conversation cases.
usage: python3 -I tests/test_chat_template.py BUILD/qv_tok [MODEL_DIR]"""
import json, subprocess, sys
qv_tok = sys.argv[1]
model_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/q-volta-models/Qwen3.8-27B"
from transformers import AutoTokenizer
hf = AutoTokenizer.from_pretrained(model_dir)
U = lambda t: {"role": "user", "content": t}
A = lambda t, r=None: ({"role": "assistant", "content": t} if r is None else {"role": "assistant", "content": t, "reasoning_content": r})
S = lambda t: {"role": "system", "content": t}
T = lambda t: {"role": "tool", "content": t}
cases = [
    dict(messages=[U("Hello")]),
    dict(messages=[U("Hello")], enable_thinking=False),
    dict(messages=[U("Hello")], enable_thinking=True),
    dict(messages=[S("You are terse."), U("Hi")]),
    dict(messages=[S("You are terse."), U("Hi")], enable_thinking=False),
    dict(messages=[S(""), U("Hi")]),
    dict(messages=[S("  padded system \n"), U("  padded user \n\n")], enable_thinking=False),
    dict(messages=[U("Hi"), A("Hello!"), U("How are you?")]),
    dict(messages=[U("Hi"), A("Hello!"), U("How are you?")], enable_thinking=False),
    dict(messages=[U("Hi"), A("Hello!", "user greeted me"), U("And?")]),
    dict(messages=[U("Hi"), A("Hello!", "user greeted me"), U("And?")], preserve_thinking=False),
    dict(messages=[U("Hi"), A("Hello!", "  spaced  \n"), U("And?"), A("Fine", "r2"), U("ok")], preserve_thinking=False),
    dict(messages=[U("Hi"), A("Hello!", "x"), U("And?"), A("Fine", "r2")], preserve_thinking=False),
    dict(messages=[S("S"), U("a"), A("b"), U("c")], reasoning_effort="low"),
    dict(messages=[S("S"), U("a")], reasoning_effort="medium"),
    dict(messages=[U("a")], reasoning_effort="medium"),
    dict(messages=[U("a")], reasoning_effort="low", enable_thinking=False),
    dict(messages=[U("a")], reasoning_effort="bogus"),
    dict(messages=[U("a"), A("b")], add_generation_prompt=False),
    dict(messages=[U("a")], add_generation_prompt=False),
    dict(messages=[{"role": "user", "content": [{"type": "text", "text": "multi"}, {"type": "text", "text": "part"}]}]),
    dict(messages=[U("Привет 你好 😀\n\tcode")]),
    dict(messages=[U("run it"), A("calling", "need tool"), {"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]),
    dict(messages=[U("q"), A("a"), T("tool out 1"), T("tool out 2"), U("next")]),
    dict(messages=[U("q"), T("t")]),
    dict(messages=[T("t"), U("q")]),
    dict(messages=[A("only assistant")]),
    dict(messages=[U("a"), S("late system")]),
    dict(messages=[]),
    dict(messages=[U("a"), {"role": "weird", "content": "x"}]),
    dict(messages=[{"role": "user", "content": None}]),
]
KW = ("enable_thinking", "reasoning_effort", "preserve_thinking", "add_generation_prompt")
def hf_render(c):
    kw = {k: c[k] for k in KW if k in c}
    kw.setdefault("add_generation_prompt", True)
    try:
        return {"text": hf.apply_chat_template(c["messages"], tokenize=False, **kw)}
    except Exception as e:
        return {"error": str(e)}
inp = "\n".join(json.dumps(c) for c in cases) + "\n"
out = [json.loads(l) for l in subprocess.run([qv_tok, model_dir + "/tokenizer.json", "chat"], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()]
ok = 0
for c, g in zip(cases, out):
    w = hf_render(c)
    same = ("text" in w and g.get("text") == w["text"]) or ("error" in w and "error" in g)
    ok += same
    if not same:
        print("MISMATCH", json.dumps(c)[:150], "\n  hf:", json.dumps(w)[:400], "\n  qv:", json.dumps(g)[:400])
print(f"chat template: {ok}/{len(cases)} match ({sum('error' in hf_render(c) for c in cases)} of the cases are expected errors)")
sys.exit(0 if ok == len(cases) else 1)
