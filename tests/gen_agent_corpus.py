#!/usr/bin/env python3
"""Synthetic agentic eval corpus (NOT training data): real grep/read tool calls over files of this repo, rendered with the
pi adapter's chat template. Each conversation is cut to exactly SEQ tokens so converter windows align with conversations.
Usage: gen_agent_corpus.py OUT.txt [N=40] [SEQ=512]"""
import sys, re, glob, subprocess, random, json
from transformers import AutoTokenizer
out, N, SEQ = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40, int(sys.argv[3]) if len(sys.argv) > 3 else 512
D = "/workspace/q-volta-models/pi-adapter"
tok = AutoTokenizer.from_pretrained(D)
tools = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
files = sorted(glob.glob("/workspace/q-volta/src/**/*.*", recursive=True) + glob.glob("/workspace/q-volta/tools/*.py") + glob.glob("/workspace/q-volta/tests/*.*"))
rx = re.compile(r"^\s*(?:def|class|static|__global__|void|int|struct)\s+[\w:<>\* ]*?(\w{5,})\s*[\(:{]", re.M)
random.seed(0); convs = []
for f in files:
    src = open(f, errors="ignore").read()
    names = sorted(set(rx.findall(src)))
    if not names: continue
    nm = random.choice(names); rel = f.replace("/workspace/q-volta/", "")
    g = subprocess.run(["grep", "-n", nm, f], capture_output=True, text=True).stdout.strip().splitlines()[:6]
    first = g[0].split(":")[0] if g else "1"
    head = subprocess.run(["sed", "-n", f"{first},{int(first)+14}p", f], capture_output=True, text=True).stdout
    msgs = [{"role": "system", "content": "You are a coding agent working in /workspace/q-volta. Inspect code with tools before answering."},
            {"role": "user", "content": f"Where is `{nm}` defined and what does it do?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": f"grep -rn '{nm}' {rel}"}}}]},
            {"role": "tool", "content": "\n".join(g)},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": f"sed -n '{first},{int(first)+14}p' {rel}"}}}]},
            {"role": "tool", "content": head},
            {"role": "assistant", "content": f"`{nm}` is defined in `{rel}` at line {first}. From the code above, it " + "implements the logic shown in the first lines of its body; I read it directly rather than guessing."}]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) >= SEQ: convs.append(tok.decode(ids[:SEQ]))
    if len(convs) == N: break
txt = "".join(convs); open(out, "w").write(txt)
ids = tok(txt, add_special_tokens=False)["input_ids"]
print(len(convs), "convs;", len(ids), "tokens; expected", len(convs) * SEQ, "; windows start with <|im_start|>:",
      sum(ids[i * SEQ] == tok.convert_tokens_to_ids("<|im_start|>") for i in range(len(convs))))
