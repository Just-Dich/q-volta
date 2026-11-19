#!/usr/bin/env python3
"""Render an agentic tool-calling prompt with the pi adapter's chat template -> int32 token file.
Usage: gen_agent_prompt.py OUT.i32 [template_dir]"""
import sys, json, numpy as np
from transformers import AutoTokenizer
out = sys.argv[1]
tdir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/q-volta-models/pi-adapter"
tok = AutoTokenizer.from_pretrained(tdir)
tools = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
msgs = [{"role": "system", "content": "You are a coding agent. Work in /repo. Use tools to inspect the code before answering."},
        {"role": "user", "content": "The function human_to_bytes in lib/utils.py rejects '1.5K'. Find it and tell me what it does."}]
text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = tok(text, add_special_tokens=False)["input_ids"]
np.array(ids, dtype=np.int32).tofile(out)
print(len(ids), "tokens"); print(text[-300:])
