"""Tokenize benchmark prompts with the model's chat template (thinking disabled) -> int32 files."""
import json, sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/q-volta-models/Qwen3.8-27B")
P = {  # identical to /workspace/bench-qwen27.py (pre-existing NInfer runs)
 "code": "Напиши на Python функцию binary_search для отсортированного списка. Добавь короткий пример вызова. Только код.",
 "prose": "Объясни простыми словами, почему сменяются времена года на Земле. Расскажи о наклоне оси и разнице между полушариями.",
}
for k, v in P.items():
    ids = tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True, tokenize=True,
                                  enable_thinking=False)
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    np.array(ids, dtype=np.int32).tofile(f"benchmarks/prompts/{k}.i32")
    print(k, len(ids), repr(tok.decode(ids)[-60:]))
