#!/usr/bin/env python3 -I
"""End-to-end vision golden: our fp32 text reference (tools/inkling_ref.py) consuming the HF-tower image embeddings
(tests/golden/inkling_vision/<img>.npz 'embeds') injected at the placeholder positions. Writes tests/golden/inkling_vision/e2e_<name>.npz
(ids, mask, logits_last, gen_ids, gen_top20_*). usage: inkling_vision_golden.py IMG_NAME 'question' [out_name] [--threads N]
Prompt = chat template (HF tokenizer) with one image part, the single <|unused_200054|> placeholder expanded to the token count."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling_ref import Inkling

def main():
    name, q = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else name
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 32)
    G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "inkling_vision")
    M = "/workspace/q-volta-models/Inkling-2-Mini"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M)
    e = np.load(os.path.join(G, name + ".npz"))["embeds"]
    ids = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True)["input_ids"]
    k = ids.index(200054)
    ids = ids[:k] + [200054] * len(e) + ids[k + 1:]
    mask = np.array([i == 200054 for i in ids])
    print("prompt tokens", len(ids), "image tokens", int(mask.sum()), flush=True)
    m = Inkling(M, expert_cache_gb=30.0)
    cache = m.new_cache()
    with torch.no_grad():
        lg = m.forward(ids, cache, inject=(mask, e))[0]
    logits_last = lg.numpy().astype(np.float32)
    gi, gv, gen = [], [], []
    for s in range(32):
        tv, ti = torch.topk(lg, 20)
        gi.append(ti.numpy()); gv.append(tv.numpy()); gen.append(int(ti[0]))
        if s == 31: break
        with torch.no_grad(): lg = m.forward([gen[-1]], cache)[0]
    np.savez_compressed(os.path.join(G, f"e2e_{out}.npz"), ids=np.array(ids, np.int32), mask=mask, logits_last=logits_last, gen_ids=np.array(gen, np.int32),
                        gen_top20_ids=np.stack(gi).astype(np.int32), gen_top20_val=np.stack(gv).astype(np.float32))
    print("generated:", repr(tok.decode(gen)))
main()
