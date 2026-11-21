#!/usr/bin/env python3 -I
"""Golden generator for Inkling-2-Mini (see tools/inkling_ref.py, docs/INKLING_SPEC.md).

  python3 -I tools/inkling_golden.py --model DIR --out tests/golden/inkling [--emu-bf16] [--only p1,p2] [--threads 48]

For each prompt writes <out>/<name>.npz with: ids (prompt token ids), logits_last (fp32 [200058], last prompt position,
after softcap 30), gen_ids (32 greedy tokens), gen_top20_ids/gen_top20_val (top-20 per decode step, incl. the prefill
step as row 0), hidden_pos (positions saved), hid_<app> (fp32 [3, 2304]: residual-stream output of each of the 48 layer
applications at positions [0, n//2, n-1]), emb, final_norm, router_<app> (int8 [n, 8] expert ids).
Also a manifest <out>/manifest.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling_ref import Inkling, C


def build_prompts(model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    P = {}

    def chat(txt):
        return tok.apply_chat_template([{"role": "user", "content": txt}], tokenize=True, add_generation_prompt=True,
                                       return_dict=True)["input_ids"]

    P["p1_en"] = chat("Explain in two sentences why the sky is blue.")
    P["p2_code"] = chat("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.")
    P["p3_multi"] = chat("请用中文简要介绍一下长城。 Puis résume ta réponse en français, en une phrase.")
    readme = open(os.path.join(model_dir, "README.md"), encoding="utf-8").read()
    body_ids = tok.encode(readme[readme.index("# Inkling 2 Mini"):], add_special_tokens=False)
    suffix = "\n\nQuestion: according to the text above, how many routed experts does each token use, and out of how many?"
    # chat() wraps the text; fit the total to ~4096 tokens by trimming the README body
    lo, hi = 1000, len(body_ids)
    while lo < hi:  # largest body length with total <= 4096
        mid = (lo + hi + 1) // 2
        if len(chat(tok.decode(body_ids[:mid]) + suffix)) <= 4096:
            lo = mid
        else:
            hi = mid - 1
    P["p4_long4k"] = chat(tok.decode(body_ids[:lo]) + suffix)
    return tok, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/q-volta-models/Inkling-2-Mini")
    ap.add_argument("--out", default="tests/golden/inkling")
    ap.add_argument("--emu-bf16", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--cache-gb", type=float, default=40.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tok, P = build_prompts(a.model)
    m = Inkling(a.model, emu_bf16=a.emu_bf16, expert_cache_gb=a.cache_gb)
    names = [n for n in P if not a.only or n.split("_")[0] in a.only.split(",") or n in a.only.split(",")]
    man = {"emu_bf16": a.emu_bf16, "prompts": {}}
    for name in names:
        ids = P[name]
        n = len(ids)
        print(f"== {name}: {n} tokens", flush=True)
        cache = m.new_cache()
        t0 = time.time()
        with torch.no_grad():
            logits, tr = m.forward(ids, cache, want_trace=True)
        tp = time.time() - t0
        out = {"ids": np.array(ids, dtype=np.int32), "logits_last": logits[0].numpy().astype(np.float32)}
        posl = [0, n // 2, n - 1]
        out["hidden_pos"] = np.array(posl)
        for k, v in tr.items():
            if k.endswith("_router"):
                if not a.no_router:
                    out["router_" + k[:-7]] = v.numpy().astype(np.int8)
            else:
                out["hid_" + k] = v[posl].numpy().astype(np.float32)
        top_ids, top_val, gen = [], [], []
        lg = logits[0]
        t1 = time.time()
        for s in range(a.gen):
            tv, ti = torch.topk(lg, 20)
            top_ids.append(ti.numpy()); top_val.append(tv.numpy())
            nxt = int(ti[0])
            gen.append(nxt)
            if s == a.gen - 1:
                break
            with torch.no_grad():
                lg = m.forward([nxt], cache)[0]
        td = time.time() - t1
        out["gen_ids"] = np.array(gen, dtype=np.int32)
        out["gen_top20_ids"] = np.stack(top_ids).astype(np.int32)
        out["gen_top20_val"] = np.stack(top_val).astype(np.float32)
        np.savez_compressed(os.path.join(a.out, name + ".npz"), **out)
        txt = tok.decode(gen)
        print(f"  prefill {tp:.1f}s, decode {td:.1f}s for {len(gen)} tokens; generated: {txt!r}", flush=True)
        man["prompts"][name] = {"n_prompt": n, "prefill_s": round(tp, 1), "decode_s": round(td, 1), "gen_text": txt}
    json.dump(man, open(os.path.join(a.out, ("manifest_emu_%s.json" if a.emu_bf16 else "manifest_%s.json") % "_".join(names)), "w"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
