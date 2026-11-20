#!/usr/bin/env python3
"""Passkey / needle-in-a-haystack generator and scorer for Q-Volta (lengths are REAL tokens from qv_tok).

  niah.py gen   --tok TOKENIZER.json --qv-tok build/qv_tok --text wikitext2_test.txt --out DIR
                --lengths 4096,8192,16384,32768 --depths 0.1,0.5,0.9 --trials 2 [--seed 1]
      writes DIR/p_<L>_<depth>_<trial>.i32 and DIR/manifest.json
      prompt = chat user turn: wikitext filler with "The pass key is NNNNN. Remember it. NNNNN is the pass key." inserted at the
               given depth, then "What is the pass key mentioned in the document? Answer with the number only.", assistant turn
               opened with thinking off; the greedy answer must contain NNNNN
  niah.py score --tok TOKENIZER.json --qv-tok build/qv_tok --manifest DIR/manifest.json --results RUN.jsonl
      RUN.jsonl = one `qv_run --gen` JSON line per prompt (same order as the manifest, or "file" matched)
"""
import argparse, json, random, struct, subprocess, sys, os


def encode(qv_tok, tok, text, special=False):
    p = subprocess.run([qv_tok, tok, "encode"] + ([] if special else ["nospecial"]), input=json.dumps(text) + "\n", capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[0])


def decode(qv_tok, tok, ids):
    p = subprocess.run([qv_tok, tok, "decode"], input=json.dumps(ids) + "\n", capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[0])


def gen(a):
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    text = open(a.text, encoding="utf-8").read()
    # filler: the wikitext test set, tokenized in pieces (the tokenizer CLI takes one JSON line)
    filler = []
    step = 200000
    for i in range(0, min(len(text), 12 * step), step):
        filler += encode(a.qv_tok, a.tok, text[i:i + step])
    # chat format, thinking off (Qwen3.5 template): the model answers directly
    head = encode(a.qv_tok, a.tok, "<|im_start|>user\nBelow is a long document. Read it carefully.\n\n", special=True)
    question = encode(a.qv_tok, a.tok, "\n\nWhat is the pass key mentioned in the document? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", special=True)
    man = []
    for L in [int(x) for x in a.lengths.split(",")]:
        for d in [float(x) for x in a.depths.split(",")]:
            for t in range(a.trials):
                key = "%05d" % rng.randint(10000, 99999)
                needle = encode(a.qv_tok, a.tok, "\nThe pass key is %s. Remember it. %s is the pass key.\n" % (key, key))
                flen = L - len(needle) - len(question) - len(head)
                start = (t * 7919) % 4000
                fill = filler[start:start + flen]
                assert len(fill) == flen, "not enough filler text"
                cut = int(flen * d)
                ids = head + fill[:cut] + needle + fill[cut:] + question
                assert len(ids) == L
                name = "p_%d_%.2f_%d.i32" % (L, d, t)
                with open(os.path.join(a.out, name), "wb") as f:
                    f.write(struct.pack("%di" % len(ids), *ids))
                man.append({"file": name, "length": L, "depth": d, "trial": t, "key": key, "needle_start": cut})
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("wrote", len(man), "prompts to", a.out)


NAMES = ["Alder", "Bramble", "Cobalt", "Dunmore", "Esker", "Fennimore", "Garnet", "Hollis", "Ironwood", "Juniper"]


def gen_multi(a):
    """K distractor needles ("The secret code for NAME is NNNNN.") spread over the document; ask for ONE of them.
    Harder than the single passkey: the answer must be bound to the right name. Needle i sits at depth (i+0.5)/K."""
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    text = open(a.text, encoding="utf-8").read()
    filler = []
    for i in range(0, min(len(text), 12 * 200000), 200000):
        filler += encode(a.qv_tok, a.tok, text[i:i + 200000])
    head = encode(a.qv_tok, a.tok, "<|im_start|>user\nBelow is a long document. Read it carefully.\n\n", special=True)
    man = []
    K = a.multi
    for L in [int(x) for x in a.lengths.split(",")]:
        for t in range(a.trials):
            names = rng.sample(NAMES, K)
            keys = ["%05d" % rng.randint(10000, 99999) for _ in names]
            target = t % K  # rotate the asked needle through the document
            question = encode(a.qv_tok, a.tok, "\n\nWhat is the secret code for %s? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % names[target], special=True)
            needles = [encode(a.qv_tok, a.tok, "\nThe secret code for %s is %s.\n" % (n, k)) for n, k in zip(names, keys)]
            flen = L - sum(len(n) for n in needles) - len(question) - len(head)
            start = (t * 7919) % 4000
            fill = filler[start:start + flen]
            assert len(fill) == flen
            ids, prev = list(head), 0
            for i, nd in enumerate(needles):
                cut = int(flen * (i + 0.5) / K)
                ids += fill[prev:cut] + nd
                prev = cut
            ids += fill[prev:] + question
            assert len(ids) == L
            name = "m_%d_%d.i32" % (L, t)
            with open(os.path.join(a.out, name), "wb") as f:
                f.write(struct.pack("%di" % len(ids), *ids))
            man.append({"file": name, "length": L, "depth": (target + 0.5) / K, "trial": t, "key": keys[target], "name": names[target]})
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("wrote", len(man), "multi-needle prompts to", a.out)


def score(a):
    man = json.load(open(a.manifest))
    res = [json.loads(l) for l in open(a.results) if l.strip().startswith("{")]
    assert len(res) == len(man), "results/manifest length mismatch %d vs %d" % (len(res), len(man))
    rows, agg = [], {}
    for m, r in zip(man, res):
        out = decode(a.qv_tok, a.tok, r["ids"])
        ok = m["key"] in out.replace(" ", "")
        rows.append({**m, "output": out, "ok": ok})
        agg.setdefault(m["length"], []).append(ok)
    summ = {str(L): {"correct": sum(v), "total": len(v)} for L, v in sorted(agg.items())}
    print(json.dumps({"summary": summ, "rows": rows}, indent=1) if a.verbose else json.dumps({"summary": summ}))
    if a.out:
        json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--tok", required=True); g.add_argument("--qv-tok", required=True); g.add_argument("--text", required=True)
    g.add_argument("--out", required=True); g.add_argument("--lengths", default="4096,8192,16384,32768")
    g.add_argument("--multi", type=int, default=0, help="K>0: K-needle variant (see gen_multi)"); g.add_argument("--depths", default="0.1,0.5,0.9"); g.add_argument("--trials", type=int, default=2); g.add_argument("--seed", type=int, default=1)
    s = sp.add_parser("score")
    s.add_argument("--tok", required=True); s.add_argument("--qv-tok", required=True); s.add_argument("--manifest", required=True)
    s.add_argument("--results", required=True); s.add_argument("--out"); s.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    (gen_multi(a) if a.multi else gen(a)) if a.cmd == "gen" else score(a)
