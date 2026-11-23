#!/usr/bin/env python3
"""Aggregate raw/*.json(l) into results.json (mean/min/max over rounds) and print a markdown table."""
import glob, json, math, os, statistics as st, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
def ms(v): return {"mean": round(st.mean(v), 2), "min": round(min(v), 2), "max": round(max(v), 2), "n": len(v)}
res = {"quality": {}, "speed": {}}
for f in sorted(glob.glob(f"{R}/q4088_*.json")):
    tag = os.path.basename(f)[6:-5]
    try: d = json.loads(open(f).read().strip().splitlines()[-1])
    except Exception: continue
    q = res["quality"].setdefault(tag, {})
    q.update(ppl_4088=d["ppl"], kl_vs_q8=d.get("kl_topk_mean"), top1_vs_q8=d.get("top1_agree"), tail_mass=d.get("ref_tail_mass_mean"))
for f in sorted(glob.glob(f"{R}/q8x512_*.json")):
    tag = os.path.basename(f)[7:-5]
    nll = []
    for ln in open(f):
        try: d = json.loads(ln)
        except Exception: continue
        if "window" in d and d.get("window") == 512: res["quality"].setdefault(tag, {})["ppl_8x512"] = d["ppl"]; nll = None; break
        nll.append(d["mean_nll"])
    if nll: res["quality"].setdefault(tag, {})["ppl_8x512"] = round(math.exp(st.mean(nll)), 4)
cells = collections.defaultdict(list)
def rows(f):
    out = [json.loads(l) for l in open(f)]
    # old files: the qv_run "mtp" stats dict overwrote the depth; recover it from run order (per round: depth 0,2,3 x 5 prompts)
    byr = collections.defaultdict(list)
    for d in out:
        if d.get("engine") == "q-volta" and "mtp_depth" not in d: byr[d["round"]].append(d)
    for r, L in byr.items():
        for i, d in enumerate(L): d["mtp_depth"] = (0, 2, 3)[i // 5]
    return out
for f in sorted(glob.glob(f"{R}/speed_*.jsonl")):
    for d in rows(f):
        if "error" in d: continue
        lab = d["label"]
        if d["engine"] == "llama.cpp":
            k = (lab, "llama-bench", f"pp{d['n_prompt']}" if d["n_prompt"] else f"tg{d['n_gen']}@d{d['n_depth']}")
            cells[k].append(d["tok_s"])
        elif d["engine"] == "q-volta":
            m = f"mtp{d['mtp_depth']}" if d["mtp_depth"] else "plain"
            cells[(lab, f"qv_{m}", d["prompt"] + "_decode")].append(d["decode_tok_s"])
            cells[(lab, f"qv_{m}", d["prompt"] + "_prefill")].append(d["prefill_tok_s"])
            if isinstance(d.get("mtp"), dict): cells[(lab, f"qv_{m}", d["prompt"] + "_accept")].append(d["mtp"]["accept_rate"])
        elif d["engine"] == "llama.cpp-server":
            m = "plain" if d["spec"] == "none" else f"mtp{d['ndraft']}"
            cells[(lab, f"llama_server_{m}", d["prompt"] + "_decode")].append(d["decode_tok_s"])
            if d.get("draft_n"): cells[(lab, f"llama_server_{m}", d["prompt"] + "_accept")].append(d["draft_n_accepted"] / d["draft_n"])
for (lab, eng, cell), v in sorted(cells.items()):
    res["speed"].setdefault(lab, {}).setdefault(eng, {})[cell] = ms(v)
json.dump(res, open(os.path.join(R, "..", "results.json"), "w"), indent=1)
print(json.dumps(res["quality"], indent=1))
for lab, e in res["speed"].items():
    print("##", lab)
    for eng, c in e.items(): print(eng, {k: v["mean"] for k, v in c.items()})
