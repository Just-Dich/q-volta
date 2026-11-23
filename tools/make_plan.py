#!/usr/bin/env python3
"""Per-tensor plan JSON for convert.py --plan from a class spec (see plan_size.py classes), optional per-layer overrides.
usage: make_plan.py OUT.json 'gu=3:64,down=4:64,qkvz=4:64,o=4:64' [--layer-up '0-5,58-63:down=5:64']
Class defaults for embed/head/mtp are NOT in the plan (use --bits / --group of convert.py); only gu/down/qkvz/o are emitted.
--layer-up 'RANGE:class=bits:group' overrides one class on a layer range (repeatable via ';')."""
import json, re, sys
LM = "model.language_model.layers."
def names(i, cls):
    full = i % 4 == 3
    a = {"gu": [f"mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"],
         "qkvz": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if full else ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
         "o": ["self_attn.o_proj"] if full else ["linear_attn.out_proj"]}[cls]
    return [f"{LM}{i}.{x}.weight" for x in a]
def parse(spec):
    d = {}
    for kv in spec.split(","):
        k, v = kv.split("=")
        b, g = v.split(":")
        d[k] = [int(b), int(g)]
    return d
out, spec = sys.argv[1], sys.argv[2]
cls = parse(spec)
plan = {}
for i in range(64):
    for c in ("gu", "down", "qkvz", "o"):
        for n in names(i, c):
            plan[n] = cls[c]
if "--layer-up" in sys.argv:
    for item in sys.argv[sys.argv.index("--layer-up") + 1].split(";"):
        rng, ov = item.split(":", 1)
        layers = set()
        for r in rng.split(","):
            a, _, b = r.partition("-")
            layers |= set(range(int(a), int(b or a) + 1))
        for k, v in parse(ov).items():
            for i in layers:
                for n in names(i, k):
                    plan[n] = v
json.dump(plan, open(out, "w"), indent=0)
print(len(plan), "tensors ->", out)
