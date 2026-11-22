import sys, torch, json
sys.path.insert(0, "/workspace/q-volta/tools")
import inkling_ref as R
from transformers.core_model_loading import Inkling2LoopDequantizeExperts as D
m = R.Inkling("/workspace/q-volta-models/Inkling-2-Mini", verbose=False)
res = {}
for L in (0, 13, 19, 20, 23):
    p = f"model.llm.layers.{L}.mlp.experts."; f = m.s.f
    for nm in ("w13_weight", "w2_weight"):
        E = [0, 5, 77, 127]
        t = {}
        for suf in ("", "_scale", "_amax", "_outer_scale_denominator"):
            k = p + nm + suf
            if k in f.keys():
                x = f.get_tensor(k) if suf == "_outer_scale_denominator" else f.get_tensor(k)[E]
                t[suf] = x
        srcs = [r"w" , ]  # patterns only used as dict keys
        pats = [nm, nm + "_scale", nm + "_amax", nm + "_outer_scale_denominator"]
        inp = {pats[i]: t.get(s) for i, s in enumerate(("", "_scale", "_amax", "_outer_scale_denominator")) if s in t}
        out = D().convert(inp, pats, ["out"], model=None)["out"]  # [4, rows, cols] bf16, un-interleaved
        for j, e in enumerate(E):
            w13, w2 = m.expert(L, e)
            ours = w13 if nm == "w13_weight" else w2
            theirs = out[j]
            if nm == "w13_weight": theirs = torch.cat([theirs[0::2], theirs[1::2]], 0)
            res[f"L{L}.{nm}.e{e}"] = float((ours.float() - theirs.float()).abs().max())
print(json.dumps(res)); print("max", max(res.values()))
