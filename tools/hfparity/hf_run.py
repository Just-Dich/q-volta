import time, sys, json, numpy as np, torch
from transformers import AutoConfig, Inkling2LoopForCausalLM
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 48)
M = "/workspace/q-volta-models/Inkling-2-Mini"
G = "/workspace/q-volta/tests/golden/inkling/"
OUT = "/workspace/q-volta-models/scratch-hfparity/out/"
names = sys.argv[1].split(",")
CH = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
t0 = time.time()
cfg = AutoConfig.from_pretrained(M)
m = Inkling2LoopForCausalLM.from_pretrained(M, config=cfg.text_config, dtype=torch.bfloat16).eval()
print("loaded", time.time() - t0, flush=True)
st = {"rec": False, "pos": None, "app": [], "router": [], "embl": [], "finl": [], "chunks": []}
def lay_hook(mod, a, out):
    if st["rec"] and len(st["pos"]): st["app"].append(out[0, st["pos"]].float().clone())
def gate_hook(mod, a, out):
    if st["rec"]: st["router"].append(out[2].clone().to(torch.int16))
def emb_hook(mod, a, out):
    if st["rec"] and len(st["pos"]): st["embl"].append(out[0, st["pos"]].float().clone())
def norm_hook(mod, a, out):
    if st["rec"] and len(st["pos"]): st["finl"].append(out[0, st["pos"]].float().clone())
for l in m.model.layers:
    l.register_forward_hook(lay_hook)
    l.mlp.gate.register_forward_hook(gate_hook)
m.model.embed_tokens.register_forward_hook(emb_hook)
m.model.norm.register_forward_hook(norm_hook)
for nm in names:
    g = np.load(G + nm + ".npz")
    ids = torch.tensor(g["ids"].astype(np.int64))[None]
    gen = [int(x) for x in g["gen_ids"]]
    hp = g["hidden_pos"]; n = ids.shape[1]
    st.update(embl=[], finl=[], chunks=[])
    t = time.time(); cache = None; routers = []; apps = []; embs = []; fins = []
    for s0 in range(0, n, CH):
        e0 = min(n, s0 + CH)
        loc = torch.tensor([int(p) - s0 for p in hp if s0 <= p < e0], dtype=torch.long)
        st.update(rec=True, pos=loc, app=[], router=[], embl=[], finl=[])
        with torch.no_grad():
            o = m(input_ids=ids[:, s0:e0], cache=cache, use_cache=True, logits_to_keep=1)
        cache = o.cache
        st["rec"] = False
        print("chunk", s0, e0, time.time() - t, flush=True)
        routers.append(torch.stack(st["router"]))          # [apps, chunk, 8]
        if len(loc):
            apps.append(torch.stack(st["app"])); embs.append(st["embl"][0]); fins.append(st["finl"][0])
    st["router"] = [torch.cat(routers, 1)[i] for i in range(len(routers[0]))]
    st["app"] = list(torch.cat(apps, 1)); st["emb"] = torch.cat(embs, 0); st["fin"] = torch.cat(fins, 0)
    print(nm, "prefill", ids.shape[1], time.time() - t, flush=True)
    lg = [o.logits[0, -1].clone()]
    for i in range(len(gen) - 1):   # teacher-forced decode with the golden ids
        with torch.no_grad():
            o2 = m(input_ids=torch.tensor([[gen[i]]]), cache=cache, use_cache=True)
        lg.append(o2.logits[0, -1].clone())
    lg = torch.stack(lg)
    top = lg.topk(20, -1)
    d = {"logits_last": lg[0].numpy(), "emb": st["emb"].numpy(), "fin": st["fin"].numpy(),
         "app": torch.stack(st["app"]).numpy(), "router": torch.stack(st["router"]).numpy(),
         "hf_argmax": lg.argmax(-1).numpy(), "top20_ids": top.indices.numpy(), "top20_val": top.values.numpy()}
    np.savez(OUT + "hf_" + nm + ".npz", **d)
    print(nm, "done", time.time() - t, "argmax==golden:", int((d["hf_argmax"] == np.array(gen)).sum()), "/", len(gen), flush=True)
