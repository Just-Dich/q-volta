# HF side of the long-context (tau != 1) check: chunked prefill of the first N ids of out/long_ids.npy
import time, sys, numpy as np, torch
from transformers import AutoConfig, Inkling2LoopForCausalLM
N = int(sys.argv[1]); CH = int(sys.argv[2]); torch.set_num_threads(48)
M = "/workspace/q-volta-models/Inkling-2-Mini"; OUT = "/workspace/q-volta-models/scratch-hfparity/out/"
cfg = AutoConfig.from_pretrained(M)
m = Inkling2LoopForCausalLM.from_pretrained(M, config=cfg.text_config, dtype=torch.bfloat16).eval()
ids = torch.tensor(np.load(OUT + "long_ids.npy")[:N].astype(np.int64))[None]
st = {"rec": False, "app": []}
for l in m.model.layers:
    l.register_forward_hook(lambda mod, a, out: st["app"].append(out[0, -1].float().clone()) if st["rec"] else None)
cache = None; t = time.time(); o = None
for s in range(0, N, CH):
    e = min(N, s + CH)
    st["rec"] = (e == N); st["app"] = []
    with torch.no_grad():
        o = m(input_ids=ids[:, s:e], cache=cache, use_cache=True, logits_to_keep=1)
    cache = o.cache
    print("chunk", s, e, time.time() - t, flush=True)
lg = o.logits[0, -1]; top = lg.topk(20)
np.savez(OUT + f"hf_long{N}.npz", logits_last=lg.numpy(), app=torch.stack(st["app"]).numpy(), top20_ids=top.indices.numpy())
print("done", time.time() - t, flush=True)
