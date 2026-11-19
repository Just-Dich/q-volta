import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import qv_format as F
out = sys.argv[1]; os.makedirs(out, exist_ok=True); torch.manual_seed(1)
for bits in (2, 3, 4, 5, 6, 8):
    for K in (5120, 6144, 17408):
        N = 300  # not a multiple of 4: exercises the tail
        W = torch.randn(N, K) * 0.02
        q, s, m = F.quantize_rtn(W, bits); pk = F.pack(q, bits); sm = torch.stack([s, m], -1)
        Wd = F.dequant(pk, sm, bits).double()
        x = torch.randn(K); r = torch.randn(N)
        y = Wd @ x.double() + r.double()
        p = f"{out}/b{bits}_k{K}"
        pk.numpy().tofile(p + ".q"); sm.numpy().astype(np.float16).tofile(p + ".sm")
        x.numpy().tofile(p + ".x"); r.numpy().tofile(p + ".r"); y.numpy().tofile(p + ".y")
