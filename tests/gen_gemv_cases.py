import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import qv_format as F
import qv_quant as Q
out = sys.argv[1]; os.makedirs(out, exist_ok=True); torch.manual_seed(1)
for bits in (2, 3, 4, 5, 6, 8):
  for g in (64, 128, 129):  # 129 = group 128 with meta8 metadata
    for K in (5120, 6144, 17408):
        N = 300  # not a multiple of 4: exercises the tail
        W = torch.randn(N, K) * 0.02
        if g == 129:
            q, S8, Z8, rs = Q.rtn(W, bits, 128, clip=True, meta8=True); pk = F.pack(q, bits); sm = Q.pack_meta(S8, Z8, rs)
            Wd = F.dequant(pk, sm, bits, 128, rs).double()
        else:
            q, s, m = F.quantize_rtn(W, bits, g); pk = F.pack(q, bits); sm = torch.stack([s, m], -1)
            Wd = F.dequant(pk, sm, bits, g).double()
        x = torch.randn(K); r = torch.randn(N)
        y = Wd @ x.double() + r.double()
        p = f"{out}/b{bits}_k{K}" + ("" if g == 64 else "_g128" if g == 128 else "_m8")
        pk.numpy().tofile(p + ".q"); sm.numpy().astype(np.float16 if g != 129 else np.uint16).tofile(p + ".sm")
        if g == 129: rs.numpy().astype(np.float16).tofile(p + ".rs")
        x.numpy().tofile(p + ".x"); r.numpy().tofile(p + ".r"); y.numpy().tofile(p + ".y")
