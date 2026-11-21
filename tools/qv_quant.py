"""Weight quantizers for the Q-Volta format: per-group asymmetric (scale, min) in fp16, q in [0, 2^bits).

Every quantizer returns (q uint8 [N,K], S [N,K/g], M [N,K/g], rs) so that any of them can be packed with
qv_format.pack and reconstructed by the engine as w = scale*q + min.  Two metadata encodings:
  fp16  (rs is None): S = scale fp16, M = min fp16                                            (16+16 bit / group)
  meta8 (rs fp16 [N]): S = s8 uint8, M = z8 uint8;  scale = s8 * rs/255,  min = -scale * z,  z = zlo + z8*zstep with
        zlo = -0.25*qmax, zstep = 1.5*qmax/255            (8+8 bit / group + 2 bytes / row; measured MSE change < 0.1%)
Methods:
  rtn   : round-to-nearest from the group min/max, optionally with a per-group clip search (MSE, column-weighted)
  gptq  : GPTQ (Frantar et al. 2022) error-compensating rounding with a calibration Hessian H = 2/N sum x x^T, static
          groups along K (no act-order, the format has contiguous groups), group parameters (with clip search) taken from
          the error-compensated weights when the sweep reaches the group.
"""
import torch

CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
RS_MARGIN = 1.25  # row scale headroom over the RTN group scales (GPTQ-compensated groups can have slightly larger ranges)


def zrange(bits):
    qmax = (1 << bits) - 1
    return -0.25 * qmax, 1.5 * qmax / 255.0


def meta8_decode(s8, z8, rs, bits):
    """s8, z8 uint8-valued [N,G]; rs fp16 [N] -> (scale, min) fp32 [N,G]. Same fp32 operation order as the CUDA kernels."""
    zlo, zst = zrange(bits)
    sc = s8.float() * (rs.float() / 255.0)[:, None]
    z = z8.float() * zst + zlo
    return sc, -(sc * z)


def row_scale(W, bits, group):
    N, K = W.shape
    Wg = W.view(N, K // group, group)
    sc = (Wg.amax(-1) - Wg.amin(-1)) / ((1 << bits) - 1)
    rs = (sc.amax(-1) * RS_MARGIN).clamp_min(1e-8).half()
    return torch.where(rs.float() > 0, rs, torch.ones_like(rs))


def _params(wg, bits, colw=None, clip=True, rs=None):
    """wg fp32 [N, g] -> (sf, mf, codeS, codeM). sf/mf are the DECODED fp32 scale/min the engine will use; clip search
    picks, per row, the range shrink with the least (column-weighted) MSE of the quantized group."""
    qmax = (1 << bits) - 1
    mn = wg.amin(-1)
    mx = wg.amax(-1)
    zlo, zst = zrange(bits)
    best = None
    for p in (CLIPS if clip else (1.0,)):
        c = 0.5 * (mx + mn)
        lo = c - (c - mn) * p
        hi = c + (mx - c) * p
        sc = (hi - lo) / qmax
        if rs is None:
            cs = sc.half()
            cm = lo.half()
            sf = cs.float()
            sf = torch.where(sf > 0, sf, torch.ones_like(sf))
            mf = cm.float()
        else:
            u = rs.float() / 255.0
            cs = torch.round(sc / u).clamp_(1, 255)
            sf = cs * u
            z = -lo / sf
            cm = torch.round((z - zlo) / zst).clamp_(0, 255)
            mf = -(sf * (cm * zst + zlo))
        q = torch.round((wg - mf[:, None]) / sf[:, None]).clamp_(0, qmax)
        d = q * sf[:, None] + mf[:, None] - wg
        e = d * d
        if colw is not None:
            e = e * (colw if colw.dim() == 2 else colw[None, :])  # [g] shared, or [rows, g] per row (batched GPTQ)
        e = e.sum(-1)
        if best is None:
            best = [e, sf, mf, cs, cm]
        else:
            m = e < best[0]
            for k, v in enumerate((e, sf, mf, cs, cm)):
                best[k] = torch.where(m, v, best[k])
    _, sf, mf, cs, cm = best
    if rs is not None:
        cs, cm = cs.to(torch.uint8), cm.to(torch.uint8)
    return sf, mf, cs, cm


def _quant_col(w, sf, mf, qmax):
    q = torch.round((w - mf) / sf).clamp_(0, qmax)
    return q, q * sf + mf


def _meta_dtype(rs):
    return torch.float16 if rs is None else torch.uint8


def rtn(W, bits, group=64, clip=False, colw=None, meta8=False):
    """W fp32 [N,K] -> (q uint8 [N,K], S, M, rs).  colw: optional per-column importance (e.g. diag H)."""
    N, K = W.shape
    qmax = (1 << bits) - 1
    G = K // group
    Wg = W.view(N, G, group)
    rs = row_scale(W, bits, group) if meta8 else None
    S = torch.empty(N, G, dtype=_meta_dtype(rs), device=W.device)
    M = torch.empty_like(S)
    sf = torch.empty(N, G, device=W.device)
    mf = torch.empty(N, G, device=W.device)
    if colw is None:
        rsf = None if rs is None else rs.repeat_interleave(G)
        a, b, cs, cm = _params(Wg.reshape(N * G, group), bits, None, clip, rsf)
        sf, mf, S, M = a.view(N, G), b.view(N, G), cs.view(N, G), cm.view(N, G)
    else:
        for j in range(G):
            sf[:, j], mf[:, j], S[:, j], M[:, j] = _params(Wg[:, j], bits, colw[j * group:(j + 1) * group], clip, rs)
    q = torch.round((Wg - mf[..., None]) / sf[..., None]).clamp_(0, qmax).to(torch.uint8)
    return q.view(N, K), S, M, rs


def hinv_chol(H, percdamp=0.01):
    """Upper Cholesky factor of H^-1 (damped). H [K,K] fp32 on device (not modified). Scale-invariant in H."""
    H = H.clone()
    d = torch.diagonal(H)
    d[d == 0] = 1.0
    d.add_(percdamp * d.mean())
    L = torch.linalg.cholesky(H)
    del H
    Hinv = torch.cholesky_inverse(L)
    del L
    U = torch.linalg.cholesky(Hinv, upper=True)
    del Hinv
    return U


def gptq(W, Hu, bits, group=64, clip=True, colw=None, block=128, meta8=False):
    """W fp32 [N,K] (device), Hu = hinv_chol(H). Returns (q uint8, S, M, rs) (see module docstring)."""
    N, K = W.shape
    qmax = (1 << bits) - 1
    rs = row_scale(W, bits, group) if meta8 else None
    W = W.clone()
    Q = torch.empty(N, K, dtype=torch.uint8, device=W.device)
    S = torch.empty(N, K // group, dtype=_meta_dtype(rs), device=W.device)
    M = torch.empty_like(S)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hb = Hu[i1:i2, i1:i2]
        sf = mf = None
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                g0 = i1 + i
                cw = None if colw is None else colw[g0:g0 + group]
                sf, mf, cs, cm = _params(W1[:, i:i + group], bits, cw, clip, rs)
                S[:, g0 // group] = cs
                M[:, g0 // group] = cm
            w = W1[:, i]
            d = Hb[i, i]
            q, dq = _quant_col(w, sf, mf, qmax)
            Q[:, i1 + i] = q.to(torch.uint8)
            err = (w - dq) / d
            W1[:, i:] -= err[:, None] * Hb[i, i:][None, :]
            Err[:, i] = err
        W[:, i2:] -= Err @ Hu[i1:i2, i2:]
    return Q, S, M, rs


def dequant_g(q, S, M, group, bits=None, rs=None):
    N, K = q.shape
    if rs is None:
        sc, mn = S.float(), M.float()
    else:
        sc, mn = meta8_decode(S, M, rs, bits)
    return (q.view(N, K // group, group).float() * sc[..., None] + mn[..., None]).view(N, K)


def pack_meta(S, M, rs):
    """-> sm array to store: fp16 [N,G,2] (rs None) or uint16 [N,G] = s8 | z8 << 8."""
    if rs is None:
        return torch.stack([S, M], -1)
    return (S.to(torch.int32) | (M.to(torch.int32) << 8))


def out_err(W, Wq, H):
    """E||(W-Wq) x||^2 per token = tr(D H D^T) for H = mean x x^T; also E||W x||^2.  Returns (err, energy)."""
    D = (W - Wq)
    e = ((D @ H) * D).sum().item()
    en = ((W @ H) * W).sum().item()
    return e, en


def hinv_chol_batched(H, percdamp=0.01):
    """H [B,K,K] fp32 -> upper Cholesky factors of the damped inverses [B,K,K] (each as hinv_chol)."""
    H = H.clone()
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    d[d == 0] = 1.0
    d.add_(percdamp * d.mean(-1, keepdim=True))
    L = torch.linalg.cholesky(H)
    del H
    Hinv = torch.cholesky_inverse(L)
    del L
    return torch.linalg.cholesky(Hinv, upper=True)


def rtn_batched(W, bits, group=64, clip=True, colw=None, meta8=True):
    """W [E,N,K] -> (q uint8 [E,N,K], S [E,N,G], M [E,N,G], rs [E,N] | None). colw [E,K] optional per-expert column weights."""
    E, N, K = W.shape
    outs = [rtn(W[e], bits, group, clip=clip, colw=None if colw is None else colw[e], meta8=meta8) for e in range(E)]
    q = torch.stack([o[0] for o in outs]); S = torch.stack([o[1] for o in outs]); M = torch.stack([o[2] for o in outs])
    rs = torch.stack([o[3] for o in outs]) if meta8 else None
    return q, S, M, rs


def gptq_batched(W, Hu, bits, group=64, clip=True, colw=None, block=128, meta8=False):
    """GPTQ over a batch of independent matrices that share one shape: W [E,N,K] (device), Hu = hinv_chol_batched(H) [E,K,K],
    colw [E,K] or None. Same algorithm as gptq() (static groups, params from error-compensated weights); the python loop over
    columns is amortised across E. Returns (Q uint8 [E,N,K], S [E,N,G], M [E,N,G], rs [E,N] | None)."""
    E, N, K = W.shape
    qmax = (1 << bits) - 1
    G = K // group
    rs = torch.stack([row_scale(W[e], bits, group) for e in range(E)]) if meta8 else None  # [E,N]
    rsf = None if rs is None else rs.reshape(E * N)
    W = W.clone()
    Q = torch.empty(E, N, K, dtype=torch.uint8, device=W.device)
    S = torch.empty(E, N, G, dtype=_meta_dtype(rs), device=W.device)
    M = torch.empty_like(S)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, :, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hb = Hu[:, i1:i2, i1:i2]
        sf = mf = None
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                g0 = i1 + i
                cw = None if colw is None else colw[:, g0:g0 + group].repeat_interleave(N, 0)
                a, b, cs, cm = _params(W1[:, :, i:i + group].reshape(E * N, group), bits, cw, clip, rsf)
                sf, mf = a.view(E, N), b.view(E, N)
                S[:, :, g0 // group] = cs.view(E, N)
                M[:, :, g0 // group] = cm.view(E, N)
            w = W1[:, :, i]
            d = Hb[:, i, i][:, None]
            q, dq = _quant_col(w, sf, mf, qmax)
            Q[:, :, i1 + i] = q.to(torch.uint8)
            err = (w - dq) / d
            W1[:, :, i:] -= err[:, :, None] * Hb[:, i, i:][:, None, :]
            Err[:, :, i] = err
        W[:, :, i2:] -= torch.bmm(Err, Hu[:, i1:i2, i2:])
    return Q, S, M, rs
