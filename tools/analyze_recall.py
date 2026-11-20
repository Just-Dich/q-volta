#!/usr/bin/env python3 -I
"""Offline recall study of sparse block retrieval (Gen 6 H-B) on dumped real K/V/queries (QV_KVDUMP of the CPU engine).

usage: analyze_recall.py DUMP_DIR OUT.json [--layers 0,1,..] [--procs N] [--needle META.json]
For every attention layer, every sampled query position p (all q heads) it computes the exact softmax over keys [0,p] and compares
with attending only to: sink block (64 tokens) + recent window R + the top-k 64-token blocks of the middle chosen per KV head.
Selectors: quest_sum / quest_max (per-block min/max key bound, sum/max over the 6 q heads of the kv group), oracle (true attention mass
per block, summed over the group = ceiling for any block-level selector), mean (query . block-mean-key), stale<j> (quest_sum scored with the
query of position p-j; only for the consecutive last-32 queries) -> what an async prefetch from the previous token(s) would see.
Metrics per (layer, query, q head): mass recall = softmax mass inside the selected set, relative L2 error of the attention output
of the renormalised sparse softmax vs exact. Aggregated per context bucket."""
import json, sys, numpy as np
from multiprocessing import Pool

args = sys.argv[1:]
d, outp = args[0], args[1]
opt = lambda n, dflt=None: args[args.index(n) + 1] if n in args else dflt
meta = dict(kv.split('=') for kv in open(d + '/meta.txt').read().split())
N, NL, HKV, HQ, D, NQ = (int(meta[k]) for k in ('n', 'layers', 'hkv', 'hq', 'd', 'nq'))
G = HQ // HKV
BS = 64
KT = [1024, 2048, 4096, 8192]
RECENTS = [1024, 4096, 16384]
qpos = np.fromfile(d + '/qpos.i32', dtype=np.int32)
nsamp = len(qpos) - 32  # strided samples first, then last 32 consecutive
LAST = np.arange(nsamp, nsamp + 32)
BUCKETS = [c for c in (2048, 4096, 8192, 16384, 32768, 65536, 131072) if c <= N + 1024]  # bucket C = queries with pos in [C-1024, C) (clipped to the dump)
needle = json.load(open(opt('--needle'))) if opt('--needle') else None

def rstart_of(p, R):  # first token of the always-resident recent window (block aligned), keys [0,p]
    return ((p + 1 - R) // BS) * BS

def run_layer(l):
    K = np.fromfile(f'{d}/k_{l}.f16', dtype=np.float16).reshape(HKV, N, D)
    V = np.fromfile(f'{d}/v_{l}.f16', dtype=np.float16).reshape(HKV, N, D)
    Q = np.fromfile(f'{d}/q_{l}.f16', dtype=np.float16).reshape(NQ, HQ, D).astype(np.float32)
    nbl = N // BS
    Kb = K[:, :nbl * BS].astype(np.float32).reshape(HKV, nbl, BS, D)
    kmin, kmax, kmean = Kb.min(2), Kb.max(2), Kb.mean(2)  # [HKV, nbl, D]
    del Kb
    res = {}  # key -> array [NQ, HQ]
    names = []
    for R in RECENTS:
        for sel in ('quest_sum', 'quest_max', 'oracle', 'mean'):
            if R != 1024 and sel not in ('quest_sum', 'oracle'): continue
            for k in KT: names.append((sel, R, k))
    for j in (1, 4):
        for k in KT: names.append((f'stale{j}', 1024, k))
    rec = {n: np.full((NQ, HQ), np.nan, np.float32) for n in names}
    err = {n: np.full((NQ, HQ), np.nan, np.float32) for n in names}
    needle_mass = np.full((32, HQ), np.nan, np.float32)
    needle_in = {k: np.full((32, HKV), np.nan, np.float32) for k in KT}
    for qi in range(NQ):
        p = int(qpos[qi]); n = p + 1
        # exact attention
        P = np.empty((HQ, n), np.float32); O = np.empty((HQ, D), np.float32)
        for g in range(HKV):
            s = (Q[qi, g * G:(g + 1) * G] @ K[g, :n].astype(np.float32).T) / np.sqrt(D)
            s -= s.max(1, keepdims=True); e = np.exp(s); P[g * G:(g + 1) * G] = e / e.sum(1, keepdims=True)
        Vf = [V[g, :n].astype(np.float32) for g in range(HKV)]
        for g in range(HKV): O[g * G:(g + 1) * G] = P[g * G:(g + 1) * G] @ Vf[g]
        Pb_all = P[:, :nbl * BS].reshape(HQ, nbl, BS).sum(2) if n >= nbl * BS else None
        if Pb_all is None:
            tmp = np.zeros((HQ, nbl * BS), np.float32); tmp[:, :n] = P; Pb_all = tmp.reshape(HQ, nbl, BS).sum(2)
        if qi in LAST and needle is not None:
            ns, nl = needle['needle_start'], needle['needle_len']
            needle_mass[qi - nsamp] = P[:, ns:ns + nl].sum(1)
        for (sel, R, k) in names:
            if sel.startswith('stale'):
                jlag = int(sel[5:]); idx = qi - jlag
                if qi not in LAST or idx < nsamp or qpos[idx] != p - jlag: continue
                qsel = Q[idx]
            else: qsel = Q[qi]
            rs = rstart_of(p, R); nb = rs // BS - 1  # middle blocks 1 .. rs/BS-1
            if rs <= BS or nb <= k // BS:
                rec[(sel, R, k)][qi] = 1.0; err[(sel, R, k)][qi] = 0.0; continue
            kb = k // BS
            mask = np.zeros((HQ, n), bool); mask[:, :BS] = True; mask[:, rs:] = True
            for g in range(HKV):
                qg = qsel[g * G:(g + 1) * G]  # [G, D]
                if sel in ('quest_sum', 'quest_max') or sel.startswith('stale'):
                    sc = np.maximum(qg[:, None, :] * kmin[g, 1:rs // BS][None], qg[:, None, :] * kmax[g, 1:rs // BS][None]).sum(2)
                    sc = sc.max(0) if sel == 'quest_max' else sc.sum(0)
                elif sel == 'mean':
                    sc = (qg @ kmean[g, 1:rs // BS].T).sum(0)
                else:  # oracle
                    sc = Pb_all[g * G:(g + 1) * G, 1:rs // BS].sum(0)
                top = np.argpartition(-sc, kb - 1)[:kb] + 1
                for b in top: mask[g * G:(g + 1) * G, b * BS:(b + 1) * BS] = True
                if needle is not None and qi in LAST and sel == 'quest_sum' and R == 1024:
                    needle_in[k][qi - nsamp, g] = float(needle['needle_start'] // BS in top)
            Pm = P * mask
            rec[(sel, R, k)][qi] = Pm.sum(1)
            Os = np.empty_like(O)
            for g in range(HKV): Os[g * G:(g + 1) * G] = (Pm[g * G:(g + 1) * G] / Pm[g * G:(g + 1) * G].sum(1, keepdims=True)) @ Vf[g]
            err[(sel, R, k)][qi] = np.linalg.norm(Os - O, axis=1) / np.linalg.norm(O, axis=1)
    return l, rec, err, needle_mass, needle_in

if __name__ == '__main__':
    layers = [int(x) for x in opt('--layers').split(',')] if opt('--layers') else list(range(NL))
    with Pool(int(opt('--procs', 8))) as pool:
        outs = pool.map(run_layer, layers, chunksize=1)
    pos = qpos[:nsamp]
    out = {'n_tokens': N, 'layers': layers, 'buckets': {}, 'last32': {}, 'notes': 'recall = softmax mass in selected set; err = rel L2 of attention output'}
    def agg(a):  # a [nq, HQ] subset -> mean over queries/heads, mean of per-head worst 10%, min
        a = a[~np.isnan(a).all(1)]
        if len(a) == 0: return None
        ph = np.nanmean(a, 0)
        return {'mean': float(np.nanmean(a)), 'p10_head': float(np.percentile(ph, 10)), 'min_head': float(ph.min()), 'nq': int(len(a))}
    for C in BUCKETS:
        sel_q = np.where((pos >= C - 1024) & (pos < C))[0]
        if sel_q is None or len(sel_q) == 0: continue
        B = {}
        for (l, rec, err, nm, ni) in outs:
            for key in rec:
                nme = f'{key[0]}|R{key[1]}|k{key[2]}'
                if key[0].startswith('stale'): continue
                B.setdefault(nme, {}).setdefault('layers', {})[l] = {'recall': agg(rec[key][sel_q]), 'err': agg(err[key][sel_q])}
        for nme, v in B.items():
            Ls = v['layers']
            v['recall_mean_over_layers'] = float(np.mean([Ls[l]['recall']['mean'] for l in Ls]))
            v['recall_min_layer'] = float(min(Ls[l]['recall']['mean'] for l in Ls))
            v['recall_p10head_mean_over_layers'] = float(np.mean([Ls[l]['recall']['p10_head'] for l in Ls]))
            v['err_mean_over_layers'] = float(np.mean([Ls[l]['err']['mean'] for l in Ls]))
            v['layers_ge_095'] = int(sum(Ls[l]['recall']['mean'] >= 0.95 for l in Ls))
        out['buckets'][str(C)] = B
    L32 = {}
    for (l, rec, err, nm, ni) in outs:
        for key in rec:
            nme = f'{key[0]}|R{key[1]}|k{key[2]}'
            if key[0] in ('quest_sum', 'oracle') or key[0].startswith('stale'):
                L32.setdefault(nme, {}).setdefault('layers', {})[l] = {'recall': agg(rec[key][LAST]), 'err': agg(err[key][LAST])}
    for nme, v in L32.items():
        Ls = {l: x for l, x in v['layers'].items() if x['recall']}
        if Ls:
            v['recall_mean_over_layers'] = float(np.mean([Ls[l]['recall']['mean'] for l in Ls]))
            v['err_mean_over_layers'] = float(np.mean([Ls[l]['err']['mean'] for l in Ls]))
            v['layers_ge_095'] = int(sum(Ls[l]['recall']['mean'] >= 0.95 for l in Ls))
    out['last32'] = L32
    if needle is not None:
        out['needle'] = {'meta': needle, 'mass_per_layer_mean_over_heads_and_queries': {str(l): float(np.nanmean(nm)) for (l, _, _, nm, _) in outs},
                         'mass_per_layer_max_head': {str(l): float(np.nanmax(np.nanmean(nm, 0))) for (l, _, _, nm, _) in outs},
                         'needle_block_selected_quest_sum_R1024': {str(k): {str(l): float(np.nanmean(ni[k])) for (l, _, _, _, ni) in outs} for k in KT}}
    np.savez_compressed(outp.replace('.json', '.npz'), qpos=qpos, **{f'L{l}|{k[0]}|R{k[1]}|k{k[2]}|{w}': arr[k].astype(np.float16) for (l, rec, err, _, _) in outs for w, arr in (('rec', rec), ('err', err)) for k in arr})
    json.dump(out, open(outp, 'w'), indent=1)
    print('wrote', outp)
