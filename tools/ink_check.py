#!/usr/bin/env python3 -I
"""Compare a qv_ink dump with a golden npz (tests/golden/inkling or a quantised-reference golden).
  python3 -I tools/ink_check.py DUMP.bin GOLDEN.npz [--json out.json]
Reports: router-id set agreement per application, hidden-state relative L2 error per application (3 positions),
last-position logits (KL, top-1, max |d|), greedy ids vs golden (teacher-free: engine feeds ITS OWN argmax), top-20 mean |dlogit| on golden ids.
"""
import sys, json, numpy as np
def load(p):
    b = open(p, 'rb').read(); a = np.frombuffer(b, dtype='<i4')
    assert a[0] == 0x4b4e4951
    N, G, V, NH, NA = (int(x) for x in a[1:6]); o = 6
    pos = a[o:o+3]; o += 3
    H = 2304
    hid = np.frombuffer(b, '<f4', 3*NH*H, o*4).reshape(3, NH, H); o += 3*NH*H
    rid = np.frombuffer(b, '<i4', N*NA*8, o*4).reshape(N, NA, 8); o += N*NA*8
    lg = np.frombuffer(b, '<f4', V, o*4); o += V
    gen = np.frombuffer(b, '<i4', G, o*4); o += G
    ti = np.frombuffer(b, '<i4', G*20, o*4).reshape(G, 20); o += G*20
    tv = np.frombuffer(b, '<f4', G*20, o*4).reshape(G, 20)
    return dict(N=N, pos=pos, hid=hid, rid=rid, lg=lg, gen=gen, ti=ti, tv=tv)
def kl(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    la = a - a.max(); la -= np.log(np.exp(la).sum()); lb = b - b.max(); lb -= np.log(np.exp(lb).sum())
    return float((np.exp(la) * (la - lb)).sum())
def main():
    d = load(sys.argv[1]); g = np.load(sys.argv[2]); r = {}
    assert d['N'] == len(g['ids'])
    keys = sorted(k for k in g.files if k.startswith('hid_app'))
    rel = np.zeros((len(keys), 3)); agree = []
    for i, k in enumerate(keys):
        ref = g[k]
        for j in range(3):
            rel[i, j] = np.linalg.norm(d['hid'][j, 1 + i] - ref[j]) / (np.linalg.norm(ref[j]) + 1e-12)
        rk = g['router_' + k[4:]]
        agree.append(np.mean([set(map(int, x)) == set(map(int, y)) for x, y in zip(rk, d['rid'][:, i])]))
    r['router_set_agree_mean'] = float(np.mean(agree)); r['router_set_agree_min'] = float(np.min(agree))
    r['hid_rel_mean'] = float(rel.mean()); r['hid_rel_max'] = float(rel.max()); r['hid_rel_final_app'] = float(rel[-1].max())
    r['hid_rel_by_app'] = [round(float(x), 5) for x in rel.max(1)]
    ref = g['logits_last']; r['kl_last'] = kl(ref, d['lg']); r['top1_last'] = int(ref.argmax()) == int(d['lg'].argmax())
    r['max_abs_logit_last'] = float(np.abs(ref - d['lg']).max())
    gi = g['gen_top20_ids']; gv = g['gen_top20_val']
    n = min(len(d['gen']), len(g['gen_ids']))
    r['gen_match_prefix'] = int(next((i for i in range(n) if d['gen'][i] != g['gen_ids'][i]), n)); r['gen_n'] = n
    r['gen_match_total'] = int(sum(d['gen'][i] == g['gen_ids'][i] for i in range(n)))
    # top-20 error at step 0 (only step whose context is identical for both)
    r['top20_mean_abs_step0'] = float(np.abs(d['lg'][gi[0]] - gv[0]).mean())
    print(json.dumps({k: v for k, v in r.items() if k != 'hid_rel_by_app'}, indent=1))
    print('hid_rel_by_app', r['hid_rel_by_app'])
    if '--json' in sys.argv: json.dump(r, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
main()
