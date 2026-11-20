#!/usr/bin/env python3 -I
"""Assemble benchmarks/results/tier_<date>.json: measured inputs (recall study, PCIe/gather bench, CPU cold-attention bench, sparse-logits
probe) + the per-token cost model at 256K. Everything labelled 'model' is arithmetic on the measured inputs, everything labelled
'extrapolation' is a fit beyond the measured context (<= 32K).
usage: tier_model.py RESULTS_DIR OUT.json [--wiki-logits DIR --niah-logits DIR ...]"""
import json, sys, numpy as np

R = sys.argv[1]; out = sys.argv[2]
STEP_MS = 19.0           # GPU decode step (task statement, measured by the GPU engine agents)
N_ATTN = 16
CTX = 262144
rec = {k: json.load(open(f'{R}/{v}')) for k, v in (('niah32k', 'recall_niah32k.json'),)}
import os
if os.path.exists(f'{R}/recall_wiki32k.json'): rec['wiki32k'] = json.load(open(f'{R}/recall_wiki32k.json'))
pcie = json.load(open(f'{R}/tier_pcie_scatter.json'))
cold = json.load(open(f'{R}/cold_cpu.json'))
res = {'date': '2026-11-20', 'host': 'Xeon E5-2683 v4 (64 logical CPUs), V100-SXM2-16GB, PCIe3 x16', 'step_ms_gpu': STEP_MS}

# ---- recall table + power-law extrapolation of the missing mass
def table(j, sel, Rw):
    t = {}
    for C, B in j['buckets'].items():
        for K in (1024, 2048, 4096, 8192):
            v = B.get(f'{sel}|R{Rw}|k{K}')
            if v: t.setdefault(K, {})[int(C)] = {'recall': v['recall_mean_over_layers'], 'err': v['err_mean_over_layers'], 'layers_ge_0.95': v['layers_ge_095'], 'min_layer': v['recall_min_layer']}
    return t
res['recall'] = {}
for name, j in rec.items():
    res['recall'][name] = {f'{sel}|R{Rw}': table(j, sel, Rw) for sel in ('quest_sum', 'oracle') for Rw in (1024, 4096, 16384)}
    res['recall'][name] = {k: v for k, v in res['recall'][name].items() if v}
    res['recall'][name]['last32_question_queries'] = {k: {kk: vv for kk, vv in v.items() if kk != 'layers'} for k, v in j['last32'].items()}
def pooled_fit(j, sel='quest_sum'):
    """EXTRAPOLATION: log(missing mass) = c0 + c1 log(M/K) + c2 log(R), M = cold tokens (n - R - 64), K = fetched tokens, fitted over all
    (context 16K/32K, recent window R, K) points with 0.2% < missing < 100% of the measured study. Crude (log-rmse reported)."""
    rows = []
    for C, B in j['buckets'].items():
        n = int(C)
        if n < 16384: continue
        for name, v in B.items():
            s_, Rw, k = name.split('|')
            if s_ != sel: continue
            Rw = int(Rw[1:]); K = int(k[1:]); M = n - Rw - 64; miss = 1 - v['recall_mean_over_layers']
            if M <= K or miss < 2e-3: continue
            rows.append((n, Rw, K, M, miss))
    r = np.array(rows)
    x = np.column_stack([np.ones(len(r)), np.log(r[:, 3] / r[:, 2]), np.log(r[:, 1])]); y = np.log(r[:, 4])
    c = np.linalg.lstsq(x, y, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((x @ c - y) ** 2)))
    sc = {}
    for Rw in (1024, 16384, 65536, 87000):
        for K in (4096, 8192):
            M = CTX - Rw - 64
            sc[f'R{Rw}_K{K}'] = float(max(0.0, 1 - min(1.0, np.exp(c @ [1, np.log(M / K), np.log(Rw)]))))
    return {'coef': [float(z) for z in c], 'log_rmse': rmse, 'n_points': len(rows), 'predicted_mass_recall_at_256K': sc}
res['extrapolation_to_256K'] = {'note': 'EXTRAPOLATION (fit on 16K/32K context data, evaluated at 256K = 8x beyond the largest measured context): treat as order-of-magnitude only',
                                'by_source': {name: pooled_fit(j) for name, j in rec.items()}}
res['sparse_logits_probe_32K'] = {'note': 'CPU engine, E0 weights, dense 32K prefill, then teacher-forced single-token steps with sparse attention in every attention layer (sink 64 + recent R + top-k 64-token blocks by Quest bound summed over the GQA group). KL(dense||sparse), top-1 agreement, ppl delta over T rows',
                                  'wiki_T64': json.load(open(f'{R}/sparse_logits_wiki32k.json')), 'niah_T32': json.load(open(f'{R}/sparse_logits_niah32k.json')),
                                  'niah_generation': [json.loads(l) for l in open(f'{R}/niah32k_gen.jsonl')]}

# ---- measured transfer / CPU numbers
res['needle'] = rec['niah32k']['needle']
res['pcie'] = {'h2d_gbps_pinned_64MB': 12.44, 'source': 'pcie.txt (bench_pcie)',
               'scatter_gather_kernel_16blocks_gbps': {r['fetch_tokens_per_head']: r['gbps'] for r in pcie['scatter'] if r['method'].startswith('zerocopy') and r['unit_bytes'] == 18432},
               'scatter_memcpyAsync_per_unit_1stream_gbps': {r['fetch_tokens_per_head']: r['gbps'] for r in pcie['scatter'] if r['method'] == 'memcpyAsync' and r['streams'] == 1 and r['unit_bytes'] == 18432},
               'scatter_memcpyAsync_allheads_unit_1stream_gbps': {r['fetch_tokens_per_head']: r['gbps'] for r in pcie['scatter'] if r['method'] == 'memcpyAsync' and r['streams'] == 1 and r['unit_bytes'] == 73728},
               'fetch_ms_per_layer_gather': {r['fetch_tokens_per_head']: r['ms_median'] for r in pcie['scatter'] if r['method'].startswith('zerocopy') and r['unit_bytes'] == 18432},
               'pipeline_standin': pcie['pipeline']}
res['cpu_cold_attention'] = {'full_scan_262144_tokens': cold['full_scan'], 'subset_scattered_blocks': cold['subset']}

# ---- cost model, per decoded token at 256K context
base = {r['fetch_tokens_per_head']: r for r in pcie['pipeline']}
def pipe(copy, K, mode): return next(r['step_ms_median'] for r in pcie['pipeline'] if r['copy'] == copy and r['fetch_tokens_per_head'] == K and r['mode'] == mode)
nofetch = next(r['step_ms_median'] for r in pcie['pipeline'] if r['mode'] == 'no_fetch')
cm = {'basis': 'ms per decoded token at 256K context; GPU step 19 ms; 16 attention layers; fetch = K tokens per kv head per layer (all 4 kv heads), int4 rows of 144 B', 'rows': []}
sel_ms = 16 * (4 * (CTX // 64) * 2 * 256 * 2) / 800e9 * 1e3  # fp16 min/max summaries read once per layer per token at 800 GB/s (calculated)
for K in (1024, 2048, 4096, 8192):
    fetch_mb = 16 * 4 * K * 144 * 2 / 1e6
    ov_abs = pipe('gather_kernel_16blocks', K, 'prefetch_overlapped') - nofetch if K >= 2048 else None
    cr_abs = pipe('gather_kernel_16blocks', K, 'fetch_on_critical_path') - nofetch if K >= 2048 else None
    cpu_ms = next(r['ms_per_layer_median'] for r in cold['subset'] if r['fetch_tokens_per_head'] == K and r['threads'] == 32)
    row = {'fetch_tokens_per_head': K, 'fetched_MB_per_token_all_layers': round(fetch_mb, 1),
           'HB_gather_prefetch_overlapped_step_ms': None if ov_abs is None else round(STEP_MS + ov_abs + sel_ms, 2),
           'HB_gather_prefetch_overhead_pct': None if ov_abs is None else round(100 * (ov_abs + sel_ms) / STEP_MS, 1),
           'HB_gather_on_critical_path_step_ms': None if cr_abs is None else round(STEP_MS + cr_abs + sel_ms, 2),
           'HC_cpu_topk_on_critical_path_step_ms': round(STEP_MS + 16 * (cpu_ms + 0.0165) + sel_ms, 2),
           'selection_kernel_ms_model': round(sel_ms, 2)}
    cm['rows'].append(row)
full = next(r for r in cold['full_scan'] if r['threads'] == 32)
cm['HC_exact_cpu_all_cold_step_ms'] = round(STEP_MS + 16 * full['ms_per_layer_median'], 1)
cm['HC_exact_cpu_all_cold_tok_s'] = round(1e3 / cm['HC_exact_cpu_all_cold_step_ms'], 2)
cm['dense_stream_all_cold_over_pcie_step_ms'] = round(STEP_MS + 16 * (2 * 4 * (CTX - 6144) * 144) / 12.4e9 * 1e3, 1)
cm['host_pool_int4_GiB_256K'] = round(CTX * 16 * 4 * 2 * 144 / 2**30, 2)
cm['host_pool_fp16_GiB_256K'] = round(CTX * 16 * 4 * 2 * 256 * 2 / 2**30, 2)
cm['gpu_block_summaries_fp16_MiB_256K'] = round(16 * 4 * (CTX // 64) * 2 * 256 * 2 / 2**20, 0)
res['cost_model_256K'] = cm
json.dump(res, open(out, 'w'), indent=1)
print('wrote', out)
