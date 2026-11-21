#!/usr/bin/env python3 -I
"""Arithmetic model behind docs/INKLING_DESIGN.md (Inkling-2-Mini feasibility). No GPU/CPU is touched.
Inputs are (a) shapes from config.json + the safetensors header (exact), (b) MEASURED rates quoted from docs/PERFORMANCE.md /
docs/LONG_CONTEXT.md / docs/CPU_OPTIMIZATION.md (marked M below), (c) explicitly labelled assumptions (A).
usage: python3 -I tools/ink_model.py            (prints every table used in the doc)"""
import math

# ---------------------------------------------------------------- exact shapes (config + safetensors header)
H, V, VP = 2304, 201024, 200058          # hidden, padded vocab (tensor rows), unpadded vocab
NL, CORE, PASSES = 24, 12, 3             # layers, looped layers (6..17), passes
NAPP = NL + CORE * (PASSES - 1)          # 48 layer applications / token
NE, TOPK, EI = 128, 8, 1152
QH, KVH, HD = 24, 8, 128
ATTN_PL = 3 * QH * HD * H + 2 * KVH * HD * H + (QH * 16) * H + QH * H   # wq,gate,wo ; wk,wv ; wr ; sink
EXP = H * 2 * EI + EI * H                # per expert: w13 [2304,2304] + w2 [2304,1152]
CTRL_PL = 4 * (64 * 2 * H + 2 * H * 64)  # per core layer: 4 controllers (attn,k,v,mlp)
ROUTER = NE * H
EMB = V * H
MTP = 635795744 // 2                     # params (bf16 file)
GLOBAL, LOCAL = [5, 11, 17, 23], [i for i in range(24) if i not in (5, 11, 17, 23)]
G_APPS = 2 + 2 * PASSES                  # global layer applications: layers 5,23 once, 11,17 x3 = 8
L_APPS = NAPP - G_APPS                   # 40
# ---------------------------------------------------------------- measured constants (M)
HBM_PEAK = 890e9          # M: measured HBM read peak (CLAUDE.md)
GEMV_ETA_MID = 720 / 890  # M: gdn_z 4b 15.7 MB at 720 GB/s (logs/gemv_sweep.txt)
GEMV_ETA_BIG = 850 / 890  # M: lm_head 6b 0.95 GB at 850 GB/s
GEMV_SMALL_US = 5.9       # M: attn_kv 4b 1024x5120 (2.95 MB) best config 5.9 us (logs/gemv_sweep.txt)
PCIE = 12.4e9             # M: pinned H2D
DRAM = 54e9               # M: CPU sustained read (CPU_OPTIMIZATION.md)
TC_ATTN_TFLOPS = 27e12    # M: tensor-core prefill attention 26.4-29.5 TFLOP/s at head_dim 256
GEMM_TFLOPS = 77e12       # M: cuBLAS HGEMM share of Qwen prefill (PERFORMANCE.md)
DEQ_BW = 751e9            # M: dequant_rows_f16 (read+write bytes / time)
ATTN_NS_PER_KEY = 1.52e-3 / 262144 * 1e9   # M-derived: (1/22.9 - 1/51.7) s / 16 layers / 262,001 keys = 5.8 ns/key/layer (Qwen e1, int4 KV)
CPU_ATTN_MS_256K = 10.4   # M: CPU int4 attention, one layer, 262,144 tokens, 32 threads
VRAM_GIB = 15.77


def bpw(bits, group, meta8):
    return bits + (16.0 / group if meta8 else 32.0 / group)   # meta8 = 8-bit scale + 8-bit zero per group


def gib(x): return x / 2**30


def rep_budget():
    print("== 1. parameter budget (text model, vision excluded) ==")
    rows = [("embed (lookup, 1 row/token)", EMB), ("unembed / lm_head", EMB), ("attention (24 layers)", ATTN_PL * NL),
            ("routed experts (24 x 128)", EXP * NE * NL), ("router (24 x 128x2304)", ROUTER * NL),
            ("loop controllers (12 core layers)", CTRL_PL * CORE), ("MTP drafter (3 layers + markov head)", MTP)]
    tot = 0
    for n, p in rows:
        tot += p
        print(f"{n:42s} {p/1e6:10.1f} M  bf16 {p*2/1e9:7.3f} GB")
    print(f"{'total (rows above)':42s} {tot/1e6:10.1f} M")
    print(f"per expert {EXP/1e6:.3f} M params; per layer experts {EXP*NE/1e9:.4f} B; attention per layer {ATTN_PL/1e6:.3f} M")
    print(f"unique active/token: attn {ATTN_PL*NL/1e9:.3f} + experts {EXP*TOPK*NL/1e9:.3f} + head {EMB/1e9:.3f} = "
          f"{(ATTN_PL*NL+EXP*TOPK*NL+EMB)/1e9:.2f} B (+embed row); extra per pass {(ATTN_PL+EXP*TOPK)*CORE/1e9:.3f} B")
    print()
    print("Q-Volta size of experts / attention / head by (bits, scheme), GB (1e9):")
    for nm, g, m8 in (("g64 fp16 meta", 64, False), ("g128 fp16 meta", 128, False), ("g64 meta8", 64, True), ("g128 meta8", 128, True)):
        line = f"{nm:15s}"
        for b in (3, 4, 5):
            w = bpw(b, g, m8)
            line += f" | {b}b ({w:.3f} bpw): exp {EXP*NE*NL*w/8/1e9:6.2f} attn {ATTN_PL*NL*w/8/1e9:5.2f} head {EMB*w/8/1e9:5.2f}"
        print(line)
    print("checkpoint as shipped: experts NVFP4 (layers 0-19) 4.5 bpw = %.2f GB, MXFP8 (layers 20-23) 8.25 bpw = %.2f GB, both = %.2f GB"
          % (EXP*NE*20*4.5/8/1e9, EXP*NE*4*8.25/8/1e9, (EXP*NE*20*4.5+EXP*NE*4*8.25)/8/1e9))


PROFILES = {  # name: (expert bpw, attention bpw, head bpw, description)
    "Q3  (e3/a4/h5, g128 m8)": (bpw(3, 128, True), bpw(4, 128, True), bpw(5, 128, True)),
    "Q4  (e4/a5/h6, g128 m8)": (bpw(4, 128, True), bpw(5, 128, True), bpw(6, 128, True)),
    "FP4 (NVFP4 native experts, 4 fp8 layers -> 4.5)": (4.5, bpw(5, 128, True), bpw(6, 128, True)),
    "Q5  (e5/a6/h6, g128 m8)": (bpw(5, 128, True), bpw(6, 128, True), bpw(6, 128, True)),
}
ROUTER_B = ROUTER * NL * 2    # fp16
CTRL_B = CTRL_PL * CORE * 1.0  # 8-bit controllers


def resident_bytes(p):
    e, a, h = p
    return dict(experts=EXP * NE * NL * e / 8, attn=ATTN_PL * NL * a / 8, head=VP * H * h / 8, router=ROUTER_B, ctrl=CTRL_B,
                mtp=MTP * 4.5 / 8)


def rep_resident():
    print("== 1b. VRAM-resident weights (embed on host, MTP at 4.5 bpw) ==")
    for n, p in PROFILES.items():
        r = resident_bytes(p)
        core = r["experts"] * CORE / NL
        print(f"{n:50s} experts {r['experts']/1e9:6.2f} attn {r['attn']/1e9:5.2f} head {r['head']/1e9:5.2f} router {r['router']/1e9:5.3f} "
              f"ctrl {r['ctrl']/1e9:5.3f} mtp {r['mtp']/1e9:5.3f} => {sum(r.values())/1e9:6.2f} GB = {gib(sum(r.values())):5.2f} GiB (core-layer experts only {core/1e9:5.2f} GB)")


def tok_read_bytes(p, distinct_experts_per_app=TOPK, rows=1):
    e, a, h = p
    ex = distinct_experts_per_app * EXP * NAPP * e / 8
    at = ATTN_PL * NAPP * a / 8
    hd = VP * H * h / 8
    return ex, at, hd, ROUTER * NAPP * 2, CTRL_PL * CORE * PASSES * 1.0


def rep_decode():
    print("== 2. decode, 1 token ==")
    print(f"expert bytes/app (8 experts) at 3.125/4.125/4.5 bpw: {[round(TOPK*EXP*b/8/1e6,1) for b in (3.125,4.125,4.5)]} MB; w13 8 experts {[round(TOPK*H*2*EI*H/H*0+TOPK*5308416*b/8/1e6,1) for b in (3.125,4.125,4.5)]} MB, w2 {[round(TOPK*2654208*b/8/1e6,1) for b in (3.125,4.125,4.5)]} MB")
    n_k = 14 * NAPP + 8      # ASSUMED ~14 dependent kernels per layer application + head/sampling
    for n, p in PROFILES.items():
        ex, at, hd, rt, ct = tok_read_bytes(p)
        tot = ex + at + hd + rt + ct
        t_bw100 = tot / HBM_PEAK
        t_mid = tot / (GEMV_ETA_MID * HBM_PEAK)
        # model: big kernels (head) at 0.955, others at 0.81 + fixed 2.6 us / kernel (5.9 us measured for a 2.95 MB GEMV minus 2.95MB/890GB/s)
        t = (hd / (GEMV_ETA_BIG * HBM_PEAK)) + (tot - hd) / (GEMV_ETA_MID * HBM_PEAK)
        ovh = n_k * 2.6e-6
        print(f"{n:50s} read/token: exp {ex/1e9:5.3f} attn {at/1e9:5.3f} head {hd/1e9:5.3f} rt+ctrl {(rt+ct)/1e9:5.3f} = {tot/1e9:5.3f} GB | "
              f"100%@890: {t_bw100*1e3:5.2f} ms ({1/t_bw100:5.0f} tok/s) | eta {GEMV_ETA_MID:.2f}: {t*1e3:5.2f} ms ({1/t:4.0f}) | +{n_k} kernels x2.6us: {(t+ovh)*1e3:5.2f} ms ({1/(t+ovh):4.0f} tok/s)")
        cpu = tot / DRAM
        print(f"{'':50s} CPU @54 GB/s: {cpu*1e3:6.1f} ms ({1/cpu:5.1f} tok/s ceiling); at 85% {1/(cpu/0.85):5.1f}, at 70% {1/(cpu/0.70):5.1f} tok/s")
    print()
    print("expert kernel shapes (3.125 bpw): w13 8x[2304x2304] = %.1f MB, w2 8x[2304x1152] = %.1f MB; single expert w13 %.2f MB"
          % (TOPK*5308416*3.125/8/1e6, TOPK*2654208*3.125/8/1e6, 5308416*3.125/8/1e6))
    for lab, mb in (("w13 grouped", TOPK*5308416*3.125/8/1e6), ("w2 grouped", TOPK*2654208*3.125/8/1e6), ("w13 single expert", 5308416*3.125/8/1e6), ("w2 single expert", 2654208*3.125/8/1e6)):
        by = mb * 1e6
        t_floor = GEMV_SMALL_US * 1e-6 - 2.95e6 / HBM_PEAK   # fixed part
        t_est = t_floor + by / (0.81 * HBM_PEAK)
        print(f"  {lab:18s} {mb:5.1f} MB: est {t_est*1e6:5.1f} us (fixed {t_floor*1e6:.1f} us + bytes at eta .81) -> {by/t_est/1e9:5.0f} GB/s effective")
    print("  8 separate launches per matrix instead of 1 grouped: w13 %.0f us vs %.0f us" % (8*(t_floor+5308416*3.125/8/(0.81*HBM_PEAK))*1e6, (t_floor+TOPK*5308416*3.125/8/(0.81*HBM_PEAK))*1e6))


def distinct(T, k=TOPK, n=NE):
    return n * (1 - (1 - k / n) ** T)


def rep_mtp():
    print("== 5. MTP verification with MoE (uniform-routing upper bound for distinct experts) ==")
    p = PROFILES["Q3  (e3/a4/h5, g128 m8)"]
    print("distinct experts per layer for T verified tokens (uniform independent top-8 of 128):", {T: round(distinct(T), 1) for T in (1, 2, 3, 4, 5, 8, 9)})
    e, a, h = p

    def t_step(T, share=1.0):
        d = min(NE, 8 + (distinct(T) - 8) * share)     # share=1: independent routing; share<1: positively correlated (fewer distinct)
        by = d * EXP * NAPP * e / 8 + ATTN_PL * NAPP * a / 8 + VP * H * h / 8 + ROUTER * NAPP * 2 + CTRL_PL * CORE * PASSES
        return by / (GEMV_ETA_MID * HBM_PEAK) + (14 * NAPP + 8) * 2.6e-6 * (1 + 0.05 * (T - 1))   # +5%/row kernel time for T rows (Qwen: multi-token GEMV ~9% slower per call)
    t1 = t_step(1)
    # MTP draft: 3 dense layers (87.3M params each) + markov table + head (5.125 bpw) one block pass; attention part over ctx small
    draft_b = (MTP - V * 256) * 4.5 / 8 + V * 256 * 4.5 / 8 + VP * H * h / 8
    t_draft = draft_b / (GEMV_ETA_MID * HBM_PEAK) + 40 * 2.6e-6
    print(f"plain step (model) {t1*1e3:.2f} ms; draft pass (3 dense layers + head, one block) {t_draft*1e3:.2f} ms (A: 40 kernels)")
    print("verify step time (ms) / ratio to plain, by tokens verified T (independent routing | half the extra experts shared):")
    for T in (2, 3, 4, 5, 6, 9):
        print(f"  T={T}: {t_step(T)*1e3:6.2f} ms = {t_step(T)/t1:4.2f}x | {t_step(T,0.5)*1e3:6.2f} ms = {t_step(T,0.5)/t1:4.2f}x   (Qwen dense measured 1.11x/1.14x/1.25x for 2/3/4)")
    print("speedup = tokens/cycle * t_plain / (t_draft + t_verify(D+1)), tokens/cycle = 1 + sum_{i<=D} alpha^i  (geometric acceptance, A):")
    for share in (1.0, 0.5):
        print(f"  routing overlap factor {share} ({'independent' if share==1 else 'half the extra experts shared'}):")
        print("   alpha |" + "".join(f"  D={D}  " for D in (1, 2, 3, 4, 6, 8)))
        for al in (0.5, 0.6, 0.75, 0.9, 0.95):
            line = f"   {al:4.2f}  |"
            for D in (1, 2, 3, 4, 6, 8):
                tpc = 1 + sum(al ** i for i in range(1, D + 1))
                line += f" {tpc*t1/(t_draft+t_step(D+1,share)):5.2f}x"
            print(line)

    print("MTP at long context (attention K/V read once per verify pass, rows share it; row compute +5%/row, A; draft layer adds 1 global app of KV):")
    for ctx in (4096, 131072, 262144, 478000):
        att1 = G_APPS * ctx * ATTN_NS_PER_KEY * 1e-9            # M-derived kernel cost, 1 row
        dr_att = 1 * ctx * ATTN_NS_PER_KEY * 1e-9
        line = f"  ctx {ctx//1024:4d}K plain {(t1+att1)*1e3:5.1f} ms ({1/(t1+att1):5.1f} tok/s):"
        for al, D in ((0.75, 3), (0.9, 3), (0.9, 6)):
            tpc = 1 + sum(al ** i for i in range(1, D + 1))
            cyc = t_draft + dr_att + t_step(D + 1, 1.0) + att1 * (1 + 0.05 * D)
            line += f" a={al} D={D}: {tpc*(t1+att1)/cyc:4.2f}x"
        print(line)
    # CPU: bytes scale with distinct experts, nothing else is cheap
    print("CPU (DRAM-bound, same byte model, 54 GB/s, eta .85):")
    def tc(T):
        d = distinct(T)
        by = d * EXP * NAPP * e / 8 + ATTN_PL * NAPP * a / 8 + VP * H * h / 8
        return by / (0.85 * DRAM)
    for T in (1, 2, 3, 4):
        print(f"  T={T}: {tc(T)*1e3:6.1f} ms = {tc(T)/tc(1):4.2f}x plain")
    for al in (0.6, 0.75, 0.9):
        line = f"   alpha {al:4.2f} |"
        for D in (1, 2, 3):
            tpc = 1 + sum(al ** i for i in range(1, D + 1))
            tdr = draft_b / (0.85 * DRAM)
            line += f" D={D}: {tpc*tc(1)/(tdr+tc(D+1)):4.2f}x"
        print(line)


def rep_prefill():
    print("== 3. prefill ==")
    p = PROFILES["Q3  (e3/a4/h5, g128 m8)"]
    e, a, h = p
    fl_attn_w = 2 * ATTN_PL * NAPP
    fl_exp = 2 * EXP * TOPK * NAPP
    print(f"GEMM FLOPs/token: attention weights {fl_attn_w/1e9:.2f} G, experts {fl_exp/1e9:.2f} G, total {(fl_attn_w+fl_exp)/1e9:.2f} GFLOP (+router/controllers ~0.2 G)")
    deq_exp_app = EXP * NE * (e / 8 + 2) / DEQ_BW   # read quantized + write fp16 for ALL 128 experts of one layer application
    deq_attn_app = ATTN_PL * (a / 8 + 2) / DEQ_BW
    t_deq = NAPP * (deq_exp_app + deq_attn_app)
    print(f"dequant of all 128 experts per layer application: {deq_exp_app*1e3:.2f} ms; whole token-chunk pass (48 apps, + attention weights {NAPP*deq_attn_app*1e3:.1f} ms): {t_deq*1e3:.0f} ms per CHUNK")
    for eff_e in (77e12, 40e12, 25e12):
        t_tok = fl_attn_w / GEMM_TFLOPS + fl_exp / eff_e
        print(f" expert-GEMM efficiency {eff_e/1e12:.0f} TFLOP/s (attention GEMMs 77): per-token GEMM time {t_tok*1e6:.0f} us ->")
        line = "   chunk C: "
        for C in (512, 1024, 2048, 4096, 8192, 16384):
            tt = t_deq + C * t_tok
            line += f"{C}: {C/tt:5.0f} tok/s (deq {t_deq/tt*100:2.0f}%) | "
        print(line, f"\n   fused dequant-in-GEMM (no slab pass), asymptote {1/t_tok:5.0f} tok/s")
        print(f"   chunk for dequant <= 10% of time: C >= {t_deq/(0.1/0.9)/t_tok:.0f} tokens")
    print("prefill attention cost (global layer applications, 6144*T^2 FLOP each, tensor-core kernel):")
    for apps, nm in ((G_APPS, "KV per pass (8 apps)"), (G_APPS - 4, "KV shared across passes (4 apps)")):
        for TF in (27e12, 18e12):
            line = f" {nm}, {TF/1e12:.0f} TFLOP/s: "
            for T in (32768, 131072, 262144, 524288, 1048576):
                line += f"{T//1024}K {apps*6144*T*T/TF:7.0f}s | "
            print(line)
    loc = 4 * QH * HD * 1024 * L_APPS
    print(f" local layers (window 1024): {loc/1e9:.2f} GFLOP/token vs GEMM {(fl_attn_w+fl_exp)/1e9:.1f} GFLOP/token")
    print("whole-prompt prefill estimate (GEMM at 40 TFLOP/s experts, chunk 4096 dequant, + global attention 27 TFLOP/s, 8 apps):")
    t_tok = fl_attn_w / GEMM_TFLOPS + fl_exp / 40e12
    for T in (32768, 131072, 262144, 524288, 1048576):
        g = T * t_tok + (T / 4096) * t_deq
        at = G_APPS * 6144 * T * T / TC_ATTN_TFLOPS
        loc_t = T * loc / TC_ATTN_TFLOPS
        tot = g + at + loc_t
        print(f"  T={T//1024:5d}K: GEMM+dequant {g:7.0f}s, global attn {at:7.0f}s, local attn {loc_t:5.0f}s => {tot:7.0f}s = {T/tot:5.0f} tok/s average")


def rep_context():
    print("== 4. KV and context ==")
    el = 2 * KVH * HD
    for nm, bits in (("fp16", 16.0), ("int8 (+fp16 s/z per 64)", 8.5), ("int4 (kv4 record, 4.5 bit)", 4.5)):
        print(f"{nm:26s} per app-token {el*bits/8:6.0f} B | 8 apps: {8*el*bits/8/1024:6.2f} KiB/token | 4 apps: {4*el*bits/8/1024:5.2f} KiB/token | "
              + " ".join(f"{T//1024}K: {8*el*bits/8*T/2**30:5.2f}/{4*el*bits/8*T/2**30:5.2f} GiB" for T in (131072, 262144, 524288, 1048576)))
    ring = L_APPS * 1024 * el * 2
    print(f"local rings fp16: {L_APPS} apps x 1024 x {el*2} B = {ring/2**20:.0f} MiB (constant); sconv/state negligible")
    mtp_kv = el * 4.5 / 8
    print(f"MTP drafter global layer context KV (int4): +{mtp_kv:.0f} B/token (A: k_proj_ctx/v_proj_ctx of one global layer)")
    ovh = 1.2    # GiB, A: 0.72 measured on Qwen (CUDA ctx + scratch) + ~0.5 MoE/prefill/MTP snapshot buffers
    print(f"VRAM {VRAM_GIB} GiB - overhead {ovh} GiB (A) - rings {ring/2**30:.2f} GiB; context at int4 with MTP layer (9 apps | 5 apps shared):")
    for n, p in PROFILES.items():
        r = sum(resident_bytes(p).values())
        free = VRAM_GIB - gib(r) - ovh - ring / 2**30
        if free <= 0:
            print(f"  {n:50s} weights {gib(r):5.2f} GiB: does not fit"); continue
        c9 = free * 2**30 / (9 * el * 4.5 / 8); c5 = free * 2**30 / (5 * el * 4.5 / 8)
        c9f = free * 2**30 / (9 * el * 16 / 8)
        print(f"  {n:50s} weights {gib(r):5.2f} GiB, free for KV {free:5.2f} GiB -> {c9/1024:6.0f}K ctx (per-pass KV) | {c5/1024:6.0f}K (shared) | fp16 KV: {c9f/1024:5.0f}K")
    # hybrid: core experts on GPU, non-core on host
    print("hybrid residency (experts of the 12 non-core layers in host RAM, computed by the CPU):")
    for n, p in PROFILES.items():
        r = resident_bytes(p); r["experts"] = r["experts"] * CORE / NL
        w = sum(r.values())
        free = VRAM_GIB - gib(w) - ovh - ring / 2**30
        c9 = free * 2**30 / (9 * el * 4.5 / 8)
        host = resident_bytes(p)["experts"] * (NL - CORE) / NL
        print(f"  {n:50s} GPU weights {gib(w):5.2f} GiB, free {free:5.2f} GiB -> {c9/1024:6.0f}K ctx int4 (9 apps); host-side experts {host/1e9:5.2f} GB")
    print("decode attention time per token (global apps x ctx x ns/key): measured-derived 5.8 ns/key/app vs bandwidth-bound 1152 B @ 800 GB/s = 1.44 ns/key/app")
    for T in (32768, 131072, 262144, 524288, 1048576):
        for apps in (8, 4):
            print(f"  ctx {T//1024:5d}K, {apps} apps: kernel as measured {apps*T*ATTN_NS_PER_KEY*1e-6:7.1f} ms | bandwidth-bound {apps*T*1152/800e9*1e3:6.1f} ms | CPU exact int4 (10.4 ms/256K/app, linear) {apps*T/262144*CPU_ATTN_MS_256K:7.1f} ms")
    print("host tier (H-B) at 1M: fetch K tokens/kv head/app: bytes/token and PCIe time")
    for K in (2048, 4096, 8192):
        b = 8 * K * 8 * 1152 / 8 * 1  # 8 apps x (K tokens x 8 kv heads) ; record is 1152 B per token for all 8 heads => per head 144 B
        b = 8 * K * 1152
        print(f"  K={K}: {b/1e6:6.1f} MB/token = {b/PCIE*1e3:5.2f} ms of PCIe at 12.4 GB/s (8 apps; Qwen doc: 16 apps 75 MB at K=4K)")
    print("summary summaries for selection at 1M: min/max fp16 per (app, kv head, 64-token block): "
          f"{8*(1048576//64)*8*2*HD*2/2**20:.0f} MiB (fp16) / {8*(1048576//64)*8*2*HD/2**20:.0f} MiB (int8)")


def rep_residency():
    print("== 6. expert residency (weights > VRAM) ==")
    e = 4.5
    per_expert = EXP * e / 8
    print(f"one expert (w13+w2) at {e} bpw = {per_expert/1e6:.2f} MB; PCIe {PCIE/1e9:.1f} GB/s -> {per_expert/PCIE*1e3:.3f} ms per expert fetch")
    print(f"token touches {TOPK*NAPP} expert slots (8 x 48); if every slot missed: {TOPK*NAPP*per_expert/1e9:.2f} GB = {TOPK*NAPP*per_expert/PCIE*1e3:.0f} ms over PCIe")
    for hit in (0.5, 0.7, 0.8, 0.9, 0.95):
        mb = (1 - hit) * TOPK * NAPP * per_expert
        print(f"  hit rate {hit:4.2f}: miss traffic {mb/1e6:7.1f} MB/token -> PCIe {mb/PCIE*1e3:6.1f} ms (+ GPU compute ~5 ms, overlap <= prefetch) | CPU computes the misses instead (DRAM 54 GB/s): {mb/DRAM*1e3:5.1f} ms")
    nc = TOPK * (NL - CORE) * per_expert
    print(f"core-resident / non-core on host: non-core slots/token = {TOPK*(NL-CORE)}, {nc/1e6:.0f} MB (NVFP4 4.5 bpw) -> PCIe {nc/PCIE*1e3:.1f} ms | CPU {nc/DRAM*1e3:.1f} ms (+ 12 x ~35 us sync round trips)")
    x = EXP * (NL - CORE) * NE
    print(f"  host-resident non-core experts: {x*4.5/8/1e9:.2f} GB at 4.5 bpw; GPU-resident core experts {EXP*NE*CORE*4.5/8/1e9:.2f} GB at 4.5 bpw")
    uni = lambda cap_frac: cap_frac
    print("uniform-routing hit rate = resident fraction of experts (LRU on skewed routing is better, unmeasured):")
    for gb in (4, 6, 8, 10, 12):
        tot = EXP * NE * NL * 4.5 / 8 / 1e9
        print(f"  {gb:2d} GB expert cache of {tot:.1f} GB: hit {min(1,gb/tot):.2f} -> miss time {(1-min(1,gb/tot))*TOPK*NAPP*per_expert/PCIE*1e3:6.1f} ms/token")


if __name__ == "__main__":
    for f in (rep_budget, rep_resident, rep_decode, rep_prefill, rep_context, rep_mtp, rep_residency):
        f(); print()
