#!/usr/bin/env python3 -I
"""Run the Inkling-2-Mini torch reference (tools/inkling_ref.py) over a token sequence (prefill in chunks, optionally followed by
greedy decode steps) and save what the MTP/DSpark drafter needs (see docs/INKLING_MTP.md):

  feat   fp32 [n, 4, 2304]  target features per position: the layer INPUT (running partial block) of layer visits 5, 23, 41
                            (= outputs of applications 4, 22, 40) and the final-norm output (visit 48)
  ids    int32 [n]          the token ids (prompt + greedy continuation)
  argmax int32 [n]          argmax of the target logits at each position (prediction of token t+1)
  n_prompt                  number of prompt tokens

  python3 -I tools/inkling_mtp_extract.py --ids ids.npy --out feats.npz [--gen 96] [--chunk 512] [--threads 48]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling_ref import Inkling, C

KEYS = ["app04_L4_p0", "app22_L10_p1", "app40_L16_p2"]  # outputs == inputs of visits 5, 23, 41


def feats_from_trace(tr):
    return torch.stack([tr[k] for k in KEYS] + [tr["final_norm"]], 1)  # [n,4,H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/q-volta-models/Inkling-2-Mini")
    ap.add_argument("--job", nargs=3, action="append", metavar=("IDS.npy", "OUT.npz", "GEN"), required=True,
                    help="repeatable; one process runs all jobs so the lazily dequantised experts are shared")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--cache-gb", type=float, default=20.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m = Inkling(a.model, expert_cache_gb=a.cache_gb, verbose=False)
    for jb in a.job:
        run_job(m, a, jb[0], jb[1], int(jb[2]))


def run_job(m, a, ids_path, out, gen):
    ids = [int(x) for x in np.load(ids_path)]
    cache = m.new_cache()
    F, A = [], []
    t0 = time.time()
    pos = 0
    while pos < len(ids):
        ch = ids[pos:pos + a.chunk]
        lg, tr = m.forward(ch, cache, want_trace=True, all_logits=True)
        F.append(feats_from_trace(tr).clone())
        A.append(lg.argmax(-1).to(torch.int32))
        pos += len(ch)
        print(f"prefill {pos}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
        del lg, tr
    n_prompt = len(ids)
    for s in range(gen):
        nxt = int(A[-1][-1])
        ids.append(nxt)
        lg, tr = m.forward([nxt], cache, want_trace=True, all_logits=True)
        F.append(feats_from_trace(tr).clone())
        A.append(lg.argmax(-1).to(torch.int32))
        if s % 8 == 0:
            print(f"gen {s}/{gen} {time.time() - t0:.0f}s", flush=True)
    feat = torch.cat(F, 0).numpy()
    am = torch.cat(A, 0).numpy()
    np.savez(out, feat=feat, ids=np.array(ids, np.int32), argmax=am, n_prompt=n_prompt)
    print("saved", out, feat.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
