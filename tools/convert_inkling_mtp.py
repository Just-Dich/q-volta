#!/usr/bin/env python3 -I
"""Inkling-2-Mini MTP (DSpark) drafter -> sidecar Q-Volta container `<model>-mtp.qv` (docs/INKLING_MTP.md).
The main container is untouched. Tensors (names = checkpoint names with an `mtp.` prefix):
  mtp.layers.I.self_attn.{q,k,v,o}_proj.weight, k_proj_ctx / v_proj_ctx, mlp.{gate,up,down}_proj.weight   quantized (RTN+clip, g64 meta8, --bits)
  mtp.layers.I.{input_layernorm,post_attention_layernorm,dflare_norm}.weight, self_attn.{q,k}_norm / k_norm_ctx  f32
  mtp.norm.weight, mtp.mask_embed, mtp.fuse_alpha [4,4] (= softmax(fuse_weight, -1))                          f32
  mtp.markov.rnn_wa [768,512] / rnn_wb [768,4608] f16 (rnn_proj columns [0:512] / [512:5120]) + rnn_proj.bias f32, mtp.markov.dflare_norm.weight f32
  mtp.markov.w1 [200058,256]  previous-token embedding of the recurrence                                       q (--head-bits, g64)
  mtp.markov.fold_weight [200058,256] = unembed @ hidden_out.weight (fp64, folded once as vLLM does)          q (--head-bits, g64)
  mtp.markov.fold_bias [200058] = unembed @ hidden_out.bias                                                     f32
Usage: python3 -I tools/convert_inkling_mtp.py --out /workspace/q-volta-models/qv/inkling2-mini-mtp.qv [--bits 5] [--head-bits 6]
"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_format as F
import qv_quant as Q

MD = "/workspace/q-volta-models/Inkling-2-Mini"


def quant(w, W, name, mat, bits, group=64, dev="cpu"):
    Wd = mat.float().to(dev)
    q, S, M, r = Q.rtn(Wd, bits, group, clip=True, meta8=True)
    dq = Q.dequant_g(q, S, M, group, bits, r)
    err = float(((dq - Wd) ** 2).sum() / (Wd ** 2).sum()) ** 0.5
    W.add_quant(name, F.pack(q, bits).cpu(), Q.pack_meta(S, M, r).cpu(), bits, list(mat.shape), group, r.cpu())
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=5, help="bits of the drafter transformer matrices")
    ap.add_argument("--head-bits", type=int, default=6, help="bits of markov w1 / fold_weight")
    ap.add_argument("--dev", default="cpu")
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    mc = cfg["mtp_config"]
    f = safe_open(os.path.join(a.model, "mtp.safetensors"), "pt")
    main_f = safe_open(os.path.join(a.model, "model-00001-of-00001.safetensors"), "pt")
    W = F.ContainerWriter(a.out)
    errs = {}
    for i in range(mc["num_hidden_layers"]):
        P = f"layers.{i}."
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "k_proj_ctx", "v_proj_ctx"):
            errs[P + n] = quant(f, W, f"mtp.{P}self_attn.{n}.weight", f.get_tensor(f"{P}self_attn.{n}.weight"), a.bits, dev=a.dev)
        for n in ("gate_proj", "up_proj", "down_proj"):
            errs[P + n] = quant(f, W, f"mtp.{P}mlp.{n}.weight", f.get_tensor(f"{P}mlp.{n}.weight"), a.bits, dev=a.dev)
        for n in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            W.add_dense(f"mtp.{P}{n}.weight", f.get_tensor(f"{P}{n}.weight"), "f32")
        for n in ("q_norm", "k_norm", "k_norm_ctx"):
            W.add_dense(f"mtp.{P}self_attn.{n}.weight", f.get_tensor(f"{P}self_attn.{n}.weight"), "f32")
    W.add_dense("mtp.norm.weight", f.get_tensor("norm.weight"), "f32")
    W.add_dense("mtp.mask_embed", f.get_tensor("mask_embed"), "f32")
    W.add_dense("mtp.fuse_alpha", torch.softmax(f.get_tensor("fuse_weight").float(), -1), "f32")
    mh = "markov_head."
    rw = f.get_tensor(mh + "rnn_proj.weight")  # [768, 5120] = [state(256) | prev embedding(256) | hidden(2304) | f_rnn(2304)]
    W.add_dense("mtp.markov.rnn_wa", rw[:, :512].contiguous(), "f16")   # recurrence part, applied per step
    W.add_dense("mtp.markov.rnn_wb", rw[:, 512:].contiguous(), "f16")   # [hidden, f_rnn] part, applied to all steps at once
    W.add_dense("mtp.markov.rnn_proj.bias", f.get_tensor(mh + "rnn_proj.bias"), "f32")
    W.add_dense("mtp.markov.dflare_norm.weight", f.get_tensor(mh + "dflare_norm.weight"), "f32")
    uv = 200058
    w1 = f.get_tensor(mh + "markov_w1.weight")[:uv]
    errs["markov_w1"] = quant(f, W, "mtp.markov.w1", w1, a.head_bits, dev=a.dev)
    # fold hidden_out into the unembedding (fp64, as vLLM Inkling2MarkovHead.fold)
    ho_w = f.get_tensor(mh + "hidden_out.weight").double()  # [2304, 256]
    ho_b = f.get_tensor(mh + "hidden_out.bias").double()
    U = main_f.get_slice("model.llm.unembed.weight")
    fw = torch.empty(uv, ho_w.shape[1])
    fb = torch.empty(uv)
    for r0 in range(0, uv, 8192):
        r1 = min(uv, r0 + 8192)
        u = U[r0:r1].double()
        fw[r0:r1] = (u @ ho_w).float()
        fb[r0:r1] = (u @ ho_b).float()
    errs["fold_weight"] = quant(f, W, "mtp.markov.fold_weight", fw, a.head_bits, dev=a.dev)
    W.add_dense("mtp.markov.fold_bias", fb, "f32")
    conf = dict(arch="inkling2_mtp", vocab_size=201024, eos_token_id=200006, hidden_size=mc["hidden_size"], num_hidden_layers=mc["num_hidden_layers"],
                num_attention_heads=mc["num_attention_heads"], num_key_value_heads=mc["num_key_value_heads"], head_dim=mc["head_dim"],
                intermediate_size=mc["intermediate_size"], rms_norm_eps=mc["rms_norm_eps"], rope_theta=mc["rope_theta"],
                local_layer_ids=mc["local_layer_ids"], sliding_window_size=mc["sliding_window_size"], block_size=mc["block_size"],
                markov_rank=mc["markov_rank"], target_layer_ids=mc["target_layer_ids"], unpadded_vocab=uv,
                final_logit_softcap=30.0, bits=a.bits, head_bits=a.head_bits)
    W.close(conf)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 2**20:.1f} MiB in {time.time() - t0:.0f}s; rel rms weight err "
          f"mean {sum(errs.values()) / len(errs):.4f} max {max(errs.values()):.4f}")
    json.dump(errs, open(a.out + ".report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
