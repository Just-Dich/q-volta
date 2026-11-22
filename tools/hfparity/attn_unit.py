# Unit-level fp32 vs fp32 check of ONE attention layer (HF PR module vs our inkling_ref.attention) at positions where
# tau != 1.  usage: attn_unit.py LAYER T [N_FLOOR]  (N_FLOOR overrides log_scaling_n_floor on BOTH sides)
import sys, json, copy, numpy as np, torch
sys.path.insert(0, "/workspace/q-volta/tools")
import inkling_ref as R
from transformers import AutoConfig
from transformers.models.inkling2_loop import modeling_inkling2_loop as MM
L = int(sys.argv[1]); T = int(sys.argv[2]); floor = int(sys.argv[3]) if len(sys.argv) > 3 else None
torch.set_num_threads(24); torch.manual_seed(0)
M = "/workspace/q-volta-models/Inkling-2-Mini"
tc = AutoConfig.from_pretrained(M).text_config
tc._attn_implementation = "eager"
if floor: tc.log_scaling_n_floor = floor; R.C.LOG_N = floor
glob = L in R.C.GLOBAL
cls = MM.Inkling2LoopFullAttention if glob else MM.Inkling2LoopSlidingAttention
mod = cls(tc, L).float().eval()
m = R.Inkling(M, verbose=False)
a = f"model.llm.layers.{L}.attn."
sd = mod.state_dict()
mp = {"wq_du.weight": "q_proj.linear.weight", "q_l2_clip.scale": "q_proj.norm.scale", "wk_dv.weight": "k_proj.linear.weight",
      "wv_dv.weight": "v_proj.linear.weight", "wr_du.weight": "relative_position_bias.linear.weight",
      "rel_l2_clip.scale": "relative_position_bias.norm.scale", "rel_logits_proj.proj": "relative_position_bias.proj",
      "gate_attn.weight": "o_proj.gate_proj.weight", "wo_ud.weight": "o_proj.linear.weight", "attn_sink_proj.weight": "sink_proj.weight"}
if glob:
    mp.update({"k_sconv.static_w": "k_proj.sconv.static_weight", "k_sconv.weight_proj": "k_proj.sconv.weight_proj.weight",
               "v_sconv.static_w": "v_proj.sconv.static_weight", "v_sconv.weight_proj": "v_proj.sconv.weight_proj.weight"})
else:
    ks = [k for k in sd if "sconv" in k]; print("local sconv keys:", ks)
    mp.update({"k_sconv.weight": [k for k in ks if k.startswith("k_proj")][0], "v_sconv.weight": [k for k in ks if k.startswith("v_proj")][0]})
new = {}
for ck, hk in mp.items():
    w = m.w(a + ck)
    if hk not in sd: raise SystemExit(f"missing {hk}: {list(sd)}")
    new[hk] = w.reshape(sd[hk].shape) if w.numel() == sd[hk].numel() else w
missing = set(sd) - set(new); print("unmapped HF keys:", missing)
mod.load_state_dict(new, strict=False)
def inp(nw):
    x = torch.randn(T, 2304) * 1.0
    return R.rms_w(x, m.lw(L, nw))
if glob: hq, hk_, hv = inp("q_input_norm.weight"), inp("k_input_norm.weight"), inp("v_input_norm.weight")
else: hq = inp("attn_norm.weight"); hk_ = hq.clone(); hv = hq.clone()
pos = torch.arange(T)
with torch.no_grad():
    ours = m.attention(L, 0, m.new_cache(), hq, hk_, hv, pos)
    ours_nt = None
    R.ABL.add('no_tau'); ours_nt = m.attention(L, 0, m.new_cache(), hq, hk_, hv, pos) if glob else None; R.ABL.discard('no_tau')
    dist = pos[:, None] - pos[None, :]
    mask = torch.zeros(T, T).masked_fill(dist < 0, float("-inf"))
    if not glob: mask = mask.masked_fill(dist >= tc.sliding_window_size, float("-inf"))
    out, _ = mod(hidden_states=hq[None], key_hidden_states=hk_[None], value_hidden_states=hv[None], attention_mask=mask[None, None],
                 relative_distance=dist, conv_mask=None, cache=None, position_ids=pos[None])
hf = out[0]
rel = lambda a, b: float((a - b).norm() / b.norm())
last = slice(T - 64, T)
res = {"layer": L, "T": T, "n_floor": tc.log_scaling_n_floor, "tau_last": float(1 + 0.1 * np.log(max(T / tc.log_scaling_n_floor, 1))),
       "rel_err_hf_vs_ref_all": rel(hf, ours), "rel_err_hf_vs_ref_last64": rel(hf[last], ours[last]),
       "max_abs_hf_vs_ref": float((hf - ours).abs().max())}
if ours_nt is not None:
    res["rel_diff_ref_vs_ref_no_tau_last64"] = rel(ours[last], ours_nt[last]); res["rel_err_hf_vs_ref_no_tau_last64"] = rel(hf[last], ours_nt[last])
print(json.dumps(res))
open("/workspace/q-volta-models/scratch-hfparity/out/attn_unit.jsonl", "a").write(json.dumps(res) + "\n")
