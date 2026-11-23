#!/usr/bin/env python3
"""Golden image embeddings from the HF PR #51050 vision tower (Inkling2VisionModel + Inkling2ImageProcessor).

Two stages (the PR source is used only through PYTHONPATH in a user-level process; nothing is installed):
  proc:   PYTHONPATH=<tf>/src python3 vision_ref.py proc  IMG_DIR OUT_DIR           (needs torchvision: the HF image processor)
  tower:  PYTHONPATH=<tf>/src <venv>/bin/python vision_ref.py tower OUT_DIR [MODEL_DIR] [threads]   (loads ONLY the 0.64 GB vision weights)
`proc` writes OUT_DIR/<name>.pix.npz (pixel_values [n,40,40,3] fp32, grid_thw) with the HF processor, applied to the PIL image.
`tower` writes OUT_DIR/<name>.npz: pixel_values, grid, x0 (embeddings after pre_layernorm incl. class token), last_hidden (fp32 tower),
embeds (fp32 tower output [n,2304], the golden), embeds_bf16 (bf16 tower, the dtype HF runs by default).
"""
import sys, os, glob, numpy as np

def proc(img_dir, out):
    from PIL import Image
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
    ip = Inkling2ImageProcessor()
    os.makedirs(out, exist_ok=True)
    for p in sorted(glob.glob(img_dir + "/*.png") + glob.glob(img_dir + "/*.jpg")):
        name = os.path.splitext(os.path.basename(p))[0]
        im = Image.open(p); im.load()
        r = ip(images=im, return_tensors="pt")
        np.savez_compressed(f"{out}/{name}.pix.npz", pixel_values=r["pixel_values"].numpy(), grid=r["image_grid_thw"].numpy()[0])
        print(name, im.size, im.mode, tuple(r["pixel_values"].shape), r["image_grid_thw"].tolist())

def tower(out, model="/workspace/q-volta-models/Inkling-2-Mini", threads=16):
    import torch, json
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
    torch.set_num_threads(int(threads))
    cfg = AutoConfig.from_pretrained(model).vision_config
    cfg._attn_implementation = "eager"
    f = safe_open(model + "/model-00001-of-00001.safetensors", "pt")
    sd = {}
    for k in f.keys():
        if not k.startswith("model.visual."): continue
        n = k[len("model.visual."):]
        n = n.replace("backbone.embeddings.", "embeddings.").replace("backbone.ln_pre.", "pre_layernorm.").replace("backbone.layers.", "layers.")
        n = n.replace("proj_in.", "merger.linear_1.").replace("proj_out.", "merger.linear_2.")
        n = n.replace(".ln_1.", ".layernorm_before.").replace(".ln_2.", ".layernorm_after.").replace(".mlp.c_fc.", ".mlp.fc1.").replace(".mlp.c_proj.", ".mlp.fc2.")
        n = n.replace(".ls_1.gamma", ".lambda_1").replace(".ls_2.gamma", ".lambda_2")
        if n == "embeddings.position_embedding": n += ".weight"
        sd[n] = f.get_tensor(k)
    res = {}
    for dt in (torch.float32, torch.bfloat16):
        m = Inkling2VisionModel(cfg).to(dt).eval()
        miss, unexp = m.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=False)
        miss = [x for x in miss if "rotary" not in x]
        assert not miss and not unexp, (miss, unexp)
        res[dt] = m
    for p in sorted(glob.glob(out + "/*.pix.npz")):
        name = os.path.basename(p)[:-8]
        z = np.load(p)
        pv = torch.from_numpy(z["pixel_values"]); grid = torch.from_numpy(z["grid"])[None]
        o = {"pixel_values": z["pixel_values"], "grid": z["grid"]}
        for dt, key in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            m = res[dt]; cap = {}
            h = m.pre_layernorm.register_forward_hook(lambda mod, a, out_: cap.__setitem__("x0", out_.detach().float().numpy()))
            with torch.no_grad():
                r = m(pixel_values=pv.to(dt), image_grid_thw=grid)
            h.remove()
            if key == "fp32":
                o["x0"] = cap["x0"]; o["last_hidden"] = r.last_hidden_state.numpy(); o["embeds"] = r.pooler_output.numpy()
            else:
                o["embeds_bf16"] = r.pooler_output.float().numpy()
        e, b = o["embeds"], o["embeds_bf16"]
        print(name, e.shape, "bf16-vs-fp32 rel L2 %.4g" % (np.linalg.norm(e - b) / np.linalg.norm(e)), "max|e| %.2f" % np.abs(e).max())
        np.savez_compressed(f"{out}/{name}.npz", **o)

if __name__ == "__main__":
    if sys.argv[1] == "proc": proc(sys.argv[2], sys.argv[3])
    else: tower(*sys.argv[2:])
