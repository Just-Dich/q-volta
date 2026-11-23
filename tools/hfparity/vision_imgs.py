#!/usr/bin/env python3
"""Make the small synthetic test images of tests/golden/inkling_vision (deterministic; PIL + numpy only).
usage: vision_imgs.py OUT_DIR [photo_source.jpg]
"""
import sys, os, numpy as np
from PIL import Image, ImageDraw, ImageFont

def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    # 1. gradient, 200x120 (not a multiple of 40: right/bottom padding)
    h, w = 120, 200
    y, x = np.mgrid[0:h, 0:w]
    g = np.stack([x * 255 // (w - 1), y * 255 // (h - 1), (x + y) * 255 // (w + h - 2)], -1).astype(np.uint8)
    Image.fromarray(g).save(f"{out}/gradient_200x120.png")
    # 2. rendered text HELLO, 320x120 (aspect 8:3)
    im = Image.new("RGB", (320, 120), (255, 255, 255)); d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 84)
    except Exception:
        f = ImageFont.load_default()
    d.text((12, 8), "HELLO", fill=(0, 0, 0), font=f)
    im.save(f"{out}/hello_320x120.png")
    # 3. big red square on blue, 224x224
    a = np.zeros((224, 224, 3), np.uint8); a[:] = (20, 40, 220); a[48:176, 48:176] = (230, 20, 20)
    Image.fromarray(a).save(f"{out}/redsquare_224x224.png")
    # 4. photo-like: a real JPEG (a diagram/photo from the local disk), downscaled to 240x160 and saved as JPEG q90
    src = sys.argv[2] if len(sys.argv) > 2 else None
    if src and os.path.exists(src):
        p = Image.open(src).convert("RGB").resize((240, 160), Image.BICUBIC)
    else:
        rng = np.random.RandomState(0)
        n = rng.rand(10, 15, 3); p = Image.fromarray((n * 255).astype(np.uint8)).resize((240, 160), Image.BICUBIC)
    p.save(f"{out}/photo_240x160.jpg", quality=90)
    # 5. RGBA with transparency (alpha composited onto the 8px checkerboard by the processor), 120x80
    r = np.zeros((80, 120, 4), np.uint8); r[..., :3] = (30, 160, 60); r[..., 3] = 255
    r[20:60, 30:90, 3] = 100; r[:, :20, 3] = 0
    Image.fromarray(r, "RGBA").save(f"{out}/rgba_120x80.png")
    # 6. tiny: 40x40 (exactly one token) and 33x47 odd size
    t = np.random.RandomState(1).randint(0, 256, (47, 33, 3), np.uint8)
    Image.fromarray(t).save(f"{out}/noise_33x47.png")

main()
