from PIL import Image
import torch
import traceback
from flux_bridge import get_flux_bridge

br = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")

dims_to_test = [
    (256, 160), # Multiple of 32
    (256, 192), # Multiple of 64
    (512, 512),
    (320, 240),
]

for w, h in dims_to_test:
    img = Image.new('RGB', (w, h), color='blue')
    print(f"\n--- Testing {w}x{h} ---")
    try:
        with torch.inference_mode():
            res = br.generate_img2img("test", img, width=w, height=h, strength=0.5, steps=1)
            print(f"Success for {w}x{h}!")
    except Exception as e:
        print(f"FAILED for {w}x{h}:", str(e))
