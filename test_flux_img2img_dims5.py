from PIL import Image
import torch
import gc
from flux_bridge import get_flux_bridge

br = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")

def force_test(w, h):
    print(f"\n--- Testing {w}x{h} in Bridge ---")
    img = Image.new('RGB', (w, h), color = 'red')
    try:
        with torch.inference_mode():
            res = br.generate_img2img("test", img, width=w, height=h, strength=0.5, steps=4)
            print("Success!", res.size if res else None)
    except Exception as e:
        print("FAILED:", str(e))

force_test(256, 192) # Multiples of 64

