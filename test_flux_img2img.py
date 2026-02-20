from PIL import Image
import torch
import gc
from flux_bridge import get_flux_bridge

br = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")
img = Image.new('RGB', (240, 160), color = 'red')

try:
    with torch.inference_mode():
        res = br.generate_img2img("test", img, width=240, height=160, strength=0.5, steps=4)
        print("Success!", res.size if res else None)
except Exception as e:
    print("FAILED:", str(e))
