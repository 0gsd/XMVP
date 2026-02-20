from PIL import Image
import torch
from flux_bridge import get_flux_bridge

br = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")

for h in [160, 192, 224, 256]:
    img = Image.new('RGB', (256, h), color='blue')
    print(f"Testing 256x{h}...")
    try:
        with torch.inference_mode():
            res = br.generate_img2img("test", img, width=256, height=h, strength=0.5, steps=2)
            print(f"Success for 256x{h}!")
            break
    except Exception as e:
        print(f"FAILED for 256x{h}:", str(e))
        import traceback
        traceback.print_exc()

