from PIL import Image
import torch
from flux_bridge import get_flux_bridge

br = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")
img = Image.new('RGB', (256, 160), color='blue')

# Trying WITH height and width kwargs explicit (which we current omit)
try:
    with torch.inference_mode():
        # bypassing the bridge wrapper to hit pipeline directly
        res = br.img2img_pipeline(
            prompt="test",
            image=img,
            height=160,
            width=256,
            strength=0.5,
            num_inference_steps=2
        )
        print("Success passing width/height directly!")
except Exception as e:
    print("FAILED with width/height explicitly passed:", e)

