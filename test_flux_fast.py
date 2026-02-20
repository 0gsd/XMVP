import torch
import os
from flux_bridge import get_flux_bridge
from PIL import Image

def run_test():
    path = "/Volumes/XMVPX/mw/flux-root"  
    if not os.path.exists(path):
        path = "/Users/m3u/METMcloud/METMroot/tools/fmv/mvp/0weights/flux-klein.safetensors"
    bridge = get_flux_bridge(path)
    img = Image.new("RGB", (256, 160), color="red")
    
    # We only care about whether the pipeline call throws the 64GB error.
    # So we do 1 step and see if it fails instantly.
    print("Test 2: Modifying kwargs to pass width/height directly in pipeline call")
    bridge.load_img2img()
    try:
        out = bridge.img2img_pipeline(
            prompt="A blue square",
            image=img,
            width=256,
            height=160,
            strength=0.7,
            num_inference_steps=1,
            guidance_scale=3.5
        )
        print("SUCCESS! Output size:", out.images[0].size)
    except Exception as e:
        print("FAILED WITH EXP:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
