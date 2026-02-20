import torch
import os
from flux_bridge import get_flux_bridge
from PIL import Image

def run_test():
    print("Testing Flux Img2Img shapes")
    path = "/Volumes/XMVPX/mw/flux-root"  # Assume this exists based on previous logs
    if not os.path.exists(path):
        # find safetensors 
        path = "/Users/m3u/METMcloud/METMroot/tools/fmv/mvp/0weights/flux-klein.safetensors"
        if not os.path.exists(path):
            print("Cannot find model path")
            return
            
    bridge = get_flux_bridge(path)
    
    # Create dummy image
    img = Image.new("RGB", (256, 160), color="red")
    
    # Try with width/height passed, which previously had a comment saying NOT to do
    print("Test 1: Passing explicit width/height 256x160")
    try:
        out = bridge.generate_img2img(
            prompt="A blue square",
            image=img,
            width=256,
            height=160,
            strength=0.7,
            steps=4
        )
        if out is None:
            print("Return was None")
        else:
            print("Success output size:", out.size)
    except Exception as e:
        print("Failed:", e)

    print("Test 2: Modifying kwargs to pass width/height directly in pipeline call")
    try:
        # We need to manually call it to bypass the current flux_bridge logic that hides it
        bridge.load_img2img()
        out = bridge.img2img_pipeline(
            prompt="A blue square",
            image=img,
            width=256,
            height=160,
            strength=0.7,
            num_inference_steps=4
        )
        print("Success manually?", out.images[0].size)
    except Exception as e:
        print("Failed manually:", e)

if __name__ == "__main__":
    run_test()
