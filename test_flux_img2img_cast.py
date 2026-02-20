import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

pipe = FluxImg2ImgPipeline.from_pretrained("/Volumes/XMVPX/mw/flux-root", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

img = Image.new('RGB', (256, 160), color='blue')
print("Testing 256x160 explicit on FluxImg2ImgPipeline...")
try:
    with torch.inference_mode():
        res = pipe(prompt="test", image=img, strength=0.5, num_inference_steps=2)
        print("Success!", res.images[0].size)
except Exception as e:
    print("FAILED:", str(e))
    import traceback
    traceback.print_exc()

