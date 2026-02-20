import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

pipe = FluxImg2ImgPipeline.from_pretrained("/Volumes/XMVPX/mw/flux-root", torch_dtype=torch.bfloat16)

def test_dim(w, h):
    print(f"\n--- Testing {w}x{h} (w%{w%32}, h%{h%32}) ---(w%{w%64}, h%{h%64})")
    img = Image.new('RGB', (w, h), color='blue')
    try:
        with torch.inference_mode():
            res = pipe(prompt="test", image=img, strength=0.5, num_inference_steps=2)
            print(f"Success! output: {res.images[0].size}")
            return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

# We know 256x160 fails (both % 32 == 0) (width % 64 == 0, height % 64 == 32)
# Let's test 256x192 (both % 64 == 0)
print("Testing multiples of 64...")
test_dim(256, 192)

