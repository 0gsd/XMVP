import torch
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image

print("Loading VAE only...")
vae = AutoencoderKL.from_pretrained("/Volumes/XMVPX/mw/flux-root/vae", torch_dtype=torch.bfloat16).to("mps")
print("VAE Loaded.")

def test_vae_encode(w, h):
    print(f"\n--- Testing {w}x{h} ---(w%{w%32}, h%{h%32}) (w%{w%64}, h%{h%64})")
    img = Image.new('RGB', (w, h), color='blue')
    
    # Preprocess image to tensor (equivalent to transforms.ToTensor() + Normalize([0.5], [0.5]))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to("mps", dtype=torch.bfloat16)

    try:
        with torch.inference_mode():
            # encode to latent space
            latents = vae.encode(img_tensor).latent_dist.sample()
            print(f"Success! Latent shape: {latents.shape}")
            return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

# 256x160 failed before (w%64==0, h%64==32)
test_vae_encode(256, 160)
# 256x192 (both % 64 == 0)
test_vae_encode(256, 192)
# 240x160 (both % 16 == 0)
test_vae_encode(240, 160)
