import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as transforms

print("Loading VAE only...")
vae = AutoencoderKL.from_pretrained("/Volumes/XMVPX/mw/flux-root/vae", torch_dtype=torch.bfloat16).to("mps")
print("VAE Loaded.")

def test_vae_encode(w, h):
    print(f"\n--- Testing {w}x{h} (w%{w%32}, h%{h%32}) ---(w%{w%64}, h%{h%64})")
    img = Image.new('RGB', (w, h), color='blue')
    
    # Preprocess image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to("mps", dtype=torch.bfloat16)

    try:
        with torch.inference_mode():
            latents = vae.encode(img_tensor).latent_dist.sample()
            print(f"Success! Latent shape: {latents.shape}")
            return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

# 256x160 failed before
test_vae_encode(256, 160)
# 256x192 is multiple of 64
test_vae_encode(256, 192)
# 256x256 is multiple of 64
test_vae_encode(256, 256)
