import torch
from test_scheduler2 import *
from diffusers import FluxImg2ImgPipeline

pipe = FluxImg2ImgPipeline.from_pretrained('/Volumes/XMVPX/mw/flux-root/klein-9b', torch_dtype=torch.float32)
pipe.to("mps")
print("Pipeline loaded.")

image = torch.randn(1, 3, 320, 480).to("mps")
gen = torch.Generator("mps").manual_seed(0)
latents = pipe._encode_vae_image(image=image, generator=gen)
print('Latents shape:', getattr(latents, 'shape', type(latents)))

pipe.scheduler.set_timesteps(12, device="mps")
timesteps = pipe.scheduler.timesteps
sigmas = pipe.scheduler.sigmas
strength = 0.4
start_idx = int(len(timesteps) * (1.0 - strength))
start_idx = max(0, min(start_idx, len(timesteps) - 1))

start_timestep = timesteps[start_idx].to("mps", dtype=torch.float32)
latent_timestep = start_timestep.expand(1)

noise = torch.randn_like(latents).to("mps")
noised_latents = pipe.scheduler.scale_noise(latents, latent_timestep, noise)
print("Latents noised successfully!")
print("Done.")
