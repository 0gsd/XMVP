import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# Simulate pipeline state
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="scheduler")

# Create a scenario like the one observed in the log:
# 12 steps, strength 0.4 -> sigmas should have 6 items according to the log.
steps = 12
strength = 0.4

scheduler.set_timesteps(steps, device="cpu")
timesteps = scheduler.timesteps
sigmas = scheduler.sigmas

print(f"Initial timesteps ({len(timesteps)}): {timesteps}")
print(f"Initial sigmas ({len(sigmas)}): {sigmas}")

if timesteps is not None and sigmas is not None and len(timesteps) > 0:
    start_idx = int(len(timesteps) * (1.0 - strength))
    start_idx = max(0, min(start_idx, len(timesteps) - 1))
    
    filtered_sigmas = sigmas[start_idx:].cpu().tolist()
    print(f"filtered_sigmas: {filtered_sigmas}")
    
    # Simulate start_timestep retrieval
    try:
        start_timestep = timesteps[start_idx].to("cpu", dtype=torch.float32)
        print(f"Accessed start_timestep from array: {start_timestep}")
    except Exception as e:
        safe_start_timestep = sigmas[start_idx] * 1000.0
        print(f"Fallback. Exception: {e}")
        start_timestep = torch.tensor(safe_start_timestep, device="cpu", dtype=torch.float32)
        
    print(f"start_timestep calculation result: {start_timestep}")
    
    # Latent injection
    effective_batch = 1
    num_channels_latents = 16
    latent_height = 10
    latent_width = 15
    shape = (effective_batch, num_channels_latents, latent_height, latent_width)
    
    image_latents = torch.ones(shape, dtype=torch.float32)
    noise = torch.randn(shape, generator=None, device="cpu", dtype=torch.float32)
    latent_timestep = start_timestep.expand(effective_batch)
    
    noised_latents = scheduler.scale_noise(image_latents, latent_timestep, noise)
    print("scale_noise succeeded.")
