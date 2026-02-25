import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image

try:
    pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    img = Image.new('RGB', (256, 160), color='white')
    
    # Generate schedule
    pipe.scheduler.set_timesteps(4, device="cpu")
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    
    print("Timesteps:", timesteps)
    print("Sigmas:", sigmas)
    
    strength = 0.5
    start_idx = int(len(timesteps) * (1.0 - strength))
    filtered_sigmas = sigmas[start_idx:].cpu().tolist()
    print("Filtered Sigmas:", filtered_sigmas)
    
    out = pipe(prompt="A red ball", image=img, sigmas=filtered_sigmas, strength=1.0)
    print("Success! Generated image")
except Exception as e:
    print("ERROR:", type(e), e)
