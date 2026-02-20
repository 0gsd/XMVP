import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

print("Setting up scheduler...")
scheduler = FlowMatchEulerDiscreteScheduler()

fixed_height = 320
fixed_width = 480
seq_len = (fixed_height // 16) * (fixed_width // 16)
m = (1.15 - 0.5) / (4096 - 256)
b = 0.5 - m * 256
mu = m * seq_len + b

scheduler.set_timesteps(12, device="cpu", mu=mu)

timesteps = getattr(scheduler, "timesteps", None)
sigmas = getattr(scheduler, "sigmas", None)

print(f"timesteps length: {len(timesteps) if timesteps is not None else 'None'}")
print(f"timesteps shape: {timesteps.shape if hasattr(timesteps, 'shape') else 'None'}")
print(f"sigmas length: {len(sigmas) if sigmas is not None else 'None'}")
print(f"timesteps tensor: {timesteps}")

strength = 0.4
start_idx = int(len(timesteps) * (1.0 - strength))
start_idx = max(0, min(start_idx, len(timesteps) - 1))

print(f"strength: {strength}")
print(f"start_idx: {start_idx}")

try:
    print(f"Attempting to access timesteps[start_idx]...")
    start_timestep = timesteps[start_idx]
    print(f"start_timestep: {start_timestep}")
except Exception as e:
    print(f"Error accessing timesteps: {e}")
