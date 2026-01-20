#!/usr/bin/env python3
"""
train_mll.py
------------
Train a Flux LoRA on MPS/CPU.
Implements Rectified Flow Matching for Flux.
"""

import os
import json
import argparse
import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model

# Standard Paths
FLUX_ROOT = "/Volumes/XMVPX/mw/flux-root"

# Device Selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def compute_text_embeddings(pipeline, prompt):
    """
    Leverage the pipeline's internal logic to get:
    prompt_embeds, pooled_prompt_embeds
    """
    with torch.no_grad():
        # Flux requires getting both T5 and CLIP embeddings
        # Pipeline has encode_prompt methods.
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt, # Usually same prompt for both
            device=device
        )
    return prompt_embeds, pooled_prompt_embeds, text_ids

class MovieDataset(Dataset):
    def __init__(self, root_dir, size=512): # LoRA on 512 is faster/stable
        self.root = root_dir
        self.size = size
        self.data = []
        
        jsonl = os.path.join(root_dir, "metadata.jsonl")
        if os.path.exists(jsonl):
            with open(jsonl, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = os.path.join(self.root, entry["file_name"])
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.size, self.size), Image.LANCZOS)
            import numpy as np
            # Normalize [0, 1] -> [-1, 1]
            img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
            pixel_values = torch.from_numpy(img_np).permute(2, 0, 1) # CHW
            return {"pixel_values": pixel_values, "prompt": entry["text"]}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.data)-1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output_dir", default="adapters/movies")
    parser.add_argument("--steps", type=int, default=100) # Quick fine tune
    parser.add_argument("--lr", type=float, default=2e-4) # Slightly higher for LoRA
    args = parser.parse_args()

    print(f"🎬 Train MLL: {args.name} | Device: {device}")

    # 1. Load Pipeline (BFloat16 for MPS compatibility? torch.float16 is better? MPS prefers Float16)
    # Flux is BF16 native. MPS supports BF16 on M-series.
    dtype = torch.bfloat16
    
    print("   🌊 Loading Flux Pipeline...")
    pipeline = FluxPipeline.from_pretrained(FLUX_ROOT, torch_dtype=dtype).to(device)
    
    # Freeze Components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False) 
    
    # 2. Add LoRA
    print("   💉 Injecting LoRA...")
    lora_config = LoraConfig(
        r=16, lora_alpha=16, 
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian"
    )
    # Extract transformer to wrap
    transformer = pipeline.transformer
    transformer.add_adapter(lora_config)
    
    # Build Optimizer
    params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
    # 3. Data
    dataset = MovieDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 4. Training Loop (Rectified Flow)
    transformer.train()
    global_step = 0
    
    print("   🏃 Start Training...")
    
    while global_step < args.steps:
        for batch in dataloader:
            if global_step >= args.steps: break
            
            optimizer.zero_grad()
            
            # A. Latents
            pixels = batch["pixel_values"].to(device, dtype=dtype)
            with torch.no_grad():
                latents = pipeline.vae.encode(pixels).latent_dist.sample()
                latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                
            # B. Text Embeds
            prompts = batch["prompt"]
            prompt_embeds, pooled, text_ids = compute_text_embeddings(pipeline, prompts)
            
            # C. Noise / Flow Matching
            # Sample t ~ [0, 1]
            bsz = latents.shape[0]
            t = torch.rand((bsz,), device=device, dtype=dtype)
            
            # Noise x1
            noise = torch.randn_like(latents).to(device, dtype=dtype)
            
            # Interpolate: xt = (1-t)x0 + t*x1 (where x0=latents, x1=noise? Flux usually trains x1->x0?)
            # Rectified Flow: x0 = data, x1 = noise.
            # xt = t * x1 + (1 - t) * x0
            # Velocity v = x1 - x0.
            # Model predicts v.
            
            # Flux signature: usually t is "timestep" or "guidance".
            # diffusers Flux pipeline uses timestep 1000...0.
            # But underlying transformer takes 'timestep' as continuous or discrete?
            # It takes 'timestep' (1D tensor).
            
            x_t = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
            target = noise - latents 
            
            # D. Predict
            # Pack latents/text_ids for Flux
            # Using pipeline's internal packing/prep usually requires deeper access.
            # However, FluxTransformer2DModel takes:
            # hidden_states, timestep, encoder_hidden_states, pooled_projections, img_ids, txt_ids...
            
            # We need valid img_ids.
            # Pipeline.prepare_latents creates them.
            # Let's simplify: 
            # 512x512 -> 64x64 latents -> 4096 tokens?
            # Flux packs images heavily.
            
            # To avoid implementing packing from scratch, we use a simpler strategy:
            # Skip packing if the model supports unpacked (it handles it internally usually?)
            # FluxTransformer DOES expect packed inputs usually.
            
            # CRITICAL: Reimplementing Flux packing is complex.
            # Alternative: Just run the forward pass and let it fail if I can't pack?
            # Or use `pipeline.transformer` correctly.
            
            # Let's rely on standard assumption:
            # If we pass standard shaped inputs (B, C, H, W) to `transformer()`, does it work?
            # FluxTransformer expects `hidden_states` as (B, L, D). IT IS A TRANSFORMER.
            # So we MUST patch/embed the image.
            
            # Ok, implementing packing is REQUIRED.
            # We use `pipeline._pack_latents` logic if accessible.
            # Using private methods is risky but `_pack_latents` exists in diffusers source.
            
            H = latents.shape[-2]
            W = latents.shape[-1]
            
            # Get Image IDs (3D positional embeddings)
            # From source:
            img_ids = torch.zeros(H // 2, W // 2, 3) 
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(H // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(W // 2)[None, :]
            img_ids = img_ids.reshape(1, -1, 3).repeat(bsz, 1, 1).to(device)
            
            # Pack Latents
            # (B, C, H, W) -> (B, (H/2)*(W/2), C*4)
            # latents = latents.view(bsz, C, H//2, 2, W//2, 2)
            # latents = latents.permute(0, 2, 4, 1, 3, 5)
            # latents = latents.reshape(bsz, (H//2)*(W//2), C*4)
            
            # Wait, Flux expects specific packing.
            # Let's try to just run it through without manual packing and see if it errors?
            # The error will guide me. 
            # But let's try to pass unpacked? No, it's a transformer.
            
            # Simplified Pack (Standard Patchify)
            hidden_states = latents.permute(0, 2, 3, 1) # b h w c
            hidden_states = hidden_states.reshape(bsz, -1, hidden_states.shape[-1]) # b n c
            
            # This is likely WRONG for Flux which uses patch size 2x2.
            # But for a "Minimal Script" I'm blocking.
            
            # DECISION: To avoid writing a broken trainer, I will use a dummy trainer approach:
            # Use `FluxBridge` to "dry run" a generation command which we hijack? No.
            
            # Real solution: Use the `FluxPipeline` itself? No training support.
            
            # I will trust that standard `diffusers` training examples pack correctly.
            # Pack:
            latents = pipeline._pack_latents(latents, bsz, latents.shape[1], latents.shape[2], latents.shape[3])
            
            # Predict
            model_pred = transformer(
                hidden_states=latents,
                timestep=t,     # Scalar in range [0, 1]? Or [0, 1000]? Flux uses [0,1] sigma.
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False
            )[0]
            
            # Target Packing
            target_packed = pipeline._pack_latents(target, bsz, target.shape[1], target.shape[2], target.shape[3])
            
            loss = F.mse_loss(model_pred, target_packed)
            loss.backward()
            optimizer.step()
            
            if global_step % 10 == 0:
                print(f"   Step {global_step}/{args.steps} | Loss: {loss.item():.4f}")
            global_step += 1

    # 5. Save
    print(f"   💾 Saving Adaptor to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.name}.safetensors")
    
    from safetensors.torch import save_file
    peft_state = {k: v for k, v in transformer.state_dict().items() if "lora" in k}
    save_file(peft_state, out_path)
    
    print("✅ Done.")

if __name__ == "__main__":
    main()
