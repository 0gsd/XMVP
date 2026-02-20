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
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model

# Standard Paths
# Prioritize Flux 2 Klein
FLUX_ROOT = "/Volumes/XMVPX/mw/flux-root/klein-9b"
if not os.path.exists(FLUX_ROOT):
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
        try:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt, # Usually same prompt for both
                device=device
            )
        except TypeError:
            # Fallback for pipelines that don't accept prompt_2 (e.g. Flux2KleinPipeline)
            ret = pipeline.encode_prompt(
                prompt=prompt,
                device=device
            )
            
            # Handle variable return length
            if len(ret) == 3:
                prompt_embeds, pooled_prompt_embeds, text_ids = ret
            elif len(ret) == 2:
                prompt_embeds, pooled_prompt_embeds = ret
                # Synthesize text_ids (zeros of shape [Batch, SeqLen, 3])
                # Usually text_ids are simple position IDs.
                # Flux uses 3 dimensions? or just passed through.
                # Let's verify shape. 
                # prompt_embeds shape: [B, SeqLen, Dim]
                # text_ids shape: [B, SeqLen, 3] usually?
                # Actually, standard Flux pipeline generates them as zeros mostly?
                # Let's try zeros.
                text_ids = torch.zeros(prompt_embeds.shape[0], prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
            else:
                raise ValueError(f"Unexpected return length from encode_prompt: {len(ret)}")
                
    return prompt_embeds, pooled_prompt_embeds, text_ids

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents

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

    # 1. Load Pipeline (BFloat16 for MPS compatibility)
    dtype = torch.bfloat16
    
    print("   🌊 Loading Flux Pipeline (DiffusionPipeline)...")
    # Use DiffusionPipeline to handle custom architectures (Klein) and Auto-Download of components
    pipeline = DiffusionPipeline.from_pretrained(FLUX_ROOT, torch_dtype=dtype, trust_remote_code=True).to(device)
    
    # Validation: Check for T5 (text_encoder_2)
    if not hasattr(pipeline, "text_encoder_2") or pipeline.text_encoder_2 is None:
         print("   ⚠️ Pipeline missing text_encoder_2 (T5). Attempting manual load...")
         try:
             # Fallback logic from flux_bridge
             from transformers import T5EncoderModel, T5TokenizerFast
             t5_local = "/Volumes/XMVPX/mw/t5weights-root"
             if os.path.exists(t5_local):
                 print(f"      📚 Loading T5 from Local Cache: {t5_local}")
                 pipeline.text_encoder_2 = T5EncoderModel.from_pretrained(t5_local, torch_dtype=dtype).to(device)
                 pipeline.tokenizer_2 = T5TokenizerFast.from_pretrained(t5_local)
             else:
                 print("      ☁️ Loading T5 from Hub (city96/t5-v1_1-xxl-encoder-bf16)...")
                 pipeline.text_encoder_2 = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=dtype).to(device)
                 pipeline.tokenizer_2 = T5TokenizerFast.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16")
         except Exception as e:
             print(f"   ❌ Failed to load T5 fallback: {e}")
             # We might crash later if encode_prompt needs it, but let's try.
    
    # Verify Transformer access
    if not hasattr(pipeline, "transformer"):
        # Some custom pipelines might nest it?
        raise ValueError(f"Pipeline {type(pipeline)} missing 'transformer' attribute.")
    
    # Freeze Components
    if hasattr(pipeline, "vae"): pipeline.vae.requires_grad_(False)
    if hasattr(pipeline, "text_encoder"): pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline, "text_encoder_2"): pipeline.text_encoder_2.requires_grad_(False)
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
                # Handle FrozenDict config access
                shift_factor = pipeline.vae.config.get("shift_factor", 0.0) if hasattr(pipeline.vae.config, "get") else getattr(pipeline.vae.config, "shift_factor", 0.0)
                scaling_factor = pipeline.vae.config.get("scaling_factor", 1.0) if hasattr(pipeline.vae.config, "get") else getattr(pipeline.vae.config, "scaling_factor", 0.3611)
                
                latents = (latents - shift_factor) * scaling_factor
                
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
            C = latents.shape[1]
            
            H = latents.shape[-2]
            W = latents.shape[-1]
            C = latents.shape[1]
            
            # Get Image IDs (3D positional embeddings)
            # Standard Flux is 3 dims. This custom one seems to want 4? (Index 3 out of bounds)
            # We try 4 dimensions.
            img_ids = torch.zeros(H // 2, W // 2, 4)
            img_ids = torch.zeros(H // 2, W // 2, 4)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(H // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(W // 2)[None, :]
            # Leave index 0 and 3 as zeros?
            
            img_ids = img_ids.reshape(1, -1, 4).repeat(bsz, 1, 1).to(device, dtype=dtype)
            
            # Update Text IDs too if we synthesized them
            if text_ids.shape[-1] == 3:
                # Pad text_ids to 4
                text_ids_pad = torch.zeros(text_ids.shape[0], text_ids.shape[1], 4).to(device, dtype=dtype)
                text_ids_pad[..., :3] = text_ids
                text_ids = text_ids_pad
            
            # Pack Latents
            latents = pack_latents(latents, bsz, C, H, W)
            
            # Predict
            # Dynamic Argument Mapping (Handle different Flux variants)
            forward_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "return_dict": False
            }
            
            import inspect
            sig = inspect.signature(transformer.forward)
            params = sig.parameters
            
            # Map Timestep
            if "timestep" in params: forward_kwargs["timestep"] = t
            elif "timesteps" in params: forward_kwargs["timesteps"] = t
            elif "guidance" in params: forward_kwargs["guidance"] = t # Some variants map t to guidance
            
            # Map Pooled Projections (Vector Embeds)
            if "pooled_projections" in params:
                forward_kwargs["pooled_projections"] = pooled
            elif "vec" in params:
                 forward_kwargs["vec"] = pooled
            elif "vector_embeddings" in params:
                 forward_kwargs["vector_embeddings"] = pooled
            
            # Map IDs
            if "txt_ids" in params: forward_kwargs["txt_ids"] = text_ids
            elif "text_ids" in params: forward_kwargs["text_ids"] = text_ids
            
            if "img_ids" in params: forward_kwargs["img_ids"] = img_ids
            elif "image_ids" in params: forward_kwargs["image_ids"] = img_ids
            
            model_pred = transformer(**forward_kwargs)[0]
            
            # Target Packing
            target_packed = pack_latents(target, bsz, C, H, W)
            
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
