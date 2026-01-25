#!/usr/bin/env python3
import os
import torch
import logging
import torch
import logging
from diffusers import FluxPipeline, FluxImg2ImgPipeline, DiffusionPipeline
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from PIL import Image

logging.basicConfig(level=logging.INFO)

class FluxBridge:
    def __init__(self, model_path, device="mps"):
        self.model_path = model_path
        self.device = device
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.img2img_pipeline = None
        
        # Check availability
        if device == "mps" and not torch.backends.mps.is_available():
            logging.warning("⚠️ MPS not available. Falling back to CPU (Slow!).")
            self.device = "cpu"
            
        self.load_pipeline(model_path)

    def load_pipeline(self, model_path):
        logging.info(f"   🌊 Loading Flux Pipeline from: {model_path}...")
        
        try:
            # Check if directory or file
            is_diffusers_dir = False
            if os.path.isdir(model_path):
                 if os.path.exists(os.path.join(model_path, "model_index.json")):
                      is_diffusers_dir = True
                 else:
                      logging.warning(f"   ⚠️ Directory found but no model_index.json. Looking for safetensors in {model_path}...")
                      # Find first .safetensors file
                      for f in os.listdir(model_path):
                           if f.endswith(".safetensors"):
                                model_path = os.path.join(model_path, f)
                                logging.info(f"      -> Found single file: {f}")
                                break

            if is_diffusers_dir:
                # Auto-Load (Generic) for directories (Handles Flux.2 Klein, etc.)
                # This uses model_index.json to determine the class (e.g. Flux2KleinPipeline)
                logging.info(f"      ✨ Using Auto-Loader (DiffusionPipeline) for {model_path}...")
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True # Needed for custom pipelines like Klein
                ).to(self.device)
            else:
                # Single File Loader (Needs explicit encoders usually if not in file)
                # Attempt 1: Try default loading (might fail if weights missing)
                try:
                    self.pipeline = FluxPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.bfloat16
                    ).to(self.device)
                except Exception as e:
                     if "CLIPTextModel" in str(e) or "text_encoder" in str(e):
                         logging.warning("   ⚠️ Flux Single File missing Encoders. Loading from Local/Hub...")
                         
                         # 1. CLIP (Standard Hub)
                         text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
                         tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                         
                         # 2. T5 (Local Preference -> Hub Fallback)
                         t5_local_path = "/Volumes/XMVPX/mw/t5weights-root"
                         if os.path.exists(t5_local_path):
                             logging.info(f"      📚 Loading T5 from Local Cache: {t5_local_path}")
                             text_encoder_2 = T5EncoderModel.from_pretrained(t5_local_path, torch_dtype=torch.bfloat16)
                             tokenizer_2 = T5TokenizerFast.from_pretrained(t5_local_path) 
                         else:
                             logging.warning("      ☁️ Local T5 not found. Downloading from Hub (city96/t5-v1_1-xxl-encoder-bf16)...")
                             text_encoder_2 = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=torch.bfloat16)
                             tokenizer_2 = T5TokenizerFast.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16")
                         
                         self.pipeline = FluxPipeline.from_single_file(
                             model_path,
                             text_encoder=text_encoder,
                             tokenizer=tokenizer,
                             text_encoder_2=text_encoder_2,
                             tokenizer_2=tokenizer_2,
                             torch_dtype=torch.bfloat16
                         ).to(self.device)
                     else:
                         raise e

            # Optimization for Mac
            if self.device == "mps":
                # Recommended for Flux on Mac
                pass 
                
            logging.info("   ✅ Flux Pipeline Ready.")
            
        except Exception as e:
            logging.error(f"   ❌ Failed to load Flux: {e}")
            self.pipeline = None

    def load_lora(self, lora_path, adapter_name="default", scale=1.0):
        """Loads a LoRA adapter."""
        if not self.pipeline: return False
        
        logging.info(f"   💉 Loading LoRA: {lora_path} (Scale: {scale})")
        try:
            self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            # FluxPipeline supports set_adapters or fuse_lora?
            # Diffusers unified LoRA support:
            self.pipeline.fuse_lora(lora_scale=scale) # Fuse for speed? Or keep separate?
            # Note: fuse_lora merges weights. If we want to switch movies, we should unfuse first?
            # For simplicity in this script (one run per movie), fusing is fine and faster.
            logging.info("   ✅ LoRA Fused.")
            return True
        except Exception as e:
            logging.error(f"   ❌ LoRA Load Failed: {e}")
            return False

    def generate(self, prompt, width=1024, height=1024, steps=4, seed=None, guidance_scale=3.5, image=None, strength=0.65):
        """
        Unified generation method.
        If 'image' is provided, performs Img2Img.
        Otherwise, performs Text2Image.
        """
        if image is not None:
             # Route to Img2Img
             if not self.img2img_pipeline:
                 self.load_img2img()
                 
             if self.img2img_pipeline:
                 return self.generate_img2img(prompt, image, strength=strength, width=width, height=height, steps=steps, seed=seed, guidance_scale=guidance_scale)
             else:
                 # Fallback to T2I (Graceful degradation for Single Model Mode)
                 logging.warning("   ⚠️ Img2Img requested but Pipeline not ready. Falling back to Text-to-Image (ignoring input image).")
                 # proceed to T2I block below...

        if not self.pipeline:
            self.load_pipeline()
            # logging.error("   ❌ Flux Pipeline not initialized.")
            # return None
            
        logging.info(f"   🎨 Flux Generating: {prompt[:40]}... ({width}x{height}, {steps} steps, G:{guidance_scale})")
        
        # Memory Cleanup (Critical for Loop Stability)
        import gc
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed) # MPS generators tricky? Use CPU for determinism if needed
            
        # Prompt Sanitization & Truncation (Fix for CLIP 77 token limit)
        # Flux uses T5 (512 tokens) and CLIP (77 tokens). Diffusers usually masks the excess,
        # but explicit truncation avoids "Batch size mismatch" or tokenizer warnings.
        # We target ~70 words / 300 chars to be safe.
        safe_prompt = prompt
        if len(prompt) > 350:
            logging.warning(f"   ✂️ Truncating long prompt ({len(prompt)} chars).")
            safe_prompt = prompt[:350]
            
        try:
            image_obj = self.pipeline(
                prompt=safe_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=guidance_scale # Configurable
            )
            image = image_obj.images[0]
            del image_obj
            
            # Post-Gen Cleanup
            import gc
            gc.collect() 
            if self.device == "mps":
                torch.mps.empty_cache()
                
            return image
        except Exception as e:
            logging.error(f"   ❌ Flux Generation Error: {e}")
            return None

    def load_img2img(self):
        """Lazy loads the Img2Img pipeline, reusing components if possible."""
        if self.img2img_pipeline: return

        logging.info("   🔄 Loading Flux Img2Img Pipeline...")
        
        try:
            # DIRECT LOADING (Avoid AutoPipeline due to Transformers conflicts)
            from diffusers import FluxImg2ImgPipeline
            import inspect

            # 1. Zero-Copy Reuse: Check if the current pipeline already supports img2img (has 'image' argument)
            # This handles custom pipelines like Flux2Klein if they are unified.
            if self.pipeline:
                call_args = inspect.signature(self.pipeline.__call__).parameters
                if "image" in call_args:
                     # RELAXED CHECK: If it has 'image', we assume it's a valid Img2Img pipeline (or Unified Pipeline).
                     # Flux2Klein and others might hide strength/denoising in kwargs or use different names,
                     # but stopping here causes a fallback to Txt2Img which breaks everything.
                     logging.info("   ✨ Pipeline natively supports 'image' input. Reusing as Img2Img (Zero-Copy).")
                     self.img2img_pipeline = self.pipeline
                     return

            if self.pipeline:
                # 2. Component Casting (Try to promote T2I to I2I)
                # FluxImg2ImgPipeline shares components with FluxPipeline
                try:
                    self.img2img_pipeline = FluxImg2ImgPipeline(**self.pipeline.components).to(self.device)
                    logging.info("   ✅ Flux Img2Img Ready (Shared Components).")
                except TypeError as e:
                    # Missing components (e.g. text_encoder_2 for quantized/distilled models)
                    logging.warning(f"   ⚠️ Cannot cast to FluxImg2ImgPipeline (Missing Components: {e}).")
                    self.img2img_pipeline = None
            
            if not self.img2img_pipeline:
                 # 3. Independent Load (if we didn't just fail casting, or if we want to try loading from disk as I2I)
                 # If casting failed, likely loading from disk as I2I will also fail if it's the same model structure?
                 # Not necessarily, maybe from_pretrained handles it differently than __init__.
                 pass

            if not self.img2img_pipeline:
                 # 3. Independent Load (Force Standard Flux Img2Img from disk)
                 logging.info("   ⚠️ Component Casting failed. Attempting independent load of FluxImg2ImgPipeline...")
                 try:
                     # Force standard class. Use ignore_mismatched_sizes if needed?
                     # We trust remote code false here to avoid loading the custom T2I pipeline again?
                     # But if we need custom components... 
                     # Let's try loading with the SAME components but bypassing checks?
                     # No, let's try strict from_pretrained.
                     self.img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(
                         self.model_path,
                         torch_dtype=torch.bfloat16,
                         trust_remote_code=True
                     ).to(self.device)
                     logging.info("   ✅ Flux Img2Img Loaded (Independent).")
                 except Exception as e_ind:
                     logging.warning(f"   ⚠️ Independent Load Failed: {e_ind}")

        except Exception as e:
            logging.error(f"   ❌ Failed to load Flux Img2Img: {e}")
        
        if not self.img2img_pipeline:
            logging.warning("   ⚠️ Flux Img2Img incompatible or failed. Fallback default: None (T2I Fallback will activate).")

    def generate_img2img(self, prompt, image, strength=0.6, width=1024, height=1024, steps=4, seed=None, guidance_scale=3.5):
        if not self.img2img_pipeline:
            self.load_img2img()
            
        if not self.img2img_pipeline:
            return None
            
        logging.info(f"   🎨 Flux Img2Img: {prompt[:40]}... (Str: {strength}, {width}x{height}, G:{guidance_scale})")
        
        # Memory Cleanup
        import gc
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        
        # Inspect signature to adapt arguments
        import inspect
        try:
             # Check if we are dealing with a pipeline call or a method
             call_method = self.img2img_pipeline.__call__
             sig = inspect.signature(call_method)
             available_args = sig.parameters.keys()
             
             kwargs = {
                 "prompt": prompt,
                 "image": image,
                 "num_inference_steps": steps,
                 "guidance_scale": guidance_scale,
                 "height": height,
                 "width": width
             }
             
             if seed is not None:
                 kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
             
             # Handle 'strength' vs 'denoising_start'
             if "strength" in available_args:
                 kwargs["strength"] = strength
             elif "denoising_start" in available_args:
                 logging.info(f"   ⚠️ Pipeline missing 'strength'. Mapping {strength} -> 'denoising_start' ({1.0 - strength:.2f})")
                 kwargs["denoising_start"] = 1.0 - strength
             else:
                 logging.warning(f"   ⚠️ Pipeline signature missing 'strength' or 'denoising_start'. Available args: {list(available_args)}")
                 # Hail Mary: Just try passing strength anyway.
                 # Many custom pipelines or wrapped renderers accept kwargs that signature inspection misses.
                 logging.info(f"   🤞 Force-passing 'strength={strength}' to pipeline (Hope it takes it)...")
                 kwargs["strength"] = strength
                 
                 # Strategy Update: Check for 'sigmas'?
                 if "sigmas" in available_args:
                     logging.info(f"   ✨ Pipeline uses 'sigmas'. Calculating noise schedule for strength {strength}...")
                     try:
                         # 1. Get full sigmas from scheduler
                         # FluxFlowMatchEulerDiscreteScheduler usually has a set_timesteps method
                         # or we can generate them.
                         # Simpler: Use the pipe's internal logic if possible, or manual generation.
                         
                         # Manual Sigma Generation for Flux (Simplified)
                         # This maps 'strength' to a sigma start index.
                         from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
                         if isinstance(self.img2img_pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
                             # Generate full steps
                             try:
                                 # Newer Flux schedulers require 'mu' (e.g. for dynamic shifting)
                                 # Standard Flux1.0 mu is often derived or default.
                                 # Let's try passing it if needed.
                                 # Inspect signature? Or just try/except with args.
                                 
                                 # Attempt 1: Standard
                                 # self.img2img_pipeline.scheduler.set_timesteps(steps, device=self.device)
                                 
                                 # Attempt 2: With 'mu' (if error previously)
                                 # What value? Default is often 1.0 or based on resolution?
                                 # Let's try passing mu (assuming 256*256 resolution base or similar?)
                                 # Actually, let's catch the error and retry with mu.
                                 
                                 # Attempt 1: Standard
                                 # We force CPU for the scheduler to avoid "MPS->Numpy" crashes later.
                                 self.img2img_pipeline.scheduler.set_timesteps(steps, device="cpu")
                             except Exception as te:
                                 # Standard Flux1.0 mu error is usually TypeError, but let's be safe.
                                 logging.warning(f"      🔧 Scheduler set_timesteps failed: {te} ({type(te).__name__}). Retrying with mu=1.0 and device='cpu'...")
                                 # FIX: Pass 'mu' (some schedulers require it for dynamic shifting)
                                 # Warning: Some diffusers versions expect 'mu' in init, others in set_timesteps.
                                 # We try passing it here.
                                 try:
                                     self.img2img_pipeline.scheduler.set_timesteps(steps, device="cpu", mu=1.0)
                                 except TypeError:
                                     # If mu is not accepted, maybe it's older pattern? Just retry without mu (already failed) or try just device?
                                     # Actually, the error was "mu must be passed". So we MUST pass it.
                                     pass

                             timesteps = self.img2img_pipeline.scheduler.timesteps
                             
                             # Calculate start index based on strength
                             # Strength 1.0 = Index 0 (Full Denoise)
                             # Strength 0.0 = Index Max (No Denoise)
                             # num_inference_steps * strength
                             start_idx = int(len(timesteps) * (1.0 - strength))
                             start_idx = max(0, min(start_idx, len(timesteps) - 1))
                             
                             # Slice sigmas/timesteps?
                             # Some pipelines want 'sigmas', some want 'timesteps'.
                             # The error message said 'sigmas' is an arg.
                             
                             # Flux uses 'sigmas'.
                             # We need the full list but start from a specific point? 
                             # Or does it accept a list of sigmas to run?
                             # Usually passing 'sigmas' overrides 'num_inference_steps'.
                             
                             # FIX: Move sigmas to CPU and convert to LIST to avoid "can't convert mps:0..." and "numpy()" errors.
                             # Lists are device-agnostic.
                             filtered_sigmas = self.img2img_pipeline.scheduler.sigmas[start_idx:].cpu().tolist()
                             
                             filtered_sigmas = self.img2img_pipeline.scheduler.sigmas[start_idx:].cpu().tolist()
                             
                             kwargs["sigmas"] = filtered_sigmas
                             logging.info(f"   Generated {len(kwargs['sigmas'])} sigmas from {steps} steps (Strength {strength}).")

                             # FIX: Failed 'Flux2KleinPipeline' rejects 'strength' arg if we passed it in "Hail Mary".
                             # Since we have sigmas, we MUST remove strength/denoising_start to avoid TypeError.
                             if "strength" in kwargs:
                                 del kwargs["strength"]
                             if "denoising_start" in kwargs:
                                 del kwargs["denoising_start"]

                             
                             # Let's try passing 'timesteps' if available? No, log didn't key it.
                             # Let's try passing 'sigmas' as the full list?
                             
                             # WAIT! If we pass 'sigmas', we override the scheduler.
                             # We want to run only the last X steps.
                             
                            # Clean up: Don't overwrite our good list!

                         
                     except Exception as exc:
                         logging.warning(f"   ⚠️ Sigma calculation failed: {exc}")


                 # If we are here, we are desperate?
                 # Only if we failed to set strength, denoising_start, OR sigmas.
                 if "sigmas" not in kwargs and "strength" not in kwargs and "denoising_start" not in kwargs:
                     logging.warning("   ⚠️ Dropping strength/denoising arguments entirely (Pipeline might do full redraw).")
                     # We only delete if they exist and we've decided they are invalid?
                     # Actually, if they are not in kwargs, we don't need to delete.
                     # But above "Hail Mary" might have put 'strength' in.
                     
                     # Check again:
                     # If "strength" is in kwargs but signature check failed...
                     # We force passed it. So we should NOT delete it here unless we are sure.
                     pass 
                 
                 # Clean up specific keys if we found a better alternative (conflicts)
                 # If we have sigmas, we might want to remove strength to avoid "ambiguous argument" errors?
                 # Standard Flux pipeline behaves fine with sigmas + strength (sigmas win).
                 # So let's REMOVE the deletion logic.
                 pass

             try:
                 out_img_obj = self.img2img_pipeline(**kwargs)
             except TypeError as te:
                 logging.error(f"   ❌ Flux Img2Img TypeError: {te}. Available args: {list(available_args)}. Attempted kwargs: {kwargs.keys()}")
                 return None

             out_img = out_img_obj.images[0]
             del out_img_obj
             
             # Post-Gen Cleanup
             import gc
             gc.collect() 
             if self.device == "mps":
                 torch.mps.empty_cache()

             return out_img
        except Exception as e:
             logging.error(f"   ❌ Flux Img2Img Error: {e}")
             return None

    def unload(self):
        """Unload Flux pipelines."""
        if self.pipeline:
             logging.info("   🗑️  Unloading Flux Engine...")
             del self.pipeline
             self.pipeline = None
             
        if self.img2img_pipeline:
             del self.img2img_pipeline
             self.img2img_pipeline = None
             
        import gc
        gc.collect()
        if self.device == "mps":
             torch.mps.empty_cache()
        logging.info("   ✅ Flux Engine Unloaded.")


# Singleton Pattern for specific use cases
_BRIDGE = None
def get_flux_bridge(path):
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = FluxBridge(path)
    return _BRIDGE


if __name__ == "__main__":
    # Test
    path = "/Volumes/XMVPX/mw/flux-root"
    if os.path.exists(path):
        bridge = FluxBridge(path)
        img = bridge.generate("A pixel art cyberpunk city", width=512, height=512)
        if img:
            img.save("test_flux.png")
            print("Saved test_flux.png")
    else:
        print(f"Skipping test, path not found: {path}")
