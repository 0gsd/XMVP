#!/usr/bin/env python3
import os
import torch
import logging
import torch
import logging
from diffusers import FluxPipeline, FluxImg2ImgPipeline
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
            if os.path.isdir(model_path):
                self.pipeline = FluxPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16
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

    def generate(self, prompt, width=1024, height=1024, steps=4, seed=None):
        if not self.pipeline:
            logging.error("   ❌ Flux Pipeline not initialized.")
            return None
            
        logging.info(f"   🎨 Flux Generating: {prompt[:40]}... ({width}x{height}, {steps} steps)")
        
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
                safe_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=0.0 # Flux Schnell
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
        from diffusers import FluxImg2ImgPipeline
        
        try:
            if self.pipeline:
                # Reuse components!
                self.img2img_pipeline = FluxImg2ImgPipeline(
                    **self.pipeline.components
                ).to(self.device)
                logging.info("   ✅ Flux Img2Img Ready (Shared Components).")
            else:
                # Fallback: Load from scratch (Expensive!)
                logging.warning("   ⚠️ Primary Pipeline not loaded. Loading Img2Img from scratch.")
                if os.path.isabs(self.model_path) and os.path.exists(self.model_path):
                     self.img2img_pipeline = FluxImg2ImgPipeline.from_single_file(
                         self.model_path,
                         torch_dtype=torch.bfloat16
                     ).to(self.device)
                else:
                     self.img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(
                         self.model_path,
                         torch_dtype=torch.bfloat16
                     ).to(self.device)
                logging.info("   ✅ Flux Img2Img Ready (Independent).")
                
        except Exception as e:
            logging.error(f"   ❌ Failed to load Flux Img2Img: {e}")

    def generate_img2img(self, prompt, image, strength=0.6, width=1024, height=1024, steps=4, seed=None):
        self.load_img2img()
        if not self.img2img_pipeline:
            return None
            
        logging.info(f"   🎨 Flux Img2Img: {prompt[:40]}... (Str: {strength}, {width}x{height})")
        
        # Memory Cleanup
        import gc
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
        try:
            out_img_obj = self.img2img_pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=0.0
            )
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
