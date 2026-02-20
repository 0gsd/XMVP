#!/usr/bin/env python3
import os
import torch
import logging
import torch
import logging
from diffusers import FluxPipeline, FluxImg2ImgPipeline, DiffusionPipeline
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from PIL import Image
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None # Handle optional dependency

import logging

logging.basicConfig(level=logging.INFO)

def _make_multiple_of_32(val: int) -> int:
    """Rounds an integer to the nearest multiple of 32 to prevent Flux VAE tensor size mismatch."""
    return int(round(val / 32) * 32)


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
                )
            else:
                # Single File Loader (Needs explicit encoders usually if not in file)
                # Attempt 1: Try default loading (might fail if weights missing)
                try:
                    self.pipeline = FluxPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.bfloat16
                    )
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
                         )
                     else:
                         raise e

            # Optimization for Mac
            if self.device == "mps" and self.pipeline:
                # Recommended for Flux on Mac
                logging.info("   ⚡ Enabling Model CPU Offload for MacOS MPS limits...")
                self.pipeline.enable_model_cpu_offload(device=self.device)
                
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

    def generate(self, prompt, width=1024, height=1024, steps=4, seed=None, guidance_scale=3.5, image=None, strength=0.5):
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
        if len(prompt) > 1024:
            logging.warning(f"   ✂️ Truncating long prompt ({len(prompt)} chars).")
            safe_prompt = prompt[:1024]
            
        try:
            with torch.inference_mode():
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
            from diffusers import FluxImg2ImgPipeline
            import inspect

            # 1. Component Casting (Primary Path)
            # Cast T2I pipeline components into a proper FluxImg2ImgPipeline.
            # This gives us native 'strength' and 'denoising_start' support
            # while reusing the already-loaded model weights (zero VRAM cost).
            if self.pipeline:
                try:
                    pipe_cls = self.pipeline.__class__.__name__
                    logging.info(f"   🔧 Casting {pipe_cls} → FluxImg2ImgPipeline (Shared Components)...")
                    
                    # Get components and log them for diagnostics
                    comps = self.pipeline.components
                    logging.info(f"   📦 Components available: {list(comps.keys())}")
                    
                    # Inject missing T5 components (distilled models like Klein/Schnell lack these)
                    if "text_encoder_2" not in comps or "tokenizer_2" not in comps:
                        logging.info("   📚 Injecting T5 encoder/tokenizer for Img2Img compatibility...")
                        t5_local_path = "/Volumes/XMVPX/mw/t5weights-root"
                        if os.path.exists(t5_local_path):
                            logging.info(f"      📚 Loading T5 from Local: {t5_local_path}")
                            comps["text_encoder_2"] = T5EncoderModel.from_pretrained(t5_local_path, torch_dtype=torch.bfloat16)
                            comps["tokenizer_2"] = T5TokenizerFast.from_pretrained(t5_local_path)
                        else:
                            logging.info("      ☁️ Loading T5 from Hub (city96/t5-v1_1-xxl-encoder-bf16)...")
                            comps["text_encoder_2"] = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=torch.bfloat16)
                            comps["tokenizer_2"] = T5TokenizerFast.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16")
                    
                    # Fill optional components the constructor expects
                    for opt_key in ["image_encoder", "feature_extractor"]:
                        if opt_key not in comps:
                            comps[opt_key] = None
                    
                    # Cast — Apply components (don't force to device explicitly here)
                    self.img2img_pipeline = FluxImg2ImgPipeline(**comps)
                    
                    # Verify the cast worked and has strength
                    cast_args = inspect.signature(self.img2img_pipeline.__call__).parameters
                    has_strength = "strength" in cast_args
                    has_denoising = "denoising_start" in cast_args
                    logging.info(f"   ✅ Flux Img2Img Ready (Shared Components). strength={has_strength}, denoising_start={has_denoising}")
                except Exception as e:
                    logging.warning(f"   ⚠️ Cannot cast to FluxImg2ImgPipeline: {type(e).__name__}: {e}")
                    self.img2img_pipeline = None
            
            if not self.img2img_pipeline:
                # 2. Independent Load (Force Standard Flux Img2Img from disk)
                logging.info("   ⚠️ Component Casting failed. Attempting independent load of FluxImg2ImgPipeline...")
                try:
                    self.img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    )
                    logging.info("   ✅ Flux Img2Img Loaded (Independent).")
                except Exception as e_ind:
                    logging.warning(f"   ⚠️ Independent Load Failed: {e_ind}")

            # 3. Last Resort: Reuse pipeline directly if it has both 'image' AND 'strength'
            if not self.img2img_pipeline and self.pipeline:
                call_args = inspect.signature(self.pipeline.__call__).parameters
                if "image" in call_args and "strength" in call_args:
                    logging.info("   ✨ Pipeline natively supports image+strength. Reusing directly.")
                    self.img2img_pipeline = self.pipeline
                elif "image" in call_args:
                    logging.warning("   ⚠️ Pipeline has 'image' but no 'strength'. Using as fallback (limited control).")
                    self.img2img_pipeline = self.pipeline

        except Exception as e:
            logging.error(f"   ❌ Failed to load Flux Img2Img: {e}")
            
        if self.device == "mps" and self.img2img_pipeline:
             logging.info("   ⚡ Enabling Model CPU Offload for FluxImg2ImgPipeline...")
             self.img2img_pipeline.enable_model_cpu_offload(device=self.device)
        
        if not self.img2img_pipeline:
            logging.warning("   ⚠️ Flux Img2Img incompatible or failed. Fallback: T2I will activate.")

    def generate_img2img(self, prompt, image, strength=0.5, width=1024, height=1024, steps=4, seed=None, guidance_scale=3.5, denoising_start=None):
        if not self.img2img_pipeline:
            self.load_img2img()
            
        if not self.img2img_pipeline:
            return None
        
        # Clamp strength to valid range
        strength = max(0.01, min(strength, 1.0))
        
        logging.info(f"   🎨 Flux Img2Img: {prompt[:40]}... (Str: {strength:.2f}, {width}x{height}, G:{guidance_scale})")
        
        # Memory Cleanup
        import gc
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        
        import inspect
        try:
            # CRITICAL: Ensure inputs are multiples of 32 for the VAE to prevent 64GB buffer size error
            fixed_width = _make_multiple_of_32(width)
            fixed_height = _make_multiple_of_32(height)
            
            # CRITICAL: Resize input image to target dimensions BEFORE passing to pipeline.
            # FluxImg2ImgPipeline derives output dims from the input image.
            # Passing height/width as separate kwargs causes a latent-space mismatch
            # that triggers a 64GB buffer allocation on MPS/Metal.
            if isinstance(image, Image.Image):
                if image.size != (fixed_width, fixed_height):
                    logging.info(f"   📐 Resizing input image {image.size} → ({fixed_width}, {fixed_height}) (multiple of 32) for Img2Img")
                    image = image.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
            
            sig = inspect.signature(self.img2img_pipeline.__call__)
            available_args = sig.parameters.keys()
            
            # NOTE: Do NOT pass height/width to Img2Img — the pipeline infers them 
            # from the input image. Passing them separately causes buffer explosions.
            kwargs = {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
            }
            
            if seed is not None:
                kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
            
            # Strength / Denoising Control
            # denoising_start overrides strength when provided (per diffusers API).
            if denoising_start is not None and "denoising_start" in available_args:
                kwargs["denoising_start"] = denoising_start
                logging.info(f"   🎛️  Using denoising_start={denoising_start:.2f} (strength ignored)")
            elif "strength" in available_args:
                kwargs["strength"] = strength
            elif "denoising_start" in available_args:
                # Map strength → denoising_start (inverse relationship)
                kwargs["denoising_start"] = 1.0 - strength
                logging.info(f"   🎛️  Mapped strength {strength:.2f} → denoising_start={1.0 - strength:.2f}")
            else:
                # Last resort: pass strength anyway (some custom pipelines accept **kwargs)
                logging.warning(f"   ⚠️ Pipeline lacks 'strength' and 'denoising_start'. Force-passing strength={strength:.2f}.")
                kwargs["strength"] = strength

            try:
                with torch.inference_mode():
                    out_img_obj = self.img2img_pipeline(**kwargs)
            except TypeError as te:
                # If strength caused the TypeError, retry without it
                if "strength" in str(te) and "strength" in kwargs:
                    logging.warning(f"   ⚠️ Pipeline rejected 'strength'. Retrying without it...")
                    del kwargs["strength"]
                    with torch.inference_mode():
                        out_img_obj = self.img2img_pipeline(**kwargs)
                else:
                    logging.error(f"   ❌ Flux Img2Img TypeError: {te}. Attempted kwargs: {list(kwargs.keys())}")
                    return None

            out_img = out_img_obj.images[0]
            del out_img_obj
            
            # Post-Gen Cleanup
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

    return _BRIDGE

def generate_via_hf_endpoint(prompt, width=1024, height=1024, steps=28, guidance=3.5, seed=None, api_key=None, endpoint_url=None, image=None, strength=0.5):
    """
    Generates an image using Hugging Face InferenceClient (fal-ai provider).
    Supports Text-to-Image and Image-to-Image.
    """
    if not InferenceClient:
        logging.error("❌ huggingface_hub not installed. Cannot use Cloud Flux.")
        return None

    if not api_key:
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        
    if not api_key:
        logging.error("❌ Missing HF_TOKEN for Cloud Flux.")
        return None
        
    # Use standard Flux.1-dev via fal-ai provider (as used in LTX Cloud Director)
    # We ignore endpoint_url unless specifically passed, but default to the Repo ID.
    model_id = "black-forest-labs/FLUX.1-dev"
    if endpoint_url and "http" not in endpoint_url:
         # If user passed a model ID as endpoint_url
         model_id = endpoint_url
    
    # Enforce multiples of 32 for HF Inference API to prevent buffer size issues
    fixed_width = _make_multiple_of_32(width)
    fixed_height = _make_multiple_of_32(height)
    
    try:
        client = InferenceClient(provider="fal-ai", api_key=api_key)
        
        # NOTE: fal-ai provider does NOT support Img2Img for Flux.
        # Skip straight to T2I to avoid wasting a round-trip on a guaranteed failure.
        if image:
            logging.info(f"   ☁️  Cloud: Img2Img not supported by fal-ai. Using T2I instead.")
            logging.info(f"   ☁️  Flux Cloud T2I (fal-ai): '{prompt[:40]}...' ({fixed_width}x{fixed_height})")
            generated_image = client.text_to_image(
                prompt=prompt,
                model=model_id,
                width=fixed_width,
                height=fixed_height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed
            )
            return generated_image
        
    except Exception as e:
        logging.error(f"   ❌ Flux Cloud Generation Failed: {e}")
        return None

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
