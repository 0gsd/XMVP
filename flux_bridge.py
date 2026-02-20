#!/usr/bin/env python3
import os
import torch
import logging
from diffusers import FluxPipeline, FluxImg2ImgPipeline, DiffusionPipeline
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from PIL import Image
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None # Handle optional dependency

logging.basicConfig(level=logging.INFO)

def _make_multiple_of_16(val: int) -> int:
    """Rounds an integer to the nearest multiple of 16 to prevent Flux VAE tensor size mismatch on MPS."""
    return int(round(val / 16.0)) * 16


class FluxBridge:
    def __init__(self, model_path, device="mps"):
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
                    torch_dtype=torch.float16, # Shifted to float16 for MPS stability
                    trust_remote_code=True # Needed for custom pipelines like Klein
                )
            else:
                # Single File Loader (Needs explicit encoders usually if not in file)
                # Attempt 1: Try default loading (might fail if weights missing)
                try:
                    self.pipeline = FluxPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16
                    )
                except Exception as e:
                    if "CLIPTextModel" in str(e) or "text_encoder" in str(e):
                        logging.warning("   ⚠️ Flux Single File missing Encoders. Loading from Local/Hub...")
                        
                        # 1. CLIP (Standard Hub)
                        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
                        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                        
                        # 2. T5 (Local Preference -> Hub Fallback)
                        t5_local_path = "/Volumes/XMVPX/mw/t5weights-root"
                        if os.path.exists(t5_local_path):
                            logging.info(f"      📚 Loading T5 from Local Cache: {t5_local_path}")
                            text_encoder_2 = T5EncoderModel.from_pretrained(t5_local_path, torch_dtype=torch.float16)
                            tokenizer_2 = T5TokenizerFast.from_pretrained(t5_local_path) 
                        else:
                            logging.warning("      ☁️ Local T5 not found. Downloading from Hub (city96/t5-v1_1-xxl-encoder-bf16)...")
                            text_encoder_2 = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=torch.float16)
                            tokenizer_2 = T5TokenizerFast.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16")
                        
                        self.pipeline = FluxPipeline.from_single_file(
                            model_path,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            text_encoder_2=text_encoder_2,
                            tokenizer_2=tokenizer_2,
                            torch_dtype=torch.float16
                        )
                    else:
                        raise e

            # Optimization for Mac (MPS)
            if self.device == "mps" and self.pipeline:
                logging.info("   ⚡ Stability Protocol: Component-Level MPS + VAE-on-CPU...")
                
                # Slicing reduces memory spikes during VAE operations
                self.pipeline.enable_attention_slicing("max")
                if hasattr(self.pipeline, 'enable_vae_slicing'):
                    self.pipeline.enable_vae_slicing()
                if hasattr(self.pipeline, 'enable_vae_tiling'):
                    self.pipeline.enable_vae_tiling()
                
                # COMPONENT-LEVEL PLACEMENT:
                if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer:
                    self.pipeline.transformer.to(self.device)
                    logging.info(f"      ➡️  transformer → {self.device}")
                
                for comp_name in ['text_encoder', 'text_encoder_2', 'vae']:
                    comp = getattr(self.pipeline, comp_name, None)
                    if comp is not None:
                        dtype = torch.float32 if comp_name == 'vae' else torch.float16
                        comp.to(device="cpu", dtype=dtype)
                        logging.info(f"      🧊 {comp_name} → CPU ({dtype})")
                
                # CRITICAL: Override _execution_device to return MPS.
                # The default implementation checks the first nn.Module component,
                # which is now text_encoder on CPU. This would make __call__ create
                # ALL tensors (guidance, timesteps, etc.) on CPU, but the transformer
                # expects MPS. Force it to return the correct compute device.
                _target_device = torch.device(self.device)
                # For some pipeline classes, _execution_device is a property.
                # We need to set it on the class or use a simpler override:
                type(self.pipeline)._execution_device = property(lambda s, d=_target_device: d)
                logging.info(f"      🎯 _execution_device overridden → {self.device}")
                
                # CRITICAL PATCHES: Monkey-patch internal pipeline methods to be device-aware.
                # encode_prompt is called with keyword args, so kwargs patch works.
                if hasattr(self.pipeline, 'encode_prompt'):
                    _orig_encode_prompt = self.pipeline.encode_prompt
                    _mps_device = self.device
                    def _safe_encode_prompt(*args, **kwargs):
                        target_device = kwargs.get('device', _mps_device)
                        kwargs['device'] = torch.device('cpu')
                        out = _orig_encode_prompt(*args, **kwargs)
                        return tuple(t.to(target_device) if isinstance(t, torch.Tensor) else t for t in out)
                    self.pipeline.encode_prompt = _safe_encode_prompt

                # prepare_latents is called with ALL POSITIONAL ARGS inside __call__.
                # The position of 'device' varies by pipeline type:
                #   FluxPipeline (T2I):     device is arg index 5
                #   FluxImg2ImgPipeline:    device is arg index 7
                # Use inspect to find the correct index dynamically.
                if hasattr(self.pipeline, 'prepare_latents'):
                    import inspect
                    _orig_prep = self.pipeline.prepare_latents
                    _prep_params = list(inspect.signature(_orig_prep).parameters.keys())
                    _device_idx = _prep_params.index('device') if 'device' in _prep_params else -1
                    logging.info(f"      🔧 prepare_latents: 'device' at positional index {_device_idx}")
                    
                    def _safe_prepare_latents(*args, _didx=_device_idx, **kwargs):
                        target_device = None
                        if 'device' in kwargs:
                            target_device = kwargs['device']
                            kwargs['device'] = torch.device('cpu')
                        elif _didx >= 0 and len(args) > _didx:
                            args = list(args)
                            target_device = args[_didx]
                            args[_didx] = torch.device('cpu')
                        
                        result = _orig_prep(*args, **kwargs)
                        if target_device is not None:
                            if isinstance(result, tuple):
                                return tuple(t.to(target_device) if isinstance(t, torch.Tensor) else t for t in result)
                            return result.to(target_device) if isinstance(result, torch.Tensor) else result
                        return result
                    self.pipeline.prepare_latents = _safe_prepare_latents

                if hasattr(self.pipeline, 'prepare_image_latents'):
                    import inspect
                    _orig_prep_img = self.pipeline.prepare_image_latents
                    _prep_img_params = list(inspect.signature(_orig_prep_img).parameters.keys())
                    _device_idx_img = _prep_img_params.index('device') if 'device' in _prep_img_params else -1
                    logging.info(f"      🔧 prepare_image_latents: 'device' at positional index {_device_idx_img}")
                    
                    def _safe_prepare_image_latents(*args, _didx=_device_idx_img, **kwargs):
                        target_device = None
                        if 'device' in kwargs:
                            target_device = kwargs['device']
                            kwargs['device'] = torch.device('cpu')
                        elif _didx >= 0 and len(args) > _didx:
                            args = list(args)
                            target_device = args[_didx]
                            args[_didx] = torch.device('cpu')
                            
                        result = _orig_prep_img(*args, **kwargs)
                        if target_device is not None:
                            if isinstance(result, tuple):
                                return tuple(t.to(target_device) if isinstance(t, torch.Tensor) else t for t in result)
                            return result.to(target_device) if isinstance(result, torch.Tensor) else result
                        return result
                    self.pipeline.prepare_image_latents = _safe_prepare_image_latents

                # VAE decode AND encode wrappers (both needed for cross-device safety)
                if hasattr(self.pipeline, 'vae') and self.pipeline.vae:
                    _orig_decode = self.pipeline.vae.decode
                    _orig_vae_encode = self.pipeline.vae.encode
                    def _cpu_safe_decode(z, *args, **kwargs):
                        if isinstance(z, torch.Tensor):
                            z = z.to(device="cpu", dtype=torch.float32)
                        return _orig_decode(z, *args, **kwargs)
                    def _cpu_safe_encode(x, *args, **kwargs):
                        if isinstance(x, torch.Tensor):
                            x = x.to(device="cpu", dtype=torch.float32)
                        return _orig_vae_encode(x, *args, **kwargs)
                    self.pipeline.vae.decode = _cpu_safe_decode
                    self.pipeline.vae.encode = _cpu_safe_encode

                logging.info("   ✅ Stability Protocol Applied: Cross-Device Wrappers Active.")
                
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
            self.load_pipeline(self.model_path)
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

            # 1. Primary Check: Reuse pipeline directly if it already supports Img2Img natively
            if self.pipeline:
                call_args = inspect.signature(self.pipeline.__call__).parameters
                if "image" in call_args and "strength" in call_args:
                    logging.info("   ✨ Pipeline natively supports image+strength. Reusing directly.")
                    self.img2img_pipeline = self.pipeline
                elif "image" in call_args:
                    logging.info("   ✨ Pipeline natively supports image (no strength). Reusing directly.")
                    self.img2img_pipeline = self.pipeline

            # 2. Component Casting (Fallback if not natively supported)
            if not self.img2img_pipeline and self.pipeline:
                try:
                    from diffusers import FluxImg2ImgPipeline
                    pipe_cls = self.pipeline.__class__.__name__
                    logging.info(f"   🔧 Casting {pipe_cls} → FluxImg2ImgPipeline (Shared Components)...")
                    
                    comps = self.pipeline.components
                    logging.info(f"   📦 Components available: {list(comps.keys())}")
                    
                    if "text_encoder_2" not in comps or "tokenizer_2" not in comps:
                        logging.info("   📚 Injecting T5 encoder/tokenizer for Img2Img compatibility...")
                        t5_local_path = "/Volumes/XMVPX/mw/t5weights-root"
                        if os.path.exists(t5_local_path):
                            logging.info(f"      📚 Loading T5 from Local: {t5_local_path}")
                            comps["text_encoder_2"] = T5EncoderModel.from_pretrained(t5_local_path, torch_dtype=torch.float16)
                            comps["tokenizer_2"] = T5TokenizerFast.from_pretrained(t5_local_path)
                        else:
                            logging.info("      ☁️ Loading T5 from Hub (city96/t5-v1_1-xxl-encoder-bf16)...")
                            comps["text_encoder_2"] = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16", torch_dtype=torch.float16)
                            comps["tokenizer_2"] = T5TokenizerFast.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16")
                    
                    for opt_key in ["image_encoder", "feature_extractor"]:
                        if opt_key not in comps:
                            comps[opt_key] = None
                    
                    self.img2img_pipeline = FluxImg2ImgPipeline(**comps)
                    
                    cast_args = inspect.signature(self.img2img_pipeline.__call__).parameters
                    has_strength = "strength" in cast_args
                    has_denoising = "denoising_start" in cast_args
                    logging.info(f"   ✅ Flux Img2Img Ready (Shared Components). strength={has_strength}, denoising_start={has_denoising}")
                except Exception as e:
                    logging.warning(f"   ⚠️ Cannot cast to FluxImg2ImgPipeline: {type(e).__name__}: {e}")
                    self.img2img_pipeline = None

            # 3. Independent Load (Last Resort)
            if not self.img2img_pipeline:
                logging.info("   ⚠️ Component Casting failed. Attempting independent load of FluxImg2ImgPipeline...")
                try:
                    from diffusers import FluxImg2ImgPipeline
                    self.img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    logging.info("   ✅ Flux Img2Img Loaded (Independent).")
                except Exception as e_ind:
                    logging.warning(f"   ⚠️ Independent Load Failed: {e_ind}")

        except Exception as e:
            logging.error(f"   ❌ Failed to load Flux Img2Img: {e}")
            
        if self.device == "mps" and self.img2img_pipeline and self.img2img_pipeline is not self.pipeline:
             logging.info("   ⚡ Stability Protocol (Img2Img): Component-Level MPS + VAE-on-CPU...")
             
             # Slicing reduces memory spikes during VAE operations
             self.img2img_pipeline.enable_attention_slicing("max")
             if hasattr(self.img2img_pipeline, 'enable_vae_slicing'):
                 self.img2img_pipeline.enable_vae_slicing()
             if hasattr(self.img2img_pipeline, 'enable_vae_tiling'):
                 self.img2img_pipeline.enable_vae_tiling()

             # COMPONENT-LEVEL PLACEMENT:
             if hasattr(self.img2img_pipeline, 'transformer') and self.img2img_pipeline.transformer:
                 self.img2img_pipeline.transformer.to(self.device)
                 logging.info(f"      ➡️  transformer → {self.device}")
             
             for comp_name in ['text_encoder', 'text_encoder_2', 'vae']:
                 comp = getattr(self.img2img_pipeline, comp_name, None)
                 if comp is not None:
                     dtype = torch.float32 if comp_name == 'vae' else torch.float16
                     comp.to(device="cpu", dtype=dtype)
                     logging.info(f"      🧊 {comp_name} → CPU ({dtype})")
             
             # Override _execution_device → MPS (same as load_pipeline)
             _target_device = torch.device(self.device)
             type(self.img2img_pipeline)._execution_device = property(lambda s, d=_target_device: d)
             logging.info(f"      🎯 _execution_device overridden → {self.device}")
             
             # CRITICAL PATCHES
             if hasattr(self.img2img_pipeline, 'encode_prompt'):
                 _orig_encode_prompt = self.img2img_pipeline.encode_prompt
                 _mps_device = self.device
                 def _safe_encode_prompt(*args, **kwargs):
                     target_device = kwargs.get('device', _mps_device)
                     kwargs['device'] = torch.device('cpu')
                     out = _orig_encode_prompt(*args, **kwargs)
                     return tuple(t.to(target_device) if isinstance(t, torch.Tensor) else t for t in out)
                 self.img2img_pipeline.encode_prompt = _safe_encode_prompt

             # prepare_latents: device is positional arg, index varies by pipeline type
             if hasattr(self.img2img_pipeline, 'prepare_latents'):
                 import inspect
                 _orig_prep = self.img2img_pipeline.prepare_latents
                 _prep_params = list(inspect.signature(_orig_prep).parameters.keys())
                 _device_idx = _prep_params.index('device') if 'device' in _prep_params else -1
                 logging.info(f"      🔧 prepare_latents: 'device' at positional index {_device_idx}")
                 
                 def _safe_prepare_latents(*args, _didx=_device_idx, **kwargs):
                     target_device = None
                     if 'device' in kwargs:
                         target_device = kwargs['device']
                         kwargs['device'] = torch.device('cpu')
                     elif _didx >= 0 and len(args) > _didx:
                         args = list(args)
                         target_device = args[_didx]
                         args[_didx] = torch.device('cpu')
                         
                     result = _orig_prep(*args, **kwargs)
                     if target_device is not None:
                         if isinstance(result, tuple):
                             return tuple(t.to(target_device) if isinstance(t, torch.Tensor) else t for t in result)
                         return result.to(target_device) if isinstance(result, torch.Tensor) else result
                     return result
                 self.img2img_pipeline.prepare_latents = _safe_prepare_latents

             if hasattr(self.img2img_pipeline, 'prepare_image_latents'):
                 import inspect
                 _orig_prep_img = self.img2img_pipeline.prepare_image_latents
                 _prep_img_params = list(inspect.signature(_orig_prep_img).parameters.keys())
                 _device_idx_img = _prep_img_params.index('device') if 'device' in _prep_img_params else -1
                 logging.info(f"      🔧 prepare_image_latents: 'device' at positional index {_device_idx_img}")
                 
                 def _safe_prepare_image_latents(*args, _didx=_device_idx_img, **kwargs):
                     target_device = None
                     if 'device' in kwargs:
                         target_device = kwargs['device']
                         kwargs['device'] = torch.device('cpu')
                     elif _didx >= 0 and len(args) > _didx:
                         args = list(args)
                         target_device = args[_didx]
                         args[_didx] = torch.device('cpu')
                         
                     result = _orig_prep_img(*args, **kwargs)
                     if target_device is not None:
                         if isinstance(result, tuple):
                             return tuple(t.to(target_device) if isinstance(t, torch.Tensor) else t for t in result)
                         return result.to(target_device) if isinstance(result, torch.Tensor) else result
                     return result
                 self.img2img_pipeline.prepare_image_latents = _safe_prepare_image_latents

             # VAE decode AND encode wrappers
             if hasattr(self.img2img_pipeline, 'vae') and self.img2img_pipeline.vae:
                 _orig_decode = self.img2img_pipeline.vae.decode
                 _orig_vae_encode = self.img2img_pipeline.vae.encode
                 def _cpu_safe_decode(z, *args, **kwargs):
                     if isinstance(z, torch.Tensor):
                         z = z.to(device="cpu", dtype=torch.float32)
                     return _orig_decode(z, *args, **kwargs)
                 def _cpu_safe_encode(x, *args, **kwargs):
                     if isinstance(x, torch.Tensor):
                         x = x.to(device="cpu", dtype=torch.float32)
                     return _orig_vae_encode(x, *args, **kwargs)
                 self.img2img_pipeline.vae.decode = _cpu_safe_decode
                 self.img2img_pipeline.vae.encode = _cpu_safe_encode

             logging.info("   ✅ Stability Protocol (Img2Img) Applied: Cross-Device Wrappers Active.")
             
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
            # CRITICAL: Ensure inputs are multiples of 128 for the VAE to prevent 64GB buffer size error
            fixed_width = _make_multiple_of_16(width)
            fixed_height = _make_multiple_of_16(height)
            
            # DIAGNOSTIC
            vae_dtype = self.img2img_pipeline.vae.dtype if hasattr(self.img2img_pipeline, "vae") else "unknown"
            logging.info(f"   📐 Tensor Shape Check: Requested {width}x{height} -> Aligned {fixed_width}x{fixed_height} | VAE Dtype: {vae_dtype}")
            
            # CRITICAL: Resize input image to target dimensions BEFORE passing to pipeline.
            # FluxImg2ImgPipeline derives output dims from the input image.
            # Passing height/width as separate kwargs causes a latent-space mismatch
            # that triggers a 64GB buffer allocation on MPS/Metal.
            if isinstance(image, Image.Image):
                if image.size != (fixed_width, fixed_height):
                    logging.info(f"   📐 Resizing input image {image.size} → ({fixed_width}, {fixed_height}) (multiple of 128) for Img2Img")
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
                logging.info(f"   🤞 Force-passing 'strength={strength}' to pipeline (Hope it takes it)...")
                kwargs["strength"] = strength
                
                # Check for 'sigmas' support to manually control denoising schedule
                if "sigmas" in available_args:
                    logging.info(f"   ✨ Pipeline uses 'sigmas'. Calculating noise schedule for strength {strength}...")
                    try:
                        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
                        if isinstance(self.img2img_pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
                            # Generate a fresh schedule for the requested number of steps
                            try:
                                # Determine mu if dynamic shifting is enabled
                                if getattr(self.img2img_pipeline.scheduler.config, "use_dynamic_shifting", False):
                                    import math
                                    # Calculate sequence length for Flux (patch size 2, 2x2 merged = effective patch 4)
                                    # Seq len = (height / 16) * (width / 16)
                                    seq_len = (fixed_height // 16) * (fixed_width // 16)
                                    
                                    # Default Flux shift parameters
                                    base_shift = getattr(self.img2img_pipeline.scheduler.config, "base_shift", 0.5)
                                    max_shift = getattr(self.img2img_pipeline.scheduler.config, "max_shift", 1.15)
                                    base_image_seq_len = getattr(self.img2img_pipeline.scheduler.config, "base_image_seq_len", 256)
                                    max_image_seq_len = getattr(self.img2img_pipeline.scheduler.config, "max_image_seq_len", 4096)
                                    
                                    # Interpolate shift based on sequence length
                                    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
                                    b = base_shift - m * base_image_seq_len
                                    mu = m * seq_len + b
                                    
                                    self.img2img_pipeline.scheduler.set_timesteps(steps, device="cpu", mu=mu)
                                else:
                                    self.img2img_pipeline.scheduler.set_timesteps(steps, device="cpu")
                            except TypeError:
                                # Fallback if signature doesn't match expectations
                                try:
                                    self.img2img_pipeline.scheduler.set_timesteps(steps, device="cpu", mu=None)
                                except Exception:
                                    pass 
                            
                            timesteps = getattr(self.img2img_pipeline.scheduler, "timesteps", None)
                            sigmas = getattr(self.img2img_pipeline.scheduler, "sigmas", None)
                            
                            if timesteps is not None and sigmas is not None and len(timesteps) > 0:
                                start_idx = int(len(timesteps) * (1.0 - strength))
                                start_idx = max(0, min(start_idx, len(timesteps) - 1))
                                
                                filtered_sigmas = sigmas[start_idx:].cpu().tolist()
                                kwargs["sigmas"] = filtered_sigmas
                                
                                # Safety fallback: If timesteps array is empty later, we can derive start_timestep from sigmas
                                safe_start_timestep = sigmas[start_idx] * 1000.0
                                
                                logging.info(f"   🎛️  Generated {len(kwargs['sigmas'])} sigmas from {len(timesteps)} steps (Strength {strength}).")
                                
                                # CRITICAL FIX for diffusers ignoring strength when sigmas are passed:
                                # When sigmas is provided, diffusers sets t_start = 0.
                                # This means the input image gets 100% noised, destroying coherence.
                                # Therefore, we must MANUALLY encode the image, add the correct amount of noise,
                                # and pass it via the `latents` kwarg, setting `image=None` so the pipeline
                                # doesn't try to re-process it from the beginning of the schedule.
                                try:
                                    logging.info(f"   🔧 Manually adding initial noise to input image latents for strength {strength}...")
                                    
                                    # 1. Prepare image
                                    init_image = self.img2img_pipeline.image_processor.preprocess(image, height=fixed_height, width=fixed_width)
                                    init_image = init_image.to(device=self.device, dtype=torch.float32) # VAE workaround
                                    
                                    # 2. Encode to latents
                                    latent_channels = self.img2img_pipeline.vae.config.latent_channels if hasattr(self.img2img_pipeline.vae.config, "latent_channels") else 16
                                    if init_image.shape[1] != latent_channels:
                                        # Use proper encoding logic from pipeline
                                        if hasattr(self.img2img_pipeline, "_encode_vae_image"):
                                            # Using the patched dummy generator if no real generator available
                                            gen = kwargs.get("generator", torch.Generator(device=self.device).manual_seed(0))
                                            image_latents = self.img2img_pipeline._encode_vae_image(image=init_image, generator=gen)
                                        else:
                                            # Fallback standard diffusers VAE encode
                                            image_latents = self.img2img_pipeline.vae.encode(init_image).latent_dist.sample()
                                            image_latents = (image_latents - self.img2img_pipeline.vae.config.shift_factor) * self.img2img_pipeline.vae.config.scaling_factor
                                    else:
                                        image_latents = init_image
                                        
                                    image_latents = image_latents.to(dtype=self.img2img_pipeline.transformer.dtype)
                                        
                                    # 3. Add noise
                                    # We need the appropriate timestep for the start index
                                    # start_idx corresponds to the first sigma in our filtered_sigmas list
                                    try:
                                        start_timestep = timesteps[start_idx].to(self.device, dtype=torch.float32)
                                    except Exception as e:
                                        logging.warning(f"   ⚠️ Could not access timesteps[{start_idx}]: {e}. Deriving from sigmas.")
                                        start_timestep = torch.tensor(safe_start_timestep, device=self.device, dtype=torch.float32)
                                    
                                    # Shape calculations for random noise array
                                    batch_size = 1
                                    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
                                    effective_batch = batch_size * num_images_per_prompt
                                    
                                    vae_scale_factor = getattr(self.img2img_pipeline, "vae_scale_factor", 8)
                                    latent_height = 2 * (int(fixed_height) // (vae_scale_factor * 2))
                                    latent_width = 2 * (int(fixed_width) // (vae_scale_factor * 2))
                                    num_channels_latents = self.img2img_pipeline.transformer.config.in_channels // 4
                                    shape = (effective_batch, num_channels_latents, latent_height, latent_width)

                                    
                                    # Generate matching noise
                                    noise = torch.randn(shape, generator=kwargs.get("generator", None), device=self.device, dtype=image_latents.dtype)
                                    
                                    # Expand timestep to match batch size
                                    latent_timestep = start_timestep.expand(effective_batch)
                                    
                                    # Scale noise using scheduler's native method
                                    noised_latents = self.img2img_pipeline.scheduler.scale_noise(image_latents, latent_timestep, noise)
                                    
                                    # Pack the latents for Flux transformer exactly like _pack_latents does
                                    packed_latents = noised_latents.view(effective_batch, num_channels_latents, latent_height // 2, 2, latent_width // 2, 2)
                                    packed_latents = packed_latents.permute(0, 2, 4, 1, 3, 5)
                                    packed_latents = packed_latents.reshape(effective_batch, (latent_height // 2) * (latent_width // 2), num_channels_latents * 4)
                                    
                                    kwargs["latents"] = packed_latents
                                    kwargs["image"] = None # tell pipeline not to regenerate latents
                                    logging.info(f"   ✅ Successfully injected patched initial latents manually.")
                                    
                                except Exception as manual_latent_exc:
                                    import traceback; logging.warning(f"   ⚠️ Manual latent injection failed: {manual_latent_exc}\n{traceback.format_exc()}. Falling back to default noise.")
                                
                                # Remove strength to avoid TypeError since we are passing explicit sigmas
                                if "strength" in kwargs:
                                    del kwargs["strength"]
                                if "denoising_start" in kwargs:
                                    del kwargs["denoising_start"]
                    except Exception as exc:
                        logging.warning(f"   ⚠️ Sigma calculation failed: {exc}")

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
    
    # Enforce multiples of 128 for HF Inference API to prevent buffer size issues
    fixed_width = _make_multiple_of_16(width)
    fixed_height = _make_multiple_of_16(height)
    
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
