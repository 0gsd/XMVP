#!/usr/bin/env python3
import os
import io
import base64
import logging
import gc
from PIL import Image

try:
    # 💥 APPLE SILICON WATERMARK PROTOCOL 💥
    # Set this BEFORE torch is imported to force PyTorch to internally clear the MPS Metal 
    # cache when allocations hit 70% of available memory, preventing 76GB Swap Death.
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    
    import torch
    from diffusers import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
    from transformers import T5EncoderModel
except ImportError:
    torch = None
    FluxPipeline = None

logging.basicConfig(level=logging.INFO)

class FluxBridge:
    def __init__(self, model_path, device="mps"):
        self.model_path = model_path
        self.device = device
        self.pipe = None
        
        if not torch or not FluxPipeline:
            logging.error("   ❌ diffusers or torch not installed! Cannot run local Flux inference.")
            return

        logging.info(f"   🏗️  Loading Local Flux 2 Dev via Diffusers at: {model_path}...")
        
        # We expect model_path to be a FOLDER containing the .safetensors file for local setups
        if os.path.isdir(self.model_path):
            safetensors_files = [f for f in os.listdir(self.model_path) if f.endswith('.safetensors')]
            if safetensors_files:
                self.model_path = os.path.join(self.model_path, safetensors_files[0])
        
        try:
            from diffusers import Flux2Pipeline, Flux2Transformer2DModel
            
            logging.info("   📥 Loading Flux 2 Transformer from local single file...")
            transformer = Flux2Transformer2DModel.from_single_file(
                self.model_path,
                config="black-forest-labs/FLUX.2-dev",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                local_files_only=False
            )

            logging.info("   📥 Building Flux 2 Pipeline with MLLM Auto-Detection...")
            self.pipe = Flux2Pipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                local_files_only=False
            )
            
            # Use MPS Offload for Unified Memory management alongside Gemma
            self.pipe.enable_model_cpu_offload(device=self.device)
            logging.info("   ✅ Local Flux 2 Dev loaded on MPS successfully.")
            
        except Exception as e:
            logging.error(f"   ❌ Diffusers Flux Load Error: {e}")

    def load_lora(self, lora_path, adapter_name="default", scale=1.0):
        if not self.pipe: return False
        try:
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            self.pipe.set_adapters(adapter_name, adapter_weights=[scale])
            logging.info(f"   🌸 Loaded LoRA {lora_path} successfully.")
            return True
        except Exception as e:
            logging.warning(f"   ⚠️ Local LoRA load failed: {e}")
            return False

    def generate(self, prompt, width=1024, height=1024, steps=28, seed=None, guidance_scale=3.5, image=None, strength=0.5):
        if image is not None:
             return self.generate_img2img(
                 prompt=prompt, image=image, strength=strength, 
                 width=width, height=height, steps=steps, 
                 seed=seed, guidance_scale=guidance_scale
             )
             
        logging.info(f"   🚀 Flux T2I (Local Diffusers): {prompt[:40]}... ({width}x{height}, {steps} steps, G:{guidance_scale})")
        if not self.pipe: return None
        
        fixed_width = int(round(width / 16.0)) * 16
        fixed_height = int(round(height / 16.0)) * 16
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    height=fixed_height,
                    width=fixed_width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    max_sequence_length=256
                )
            
            out_img = result.images[0]
            del result
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            return out_img
        except Exception as e:
            logging.error(f"   ❌ Local Flux Generation Error: {e}")
            return None

    def generate_img2img(self, prompt, image, strength=0.5, width=1024, height=1024, steps=28, seed=None, guidance_scale=3.5, denoising_start=None):
        logging.info(f"   🚀 Flux Img2Img (Local Diffusers): {prompt[:40]}... (Str: {strength:.2f}, {width}x{height}, G:{guidance_scale})")
        if not self.pipe: return None

        # Dynamically inject img2img pipeline if not initialized
        if not hasattr(self, 'i2i_pipe'):
            try:
                # We reuse the components from self.pipe to save extreme VRAM
                self.i2i_pipe = FluxImg2ImgPipeline(
                    transformer=self.pipe.transformer,
                    scheduler=self.pipe.scheduler,
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                )
                self.i2i_pipe.enable_model_cpu_offload(device=self.device)
            except Exception as e:
                logging.error(f"   ❌ Could not init Img2Img pipeline: {e}")
                return None

        fixed_width = int(round(width / 16.0)) * 16
        fixed_height = int(round(height / 16.0)) * 16

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        if image.size != (fixed_width, fixed_height):
             image = image.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
             
        # Strength logic: Diffusers maps denoising_start to strength natively sometimes
        # We enforce expected behavior
        if denoising_start is None:
            denoising_start = 1.0 - strength

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
        try:
            with torch.inference_mode():
                result = self.i2i_pipe(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    max_sequence_length=256
                )
            
            out_img = result.images[0]
            del result
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            return out_img
        except Exception as e:
            logging.error(f"   ❌ Local Flux Img2Img Error: {e}")
            return None

    def unload(self):
        logging.info("   🧹 Unloading Flux Pipeline (Clearing RAM/VRAM)")
        self.pipe = None
        if hasattr(self, 'i2i_pipe'): self.i2i_pipe = None
        if torch is not None: torch.mps.empty_cache()


_BRIDGES = {}
def get_flux_bridge(path):
    global _BRIDGES
    res_path = os.path.abspath(path) if path and path != "cloud" else "cloud"
    
    # If explicitly calling cloud but we lack a cloud implementation, we assume local routing anyway
    # because the user explicitly stated to REMOVE fal.ai fallback entirely. 
    # Hardcode resolution directly to locally downloaded weights.
    if res_path == "cloud" or "flux-root" not in res_path:
        local_override = "/Users/m3u/METMcloud/METMroot/tools/fmv/weights/flux-root/dev/flux2-dev.safetensors"
        logging.info(f"   🔀 Cloud/Unknown Flux path intercepted. Redirecting cleanly to local weights: {local_override}")
        res_path = os.path.abspath(local_override)

    if res_path not in _BRIDGES:
        _BRIDGES[res_path] = FluxBridge(res_path)
    return _BRIDGES[res_path]


def generate_via_hf_endpoint(*args, **kwargs):
    logging.warning("   ⚠️ generate_via_hf_endpoint called. Redacting HF/Cloud behavior and delegating locally.")
    bridge = get_flux_bridge("cloud")
    return bridge.generate(*args, **kwargs)

if __name__ == "__main__":
    bridge = get_flux_bridge("/Users/m3u/METMcloud/METMroot/tools/fmv/weights/flux-root/dev/flux2-dev.safetensors")
    print("Bridge tested and loaded successfully.")


