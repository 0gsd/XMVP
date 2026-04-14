"""
MFLUX BRIDGE (Apple Silicon Native MLX)
Replaces Diffusers `flux_bridge.py` to prevent "Swap Death" Memory Crashes.
Using FLUX.2-klein-4B quantized.
"""

import os
import logging
from PIL import Image

# Enable blazing fast hugs
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein
    MFLUX_AVAILABLE = True
except ImportError:
    MFLUX_AVAILABLE = False
    
class MFluxBridge:
    def __init__(self, model_path=None, quantize=8):
        self.quantize = quantize
        self.pipe = None
        self.i2i_pipe = None
        
        if not MFLUX_AVAILABLE:
            logging.error("   ❌ MFLUX not installed. Run: pip install mflux hf_transfer")
            return
            
        logging.info(f"   📥 Initializing MFLUX FLUX.2-klein-4B (Q{quantize} bit)...")
        # Initialize T2I
        try:
            self.pipe = Flux2Klein(
                model_config=ModelConfig.flux2_klein_4b(),
                quantize=self.quantize
            )
        except Exception as e:
            logging.error(f"   ❌ MFLUX T2I Init Error: {e}")

    def generate(self, prompt, image=None, strength=0.5, width=1024, height=1024, steps=4, seed=None, guidance_scale=1.0, denoising_start=None):
        # Distilled Models MUST use 1.0 Guidance, but steps are now caller-controllable.
        # Klein architecture sweet spot: 4-16 for T2I. Above 16 shows diminishing returns.
        guidance_scale = 1.0
        steps = max(4, min(steps, 16))
        
        if image is not None:
             return self.generate_img2img(prompt=prompt, image=image, strength=strength, width=width, height=height, steps=steps, seed=seed, guidance_scale=guidance_scale, denoising_start=denoising_start)
             
        logging.info(f"   🚀 MFLUX T2I: {prompt[:40]}... ({width}x{height}, {steps} steps, G:{guidance_scale})")
        if not self.pipe: return None
        
        fixed_width = int(round(width / 16.0)) * 16
        fixed_height = int(round(height / 16.0)) * 16
        
        img_seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
        
        try:
            image = self.pipe.generate_image(
                prompt=prompt,
                width=fixed_width,
                height=fixed_height,
                num_inference_steps=steps,
                seed=img_seed
            )
            return image
        except KeyboardInterrupt:
            logging.info("🚪 Ctrl-C Detected! Aborting T2I generation.")
            raise
        except Exception as e:
            logging.error(f"   ❌ Local MFLUX Generation Error: {e}")
            return None

    def generate_img2img(self, prompt, image, strength=0.5, width=1024, height=1024, steps=4, seed=None, guidance_scale=1.0, denoising_start=None):
        # Klein I2I sweet spot: 4-12 steps (init image provides structure, fewer steps needed)
        guidance_scale = 1.0
        steps = max(4, min(steps, 12))
        logging.info(f"   🚀 MFLUX Local I2I: {prompt[:40]}... (Str: {strength:.2f}, {width}x{height})")
        
        fixed_width = int(round(width / 16.0)) * 16
        fixed_height = int(round(height / 16.0)) * 16
        img_seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
        
        if image.size != (fixed_width, fixed_height):
             image = image.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
             
        # MFLUX Base pipe supports init-image seamlessly!
        temp_path = "/tmp/mflux_i2i_temp.png"
        image.save(temp_path)
        
        # MFLUX `image_strength` represents how much of the original image to preserve
        # (by skipping time steps). A value of 1.0 means 100% preservation (no noise).
        # Diffusers `strength` represents NOISE amount (1.0 = full noise).
        # We invert it here to match expectations.
        mflux_image_strength = max(0.01, min(0.99, 1.0 - strength))
        
        try:
            out_image = self.pipe.generate_image(
                prompt=prompt,
                image_path=temp_path,
                image_strength=mflux_image_strength,
                width=fixed_width,
                height=fixed_height,
                num_inference_steps=steps,
                seed=img_seed
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return out_image
        except KeyboardInterrupt:
            logging.info("🚪 Ctrl-C Detected! Aborting I2I generation.")
            raise
        except Exception as e:
            logging.error(f"   ❌ Local MFLUX I2I Error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

_bridge = None
def get_flux_bridge(path=None):
    global _bridge
    if not _bridge:
        _bridge = MFluxBridge()
    return _bridge
