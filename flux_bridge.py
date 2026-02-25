#!/usr/bin/env python3
import os
import io
import base64
import logging
from PIL import Image
import requests

try:
    import fal_client
except ImportError:
    fal_client = None

logging.basicConfig(level=logging.INFO)

# --- UTILS ---
def load_fal_key():
    key = os.environ.get("FAL_KEY")
    if key: return key
    
    # Try finding env_vars.yaml centrally
    from pathlib import Path
    import yaml
    
    # tools/fmv/env_vars.yaml (Central)
    central = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
    local = Path(__file__).resolve().parent / "env_vars.yaml"
    
    for p in [central, local]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "FAL_KEY" in data:
                        # Set to environ so fal_client picks it up automatically
                        val = data["FAL_KEY"]
                        os.environ["FAL_KEY"] = val
                        return val
            except Exception as e:
                pass
    return None

def pil_to_base64_uri(img, format="JPEG"):
    buffered = io.BytesIO()
    # Convert RGBA to RGB if saving as JPEG
    if format == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffered, format=format, quality=95)
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{b64}"

def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def _make_multiple_of_16(val: int) -> int:
    return int(round(val / 16.0)) * 16

class FluxBridge:
    def __init__(self, model_path, device="mps"):
        self.model_path = model_path
        self.device = device
        self.is_gguf = False # Legacy compat
        
        load_fal_key()
        if not fal_client:
            logging.warning("⚠️ fal_client not installed! pip install fal-client is required for Flux API.")
        if not os.environ.get("FAL_KEY") or os.environ.get("FAL_KEY") == "YOUR_FAL_KEY_HERE":
            logging.warning("⚠️ FAL_KEY not found or default in env_vars.yaml. API calls will fail.")
            
        logging.info(f"   ⚡ FluxBridge initialized for Cloud Inference via Fal.ai API.")

    def load_lora(self, lora_path, adapter_name="default", scale=1.0):
        # Fal.ai supports LoRAs by passing loras=[{"path": url, "scale": scale}]
        # But for 'local' files, would need upload logic. 
        # For our main pipeline, we're not heavily using local LoRAs with Flux Dev yet.
        logging.warning("   ⚠️ Local LoRA load requested but Fal.ai bridge currently ignores it. (Upload unsupported online yet)")
        return False

    def generate(self, prompt, width=1024, height=1024, steps=28, seed=None, guidance_scale=3.5, image=None, strength=0.5):
        if image is not None:
             return self.generate_img2img(
                 prompt=prompt, image=image, strength=strength, 
                 width=width, height=height, steps=steps, 
                 seed=seed, guidance_scale=guidance_scale
             )
             
        # TEXT TO IMAGE
        logging.info(f"   ☁️  Flux T2I (Fal.ai): {prompt[:40]}... ({width}x{height}, {steps} steps, G:{guidance_scale})")
        if not fal_client: return None
        
        fixed_width = _make_multiple_of_16(width)
        fixed_height = _make_multiple_of_16(height)
        
        args = {
            "prompt": prompt,
            "image_size": {
                "width": fixed_width,
                "height": fixed_height
            },
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            # "enable_safety_checker": False # Optional depending on endpoint
        }
        if seed is not None:
            args["seed"] = seed
            
        try:
            result = fal_client.subscribe(
                "fal-ai/flux/dev",
                arguments=args,
                with_logs=False
            )
            img_url = result.get('images', [{}])[0].get('url')
            if img_url:
                 return download_image(img_url)
            return None
        except Exception as e:
            logging.error(f"   ❌ Fal.ai Generation Error: {e}")
            return None

    def generate_img2img(self, prompt, image, strength=0.5, width=1024, height=1024, steps=28, seed=None, guidance_scale=3.5, denoising_start=None):
        logging.info(f"   ☁️  Flux Img2Img (Fal.ai): {prompt[:40]}... (Str: {strength:.2f}, {width}x{height}, G:{guidance_scale})")
        if not fal_client: return None

        fixed_width = _make_multiple_of_16(width)
        fixed_height = _make_multiple_of_16(height)

        # Handle PIL Image
        if isinstance(image, Image.Image):
             # Ensure dimensions match requested
             if image.size != (fixed_width, fixed_height):
                 image = image.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
             image_url = pil_to_base64_uri(image)
        elif isinstance(image, str):
             # Assume path or existing URL
             if image.startswith("http") or image.startswith("data:"):
                 image_url = image
             else:
                 img = Image.open(image).convert("RGB").resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
                 image_url = pil_to_base64_uri(img)
        else:
            logging.error("   ❌ Invalid image type passed to generate_img2img.")
            return None
            
        # Strength logic: Diffusers maps denoising_start to strength.
        if denoising_start is not None:
            strength = 1.0 - denoising_start

        # FAL.AI FLUX DEV STRENGTH MAPPER
        # The Fal.ai `flux/dev/image-to-image` endpoint has an extremely steep, non-linear strength curve.
        # Any strength < 0.90 behaves like a rigid ControlNet, ignoring prompts to retain the exact image constraints.
        # It only breaks free to allow "animation/redraw" between 0.93 and 0.98.
        # To maintain compatibility with local Diffusers expectations (where 0.70 is standard animation):
        # We remap the incoming 0.0->1.0 band into Fal's 0.85->1.0 band.
        if strength < 0.90:
             mapped_strength = 0.85 + (float(strength) * 0.15)
             logging.info(f"   🧮 Mapped Fal.ai Strength: {strength:.2f} -> {mapped_strength:.3f} to unlock movement.")
             strength = mapped_strength

        # Fal expects strength > 0.0 effectively.
        strength = max(0.01, min(strength, 1.0))
        
        args = {
            "prompt": prompt,
            "image_url": image_url,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            args["seed"] = seed
            
        try:
            result = fal_client.subscribe(
                "fal-ai/flux/dev/image-to-image",
                arguments=args,
                with_logs=False
            )
            img_url = result.get('images', [{}])[0].get('url')
            if img_url:
                 return download_image(img_url)
            return None
        except Exception as e:
            logging.error(f"   ❌ Fal.ai Img2Img Error: {e}")
            return None

    def unload(self):
        logging.info("   ✅ Fal.ai Bridge Unloaded (No-Op).")


_BRIDGES = {}
def get_flux_bridge(path):
    global _BRIDGES
    res_path = os.path.abspath(path)
    if res_path not in _BRIDGES:
        logging.info(f"   🏗️  Initializing Fal.ai Bridge for path: {res_path}...")
        _BRIDGES[res_path] = FluxBridge(res_path)
    return _BRIDGES[res_path]


def generate_via_hf_endpoint(*args, **kwargs):
    logging.warning("   ⚠️ generate_via_hf_endpoint called but local bridge is already using Fal.ai. Delegating to FluxBridge.")
    bridge = get_flux_bridge("cloud")
    return bridge.generate(*args, **kwargs)

if __name__ == "__main__":
    bridge = FluxBridge("")
    print("Bridge loaded. Ensure FAL_KEY is set in env_vars.yaml.")
