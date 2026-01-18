#!/usr/bin/env python3
import os
import torch
import logging
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video
import gc

logging.basicConfig(level=logging.INFO)

class LTXBridge:
    def __init__(self, model_path, device="mps"):
        self.model_path = model_path
        self.device = device
        self.txt2vid_pipe = None
        self.img2vid_pipe = None
        
        # Check availability
        if device == "mps" and not torch.backends.mps.is_available():
            logging.warning("⚠️ MPS not available. Falling back to CPU.")
            self.device = "cpu"
            
    def load_txt2vid(self):
        """Loads the Text-to-Video pipeline."""
        if self.txt2vid_pipe: return
        
        logging.info(f"   🌊 Loading LTX Pipeline (Txt2Vid) from: {self.model_path}...")
        try:
            self.txt2vid_pipe = LTXPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16, # bfloat16 might be better for MPS? Let's try float16 standard first.
                use_safetensors=True
            ).to(self.device)
            
            # CPU Offload for Mac
            self.txt2vid_pipe.enable_model_cpu_offload()
            self.txt2vid_pipe.enable_vae_tiling() # Maybe needed for 4K?
            
            logging.info("   ✅ LTX Txt2Vid Ready.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load LTX Txt2Vid: {e}")
            raise e

    def load_img2vid(self):
        """Loads the Image-to-Video pipeline."""
        if self.img2vid_pipe: return
        
        logging.info(f"   🌊 Loading LTX Pipeline (Img2Vid) from: {self.model_path}...")
        try:
            self.img2vid_pipe = LTXImageToVideoPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(self.device)
            
            self.img2vid_pipe.enable_model_cpu_offload()
            
            logging.info("   ✅ LTX Img2Vid Ready.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load LTX Img2Vid: {e}")
            raise e

    def generate(self, prompt, output_path, width=768, height=512, num_frames=121, fps=24, seed=None, image_path=None):
        """
        Generates video.
        If image_path is provided, uses Img2Vid.
        num_frames: LTX default is often 121 (for 4s at 24fps?)
        """
        try:
            logging.info(f"   🎬 LTX Generating: {prompt[:40]}... (Image: {bool(image_path)})")
            
            # Cleanup
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)

            video_frames = None
            
            if image_path and os.path.exists(image_path):
                # Img2Vid
                self.load_img2vid()
                if not self.img2vid_pipe: return False
                
                from diffusers.utils import load_image
                img = load_image(image_path).resize((width, height)) # Resize input to match target
                
                output = self.img2vid_pipe(
                    prompt=prompt,
                    image=img,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    guidance_scale=3.0,
                    max_sequence_length=512,
                    generator=generator
                )
                video_frames = output.frames[0]
                
            else:
                # Txt2Vid
                self.load_txt2vid()
                if not self.txt2vid_pipe: return False
                
                output = self.txt2vid_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    guidance_scale=3.0,
                    max_sequence_length=512,
                    generator=generator
                )
                video_frames = output.frames[0]
            
            # Save
            if video_frames:
                export_to_video(video_frames, output_path, fps=fps)
                logging.info(f"   💾 Saved to {output_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"   ❌ LTX Generation Error: {e}")
            return False

# Singleton
_BRIDGE = None
def get_ltx_bridge(path):
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = LTXBridge(path)
    return _BRIDGE

if __name__ == "__main__":
    # Test
    path = "/Volumes/XMVPX/mw/LT2X-root" # Assuming directory structure
    if os.path.exists(path):
        bridge = LTXBridge(path)
        bridge.generate("A cinematic shot of a cyberpunk city, rain, neon lights", "test_ltx.mp4", width=512, height=384, num_frames=32, fps=8)
