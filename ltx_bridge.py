#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevent OpenMP crash on Mac
import torch
import logging
import gc
# Standard diffusers import
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

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
            
    def _get_loading_kwargs(self):
        """Helper to get common loading arguments."""
        return {
            "torch_dtype": torch.bfloat16, # Optimized for Mac M3
            "use_safetensors": True
        }

    def load_txt2vid(self):
        """Loads the Text-to-Video pipeline."""
        if self.txt2vid_pipe: return
        
        logging.info(f"   🌊 Loading LTX Pipeline (Txt2Vid) from: {self.model_path}...")
        try:
            kwargs = self._get_loading_kwargs()
            
            if self.model_path.endswith(".safetensors"):
                # Use standard T5 encoder - LTX uses google/t5-v1_1-xxl
                # We can try to load it from cache or download it.
                # Assuming standard t5-v1_1-xxl is fine or if user has a local path.
                t5_path = "city96/t5-v1_1-xxl-encoder-bf16" # Small safe bet or full T5?
                # Actually, diffusers usually handles this if we don't hold it wrong. 
                # But single_file loading is tricky.
                
                # Check for local text encoder path in model_path directory if exists 
                # or just use the hub id.
                text_encoder = T5EncoderModel.from_pretrained(
                    "city96/t5-v1_1-xxl-encoder-bf16", 
                    torch_dtype=torch.bfloat16
                )

                self.txt2vid_pipe = LTXPipeline.from_single_file(
                    self.model_path,
                    text_encoder=text_encoder,
                    **kwargs
                )
            else:
                self.txt2vid_pipe = LTXPipeline.from_pretrained(
                    self.model_path,
                    **kwargs
                )
            
            # CPU Offload for Mac
            self.txt2vid_pipe.enable_model_cpu_offload(device=self.device)
            
            # Enable Memory Optimizations
            self.txt2vid_pipe.enable_vae_tiling()
            self.txt2vid_pipe.enable_attention_slicing()
            
            logging.info("   ✅ LTX Txt2Vid Ready.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load LTX Txt2Vid: {e}")
            raise e

    def load_img2vid(self):
        """Loads the Image-to-Video pipeline."""
        if self.img2vid_pipe: return
        
        logging.info(f"   🌊 Loading LTX Pipeline (Img2Vid) from: {self.model_path}...")
        try:
            kwargs = self._get_loading_kwargs()
            
            if self.model_path.endswith(".safetensors"):
                # Ensure T5 loaded (reuse from txt2vid if available? No, safe to reload/cache)
                text_encoder = T5EncoderModel.from_pretrained(
                    "city96/t5-v1_1-xxl-encoder-bf16", 
                    torch_dtype=torch.bfloat16
                )
                
                self.img2vid_pipe = LTXImageToVideoPipeline.from_single_file(
                    self.model_path,
                    text_encoder=text_encoder,
                    **kwargs
                )
            else:
                self.img2vid_pipe = LTXImageToVideoPipeline.from_pretrained(
                    self.model_path,
                    **kwargs
                )
            
            self.img2vid_pipe.enable_model_cpu_offload(device=self.device)
            
            # Enable Memory Optimizations
            self.img2vid_pipe.enable_vae_tiling()
            self.img2vid_pipe.enable_attention_slicing()
            
            logging.info("   ✅ LTX Img2Vid Ready.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load LTX Img2Vid: {e}")
            raise e

    def free_memory(self):
        """Frees up memory aggressively."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt, output_path, width=768, height=512, num_frames=121, fps=24, seed=None, image_path=None, num_inference_steps=40, guidance_scale=3.0):
        """
        Generates video using standard diffusers pipelines.
        """
        try:
            logging.info(f"   🎬 LTX Generating: {prompt[:40]}... (Image: {bool(image_path)})")
            logging.info(f"      ⚙️  Config: Device={self.device}, DType={self._get_loading_kwargs()['torch_dtype']}, Steps={num_inference_steps}")
            if self.device == "mps":
                try:
                    is_built = torch.backends.mps.is_built()
                    is_avail = torch.backends.mps.is_available()
                    logging.info(f"      🍎 MPS Status: Built={is_built}, Available={is_avail}")
                except:
                    pass
            
            # Cleanup
            self.free_memory()
            
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)

            video_frames = None
            
            if image_path and os.path.exists(image_path):
                # Img2Vid
                self.load_img2vid()
                if not self.img2vid_pipe: return False
                
                from diffusers.utils import load_image
                img = load_image(image_path).resize((width, height))
                
                output = self.img2vid_pipe(
                    prompt=prompt,
                    image=img,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_sequence_length=512,
                    generator=generator,
                    decode_timestep=0.0, # Optimized decoding
                    decode_noise_scale=None
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
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    max_sequence_length=512,
                    generator=generator,
                    decode_timestep=0.0,
                    decode_noise_scale=None
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
            import traceback
            traceback.print_exc()
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
    path = "/Volumes/XMVPX/mw/LT2X-root"
    if os.path.exists(path):
        bridge = LTXBridge(path)
        bridge.generate("A cinematic shot of a cyberpunk city", "test_ltx.mp4", width=512, height=320, num_frames=32)
