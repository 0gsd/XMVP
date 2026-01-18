import os
import sys
import torch
import torchaudio
import logging
import random
import numpy as np

# Config
MW_ROOT = "/Volumes/XMVPX/mw"
CODE_ROOT = os.path.join(MW_ROOT, "hunyuan-foley-code")
MODEL_ROOT = os.path.join(MW_ROOT, "hunyuan-foley")

# Add Code to Path
if os.path.exists(CODE_ROOT):
    sys.path.append(CODE_ROOT)
    # Also add the inner directory if needed, but 'hunyuanvideo_foley' is usually top level package in repo
    # Check structure: code_root/hunyuanvideo_foley
else:
    print(f"[-] Hunyuan Code not found at {CODE_ROOT}")

# Attempt Imports
try:
    from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
    from hunyuanvideo_foley.utils.feature_utils import feature_process
    from hunyuanvideo_foley.utils.media_utils import merge_audio_video
except ImportError as e:
    print(f"[-] Hunyuan Import Failed: {e}")
    # Fallback to prevent crash during import, but class will fail
    load_model = None

class HunyuanFoleyBridge:
    def __init__(self, model_path=MODEL_ROOT, device="auto"):
        self.model_path = model_path
        self.model_dict = None
        self.cfg = None
        self.device = self._setup_device(device)
        
        # Load Model
        if load_model:
            self.load()
        else:
            print("[-] HunyuanFoley Logic not imported.")

    def _setup_device(self, device_str):
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                # MPS Support check? The code uses torch.float16 or bfloat16 often.
                # MacOS might need explicit MPS.
                return torch.device("mps") 
            return torch.device("cpu")
        return torch.device(device_str)

    def load(self):
        config_path = os.path.join(self.model_path, "config.yaml")
        # Check if XL or XXL. Standard weighting seems to be XXL (10GB file).
        # Let's check which config exists or default to code repo config if not in weights
        if not os.path.exists(config_path):
             # Try code repo config
             config_path = os.path.join(CODE_ROOT, "configs/hunyuanvideo-foley-xxl.yaml")
        
        print(f"   🌊 Loading HunyuanFoley from {self.model_path} (Config: {config_path})...")
        
        try:
            # enable_offload=True by default for consumer cards/Mac?
            # model_size="xxl" implied by config? load_model arg has model_size
            self.model_dict, self.cfg = load_model(
                self.model_path, 
                config_path, 
                self.device, 
                enable_offload=True, # Safety for VRAM
                model_size="xxl" # Assuming standard model
            )
            print("   ✅ HunyuanFoley Loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load HunyuanFoley: {e}")

    def generate_foley(self, text_prompt, video_path, output_path, duration=None, guidance_scale=4.5, steps=30):
        if not self.model_dict:
            print("   ❌ Model not loaded.")
            return False

        print(f"   🎵 Generating Foley: '{text_prompt}' (Video: {os.path.basename(video_path)})...")
        
        try:
            # Feature Process
            # neg_prompt default
            neg_prompt = "noisy, harsh, low quality, distortion"
            
            # The model seems to assume video_path exists.
            # feature_process returns: visual_feats, text_feats, audio_len_in_s
            visual_feats, text_feats, audio_len_in_s = feature_process(
                video_path,
                text_prompt,
                self.model_dict,
                self.cfg,
                neg_prompt=neg_prompt
            )
            
            # Duration Override?
            # feature_process calculates audio_len_in_s from video.
            # We can override if needed, but keeping sync is best.
            if duration:
                # If we want to extend/shorten? 
                pass 

            # Denoise
            # Note: steps=30 (tuned down from 50 default for speed?)
            audio, sample_rate = denoise_process(
                visual_feats,
                text_feats,
                audio_len_in_s,
                self.model_dict,
                self.cfg,
                guidance_scale=guidance_scale,
                num_inference_steps=steps
            )
            
            # Save
            # audio is [1, T] tensor?
            # denoise_process returns: return audio.cpu(), sample_rate
            # audio[0] based on infer.py
            
            torchaudio.save(output_path, audio[0], sample_rate)
            return True
            
        except Exception as e:
            print(f"   ❌ Foley Generation Error: {e}")
            import traceback
            traceback.print_exc()
            return False

# Singleton
_bridge = None

def generate_foley_asset(prompt, output_path, video_path=None, duration=4.0):
    global _bridge
    if not _bridge:
        _bridge = HunyuanFoleyBridge()
        
    if not video_path:
        print("   ⚠️ No video path provided for Foley. Hunyuan is Video-to-Audio.")
        # We could generate a dummy black video here if needed, but content_producer should provide it.
        # Fallback to dummy?
        # Let's create a temp black video of duration
        temp_vid = output_path.replace(".wav", "_temp_black.mp4")
        import subprocess
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', f'color=c=black:s=512x512:d={duration}', 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', temp_vid, '-y', '-loglevel', 'error'
        ])
        video_path = temp_vid
        
    success = _bridge.generate_foley(prompt, video_path, output_path, duration=duration)
    
    # Cleanup temp video if we made it
    if "temp_black" in video_path and os.path.exists(video_path):
        os.remove(video_path)
        
    return output_path if success else None
