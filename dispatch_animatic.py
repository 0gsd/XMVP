#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import time
import torch
import gc
import random
import math
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

# MVP Shared Logic
from mvp_shared import Manifest, load_manifest, save_manifest

# MLX for Director
try:
    import mlx_lm
except ImportError:
    # Optional dependency if we just want to run flux loop? 
    # But for now hard dependency as per original file
    logging.warning("⚠️ mlx_lm not found. Director text generation will fail if needed.")
    # sys.exit(1)

# Flux Bridge
try:
    from flux_bridge import get_flux_bridge
except ImportError:
    logging.error("❌ flux_bridge not found.")
    sys.exit(1)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
# Resolution: 512x288 (16:9)
WIDTH = 512
HEIGHT = 288
DIRECTOR_MODEL_PATH = "mlx-community/gemma-2-9b-it-4bit"
ADAPTER_PATH = "adapters/director_v1"

def apply_camera_motion(image: Image.Image, zoom: float = 1.0, pan_x: int = 0, pan_y: int = 0, rotate: float = 0.0) -> Image.Image:
    """
    Applies simulated camera motion (Zoom/Pan) to the image.
    Used to create the 'Next Frame Input' for Img2Img.
    """
    w, h = image.size
    
    # 1. Zoom (Crop & Resize)
    if zoom != 1.0:
        # Center crop based on zoom factor
        # If zoom > 1.0 (Zoom In): Crop smaller area, Resize up
        # If zoom < 1.0 (Zoom Out): Paste on larger canvas? (Not handled well, assume Zoom In usually)
        
        new_w = int(w / zoom)
        new_h = int(h / zoom)
        
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = (w + new_w) // 2
        bottom = (h + new_h) // 2
        
        # Clamp
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        
    # 2. Pan (Shift & Wrap or Mirror?)
    # For Img2Img, 'Mirror' or 'Replica' padding is best to avoid black bars
    if pan_x != 0 or pan_y != 0:
        # Create a new image with same mode
        new_img = Image.new("RGB", (w, h))
        # Paste shifted
        new_img.paste(image, (-pan_x, -pan_y))
        
        # Fill edges (Simple Mirror/Smear)
        # TODO: Better inpainting? For Animatic, naive smear is okay.
        
        image = new_img
        
    return image

class DirectorEngine:
    """Manages the Director Model (Gemma) via MLX."""
    def __init__(self, model_path, adapter_path):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load(self):
        if self.model: return
        
        try:
            import mlx_lm
            logging.info(f"   🎬 Loading Director Model: {self.model_path}")
            if os.path.exists(self.adapter_path):
                logging.info(f"   ➕ With Adapter: {self.adapter_path}")
                self.model, self.tokenizer = mlx_lm.load(
                    self.model_path, 
                    adapter_path=self.adapter_path,
                    tokenizer_config={"trust_remote_code": True}
                )
            else:
                logging.warning(f"   ⚠️ Adapter not found at {self.adapter_path}. Loading Base Model.")
                self.model, self.tokenizer = mlx_lm.load(self.model_path)
        except Exception as e:
            logging.error(f"   ❌ Director Load Failed: {e}")
    
    def direct_shot(self, script_line, previous_shot_desc=None):
        """Generates a visual description from the script."""
        if not self.model: self.load()
        if not self.model: return script_line # Fallback
        
        import mlx_lm
        
        system_prompt = "You are a specialized screenplay visualization assistant. Your goal is to translate screenplay text into vivid, 'Fattened' visual descriptions aka SASSPRILLA Prompts."
        
        context = ""
        if previous_shot_desc:
            context = f"Previous Shot: {previous_shot_desc}\n"
            
        user_prompt = f"{context}Visualize this scene:\n\n{script_line}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            full_prompt = (
                f"<start_of_turn>user\n"
                f"{system_prompt}\n\n{user_prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        text = mlx_lm.generate(
            self.model, 
            self.tokenizer, 
            prompt=full_prompt,
            max_tokens=256,
            verbose=False
        )
        
        return text.strip()

def run_animatic(manifest_path, out_path, staging_dir, flux_path):
    # 1. Load Manifest
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        logging.error(f"Failed to load manifest: {e}")
        return False

    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Init Flux
    logging.info(f"   🌊 Initializing Flux Bridge from {flux_path}...")
    flux = get_flux_bridge(flux_path)
    
    # Pre-Load Img2Img Pipeline to avoid stutter on first frame
    flux.load_img2img()
    
    # 3. Init Director
    director = DirectorEngine(DIRECTOR_MODEL_PATH, ADAPTER_PATH)
    # director.load() # Lazy load on first call
    
    # 4. Loop
    logging.info(f"   🎞️ Starting Animatic Loop: {len(manifest.segs)} segments @ {WIDTH}x{HEIGHT}...")
    
    sorted_segs = sorted(manifest.segs, key=lambda s: s.id)
    last_visual = None
    last_pil_image: Image.Image = None
    
    # Animation Settings
    DRAFT_FPS = 8 # Higher FPS for smoother "Frame by Frame" feel (was 2)
    # Strength = How much to keep of previous frame (1.0 = All Prev, 0.0 = All New)
    # For Flux Img2Img pipeline, 'strength' is usually 'denoising_strength' aka 'Control'
    # Actually in diffusers: 'strength' (0.0 to 1.0). 
    # High strength (0.8) = More destruction/generation (Less of original).
    # Low strength (0.3) = Keeps original.
    # We want to KEEP the motion, so we start with a medium-low strength to preserve the 'edit' we made,
    # but let Flux dream details.
    
    DEFAULT_DENOISE = 0.65 # Balance
    
    for seg in sorted_segs:
        # Skip if done
        if seg.id in manifest.files and os.path.exists(manifest.files[seg.id]):
            logging.info(f"   ⏩ Seg {seg.id} done.")
            continue
            
        print(f"\n📍 SEGMENT {seg.id}")
        
        duration_frames = seg.end_frame - seg.start_frame
        duration_sec = max(1.0, duration_frames / 24.0)
        
        print(f"   📜 Script: \"{seg.prompt[:100]}...\" ({duration_sec:.1f}s)")
        
        # A. Direct (Gemma)
        visual_prompt = director.direct_shot(seg.prompt, last_visual)
        print(f"   🎬 Director: \"{visual_prompt[:100]}...\"")
        last_visual = visual_prompt
        
        # B. Visualize (Flux Sequence)
        num_gen_frames = max(1, int(duration_sec * DRAFT_FPS))
        
        seg_staging = staging_path / f"seg_{seg.id:03d}"
        seg_staging.mkdir(exist_ok=True)
        
        frame_paths = []
        
        # Camera Logic per Segment:
        # Randomly decide a move: Pan Left, Right, Zoom In, Zoom Out, or Static
        move_type = random.choice(["zoom_in", "zoom_out", "pan_right", "pan_left", "static", "explore"])
        
        zoom_speed = 1.0
        pan_speed_x = 0
        
        if move_type == "zoom_in": zoom_speed = 1.02 # 2% per frame
        elif move_type == "zoom_out": zoom_speed = 0.98
        elif move_type == "pan_right": pan_speed_x = -5 # shift pixels
        elif move_type == "pan_left": pan_speed_x = 5
        elif move_type == "explore":
            zoom_speed = 1.01
            pan_speed_x = random.choice([-2, 2])
            
        print(f"   🎥 Camera Grip: {move_type} (Z:{zoom_speed}, X:{pan_speed_x})")
        
        for i in range(num_gen_frames):
            frame_filename = f"frame_{i:04d}.png"
            frame_path = seg_staging / frame_filename
            
            # 1. Prepare Input
            input_image = None
            strength = DEFAULT_DENOISE
            
            if last_pil_image:
                # Apply Motion Transform (The "Edit")
                input_image = apply_camera_motion(
                    last_pil_image, 
                    zoom=zoom_speed, 
                    pan_x=pan_speed_x
                )
                
                # If this is the FIRST frame of a NEW segment, maybe we want a cut?
                # If i == 0, we might want to ignore last_pil_image from PREVIOUS segment to establish new shot.
                # Unless we want a "One Shot" flow.
                # Let's Cut on new segment i=0, unless requested otherwise?
                # Standard movie = Cut.
                if i == 0:
                     print("   ✂️  Cut! Starting new shot (Text-to-Image).")
                     input_image = None 
            
            # 2. Render
            img = flux.generate(
                prompt=visual_prompt,
                width=WIDTH,
                height=HEIGHT,
                steps=4, # Fast Schnell steps
                seed=None, # Random seed per frame? Or stable? 
                # If Img2Img, Stable seed might reduce boiling?
                # Let's try Seed = 42 + i for strict determinism per frame index
                # seed = 42 + i + (seg.id * 1000)
                image=input_image,
                strength=strength
            )
            
            if img:
                img.save(frame_path)
                frame_paths.append(str(frame_path))
                last_pil_image = img # Feedback Loop
            else:
                logging.error(f"   ❌ Frame {i} failed.")
            
            print(f"      Frame {i+1}/{num_gen_frames} | I2I: {input_image is not None}", end="\r")
            
        print("")
        
        # C. Stitch
        if not frame_paths:
             logging.warning("   ⚠️ No frames generated.")
             continue
             
        out_video_name = f"animatic_seg_{seg.id:03d}.mp4"
        out_video_path = staging_path / out_video_name
        
        list_txt_path = seg_staging / "list.txt"
        with open(list_txt_path, 'w') as f:
            for fp in frame_paths:
                f.write(f"file '{fp}'\n")
                # f.write(f"duration {1.0/DRAFT_FPS}\n") # If using simple concat
                # With -vf fps=24, we don't need line duration if we assume input is DRAFT_FPS?
                # Actually ffmpeg concat list defaults to duration?
                # Best approach: Input is sequence of images.
                f.write(f"duration {1.0/DRAFT_FPS}\n")
            # Repeat last
            f.write(f"file '{frame_paths[-1]}'\n")
            
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_txt_path),
            "-vf", f"fps=24", 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(out_video_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"   ✅ Saved {out_video_name} ({duration_sec:.1f}s)")
            manifest.files[seg.id] = str(out_video_path)
        except Exception as e:
            logging.error(f"   ❌ FFMPEG Error: {e}")

            
    # 5. Save Manifest
    save_manifest(manifest, out_path)
    logging.info(f"🎉 Animatic Wrap! Saved to {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Dispatch Animatic: High-Speed Visualization Engine")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="manifest_updated.json")
    parser.add_argument("--staging", default="componentparts")
    
    import definitions
    flux_conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-klein")
    fallback_path = flux_conf.path if flux_conf else "/Volumes/XMVPX/mw/flux-root"
    
    parser.add_argument("--flux_path", default=fallback_path)
    
    args = parser.parse_args()
    
    success = run_animatic(args.manifest, args.out, args.staging, args.flux_path)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
