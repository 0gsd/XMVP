import argparse
import logging
import os
import sys
import time
import re
import yaml
import random
import math
import shutil
import glob
from pathlib import Path
from PIL import Image

# Third Party (Check availability)
try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ OpenCV or Numpy missing. Install with: pip install opencv-python numpy")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Google GenAI SDK missing.")
    sys.exit(1)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
ENV_FILE = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
# Fallback to local if relative path fails (e.g. running from deeper dir)
if not ENV_FILE.exists():
    ENV_FILE = Path("tools/fmv/env_vars.yaml").resolve()

def load_keys():
    """Loads keys from centralized env_vars.yaml."""
    if not ENV_FILE.exists():
        logging.error(f"❌ Config not found: {ENV_FILE}")
        return []
    try:
        with open(ENV_FILE, 'r') as f:
            data = yaml.safe_load(f)
            # Try list or string
            keys = data.get("GEMINI_API_KEYS") or data.get("ACTION_KEYS_LIST")
            if isinstance(keys, str):
                return [k.strip() for k in keys.split(',') if k.strip()]
            if isinstance(keys, list):
                return keys
            return []
    except Exception as e:
        logging.error(f"❌ Config load error: {e}")
        return []

def get_random_key(keys):
    if not keys: return None
    return random.choice(keys)

# --- CLASSES ---

class FrameUpscaler:
    """
    The 'Svelte 2x-ing Machine'.
    Ported from 0opsgan/aw_redoer.py (ObsessiveForger).
    Logic: Obsessive layering -> Quantized Resize.
    """
    def __init__(self):
        pass

    def _pil_to_cv2(self, pil_img):
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv2_img):
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    def obsess(self, pil_img, intensity=0.3):
        """
        Simplified 'Mind Vision' loop.
        Adds subtle detail, contrast, and edge definition before upscaling.
        """
        cv_img = self._pil_to_cv2(pil_img)
        
        # 1. CLAHE (Contrast Limit Adaptive Histogram Equalization)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        cv_img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. Edge Enhancement
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(cv_img, -1, kernel)
        
        # Blend (Intensity)
        blended = cv2.addWeighted(cv_img, 1.0 - intensity, sharpened, intensity, 0)
        
        return self._cv2_to_pil(blended)

    def upscale(self, image_path, output_path, scale=2.0):
        """
        Resizes using the 'Pass 3' Logic (Lanczos -> Quantize).
        """
        try:
            if not os.path.exists(image_path): return False
            img = Image.open(image_path).convert("RGB")
            
            # 1. Obsess (Detail Injection)
            img = self.obsess(img, intensity=0.2)
            
            w, h = img.size
            target_w, target_h = int(w * scale), int(h * scale)
            
            # 2. Resize Method (LANCZOS is best for upscaling before quantization)
            resized = img.resize((target_w, target_h), Image.LANCZOS)
            
            # 3. Quantize (The 'Redoer' signature look)
            # Reduces color banding introduced by smooth resize
            # We use a high palette count (255) to keep it photoreal but structured
            final = resized.quantize(colors=255, method=Image.MAXCOVERAGE, dither=Image.FLOYDSTEINBERG)
            final = final.convert("RGB")
            
            final.save(output_path)
            return True
        except Exception as e:
            logging.error(f"   ❌ Upscale Failed: {e}")
            return False

class FrameInterpolator:
    """
    The 'Hallucinated Tween' Engine.
    Uses Gemini 2.5 Flash Image to generate missing frames.
    """
    def __init__(self, keys):
        self.keys = keys
        self.model = "gemini-2.5-flash-image"

    def generate_tween(self, img_path_a, img_path_b, output_path, frame_idx_a, frame_idx_b):
        """
        Generates a frame visually between A and B.
        """
        key = get_random_key(self.keys)
        client = genai.Client(api_key=key)
        
        prompt = (
            "You are a master animator, photographic collagist, and photorealist painter "
            "with a photographic memory and 25 years of art school training. "
            f"We have lost the frame between Frame {frame_idx_a} and Frame {frame_idx_b} of this video. "
            f"Please recreate the missing middle frame _exactly_, maintaining perfect continuity. "
            "Draw the intermediate movement. If a character moves, show them halfway. "
            "Do NOT add text, timecodes, or overlays."
        )
        
        try:
            # Load images as PIL
            # Gemini client handles file paths or PIL images in contents?
            # client.models.generate_content supports PIL images directly in 'contents' list.
            
            img_a = Image.open(img_path_a)
            img_b = Image.open(img_path_b)
            
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, img_a, img_b]
            )
            
            # Extract Image
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        with open(output_path, "wb") as f:
                            f.write(part.inline_data.data)
                        return True
            
            logging.warning(f"   ⚠️ No tween generated for {frame_idx_a}->{frame_idx_b}")
            return False
            
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                logging.warning(f"   ⚠️ Quota Hit (Tweening). Skipping this tween.")
            else:
                logging.error(f"   ❌ Tween Error: {e}")
            return False

def sort_frames(frame_list):
    """Sorts frames by numeric sequence."""
    # usage: frame_001.png
    def key_func(f):
        nums = re.findall(r'\d+', f.name)
        if nums:
            return int(nums[-1])
        return 0
    return sorted(frame_list, key=key_func)

def process(args):
    keys = load_keys()
    if not keys:
        logging.error("❌ No keys found.")
        sys.exit(1)
        
    input_video = None
    input_dir = None
    
    # Identify Input
    if os.path.isfile(args.input):
        input_video = Path(args.input)
    elif os.path.isdir(args.input):
        input_dir = Path(args.input)
    else:
        logging.error(f"❌ Invalid input: {args.input}")
        sys.exit(1)
        
    # Setup Output
    ts = int(time.time())
    out_root = Path(args.output) if args.output else Path(f"post_prod_{ts}")
    if not out_root.exists(): out_root.mkdir(parents=True)
    
    stage_1_frames = out_root / "1_extracted"
    stage_2_interpolated = out_root / "2_interpolated"
    stage_3_upscaled = out_root / "3_upscaled"
    
    # 1. Extract (if video)
    if input_video:
        logging.info(f"🎞️ Extracting frames from {input_video}...")
        stage_1_frames.mkdir(exist_ok=True)
        # ffmpeg extract
        cmd = f"ffmpeg -i '{input_video}' -q:v 2 '{stage_1_frames}/frame_%05d.png' -y -loglevel error"
        os.system(cmd)
        work_frames = list(stage_1_frames.glob("*.png"))
    else:
        # Just copy extracted frames to separate logic
        logging.info(f"📂 Loading frames from {input_dir}...")
        work_frames = list(input_dir.glob("*.png"))
        if not work_frames:
             # Try jpg
             work_frames = list(input_dir.glob("*.jpg"))
    
    work_frames = sort_frames(work_frames)
    if not work_frames:
        logging.error("❌ No frames found.")
        sys.exit(1)
        
    logging.info(f"   Found {len(work_frames)} source frames.")
    
    # 2. Interpolate (Tweening)
    if args.x > 1:
        logging.info(f"⚡ Interpolating (Expansion: {args.x}x)...")
        stage_2_interpolated.mkdir(exist_ok=True)
        interpolator = FrameInterpolator(keys)
        
        # Logic: 
        # Source: [A, B, C]
        # Target: [A, Tween(A,B), B, Tween(B,C), C] (for 2x)
        # We need a re-indexing strategy.
        # Simplest: Multiply indices by Expansion Factor.
        # Frame 0 -> 0      (Source)
        # Frame 1 -> 10     (Source)
        # Tween -> 5
        
        # Just creating a new sequence list
        final_sequence = []
        
        for i in range(len(work_frames) - 1):
            curr_frame = work_frames[i]
            next_frame = work_frames[i+1]
            
            # Add Current
            final_sequence.append(("source", curr_frame))
            
            # Generate Intermediates
            # If x=2, generate 1 intermediate
            # If x=4, generate 3? (Recursive? Or just parallel prompts?)
            # User example: "Frame 2 is missing... here is 1 and 3". 
            # For x=2, it's 1 tween.
            
            count_tweens = args.x - 1
            if count_tweens > 0:
                # Naive implementation: Just 1 tween for now as requested ("--x 2 ... default")
                # User specifically asked for the "flipbook logic"
                
                # Tween Output Name (Temp)
                tween_name = f"tween_{i}_{i+1}.png"
                tween_path = stage_2_interpolated / tween_name
                
                logging.info(f"   🎨 Generating Tween {i}-{i+1}...")
                success = interpolator.generate_tween(curr_frame, next_frame, tween_path, i, i+1)
                
                if success:
                    final_sequence.append(("generated", tween_path))
                else:
                    # Fallback? Duplicate current frame to keep sync?
                    logging.warning("   ⚠️ Tween failed. Duplicating current frame.")
                    final_sequence.append(("source", curr_frame)) # Duplicate
        
        # Add Last Frame
        final_sequence.append(("source", work_frames[-1]))
        
        # Prepare for Upscaling phase: Make this the new work list
        # We don't rename yet, we just pass paths.
        work_frames_tuples = final_sequence
        logging.info(f"   ✅ Interpolation Done. Total Frames: {len(work_frames_tuples)}")
        
    else:
        # No interpolation, just wrap source frames
        work_frames_tuples = [("source", f) for f in work_frames]

    # 3. Upscale (Redo)
    if args.scale > 1:
        logging.info(f"🔍 Upscaling (Scale: {args.scale}x)...")
        stage_3_upscaled.mkdir(exist_ok=True)
        upscaler = FrameUpscaler()
        
        upscaled_paths = []
        
        # Process every frame in the sequence
        # Renumber them sequentially here: frame_00001.png
        
        for idx, (ftype, path) in enumerate(work_frames_tuples):
            out_name = f"frame_{idx:05d}.png"
            out_path = stage_3_upscaled / out_name
            
            logging.info(f"   🚀 Upscaling Frame {idx+1}/{len(work_frames_tuples)} ({ftype})...")
            success = upscaler.upscale(path, out_path, scale=args.scale)
            
            if success:
                upscaled_paths.append(out_path)
            else:
                # Copy original if upscale fails (fallback)
                shutil.copy(path, out_path)
                upscaled_paths.append(out_path)
                
        logging.info(f"   ✅ Upscaling Done.")
        final_frames_dir = stage_3_upscaled
    else:
        # Just copy/renumber to stage 3 if no upscale
        stage_3_upscaled.mkdir(exist_ok=True)
        for idx, (ftype, path) in enumerate(work_frames_tuples):
             out_name = f"frame_{idx:05d}.png"
             shutil.copy(path, stage_3_upscaled / out_name)
        final_frames_dir = stage_3_upscaled
        
    # 4. Stitch
    logging.info("🧵 Stitching Video...")
    output_video = out_root / "final_output.mp4"
    # Framerate? source / expansion? 
    # If source was 24fps and we did 2x, we probably want 48fps OR slow motion?
    # Usually frame interpolation is for smoothness (increase fps).
    # Default to 24fps?
    fps = 24 * args.x 
    # Actually, simpler: 12fps is default animation. 24 is cinema.
    # If standard is 12, and we double, maybe 24.
    
    cmd = f"ffmpeg -framerate 24 -i '{final_frames_dir}/frame_%05d.png' -c:v libx264 -pix_fmt yuv420p '{output_video}' -y -loglevel error"
    os.system(cmd)
    
    logging.info(f"🎉 Post Production Complete: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Post Production: The Svelte 2x Machine")
    parser.add_argument("input", help="Input Video file or Directory of frames")
    parser.add_argument("--output", help="Output Directory")
    parser.add_argument("-x", type=int, default=2, help="Frame Expansion Factor (Tweening). Default: 2")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale Factor. Default: 2.0")
    
    args = parser.parse_args()
    
    process(args)

if __name__ == "__main__":
    main()
