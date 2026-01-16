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
import subprocess
from pathlib import Path
from PIL import Image
try:
    from mvp_shared import load_xmvp
except ImportError:
    # If not in path, try relative
    sys.path.append(str(Path(__file__).parent))
    from mvp_shared import load_xmvp
import json

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

def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def run_ascii_forge(input_video, output_video):
    """
    Runs the ascii_forge.py script on the input video.
    """
    try:
        # Resolve path relative to this script
        # post_production.py is in tools/fmv/mvp/v0.5/
        # ascii_forge is in tools/fmv/mvp/spearmint/ascii_forge/
        # So we go up 2 levels: v0.5 -> mvp -> fmv (Wait, root is tools/fmv ?)
        # Let's check path structure:
        # /Users/0gs/METMcloud/METMroot/tools/fmv/mvp/v0.5/post_production.py
        # /Users/0gs/METMcloud/METMroot/tools/fmv/mvp/spearmint/ascii_forge/ascii_forge.py
        # parent = v0.5
        # parent.parent = mvp
        # parent.parent / spearmint / ... -> Correct.
        
        forge_script = Path(__file__).resolve().parent.parent / "spearmint" / "ascii_forge" / "ascii_forge.py"
        if not forge_script.exists():
             logging.error(f"ASCII Forge script not found at {forge_script}")
             return False
             
        forge_input_dir = forge_script.parent / "inputs"
        forge_output_dir = forge_script.parent / "outputs"
        ensure_dir(forge_input_dir)
        ensure_dir(forge_output_dir)
        
        # Clean forge input
        for f in forge_input_dir.glob("*"): 
            try: f.unlink()
            except: pass
        
        # Copy source
        temp_input = forge_input_dir / input_video.name
        shutil.copy(input_video, temp_input)
        
        # Run Forge
        # Using default settings or arguments? The user requested "just like music-visualizer"
        # Music Visualizer uses defaults in the function I copied?
        # Code in cartoon_producer: cmd = ["python3", str(forge_script), "--brightness", "120", "--saturation", "140"]
        cmd = ["python3", str(forge_script), "--brightness", "120", "--saturation", "140"]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find output
        expected_output = forge_output_dir / f"ascii_{input_video.name}"
        if expected_output.exists():
            shutil.move(expected_output, output_video)
            return True
        else:
             logging.error("ASCII Forge did not produce expected output.")
             return False
             
    except Exception as e:
        logging.error(f"ASCII Forge failed: {e}")
        return False

def blend_videos(base_video, overlay_video, output_path, opacity=0.33):
    """
    Blends overlay_video onto base_video with specified opacity.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_video),
            "-i", str(overlay_video),
            "-filter_complex", f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[ov];[0:v][ov]overlay",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logging.error(f"Video Blending failed: {e}")
        return False

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
        Supports Hybrid Upscaling for scale > 2.0 (Style x2 -> Nearest Neighbor xRemaining).
        """
        try:
            if not os.path.exists(image_path): return False
            img = Image.open(image_path).convert("RGB")
            
            # 1. Obsess (Detail Injection)
            img = self.obsess(img, intensity=0.2)
            
            w, h = img.size
            
            # Hybrid Logic for Scale > 2 (e.g. 4x)
            # User wants "native repainting style" (approx 2x) then "forceful double" (Nearest)
            if scale > 2.0:
                # Stage A: Style Upscale (x2)
                intermediate_scale = 2.0
                int_w, int_h = int(w * intermediate_scale), int(h * intermediate_scale)
                
                # Resize (Lanczos)
                resized = img.resize((int_w, int_h), Image.LANCZOS)
                
                # Quantize (The 'Redoer' signature look)
                styled = resized.quantize(colors=255, method=Image.MAXCOVERAGE, dither=Image.FLOYDSTEINBERG)
                styled = styled.convert("RGB")
                
                # Stage B: Forceful Resize (Nearest Neighbor) to Final
                final_w, final_h = int(w * scale), int(h * scale)
                final = styled.resize((final_w, final_h), Image.NEAREST)
                
            else:
                # Standard Logic (Direct to Scale)
                target_w, target_h = int(w * scale), int(h * scale)
                
                # 2. Resize Method (LANCZOS is best for upscaling before quantization)
                resized = img.resize((target_w, target_h), Image.LANCZOS)
                
                # 3. Quantize
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
    def __init__(self, keys, context=""):
        self.keys = keys
        self.context = context
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
        
        if self.context:
             prompt += f"\n\nPROJECT CONTEXT/STYLE GUIDE:\n{self.context}"
        
        try:
            # Load images as PIL
            # Gemini client handles file paths or PIL images in contents?
            # client.models.generate_content supports PIL images directly in 'contents' list.
            
            img_a = Image.open(img_path_a)
            img_b = Image.open(img_path_b)
            
            # Put Images FIRST for Gemini 2.0/2.5 Context Priority
            prompt += "\n(Refer to the two images above as the Start and End frames. Generate the Middle Frame.)"
            
            response = client.models.generate_content(
                model=self.model,
                contents=[img_a, img_b, prompt]
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
    audio_path = None
    
    if input_video:
        logging.info(f"🎞️ Extracting frames from {input_video}...")
        stage_1_frames.mkdir(exist_ok=True)
        # ffmpeg extract
        cmd = f"ffmpeg -i '{input_video}' -q:v 2 '{stage_1_frames}/frame_%05d.png' -y -loglevel error"
        os.system(cmd)
        work_frames = list(stage_1_frames.glob("*.png"))
        
        # Extract Audio
        audio_candidate = out_root / "source_audio.mp3"
        # Check if source has audio
        cmd_audio = f"ffmpeg -i '{input_video}' -vn -acodec libmp3lame -q:a 2 '{audio_candidate}' -y -loglevel error"
        if os.system(cmd_audio) == 0 and audio_candidate.exists():
             logging.info("   🔊 Audio extracted.")
             audio_path = audio_candidate
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
        
        # XML Context Lookahead
        context_str = ""
        possible_xml = None
        if input_video:
             possible_xml = input_video.with_suffix('.xml')
        elif input_dir:
             possible_xml = input_dir / "project.xml" # Convention?
             
        if possible_xml and possible_xml.exists():
             try:
                 logging.info(f"   📜 Found XML Context: {possible_xml.name}")
                 cssv_raw = load_xmvp(possible_xml, "CSSV")
                 story_raw = load_xmvp(possible_xml, "Story")
                 
                 parts = []
                 if cssv_raw:
                     c = json.loads(cssv_raw)
                     # Safely get deeply nested keys
                     vision = c.get('vision', "")
                     situation = c.get('situation', "")
                     parts.append(f"VISUAL STYLE: {vision}")
                     parts.append(f"SCENARIO: {situation}")
                     
                 if story_raw:
                     s = json.loads(story_raw)
                     chars = s.get('characters', [])
                     parts.append(f"CHARACTERS: {', '.join(chars)}")
                     
                 context_str = "\n".join(parts)
                 if context_str:
                     logging.info(f"   🧠 Injected Context ({len(context_str)} chars)")
             except Exception as e:
                 logging.warning(f"   ⚠️ XML Load Failed: {e}")

        interpolator = FrameInterpolator(keys, context=context_str)
        
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
                    # TODO: If x > 2, we might want to fill more?
                    # For MVP v0.5, strict flipbook (A, Tween, B) is fine.
                    # Duplicating Tween for higher X? Or defaulting logic?
                    # Let's just stick one tween. If x=4, framerate increases, creates simple hold?
                    # Actually, let's just do 1 tween.
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

    # 2.5 RESTYLE (ASCII)
    if args.restyle and args.restyle.lower() == 'ascii':
        logging.info("🎨 Restyling (ASCII Overlay)...")
        stage_restyle = out_root / "2.5_restyled"
        stage_restyle.mkdir(exist_ok=True)
        
        # A. Stitch current frames to temp video
        temp_stitch_list = stage_restyle / "temp_list.txt"
        temp_video = stage_restyle / "temp_input.mp4"
        
        with open(temp_stitch_list, 'w') as f:
            for _, path in work_frames_tuples:
                f.write(f"file '{path}'\n")
                # Default duration? If we just want a sequence for processing, FFmpeg default is fine?
                # Actually, "concat" demuxer without duration assumes each image is a frame? 
                # No, standard image list concat usually needs duration if they are images.
                # However, if we use glob or pattern it's easier.
                # Let's iterate.
                f.write("duration 0.0416\n") # 24fps equivalent
                
        # Actually, piping images to ffmpeg is safer than concat demuxer for raw frames
        # But paths are scattered (source vs interpolated).
        # Let's use the list file with explicit duration.
        # Last frame needs to be repeated or just handled.
        
        # Alternative: Copy all current frames to a temp dir in order, then ffmpeg -i %05d
        temp_frame_dir = stage_restyle / "temp_frames"
        temp_frame_dir.mkdir(exist_ok=True)
        for idx, (_, path) in enumerate(work_frames_tuples):
             shutil.copy(path, temp_frame_dir / f"frame_{idx:05d}.png")
             
        # Stitch
        # Assuming 24fps for processing
        cmd_stitch = f"ffmpeg -framerate 24 -i '{temp_frame_dir}/frame_%05d.png' -c:v libx264 -pix_fmt yuv420p '{temp_video}' -y -loglevel error"
        os.system(cmd_stitch)
        
        if temp_video.exists():
            # B. Run Forge
            ascii_video = stage_restyle / "ascii_ver.mp4"
            logging.info("   🔥 Forging ASCII...")
            if run_ascii_forge(temp_video, ascii_video):
                # C. Blend
                blended_video = stage_restyle / "blended.mp4"
                logging.info("   🎨 Blending...")
                if blend_videos(temp_video, ascii_video, blended_video, opacity=0.33):
                    # D. Extract Frames back
                    logging.info("   🎞️ Extracting Restyled Frames...")
                    final_restyle_dir = stage_restyle / "frames"
                    final_restyle_dir.mkdir(exist_ok=True)
                    
                    cmd_extract = f"ffmpeg -i '{blended_video}' -q:v 2 '{final_restyle_dir}/frame_%05d.png' -y -loglevel error"
                    os.system(cmd_extract)
                    
                    # Update work_frames_tuples
                    new_frames = list(final_restyle_dir.glob("*.png"))
                    new_frames = sort_frames(new_frames)
                    
                    if new_frames:
                        work_frames_tuples = [("restyled", p) for p in new_frames]
                        logging.info(f"   ✅ Restyle Complete. New Frame Count: {len(work_frames_tuples)}")
                    else:
                        logging.error("   ❌ Restyle Extraction Failed. Proceeding with original frames.")
                else:
                    logging.error("   ❌ Blend Failed.")
            else:
                 logging.error("   ❌ ASCII Forge Failed.")
        else:
             logging.error("   ❌ Stitched temp video failed.")

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
    # Framerate Logic:
    # If source was X fps, and we expanded twice, we have 2X frames.
    # To keep Duration constant (Audio Sync), we must double FPS.
    # But usually source FPS isn't known for extracted frames unless probed.
    # Assuming standard input of 12fps or 24fps?
    # If args.x is used (Interpolation), we usually WANT to increase fluidity (e.g. 12->24).
    # So we should multiply base fps (which we'll assume is ~24 or 12?)
    
    # Safe Default: 24fps as base.
    # If we expanded by args.x... wait.
    # If we have 1 sec of audio and 12 frames.
    # If we expand to 24 frames (x=2), and set FPS to 24, duration is 1 sec. Perfect.
    # So if original was "Animation" (approx 12?), then 12 * x is correct.
    # Let's assume input matches the expansion intent.
    
    target_fps = 12 * args.x 
    # Or should we just stick to 24?
    # User usually wants 24fps final.
    # Let's stick to 24 forced if no better info.
    target_fps = 24
    
    # Check if we have audio
    audio_cmd = ""
    if audio_path:
        audio_cmd = f"-i '{audio_path}' -map 0:v -map 1:a -c:a copy"
    
    cmd = f"ffmpeg -framerate {target_fps} -i '{final_frames_dir}/frame_%05d.png' {audio_cmd} -c:v libx264 -pix_fmt yuv420p '{output_video}' -y -loglevel error"
    os.system(cmd)
    
    logging.info(f"🎉 Post Production Complete: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Post Production: The Svelte 2x Machine")
    parser.add_argument("input", help="Input Video file or Directory of frames")
    parser.add_argument("--output", help="Output Directory")
    parser.add_argument("-x", type=int, default=2, help="Frame Expansion Factor (Tweening). Default: 2")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale Factor. Default: 2.0")
    parser.add_argument("--restyle", type=str, default=None, help="Restyle Mode (e.g. 'ascii').")
    
    args = parser.parse_args()
    
    process(args)

if __name__ == "__main__":
    main()
