import argparse
import logging
import os
import sys

# Hack for Mac (OpenMP Conflict)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    import definitions 
    from definitions import Modality, BackendType
    from flux_bridge import get_flux_bridge
except ImportError:
    # If not in path, try relative
    sys.path.append(str(Path(__file__).parent))
    from mvp_shared import load_xmvp
    import definitions 
    from definitions import Modality, BackendType
    from flux_bridge import get_flux_bridge
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

    except Exception as e:
        logging.error(f"ASCII Forge failed: {e}")
        return False

def get_duration(media_path):
    """Returns duration in seconds using ffprobe (Copied from music_video.py)."""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(media_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"[-] Error getting duration for {media_path}: {e}")
        return None

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

    def upscale(self, image_path, output_path, scale=2.0, local=False, more=False):
        """
        Resizes and enhances.
        Legacy: 'Pass 3' Logic (Lanczos -> Quantize).
        Local: Flux Img2Img Upscale.
        """
        try:
            if not os.path.exists(image_path): return False
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            
            # --- LOCAL FLUX UPSCALER ---
            if local:
                # 1. Base Upscale (Flux) ~ 2x
                # If scale >= 2, we do at least 2x via Flux
                # If scale < 2, just resize? Assuming scale=2 base.
                
                target_w, target_h = int(w * 2), int(h * 2)
                logging.info(f"   ✨ Flux Upscaling to {target_w}x{target_h}...")
                
                # Resize first (Conditioning)
                img = img.resize((target_w, target_h), Image.LANCZOS)
                
                # Flux Denoise (Add Detail)
                # Load Bridge
                config = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-schnell")
                bridge = get_flux_bridge(config.path)
                
                # Prompt? "High resolution, sharp details, 4k"
                enhanced = bridge.generate_img2img(
                    prompt="high resolution, sharp focus, detailed, cinematic, best quality, 4k",
                    image=img,
                    width=target_w,
                    height=target_h,
                    strength=0.30, # Low strength to preserve geometry but add texture
                    steps=4
                )
                
                if not enhanced: 
                    logging.warning("   ⚠️ Flux Upscale failed. Using Lanczos.")
                    enhanced = img # Fallback
                
                img = enhanced
                
                # 2. Secondary Upscale (--more) -> 4x Total
                if more and scale >= 4.0:
                    final_w, final_h = int(w * 4), int(h * 4)
                    logging.info(f"   ➕ Plus Upscale: Resampling to {final_w}x{final_h} (Lanczos)...")
                    img = img.resize((final_w, final_h), Image.LANCZOS)
                    
                img.save(output_path)
                return True

            # --- LEGACY UPSCALER ---
            
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

    def generate_tween(self, img_path_a, img_path_b, output_path, frame_idx_a, frame_idx_b, local=False):
        """
        Generates a frame visually between A and B.
        Legacy: Gemini Hallucination.
        Local: Flux Img2Img Tweening.
        """
        if local:
            try:
                # 1. Load Images
                img_a = Image.open(img_path_a).convert("RGB")
                img_b = Image.open(img_path_b).convert("RGB")
                
                # 2. Simple Blend (50%)
                blended = Image.blend(img_a, img_b, 0.5)
                
                # 3. Flux Denoise (Img2Img)
                # Strength 0.55 -> Enough to hallucinate smooth transitions but keep structure
                config = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-schnell")
                bridge = get_flux_bridge(config.path)
                
                # Prompt? Just describe movement or generic?
                # "cinematic shot, smooth motion, high quality"
                prompt = "cinematic aesthetic, smooth motion blur, high quality, consistent lighting"
                w, h = blended.size
                
                result = bridge.generate_img2img(
                    prompt=prompt,
                    image=blended,
                    width=w, 
                    height=h, 
                    strength=0.55,
                    steps=4
                )
                
                if result:
                    result.save(output_path)
                    return True
                else:
                    return False
            except Exception as e:
                logging.error(f"   ❌ Local Tween Failed: {e}")
                return False

        # --- LEGACY GEMINI LOGIC ---
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
    
    # Check if output is a filename or dir
    # Heuristic: If extension is .mp4, .mov, .avi, etc., assume filename.
    target_video_file = None
    
    if args.output:
        potential_out = Path(args.output)
        if potential_out.suffix.lower() in ['.mp4', '.mov', '.mkv', '.avi']:
            # User provided a filename!
            target_video_file = potential_out.resolve()
            out_root = target_video_file.parent / f"work_{target_video_file.stem}_{ts}"
        else:
            out_root = potential_out
    else:
        # Default to z_test-outputs if no output specified
        # This prevents polluting the root directory
        base_outputs = Path(__file__).resolve().parent / "z_test-outputs" / "post_prod"
        ensure_dir(base_outputs)
        out_root = base_outputs / f"post_prod_{ts}"
        
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
                success = interpolator.generate_tween(curr_frame, next_frame, tween_path, i, i+1, local=args.local)
                
                if success:
                    final_sequence.append(("generated", tween_path))
                    
                    # More Mode: Secondary Interpolation (Wait, does user want to interp AGAIN between A->Tween and Tween->B?)
                    # User: "with a bonus boolean, --more, we do ANOTHER round of frame interp using the simple tweening"
                    # So: A -> Simple -> Tween -> Simple -> B
                    # This means we should add a simple blend around the tween.
                    if args.more:
                        # 1. A -> Tween (Simple Blend)
                        p1_name = f"tween_{i}_pre_more.png"
                        p1_path = stage_2_interpolated / p1_name
                        try:
                           img_a = Image.open(curr_frame).convert("RGB")
                           img_t = Image.open(tween_path).convert("RGB")
                           Image.blend(img_a, img_t, 0.5).save(p1_path)
                           # Insert BEFORE Tween
                           # But we already appended tween. No, we appended ("generated", tween_path)
                           # So we should pop or insert correctly.
                           final_sequence.pop() # Remove Tween for a sec
                           final_sequence.append(("simple_more", p1_path))
                           final_sequence.append(("generated", tween_path))
                           
                           # 2. Tween -> B (Simple Blend)
                           p2_name = f"tween_{i}_post_more.png"
                           p2_path = stage_2_interpolated / p2_name
                           img_b = Image.open(next_frame).convert("RGB")
                           Image.blend(img_t, img_b, 0.5).save(p2_path)
                           final_sequence.append(("simple_more", p2_path))
                        except Exception as e:
                            logging.warning(f"Simple More interp failed: {e}")
                            
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
            # If args.more is True, we pass it. If args.scale (which defaults to 2.0) is unchanged but 'more' is on, 
            # we should assume 4x?
            # User: "default is 2x frame and 2x size, if --more is enabled we get 4x frames and 4x size."
            # So if more is On, we force scale 4.0?
            # Or we let upscaler handle it?
            # Upscaler logic handles "if more and scale >= 4".
            
            target_scale = args.scale
            if args.more:
                target_scale = 4.0
                
            success = upscaler.upscale(path, out_path, scale=target_scale, local=args.local, more=args.more)
            
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
    if args.local:
        # If we did 2x interp, doubling frames. 12->24?
        # If --more, we did 4x interp. 12->48?
        # Assuming input is LTX generic (24fps?) post-interpolation.
        # Actually usually if we interpolate we want SLOW MOTION or SMOOTHER MOTION.
        # If we keep FPS same -> Slow Motion.
        # If we increase FPS -> Smoother.
        # "music-video" implies sync. So duration must match.
        # If we have 4x frames, we need 4x FPS to keep duration.
        base_fps = 24
        multiplier = 2 if args.x > 1 else 1 # Simple assumption
        if args.more: multiplier = 4
        
        target_fps = base_fps * multiplier
    
    # Check if we have audio
    audio_cmd = ""
    
    # --- MUSIC VIDEO LOGIC ---
    # Merge of music_video.py
    if args.mu:
        if os.path.exists(args.mu):
            audio_path = Path(args.mu)
            logging.info(f"🎵 Music Video Mode: Syncing to {audio_path.name}...")
            
            # 1. Measure Audio Duration
            audio_dur = get_duration(audio_path)
            
            # 2. Measure Expected Video Duration at Current Target FPS
            # We have len(work_frames_tuples) frames (or we should count final frames dir?)
            # Usually final_frames_dir has the 'work_frames_tuples' equivalent.
            # Let's count explicitly to be safe.
            final_files = list(final_frames_dir.glob("frame_*.png")) 
            num_frames = len(final_files)
            
            if audio_dur and num_frames > 0:
                expected_dur = num_frames / target_fps
                diff = abs(audio_dur - expected_dur)
                
                logging.info(f"   ⏱️ Audio: {audio_dur:.2f}s | Frames: {num_frames} | @{target_fps}fps = {expected_dur:.2f}s")
                logging.info(f"   ⚠️ Delta: {diff:.2f}s")
                
                # 3. Stretch/Squeeze Threshold (10s) OR Forced Stitch
                if diff <= 10.0 or args.stitch_audio:
                    # Calculate Perfect FPS
                    new_fps = num_frames / audio_dur
                    logging.info(f"   ✨ Auto-Sync: Adjusting FPS {target_fps} -> {new_fps:.4f} to match audio (Stitch Mode: {args.stitch_audio}).")
                    target_fps = new_fps
                else:
                    logging.warning("   ⚠️ Delta > 10s. Keeping original FPS (Audio might cut off or videeo silent). Use --stitch-audio to force.")
                    
        else:
             logging.warning(f"   ⚠️ Audio file not found: {args.mu}")

    if audio_path:
        audio_cmd = f"-i '{audio_path}' -map 0:v -map 1:a -c:a copy"
        # If we did the squeez, we want shortest? 
        # If we calculated exact FPS, it should match.
        # But for safety, -shortest is good logic if we prioritized audio.
        if args.mu: audio_cmd += " -shortest"
    
    cmd = f"ffmpeg -framerate {target_fps} -i '{final_frames_dir}/frame_%05d.png' {audio_cmd} -c:v libx264 -pix_fmt yuv420p '{output_video}' -y -loglevel error"
    os.system(cmd)
    
    # 6. Finalize (Move to Target)
    if target_video_file:
         logging.info(f"🚚 Moving final output to {target_video_file}...")
         # Check if target dir exists
         if not target_video_file.parent.exists():
             target_video_file.parent.mkdir(parents=True, exist_ok=True)
         shutil.copy(output_video, target_video_file)
         logging.info(f"✨ Custom Output Saved: {target_video_file}")
    else:
         logging.info(f"🎉 Post Production Complete: {output_video}")


def stitch_videos(log, output_filename):
    """Stitches mp4s using FFmpeg concat."""
    # Create unique list file in componentparts to avoid collisions
    # We assume DIR_PARTS or similar exists, or we use the dir of the first file
    
    # Heuristic: Find a directory to write the list file to
    work_dir = Path("componentparts")
    if log and isinstance(log, list) and len(log) > 0 and 'local_file' in log[0]:
         work_dir = Path(log[0]['local_file']).parent

    work_dir.mkdir(parents=True, exist_ok=True)
    
    ts = int(time.time())
    list_file = work_dir / f"stitch_list_{ts}.txt"
    valid_files = []
    
    with open(list_file, 'w') as f:
        for entry in log:
            if 'local_file' in entry and os.path.exists(entry['local_file']):
                abs_path = os.path.abspath(entry['local_file'])
                f.write(f"file '{abs_path}'\n")
                valid_files.append(abs_path)
    
    if not valid_files:
        logging.warning("   ❌ No videos to stitch.")
        return

    logging.info(f"   🧵 Combining {len(valid_files)} clips...")
    # ffmpeg concat
    # -safe 0 to allow relative paths
    cmd = f"ffmpeg -f concat -safe 0 -i {list_file} -c copy {output_filename} -y"
    
    ret = os.system(cmd)
    if ret == 0:
        # 6. Finalize (Move to Target)
        # Note: target_video_file and final_output are not defined in this function's scope.
        # This block seems intended for the main `process` function's final output handling.
        # Assuming `output_filename` is the `final_output` here.
        # And `target_video_file` would be `args.output` if it were passed.
        # For now, just logging the successful stitch.
        logging.info(f"   ✅ stitched: {output_filename}")
        try:
            os.remove(list_file)
        except:
             pass
    else:
        logging.error("   ❌ FFmpeg failed.")

def main():
    parser = argparse.ArgumentParser(description="Post Production: The Svelte 2x Machine")
    parser.add_argument("input", nargs='?', help="Input Video file or Directory of frames")
    parser.add_argument("--output", help="Output Directory")
    parser.add_argument("-x", type=int, default=2, help="Frame Expansion Factor (Tweening). Default: 2")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale Factor. Default: 2.0")
    parser.add_argument("--restyle", type=str, default=None, help="Restyle Mode (e.g. 'ascii').")
    parser.add_argument("--local", action="store_true", help="Run Locally (Flux Img2Img)")
    parser.add_argument("--more", action="store_true", help="Enable Secondary Interpolation/Upscale (4x Total)")
    parser.add_argument("--mu", type=str, help="Audio File for Music Video Sync")
    parser.add_argument("--stitch-audio", action="store_true", help="Force-stitch frames to match audio duration (Ignore Delta).")
    
    args = parser.parse_args()
    
    if args.input:
        process(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

