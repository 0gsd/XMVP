import re
import os
import json
import time
import math
import logging
import argparse
import subprocess
import random
import yaml
import itertools
import shutil
import warnings
import pyloudnorm as pyln
from pathlib import Path
import librosa
import numpy as np
# Monkeypatch for Scipy 1.13+ vs Librosa < 0.10 compatibility
try:
    import scipy.signal
    if not hasattr(scipy.signal, "hann"):
        if hasattr(scipy.signal.windows, "hann"):
             scipy.signal.hann = scipy.signal.windows.hann
except ImportError:
    pass

from google import genai
from google.genai import types
from text_engine import TextEngine
from truth_safety import TruthSafety
from mvp_shared import VPForm, CSSV, Constraints, Story, Portion, save_xmvp, load_xmvp, load_manifest
from vision_producer import get_chaos_seed
import frame_canvas # Support FC Mode
try:
    from flux_bridge import get_flux_bridge
except ImportError:
    pass # Handle locally inside function if missing
import definitions
from definitions import Modality, BackendType, get_active_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default Paths (Can be overridden by args)
DEFAULT_TF = Path("/Users/0gs/METMcloud/METMroot/tools/fmv/fbf_data")
DEFAULT_VF = Path("/Volumes/XMVPX/fmv_corpus")

# Model Configuration
# Model Configuration
# IMAGE_MODEL = "gemini-2.5-flash-image" # Default (Fast/Capable) - DEPRECATED for Registry
FLIPBOOK_STYLE_PROMPT = "You are a commercial animator working in a large production house with Fortune 500 clients in the 1990's. though you have SOME room for artistic expression and deviating from the frame prompt you are given, in order to fulfill your job as an animator, you need to ensure that you reference the previous frame (provided helpfully as context) and draw the exact next frame in the scene's (and thus the narrative's) trajectory. VISUAL:"

ANI_INSTRUCTION = """
CRITICAL VISUAL INSTRUCTION:
- This is a sequential frame in a HAND-DRAWN animation.
- STRICTLY NO TEXT, NO TIMECODES, NO "FRAME XX/YY" OVERLAYS.
- The image must be pure visual content.
- Match the style of the previous frame (if provided) EXACTLY.
- Motion should be incremental and fluid.
"""

def load_keys(env_path):
    """Loads keys from env_vars.yaml."""
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                secrets = yaml.safe_load(f)
                keys_str = secrets.get("ACTION_KEYS_LIST", "")
                if keys_str:
                    return [k.strip() for k in keys_str.split(',') if k.strip()]
        except Exception as e:
            logging.error(f"Failed to load keys: {e}")
    return []

def get_random_key(keys):
    return random.choice(keys) if keys else os.environ.get("GEMINI_API_KEY")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_original_video(video_id_stem, search_dirs):
    """Finds the original video in specified directories."""
    for folder in search_dirs:
        if not folder.exists(): continue
        
        # 1. Direct match attempt
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.webm']:
            cand = folder / f"{video_id_stem}{ext}"
            if cand.exists(): return cand

def analyze_audio(audio_path):
    """
    Analyzes audio for BPM and Duration.
    Returns (bpm, duration, beat_frames, suggested_fps)
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Tempo is an array or float. Normalize.
        bpm = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)
        
        # Calculate Frames Per Beat (FPB) targeting ~4-12 FPS range
        # 120 BPM = 2 beats/sec. 
        # If we want 2 frames per beat -> 4 FPS.
        # If we want 4 frames per beat -> 8 FPS.
        
        # Let's standardize on 4 frames per beat for visualizer smoothness
        frames_per_beat = 4 
        bps = bpm / 60.0
        fps = bps * frames_per_beat
        
        logging.info(f"   🎵 Audio Analysis: {bpm:.1f} BPM | {duration:.1f}s")
        logging.info(f"   🎵 Derived FPS: {fps:.2f} (based on {frames_per_beat} frames/beat)")
        
        return bpm, duration, frames_per_beat, fps
        
    except Exception as e:
        logging.error(f"   ❌ Audio Analysis Failed: {e}")
        return 120.0, 60.0, 4, 8.0 # Fallback

def analyze_audio_profile(audio_path, duration):
    """
    Creates a 'Sonic Map' of the track using LUFS (Loudness) and Dynamic Range.
    Returns a string describing energy levels and sudden changes over time.
    """
    try:
        # Load audio (Librosa)
        y, sr = librosa.load(audio_path, sr=None)
        
        # Pyloudnorm Setup
        meter = pyln.Meter(sr) # create BS.1770 meter
        
        # Calculate Integrated Loudness (Overall)
        try:
            loudness_overall = meter.integrated_loudness(y)
        except Exception:
            loudness_overall = -24.0 # Default fallback
            
        logging.info(f"   📊 Audio Integrated Loudness: {loudness_overall:.2f} LUFS")

        # Dynamic Analysis: Windowed LUFS
        # Segment into chunks (e.g., 8 segments for narrative structure)
        num_segments = 8
        seg_samples = len(y) // num_segments
        
        profile_str = []
        lufs_values = []
        
        for i in range(num_segments):
            start = i * seg_samples
            end = (i + 1) * seg_samples
            chunk = y[start:end]
            
            # Check length to avoid PyLoudnorm errors on tiny chunks
            if len(chunk) < sr * 0.4: # < 400ms might error
                chunk_lufs = loudness_overall # Fallback
            else:
                try:
                    chunk_lufs = meter.integrated_loudness(chunk)
                except ValueError:
                    # Silence or too short
                    chunk_lufs = -70.0 
            
            lufs_values.append(chunk_lufs)

        # Detect Deltas (Changes)
        # Relative to Overall
        for i, val in enumerate(lufs_values):
            time_start = int((i / num_segments) * duration)
            
            diff_from_avg = val - loudness_overall
            
            # Determine previous value for Delta
            prev_val = lufs_values[i-1] if i > 0 else val
            delta = val - prev_val
            
            emoji = "🔉"
            desc = "Quiet"
            
            # 1. Absolute Levels
            if val > -14.0:
                emoji = "🔊"
                desc = "High Energy"
            elif val > -24.0:
                emoji = "🎵"
                desc = "Moderate"
            else:
                emoji = "🔉"
                desc = "Quiet"
                
            # 2. Dynamic Deltas (Overrides)
            if delta > 6.0:
                emoji = "💥"
                desc = "SUDDEN IMPACT"
            elif delta < -10.0:
                emoji = "🤫"
                desc = "Sudden Drop"
            elif val > -10.0: # Very Loud
                emoji = "🔥"
                desc = "MAX INTENSITY"
                
            profile_str.append(f"[{time_start}s: {emoji} {desc} ({val:.1f} LUFS)]")
            
        return " -> ".join(profile_str)
    except Exception as e:
        logging.warning(f"   ⚠️ Could not generate audio profile: {e}")
        return "Audio Profile Unavailable."

def run_ascii_forge(input_video, output_video):
    """
    Runs the ascii_forge.py script on the input video.
    """
    try:
        forge_script = Path(__file__).resolve().parent.parent.parent / "spearmint" / "ascii_forge" / "ascii_forge.py"
        if not forge_script.exists():
             logging.error(f"ASCII Forge script not found at {forge_script}")
             return False
             
        # Create inputs/outputs dir for forge if needed, or modify forge to accept args.
        # The current forge script processes everything in inputs/*.mp4
        # We will symlink or copy our video to its input, run it, move output back.
        
        forge_input_dir = forge_script.parent / "inputs"
        forge_output_dir = forge_script.parent / "outputs"
        ensure_dir(forge_input_dir)
        ensure_dir(forge_output_dir)
        
        # Clean forge input
        for f in forge_input_dir.glob("*"): f.unlink()
        
        # Link source
        temp_input = forge_input_dir / input_video.name
        # Symlink might fail on some OS/filesystems, copy is safer for small clips
        import shutil
        shutil.copy(input_video, temp_input)
        
        # Run Forge
        cmd = ["python3", str(forge_script), "--brightness", "120", "--saturation", "140"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find output
        # Forge typically names it "ascii_" + filename
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
        
        # 2. Recursive Search
        candidates = list(folder.rglob(f"{video_id_stem}.*"))
        valid_cands = [c for c in candidates if c.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.webm']]
        
        if valid_cands:
            return valid_cands[0]
        
    return None

def generate_frame_universal(index, prompt, output_dir, key_cycle, width=768, height=768, aspect_ratio="1:1", model=None, prev_frame_path=None, pg_mode=False, force_local=False):
    """
    Worker function using Universal Backend (Flux Local or Gemini/Imagen Cloud).
    """
    if model is None:
        model = get_active_model(Modality.IMAGE).name # String name from Registry

    # Resolve Model Config to check backend
    config = definitions.MODAL_REGISTRY[Modality.IMAGE].get(model)
    if not config:
        # Fallback for legacy string names not in registry
        # Assume Cloud if unknown
        is_local = False
    else:
        is_local = (config.backend == definitions.BackendType.LOCAL)

    # STRICT LOCAL ENFORCEMENT
    if force_local and not is_local:
        logging.error(f"❌ STRICT LOCAL ENFORCEMENT FAILED: Model '{model}' is not configured as LOCAL.")
        logging.error("   Please update active_models.json or definitions.py to point to a local model (e.g. flux-schnell).")
        # Fail hard
        raise RuntimeError("Local execution enforced but model is cloud-based/unknown.")

    target_path = output_dir / f"frame_{index:04d}.png"
    if target_path.exists():
        return True # Skip if already done

    # Retry Loop
    max_retries = 5 # User Requested Boost
    for attempt in range(max_retries):
        
        # --- LOCAL FLUX PATH ---
        if is_local:
            logging.info(f"🎨 Rendering Frame {index} (Attempt {attempt+1}/{max_retries}) [Flux Local]...")
            try:
                # Get Bridge
                try:
                    from flux_bridge import get_flux_bridge
                except ImportError:
                    logging.error("Flux Bridge not found.")
                    return False
                
                # Check path from config
                # config.path is the path to weights
                bridge = get_flux_bridge(config.path)
                
                # Generate
                # Flux Schnell doesn't support aspect_ratio arg directly usually, depends on bridge
                # We'll stick to square for now unless width/height set
                # Standardize to requested dimensions
                img = bridge.generate(prompt, width=width, height=height) 
                
                if img:
                    # Save (No resize needed if generated at target)
                    img.save(target_path)
                    return True
                else:
                    logging.warning("Flux returned None.")
                    continue
                    
            except Exception as e:
                logging.error(f"Flux Generation Failed: {e}")
                continue # Retry? Or fail? Retry.

        # --- CLOUD GEMINI/IMAGEN PATH ---
        else:
            # ROTATION: Get next key from cycle
            current_key = next(key_cycle)
            logging.info(f"🎨 Rendering Frame {index} (Attempt {attempt+1}/{max_retries}) [Cloud Key Rotation]...")
            
            # Instantiate fresh client with rotated key
            client = genai.Client(api_key=current_key)
            
            try:
                
                # MODE SWITCH
                if "imagen" in model.lower():
                    # Method A: Imagen (generate_images)
                    response = client.models.generate_images(
                        model=model, 
                        prompt=prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio=aspect_ratio, 
                        )
                    )
                    if response.generated_images:
                        image = response.generated_images[0]
                        if image.image:
                            image.image.save(target_path)
                            return True
                else:
                    # Method B: Gemini Flash (generate_content)
                    
                    # --- DIRECTOR MODE (Gemini 2.0 Flash) ---
                    # If model is 2.0 Flash, we use it to DESCRIBE the next frame, then use 2.5 Image to RENDER it.
                    if model == "gemini-2.0-flash":
                        director_prompt = f"""
                        You are a Lead Animator. 
                        Goal: Describe the EXACT VISUALS for the next frame in this animation sequence.
                        Context: Previous frame is attached (if available). 
                        Current Prompt: "{prompt}".
                        
                        CRITICAL: 
                        1. Maintain the STYLE described in the prompt exactly.
                        2. If the 'Action' is similar to the previous frame, describe INCREMENTAL MOTION (flipbook style). Do not change camera angles or character designs unless explicitly told to.
                        3. If the 'Action' is a new beat, describe the new scene but KEEP THE VISUAL STYLE CONSISTENT.
                        4. ABSOLUTELY NO TEXT DESCRIPTIONS IN THE IMAGE. Do not ask for "Frame 1" text.

                        Output: A dense, visual description of the image to generate. No conversational filler.
                        """
                        
                        director_contents = []
                        
                        # Attach context if available (Put Image FIRST for better attention)
                        if prev_frame_path and prev_frame_path.exists():
                            try:
                                img_bytes = prev_frame_path.read_bytes()
                                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                                director_contents.append(img_part)
                                director_prompt += "\n(Refer to the image above as the PREVIOUS FRAME to maintain continuity from.)"
                            except Exception as ex:
                                logging.warning(f"   ⚠️ Failed to attach context image: {ex}")
                                
                        director_contents.append(director_prompt)
                        
                        # 1. Ask Director for Description
                        director_response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=director_contents
                        )
                        
                        if director_response.text:
                            raw_desc = director_response.text
                            logging.info(f"   🎬 Director's Direction: {raw_desc[:100]}...")
                            
                            # 1.5 Truth & Safety Enforce
                            # Ensure the Director's output isn't crazy
                            ts = TruthSafety() # Uses Text Keys for verification
                            refined_prompt = ts.refine_prompt(raw_desc, context_dict={"Role": "Director Output"}, pg_mode=pg_mode)
                            
                            # BREATHING ROOM (User Request)
                            # Prevent "Double Tap" spamming the API (Director -> Renderer instant hit)
                            time.sleep(0.5)

                            # 2. Use Renderer
                            render_model = "gemini-2.5-flash-image"
                            # Force "Generate an image of" prefix AND explicit size
                            render_prompt = f"Generate an image of: {refined_prompt}"
                            
                            logging.info(f"      🎨 Render Prompt: {render_prompt[:50]}...")
                            
                            img_response = client.models.generate_images(
                                model=render_model,
                                prompt=render_prompt,
                                config=types.GenerateImagesConfig(
                                    number_of_images=1,
                                    aspect_ratio=aspect_ratio
                                )
                            )
                            if img_response.generated_images:
                                image = img_response.generated_images[0]
                                if image.image:
                                    image.image.save(target_path)
                                    return True
                    else:
                        # Direct Call (Legacy Gemini)
                        response = client.models.generate_images(
                            model=model, 
                            prompt=prompt,
                            config=types.GenerateImagesConfig(
                                number_of_images=1,
                                aspect_ratio=aspect_ratio, 
                            )
                        )
                        if response.generated_images:
                            image = response.generated_images[0]
                            if image.image:
                                image.image.save(target_path)
                                return True
                                
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    # RAMP COOLDOWN: 5, 10, 20, 40, 60
                    delay = 5 * (2 ** attempt)
                    if delay > 60: delay = 60
                    logging.warning(f"Frame {index}: Quota exceeded. Coping... (Waiting {delay}s)")
                    time.sleep(delay)
                else:
                    logging.error(f"Frame {index} failed: {e}") 
        
        # Backoff before retry
        if attempt < max_retries - 1:
            time.sleep(2)

    logging.error(f"Frame {index} failed after {max_retries} attempts.")
    return False

def scan_projects(tf_dir):
    """
    Scans the transcript folder for subdirectories containing 'analysis.json'.
    Returns a list of project paths or analysis dicts.
    """
    projects = []
    if not tf_dir.exists():
        logging.error(f"Transcript folder not found: {tf_dir}")
        return []
        
    for item in tf_dir.iterdir():
        if item.is_dir():
            analysis_file = item / "analysis.json"
            if analysis_file.exists():
                projects.append(item)
    
    projects.sort(key=lambda x: x.name) # Alphabetical/Predictable order
    return projects

def process_project(project_dir, vf_dir, key_cycle, args, output_root, keys, text_engine):
    # Use passed TextEngine instance (Singleton)
    te = text_engine
    
    project_name = project_dir.name
    logging.info(f"▶️ Processing Project: {project_name}")
    
    # Resolution Setup
    target_w = args.w if (args.local and args.w) else args.kid
    target_h = args.h if (args.local and args.h) else args.kid
    logging.info(f"   📐 Target Resolution: {target_w}x{target_h}")
    
    analysis_file = project_dir / "analysis.json"
    metadata_file = project_dir / "metadata.json"
    
    descriptions = []
    
    # Bypass for Creative Agency / XB
    # Bypass for Creative Agency / XB / Music Visualizer / Music Agency
    if args.vpform in ["creative-agency", "music-visualizer", "music-video"] or args.xb:
        descriptions = ["Placeholder"] # Will be overwritten by logic below
    else:    
        with open(analysis_file, 'r') as f:
            descriptions = json.load(f)
        
    duration = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            meta = json.load(f)
            duration = meta.get("duration")
            
    if not duration:
        logging.warning("Duration missing in metadata. Assuming 1 description = 1 second.")
        duration = len(descriptions)
    
    # 1. Find Source Video (for Audio)
    original_video = find_original_video(project_name, [vf_dir])
    if not original_video:
        if args.mu and os.path.exists(args.mu):
             logging.info(f"   🎵 Using provided music track: {args.mu}")
        else:
             logging.warning(f"Original video for {project_name} not found in {vf_dir}. Audio stitching will be skipped.")
    else:
        logging.info(f"Found original video: {original_video}")

    # 2. Calculate Target Frames & Prompts
    prompts = []
    
    # FBF Mode: Exact Match + Expansion
    if args.vpform == "fbf-cartoon":
        # BUG FIX: Cast directly to int for expansion, do not mutate args.fps for subsequent loops
        expansion = max(1, int(args.fps)) 
        total_frames = len(descriptions) * expansion
        
        # Local variable for this specific project's Output FPS
        project_fps = args.fps 
        
        if duration and duration > 0:
            real_fps = total_frames / duration
            logging.info(f"   ⚡ FBF Mode: {len(descriptions)} src rows * {expansion}x expansion = {total_frames} frames.")
            logging.info(f"   ⏱️  matches {duration:.2f}s audio => {real_fps:.2f} Output FPS")
            project_fps = real_fps # Use calculated FPS for stitching, but don't save to args
        else:
            logging.warning("   ⚠️ Duration unknown. Defaulting to 12 FPS.")
            project_fps = 12
            
        style_prefix = """
        You are a commercial animator working in a large production house with Fortune 500 clients in the 1990's. though you have SOME room for artistic expression and deviating from the frame prompt you are given, in order to fulfill your job as an animator, you need to ensure that you reference the previous frame (provided helpfully as context) and draw the exact next frame in the scene's (and thus the narrative's) trajectory.
        VISUAL: """
        
        for i, raw_desc in enumerate(descriptions):
             # HARDEN: Strip "Frame XX:" prefixes
             clean_desc = re.sub(r"^(Frame\s*\d+\s*[:.-]\s*)", "", raw_desc, flags=re.IGNORECASE).strip()
             
             # Lookahead for interpolation context
             next_desc = descriptions[i+1] if i < len(descriptions) - 1 else clean_desc
             clean_next = re.sub(r"^(Frame\s*\d+\s*[:.-]\s*)", "", next_desc, flags=re.IGNORECASE).strip()
             
             # Expand
             for k in range(expansion):
                 if k == 0:
                     # Keyframe (Anchor)
                     final_prompt = f"{style_prefix} {clean_desc}"
                 else:
                     # Interpolation Frame
                     # "Subtly vary everything to simulate natural imperfections... but exact precision."
                     interp_instruction = (
                         f"INTERPOLATION ({k}/{expansion}): "
                         "Slightly evolve this frame to simulate natural hand-drawn 'line boil' and subtle movement. "
                         f"Maintain continuity. Transitioning towards: {clean_next}."
                     )
                     final_prompt = f"{style_prefix} {clean_desc} \n{interp_instruction}"
                     
                 prompts.append(final_prompt)
             
                 prompts.append(final_prompt)
             
    elif args.vpform == "music-visualizer":
        # === MUSIC VISUALIZER MODE ===
        if not args.mu or not os.path.exists(args.mu):
             logging.error("❌ Music Visualizer requires --mu [audio_path]")
             return
             
        # 1. Analyze Audio
        if args.bpm and args.bpm > 0:
             # Manual Override
             bpm = float(args.bpm)
             # Get duration via soundfile if possible, or fallback
             try:
                 import soundfile as sf
                 f = sf.SoundFile(args.mu)
                 duration = len(f) / f.samplerate
             except:
                 # Last ditch effort if librosa also fails
                 logging.warning("⚠️ Could not read duration. Defaulting to 60s.")
                 duration = 60.0
             
             # Calculate derived metrics
             logging.info(f"   🎹 BPM Override: {bpm}")
             frames_per_beat = 4
             bps = bpm / 60.0
             fps = bps * frames_per_beat
             
        else:
             bpm, duration, fpb, fps = analyze_audio(args.mu)
             
        target_frames = int(duration * fps)
        project_fps = fps
        target_duration = duration
        
        logging.info(f"   🎹 Visualizer Target: {target_frames} frames @ {fps:.2f} FPS ({duration:.1f}s)")
        
        # 2. Generate Abstract Prompts
        # No "Story", just pure visual evolution.
        prompts = []
        
        # Style Definition
        base_style = args.style if args.style != "Indie graphic novel artwork. Precise, uniform, dead-weight linework. Highly stylized, elegantly sophisticated, and with an explosive, highly saturated pop-color palette." else "Abstract, pixel art, Stan Brakhage style, melting film on hot projectors, unique fractal algorithm, highly saturated colors"
        
        # Generate evolution
        # We want "Patterns" and "One path".
        # We can simulate this by evolving a noise seed or just descriptive interpolation.
        
        style_prefix = base_style
        
        # Visualizer Loop
        for i in range(target_frames):
             progress = i / target_frames
             # Evolve description
             # Phase 1: 0-0.3 (Buildup)
             # Phase 2: 0.3-0.7 (Chaos)
             # Phase 3: 0.7-1.0 (Resolution)
             
             phase = "forming patterns"
             if progress > 0.3: phase = "melting into chaotic fractals"
             if progress > 0.7: phase = "crystallizing into pure light"
             
             raw_desc = f"Visualizer Beat {i}. A single continuous abstract form {phase}. Progress: {int(progress*100)}%."
             prompts.append(f"Style: {style_prefix}. Action: {raw_desc} (Frame {i+1}/{target_frames})")

    elif args.vpform == "music-video":
        print(f"   Mode: Music Video Agency (Legacy: music-agency)")
        # === MUSIC VIDEO AGENCY MODE ===
        # Combo: Audio Sync (Timing) + Creative Story (Content)
        
        if not args.mu or not os.path.exists(args.mu):
             logging.error("❌ Music Agency requires --mu [audio_path]")
             return
             
        # 1. Analyze Audio (Timing)
        if args.bpm and args.bpm > 0:
             # Manual Override
             bpm = float(args.bpm)
             try:
                 import soundfile as sf
                 f = sf.SoundFile(args.mu)
                 duration = len(f) / f.samplerate
             except:
                 duration = 60.0
             frames_per_beat = 4
             bps = bpm / 60.0
             fps = bps * frames_per_beat
             
             # VSPEED Override
             if args.vspeed and args.vspeed != 8.0:
                 # User wants specific FPS override
                 logging.info(f"   🏎️  VSpeed Override: {args.vspeed} FPS (Default 8.0)")
                 fps = args.vspeed
        else:
             bpm, duration, fpb, fps = analyze_audio(args.mu)
             # VSPEED Override (even for auto-detected)
             if args.vspeed and args.vspeed != 8.0:
                  logging.info(f"   🏎️  VSpeed Override: {args.vspeed} FPS (Detected {fps:.2f})")
                  fps = args.vspeed
        target_frames = int(duration * fps)
        project_fps = fps
        target_duration = duration
        
        logging.info(f"   🎹 Music Agency Target: {target_frames} frames @ {fps:.2f} FPS ({duration:.1f}s) | BPM: {bpm}")
        
        # 2. Generate Story (Content)
        logging.info("   🧠 Agency Brain: Dreaming up a narrative for this track...")
        # te is already initialized
        text_engine = te
        
        # Seeds
        seeds = []
        if args.cs > 0:
            logging.info(f"      🎲 Rolling {args.cs} Chaos Seeds...")
            seeds = [get_chaos_seed() for _ in range(args.cs)]
            
        prompt_concept = args.prompt if args.prompt else "A cinematic music video."
        
        logging.info(f"      Seeds: {seeds}")
        logging.info(f"      Prompt: {prompt_concept}")
        
        # Determine number of 'beats' to ask for.
        # If we have target_frames, we don't want a unique description for EVERY frame (too jittery).
        # We want scenes/beats. 
        # Let's say 1 beat every 2-4 seconds?
        # Duration / 3?
        est_beats = max(5, int(target_duration / 3.0))
        
        # AUDIO PROFILE
        sonic_map = analyze_audio_profile(args.mu, target_duration)
        logging.info(f"      🎵 Sonic Map: {sonic_map}")
        
        story_req = (
            f"Create a VISUAL SCREENPLAY for a {target_duration}s music video (Animated).\n"
            f"Concept: {prompt_concept}\n"
            f"Chaos Seeds to weave in: {seeds}\n"
            f"Music Vibe: {bpm} BPM.\n"
            f"Audio Profile (Energy/Mood over time): {sonic_map}\n"
            f"Constraints: We need approx {est_beats} distinct visual scenes/beats to span the song.\n"
            f"Critical: Match the visual intensity to the Audio Profile (e.g. Calm visuals for 'Quiet', Intense for 'High Energy').\n"
            "Output JSON: { 'title': '...', 'synopsis': '...', 'beats': ['Visual description 1', 'Visual description 2', ...] }"
        )
        
        try:
            raw = text_engine.generate(story_req, json_schema=True)
            story_data = json.loads(raw)
            if isinstance(story_data, list):
                story_data = story_data[0]
            source_content = story_data.get('beats', [])
            context_str = f"Music Video: {story_data.get('title')}"
            logging.info(f"   📜 Generated {len(source_content)} story beats.")
            
        except Exception as e:
            logging.error(f"   ❌ Writer Failed: {e}")
            return

        # 3. Distribute Beats
        if not source_content:
             source_content = ["Band performing on stage."]

        num_beats = len(source_content)
        frames_per_beat = target_frames / num_beats
        
        # Style
        style_prefix = args.style
        
        prompts = []
        for i in range(target_frames):
             beat_idx = int(i / frames_per_beat)
             beat_idx = min(beat_idx, num_beats - 1)
             
             raw_desc = source_content[beat_idx]
             prompts.append(f"Style: {style_prefix}. Action: {raw_desc} (Frame {i+1}/{target_frames})")

    elif args.vpform == "creative-agency" or args.xb:
        # CREATIVE AGENCY MODE / RE-RENDER MODE
        
        # 2a. Determine Duration
        target_duration = 60.0 # Default
        if args.slength > 0:
             target_duration = args.slength
        
        # MUSIC OVERRIDE
        if args.mu and os.path.exists(args.mu):
             cmd = ['ffprobe', '-i', str(args.mu), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
             try:
                 res = subprocess.run(cmd, capture_output=True, text=True)
                 target_duration = float(res.stdout.strip())
                 logging.info(f"   🎵 Music Sync Engine: Locked to track duration {target_duration:.2f}s")
             except:
                 logging.warning("   ⚠️ Failed to probe music duration. Fallback to default.")
                 
        # 2b. Frame Count
        # We target the requested FPS (default 12 or 24)
        project_fps = 12 # Default for cartoons
        if args.fps > 1: project_fps = args.fps
        
        target_frames = int(target_duration * project_fps)
        logging.info(f"   ⏱️  Target: {target_frames} frames @ {project_fps} FPS ({target_duration:.1f}s)")
        
        # 2c. Source Material
        source_content = []
        context_str = ""
        
        
        if args.vpform == "music-visualizer":
            # === MUSIC VISUALIZER MODE ===
            if not args.mu or not os.path.exists(args.mu):
                 logging.error("❌ Music Visualizer requires --mu [audio_path]")
                 return
                 
            # 1. Analyze Audio
            bpm, duration, fpb, fps = analyze_audio(args.mu)
            target_frames = int(duration * fps)
            project_fps = fps
            target_duration = duration
            
            logging.info(f"   🎹 Visualizer Target: {target_frames} frames @ {fps:.2f} FPS ({duration:.1f}s)")
            
            # 2. Generate Abstract Prompts
            # No "Story", just pure visual evolution.
            prompts = []
            
            # Style Definition
            base_style = args.style if args.style != "Indie graphic novel artwork. Precise, uniform, dead-weight linework. Highly stylized, elegantly sophisticated, and with an explosive, highly saturated pop-color palette." else "Abstract, pixel art, Stan Brakhage style, melting film on hot projectors, unique fractal algorithm, highly saturated colors"
            
            style_prefix = base_style
            
            # Visualizer Loop
            for i in range(target_frames):
                 progress = i / target_frames
                 # Evolve description
                 phase = "forming patterns"
                 if progress > 0.3: phase = "melting into chaotic fractals"
                 if progress > 0.7: phase = "crystallizing into pure light"
                 
                 raw_desc = f"Visualizer Beat {i}. A single continuous abstract form {phase}. Progress: {int(progress*100)}%."
                 prompts.append(f"Style: {style_prefix}. Action: {raw_desc} (Frame {i+1}/{target_frames})")
                 
            # Set source_content to something not empty so we skip the default loop
            source_content = ["Visualizer Mode Active"] 

        elif args.xb:
             # RE-RENDER from XML
             logging.info(f"   📂 Ingesting Manifest: {args.xb}")
             # We need to parse portions from the XML
             # Rudimentary parsing or use mvp_shared helpers if robust enough.
             # Let's extract Portions raw for now.
             import xml.etree.ElementTree as ET
             try:
                 tree = ET.parse(args.xb)
                 root = tree.getroot()
                 # Find Portions JSON
                 p_node = root.find("Portions")
                 if p_node is not None and p_node.text:
                     portions_data = json.loads(p_node.text)
                     # Convert to list of content strings
                     for p in portions_data:
                         source_content.append(p.get('content', ''))
                     context_str = f"Re-render of {args.xb}"
                 else:
                     logging.error("   ❌ No Portions found in XML.")
                     return
             except Exception as e:
                 logging.error(f"   ❌ XML Parse Error: {e}")
                 return

        else:
            # CREATIVE AGENCY - Generate Story
            logging.info("   🧠 Creative Agency: Dreaming...")
            # text_engine already passed in args
            
            # Seeds
            seeds = []
            if args.cs > 0:
                logging.info(f"      🎲 Rolling {args.cs} Chaos Seeds...")
                seeds = [get_chaos_seed() for _ in range(args.cs)]
                
            prompt_concept = args.prompt if args.prompt else "A surreal sequence."
            
            logging.info(f"      Seeds: {seeds}")
            logging.info(f"      Prompt: {prompt_concept}")
            
            story_req = (
                f"Create a specific, visual screenplay for a {target_duration}s animation.\n"
                f"Concept: {prompt_concept}\n"
                f"Chaos Seeds to weave in: {seeds}\n"
                f"Constraints: {target_frames} visual beats needed (approx).\n"
                "Output JSON: { 'title': '...', 'synopsis': '...', 'beats': ['Visual description 1', 'Visual description 2', ...] }"
            )
            
            try:
                raw = text_engine.generate(story_req, json_schema=True)
                story_data = json.loads(raw)
                if isinstance(story_data, list):
                    story_data = story_data[0]
                source_content = story_data.get('beats', [])
                context_str = f"Title: {story_data.get('title')}. Synopsis: {story_data.get('synopsis')}"
                
                # Save Story/Project info for later XML export
                # (We'll do a simple hallucination or pass it through)
                
            except Exception as e:
                logging.error(f"   ❌ Writer Failed: {e}")
                return
                
        # 2d. Distribute Beats to Frames
        # If we have N beats and T frames.
        # Simple stretch: Each beat gets T/N frames.
        if not source_content:
             source_content = ["Static noise."]
             
        num_beats = len(source_content)
        frames_per_beat = target_frames / num_beats
        
        # Use new Flipbook Style
        # Override with args.style if present (already defaulted to Indie Graphic Novel)
        style_prefix = args.style
        
        for i in range(target_frames):
             beat_idx = int(i / frames_per_beat)
             beat_idx = min(beat_idx, num_beats - 1)
             
             raw_desc = source_content[beat_idx]
             # Dynamically append frame index to encourage movement in the LLM's mind
             # We inject the STYLE here so it travels with the prompt to the Director.
             prompts.append(f"Style: {style_prefix}. Action: {raw_desc} (Frame {i+1}/{target_frames})")
             
    else:
        # Interpolation Mode (Legacy)
        # 2. Calculate Target Frames
        # Legacy treats args.fps as direct Output FPS
        project_fps = args.fps # Use directly
        target_frames = math.ceil(duration * project_fps) if duration else 120
        logging.info(f"Legacy Target Frames: {target_frames} (@ {project_fps} FPS)")
        
        # 3. Expand Descriptions (Style Injection)
        num_desc = len(descriptions)
        
        style_prefix = """
        You are a commercial animator working in a large production house with Fortune 500 clients in the 1990's. though you have SOME room for artistic expression and deviating from the frame prompt you are given, in order to fulfill your job as an animator, you need to ensure that you reference the previous frame (provided helpfully as context) and draw the exact next frame in the scene's (and thus the narrative's) trajectory.
        VISUAL: """
        
        for i in range(target_frames):
            if num_desc > 0:
                desc_idx = int( (i / target_frames) * num_desc )
                desc_idx = min(desc_idx, num_desc - 1)
                raw_desc = descriptions[desc_idx]
                # Harden here too? Why not.
                raw_desc = re.sub(r"^(Frame\s*\d+\s*[:.-]\s*)", "", raw_desc, flags=re.IGNORECASE).strip()
            else:
                raw_desc = "A static screen of colorful noise."
                
            prompts.append(f"{style_prefix} {raw_desc}")

    # 4. Prepare Output
    ts = int(time.time())
    # Standardize output folder naming for MVP
    ts = int(time.time())
    # Standardize output folder naming for MVP
    # If the project name already looks like a final folder (MusicVideo_...), don't prepend "cartoon_"
    if project_name.startswith("MusicVideo_") or project_name.startswith("Agency_"):
        project_out = output_root / f"{project_name}" # Use exact name we generated in main
    else:
        project_out = output_root / f"cartoon_{project_name}_{ts}"
    frames_dir = project_out / "frames"
    ensure_dir(frames_dir)
    
    # 5. Generation Loop
    success_count = 0
    # Client is now created per frame for rotation
    
    # Calculate project_out here if needed for Visualizer? 
    # It's calculated above.

    
    # Limit for testing?
    if args.limit and args.limit > 0:
        prompts = prompts[:args.limit]
        logging.info(f"⚠️ Limit applied: Generating only {args.limit} frames.")

    # TWEENING LOOP (Phase 3)
    # Iterate in steps of 2
    # Frame i: Generate
    # Frame i+1: Tween (from i and i+2)
    # Frame i+2: Generate
    
    step_size = 2 if args.fc else 1 # Only tween in FC mode (for now)
    
    # We need a list to index into (prompts)
    # but we ALSO need to handle the fact that we might run out of prompts?
    # No, prompts list assumes 1 prompt per frame?
    # Yes, len(prompts) == target_frames roughly.
    
    last_generated_img = None
    last_generated_idx = -1
    
    # We iterate i from 0 to len(prompts) with step
    # BUT we need to be careful about not skipping the last one if ODD.
    
    # Actually, simpler logic:
    # Iterate normally.
    # If fc and i % 2 != 0 (Odd frame):
    #    Skip standard generation. Mark for "Tween Backfill".
    #    But we can't backfill until we have i+1 (Even).
    #    So we generate i (Even).
    #    If we have i-2 (Previous Even), we TWEEN i-1.
    
    # === NEW RECURSIVE LOGIC (5-Frame Buildout) ===
    # 1 (Flux) -> 5 (Flux) -> 3 (Flux Tween) -> 2,4 (Pixel Fills)
    recursive_mode = (getattr(args, 'fc', False) and getattr(args, 'vpform', '') in ['music-video', 'full-movie'] and getattr(args, 'local', False))

    if recursive_mode:
        logging.info("🚀 Engaging 5-Frame Recursive Flux Buildout (Local FC Mode)...")
        i_idx = 0
        N = len(prompts)
        
        while i_idx < N:
            # 1. Start Frame (Frame Index i+1)
            frame_start_idx = i_idx + 1
            start_path = frames_dir / f"frame_{frame_start_idx:04d}.png"
            
            start_img = None
            if i_idx == 0:
                 # Generate First Frame via Flux (Seed)
                 logging.info(f"   🎨 Batch Start: Frame {frame_start_idx} (Flux Seed)...")
                 start_img = frame_canvas.generate_seed_image(prompts[i_idx], te, init_dim=args.kid)
                 if start_img:
                     start_img.save(start_path)
                     last_generated_img = start_img
                     success_count += 1
            else:
                 # Load existing start frame
                 if start_path.exists():
                      from PIL import Image
                      start_img = Image.open(start_path)
                 else:
                      logging.warning(f"   ⚠️ Start Frame {frame_start_idx} missing in batch. Regenerating...")
                      start_img = frame_canvas.generate_seed_image(prompts[i_idx], te, init_dim=args.kid)
                      if start_img: start_img.save(start_path)

            if not start_img:
                logging.error("   ❌ Batch Start Failed. Aborting batch.")
                i_idx += 1
                continue

            # 2. Determine End Frame (Target: +4 frames)
            step = 4
            next_i = min(i_idx + step, N - 1)
            
            if next_i <= i_idx:
                 break # End of sequence
                 
            # 3. Generate End Frame via Flux
            frame_end_idx = next_i + 1
            end_path = frames_dir / f"frame_{frame_end_idx:04d}.png"
            logging.info(f"   🎨 Batch End: Frame {frame_end_idx} (Flux)...")
            
            end_img = frame_canvas.generate_seed_image(prompts[next_i], te, init_dim=args.kid)
            if end_img:
                end_img.save(end_path)
                success_count += 1
            else:
                 logging.error(f"   ❌ Batch End Frame {frame_end_idx} Failed.")
                 # Fallback: Just advance one step linearly if Flux fails?
                 i_idx += 1
                 continue

            # 4. Generate Mid Frame via Tween+Flux (i + next_i // 2)
            mid_i = (i_idx + next_i) // 2
            mid_img = None
            
            if mid_i > i_idx and mid_i < next_i:
                 frame_mid_idx = mid_i + 1
                 mid_path = frames_dir / f"frame_{frame_mid_idx:04d}.png"
                 logging.info(f"   ✨ Batch Mid: Frame {frame_mid_idx} (Flux Tween)...")
                 
                 tween_mid = frame_canvas.tween_frames(start_img, end_img, blend=0.5)
                 # Refine Tween with Flux
                 mid_img = frame_canvas.refine_tween(tween_mid, prompts[mid_i], width=target_w, height=target_h, text_engine=te)
                 if mid_img:
                      mid_img.save(mid_path)
                      success_count += 1
                 else:
                      logging.warning("   ⚠️ Mid-Tween Failed.")
                      mid_img = tween_mid # Fallback to raw tween

            # 5. Fill Gaps with Brute Force Blend (Tween)
            # Gap 1: Start -> Mid
            gap1_i = i_idx + 1
            if gap1_i < mid_i and mid_img:
                 logging.info(f"   🌪️  Batch Fill A: Frame {gap1_i+1} (Brute Force Blend)...")
                 # Tween Start <-> Mid (50% blend)
                 fill1 = frame_canvas.tween_frames(start_img, mid_img, blend=0.5)
                 if fill1:
                      fill1.save(frames_dir / f"frame_{gap1_i+1:04d}.png")
                      success_count += 1

            # Gap 2: Mid -> End
            gap2_i = mid_i + 1
            if gap2_i < next_i and mid_img and end_img:
                  logging.info(f"   🌪️  Batch Fill B: Frame {gap2_i+1} (Brute Force Blend)...")
                  # Tween Mid <-> End (50% blend)
                  fill2 = frame_canvas.tween_frames(mid_img, end_img, blend=0.5)
                  if fill2:
                       fill2.save(frames_dir / f"frame_{gap2_i+1:04d}.png")
                       success_count += 1

            # Advance loop: The End Frame becomes the new Start Frame
            i_idx = next_i
            last_generated_img = end_img

    legacy_prompts = [] if recursive_mode else prompts
    for i, p in enumerate(legacy_prompts):
        index = i + 1
        
        # Pass model from args
        model_to_use = getattr(args, 'model', None)
        if not model_to_use:
             model_to_use = get_active_model(Modality.IMAGE).name

        # Define Previous Frame Path
        prev_frame = None
        if i > 0:
             prev_frame = frames_dir / f"frame_{i:04d}.png" # Wait, this looks for i (0-based) named frame_i? No frame_{i:04d} is frame i. 
             # Logic earlier: index = i+1. So frame_{index:04d} is CURRENT.
             # So prev_frame is frame_{i:04d} (index-1). Correct.
        
        # FC MODE CHECK
        if args.fc:
            # Code Painter Mode
            # Reuse 'te' from top of function to preserve key rotation
            pass
            
            # Context

            # We inject Past (i-1) and Future (i+1) prompts to help the model flow
            prev_p = prompts[i-1] if i > 0 else "Start of sequence."
            next_p = prompts[i+1] if i < len(prompts) - 1 else "End of sequence."
            
            ctx = {
                "global_concept": context_str if 'context_str' in locals() else "Animation",
                "frame_index": index,
                "total_frames": target_frames if 'target_frames' in locals() else 0,
                "prev_frame_summary": prev_p, 
                "next_frame_summary": next_p
            }
            
            # TWEEN LOGIC:
            # We ONLY generate on EVENS (0, 2, 4...)
            # Odd frames (1, 3, 5) are skipped here and filled AFTER we get the next Even.
            
            is_even = (i % 2 == 0)
            is_last = (i == len(prompts) - 1)
            
            current_img = None
            
            if is_even or (is_last and not is_even): # Always generate last frame to cap the sequence
                logging.info(f"   🎨 FC Code Painter: Frame {index} (Keyframe)...")
                # Recursive Gen
                current_img = frame_canvas.generate_recursive(
                    prompt=p, # The full prompt
                    width=target_w, height=target_h, 
                    context=ctx, text_engine=te,
                    prev_img=last_generated_img,
                    init_dim=args.kid # Pass KID
                )
                
                target_path = frames_dir / f"frame_{index:04d}.png"
                if current_img:
                    current_img.save(target_path)
                    print("K", end="", flush=True) # K for Keyframe
                    success_count += 1
                    
                    # BACKFILL TWEEN check
                    # If we just generated i (Even), and i > 1, it means we skipped i-1 (Odd).
                    # We need to tween i-1 using (i-2) and (i).
                    if i > 1 and last_generated_img:
                        # Index of gap frame: i-1 (0-based) -> frame_{(i):04d}
                        gap_idx = i - 1
                        gap_frame_num = gap_idx + 1
                        gap_prompt = prompts[gap_idx]
                        
                        logging.info(f"      ✨ Tweening Gap: Frame {gap_frame_num}...")
                        
                        # Tween
                        tween_img = frame_canvas.tween_frames(last_generated_img, current_img, blend=0.5)
                        # Refine
                        refined_tween = frame_canvas.refine_tween(tween_img, gap_prompt, model=None, text_engine=te, width=target_w, height=target_h)
                        
                        # Save
                        gap_path = frames_dir / f"frame_{gap_frame_num:04d}.png"
                        if refined_tween:
                            refined_tween.save(gap_path)
                            print("t", end="", flush=True) # t for tween
                            success_count += 1
                            
                    # Update Memory
                    last_generated_img = current_img
                    last_generated_idx = i
                else:
                    logging.error(f"   [-] Frame {index} failed (Code Gen).")
                    
                    # FALLBACK FOR FC MODE
                    logging.warning(f"Frame {index}: FC Code Gen Failed. Triggering Stutter Clone.")
                    target_path = frames_dir / f"frame_{index:04d}.png"
                    
                    clone_source = None
                    if prev_frame and prev_frame.exists():
                        clone_source = prev_frame
                    else:
                         for back_step in range(1, 500):
                             test_idx = index - back_step
                             if test_idx < 1: break
                             test_path = frames_dir / f"frame_{test_idx:04d}.png"
                             if test_path.exists():
                                 clone_source = test_path
                                 break
                    
                    if clone_source:
                        shutil.copy(clone_source, target_path)
                        success_count += 1 
                        logging.warning(f"   🩹 Stuttered: Cloned {clone_source.name} -> {target_path.name}")
                        print("x", end="", flush=True)
                        
                        # RECOVERY FOR BACKFILL
                        # We must load this cloned frame as 'current_img' so we can backfill the gap (i-1)
                        try:
                            from PIL import Image
                            current_img = Image.open(target_path)
                            # Update Memory
                            last_generated_img = current_img
                            last_generated_idx = i
                            
                            # TRIGGER BACKFILL (Copy-Paste of Logic Above)
                            if i > 1:
                                gap_idx = i - 1
                                gap_frame_num = gap_idx + 1
                                gap_prompt = prompts[gap_idx]
                                
                                # Since 'current' is a clone of 'prev' (likely), the tween will be static.
                                # But we MUST produce the file.
                                tween_img = current_img.copy() # Just clone it again for stability
                                
                                gap_path = frames_dir / f"frame_{gap_frame_num:04d}.png"
                                tween_img.save(gap_path)
                                print("x", end="", flush=True) # x for stutter-fill
                                success_count += 1
                                
                        except Exception as e_rec:
                            logging.error(f"   Recovery Backfill Failed: {e_rec}")

                    else:
                        print("!", end="", flush=True) # Fatal gap
            
            else:
                # ODD FRAME (Skip)
                # We will backfill this when we hit the next Even.
                print("_", end="", flush=True) # _ for skip
                pass 
                
            continue # Skip standard gen
        else:
            # STANDARD GENERATION (Gemini/Imagen/Universal)
            success = generate_frame_universal(
                index=index,
                prompt=p,
                output_dir=frames_dir,
                key_cycle=key_cycle,
                model=model_to_use,
                prev_frame_path=prev_frame,
                pg_mode=args.pg,
                # Custom Resolution Logic: If Local, allow override. Else KID.
                width=args.w if (args.local and args.w) else args.kid,
                height=args.h if (args.local and args.h) else args.kid,
                force_local=args.local
            )
            
            if success:
                print(".", end="", flush=True)
                success_count += 1
                continue
            
            # FAILURE FALLBACK (Gap Filling)
            # If generation fails, we MUST fill the gap or FFmpeg will stop stitching at this frame.
            logging.warning(f"Frame {index}: Generation Failed. Duplicating previous frame to maintain sync.")
             
            target_path = frames_dir / f"frame_{index:04d}.png"
            
            # Search backwards for ANY valid frame to clone (Stutter Step)
            # This protects against chain failures where Frame N-1 is also missing.
            clone_source = None
            if prev_frame and prev_frame.exists():
                clone_source = prev_frame
            else:
                # Deep Search (Look back 500 frames)
                for back_step in range(1, 500):
                    test_idx = index - back_step
                    if test_idx < 1: break
                    test_path = frames_dir / f"frame_{test_idx:04d}.png"
                    if test_path.exists():
                        clone_source = test_path
                        break
            
            if clone_source:
                shutil.copy(clone_source, target_path)
                success_count += 1 
                logging.warning(f"   🩹 Stuttered: Cloned {clone_source.name} -> {target_path.name}")
            else:
                # If Frame 1 fails or no history, create a black frame.
                try:
                    from PIL import Image
                    img = Image.new('RGB', (768, 768), color='black')
                    img.save(target_path)
                    success_count += 1
                    logging.warning(f"   ⚫ Black Frame: Created {target_path.name}")
                except ImportError:
                    logging.error("PIL missing. Cannot create fallback frame.")
             
            print("x", end="", flush=True)

        # DELAY (Quota protection)
        if hasattr(args, 'delay') and args.delay > 0:
            # Add small Jitter (0-0.5s) to avoid harmonic resonance with API windows
            jitter = random.uniform(0.0, 0.5)
            time.sleep(args.delay + jitter)
             
    print("\n")
    
    # 6. Assembly (Stitch & Mux)
    if success_count > 0:
        raw_video = project_out / "raw_video.mp4"
        final_video = project_out / "final_cut.mp4"
        
        logging.info("🧵 Stitching Video...")
        # Recalculate frames pattern because Visualizer might differ? No, standard logic.
        frames_pattern = frames_dir.resolve() / "frame_%04d.png"
        
        # Ensure FPS is valid
        final_fps = project_fps if args.vpform in ["music-visualizer", "music-agency"] else args.fps if args.fps > 1 else 12 
        # Wait, local var project_fps is set for visualizer.
        # But if we are in Legacy mode, it's args.fps.
        # Let's trust project_fps variable if set.
        current_fps = project_fps if 'project_fps' in locals() else args.fps
        
        # 0.5 Re-sequence Frames (Gap Fixing)
        # Ensure frame_0001, frame_0002, frame_0003... are contiguous
        logging.info("   🔧 Re-sequencing frames for stitching...")
        all_frames = sorted(list(frames_dir.glob("frame_*.png")))
        temp_stitch_dir = project_out / "temp_stitch_frames"
        if temp_stitch_dir.exists(): shutil.rmtree(temp_stitch_dir)
        temp_stitch_dir.mkdir()
        
        for i, src in enumerate(all_frames):
            dst = temp_stitch_dir / f"frame_{i+1:04d}.png"
            
            # SAFE STITCHING: Resize if mismatch
            # We use args.w/h if present, else args.kid, else 768
            target_w = args.w if args.w else (args.kid if args.kid else 768)
            target_h = args.h if args.h else (args.kid if args.kid else 768)
            
            try:
                from PIL import Image
                with Image.open(src) as img:
                    if img.size != (target_w, target_h):
                        logging.debug(f"   Resize {src.name} {img.size} -> {target_w}x{target_h}")
                        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
                        img_resized.save(dst)
                    else:
                        shutil.copy(src, dst)
            except ImportError:
                shutil.copy(src, dst) # Fallback if PIL missing
            
        frames_pattern = temp_stitch_dir / "frame_%04d.png"

        # 1. Video Only
        cmd_vid = [
            "ffmpeg", "-y",
            "-framerate", str(current_fps),
            "-i", str(frames_pattern),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(raw_video)
        ]
        
        try:
            subprocess.run(cmd_vid, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logging.error(f"FFmpeg stitching failed: {e}")
            return

        # 2. Mux with Audio (if available or provided via --mu)
        audio_source = None
        
        # Priority 1: Explicit Audio File (--mu)
        if args.mu and os.path.exists(args.mu):
             audio_source = Path(args.mu)
             logging.info(f"   🎵 Using provided music track: {audio_source}")
        # Priority 2: Inferred Original Video (Legacy/FBF)
        elif original_video and original_video.exists():
             audio_source = original_video
             logging.info(f"   🎵 Using original video audio: {audio_source}")
        
        if audio_source and raw_video.exists():
            logging.info("🔊 Muxing Audio...")
            cmd_mix = [
                "ffmpeg", "-y",
                "-i", str(raw_video),
                "-i", str(audio_source),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-shortest", # Clip video to audio length (crucial for music videos)
                str(final_video)
            ]
            try:
                subprocess.run(cmd_mix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"✅ FINAL CUT (with Audio): {final_video}")
            except Exception as e:
                logging.error(f"Audio mux failed: {e}. Fallback to raw video.")
        else:
             logging.info(f"✅ FINAL CUT (Silent): {raw_video}")
             
        # 3. Post-Processing (Music Visualizer Only)
        if args.vpform == "music-visualizer" and raw_video.exists():
            logging.info("🔥 Entering Post-Production: Obsessive Video Repainting (ASCII Forge)...")
            
            ascii_vid = project_out / "ascii_version.mp4"
            if run_ascii_forge(raw_video, ascii_vid):
                logging.info(f"   ✅ ASCII Version Forged: {ascii_vid}")
                
                # Blend
                blended_vid = project_out / "final_blend.mp4"
                logging.info("   🎨 Blending ASCII Overlay (33%)...")
                # Use final_video logic? No, we have raw_video.
                if blend_videos(raw_video, ascii_vid, blended_vid, opacity=0.33):
                     logging.info(f"   ✅ FINAL VISUALIZER MASTER: {blended_vid}")
                     
                     # Mux Audio
                     final_master = project_out / "master_release.mp4"
                     if args.mu:
                         cmd_remix = [
                            "ffmpeg", "-y",
                            "-i", str(blended_vid),
                            "-i", str(args.mu),
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-shortest",
                            str(final_master)
                         ]
                         try:
                             subprocess.run(cmd_remix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                             logging.info(f"   🏆 MASTER RELEASE: {final_master}")
                         except:
                             logging.warning("   ⚠️ Audio remix failed.")
             
        # 3. Post-Processing (Music Visualizer Only)
        if args.vpform == "music-visualizer" and final_video.exists():
            logging.info("🔥 Entering Post-Production: Obsessive Video Repainting (ASCII Forge)...")
            
            ascii_vid = project_out / "ascii_version.mp4"
            if run_ascii_forge(raw_video, ascii_vid):
                logging.info(f"   ✅ ASCII Version Forged: {ascii_vid}")
                
                # Blend
                blended_vid = project_out / "final_blend.mp4"
                logging.info("   🎨 Blending ASCII Overlay (33%)...")
                if blend_videos(final_video, ascii_vid, blended_vid, opacity=0.33):
                     logging.info(f"   ✅ FINAL VISUALIZER MASTER: {blended_vid}")
                     # Ensure audio is present (blend might strip it if checking video streams only?)
                     # Blend takes video from 0 and 1. Audio from 0? 
                     # ffmpeg complex filter usually drops audio unless mapped.
                     # Let's Remix audio just to be safe.
                     final_master = project_out / "master_release.mp4"
                     if args.mu:
                         cmd_remix = [
                            "ffmpeg", "-y",
                            "-i", str(blended_vid),
                            "-i", str(args.mu),
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-shortest",
                            str(final_master)
                         ]
                         try:
                             subprocess.run(cmd_remix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                             logging.info(f"   🏆 MASTER RELEASE: {final_master}")
                         except:
                             logging.warning("   ⚠️ Audio remix failed.")

    # 7. Frame Cleanup (Keep First Frame Only)
    # -------------------------------------------------------------------------
    try:
        logging.info("🧹 Cleaning up frames (Keeping only frame_0001.png)...")
        deleted_count = 0
        for frame_file in frames_dir.glob("frame_*.png"):
            if frame_file.name != "frame_0001.png":
                try:
                    frame_file.unlink()
                    deleted_count += 1
                except Exception as e_del:
                    logging.warning(f"Failed to delete {frame_file.name}: {e_del}")
        logging.info(f"   Deleted {deleted_count} intermediate frames.")
    except Exception as e_clean:
         logging.error(f"Cleanup failed: {e_clean}")

    # 8. XMVP Export (Storyboard Mode)
    # -------------------------------------------------------------------------
    try:
        logging.info("📜 Generating XMVP XML (Storyboard)...")
        import sys
        # Assuming we are in tools/fmv/mvp/v0.5 already, but for robustness:
        mvp_path = Path(__file__).resolve().parent
        if str(mvp_path) not in sys.path:
            sys.path.append(str(mvp_path))
        
        import mvp_shared
        from mvp_shared import VPForm, CSSV, Constraints, Story, Portion
        
        # A. Hallucinate Metadata
        # Use a fresh client/key
        # A. Hallucinate Metadata
        # Use a fresh client/key
        meta_key = get_random_key(keys)
        if not meta_key:
             meta_key = os.environ.get("GEMINI_API_KEY")
        
        if not meta_key:
             logging.warning("   ⚠️ No API Key available for Metadata Hallucination. Skipping context.")
             meta_result = {"title": f"Cartoon {project_name}", "synopsis": "A generated cartoon.", "characters": [], "themes": [], "vibe": "Hand Drawn", "vibe": "Hand Drawn"}
        else:
             meta_client = genai.Client(api_key=meta_key)
             # ... continue logic below

        
        # Sample descriptions to save tokens/time
        sample_size = 20
        if len(descriptions) > sample_size:
            step = len(descriptions) // sample_size
            sample_desc = descriptions[::step][:sample_size]
        else:
            sample_desc = descriptions
            
        logging.info("   🔮 Hallucinating Story Context from Frames...")
        
        meta_prompt = f"""
        Analyze these visual frame descriptions from a video storyboard and hallucinate the underlying Movie Metadata.
        
        FRAMES:
        {json.dumps(sample_desc)}
        
        OUTPUT JSON ONLY:
        {{
            "title": "A catchy title",
            "synopsis": "A 2-sentence summary of the plot arc implied by these frames.",
            "characters": ["Names of implied characters"],
            "themes": ["Theme 1", "Theme 2"],
            "vibe": "The visual aesthetic (e.g. Noir, Cyberpunk)"
        }}
        """
        
        meta_result = {"title": f"Cartoon {project_name}", "synopsis": "A generated cartoon.", "characters": [], "themes": [], "vibe": "Hand Drawn"}
        try:
            # Use Flash 2.5 (Gemini L)
            resp = meta_client.models.generate_content(model="gemini-2.5-flash-image", contents=meta_prompt)
            if resp.text:
                clean_json = re.sub(r"```json|```", "", resp.text).strip()
                meta_result = json.loads(clean_json)
        except Exception as e_meta:
            logging.warning(f"Metadata hallucination failed: {e_meta}")
            
        # B. Construct Models
        vp = VPForm(
            name="cartoon-storyboard", 
            fps=args.fps, 
            description="Automated Cartoon Storyboard from FBF Analysis",
            mime_type="video/mp4" # Target
        )
        
        cssv = CSSV(
            constraints=Constraints(width=768, height=768, fps=24, target_segment_length=4.0),
            scenario=f"A story derived from {project_name}.",
            situation=meta_result.get('synopsis', 'Visual sequence.'),
            vision=f"Style: {meta_result.get('vibe', 'Animation')}. Created by Cartoon Producer."
        )
        
        story = Story(
            title=meta_result.get('title', project_name),
            synopsis=meta_result.get('synopsis', ''),
            characters=meta_result.get('characters', ['Unknown']),
            theme=meta_result.get('themes', ['Animation'])[0] if meta_result.get('themes') else "General"
        )
        
        # C. Map Portions
        # Map each Prompt to a Portion
        portions = []
        # Use prompts list which has the hardened/expanded prompts
        # Segment length? 
        # If FBF, each prompt is 1 frame? 
        # But Portion is a narrative chunk.
        # Let's map 1:1 for maximum granularity or aggregate? 
        # 1:1 is huge for XMVP if thousands of frames.
        # But 'process_project' calculated 'prompts' list.
        # If explicit fps expansion was used, prompts has duplicates/interpolations.
        # Let's use the Original 'descriptions' for the Portions (the Narrative beats)
        # NOT the interpolated frames.
        
        for i, desc in enumerate(descriptions):
             portions.append(Portion(
                 id=i+1,
                 duration_sec=2.0, # Arbitrary default for viewing
                 content=desc
             ))
             
        xmvp_data = {
            "VPForm": vp,
            "CSSV": cssv,
            "Story": story,
            "Portions": [p.model_dump() for p in portions]
        }
        
        xml_name = f"storyboard_{project_name}_{ts}.xml"
        xml_path = project_out / xml_name
        mvp_shared.save_xmvp(xmvp_data, xml_path)
        logging.info(f"   📜 XMVP Saved: {xml_path}")
        
        # --- WAN VIDEO HIJACK (Full Movie) ---
        if args.vpform == "full-movie":
            logging.info("🎬 Mode: full-movie (Wan 2.1 Video Engine). Hijacking FBF Loop.")
            
            # Convert XMVP Portions to Manifest JSON
            # We need a 'manifest.json' for dispatch_wan
            manifest_path = project_out / "manifest.json"
            manifest = {
                "vpform": "full-movie",
                "portions": [p.model_dump() for p in portions]
            }
            # Note: Portions in cartoon mode typically lack 'audio_path'.
            # dispatch_wan assumes some audio might exist or generates silence?
            # User said: "We will generate all of the dialogue lines BEFORE..."
            # In cartoon_producer, 'generate_visual_script' creates the text.
            # But where is the audio?
            # If using 'transcript folder', audio might be aligned? 
            # Or assume text-to-video only (Wan supports it).
            # We will pass what we have.
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            import dispatch_wan
            success = dispatch_wan.run_wan_pipeline(
                manifest_path=manifest_path,
                out_path=project_out / "manifest_updated.json",
                staging_dir=project_out / "parts",
                args=args
            )
            
            if success:
                logging.info("✅ Wan Pipeline Complete.")
                return # Skip FBF Loop
            else:
                logging.error("❌ Wan Pipeline Failed.")
                return

        # 6. Cleanup
        # "Delete all but frame one from the frames folder"
        logging.info("   🧹 Cleaning up frames...")
        try:
            frames_to_keep = ["frame_0001.png"]
            if frames_dir.exists():
                for f in frames_dir.iterdir():
                    if f.name not in frames_to_keep:
                        f.unlink()
        except Exception as e:
            logging.warning(f"   ⚠️ Cleanup failed: {e}")
        
    except Exception as e_xml:
        logging.error(f"XMVP Generation Failed: {e_xml}")

def get_project_duration(project_dir):
    """
    Quickly peeks at metadata.json to get duration.
    Returns 0.0 if not found.
    """
    meta_path = project_dir / "metadata.json"
    if meta_path.exists():
        try:
             with open(meta_path, 'r') as f:
                 data = json.load(f)
                 return float(data.get("duration", 0.0))
        except: pass
    # Fallback to analysis length? 
    # For speed, we might skip this unless necessary, but let's be robust.
    analysis_path = project_dir / "analysis.json"
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r') as f:
                data = json.load(f)
                return float(len(data)) # customized fallback: 1 sec per line
        except: pass
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Cartoon Producer v1.1: The Creative Agency")
    
    # CLI Polish: Positional Args for Aliases
    definitions.add_global_vpform_args(parser)

    parser.add_argument("--vpform", type=str, default=None, help="VP Form (default: creative-agency)") # Default via logic
    parser.add_argument("--tf", type=Path, default=DEFAULT_TF, help="Transcript Folder (Source)")
    parser.add_argument("--vf", type=Path, default=DEFAULT_VF, help="Video Folder (Corpus)")
    parser.add_argument("--fps", type=int, default=4, help="Expansion Factor (FBF) or Output FPS (Legacy). Default: 1")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between requests in seconds (Default: 5.0)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of frames per project (for testing)")
    parser.add_argument("--project", type=str, default=None, help="Specific project name to process (Optional)")
    parser.add_argument("--smin", type=float, default=0.0, help="Minimum duration in seconds")
    parser.add_argument("--smax", type=float, default=None, help="Maximum duration in seconds")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle projects before processing")
    # v1.1 Args
    parser.add_argument("--xb", type=str, help="Path to XMVP XML Manifest for re-rendering")
    parser.add_argument("--mu", type=str, help="Path to Music/Audio file for sync")
    parser.add_argument("--prompt", type=str, help="Creative Prompt for Agency Mode")
    parser.add_argument("--style", type=str, default="Indie graphic novel artwork. Precise, uniform, dead-weight linework. Highly stylized, elegantly sophisticated, and with an explosive, highly saturated pop-color palette.", help="Visual Style Definition")
    parser.add_argument("--slength", type=float, default=60.0, help="Target length in seconds (if no music)")
    parser.add_argument("--cs", type=int, default=0, choices=[0, 1, 2, 3], help="Chaos Seeds Level (0=None, 0-3)")
    parser.add_argument("--bpm", type=float, help="Manual BPM override for music modes (bypasses detection)")
    parser.add_argument("--pg", action="store_true", help="Enable PG Mode (Relaxed Celebrity/Strict Child Safety)")
    
    # v1.2 FC Integration
    parser.add_argument("--vspeed", type=float, default=8.0, help="Visualizer Speed (FPS) for music-agency. Default 8. Supports 2, 4, 16.")
    parser.add_argument("--fc", action="store_true", help="Enable Frame & Canvas (Code Painter) Mode")
    parser.add_argument("--kid", type=int, default=512, help="Keyframe Init Dimension (default: 512). Higher = Better composition before downscale.")
    parser.add_argument("--local", action="store_true", help="Local Mode (Use Gemma + Flux)") # NEW
    parser.add_argument("--w", type=int, help="Override width (Local Only, e.g. 1920)")
    parser.add_argument("--h", type=int, help="Override height (Local Only, e.g. 1080)")
    
    args, unknown = parser.parse_known_args()

    # Handle 'run' in unknown if not captured by cli_args
    for u in unknown:
        if u.lower() == "run":
            pass # Ignored command
        elif u.startswith("-"):
            logging.warning(f"Unknown flag ignored: {u}")
    
    # CLI Polish: Handle Aliases via Global Registry
    # This automatically resolves 'music-agency', 'mv', 'viz' -> 'music-video'
    args.vpform = definitions.parse_global_vpform(args, current_default="creative-agency")
    
    logging.info(f"   CLI: Resolved VPForm: {args.vpform}")

    # Auto-Carbonation (Sassprilla)
    # Check for "Song Title Case" prompts (Short, Title Case, No Periods)
    if args.prompt:
        p_clean = args.prompt.strip()
        # Heuristic: Title Case, No periods (not a sentence), relatively short (< 60 chars)
        if p_clean.istitle() and "." not in p_clean and len(p_clean) < 80:
            logging.info(f"🫧 Auto-Carbonating Title Prompt: '{p_clean}'...")
            try:
                import sassprilla_carbonator
                expanded = sassprilla_carbonator.carbonate_prompt(p_clean)
                if expanded:
                    logging.info(f"   ✨ Expanded to {len(expanded)} chars.")
                    args.prompt = expanded
            except Exception as e:
                logging.warning(f"   ⚠️ Carbonation failed: {e}")
    
    # Setup Paths
    # Point to CENTRAL env_vars.yaml in tools/fmv/
    env_file = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
    if not env_file.exists():
         logging.warning(f"Central env_vars.yaml not found at {env_file}. Attempting fallback...")
         env_file = Path(__file__).resolve().parent / "env_vars.yaml"
    
    # Using mvp/v0.5 specific output dir
    base_dir = Path(__file__).resolve().parent # v0.5
    output_root = base_dir / "z_test-outputs" / "cartoons"
    ensure_dir(output_root)

    keys = load_keys(env_file)
    if not keys:
        logging.warning("No API keys found in env_vars.yaml! Checking GEMINI_API_KEY env var...")
        if not os.environ.get("GEMINI_API_KEY"):
            logging.error("❌ No API Keys found. Aborting.")
            return

    logging.info("🎬 Cartoon Producer Initialized.")
    logging.info(f"   Mode: {args.vpform}")
    logging.info(f"   Key Pool Size: {len(keys)}")
    logging.info(f"   Transcript Folder: {args.tf}")
    logging.info(f"   Video Folder: {args.vf}")
    
    # 0. Load Keys (Already done at startup)
    
    # 0.5 Full-Movie Resolution Default
    if args.vpform == "full-movie":
        if not args.w: args.w = 512
        if not args.h: args.h = 288
        logging.info(f"   🎥 Full Movie Mode: Enforcing {args.w}x{args.h}")
        
    # --- PRODUCER MODE SWITCHING ---
    # --- PRODUCER MODE SWITCHING ---
    if args.local:
        logging.info("🔌 Local Mode Requested. Switching Registry to Local Models...")
        try:
            # Switch Text -> Gemma (Use 4-bit Quantized by default for Safety)
            definitions.set_active_model(Modality.TEXT, "gemma-2-9b-it")
            os.environ["TEXT_ENGINE"] = "local_gemma"
            
            # Resolve Gemma Path Dynamically
            gemma_conf = definitions.MODAL_REGISTRY[Modality.TEXT].get("gemma-2-9b-it")
            if gemma_conf and gemma_conf.path:
                # Force the 4-bit safe path if the registry points to raw
                # Or trust TextEngine's default? 
                # TextEngine default is "mlx-community/gemma-2-9b-it-4bit"
                # But here we are setting env var override. 
                # Let's set it to the safe default explicitly to fail-safe.
                os.environ["LOCAL_MODEL_PATH"] = "mlx-community/gemma-2-9b-it-4bit"
                logging.info(f"   📍 Text Model Path: {os.environ['LOCAL_MODEL_PATH']}")
                
            # Switch Image -> Flux
            definitions.set_active_model(Modality.IMAGE, "flux-schnell")
            # Switch Video -> LTX (Future proofing)
            definitions.set_active_model(Modality.VIDEO, "ltx-video")
        except Exception as e:
            logging.error(f"❌ Failed to switch to Local Mode: {e}")
            
    # Initialize Engines
    # ----------------
    # Text Engine (Global Singleton)
    text_engine = TextEngine(config_path="env_vars.yaml")
    
    # Re-verify local status if requested
    if args.local and text_engine.backend != "local_gemma":
        logging.warning("⚠️ Local Mode requested but TextEngine didn't switch? Forcing...")
        text_engine.backend = "local_gemma"
        text_engine._init_local_model()

    # Truth Safety
    safety = TruthSafety()
    
    # 1. Model Selection (Registry Aware)
    # Start with the active model (which might have been set to Flux by --local block above)
    model = get_active_model(Modality.IMAGE).name
    
    if args.vpform == "fbf-cartoon":
         # Force Image Model? Or respect Registry?
         # Legacy behavior was specific. Let's stick to Registry unless overridden.
         pass
    elif args.vpform in ["creative-agency", "music-visualizer", "music-agency", "music-video", "full-movie"]:
        # Only switch to Gemini 2.0 (Director Mode) if NOT local.
        # Local mode uses Flux directly.
        if not args.local:
            model = "gemini-2.0-flash"
            logging.info(f"   🎨 {args.vpform}: Using Gemini 2.0 Flash (Director) -> {get_active_model(Modality.IMAGE).name} (Renderer).")
        else:
            logging.info(f"   🎨 {args.vpform}: Using Local Pipeline (Gemma Director -> Flux Renderer).")
        
    logging.info(f"   Model: {model}")
    
    # Scan (Only needed for legacy FBF/Transcript mode)
    if args.vpform in ["creative-agency", "music-visualizer", "music-agency", "music-video", "full-movie"] or args.xb:
        # Virtual Project
        try:
             args.model = model
             random.shuffle(keys)
             key_cycle = itertools.cycle(keys)
             
             # Create a valid Dummy Path or use the XML name
             if args.xb:
                 p_name = Path(args.xb).stem
                 p_dir = Path(args.xb).parent / p_name
             else:
                 # USEFUL NAMING: Use Music Filename or Default
                 if args.mu:
                     clean_stem = Path(args.mu).stem.replace(" ", "_").replace(".", "_")
                     p_name = f"{clean_stem}_{int(time.time())}" # Add timestamp to avoid collisions? Or just overwrite? 
                     # User hates clutter. Let's just use Stem + Date-Time or just Stem if we want single folder.
                     # Let's use Stem_Agency
                     p_name = f"{clean_stem}_Video"
                 else:
                     p_name = f"Agency_Creative_{int(time.time())}"
                     
                 p_dir = output_root / p_name
                 ensure_dir(p_dir)
             
             # Mock a Path object that has useable join behavior
             # We want correct subdir behavior: proj / "analysis.json" -> output_dir / "analysis.json"
             # But process_project expects proj to be the PARENT of the project folder?
             # Actually, scan_projects returns the DIRECTORY of the project.
             # So MockPath should probably represent p_dir directly.
             
             # The issue is usually definitions.py expects specific structure.
             # Let's just pass p_dir (Path object) directly, but ensure it exists.
             # Wait, process_project usually iterates files in proj.
             # For Agency mode, there are no files yet.
             # Let's see how process_project uses it. 
             # It likely does: analysis_file = project / "analysis.json"
             
             # So we can just use p_dir!
             mock_proj = p_dir
             
             # We need to bypass the 'analysis.json' read in process_project if in this mode.
             # Wait, process_project reads analysis.json immediately.
             # We should inject the logic INSIDE process_project (done above) 
             # and ensure it doesn't crash on missing file if mode is set.
             # Let's verify process_project logic...
             # It starts: with open(analysis_file)... CRASH.
             # We must fix process_project start block.
             
             pass # Logic handled in next edit block to process_project
             
             # Call logic
             # We need to ensure process_project doesn't fail early.
             # Call logic
             # We need to ensure process_project doesn't fail early.
             process_project(mock_proj, args.vf, key_cycle, args, output_root, keys, text_engine)

        except Exception as e:
             logging.error(f"Agency Job Failed: {e} | Type: {type(e)}")
             import traceback
             traceback.print_exc()
             return
        return

    projects = scan_projects(args.tf)
    logging.info(f"   Found {len(projects)} projects.")
    
    if args.project:
        # Filter for specific project
        projects = [p for p in projects if p.name == args.project]
        if not projects:
            logging.error(f"Project '{args.project}' not found!")
            return
    
    # Shuffle
    if args.shuffle:
        logging.info("   🎲 Shuffling project list...")
        random.shuffle(projects)

    processed_count = 0
    
    for proj in projects:
        # Filter by Duration
        if args.smin > 0 or args.smax is not None:
            dur = get_project_duration(proj)
            if dur < args.smin:
                continue
            if args.smax is not None and dur > args.smax:
                continue
        
        try:
            # Pass model explicitly? Or set global/arg?
            # Let's pass args and handle logic inside process_project
            args.model = model 
            
            # Init Cycle
            # Ensure keys exist
            if not keys:
                keys = load_keys(Path("env_vars.yaml"))
                if not keys: keys = [os.environ.get("GEMINI_API_KEY")]
            
            random.shuffle(keys)
            key_cycle = itertools.cycle(keys)
            
            process_project(proj, args.vf, key_cycle, args, output_root, keys, text_engine)
            processed_count += 1
        except Exception as e:
            logging.error(f"Failed to process {proj.name}: {e}")
            
    if processed_count == 0:
        logging.warning("No projects matched criteria (or none processed).")

if __name__ == "__main__":
    main()
