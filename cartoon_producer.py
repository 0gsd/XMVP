#!/usr/bin/env python3
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

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default Paths (Can be overridden by args)
DEFAULT_TF = Path("/Users/0gs/METMcloud/METMroot/tools/fmv/fbf_data")
DEFAULT_VF = Path("/Volumes/ORICO/fmv_corpus")

# Model Configuration
IMAGE_MODEL = "gemini-2.5-flash-image" # Default (Fast/Capable)
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

def generate_frame_gemini(index, prompt, output_dir, key_cycle, width=768, height=768, aspect_ratio="1:1", model=IMAGE_MODEL, prev_frame_path=None, pg_mode=False):
    """
    Worker function using Gemini Image API (Polyglot: Imagen vs Gemini) with Round-Robin Rotation.
    """
    target_path = output_dir / f"frame_{index:04d}.png"
    if target_path.exists():
        return True # Skip if already done

    # Retry Loop
    # Retry Loop
    max_retries = 5 # User Requested Boost
    for attempt in range(max_retries):
        # ROTATION: Get next key from cycle
        current_key = next(key_cycle)
        logging.info(f"🎨 Rendering Frame {index} (Attempt {attempt+1}/{max_retries}) [Key Rotation]...")
        
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
                        # Force "Generate an image of" prefix AND explicit silence instruction
                        # Force "Generate an image of" prefix AND explicit silence instruction
                        # SIMPLIFICATION: Imagen 3 prefers direct captions. Removing meta-tags.
                        contents_payload = [f"Image of {refined_prompt} . {ANI_INSTRUCTION}"]
                        model_to_use = render_model
                    else:
                        logging.warning("   ⚠️ Director failed to return text. Using raw prompt.")
                        contents_payload = [ar_prompt]
                        model_to_use = "gemini-2.5-flash-image"

                else:
                    # Standard (FBF / Other)
                    # For FBF, we don't do context injection because 2.5 Image doesn't support it well (returns text).
                    contents_payload = [ar_prompt]
                    model_to_use = model

                # 3. Generate Image
                response = client.models.generate_content(
                    model=model_to_use,
                    contents=contents_payload
                )
                
                # Check for inline data
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            # Save bytes
                            with open(target_path, "wb") as f:
                                f.write(part.inline_data.data)
                            return True
                        elif part.text:
                            logging.warning(f"Frame {index}: Model returned TEXT instead of IMAGE: {part.text[:200]}...")
            
            logging.warning(f"Frame {index}: No image data returned. Retrying...")
            
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

def process_project(project_dir, vf_dir, key_cycle, args, output_root, keys):
    """
    Runs the pipeline for a single project.
    """
    project_name = project_dir.name
    logging.info(f"🚀 Processing Project: {project_name}")
    
    analysis_file = project_dir / "analysis.json"
    metadata_file = project_dir / "metadata.json"
    
    descriptions = []
    
    # Bypass for Creative Agency / XB
    # Bypass for Creative Agency / XB / Music Visualizer / Music Agency
    if args.vpform in ["creative-agency", "music-visualizer", "music-agency"] or args.xb:
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

    elif args.vpform == "music-agency":
        # === MUSIC AGENCY MODE ===
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
        else:
             bpm, duration, fpb, fps = analyze_audio(args.mu)
        target_frames = int(duration * fps)
        project_fps = fps
        target_duration = duration
        
        logging.info(f"   🎹 Music Agency Target: {target_frames} frames @ {fps:.2f} FPS ({duration:.1f}s) | BPM: {bpm}")
        
        # 2. Generate Story (Content)
        logging.info("   🧠 Agency Brain: Dreaming up a narrative for this track...")
        text_engine = TextEngine()
        
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
        
        story_req = (
            f"Create a VISUAL SCREENPLAY for a {target_duration}s music video (Animated).\n"
            f"Concept: {prompt_concept}\n"
            f"Chaos Seeds to weave in: {seeds}\n"
            f"Music Vibe: {bpm} BPM.\n"
            f"Constraints: We need approx {est_beats} distinct visual scenes/beats to span the song.\n"
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
            text_engine = TextEngine()
            
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

    for i, p in enumerate(prompts):
        index = i + 1
        
        # Pass model from args
        model_to_use = getattr(args, 'model', IMAGE_MODEL)

        # Simple Aspect Ratio logic? Assuming Square for now as per ppfad defaults (768x768)
        # Gemini often prefers '1:1', '3:4', '4:3', '16:9'
        
        # Define Previous Frame Path
        prev_frame = None
        if i > 0:
             prev_frame = frames_dir / f"frame_{i:04d}.png"
             
        if generate_frame_gemini(index, p, frames_dir, key_cycle, aspect_ratio="1:1", model=model_to_use, prev_frame_path=prev_frame, pg_mode=args.pg):
             print(".", end="", flush=True)
             success_count += 1
        else:
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
        audio_source = original_video
        if args.mu and os.path.exists(args.mu):
             audio_source = Path(args.mu)
             logging.info(f"   🎵 Using provided music track: {audio_source}")
        
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
        meta_key = get_random_key(keys)
        meta_client = genai.Client(api_key=meta_key)
        
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
    parser.add_argument("--vpform", type=str, default="creative-agency", help="VP Form (default: creative-agency)") # New Default!
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
    
    args = parser.parse_args()
    
    # Setup Paths
    # Point to CENTRAL env_vars.yaml in tools/fmv/
    env_file = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
    if not env_file.exists():
         logging.warning(f"Central env_vars.yaml not found at {env_file}. Attempting fallback...")
         env_file = Path(__file__).resolve().parent / "env_vars.yaml"
    
    # Using mvp/v0.5 specific output dir
    base_dir = Path(__file__).resolve().parent # v0.5
    output_root = base_dir / "z_test-outputs"
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
    
    # Model Selection
    model = IMAGE_MODEL # Default Imagen 4
    if args.vpform == "fbf-cartoon":
        model = "gemini-2.5-flash-image"
        logging.info("   ⚡ FBF Mode: Using Gemini 2.5 Flash Image (L-Tier) for speed/quota.")
    elif args.vpform in ["creative-agency", "music-visualizer", "music-agency"]:
        # Changed to Gemini 2.0 Flash for Multimodal Context (Previous Frame Awareness)
        # generated_frame_gemini will detect this model and use it as a "Director" for the 2.5 Image model.
        model = "gemini-2.0-flash" 
        logging.info(f"   🎨 {args.vpform}: Using Gemini 2.0 Flash (Director) -> 2.5 Flash Image (Renderer).")
        
    logging.info(f"   Model: {model}")
    
    # Scan (Only needed for legacy FBF/Transcript mode)
    if args.vpform in ["creative-agency", "music-visualizer", "music-agency"] or args.xb:
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
                 p_name = f"agency_job_{int(time.time())}"
                 p_dir = output_root / p_name
                 ensure_dir(p_dir)
             
             # Mock a Path object that has .name for the process function
             # Actually, we should just refactor process_project to take a name, but
             # we will monkey-patch or pass a Mock object to minimize drift.
             class MockPath:
                 def __init__(self, name): self.name = name
                 def __truediv__(self, other): return Path(f"/tmp/{self.name}") / other
                 
             mock_proj = MockPath(p_name)
             
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
             process_project(mock_proj, args.vf, key_cycle, args, output_root, keys)

        except Exception as e:
             logging.error(f"Agency Job Failed: {e}")
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
            random.shuffle(keys)
            key_cycle = itertools.cycle(keys)
            
            process_project(proj, args.vf, key_cycle, args, output_root, keys)
            processed_count += 1
        except Exception as e:
            logging.error(f"Failed to process {proj.name}: {e}")
            
    if processed_count == 0:
        logging.warning("No projects matched criteria (or none processed).")

if __name__ == "__main__":
    main()
