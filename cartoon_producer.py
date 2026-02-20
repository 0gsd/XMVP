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
import torch
from PIL import Image, ImageDraw, ImageFont
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
from mvp_shared import VPForm, CSSV, Constraints, Story, Portion, save_xmvp, load_xmvp, load_manifest, load_nicotime_context
from vision_producer import get_chaos_seed
import frame_canvas # Support FC Mode
try:
    from flux_bridge import get_flux_bridge
except ImportError:
    pass # Handle locally inside function if missing
try:
    from wan_bridge import get_wan_bridge
except ImportError:
    pass
import definitions
from definitions import Modality, BackendType, get_active_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plan_shots(total_frames, fps, bpm=None):
    """
    Generates a list of shots (StartFrame, EndFrame) covering the total duration.
    - Average Duration: 4.0s
    - Range: 1.0s - 6.0s
    - Beat Sync: If BPM provided, snaps cut points to nearest beat.
    """
    shots = []
    current_frame = 0
    
    # Calculate frames per beat
    frames_per_beat = None
    if bpm and bpm > 0:
        frames_per_beat = fps * (60.0 / bpm)
        logging.info(f"   🥁 Beat Sync Active: {frames_per_beat:.1f} frames/beat")
        
    while current_frame < total_frames:
        # 1. Propose generic duration (1s-6s, avg 4s)
        # Weight towards 4s
        base_sec = random.triangular(1.0, 6.0, 4.0)
        dur_frames = int(base_sec * fps)
        
        target_end = current_frame + dur_frames
        
        # 2. Beat Snap (The "Loose Coupling")
        if frames_per_beat:
            # Find nearest beat boundary to target_end
            raw_beat = target_end / frames_per_beat
            nearest_beat = round(raw_beat)
            # Don't snap if it makes clip too short (<1s)
            valid_snap = int(nearest_beat * frames_per_beat)
            if valid_snap > current_frame + fps:
                target_end = valid_snap
                
        # 3. Cap at total
        if target_end >= total_frames:
            target_end = total_frames
            dur_frames = target_end - current_frame
            shots.append((current_frame, target_end))
            break
            
        # 4. Enforce Wan Constraints (Length % 4 == 1)
        # Wan FLF generates N frames between A and B (inclusive).
        # We need the SEGMENT length (End - Start + 1) to be 4n+1?
        # Actually Wan logic: num_frames = (end - start) + 1?
        # Wait, if we have Frame 0 and Frame 24.
        # We generate 0..24. Total 25 frames.
        # 25 = 4*6 + 1. Valid.
        # So (End - Start) must be divisible by 4.
        
        seg_len = target_end - current_frame
        remainder = seg_len % 4
        if remainder != 0:
            target_end -= remainder # Round down to nearest mod 4
            
        if target_end <= current_frame:
             # If rounding killed it, force forward
             target_end = current_frame + 4
        
        shots.append((current_frame, target_end))
        current_frame = target_end
        
    return shots

def run_wan_keyframe_anim(args, prompts, project_fps, out_root, duration, bpm=None):
    """
    Orchestrates the Wan 2.1 Keyframe Animation workflow using Variable Shot Lengths.
     1. Plan Shots (Variable duration, beat-synced)
     2. Generate Keyframes (Flux) for Shot Boundaries.
     3. Generate Intervals (Wan FLF).
     4. Stitch.
    """
    kf_dir = out_root / "keyframes"
    ensure_dir(kf_dir)
    vid_dir = out_root / "segments"
    ensure_dir(vid_dir)
    
    # 0. Plan Shots
    total_frames = int(duration * project_fps)
    shot_list = plan_shots(total_frames, project_fps, bpm)
    
    logging.info(f"\\n=== PHASE 0: SHOT PLANNING ({len(shot_list)} Shots) ===")
    logging.info(f"   Avg Duration: {duration/len(shot_list):.2f}s")
    
    # 1. Generate Keyframes (Flux)
    logging.info("\\n=== PHASE 1: GENERATING KEYFRAMES (FLUX) ===")
    
    # Collect unique boundary frames needed (Start of Shot 0, End of Shot 0/Start of Shot 1...)
    # Actually, distinct shots might NOT share frames if we want "Cuts".
    # But Wan FLF needs First and Last.
    # If Shot A is 0-96. Shot B is 96-144.
    # Frame 96 is the end of A and start of B.
    # So yes, they share keyframes for continuity.
    # UNLESS we want hard cuts?
    # User said: "not like a crazy collage of morphing... make it feel like a real movie composed of clips"
    # If we share frames, Wan will morph A->B.
    # If we want a CUT, Shot A (0-96) and Shot B (97-150) should be distinct.
    # But Wan FLF generates the video *between* the frames.
    # If we want a continuous video file, we need continuity?
    # Actually, real movies have CUTS.
    # So Shot A ends. CUT. Shot B starts.
    # Frame 96 (End of A) and Frame 97 (Start of B) can be totally different images!
    # They don't need to morph.
    # So we should generate:
    # Keyframe A_Start, Keyframe A_End -> Output Video A
    # Keyframe B_Start, Keyframe B_End -> Output Video B
    # Concatenate.
    # This avoids the "Morphing World" problem entirely!
    # Frame A_End and B_Start will be sequential in the final video but visual discontinuities (Cuts).
    
    # So... we need 2 keyframes per shot.
    # Shot 1: KF_0_Start, KF_0_End.
    # Shot 2: KF_1_Start, KF_1_End.
    
    # Load Flux Bridge
    try:
        from flux_bridge import get_flux_bridge
        flux_bridge = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")
        if not flux_bridge: raise ImportError("No Flux Bridge")
    except Exception as e:
        logging.error(f"Failed to load Flux: {e}")
        return

    shot_assets = [] # Stores (output_path, num_frames)
    
    for i, (start_f, end_f) in enumerate(shot_list):
        logging.info(f"\\n   🎬 Shot {i}: Frames {start_f}-{end_f} ({(end_f-start_f)/project_fps:.1f}s)")
        
        # Define Keyframe Paths
        kf_start_path = kf_dir / f"shot_{i:03d}_start_f{start_f}.png"
        kf_end_path = kf_dir / f"shot_{i:03d}_end_f{end_f}.png"
        
        # Get Prompts
        # We use the prompt at the START time for the whole shot?
        # Or Start prompt and End prompt?
        # If prompts change 1/sec, and shot is 4s.
        # Start Prompt = P[0]. End Prompt = P[4].
        # Wan interpolates P[0]->P[4].
        
        p_idx_start = min(start_f, len(prompts)-1)
        p_idx_end = min(end_f, len(prompts)-1)
        
        prompt_start = prompts[p_idx_start]
        prompt_end = prompts[p_idx_end]
        
        # Generate Start Keyframe (If not exists)
        if not kf_start_path.exists():
            clean_s = re.sub(r"\(Frame \d+/\d+\)", "", prompt_start).strip()
            scold_s = f"{clean_s} {ANI_INSTRUCTION} --no text --no letters --no watermarks"
            logging.info(f"      🖼️  KF Start: {clean_s[:30]}...")
            img = flux_bridge.generate(prompt=scold_s, width=args.w, height=args.h, steps=12)
            if img: img.save(kf_start_path)
            
        # Generate End Keyframe
        if not kf_end_path.exists():
            clean_e = re.sub(r"\(Frame \d+/\d+\)", "", prompt_end).strip()
            scold_e = f"{clean_e} {ANI_INSTRUCTION} --no text --no letters --no watermarks"
            # Optimization: If Start/End prompts are identical (static shot), maybe Flux varies it slightly?
            # Yes, Flux seed is random. So it will look like "Time passing".
            logging.info(f"      🖼️  KF End:   {clean_e[:30]}...")
            img = flux_bridge.generate(prompt=scold_e, width=args.w, height=args.h, steps=12)
            if img: img.save(kf_end_path)
            
        shot_assets.append({
            "id": i,
            "start_img": kf_start_path,
            "end_img": kf_end_path,
            "prompt": prompt_start, # Use start prompt for Wan conditioning
            "num_frames": end_f - start_f + 1,
            "out_video": vid_dir / f"shot_{i:03d}.mp4"
        })

    del flux_bridge
    import gc; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    
    # 2. Generate Intervals (Wan FLF)
    logging.info("\\n=== PHASE 2: GENERATING SHOTS (WAN FLF) ===")
    
    try:
        from wan_bridge import get_wan_bridge
        wan_bridge = get_wan_bridge(task_type="flf")
        wan_bridge.load_model()
    except Exception as e:
        logging.error(f"Failed to load Wan: {e}")
        return
        
    final_segments = []
    
    for shot in shot_assets:
        out_path = shot["out_video"]
        final_segments.append(out_path)
        
        if out_path.exists(): continue
        
        logging.info(f"   📹 Rendering Shot {shot['id']} ({shot['num_frames']} frames)...")
        
        clean_p = re.sub(r"\(Frame \d+/\d+\)", "", shot["prompt"]).strip()
        
        # Enforce 4n+1 Constraint again just in case
        valid_n = shot["num_frames"]
        if (valid_n - 1) % 4 != 0:
             # Should be handled by planner, but safety net
             valid_n = round((valid_n - 1) / 4) * 4 + 1
             
        success = wan_bridge.generate_flf(
            prompt=clean_p,
            first_frame_path=str(shot["start_img"]),
            last_frame_path=str(shot["end_img"]),
            output_path=str(out_path),
            width=args.w,
            height=args.h,
            num_frames=valid_n
        )
        if not success:
            logging.error(f"Failed Shot {shot['id']}")

    wan_bridge.unload()

    # 3. Stitch
    logging.info("\\n=== PHASE 3: STITCHING ===")
    list_path = out_root / "shot_list.txt"
    final_video = out_root / "final_feature.mp4"
    
    with open(list_path, 'w') as f:
        for seg in final_segments:
            f.write(f"file '{seg}'\\n")
            # No inpoint needed if we are doing hard cuts.
            # We want the FULL clip.
            
    cmd_stitch = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy", str(final_video)
    ]
    subprocess.run(cmd_stitch, check=False)
    
    if args.mu and os.path.exists(args.mu) and final_video.exists():
        logging.info("   🎵 Adding Audio...")
        final_with_audio = out_root / "final_with_audio.mp4"
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(final_video),
            "-i", str(args.mu),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-shortest",
            str(final_with_audio)
        ]
        subprocess.run(cmd_audio, check=False)
        logging.info(f"✅ COMPLETE: {final_with_audio}")
    elif final_video.exists():
        logging.info(f"✅ COMPLETE: {final_video}")

# Default Paths (Can be overridden by args)
DEFAULT_TF = Path("/Users/0gs/METMcloud/METMroot/tools/fmv/fbf_data")
DEFAULT_VF = Path("/Volumes/XMVPX/fmv_corpus")

# Model Configuration
# Model Configuration
# IMAGE_MODEL = "imagen-3.0-generate-001" # Default (Fast/Capable) - DEPRECATED for Registry
FLIPBOOK_STYLE_PROMPT = "You are a commercial animator. DRAW ONLY VISUALS. NO TEXT. NO NUMBERS. NO METADATA OVERLAYS. VISUAL:"

ANI_INSTRUCTION = """
CRITICAL VISUAL INSTRUCTION:
- This is a sequential frame in a HAND-DRAWN animation.
- STRICTLY NO TEXT, NO TIMECODES, NO "FRAME XX/YY" OVERLAYS.
- DO NOT INCLUDE ANY WORDS, NUMBERS, LOGOS, or WATERMARKS.
- The image must be pure visual content.
- Match the style of the previous frame (if provided) EXACTLY.
- Motion should be incremental and fluid.
- IF YOU ARE TEMPTED TO WRITE TEXT, DRAW A cloud INSTEAD.
"""

# -----------------------------------------------------------------------------
# ANSI VIDEO CONFIGURATION
# -----------------------------------------------------------------------------

# Character palette ordered by brightness (0.0 -> 1.0)
# Includes half-blocks for more detailed "subpixels"
ANSI_CHARS = " ░▒▓█▀▄▌▐"
ANSI_CHAR_BRIGHTNESS = {
    ' ': 0.0,
    '░': 0.25,
    '▒': 0.50,
    '▓': 0.75,
    '█': 1.0,
    '▀': 0.5,  # Top half
    '▄': 0.5,  # Bottom half
    '▌': 0.5,  # Left half  
    '▐': 0.5,  # Right half
}

# Color palettes (gradient from dark to bright)
ANSI_PALETTES = {
    "amber": [(30, 20, 10), (80, 50, 20), (180, 120, 60), (255, 200, 150)],
    "cyber": [(0, 20, 15), (0, 60, 50), (0, 140, 100), (0, 255, 180)],
    "vaporwave": [(40, 20, 50), (100, 40, 120), (180, 80, 180), (255, 140, 220)],
    "fire": [(30, 10, 5), (120, 40, 10), (220, 100, 20), (255, 180, 80)],
    "ocean": [(10, 20, 40), (30, 60, 100), (60, 120, 180), (120, 200, 255)],
    "matrix": [(0, 15, 0), (0, 50, 0), (0, 120, 0), (50, 255, 50)],
    "blood": [(20, 5, 5), (80, 20, 20), (160, 40, 40), (255, 80, 80)],
    "ice": [(20, 30, 40), (60, 100, 140), (120, 180, 220), (200, 240, 255)],
    "sunset": [(40, 20, 30), (140, 60, 50), (220, 120, 80), (255, 200, 150)],
    "monochrome": [(20, 20, 20), (80, 80, 80), (160, 160, 160), (240, 240, 240)],
}

# Prompt keywords that trigger specific palettes
ANSI_PALETTE_KEYWORDS = {
    "fire": "fire", "flame": "fire", "burn": "fire", "lava": "fire", "hell": "fire",
    "ocean": "ocean", "sea": "ocean", "water": "ocean", "underwater": "ocean", "aqua": "ocean",
    "cyber": "cyber", "neon": "cyber", "hacker": "matrix", "matrix": "matrix", "digital": "matrix",
    "blood": "blood", "horror": "blood", "dark": "blood", "evil": "blood", "death": "blood",
    "ice": "ice", "cold": "ice", "frozen": "ice", "winter": "ice", "snow": "ice",
    "sunset": "sunset", "dawn": "sunset", "dusk": "sunset", "warm": "sunset",
    "vapor": "vaporwave", "retro": "vaporwave", "80s": "vaporwave", "synthwave": "vaporwave",
    "space": "cyber", "robot": "cyber", "machine": "cyber", "transform": "cyber",
    "cartoon": "amber", "saturday": "amber", "morning": "amber", "1992": "amber", "1990": "amber",
}

def detect_ansi_palette(prompt: str, style: str = None) -> str:
    """Detects the best palette based on prompt and style keywords."""
    search_text = (prompt + " " + (style or "")).lower()
    
    for keyword, palette_name in ANSI_PALETTE_KEYWORDS.items():
        if keyword in search_text:
            return palette_name
    
    return "amber"  # Default warm CRT look

def get_palette_color(brightness: float, palette_name: str) -> tuple:
    """Interpolate a color from the palette based on brightness (0.0-1.0)."""
    palette = ANSI_PALETTES.get(palette_name, ANSI_PALETTES["amber"])
    
    brightness = max(0.0, min(1.0, brightness))
    n = len(palette)
    idx = brightness * (n - 1)
    i = int(idx)
    frac = idx - i
    
    if i >= n - 1:
        return palette[-1]
    
    c1, c2 = palette[i], palette[i + 1]
    return tuple(int(c1[j] + frac * (c2[j] - c1[j])) for j in range(3))

def generate_ansi_frame(index: int, prompt: str, output_dir: Path, 
                        canvas_width: int, canvas_height: int,
                        text_engine, prev_frame_text: str = None,
                        palette_name: str = "amber") -> tuple:
    """
    Uses the LLM to 'draw' a frame using ANSI block characters.
    Returns (frame_path, frame_text) for feedback to next frame.
    """
    target_path = output_dir / f"frame_{index:04d}.png"
    target_txt = output_dir / f"frame_{index:04d}.txt"
    
    if target_path.exists() and target_txt.exists():
        # Skip if already done, but return text for context
        return target_path, target_txt.read_text()
    
    # Build the drawing instruction
    char_list = "█ (solid/bright), ▓ (dark shade), ▒ (medium shade), ░ (light shade), [space] (empty/black), ▀ ▄ ▌ ▐ (half-blocks)"
    
    base_instruction = f"""You are an ANIMATOR drawing frame-by-frame pixel art for a CARTOON.

🎬 THIS IS ANIMATION - each frame shows characters, objects, and scenes in motion!

CANVAS: {canvas_width} columns × {canvas_height} rows
CHARACTERS TO USE: {char_list}

🎨 SCENE TO DRAW:
{prompt}

⚠️ CRITICAL RULES:
1. DRAW ACTUAL OBJECTS - characters, buildings, landscapes, vehicles, creatures
2. NEVER fill the entire canvas with one character - that creates a blank frame!
3. Use HIGH CONTRAST: █ for bright objects, [space] for dark background
4. Create SILHOUETTES and SHAPES that are visually distinct
5. Objects should have EDGES - use ▓▒░ for shading and transitions
6. Use half-blocks (▀▄▌▐) for smooth curves and diagonal lines

📐 OUTPUT FORMAT:
- EXACTLY {canvas_height} lines
- Each line has EXACTLY {canvas_width} characters
- NO text, commentary, or code blocks - just the raw character grid

💡 DRAWING TIPS:
- Dark background (spaces) + bright subject (█▓) = visible art
- A character should occupy 10-30% of the canvas, not 100%
- Include ground/horizon lines to anchor the scene
- Motion = slight position shifts from previous frame"""

    if prev_frame_text:
        base_instruction += f"""

🔄 PREVIOUS FRAME (animate from this - small incremental motion only):
```
{prev_frame_text}
```

Move objects slightly, don't redraw everything! This is animation, not random frames."""

    base_instruction += "\n\n🎬 Draw the frame NOW (raw characters only):"

    # Generate with LLM
    max_retries = 3
    frame_text = None
    
    for attempt in range(max_retries):
        try:
            raw_output = text_engine.generate(base_instruction, temperature=0.7)
            
            # Parse output - extract the character grid
            # Look for lines that contain our block characters
            valid_block_chars = set('█▓▒░▀▄▌▐ ')
            lines = []
            
            for line in raw_output.strip().split('\n'):
                # Skip markdown fences and common preamble
                stripped = line.strip()
                if stripped.startswith('```') or stripped.startswith('Here') or stripped.startswith('Frame'):
                    continue
                if len(stripped) == 0:
                    continue
                    
                # Count block chars vs other chars
                block_count = sum(1 for c in line if c in valid_block_chars)
                other_count = len(line) - block_count
                
                # Accept line if it's mostly block chars (>50%) or has significant block content
                if block_count > 0 and (block_count >= other_count or block_count >= canvas_width * 0.3):
                    lines.append(line)
                # Also accept lines that are mostly spaces with some block chars
                elif len(line) >= canvas_width * 0.5 and block_count >= 5:
                    lines.append(line)
            
            # Validate and normalize dimensions
            if len(lines) < canvas_height * 0.5:
                logging.warning(f"   Frame {index} Attempt {attempt+1}: Got {len(lines)} lines, expected ~{canvas_height}")
                continue
            
            # Normalize to exact dimensions    
            normalized = []
            for i, line in enumerate(lines[:canvas_height]):
                # Pad or truncate to exact width
                if len(line) < canvas_width:
                    line = line + ' ' * (canvas_width - len(line))
                elif len(line) > canvas_width:
                    line = line[:canvas_width]
                normalized.append(line)
            
            # Pad missing rows
            while len(normalized) < canvas_height:
                normalized.append(' ' * canvas_width)
            
            frame_text = '\n'.join(normalized)
            break
            
        except Exception as e:
            logging.warning(f"   Frame {index} Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    if not frame_text:
        logging.error(f"   Frame {index}: All attempts failed. Creating blank frame.")
        frame_text = '\n'.join([' ' * canvas_width] * canvas_height)
    
    # Save raw text
    target_txt.write_text(frame_text)
    
    # Render to PNG
    render_ansi_to_png(frame_text, target_path, palette_name, canvas_width, canvas_height)
    
    return target_path, frame_text

def render_ansi_to_png(frame_text: str, output_path: Path, palette_name: str,
                       canvas_width: int, canvas_height: int,
                       pixel_size: int = 8) -> bool:
    """
    Converts ANSI character art to a PNG image.
    Each character becomes a pixel_size × pixel_size block.
    """
    try:
        # Calculate image dimensions
        img_width = canvas_width * pixel_size
        img_height = canvas_height * pixel_size
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), (0, 0, 0))
        pixels = img.load()
        
        lines = frame_text.split('\n')
        
        for row, line in enumerate(lines):
            if row >= canvas_height:
                break
            for col, char in enumerate(line):
                if col >= canvas_width:
                    break
                
                # Get brightness for this character
                brightness = ANSI_CHAR_BRIGHTNESS.get(char, 0.0)
                
                # Get color from palette
                color = get_palette_color(brightness, palette_name)
                
                # Fill the pixel block
                for py in range(pixel_size):
                    for px in range(pixel_size):
                        img_x = col * pixel_size + px
                        img_y = row * pixel_size + py
                        
                        # Handle half-blocks specially
                        if char == '▀':  # Top half
                            if py < pixel_size // 2:
                                pixels[img_x, img_y] = color
                            else:
                                pixels[img_x, img_y] = get_palette_color(0.0, palette_name)
                        elif char == '▄':  # Bottom half
                            if py >= pixel_size // 2:
                                pixels[img_x, img_y] = color
                            else:
                                pixels[img_x, img_y] = get_palette_color(0.0, palette_name)
                        elif char == '▌':  # Left half
                            if px < pixel_size // 2:
                                pixels[img_x, img_y] = color
                            else:
                                pixels[img_x, img_y] = get_palette_color(0.0, palette_name)
                        elif char == '▐':  # Right half
                            if px >= pixel_size // 2:
                                pixels[img_x, img_y] = color
                            else:
                                pixels[img_x, img_y] = get_palette_color(0.0, palette_name)
                        else:
                            pixels[img_x, img_y] = color
        
        img.save(output_path)
        return True
        
    except Exception as e:
        logging.error(f"   Failed to render ANSI frame: {e}")
        return False


def redraw_frame_as_ansi(source_image_path: Path, index: int, output_dir: Path,
                          canvas_width: int, canvas_height: int,
                          text_engine, prev_frame_text: str = None,
                          palette_name: str = "amber", style_prompt: str = None) -> tuple:
    """
    Redraws a source image as ANSI block character art.
    
    1. Optionally describes the source image using vision
    2. LLM draws the scene using block characters
    3. Renders to PNG
    
    Returns: (png_path, frame_text)
    """
    target_path = output_dir / f"frame_{index:04d}.png"
    target_txt = output_dir / f"frame_{index:04d}.txt"
    
    # Load and analyze source image
    try:
        from PIL import Image
        src_img = Image.open(source_image_path).convert("RGB")
        src_w, src_h = src_img.size
        
        # Quick color analysis - sample dominant colors
        small = src_img.resize((16, 16), Image.BILINEAR)
        pixels = list(small.getdata())
        
        # Calculate average brightness zones (top, middle, bottom)
        thirds = len(pixels) // 3
        top_bright = sum(sum(p) for p in pixels[:thirds]) / (thirds * 3 * 255)
        mid_bright = sum(sum(p) for p in pixels[thirds:2*thirds]) / (thirds * 3 * 255)
        bot_bright = sum(sum(p) for p in pixels[2*thirds:]) / (thirds * 3 * 255)
        
        # Simple scene description based on brightness
        brightness_hint = f"Top: {int(top_bright*100)}%, Mid: {int(mid_bright*100)}%, Bottom: {int(bot_bright*100)}% brightness"
        
    except Exception as e:
        logging.warning(f"   Could not analyze source image: {e}")
        brightness_hint = "Unknown composition"
    
    # Build prompt for LLM to draw the frame
    base_instruction = f"""You are a pixel artist recreating a video frame using ONLY these block characters:
VALID CHARACTERS: █ ▓ ▒ ░ ▀ ▄ ▌ ▐ [space]

CANVAS SIZE: {canvas_width} columns × {canvas_height} rows

SOURCE IMAGE ANALYSIS:
- Resolution: {src_w}x{src_h} pixels
- Brightness distribution: {brightness_hint}
{f"- Style: {style_prompt}" if style_prompt else ""}

YOUR TASK:
1. Recreate the source frame's VISUAL STRUCTURE using blocks
2. Use █ for bright/foreground areas, ░ or space for dark/background
3. Use ▒▓ for midtones and transitions
4. Use half-blocks (▀▄▌▐) for fine edges and details
5. Maintain the composition: where bright areas are, where dark areas are

RULES:
1. Output EXACTLY {canvas_height} lines, each with EXACTLY {canvas_width} characters
2. Use ONLY the valid block characters - NO letters, numbers, or other symbols
3. NO commentary, just the character grid"""

    if prev_frame_text:
        base_instruction += f"""

PREVIOUS FRAME (maintain visual continuity):
```
{prev_frame_text}
```

Draw the NEXT frame with small incremental changes only."""

    base_instruction += "\n\nDraw the frame now (raw characters only, no markdown):"

    # Generate with LLM
    max_retries = 3
    frame_text = None
    valid_block_chars = set('█▓▒░▀▄▌▐ ')
    
    for attempt in range(max_retries):
        try:
            raw_output = text_engine.generate(base_instruction, temperature=0.7)
            
            # Parse output
            lines = []
            for line in raw_output.strip().split('\n'):
                stripped = line.strip()
                if stripped.startswith('```') or stripped.startswith('Here') or stripped.startswith('Frame'):
                    continue
                if len(stripped) == 0:
                    continue
                
                block_count = sum(1 for c in line if c in valid_block_chars)
                other_count = len(line) - block_count
                
                if block_count > 0 and (block_count >= other_count or block_count >= canvas_width * 0.3):
                    lines.append(line)
                elif len(line) >= canvas_width * 0.5 and block_count >= 5:
                    lines.append(line)
            
            if len(lines) < canvas_height * 0.5:
                logging.warning(f"   Frame {index} Attempt {attempt+1}: Got {len(lines)} lines, expected ~{canvas_height}")
                continue
            
            # Normalize dimensions
            normalized = []
            for line in lines[:canvas_height]:
                if len(line) < canvas_width:
                    line = line + ' ' * (canvas_width - len(line))
                elif len(line) > canvas_width:
                    line = line[:canvas_width]
                normalized.append(line)
            
            while len(normalized) < canvas_height:
                normalized.append(' ' * canvas_width)
            
            frame_text = '\n'.join(normalized)
            break
            
        except Exception as e:
            logging.warning(f"   Frame {index} Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    if not frame_text:
        logging.error(f"   Frame {index}: All attempts failed. Creating blank frame.")
        frame_text = '\n'.join([' ' * canvas_width] * canvas_height)
    
    # Save raw text
    target_txt.write_text(frame_text)
    
    # Render to PNG
    render_ansi_to_png(frame_text, target_path, palette_name, canvas_width, canvas_height)
    
    return target_path, frame_text

def load_keys(env_path):
    """Loads keys from env_vars.yaml."""
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                secrets = yaml.safe_load(f)
                
                # 1. Populate os.environ with ALL secrets (e.g. HF_TOKEN, FLUX_KLEIN_ENDPOINT)
                if secrets:
                    for k, v in secrets.items():
                        if k not in os.environ and isinstance(v, (str, int, float)):
                            os.environ[k] = str(v)
                            # logging.info(f"   🔑 Loaded Env Var: {k}")

                # 2. Extract Rotation Keys
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

def analyze_audio(audio_path, fsync=1.0):
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
        # But scale by fsync!
        frames_per_beat = 4 
        bps = bpm / 60.0
        fps = (bps * frames_per_beat) * fsync
        
        logging.info(f"   🎵 Audio Analysis: {bpm:.1f} BPM | {duration:.1f}s")
        logging.info(f"   🎵 Derived FPS: {fps:.2f} (based on {frames_per_beat} frames/beat * fsync {fsync})")
        
        return bpm, duration, frames_per_beat, fps
        
    except Exception as e:
        logging.error(f"   ❌ Audio Analysis Failed: {e}")
        return 120.0, 60.0, 4, 8.0 * fsync # Fallback

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

def run_wan_keyframe_anim(args, prompts, project_fps, out_root, duration):
    """
    Orchestrates the Wan 2.1 Keyframe Animation workflow:
    1. Generate Keyframes (Flux)
    2. Generate Intervals (Wan FLF)
    3. Stitch
    """
    kf_dir = out_root / "keyframes"
    ensure_dir(kf_dir)
    vid_dir = out_root / "segments"
    ensure_dir(vid_dir)
    
    # 1. Generate Keyframes (Flux)
    logging.info("\n=== PHASE 1: GENERATING KEYFRAMES (FLUX) ===")
    
    # Keyframe Strategy:
    # If project_fps = 8, then we need 1 keyframe per second?
    # Or 1 keyframe every N frames?
    # User said: "generate frame 1... generate frame 8... Wan 1-8".
    # This implies Keyframes are at 1s intervals (if 8fps).
    # Prompts list usually matches frames or beats.
    # In 'music-video', 'prompts' logic depends on source.
    # If source_content (XML) -> one per beat/description.
    # If auto-generated -> one per frame (music-visualizer) or one per beat.
    
    # We need to sub-sample prompts if they are dense.
    # Let's assume 'prompts' corresponds to the target timeline.
    # If 'prompts' has N items, we treat them as keyframes?
    # No, usually prompts list in 'music-video' logic is populated PER FRAME if visualizer, or PER BEAT.
    
    # Let's standardize: We generate a Keyframe every X seconds (e.g. 1s).
    # Interval = int(project_fps) frames.
    
    interval = int(project_fps) if project_fps >= 1 else 24
    num_keyframes = math.ceil(len(prompts) / interval) if len(prompts) > interval else len(prompts)
    # Actually if prompts are descriptions (beats), we might have fewer prompts than frames.
    
    # Re-evaluate 'prompts' variable from process_project.
    # In 'music-video', prompts were not fully populated in the snippets I saw? 
    # Let's look at process_project logic (Lines 800+).
    # ... It iterates target_frames.
    
    # Simplification: We will sample the provided prompts at Keyframe Indices.
    # If prompts list is short (beats), we cycle?
    # Let's assume prompts[i] corresponds to time t=i/fps?
    # No, prompts in music-video seems to be frame-by-frame text list (target_content = ...).
    
    # We will generate Keyframe images 0, Interval, 2*Interval...
    kf_paths = []
    
    # Load Flux Bridge
    try:
        from flux_bridge import get_flux_bridge
        flux_bridge = get_flux_bridge("/Volumes/XMVPX/mw/flux-root") # Default local path
        if not flux_bridge: raise ImportError("No Flux Bridge")
    except Exception as e:
        logging.error(f"Failed to load Flux for keyframes: {e}")
        return

    prev_kf = None
    
    total_frames = int(duration * project_fps)
    key_indices = range(0, total_frames, interval)
    
    for i, frame_idx in enumerate(key_indices):
        kf_path = kf_dir / f"kf_{i:04d}_f{frame_idx}.png"
        kf_paths.append(kf_path)
        
        if kf_path.exists():
            prev_kf = kf_path
            continue
            
        # Get Prompt
        # Map frame_idx to prompt list
        p_idx = min(frame_idx, len(prompts)-1)
        prompt = prompts[p_idx]
        
        # Strip "Frame X" meta
        clean_prompt = re.sub(r"\(Frame \d+/\d+\)", "", prompt).strip()
        
        logging.info(f"   🖼️  Generating Keyframe {i} (Frame {frame_idx}): {clean_prompt[:40]}...")
        
        # Generate with Flux
        # using 'prev_kf' as Img2Img input?
        # User said: "use Flux to generate frame 1 (using knowledge)... generate frame 8... using knowledge".
        # Might benefit from slight img2img or just pure txt2img with consistent seed/style.
        # Let's use Txt2Img for KF1, and maybe weak Img2Img for KF2 to keep consistency?
        # Actually user implied frame-to-frame logic in Wan. Flux frames should be distinct poses?
        # Let's allow Flux to be creative. Txt2Img (img=None).
        # OR using prev_kf with low strength?
        # Let's implement Txt2Img for now to ensure we hit the prompts.
        
        img = flux_bridge.generate(
            prompt=clean_prompt,
            width=args.w if args.w else 1280, 
            height=args.h if args.h else 720,
            image=None, # Pure Gen
            steps=8
        )
        if img:
            img.save(kf_path)
            prev_kf = kf_path
        else:
            logging.error("Flux failed.")
            return

    # Unload Flux
    del flux_bridge
    import gc; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    
    # 2. Generate Intervals (Wan FLF)
    logging.info("\n=== PHASE 2: GENERATING INTERVALS (WAN FLF) ===")
    
    try:
        from wan_bridge import get_wan_bridge
        wan_bridge = get_wan_bridge(task_type="flf")
        wan_bridge.load_model()
    except Exception as e:
        logging.error(f"Failed to load Wan: {e}")
        return
        
    segment_paths = []
    
    # Iterate pairs
    for i in range(len(kf_paths) - 1):
        start_img = kf_paths[i]
        end_img = kf_paths[i+1]
        seg_path = vid_dir / f"seg_{i:04d}.mp4"
        segment_paths.append(seg_path)
        
        if seg_path.exists(): continue
        
        # Calculate Number of Frames needed
        # Interval is e.g. 8 frames.
        # We have Frame 0 and Frame 8.
        # We need frames 0, 1, 2, 3, 4, 5, 6, 7, 8. (9 frames).
        # Wan FLF expects Start/End.
        # frame_num=9.
        # If interval=8, we need 8+1=9 frames.
        num_frames_needed = interval + 1
        
        # Ensure 4n+1 constraint?
        # 9 is 4(2)+1. Good.
        # 49 is 4(12)+1. Good.
        # If interval=24, n=25. 25 = 4(6)+1. Good.
        # If interval=12, n=13. 13 = 4(3)+1. Good.
        # Seems standard FPS values align well!
        
        # Prompt: Use the Start prompt? Or average?
        # Use Start Prompt.
        key_idx = key_indices[i]
        prompt = prompts[min(key_idx, len(prompts)-1)]
        clean_prompt = re.sub(r"\(Frame \d+/\d+\)", "", prompt).strip()
        
        success = wan_bridge.generate_flf(
            prompt=clean_prompt,
            first_frame_path=str(start_img),
            last_frame_path=str(end_img),
            output_path=str(seg_path),
            width=args.w if args.w else 1280,
            height=args.h if args.h else 720,
            num_frames=num_frames_needed
        )
        
        if not success:
            logging.error(f"Wan failed on segment {i}")
            # continue?
            
    # Unload Wan
    wan_bridge.unload()
    
    # 3. Stitch
    logging.info("\n=== PHASE 3: STITCHING ===")
    
    list_path = out_root / "segments.txt"
    final_video = out_root / "final_output.mp4"
    
    unique_segments = []
    # Logic: Segment 0 includes Frame 0..8.
    # Segment 1 includes Frame 8..16.
    # Frame 8 is duplicated!
    # We need to trim the FIRST frame of subsequent segments?
    # Or keep overlap for smoothness?
    # Strictly, Segment 0: 0,1..8. Segment 1: 8,9..16.
    # If we concat, we get 0..8, 8..16. Double frame 8.
    # We should trim.
    # ffmpeg concat filter can trim?
    # Or complex filter.
    # Easier: Just accept 1 duplicate frame per second (minor jitter) or fix.
    # Fix: use 'inpoint' in concat list? (Requires concat demuxer advanced syntax? No, duration/inpoint).
    # Concat Demuxer supports 'inpoint'.
    # file 'seg_0.mp4'
    # file 'seg_1.mp4'
    # inpoint 0.04 (approx 1 frame duration)
    
    # Actually, simpler: Wan FLF2V usually generates the End Frame as the Last Frame.
    # So duplicating it is real.
    # Let's generate the file list.
    
    with open(list_path, 'w') as f:
        for i, seg in enumerate(segment_paths):
            f.write(f"file '{seg}'\n")
            if i > 0:
                # Skip first frame of subsequent segments to avoid duplication
                # Frame duration at 24fps (Wan native output) is ~0.0416s
                # wait, Wan outputs 24fps file usually.
                # If we want 8fps output, we need to correct speed or just extract frames?
                # If we Stitch first, we get a 24fps video.
                # Then we can retiming it.
                # The 'inpoint' directive works.
                # Assuming 24fps output from Bridge:
                # 1 frame = 1/24 sec.
                f.write("inpoint 0.041666 \n") 
                
    # Stitch Command
    logging.info("   🧵 Stitching...")
    # ffmpeg concat demuxer
    # Note: 'inpoint' works for the PRECEDING file or following?
    # Syntax: 
    # file A
    # file B
    # inpoint X
    # Applies to B.
    
    cmd_stitch = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy", str(final_video)
    ]
    subprocess.run(cmd_stitch, check=False)
    
    # Add Audio if --mu
    if args.mu and os.path.exists(args.mu) and final_video.exists():
        logging.info("   🎵 Adding Audio...")
        # Audio Muxing
        final_with_audio = out_root / "final_with_audio.mp4"
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(final_video),
            "-i", str(args.mu),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-shortest",
            str(final_with_audio)
        ]
        subprocess.run(cmd_audio, check=False)
        logging.info(f"✅ COMPLETE: {final_with_audio}")
    elif final_video.exists():
        logging.info(f"✅ COMPLETE: {final_video}")


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
        # Updated to scale base_video (0) to match overlay_video (1)
        # Scale Source to Unicode Resolution.
        # [0:v][1:v]scale2ref[base][over] -> base is scaled 0, over is ref 1
        filter_complex = f"[0:v][1:v]scale2ref[base][over];[over]format=rgba,colorchannelmixer=aa={opacity}[over_alpha];[base][over_alpha]overlay=format=auto"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_video),
            "-i", str(overlay_video),
            "-filter_complex", filter_complex,
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

def generate_frame_universal(index, prompt, output_dir, key_cycle, width=768, height=768, aspect_ratio="1:1", model=None, prev_frame_path=None, pg_mode=False, force_local=False, strength=0.75):
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
        logging.error("   Please update active_models.json or definitions.py to point to a local model (e.g. flux-klein).")
        # Fail hard
        raise RuntimeError("Local execution enforced but model is cloud-based/unknown.")

    # --- NICOTIME INJECTION ---
    nico_context = load_nicotime_context(prompt)
    if nico_context:
        logging.info(f"   🧠 Nicotime Context Found: {nico_context[:60]}...")
        prompt += nico_context

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
                
                # --- DIRECTOR MODE (LOCAL) ---
                # Use Local Text Engine (Gemma) to "Direct" the shot before Rendering
                # This matches the Cloud "Two-Pass" logic (Describe -> Render)
                
                final_prompt = prompt
                img_input = None
                
                # 1. Restore Feedback (Img2Img) for Sequential Consistency
                if prev_frame_path and prev_frame_path.exists():
                    try:
                        from PIL import Image
                        img_input = Image.open(prev_frame_path).convert("RGB")
                        logging.info(f"   🔄 Feedback: Using {prev_frame_path.name} as Img2Img input.")
                    except Exception as e:
                        logging.warning(f"   ⚠️ Failed to load previous frame for feedback: {e}")

                # 2. Director Mode (Gemma)
                try:
                    from text_engine import get_engine
                    te = get_engine()
                    if te:
                        # Context-Aware Directing
                        ctx = "This is the FIRST frame."
                        if img_input:
                            ctx = "Previous frame exists. You MUST describe the NEXT sequential frame in a continuous animation."
                            
                        director_instruction = (
                            f"You are a Lead Animator. {ctx}\n"
                            f"Action: '{prompt}'.\n"
                            "Task: Output a dense, 20-30 word visual description for the renderer. "
                            "Focus on lighting, composition, and style. \n"
                            "CRITICAL: If the action is similar to previous, describe MINOR changes only (flipbook consistency). NO FILLER."
                        )
                        # Quick generation
                        visual_desc = te.generate(director_instruction, temperature=0.7)
                        
                        # CLEANUP: Handle JSON output if Director decides to be structured
                        if visual_desc and visual_desc.strip().startswith("{"):
                             try:
                                 import json
                                 # Try to find the first { and last }
                                 start = visual_desc.find("{")
                                 end = visual_desc.rfind("}")
                                 if start != -1 and end != -1:
                                     json_str = visual_desc[start:end+1]
                                     data = json.loads(json_str)
                                     # Extract best key
                                     for key in ["description", "Description", "action", "Action", "visual", "Visuals"]:
                                          if key in data:
                                              visual_desc = data[key]
                                              break
                             except:
                                 pass # Fallback to raw text
                        
                        if visual_desc and len(visual_desc) > 10:
                            logging.info(f"   🎬 Local Director: {visual_desc[:60]}...")
                            final_prompt = visual_desc
                except Exception as e_dir:
                    logging.warning(f"   ⚠️ Local Director failed: {e_dir}. Using raw prompt.")

                # Generate
                # Flux Klein typically benefits from more steps than Schnell (4). 8 is a good balance, 12 is better.
                # Standardize to requested dimensions
                # CRITICAL FIX: Ensure 'image' is passed to enable Img2Img Feedback Loop
                # If img_input is None (First Frame), it acts as Txt2Img.
                logging.info(f"   🚀 Flux Generating with Image Input: {img_input is not None}")
                
                # Use passed strength argument (default 0.75 from signature)
                img = bridge.generate(prompt=final_prompt, width=width, height=height, steps=12, image=img_input, strength=strength)
                
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
            # HF ENDPOINT PATH for Flux
            if model == "flux-2-klein-hf":
                try:
                    from flux_bridge import generate_via_hf_endpoint
                    logging.info(f"🎨 Rendering Frame {index} (Attempt {attempt+1}/{max_retries}) [Cloud Flux HF]...")
                    
                    # Load Feedback Image if available (Cloud Context)
                    img_input = None
                    if prev_frame_path and prev_frame_path.exists():
                        try:
                            from PIL import Image
                            img_input = Image.open(prev_frame_path).convert("RGB")
                            logging.info(f"   🔄 Cloud Feedback: Using {prev_frame_path.name} as Img2Img input.")
                        except Exception as e:
                            logging.warning(f"   ⚠️ Failed to load previous frame for cloud feedback: {e}")

                    img = generate_via_hf_endpoint(
                        prompt=prompt, 
                        width=width, 
                        height=height, 
                        steps=28, 
                        guidance=3.5,
                        seed=None, # Random
                        image=img_input,
                        strength=strength # Default 0.75 passed from arg
                    )
                    
                    if img:
                        img.save(target_path)
                        return True
                    else:
                         logging.warning("HF Endpoint returned None. Falling back to Imagen 3...")
                         # Fallback Logic: Switch model to Imagen 3 (Working)
                         model = "imagen-3.0-generate-001"
                         # FALL THROUGH
                         
                except ImportError:
                    logging.error("Could not import generate_via_hf_endpoint. Falling back to Imagen 3...")
                    model = "imagen-3.0-generate-001"
                except Exception as e_hf:
                     logging.error(f"HF Endpoint Failed: {e_hf}. Falling back to Imagen 3...")
                     model = "imagen-3.0-generate-001"
                     # FALL THROUGH
            
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
                            render_model = "imagen-3.0-generate-001"
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

    return False

def load_full_xmvp(path):
    """
    Parses an entire XMVP XML file and returns a dict with keys like 'Story', 'Portions', etc.
    Reconstructs Pydantic models where possible/necessary, or return dicts.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        data = {}
        
        for child in root:
            if child.text:
                try:
                    data[child.tag] = json.loads(child.text)
                except:
                    data[child.tag] = child.text
        return data
    except Exception as e:
        logging.error(f"Failed to parse XMVP XML: {e}")
        return None

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
    target_w = args.w if args.w else args.kid
    target_h = args.h if args.h else args.kid
    logging.info(f"   📐 Target Resolution: {target_w}x{target_h}")
    
    analysis_file = project_dir / "analysis.json"
    metadata_file = project_dir / "metadata.json"
    
    descriptions = []
    manifest_segments = [] # Store full segment objects if available
    
    # Bypass for Creative Agency / XB
    # Bypass for Creative Agency / XB / Music Visualizer / Music Agency
    # Bypass for Creative Agency / XB / Music Visualizer / Music Agency / ANSI Video
    if args.vpform in ["creative-agency", "music-visualizer", "music-video", "cartoon-video", "ansi-video", "ansi-redraw"] or args.xb:
        # Check if we have an explicit XML to load
        if args.xb and os.path.exists(args.xb):
             try:
                 logging.info(f"   📂 Loading XMVP from: {args.xb}")
                 xmvp_data = load_full_xmvp(args.xb)
                 # Extract 'Portions' as descriptions/beats (Legacy) OR 'Manifest' (New)
                 if xmvp_data:
                     # 1. Try Portions (Legacy)
                     if xmvp_data.get("Portions"):
                         # Handle list of dicts (if loaded from JSON)
                         raw_portions = xmvp_data["Portions"]
                         if raw_portions and isinstance(raw_portions[0], dict):
                            descriptions = [p.get('content', 'Placeholder') for p in raw_portions]
                         else:
                            descriptions = [] # fallback
                         logging.info(f"   ✅ Loaded {len(descriptions)} beats from XML Portions.")
                     
                     # 2. Try Manifest (New Standard)
                     elif xmvp_data.get("Manifest") and isinstance(xmvp_data["Manifest"], dict) and xmvp_data["Manifest"].get("segs"):
                         raw_segs = xmvp_data["Manifest"]["segs"]
                         descriptions = [s.get('prompt', 'Placeholder') for s in raw_segs]
                         # Capture full segments for timeline logic
                         manifest_segments = raw_segs
                         logging.info(f"   ✅ Loaded {len(descriptions)} beats from XML Manifest (with timing data).")

                     # RETCON PATH: If --retcon, we ignore these descriptions (or use them as seed?)
                     # For now, if --retcon, we treat them as 'Old Draft' and rewrite.
                     if args.retcon and len(descriptions) > 0 and descriptions[0] != "Placeholder":
                          logging.info("   🌀 Retcon Mode Active: Ignoring XML beats, rewriting based on Concept...")
                          descriptions = ["Placeholder (Retconning)"] 
                     
                     # STORY CONTEXT: Always check for Story to inform the prompt, especially for Retcon
                     if xmvp_data.get("Story") and not args.prompt:
                         s = xmvp_data["Story"]
                         # Handle if it's a dict (parsed JSON) or string
                         if isinstance(s, dict):
                             title = s.get("title", "Untitled")
                             synopsis = s.get("synopsis", "")
                             if synopsis:
                                 args.prompt = f"{title}: {synopsis}"
                                 logging.info(f"   🧠 Retcon Context (Story): {args.prompt}")
                         elif isinstance(s, str):
                             # Maybe it didn't parse as JSON?
                             args.prompt = s[:200]
                             logging.info(f"   🧠 Retcon Context (Raw Story): {args.prompt}")

                 if not descriptions:
                      descriptions = ["Placeholder"]
             except Exception as e:
                 logging.error(f"   ❌ Failed to load XMVP: {e}")
                 descriptions = ["Placeholder"]
        else:
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
        # Default to 4 if not set
        fbf_fps = args.fps if args.fps else 4
        expansion = max(1, int(fbf_fps))
        total_frames = len(descriptions) * expansion
        
        # Local variable for this specific project's Output FPS
        project_fps = fbf_fps 
        
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
             
             # User Override Check
             if args.fps:
                 logging.info(f"   🏎️  FPS Override: {args.fps}")
                 fps = float(args.fps)
             elif args.vspeed:
                 logging.info(f"   🏎️  VSpeed Override: {args.vspeed}")
                 fps = float(args.vspeed)

             
        else:
             bpm, duration, fpb, fps = analyze_audio(args.mu, fsync=args.fsync)
             # User Override Check
             if args.fps:
                 logging.info(f"   🏎️  FPS Override: {args.fps} (ignoring audio-derived {fps:.2f})")
                 fps = float(args.fps)
             elif args.vspeed:
                 logging.info(f"   🏎️  VSpeed Override: {args.vspeed} (ignoring audio-derived {fps:.2f})")
                 fps = float(args.vspeed)
             
        target_frames = int(duration * fps)
        project_fps = fps
        target_duration = duration
        
        logging.info(f"   🎹 Visualizer Target: {target_frames} frames @ {fps:.2f} FPS ({duration:.1f}s)")
        
        # 2. Generate Abstract Prompts
        # No "Story", just pure visual evolution.
        prompts = []
        
        # Style Definition
        # Visualizer Theme
        visual_concept = args.prompt if args.prompt else "A single continuous abstract form"

        # Style Logic
        # Fix: Check against CLI DEFAULT "high resolution 4K UHD video"
        # If user didn't touch it, it will be that string.
        cli_default_style = "high resolution 4K UHD video"
        
        if args.style == cli_default_style:
             # User left default style. Switch to Visualizer Default.
             base_style = "Abstract mathematical pixel interpolation neural network simulation, highly detailed pixel art, Stan Brakhage style, melting film on hot projectors, unique fractal algorithm, highly saturated colors, glitch art, datamoshing"
        else:
             # User provided custom style
             base_style = args.style

        style_prefix = base_style
        
        # Audio Feature Extraction (Scipy/Librosa)
        # We need per-frame features.
        import numpy as np
        
        frames_energy = []
        frames_timbre = []
        
        try:
            logging.info("   🔬 Running Scipy Audio Analysis (RMS & Spectral Centroid)...")
            y, sr = librosa.load(args.mu, sr=None)
            
            # Hop length = samples per frame
            # fps = project_fps
            hop_length = int(sr / project_fps)
            
            # 1. RMS (Energy)
            rmse = librosa.feature.rms(y=y, hop_length=hop_length)
            # Normalize to 0-1
            rmse = (rmse - np.min(rmse)) / (np.max(rmse) - np.min(rmse) + 1e-6)
            frames_energy = rmse[0]
            
            # 2. Spectral Centroid (Timbre/Brightness)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
            # Normalize to 0-1
            cent = (cent - np.min(cent)) / (np.max(cent) - np.min(cent) + 1e-6)
            frames_timbre = cent[0]
            
            # Pad or Truncate to match target_frames
            if len(frames_energy) < target_frames:
                # Pad with last value
                pad_width = target_frames - len(frames_energy)
                frames_energy = np.pad(frames_energy, (0, pad_width), mode='edge')
                frames_timbre = np.pad(frames_timbre, (0, pad_width), mode='edge')
                
        except Exception as e_audio:
            logging.warning(f"   ⚠️ Audio Feature Extraction Failed: {e_audio}. Using static values.")
            frames_energy = np.zeros(target_frames)
            frames_timbre = np.zeros(target_frames)

        # Visualizer Loop
        for i in range(target_frames):
             progress = i / target_frames
             
             # Audio Features for this frame
             # Safety clamp index
             idx = min(i, len(frames_energy)-1)
             energy = frames_energy[idx]
             timbre = frames_timbre[idx]
             
             # 1. Energy Modulation (Action Intensity)
             # Low Energy (0.0 - 0.3): Inert, Slow, Ambient
             # Mid Energy (0.3 - 0.7): Moving, Rythmic, Flowing
             # High Energy (0.7 - 1.0): Exploding, Chaotic, Glitching
             
             intensity_adj = "inert"
             if energy > 0.2: intensity_adj = "flowing"
             if energy > 0.4: intensity_adj = "pulsating"
             if energy > 0.6: intensity_adj = "vibrating violently"
             if energy > 0.8: intensity_adj = "exploding in recursive chaos"
             
             # 2. Timbre Modulation (Visual Texture/Color)
             # Low Centroid (Bass): Dark, Heavy, Liquid, Deep colors
             # High Centroid (Treble): Bright, Sharp, Crystalline, Neon, White noise
             
             texture_desc = "deep liquid forms"
             if timbre > 0.3: texture_desc = "geometric patterns"
             if timbre > 0.6: texture_desc = "crystalline shards and electric sparks"
             if timbre > 0.8: texture_desc = "blinding white noise and sharp glitches"
             
             # Phase description (Narrative Arc still applies gently)
             phase_desc = "emerging"
             if progress > 0.5: phase_desc = "evolving"
             if progress > 0.9: phase_desc = "resolving"
             
             # Combine User Prompt + Audio Reactive Descriptors
             # "Visualizer Beat 10. [User Prompt]. The image is [Intensity] and composed of [Texture]. [Phase]. Progress: 10%."
             
             # SCIPY INJECTION:
             raw_desc = (
                 f"Visualizer Beat {i}. {visual_concept}. "
                 f"The motion is {intensity_adj}. "
                 f"The visuals are composed of {texture_desc}. "
                 f"Audio Energy: {energy:.2f}. " 
             )
             
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
             
             # VSPEED/FPS Override
             if args.fps:
                  logging.info(f"   🏎️  FPS Override: {args.fps}")
                  fps = float(args.fps)
             elif args.vspeed:
                  logging.info(f"   🏎️  VSpeed Override: {args.vspeed}")
                  fps = float(args.vspeed)

        else:
             bpm, duration, fpb, fps = analyze_audio(args.mu, fsync=args.fsync)
             # VSPEED/FPS Override (even for auto-detected)
             if args.fps:
                  logging.info(f"   🏎️  FPS Override: {args.fps} (ignoring detected {fps:.2f})")
                  fps = float(args.fps)
             elif args.vspeed:
                  logging.info(f"   🏎️  VSpeed Override: {args.vspeed} (ignoring detected {fps:.2f})")
                  fps = float(args.vspeed)
        target_frames = int(duration * fps)
        project_fps = fps
        target_duration = duration
        
        logging.info(f"   🎹 Music Agency Target: {target_frames} frames @ {fps:.2f} FPS ({duration:.1f}s) | BPM: {bpm}")
        
        # 2. Generate Story (Content)
        # IF WE HAVE DESCRIPTIONS FROM XML (and NOT retconning), USE THEM.
        source_content = []
        
        # Check if descriptions were loaded from XML (and are not just "Placeholder")
        xml_loaded = False
        if len(descriptions) > 0 and descriptions[0] != "Placeholder" and descriptions[0] != "Placeholder (Retconning)":
             logging.info("   📜 Using existing beats from XMVP XML.")
             source_content = descriptions
             xml_loaded = True
             
        if not xml_loaded:
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
        est_beats = max(5, int(target_duration / 3.0))
        
        if not xml_loaded:
            # AUDIO PROFILE
            sonic_map = analyze_audio_profile(args.mu, target_duration)
            logging.info(f"      🎵 Sonic Map: {sonic_map}")
            
            story_req = (
                f"Create a VISUAL SCREENPLAY for a {target_duration}s music video (Animated Cartoon style (classic 2D, non-photorealistic)).\n"
                f"Concept: {prompt_concept}\n"
                f"Chaos Seeds to weave in: {seeds}\n"
                f"Music Vibe: {bpm} BPM.\n"
                f"Audio Profile (Energy/Mood over time): {sonic_map}\n"
                f"Constraints: We need approx {est_beats} distinct visual scenes/beats to span the song.\n"
                f"Critical: Match the visual intensity to the Audio Profile (e.g. Calm visuals for 'Quiet', Intense for 'High Energy').\n"
                "Output JSON: { 'title': '...', 'synopsis': '...', 'beats': ['Visual description 1', 'Visual description 2', ...] }"
            )
            
            try:
                # RETRY LOOP for Story Generation (Stochastic Guard)
                max_writer_retries = 3
                story_data = None
                
                for attempt in range(max_writer_retries):
                    try:
                        logging.info(f"      ✍️  Calling Writer (Attempt {attempt+1}/{max_writer_retries})...")
                        raw = text_engine.generate(story_req, json_schema=True)
                        story_data = json.loads(raw)
                        if isinstance(story_data, list):
                            story_data = story_data[0]
                            
                        # Validate keys
                        if 'beats' not in story_data:
                            raise ValueError("Missing 'beats' in JSON.")
                            
                        break # Success
                    except Exception as e_inner:
                         logging.warning(f"      ⚠️ Writer Attempt {attempt+1} Failed: {e_inner}")
                         if attempt == max_writer_retries - 1: raise e_inner
                         time.sleep(1)

                source_content = story_data.get('beats', [])
                context_str = f"Music Video: {story_data.get('title')}"
                logging.info(f"   📜 Generated {len(source_content)} story beats.")
                
            except Exception as e:
                logging.error(f"   ❌ Writer Failed: {e}")
                return

        # 3. Distribute Beats
        
        # Style
        # Override default style for Music Video to "Saturday Morning Cartoon" if user didn't specify one
        cli_default_style = "high resolution 4K UHD video"
        
        if args.style.strip() == cli_default_style:
             style_prefix = "professionally animated cartoon"
        else:
             style_prefix = args.style
        
        # 3. Distribute Beats (Shot-Aware Timeline Construction)
        timeline = [] # List of dicts: { 'prompt': str, 'is_cut': bool, 'shot_idx': int }
        
        # Check if we have Manifest Segments with valid durations
        has_rich_manifest = (xml_loaded and len(manifest_segments) > 0 and 'duration' in manifest_segments[0])
        
        if has_rich_manifest:
             logging.info("   🎬 detailed Manifest detected. Constructing Shot-Aware Timeline...")
             
             current_frame_count = 0
             
             for idx, seg in enumerate(manifest_segments):
                 # Get duration
                 dur = seg.get('duration', 4.0)
                 story_beat = seg.get('prompt', getattr(seg, 'content', 'Action'))
                 
                 # Calculate frames for this shot
                 shot_frames = int(dur * project_fps)
                 if shot_frames < 1: shot_frames = 1
                 
                 # Append to timeline
                 for f in range(shot_frames):
                     # Stop if we exceed target
                     if len(timeline) >= target_frames: break
                     
                     is_cut = (f == 0) # First frame of shot is a CUT
                     
                     # Style Injection
                     # Use the specific shot style if available, else global
                     # User Request: "professionally animated cartoon at [FPS] of {content}"
                     # User Request: Enforce specific cartoon style phrasing
                     # We use the style_prefix we calculated above
                     full_prompt = f"{style_prefix} at {project_fps} FPS of {story_beat}"
                     
                     timeline.append({
                         'prompt': full_prompt,
                         'is_cut': is_cut,
                         'shot_idx': idx
                     })
                     
                 if len(timeline) >= target_frames: break
                 
             # FILLER: If timeline is shorter than target_frames (e.g. manifest explicitly sums to less than audio)
             # We extend the last shot
             if len(timeline) < target_frames:
                 missing = target_frames - len(timeline)
                 logging.warning(f"   ⚠️ Timeline underflow: {len(timeline)} frames generated vs {target_frames} target. Extending final shot.")
                 last_entry = timeline[-1] if timeline else {'prompt': "Static", 'is_cut': True, 'shot_idx': -1}
                 extra_entry = last_entry.copy()
                 extra_entry['is_cut'] = False # Extension is not a cut
                 
                 for _ in range(missing):
                     timeline.append(extra_entry)
                     
             # SYNC: Populate 'prompts' list for compatibility with FC/Recursive modes
             prompts = [item['prompt'] for item in timeline]
             logging.info(f"   ✅ Synchronized {len(prompts)} prompts from Timeline.")
                     
        else:
             # LEGACY "PEANUT BUTTER" MODE (Average Distribution)
             if not source_content:
                  source_content = ["Band performing on stage."]

             # BEAT EXPANSION (Granularity Check)
             # If beats are too sparse (e.g. >10s per beat), we need to expand them for animation.
             avg_beat_duration = target_duration / len(source_content)
             if avg_beat_duration > 10.0 and args.wan:
                 logging.info(f"   🤏 Beats are sparse ({avg_beat_duration:.1f}s/beat). Expanding granularity...")
                 
                 # Target ~5s per beat
                 target_beat_count = int(target_duration / 5.0)
                 expansion_prompt = (
                     f"The current visual script has {len(source_content)} beats, but we need {target_beat_count} distinct visual moments for the animation (one every ~5s).\n"
                     f"Current Script: {json.dumps(source_content)}\n"
                     f"Task: Expand this into {target_beat_count} detailed, sequential visual descriptions. Maintain the narrative flow but break it down into smaller, granular actions.\n"
                     "Output JSON: { 'beats': ['Detailed beat 1', 'Detailed beat 2', ...] }"
                 )
                 
                 try:
                     raw_exp = text_engine.generate(expansion_prompt, json_schema=True)
                     exp_data = json.loads(raw_exp)
                     if 'beats' in exp_data and len(exp_data['beats']) > len(source_content):
                         source_content = exp_data['beats']
                         logging.info(f"   ✨ Expanded to {len(source_content)} beats ({target_duration/len(source_content):.1f}s/beat).")
                 except Exception as e_exp:
                     logging.warning(f"   ⚠️ Beat expansion failed: {e_exp}. Using original beats.")
     
             num_beats = len(source_content)
             frames_per_beat = target_frames / num_beats
             
             logging.info(f"   🥜 Constructing Average Pacing Timeline ({len(source_content)} beats over {target_frames} frames)...")
             
             prompts = []
             for i in range(target_frames):
                  beat_idx = int(i / frames_per_beat)
                  beat_idx = min(beat_idx, num_beats - 1)
                  
                  raw_desc = source_content[beat_idx]
                  
                  # Determine Cut (heuristic)
                  # If beat_idx changed from previous frame
                  prev_beat_idx = int((i-1) / frames_per_beat) if i > 0 else -1
                  is_cut = (beat_idx != prev_beat_idx)
                  
                  # User Request: Enforce specific cartoon style phrasing
                  # User Request: Enforce specific cartoon style phrasing
                  full_prompt = f"{style_prefix} at {project_fps} FPS of {raw_desc}"
                  
                  timeline.append({
                      'prompt': full_prompt,
                      'is_cut': is_cut,
                      'shot_idx': beat_idx
                  })
                  prompts.append(full_prompt)


    elif args.vpform == "cartoon-video":
        # CARTOON VIDEO (Img2Img Video Redraw)
        if not args.mu or not os.path.exists(args.mu):
             logging.error("❌ cartoon-video requires input video (pass as arg or --mu)")
             return

        # 1. Setup
        project_fps = args.fps # Default 4
        logging.info(f"   🎞️  Cartoon Video: Redrawing {Path(args.mu).name} @ {project_fps} FPS...")
        
        frames_dir = project_dir / "frames"
        ensure_dir(frames_dir)
        
        # 2. Extract Audio
        audio_path = project_dir / "extracted_audio.wav"
        has_audio = False
        try:
            cmd_a = ['ffmpeg', '-y', '-i', args.mu, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)]
            subprocess.run(cmd_a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                 has_audio = True
                 logging.info(f"   🎵 Audio extracted: {audio_path}")
        except Exception as e_audio:
            logging.warning(f"   ⚠️ Could not extract audio (Silent input?): {e_audio}")
            has_audio = False
        
        # 3. Extract Frames
        # Calculate vspeed step or just let ffmpeg handle fps
        logging.info("   🎞️  Extracting Source Frames...")
        cmd_f = ['ffmpeg', '-y', '-i', args.mu, '-vf', f'fps={project_fps}', str(frames_dir / 'source_%04d.png')]
        subprocess.run(cmd_f, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        source_frames = sorted(list(frames_dir.glob("source_*.png")))
        logging.info(f"   found {len(source_frames)} source frames.")
        
        # 4. Img2Img Loop
        from flux_bridge import get_flux_bridge
        
        bridge = None
        if args.local and not args.cloud: # Only load local bridge if local
             # Robust Flux Loading
             try:
                  flux_path = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE]["flux-schnell"].path
             except:
                  flux_path = "/Volumes/XMVPX/mw/flux-root"
             bridge = get_flux_bridge(flux_path)
        
        prompt = "exactly precise reproduction of this image in terms of content and image/scene structure, but improved to a different asethetic style and standard, like an uncanny Octane Render Unreal Engine 3D real-life photorealistic artistic reimagining. 8k, highly detailed."
        if args.prompt: prompt = args.prompt
        
        # --- MEMORY OPTIMIZATION ---
        # Ensure Img2Img is loaded (Local Only)
        if args.local and not args.cloud:
             bridge.load_img2img()
             if bridge.pipeline and bridge.img2img_pipeline and bridge.pipeline is not bridge.img2img_pipeline:
                 logging.info("   📉 Optimizing Memory: Unloading duplicate Txt2Img pipeline to make room for Img2Img...")
                 del bridge.pipeline
                 bridge.pipeline = None
                 import gc
                 gc.collect()
                 if torch.backends.mps.is_available():
                     torch.mps.empty_cache()
        # ---------------------------
        
        # from PIL import Image
        
        # Scaling Logic (50% Default)
        target_w = args.w
        target_h = args.h
        
        if not target_w and source_frames:
             try:
                 with Image.open(source_frames[0]) as probe:
                     w, h = probe.size
                     # Default to 50% scale
                     target_w = int(w * 0.5)
                     target_h = int(h * 0.5)
                     logging.info(f"   📉 Auto-Scaling: {w}x{h} -> {target_w}x{target_h} (50%)")
             except Exception as e:
                 logging.warning(f"   ⚠️ Could not probe first frame: {e}")
                 target_w = 1024
                 target_h = 576
        
        # Enforce 16-pixel alignment (Crucial for Flux VAE)
        if target_w and target_h:
            orig_w, orig_h = target_w, target_h
            target_w = (target_w // 16) * 16
            target_h = (target_h // 16) * 16
            if target_w != orig_w or target_h != orig_h:
                 logging.warning(f"   ⚠️ Snapping resolution to 16px grid: {orig_w}x{orig_h} -> {target_w}x{target_h}")

        # 4. Img2Img Loop
        last_output_img = None
        
        for i, src in enumerate(source_frames):
            idx = i + 1
            dst = frames_dir / f"frame_{idx:04d}.png"
            
            if dst.exists(): 
                # If existing, load it into memory for next frame's blending
                try: 
                    last_output_img = Image.open(dst).convert("RGB")
                    # Ensure size match for blending later
                    if last_output_img.size != (target_w, target_h):
                        last_output_img = last_output_img.resize((target_w, target_h), Image.LANCZOS)
                except:
                    pass
                continue
                
            logging.info(f"   🎨 Redrawing Frame {idx}/{len(source_frames)}...")
            try:
                src_img = Image.open(src).convert("RGB")
                
                # Resize src to target if needed (to match last_output/target)
                if src_img.size != (target_w, target_h):
                    src_img = src_img.resize((target_w, target_h), Image.LANCZOS)
                
                # TEMPORAL BLENDING (Stability Hack)
                # Blend 30% of previous output into current source
                input_img = src_img
                if last_output_img:
                    # alpha=0.3 means 30% image2 (last_output), 70% image1 (src)
                    logging.info(f"      🌀 Blending 30% of Frame {idx-1} for stability...")
                    input_img = Image.blend(src_img, last_output_img, alpha=0.3)
                
                # CLOUD MODE (Imagen 3 Director Pipeline)
                if args.cloud or not args.local:
                     # DIRECTOR MODE: Gemini 2.0 Flash (Describe) -> Imagen 3 (Generate)
                     # We use the key cycle reused from main() or instantiate a fresh client
                     # Grab a key from the passed 'keys' list (guaranteed valid by main)
                     api_key = keys[0] if keys else os.environ.get("GEMINI_API_KEY") 
                     client = genai.Client(api_key=api_key)

                     # 1. Describe (Director)
                     description_prompt = (
                         f"You are a Lead Animator. Analyze this frame from a video.\n"
                         f"Goal: Describe the VISUALS in extreme detail so they can be re-drawn in a specific style.\n"
                         f"Style Target: {prompt}\n" # User's prompt IS the style target
                         f"Action: Describe the character pose, lighting, setting, and composition exactly.\n"
                         f"Output: A dense, visual description suitable for an image generator."
                     )
                     
                     # Prepare Input
                     # Convert PIL to bytes
                     import io
                     img_byte_arr = io.BytesIO()
                     input_img.save(img_byte_arr, format='PNG')
                     img_bytes = img_byte_arr.getvalue()
                     
                     desc_contents = [
                         types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                         description_prompt
                     ]
                     
                     try:
                         # Retry Loop for Description
                         director_desc = ""
                         for _ in range(3):
                             try:
                                 resp = client.models.generate_content(
                                     model="gemini-2.0-flash",
                                     contents=desc_contents
                                 )
                                 if resp.text:
                                     director_desc = resp.text
                                     break
                             except Exception as e_d:
                                 logging.warning(f"   ⚠️ Director Describe Failed: {e_d}. Retrying...")
                                 time.sleep(1)
                         
                         if not director_desc:
                             logging.error("   ❌ Director failed to describe frame. Skipping.")
                             continue
                             
                         logging.info(f"   🎬 Director: {director_desc[:60]}...")
                         
                         # 2. Render (Flux1.dev Txt2Img)
                         # We use the detailed description AS the prompt for Flux.
                         # Pass image=None to force Txt2Img for maximum stylization as requested.
                         from flux_bridge import generate_via_hf_endpoint
                         
                         out_img = None
                         
                         # Retry Loop for Render
                         for _ in range(3):
                             try:
                                 out_img = generate_via_hf_endpoint(
                                     prompt=director_desc,
                                     width=target_w,
                                     height=target_h,
                                     steps=28,
                                     guidance=3.5,
                                     image=None, # Txt2Img Mode
                                     strength=1.0, # Irrelevant for Txt2Img but good practice
                                     seed=42+i
                                 )
                                 if out_img:
                                     # Resize back to exact target just in case
                                     if out_img.size != (target_w, target_h):
                                         out_img = out_img.resize((target_w, target_h), Image.LANCZOS)
                                     break
                             except Exception as e_r:
                                  logging.warning(f"   ⚠️ Render Failed: {e_r}")
                                  time.sleep(1)
                                  
                     except Exception as e_cloud:
                         logging.error(f"   ❌ Cloud Director Pipeline Failed: {e_cloud}")
                         out_img = None
                         
                else: 
                     # LOCAL MODE
                     out_img = bridge.generate_img2img(
                        prompt=prompt,
                        image=input_img,
                        strength=getattr(args, 'strength', 75) / 100.0, 
                        width=target_w,
                        height=target_h,
                        seed=42 + i
                     )
                
                if out_img:
                    out_img.save(dst)
                    last_output_img = out_img
            except Exception as e:
                logging.error(f"Frame {idx} fail: {e}")
                
        # 5. Stitch
        logging.info("   🧵 Stitching...")
        
        # Check if frames exist
        found_frames = list(frames_dir.glob("frame_*.png"))
        if not found_frames:
             logging.error("❌ No frames generated! Aborting stitch.")
             return

        out_vid = project_dir / "final_output.mp4"
        # use existing stitch logic or ffmpeg direct
        # Pattern frame_%04d.png
        cmd_s = [
            'ffmpeg', '-y', '-framerate', str(project_fps),
            '-i', str(frames_dir / 'frame_%04d.png'),
        ]
        
        # Conditionally add audio if valid
        if has_audio and audio_path.exists():
            cmd_s.extend(['-i', str(audio_path), '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k'])
        else:
            logging.info("   🔇 Stitching generic silent video (no audio source found).")
            cmd_s.extend(['-c:v', 'libx264']) # Video only
            
        cmd_s.extend([
            '-pix_fmt', 'yuv420p', '-crf', '18',
            '-shortest',
            str(out_vid)
        ])
        
        subprocess.run(cmd_s, check=True)
        logging.info(f"   ✅ Done: {out_vid}")
        return

    elif args.vpform == "ansi-video":
        # =======================================================================
        # ANSI VIDEO MODE - LLM draws each frame with block characters
        # =======================================================================
        
        if not args.mu or not os.path.exists(args.mu):
            logging.error("❌ ansi-video requires audio file (--mu)")
            return
        
        # 1. CANVAS SETUP
        # --w and --h specify CHARACTER counts (not pixels), defaulting to 60x40
        canvas_width = args.w if args.w else 60
        canvas_height = args.h if args.h else 40
        
        # Calculate pixel size to hit reasonable output resolution
        # Target ~480x320 minimum, scale up as needed
        min_output_width = 480
        pixel_size = max(8, min_output_width // canvas_width)
        output_width = canvas_width * pixel_size
        output_height = canvas_height * pixel_size
        
        logging.info(f"   🎨 ANSI Video Mode: {canvas_width}×{canvas_height} chars → {output_width}×{output_height} pixels")
        
        # 2. AUDIO ANALYSIS
        logging.info(f"   🎵 Analyzing audio: {args.mu}")
        bpm, duration, beat_frames, suggested_fps = analyze_audio(args.mu, fsync=args.fsync)
        
        # User FPS override
        project_fps = args.fps if args.fps else suggested_fps
        if project_fps < 1:
            project_fps = 4  # Minimum sane FPS
        
        total_frames = int(duration * project_fps)
        logging.info(f"   ⏱️  Target: {total_frames} frames @ {project_fps} FPS ({duration:.1f}s) | BPM: {bpm}")
        
        # 3. PALETTE DETECTION
        prompt_text = args.prompt if args.prompt else "Abstract animation"
        style_text = args.style if args.style else ""
        palette_name = detect_ansi_palette(prompt_text, style_text)
        logging.info(f"   🎨 Detected Palette: {palette_name}")
        
        # 4. STORY GENERATION (reuse existing music-video writer logic)
        logging.info("   🧠 Generating narrative beats...")
        
        # Analyze audio profile for mood
        sonic_map = analyze_audio_profile(args.mu, duration)
        
        # Calculate number of story beats (1 beat per ~3s)
        num_beats = max(5, int(duration / 3.0))
        
        story_request = (
            f"Create a VISUAL STORYBOARD for a {duration:.0f}s animation.\n"
            f"Style: ANSI pixel art / chunky block characters\n"
            f"Concept: {prompt_text}\n"
            f"Music Vibe: {bpm} BPM\n"
            f"Audio Profile: {sonic_map}\n"
            f"Constraints: Need exactly {num_beats} visual beats/scenes.\n"
            f'Output ONLY valid JSON: {{"title": "...", "beats": ["Scene 1", "Scene 2", ...]}}\n'
            "Keep descriptions VISUAL and SIMPLE - 10 words max per beat!"
        )
        
        story_beats = []
        
        # Retry loop for story generation
        for story_attempt in range(3):
            try:
                raw = te.generate(story_request, json_schema=True)
                
                # Try to extract JSON from the response
                # First, try direct parse
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # Try to find JSON object in the response
                    start = raw.find('{')
                    end = raw.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        try:
                            data = json.loads(raw[start:end+1])
                        except json.JSONDecodeError:
                            # Try fixing common issues (trailing commas, etc)
                            json_str = raw[start:end+1]
                            # Remove trailing commas before ] or }
                            import re as json_re
                            json_str = json_re.sub(r',\s*([}\]])', r'\1', json_str)
                            data = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError("No JSON found", raw, 0)
                
                if isinstance(data, list):
                    data = data[0]
                story_beats = data.get('beats', [])
                
                if story_beats and len(story_beats) >= 3:
                    logging.info(f"   📜 Generated {len(story_beats)} story beats")
                    break
                else:
                    logging.warning(f"   ⚠️ Story attempt {story_attempt+1}: Only got {len(story_beats)} beats, retrying...")
                    
            except Exception as e:
                logging.warning(f"   ⚠️ Story attempt {story_attempt+1} failed: {e}")
                if story_attempt == 2:
                    # Final fallback: create varied beats from the prompt
                    logging.info("   🔄 Using prompt-derived beats as fallback...")
                    story_beats = [
                        f"Wide establishing shot: silhouette of {prompt_text} against sky, horizon line at bottom third",
                        f"Medium shot: {prompt_text} character(s) visible, distinct shapes against background",
                        f"Action sequence: movement and energy, dynamic poses, motion blur effects",
                        f"Close-up dramatic moment: large central figure fills 40% of frame",
                        f"Climax: explosion of shapes, maximum visual contrast",
                        f"Resolution: characters small against vast landscape, peaceful composition"
                    ]
                    # Pad to num_beats if needed
                    while len(story_beats) < num_beats:
                        story_beats.append(f"Scene {len(story_beats)+1}: {prompt_text} characters in distinct poses against contrasting background")
        
        if not story_beats:
            story_beats = [prompt_text]
        
        # 5. DISTRIBUTE BEATS TO FRAMES
        frames_per_beat = total_frames / len(story_beats)
        prompts = []
        
        for i in range(total_frames):
            beat_idx = min(int(i / frames_per_beat), len(story_beats) - 1)
            beat_desc = story_beats[beat_idx]
            
            # Add frame-specific context with visual direction
            progress_pct = int((i / total_frames) * 100)
            
            # Determine visual style based on story position
            if progress_pct < 15:
                composition = "wide shot, establish setting, horizon visible"
            elif progress_pct < 40:
                composition = "medium shot, characters filling 30% of frame"
            elif progress_pct < 70:
                composition = "dynamic action, diagonal lines, high contrast"
            elif progress_pct < 90:
                composition = "close-up, subject fills 50% of canvas"
            else:
                composition = "wide pullback, small figures in vast scene"
            
            frame_prompt = f"[{progress_pct}% through story] {beat_desc}\nCOMPOSITION: {composition}"
            prompts.append(frame_prompt)
        
        logging.info(f"   ✅ Distributed {len(prompts)} frame prompts")
        
        # 6. FRAME GENERATION LOOP
        project_out = output_root / f"ansi_{Path(args.mu).stem}_{int(time.time())}"
        ensure_dir(project_out)
        
        frames_dir = project_out / "frames"
        ensure_dir(frames_dir)
        
        prev_frame_text = None
        success_count = 0
        
        logging.info(f"   🖌️  Drawing {total_frames} ANSI frames...")
        print("   Progress: ", end="", flush=True)
        
        for i, prompt in enumerate(prompts):
            idx = i + 1
            
            try:
                frame_path, frame_text = generate_ansi_frame(
                    index=idx,
                    prompt=prompt,
                    output_dir=frames_dir,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    text_engine=te,
                    prev_frame_text=prev_frame_text,
                    palette_name=palette_name
                )
                
                if frame_path.exists():
                    success_count += 1
                    print(".", end="", flush=True)
                    prev_frame_text = frame_text  # Feedback for next frame
                else:
                    print("x", end="", flush=True)
                    
            except Exception as e:
                logging.warning(f"   Frame {idx} failed: {e}")
                print("!", end="", flush=True)
            
            # Rate limiting
            if hasattr(args, 'delay') and args.delay > 0:
                time.sleep(args.delay)
            
            # Memory cleanup every 10 frames to prevent GPU exhaustion
            if idx % 10 == 0:
                import gc
                gc.collect()
                # Clear MPS cache if using Metal
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except:
                    pass
                # Clear MLX cache
                try:
                    import mlx.core as mx
                    mx.metal.clear_cache()
                except:
                    pass
                # Brief pause to let GPU recover
                time.sleep(0.5)
        
        print(f"\n   ✅ Generated {success_count}/{total_frames} frames")
        
        # 7. VIDEO ASSEMBLY
        if success_count > 0:
            raw_video = project_out / "raw_video.mp4"
            final_video = project_out / "final_ansi.mp4"
            
            logging.info("   🎬 Stitching video...")
            
            cmd_stitch = [
                'ffmpeg', '-y',
                '-framerate', str(project_fps),
                '-i', str(frames_dir / 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                str(raw_video)
            ]
            
            try:
                subprocess.run(cmd_stitch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logging.error(f"   ❌ FFmpeg stitch failed: {e}")
                return
            
            # Mux audio
            logging.info("   🔊 Muxing audio...")
            cmd_mux = [
                'ffmpeg', '-y',
                '-i', str(raw_video),
                '-i', str(args.mu),
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(final_video)
            ]
            
            try:
                subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"   ✅ ANSI VIDEO COMPLETE: {final_video}")
            except Exception as e:
                logging.error(f"   ❌ Audio mux failed: {e}")
                logging.info(f"   📹 Silent version: {raw_video}")
        
        return

    elif args.vpform == "ansi-redraw":
        # =======================================================================
        # ANSI REDRAW MODE - LLM redraws each video frame as block character art
        # =======================================================================
        
        if not args.mu or not os.path.exists(args.mu):
            logging.error("❌ ansi-redraw requires input video (--mu)")
            return

        # 1. CANVAS SETUP
        canvas_width = args.w if args.w else 120
        canvas_height = args.h if args.h else 80
        logging.info(f"   📐 ANSI Canvas: {canvas_width}×{canvas_height} characters")
        
        # 2. PALETTE DETECTION (from prompt/style)
        style_prompt = args.prompt or args.style or ""
        palette_name = detect_ansi_palette(style_prompt)
        logging.info(f"   🎨 Palette: {palette_name}")
        
        # 3. INITIALIZE TEXT ENGINE
        te = TextEngine()
        logging.info(f"   🧠 Text Engine: {te.backend}")
        
        # 4. EXTRACT FRAMES FROM INPUT VIDEO
        project_fps = args.fps if args.fps else 4
        
        # Probe video duration
        cmd_probe = ['ffprobe', '-i', str(args.mu), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
        try:
            res = subprocess.run(cmd_probe, capture_output=True, text=True)
            duration = float(res.stdout.strip())
        except:
            duration = 60.0
            logging.warning("   ⚠️ Could not probe video duration. Assuming 60s.")
        
        total_frames = int(duration * project_fps)
        logging.info(f"   🎞️  Extracting source frames: {total_frames} frames @ {project_fps} FPS")
        
        # Setup project directory
        project_out = output_root / f"ansi_redraw_{Path(args.mu).stem}_{int(time.time())}"
        ensure_dir(project_out)
        
        source_frames_dir = project_out / "source"
        ensure_dir(source_frames_dir)
        
        frames_dir = project_out / "frames"
        ensure_dir(frames_dir)
        
        # Extract source frames
        cmd_extract = [
            'ffmpeg', '-y',
            '-i', str(args.mu),
            '-vf', f'fps={project_fps}',
            str(source_frames_dir / 'source_%04d.png')
        ]
        
        try:
            subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logging.error(f"   ❌ FFmpeg extract failed: {e}")
            return
        
        source_frames = sorted(list(source_frames_dir.glob("source_*.png")))
        logging.info(f"   ✅ Extracted {len(source_frames)} source frames")
        
        if len(source_frames) == 0:
            logging.error("   ❌ No source frames extracted!")
            return
        
        # 5. FRAME REDRAW LOOP
        prev_frame_text = None
        success_count = 0
        
        logging.info(f"   🖌️  Redrawing {len(source_frames)} frames as ANSI art...")
        print("   Progress: ", end="", flush=True)
        
        for i, src_frame in enumerate(source_frames):
            idx = i + 1
            
            try:
                frame_path, frame_text = redraw_frame_as_ansi(
                    source_image_path=src_frame,
                    index=idx,
                    output_dir=frames_dir,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    text_engine=te,
                    prev_frame_text=prev_frame_text,
                    palette_name=palette_name,
                    style_prompt=style_prompt
                )
                
                prev_frame_text = frame_text
                success_count += 1
                print("█", end="", flush=True)
                
            except Exception as e:
                logging.error(f"   Frame {idx} failed: {e}")
                print("░", end="", flush=True)
            
            # Memory cleanup every 10 frames (for local mode)
            if idx % 10 == 0 and args.local:
                gc.collect()
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except:
                    pass
                try:
                    import mlx.core as mx
                    mx.metal.clear_cache()
                except:
                    pass
                time.sleep(0.5)
        
        print(f"\n   ✅ Generated {success_count}/{len(source_frames)} frames")
        
        # 6. VIDEO ASSEMBLY
        if success_count > 0:
            raw_video = project_out / "raw_video.mp4"
            final_video = project_out / "final_ansi_redraw.mp4"
            
            logging.info("   🎬 Stitching video...")
            
            cmd_stitch = [
                'ffmpeg', '-y',
                '-framerate', str(project_fps),
                '-i', str(frames_dir / 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                str(raw_video)
            ]
            
            try:
                subprocess.run(cmd_stitch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logging.error(f"   ❌ FFmpeg stitch failed: {e}")
                return
            
            # Extract and mux original audio
            logging.info("   🔊 Muxing audio from source video...")
            cmd_mux = [
                'ffmpeg', '-y',
                '-i', str(raw_video),
                '-i', str(args.mu),
                '-map', '0:v',
                '-map', '1:a?',  # ? makes audio optional
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(final_video)
            ]
            
            try:
                subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"   ✅ ANSI REDRAW COMPLETE: {final_video}")
            except Exception as e:
                logging.error(f"   ❌ Audio mux failed: {e}")
                logging.info(f"   📹 Silent version: {raw_video}")
        
        return

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
            bpm, duration, fpb, fps = analyze_audio(args.mu, fsync=args.fsync)
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
             # User Request: Enforce specific cartoon style phrasing
             prompts.append(f"professionally animated cartoon at {project_fps} FPS of {raw_desc}")
             
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
                 start_img = frame_canvas.generate_seed_image(prompts[i_idx], te, init_dim=args.kid, width=target_w, height=target_h)
                 if start_img:
                     start_img.save(start_path)
                     last_generated_img = start_img
                     success_count += 1
            else:
                 # Load existing start frame
                 if start_path.exists():
                      # from PIL import Image # Removed to fix UnboundLocalError
                      start_img = Image.open(start_path)
                 else:
                      logging.warning(f"   ⚠️ Start Frame {frame_start_idx} missing in batch. Regenerating...")
                      start_img = frame_canvas.generate_seed_image(prompts[i_idx], te, init_dim=args.kid, width=target_w, height=target_h)
                      if start_img: start_img.save(start_path)

            if not start_img:
                logging.error("   ❌ Batch Start Failed. Aborting batch.")
                i_idx += 1
                continue
                
            # RESIZE ENFORCEMENT
            if start_img.size != (target_w, target_h):
                 # logging.info(f"   📏 Resizing Start Frame: {start_img.size} -> {target_w}x{target_h}")
                 start_img = start_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                 start_img.save(start_path) # Save corrected size

            # 2. Determine End Frame (Target: +4 frames)
            step = 4
            next_i = min(i_idx + step, N - 1)
            
            if next_i <= i_idx:
                 break # End of sequence
                 
            # 3. Generate End Frame via Flux
            frame_end_idx = next_i + 1
            end_path = frames_dir / f"frame_{frame_end_idx:04d}.png"
            logging.info(f"   🎨 Batch End: Frame {frame_end_idx} (Flux)...")
            
            end_img = frame_canvas.generate_seed_image(prompts[next_i], te, init_dim=args.kid, width=target_w, height=target_h)
            if end_img:
                if end_img.size != (target_w, target_h):
                     # logging.info(f"   📏 Resizing End Frame: {end_img.size} -> {target_w}x{target_h}")
                     end_img = end_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
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
            
            # Helper for Img2Img
            def do_flux_tween(img_a, img_b, prompt_text, out_p):
                 # 1. Tween (50%)
                 t_img = frame_canvas.tween_frames(img_a, img_b, blend=0.5)
                 if not t_img: return None
                 
                 # 2. Img2Img Refine
                 # Ensure we have the bridge (re-used from Seed gen or init here)
                 # We need to get the bridge instance. 
                 # Since we are in local recursion, we know we are using Flux.
                 # We can get it from definitions or cached.
                 img_conf = definitions.get_active_model(Modality.IMAGE)
                 from flux_bridge import get_flux_bridge
                 bridge = get_flux_bridge(img_conf.path)
                 
                 if bridge:
                      strength = getattr(args, 'strength', 75) / 100.0
                      logging.info(f"   ✨ Flux Img2Img: {out_p.name} (Str: {strength})...")
                      # Use a lower strength to preserve the "morph" but add details
                      return bridge.generate(prompt_text, image=t_img, strength=strength, width=target_w, height=target_h)
                 return t_img # Fallback
            
            if mid_i > i_idx and mid_i < next_i:
                 frame_mid_idx = mid_i + 1
                 mid_path = frames_dir / f"frame_{frame_mid_idx:04d}.png"
                 logging.info(f"   ✨ Batch Mid: Frame {frame_mid_idx} (Flux Img2Img)...")
                 
                 mid_img = do_flux_tween(start_img, end_img, prompts[mid_i], mid_path)
                 if mid_img:
                      mid_img.save(mid_path)
                      success_count += 1
                 else:
                      logging.warning("   ⚠️ Mid-Tween Failed.")
                      mid_img = frame_canvas.tween_frames(start_img, end_img, blend=0.5)

            # 5. Fill Gaps with Flux Img2Img
            # Gap 1: Start -> Mid
            gap1_i = i_idx + 1
            if gap1_i < mid_i and mid_img and gap1_i < len(prompts):
                 logging.info(f"   ✨ Batch Fill A: Frame {gap1_i+1} (Flux Img2Img)...")
                 fill1 = do_flux_tween(start_img, mid_img, prompts[gap1_i], frames_dir / f"frame_{gap1_i+1:04d}.png")
                 if fill1:
                      fill1.save(frames_dir / f"frame_{gap1_i+1:04d}.png")
                      success_count += 1

            # Gap 2: Mid -> End
            gap2_i = mid_i + 1
            if gap2_i < next_i and mid_img and end_img and gap2_i < len(prompts):
                  logging.info(f"   ✨ Batch Fill B: Frame {gap2_i+1} (Flux Img2Img)...")
                  fill2 = do_flux_tween(mid_img, end_img, prompts[gap2_i], frames_dir / f"frame_{gap2_i+1:04d}.png")
                  if fill2:
                       fill2.save(frames_dir / f"frame_{gap2_i+1:04d}.png")
                       success_count += 1

            # Advance loop: The End Frame becomes the new Start Frame
            i_idx = next_i
            last_generated_img = end_img
            
    # WAN 2.1 KEYFRAME ANIMATION WORKFLOW
    if args.wan and args.local:
        logging.info("🚀 Starting Wan 2.1 Keyframe Animation Workflow...")
        run_wan_keyframe_anim(args, prompts, project_fps, project_out, duration) # Passed project_out is 'output_root' equivalent?
        # Note: process_project args: (project_dir, vf_dir, key_cycle, args, output_root, keys, text_engine)
        # We need check output_root variable name. usually 'project_out' or similar.
        # Let's check context. 'frames_dir' is used in loop.
        # 'project_out' usually defined earlier.
        # Actually in scan view, output_dir passed to generate_frame_universal is frames_dir.
        # Let's assume 'frames_dir' parent is project root.
        return

    legacy_prompts = []
    # If timeline exists (New Logic), use it. If not (Legacy modes like visualizer), construct pseudo-timeline or handling.
    if 'timeline' in locals() and timeline:
        pass # We will use timeline in loop
    else:
        # Fallback for modes that didn't generate a timeline (e.g. fbf-cartoon, creative-agency, legacy interpolation)
        # We wrap the 'prompts' list into a basic timeline
        timeline = []
        for i, p in enumerate(prompts):
            timeline.append({
                'prompt': p,
                'is_cut': (i==0), # Only cut on first frame? Or just standard flow
                'shot_idx': 0
            })
            
    # Normalize Prompt Source
    # The loop below uses 'timeline'.
    
    for i, frame_data in enumerate(timeline):
        if recursive_mode: break # Skip standard generation if recursive mode handled it
        
        index = i + 1
        p = frame_data['prompt']
        is_cut = frame_data.get('is_cut', False)

        index = i + 1
        
        model_to_use = getattr(args, 'model', None)
        if not model_to_use:
             model_to_use = get_active_model(Modality.IMAGE).name

        # OVERRIDE: Cloud Mode -> Always use Flux 2 Klein HF (per user request)
        if not args.local:
             # Check if we are doing image generation (implied by this loop)
             # Previously only music-visualizer was overridden.
             # Now apply to all.
             if model_to_use != "flux-2-klein-hf":
                 # logging.info("   ☁️  Cloud Mode: Enforcing 'flux-2-klein-hf' (HF Endpoint)...")
                 model_to_use = "flux-2-klein-hf"

        # Define Previous Frame Path
        prev_frame = None
        if i > 0:
             # ORIGINAL: Grab previous frame
             prev_frame = frames_dir / f"frame_{i:04d}.png" 
             
             # HARD CUT IMPLEMENTATION
             if is_cut:
                 logging.info(f"   ✂️  HARD CUT DETECTED at Frame {index}. Clearing context.")
                 prev_frame = None # Force Flux to generate fresh
             elif prev_frame and prev_frame.exists():
                 # Normal flow
                 pass
             else:
                 prev_frame = None
        
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
                            # from PIL import Image
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
                width=target_w,
                height=target_h,
                force_local=args.local,
                strength=getattr(args, "strength", 75) / 100.0
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
                    # from PIL import Image
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
                # from PIL import Image
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
        if args.vpform == "music-visualizer":
            # Determine Source (Prefer Final, then Raw)
            source_vid = final_video if final_video.exists() else (raw_video if raw_video.exists() else None)
            
            if source_vid:
                blend_pct = getattr(args, 'blend', 66)
                
                if blend_pct <= 0:
                     logging.info("ℹ️ Blend is 0. Skipping Unicode Redraw.")
                else:
                    logging.info(f"🔥 Entering Post-Production: Unicode Redraw (Opacity {blend_pct}%)...")
                    
                    unicode_vid = project_out / "unicode_version.mp4"
                    
                    # Determine Width (Default to 128 for visualizer if not set)
                    # User Request: Default CW = W / 2
                    target_cw = 128
                    if args.cw:
                        target_cw = int(args.cw)
                    elif args.w:
                        target_cw = int(int(args.w) / 2)
                    
                    if run_unicode_forge(source_vid, unicode_vid, width=target_cw):
                        logging.info(f"   ✅ Unicode Version Forged: {unicode_vid}")
                        
                        final_master = project_out / "master_release.mp4"
                        
                        if blend_pct >= 100:
                             logging.info("   💯 100% Opacity: Unicode Video is the Master.")
                             shutil.copy(unicode_vid, final_master)
                        else:
                             # Blend
                             blended_vid = project_out / "final_blend.mp4"
                             opacity = blend_pct / 100.0
                             logging.info(f"   🎨 Blending Unicode Overlay ({blend_pct}%)...")
                             
                             # Assumption: blend_videos is available (imported or defined)
                             # If not, we might fail here. But it was in the code we replaced.
                             try:
                                 if blend_videos(source_vid, unicode_vid, blended_vid, opacity=opacity):
                                     logging.info(f"   ✅ FINAL VISUALIZER MASTER: {blended_vid}")
                                     
                                     # Remux Audio (Blend likely strips it)
                                     if args.mu or source_vid == final_video:
                                         # If source was final_vid, it had audio. 
                                         # If args.mu present, use that.
                                         audio_src = args.mu if args.mu else source_vid
                                         
                                         cmd_remix = [
                                            "ffmpeg", "-y",
                                            "-i", str(blended_vid),
                                            "-i", str(audio_src),
                                            "-map", "0:v", "-map", "1:a?", # Use audio from source 
                                            "-c:v", "copy", "-shortest",
                                            str(final_master)
                                         ]
                                         subprocess.run(cmd_remix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                         logging.info(f"   🏆 MASTER RELEASE: {final_master}")
                                 else:
                                     logging.warning("   ⚠️ Blending failed.")
                             except Exception as e_blend:
                                 logging.error(f"   Blend/Remux failed: {e_blend}")

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
            fps=args.fps if args.fps else 24.0, 
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

# -----------------------------------------------------------------------------
# UNICODE VIDEO REFACTOR (Direct Image -> Unicode)
# -----------------------------------------------------------------------------

CELL_W = 12
CELL_H = 22 # Standard terminal aspect ratio

def load_multi_font_unicode():
    """
    Load multiple fonts for complete Unicode coverage.
    Returns a FontSet with symbol font and emoji font.
    """
    from PIL import ImageFont
    font_size = 18
    fonts = {"symbol": None, "emoji": None, "fallback": None}
    
    # Load symbol font (for braille, math, music, arrows, etc.)
    symbol_paths = [
        "/System/Library/Fonts/Apple Symbols.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
    ]
    for p in symbol_paths:
        if os.path.exists(p):
            try:
                fonts["symbol"] = ImageFont.truetype(p, font_size)
                # print(f"   ✓ Symbol font: {os.path.basename(p)}")
                break
            except:
                continue
    
    # Load emoji font (for color emoji)
    emoji_paths = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/Library/Fonts/NotoColorEmoji.ttf",
    ]
    candidate_sizes = [20, 32, 40, 48, 64, 96, 160, 18] # Try known sizes first, then fallback
    for p in emoji_paths:
        if os.path.exists(p):
            for size in candidate_sizes:
                try:
                    f = ImageFont.truetype(p, size)
                    fonts["emoji"] = f
                    # print(f"   ✓ Emoji font: {os.path.basename(p)} @ {size}px")
                    break
                except:
                    continue
            if fonts["emoji"]:
                break
    
    # Fallback font
    if not fonts["fallback"]:
        fonts["fallback"] = ImageFont.load_default()
    
    return fonts

def is_emoji_unicode(char: str) -> bool:
    """Check if a character needs the emoji font."""
    if not char: return False
    cp = ord(char[0])
    # Simplified ranges for performance
    if 0x1F300 <= cp <= 0x1FAFF: return True # Most emojis
    if 0x2600 <= cp <= 0x27BF: return True # Misc symbols/dingbats
    if 0x2B50 == cp: return True # Star
    return False

def image_to_unicode_grid(img: Image.Image, width: int = 64) -> list:
    """
    Converts an image to a grid of (char, r, g, b, br, bg, bb) tuples.
    Uses luminance mapping for characters and full RGB for colors.
    """
    # 1. Resize
    # Calculate height based on aspect ratio + terminal correction
    # Terminal cells are ~0.55 aspect ratio (W/H)
    aspect_ratio = img.height / img.width
    terminal_correction = 0.55
    height = int(width * aspect_ratio * terminal_correction)
    
    # High-quality resize
    small = img.resize((width, height), Image.Resampling.LANCZOS)
    pixels = small.load()
    
    # 2. Lumimance Mapping
    # chars ordered by visual density
    ramp = " .:-=+*#%@"
    
    grid = []
    
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = pixels[x, y][:3]
            
            # Simple luminance
            lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            
            # Index into ramp
            idx = int(lum * (len(ramp) - 1))
            char = ramp[idx]
            
            # Exact background color matching source pixel
            bg_col = (r, g, b)
            
            # Contrasting character color for readability (Option A)
            if lum < 0.5:
                fg = (255, 255, 255) # White text on dark bg
            else:
                fg = (0, 0, 0)       # Black text on bright bg
                
            # Optimization: solid block for max density
            if lum > 0.9:
                char = '█'
                # For solid block, let the block take the actual pixel color
                # since the block covers the background fully anyway
                fg = bg_col
            
            # 7-tuple: char, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b
            row.append((char, fg[0], fg[1], fg[2], bg_col[0], bg_col[1], bg_col[2]))
            
        grid.append(row)
        
    return grid

def render_unicode_frame(grid: list, fonts) -> Image.Image:
    """Renders the grid to a high-res PIL Image."""
    if not grid: return None
    
    rows = len(grid)
    cols = len(grid[0])
    
    img_w = cols * CELL_W
    img_h = rows * CELL_H
    
    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Font Setup
    if isinstance(fonts, dict):
        base_font = fonts.get("symbol") or fonts.get("fallback")
        emoji_font = fonts.get("emoji") or base_font
    else:
        base_font = emoji_font = fonts

    for r_idx, row in enumerate(grid):
        y = r_idx * CELL_H
        for c_idx, cell in enumerate(row):
            x = c_idx * CELL_W
            
            char, r, g, b, br, bg, bb = cell
            
            # Draw Background
            if br > 0 or bg > 0 or bb > 0:
                draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], fill=(br, bg, bb))
            elif char == '█':
                 draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], fill=(r, g, b))
                 continue
            
            if char == ' ' or (r+g+b) < 10: continue
            
            font = emoji_font if is_emoji_unicode(char) else base_font
            
            try:
                draw.text((x, y), char, fill=(r, g, b), font=font)
            except:
                pass
                
    return img

def run_unicode_forge(input_vid: Path, output_vid: Path, width: int = 64, fps_override: float = None) -> bool:
    """
    Core logic for Unicode Video Transformation.
    Extracts frames, renders them as Unicode grid, and stitches them back.
    """
    if not input_vid.exists():
        logging.error(f"❌ Input video not found: {input_vid}")
        return False
        
    # Use parent dict for temp output to keep it somewhat local/clean
    # Or use a dedicated temp folder side-by-side
    temp_root = input_vid.parent / f"temp_unicode_{input_vid.stem}_{int(time.time())}"
    frames_dir = temp_root / "raw_frames"
    rendered_dir = temp_root / "rendered_frames"
    ensure_dir(frames_dir)
    ensure_dir(rendered_dir)
    
    logging.info(f"🔨 Unicode Forge: {input_vid.name} -> {output_vid.name}")
    logging.info(f"   Temp Workdir: {temp_root}")
    
    # 1. Extract Frames
    logging.info("   🎞️  Extracting frames...")
    # Use higher quality extraction?
    cmd_extract = [
        "ffmpeg", "-i", str(input_vid),
        str(frames_dir / "frame_%05d.png")
    ]
    try:
        subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.error(f"Frame extraction failed: {e}")
        return False
    
    # 2. Analyze Audio/FPS
    fps = 24.0
    if fps_override:
        fps = fps_override
    else:
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_vid)
            ]).decode().strip()
            num, den = map(int, probe.split('/'))
            fps = num / den
            logging.info(f"   Detected Source FPS: {fps:.2f}")
        except:
            logging.warning("   FPS detection failed. Defaulting to 24.")
    
    # 3. Initialize Fonts
    logging.info("   🅰️  Loading Fonts...")
    fonts = load_multi_font_unicode()
    
    # 4. Render Loop
    frame_files = sorted(list(frames_dir.glob("frame_*.png")))
    total = len(frame_files)
    
    logging.info(f"   🎨 Rendering {total} frames as Unicode...")
    
    from PIL import Image
    
    for i, fpath in enumerate(frame_files):
        try:
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                
                # Convert
                grid = image_to_unicode_grid(img, width=width)
                
                # Render
                out_img = render_unicode_frame(grid, fonts)
                
                # Save
                out_path = rendered_dir / fpath.name
                out_img.save(out_path)
                
            # VERBOSE LOGGING
            if i % 10 == 0 or i == total - 1:
                logging.info(f"      Frame {i+1}/{total} ({(i+1)/total*100:.1f}%) processed.")
                
        except Exception as e:
            logging.error(f"Failed frame {i}: {e}")
            
    print("")
    
    # 5. Stitch Video
    logging.info("   🧵 Stitching Unicode Video...")
    
    # Check if target exists
    if output_vid.exists(): output_vid.unlink()
    
    cmd_stitch = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(rendered_dir / "frame_%05d.png"),
        # Note: We do NOT map audio here by default, as usage might differ.
        # But for 'forge', we usually want a clean video stream to then mix.
        # However, to save steps, let's mix if input has audio? 
        # Better to keep 'forge' as visual transformation only, and let caller handle audio muxing.
        # BUT, process_cartoon_video_unicode wants audio.
        # Let's map audio if present in Input.
        "-i", str(input_vid),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_vid)
    ]
    
    try:
        subprocess.run(cmd_stitch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"✅ Forge Complete: {output_vid}")
        
        # Cleanup
        try:
             import shutil
             shutil.rmtree(temp_root)
             logging.info("   🧹 Cleanup complete.")
        except: pass
        
        return True
    except Exception as e:
        logging.error(f"Stitching failed: {e}")
        return False

def process_cartoon_video_unicode(args):
    """
    Main entry point for 'cartoon-video' refactor.
    Converts input video -> Unicode Animation -> Output Video.
    """
    if not args.mu or not os.path.exists(args.mu):
        logging.error("❌ Input video (--mu) required for cartoon-video mode.")
        return

    input_vid = Path(args.mu)
    project_name = f"unicode_{input_vid.stem}_{int(time.time())}"
    output_dir = Path("z_test-outputs") / "cartoons" / project_name
    ensure_dir(output_dir)
    
    final_vid = output_dir / f"{input_vid.stem}_unicode.mp4"
    
    target_cw = 64 # Default
    if args.cw:
        target_cw = int(args.cw)
    elif args.w:
        target_cw = int(int(args.w) / 2)
        
    success = run_unicode_forge(input_vid, final_vid, width=target_cw)
    
    if success:
        logging.info(f"🚀 Cartoon Video Complete: {final_vid}")
    else:
        logging.error("❌ Cartoon Video Failed.")


def main():
    parser = argparse.ArgumentParser(description="Cartoon Producer v1.1: The Creative Agency")
    
    # CLI Polish: Positional Args for Aliases
    definitions.add_global_vpform_args(parser)

    parser.add_argument("--vpform", type=str, default=None, help="VP Form (default: creative-agency)") # Default via logic
    parser.add_argument("--tf", type=Path, default=DEFAULT_TF, help="Transcript Folder (Source)")
    parser.add_argument("--vf", type=Path, default=DEFAULT_VF, help="Video Folder (Corpus)")
    parser.add_argument("--fps", type=float, default=0.0, help="Expansion Factor (FBF) or Output FPS (Legacy). Default: Auto-calculated if 0")
    parser.add_argument("--fsync", type=float, default=1.0, help="FPS Sync Multiplier (0.1 - 6.0). Scales auto-calculated FPS.")
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
    parser.add_argument("--style", type=str, default="high resolution 4K UHD video", help="Visual Style Definition")
    
    # Sanitization Helper
    def sanitize_arg(val):
        if not val: return val
        for q in ['"', "'", "“", "”", "‘", "’"]:
             val = val.strip(q)
        return val
    parser.add_argument("--slength", type=float, default=60.0, help="Target length in seconds (if no music)")
    parser.add_argument("--cs", type=int, default=0, choices=[0, 1, 2, 3], help="Chaos Seeds Level (0=None, 0-3)")
    parser.add_argument("--bpm", type=float, help="Manual BPM override for music modes (bypasses detection)")
    parser.add_argument("--pg", action="store_true", help="Enable PG Mode (Relaxed Celebrity/Strict Child Safety)")
    
    # v1.2 FC Integration
    parser.add_argument("--vspeed", type=float, default=None, help="Visualizer Speed (FPS) for music-agency. Default 8. Supports 2, 4, 16.")
    parser.add_argument("--fc", action="store_true", help="Enable Frame & Canvas (Code Painter) Mode")
    parser.add_argument("--retcon", action="store_true", help="Retcon Mode: Rewrite the script/beats of the input XML")
    parser.add_argument("--wan", action="store_true", help="Use Wan 2.1 Keyframe Animation workflow (Local Only)")
    parser.add_argument("--kid", type=int, default=512, help="Keyframe Init Dimension (default: 512). Higher = Better composition before downscale.")
    parser.add_argument("--local", action="store_true", help="Local Mode (Use Gemma + Flux)") # NEW
    parser.add_argument("--cloud", action="store_true", help="Cloud Mode (Use HF Endpoints / Gemini)") # Explicit Cloud Flag
    parser.add_argument("--w", type=int, help="Override width (e.g. 1920)")
    parser.add_argument("--h", type=int, help="Override height (e.g. 1080)")
    parser.add_argument("--strength", type=int, default=50, help="Img2Img noise strength 1-99 (Default: 50). Lower = MORE frame coherence. 10-30: very stable. 30-50: balanced. 50-80: creative/different.")
    parser.add_argument("--blend", type=int, default=66, help="Unicode Overlay Opacity 0-100 (Default: 66). 0 = Skip Unicode Redraw.")
    parser.add_argument("--cw", type=int, help="Unicode Character Width (Grid Width). Default: w/2 or 64.")
    
    args, unknown = parser.parse_known_args()

    # Sniff for positional video file (cartoon-video)
    if not args.mu:
        for u in unknown:
            if u.lower().endswith(('.mp4', '.mov', '.avi')):
                args.mu = u
                logging.info(f"   🎥 Found input video in args: {args.mu}")
                break

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

    # --- UNICODE VIDEO INTERCEPT ---
    if args.vpform == "cartoon-video":
        process_cartoon_video_unicode(args)
        return
    # -------------------------------

    # CLI Polish: Inferred Prompt from Positional Args
    # If explicit --prompt is missing, check if we have any 'orphan' string in cli_args 
    # that wasn't used as a VPForm alias.
    if not args.prompt and getattr(args, "cli_args", None):
        orphans = []
        for val in args.cli_args:
            if val.lower() == "run": continue
            # If it's not a known alias/form, assume it's the prompt
            if not definitions.resolve_vpform(val):
                orphans.append(val)
        
        if orphans:
            args.prompt = " ".join(orphans)
            logging.info(f"   📝 Inferred Prompt from args: '{args.prompt}'")

    # Sanitize inputs
    args.style = sanitize_arg(args.style)
    if args.prompt: args.prompt = sanitize_arg(args.prompt)

    # Auto-Carbonation (Sassprilla)
    # Check for "Song Title Case" prompts (Short, Title Case, No Periods)
    if args.prompt:
        # Sanitization: Strip quotes (Standard & Smart) just in case shell passed them
        for q in ['"', "'", "“", "”", "‘", "’"]:
             args.prompt = args.prompt.strip(q)
        
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
                
            # Switch Image -> Flux (Upgrade to Klein for Quality)
            definitions.set_active_model(Modality.IMAGE, "flux-klein")
            # Switch Video -> LTX (Future proofing)
            definitions.set_active_model(Modality.VIDEO, "ltx-video")
        except Exception as e:
            logging.error(f"❌ Failed to switch to Local Mode: {e}")
    
    # Cloud Mode: Force Gemini API for text
    elif args.cloud:
        logging.info("☁️  Cloud Mode Requested. Using Gemini API for text generation...")
        os.environ["TEXT_ENGINE"] = "gemini_cloud"
            
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
    elif args.vpform in ["creative-agency", "music-visualizer", "music-agency", "music-video", "full-movie", "ansi-video", "ansi-redraw"]:
        # Only switch to Gemini 2.0 (Director Mode) if NOT local.
        # Local mode uses Flux directly.
        if not args.local:
            model = "gemini-2.5-flash-image"
            logging.info(f"   🎨 {args.vpform}: Using Gemini 2.5 Flash Image (Native Img2Img).")
        else:
            logging.info(f"   🎨 {args.vpform}: Using Local Pipeline (Gemma Director -> Flux Renderer).")
        
    logging.info(f"   Model: {model}")
    
    # Scan (Only needed for legacy FBF/Transcript mode)
    if args.vpform in ["creative-agency", "music-visualizer", "music-agency", "music-video", "full-movie", "cartoon-video", "ansi-video", "ansi-redraw"] or args.xb:
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
