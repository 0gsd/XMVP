#!/usr/bin/env python3
"""
Algo-Rhythm Engine (algo_rhythm.py)
Generates high-resolution Unicode video grids driven by Gemini 2.0 Scene Plans (JSON)
Synced to beat metadata from an input audio track. No Diffusion API calls used.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

import librosa
import numpy as np
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SKILL_PATH = os.path.join(Path.home(), ".gemini/antigravity/skills/algo-rhythm/SKILL.md")

# Default Grid-to-Pixel scaling
CELL_W = 10
CELL_H = 10

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)

def analyze_audio_simple(audio_path):
    """Fallback audio analysis using librosa to get BPM and track length"""
    logging.info(f"Analyzing audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo
    
    # Very basic sectioning -> split into roughly 16-bar intervals (for 4/4) assuming a basic structure.
    sections = []
    bars = int(duration / (60/bpm) / 4)
    sec_names = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
    
    current_time = 0.0
    section_len = duration / max(len(sec_names), bars / 4) # Simple heuristic divider
    
    for i in range(len(sec_names)):
        if current_time >= duration: break
        end_time = min(duration, current_time + section_len)
        sections.append({
            "name": sec_names[i],
            "start_sec": current_time,
            "end_sec": end_time
        })
        current_time = end_time
        
    return bpm, duration, sections

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from mvp_shared import load_api_keys
            keys = load_api_keys()
            api_key = keys[0] if keys else None
        except ImportError:
            try:
                from cartoon_producer import load_keys
                env_path = os.path.join(os.path.dirname(__file__), "env_vars.yaml")
                keys = load_keys(env_path) if os.path.exists(env_path) else []
                api_key = keys[0] if keys else None
            except Exception as e:
                logging.warning(f"Error loading keys: {e}")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)

def generate_logical_keyframes(bpm, duration_sec, sections, args):
    client = get_gemini_client()
    
    skill_text = ""
    if os.path.exists(SKILL_PATH):
        with open(SKILL_PATH, 'r') as f:
            skill_text = f.read()

    # Calculate logical grid dimensions
    # We maintain aspect ratio of target w/h but keep it small enough for LLM
    target_aspect = args.w / args.h
    log_w = 60
    log_h = int(60 / target_aspect)
    
    # Approx 2-4 keyframes per section depending on length
    keyframes_target = []
    
    for sec in sections:
        # One at start of section
        keyframes_target.append(sec["start_sec"])
        # One in middle
        mid = sec["start_sec"] + (sec["end_sec"] - sec["start_sec"]) / 2
        keyframes_target.append(mid)
    
    # Cap total keyframes per batch to avoid blowing up context window
    MAX_KF_PER_BATCH = 15
    batches = [keyframes_target[i:i + MAX_KF_PER_BATCH] for i in range(0, len(keyframes_target), MAX_KF_PER_BATCH)]
    
    all_keyframes = []

    schema = {
        "type": "object",
        "properties": {
            "keyframes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "time_sec": {"type": "number"},
                        "section": {"type": "string"},
                        "px": {"type": "string", "description": f"Compact hex string of {log_w*log_h} pixels"}
                    },
                    "required": ["time_sec", "section", "px"]
                }
            }
        },
        "required": ["keyframes"]
    }

    for b_idx, batch_times in enumerate(batches):
        logging.info(f"Generating Logical Keyframes Batch {b_idx+1}/{len(batches)} (Frames: {len(batch_times)})...")
        
        prompt = f"""You are the ALGO-RHYTHM Engine.
{skill_text}

USER PROMPT: {args.prompt}
STYLE: {args.style}

AUDIO INFO:
BPM: {bpm:.2f}
Duration: {duration_sec:.2f}s
Sections: {json.dumps(sections, indent=2)}

LOGICAL GRID: {log_w}x{log_h} (Total: {log_w*log_h} exact pixels)

TASK: Provide exact pixel maps for the following timestamps (seconds) in the song:
{json.dumps(batch_times)}

Draw scenes matching the prompt and style. Use color carefully to build shape and form.
Do not return black frames! Even backgrounds should have deep hues.
The 'px' field MUST be EXACTLY {log_w*log_h * 6} characters long, containing raw hex triplets (RRGGBB) spanning row-by-row left-to-right.
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.4
                )
            )
            batch_data = json.loads(response.text)
            all_keyframes.extend(batch_data.get("keyframes", []))
        except Exception as e:
            logging.error(f"Error generating batch {b_idx+1}: {e}")
            break
            
    # Sort them just to be sure
    all_keyframes.sort(key=lambda x: x["time_sec"])

    return {
        "song": {
            "fps": 8.0, 
            "total_frames": int(duration_sec * 8.0),
            "bpm": bpm
        },
        "logical_grid": [log_w, log_h],
        "target_grid": [args.w, args.h],
        "keyframes": all_keyframes
    }

def interpolate_color(c1, c2, t):
    """Interpolate between two RGB tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def render_video(args, scene_plans, bpm, out_video_path):
    w, h = args.w, args.h
    vid_w, vid_h = w * CELL_W, h * CELL_H
    
    fps = scene_plans.get("song", {}).get("fps", 8.0)
    import librosa
    dur = librosa.get_duration(path=args.mu)
    total_frames = scene_plans.get("song", {}).get("total_frames", int(fps * dur))
    
    log_w, log_h = scene_plans.get("logical_grid", [60, 40])
    keyframes = scene_plans.get("keyframes", [])
    
    if not keyframes:
        logging.error("No keyframes found in logical plan to render!")
        return
        
    # Pre-parse hex strings into RGB arrays for faster access
    for kf in keyframes:
        px_str = kf.get("px", "")
        # Fallback if LLM broke string length
        expected_len = log_w * log_h * 6
        if len(px_str) < expected_len:
            logging.warning(f"Keyframe at {kf['time_sec']}s has short px string! Padding with black.")
            px_str = px_str.ljust(expected_len, '0')
            
        kf["rgb_array"] = []
        for i in range(0, log_w * log_h * 6, 6):
            if i+6 <= len(px_str):
                c = px_str[i:i+6]
                try:
                    kf["rgb_array"].append(hex_to_rgb(c))
                except Exception:
                    kf["rgb_array"].append((0,0,0))
                    
    beat_interval = 60.0 / bpm
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f"{vid_w}x{vid_h}",
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-i', args.mu,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        out_video_path
    ]
    
    logging.info(f"Starting FFMPEG pipe to generate {out_video_path} at {vid_w}x{vid_h}, {fps} FPS")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    scale_x = w / log_w
    scale_y = h / log_h
    
    for frame in range(total_frames):
        t = frame / fps
        
        # Determine Current Beat
        beat = (t / beat_interval)
        beat_phase = beat % 1.0
        on_beat = beat_phase < 0.15 # 15% pulse window
        
        # Find current and next keyframes for interpolation
        kf_a = keyframes[0]
        kf_b = keyframes[-1]
        
        for i in range(len(keyframes) - 1):
            if keyframes[i]["time_sec"] <= t < keyframes[i+1]["time_sec"]:
                kf_a = keyframes[i]
                kf_b = keyframes[i+1]
                break
                
        # Time interpolation factor between keyframes
        lerp_t = 0.0
        if kf_b["time_sec"] > kf_a["time_sec"]:
            lerp_t = (t - kf_a["time_sec"]) / (kf_b["time_sec"] - kf_a["time_sec"])
            
        img = Image.new('RGB', (vid_w, vid_h), color=(0,0,0))
        draw = ImageDraw.Draw(img)
        
        arr_a = kf_a.get("rgb_array", [])
        arr_b = kf_b.get("rgb_array", [])
        
        # Render the logical grid upscale
        for ly in range(log_h):
            for lx in range(log_w):
                idx = ly * log_w + lx
                
                c1 = arr_a[idx] if idx < len(arr_a) else (0,0,0)
                c2 = arr_b[idx] if idx < len(arr_b) else (0,0,0)
                
                color = interpolate_color(c1, c2, lerp_t)
                
                if on_beat:
                    # Flash brighter
                    color = tuple(min(255, c + 30) for c in color)
                
                # Upscale to Target Grid
                start_x = int(lx * scale_x)
                end_x = int((lx + 1) * scale_x)
                start_y = int(ly * scale_y)
                end_y = int((ly + 1) * scale_y)
                
                # We can draw chunks as rectangles rather than string characters
                # Because the LLM generated literal pixels, characters don't make sense anymore!
                px_sx = start_x * CELL_W
                px_ex = end_x * CELL_W
                px_sy = start_y * CELL_H
                px_ey = end_y * CELL_H
                
                draw.rectangle([px_sx, px_sy, px_ex, px_ey], fill=color)
                        
        process.stdin.write(img.tobytes())
        if frame % int(fps*2) == 0:
            logging.info(f"Rendered [{frame}/{total_frames}] ({(frame/total_frames)*100:.1f}%)")
            
    process.stdin.close()
    process.wait()
    logging.info(f"Finished rendering {out_video_path}.")

def main():
    p = argparse.ArgumentParser("algo_rhythm", description="Unicode grid painter based on LLM narrative")
    p.add_argument("--prompt", required=True, type=str, help="Overall scene/narrative description")
    p.add_argument("--mu", required=True, type=str, help="Input audio track (.aif, .mp3, .wav)")
    p.add_argument("--style", type=str, default="retro", help="Overall style or color palette cues")
    p.add_argument("--w", type=int, default=480, help="Grid Width in characters")
    p.add_argument("--h", type=int, default=320, help="Grid Height in characters")
    p.add_argument("--fsync", type=float, default=1.0, help="Framerate sync multiplier / base frame duration modifier")
    
    args = p.parse_args()
    
    if not os.path.exists(args.mu):
        logging.error(f"Audio file not found: {args.mu}")
        sys.exit(1)
        
    bpm, dur_sec, sections = analyze_audio_simple(args.mu)
    logging.info(f"Parsed Audio - BPM: {bpm:.1f}, Dur: {dur_sec:.1f}s")
    
    plan_file = "algo_rhythm_plan.json"
    if os.path.exists(plan_file):
        logging.info("Loading existing scene plans...")
        with open(plan_file, "r") as f:
            scene_plans = json.load(f)
    else:
        scene_plans = generate_logical_keyframes(bpm, dur_sec, sections, args)
        with open(plan_file, "w") as f:
            json.dump(scene_plans, f, indent=2)
            
    out_video = os.path.splitext(args.mu)[0] + "_algorhythm.mp4"
    render_video(args, scene_plans, bpm, out_video)

if __name__ == "__main__":
    main()
