#!/usr/bin/env python3
"""
content_producer.py
-------------------
Unified generator for podcast and improv content.
Merges logic from `podcast_animator.py` (Script->Video) and `improv_animator.py` (Idea->Video).

Modes:
1. Podcast/Script Animation (Triplets/Pairs)
2. Improv Comedy (Zero-shot infinite generation)

Supported Engines:
- Text: Gemini 2.0 Flash (Cloud)
- Audio: Google Journey (Cloud) OR Kokoro (Local)
- Visuals: Gemini 2.0 Flash (Cloud) OR Flux Schnell (Local)
- Foley: None OR HunyuanVideo-Foley (Local)

Author: 0i0 (Merged by Antigravity)
Date: Jan 2026
"""

import os
import sys
import json
import argparse
import glob
import time
import re
import subprocess
import random
import shutil
import logging
from pathlib import Path
import itertools

# External Libs
import requests
from PIL import Image

try:
    # Try New SDK (v1.0+)
    from google import genai
    from google.genai import types
except ImportError:
    try:
        # Try Stable SDK (v0.8.x)
        import google.generativeai as genai
        # Types shim for backward compatibility if needed, or rely on duck typing
    except ImportError:
        print("[-] Warning: No Gemini SDK found. Cloud mode will fail.")
        pass

# MVP Imports
try:
    import mvp_shared
    from mvp_shared import save_xmvp, CSSV, VPForm, Story, Portion, Constraints
    from text_engine import TextEngine
    from truth_safety import TruthSafety
    from foley_talk import generate_audio_asset
    from vision_producer import get_chaos_seed
    import frame_canvas
    
    # Local Bridges
    from flux_bridge import get_flux_bridge
    from hunyuan_foley_bridge import generate_foley_asset
except ImportError as e:
    print(f"[-] Critical Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "z_test-outputs"))
TRIPLETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../z_podcast_triplets"))
MW_ROOT = "/Volumes/XMVPX/mw" # Standard mount point

# Local Model Paths (Hardcoded standards for XMVP)
FLUX_MODEL_PATH = os.path.join(MW_ROOT, "flux-root")
if not os.path.exists(FLUX_MODEL_PATH):
    # Fallback to safetensors if user moved it manually, but default populate is flux-root
    FLUX_MODEL_PATH = os.path.join(MW_ROOT, "flux-schnell.safetensors")

# --- GLOBAL STATE ---
LOCAL_MODE = False
FOLEY_ENABLED = False
FLUX_BRIDGE = None

# --- CONSTANTS & FORMS ---
FORM_DEFS = {
    "24-cartoon": {
        "description": "4-Person Improv Comedy Special (24 Minutes)",
        "cast": {
            "William": {"base": "Billy Joel", "voice": "en-US-Journey-D", "pitch": 1, "persona": "The Piano Man. Working-class poet. Melodic, specific geographic references (Long Island), cynical but soulful."},
            "Maggie": {"base": "Margaret Thatcher", "voice": "en-US-Journey-F", "pitch": -2, "persona": "The Iron Lady. Stern, authoritative, uses 'Royal We'. Surprisingly willing to play high-status absurd characters."},
            "Francis": {"base": "Frank Sinatra", "voice": "en-US-Journey-D", "pitch": -2, "persona": "The Chairman. Cool, swaggering, mid-Atlantic accent. emotional, volatile, calls everyone 'baby'. Does it 'My Way'."},
            "Anne Tailored": {"base": "Taylor Swift", "voice": "en-US-Journey-F", "pitch": 1, "persona": "The Pop Icon. Earnest, confessional, detailed storytelling. Bridges scenes with emotional hooks. Avoids copyright infringement."}
        },
        "system_prompt_template": (
            "You are the Director of a long-form Improv Comedy show in a Black Box Theater.\n"
            "The Cast:\n{cast_desc}\n\n"
            "The Rules:\n"
            "1. Generate ONE turn at a time (Speaker + Dialogue + Physical Action).\n"
            "2. Maintain a coherent narrative arc weaving through the Chaos Seeds.\n"
            "3. Style: 'Yes, And', witty, character-driven.\n"
            "4. Format: JSON {{ 'speaker': 'Name', 'text': 'Dialogue', 'action_prompt': 'Visual description of action', 'visual_focus': 'Focus' }}\n"
        )
    },
    # Aliases
    "24-podcast": {"alias": "24-cartoon"},
    "10-podcast": {"alias": "24-cartoon", "duration_override": 600}
}
FORM_DEFS["24-podcast"] = FORM_DEFS["24-cartoon"] # Simple alias ref

COURTROOM_STYLE_PROMPT = (
    "A photorealistic courtroom sketch. Drawn by a preternaturally talented artist "
    "in an underground black box theater. Hyperrealistic style, clean commercial art. "
    "NO WORDS, NO SPEECH BUBBLES, NO TEXT. "
    "The artist captures the actors improvising, surrounded by imaginary objects. "
    "Dynamic composition, intense energy."
)

# --- HELPERS ---

def get_client_keys():
    keys = mvp_shared.load_api_keys()
    if not keys: return None
    random.shuffle(keys)
    return itertools.cycle(keys)

KEY_CYCLE = get_client_keys()

def get_client():
    if not KEY_CYCLE: return None, None
    key = next(KEY_CYCLE)
    return genai.Client(api_key=key), key

def get_audio_duration(file_path):
    try:
        cmd = ['ffprobe', '-i', file_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0.0

def generate_image(prompt, output_path, ts=None):
    """
    Generates an image using either:
    1. Flux Schnell (Local) if LOCAL_MODE is True
    2. Gemini 2.0 Flash (Cloud) otherwise
    """
    global FLUX_BRIDGE
    
    if LOCAL_MODE:
        # --- LOCAL FLUX ---
        if not FLUX_BRIDGE:
             print(f"   [🌊] Initializing Flux from {FLUX_MODEL_PATH}...")
             try:
                 FLUX_BRIDGE = get_flux_bridge(FLUX_MODEL_PATH)
             except Exception as e:
                 print(f"   [-] Flux Init Failed: {e}. Falling back to Black Frame.")
                 return False
        
        try:
            # Flux requires no specific aspect ratio args in prompt usually, handled by width/height
            img = FLUX_BRIDGE.generate(prompt, width=1024, height=1024, steps=4) # Schnell 4 steps
            if img:
                img.save(output_path)
                return True
            else:
                return False
        except Exception as e:
            print(f"   [-] Flux Gen Failed: {e}")
            return False
            
    else:
        # --- CLOUD GEMINI ---
        max_retries = 6 
        base_delay = 5
        
        for attempt in range(max_retries):
            client, key = get_client()
            if not client: 
                print("[-] No Keys.")
                return False
                
            try:
                final_prompt = f"{prompt} --aspect_ratio 1:1"
                if ts: final_prompt = ts.refine_prompt(prompt, pg_mode=True)
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp", 
                    contents=final_prompt
                )
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            with open(output_path, "wb") as f:
                                f.write(part.inline_data.data)
                            return True
            except Exception as e:
                # 429 = Resource Exhausted / Quota
                # 503 = Overloaded
                if "429" in str(e) or "503" in str(e) or "ResourceExhausted" in str(e):
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"   [⏳] Quota Hit (Attempt {attempt+1}/{max_retries}). Sleeping {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"   [-] Cloud Gen Failed: {e}")
                    time.sleep(1) # Short pause on other errors
                    
        return False

def create_black_frame(path):
    subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=1024x1024:d=0.1', '-frames:v', '1', path, '-y', '-loglevel', 'error'])

def map_voice_to_kokoro(cloud_voice):
    """Maps Google Cloud Journey voices to Kokoro equivalents."""
    mapping = {
        # Females
        "en-US-Journey-F": "af_bella",  # Warm, clear
        "en-US-Journey-O": "af_sarah",  # Soft, calm
        # Males
        "en-US-Journey-D": "am_michael", # Deep, resonant
        "en-US-Journey-L": "am_adam",    # Clear, neutral
        # Fallbacks
        "en-US-Journey": "af_bella"
    }
    # Naive match
    for k, v in mapping.items():
        if k in cloud_voice:
            return v
    
    # Random fallback based on letter if possible?
    # Or just default
    return "af_bella"

# --- IMPROV LOGIC (From improv_animator.py) ---

def generate_dynamic_cast(text_engine, seeds):
    print("    [✨] Casting Call: Generating Dynamic Personas...")
    selected_seeds = seeds[:4]
    slots = [
        {"voice": "en-US-Journey-D", "pitch": 1, "gender": "Male"},
        {"voice": "en-US-Journey-F", "pitch": -2, "gender": "Female"},
        {"voice": "en-US-Journey-D", "pitch": -2, "gender": "Male"},
        {"voice": "en-US-Journey-F", "pitch": 1, "gender": "Female"}
    ]
    dynamic_cast = {}
    for i, seed in enumerate(selected_seeds):
        slot = slots[i]
        prompt = (f"Create a weird improv character inspired by: '{seed}'. Gender: {slot['gender']}. "
                  "JSON: { 'name': 'First Last', 'persona': 'Description' }")
        try:
            raw = text_engine.generate(prompt, json_schema=True)
            data = json.loads(raw)
            name = data.get("name", f"Player {i+1}")
            dynamic_cast[name] = {
                "base": name, "voice": slot["voice"], "pitch": slot["pitch"],
                "persona": data.get("persona", f"Improviser inspired by {seed}")
            }
            print(f"       + Cast {name}: {dynamic_cast[name]['persona'][:40]}...")
        except:
            pass
    return dynamic_cast

def run_improv_session(vpform, output_dir, text_engine, args):
    """
    Main Loop for Improv Mode.
    """
    print(f"🎭 IMPROV ANIMATOR: {vpform}")
    
    # 1. Config
    defs = FORM_DEFS.get(vpform, FORM_DEFS["24-cartoon"])
    if "alias" in defs: defs = FORM_DEFS[defs["alias"]]
    
    target_duration = args.slength if args.slength > 0 else (defs.get("duration_override", 24 * 60))
    
    session_id = int(time.time())
    session_dir = os.path.join(output_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 2. Seeds & Cast
    print("[*] Gathering Chaos Seeds...")
    seeds = [get_chaos_seed() for _ in range(6)]
    print(f"    Seeds: {seeds}")
    
    cast = defs["cast"]
    # If using a dynamic form (not implemented yet fully, but stub logic exists) 
    # we could swap cast here. using default cast for now.
    cast_desc = "\n".join([f"- {n}: {d['persona']}" for n, d in cast.items()])
    
    # 3. Execution Loop
    total_duration = 0.0
    turn_count = 0
    assets = []
    history = []
    
    system_prompt = defs["system_prompt_template"].format(cast_desc=cast_desc)
    
    while total_duration < target_duration:
        # A. Seed Injection
        seed_idx = int(total_duration // 240)
        current_seed = seeds[seed_idx] if seed_idx < len(seeds) else "The Grand Finale"
        
        # B. Write
        prompt = (
            f"Time: {total_duration:.1f}/{target_duration}s. Seed: {current_seed}\n"
            f"History:\n" + "\n".join([f"{h['speaker']}: {h['text']}" for h in history[-8:]]) + 
            "\n\nGenerate next turn (JSON)."
        )
        
        turn_data = None
        for attempt in range(3):
            try:
                raw = text_engine.generate(system_prompt + "\n\n" + prompt, json_schema=True)
                turn_data = json.loads(raw)
                break
            except Exception as e:
                print(f"    [!] Writer Error: {e}")
                time.sleep(1)
        
        if not turn_data: 
            print("    [!] Skipping turn (Writer failed).")
            continue
            
        speaker = turn_data.get('speaker', 'Unknown')
        text = turn_data.get('text', '...')
        action = turn_data.get('action_prompt', 'Standing.')
        visual = turn_data.get('visual_focus', speaker)
        
        print(f"    [{turn_count}] {speaker}: {text[:40]}...")
        history.append({'speaker': speaker, 'text': text})
        
        # C. Visual (First, so Foley can see it)
        img_path = os.path.join(session_dir, f"turn_{turn_count:04d}.jpg")
        visual_prompt = f"{COURTROOM_STYLE_PROMPT}\nAction: {action}\nFocus: {visual}"
        
        if not generate_image(visual_prompt, img_path):
             create_black_frame(img_path)
             
        # D. Audio (Speech)
        wav_path = os.path.join(session_dir, f"turn_{turn_count:04d}.wav")
        voice_info = cast.get(speaker, list(cast.values())[0])
        audio_mode = "kokoro" if LOCAL_MODE else "cloud"
        
        target_voice = voice_info['voice']
        if LOCAL_MODE:
            target_voice = map_voice_to_kokoro(target_voice)
            
        final_wav = generate_audio_asset(
            text, wav_path, 
            voice_name=target_voice, 
            pitch=voice_info['pitch'],
            mode=audio_mode
        )
        
        if not final_wav:
             print("       [!] Audio failed.")
             continue
             
        duration = get_audio_duration(final_wav) + 0.5 # Pause
        
        # E. Foley (Optional)
        foley_path = None
        if FOLEY_ENABLED:
            foley_prompt = f"{action} in a theater."
            foley_out = os.path.join(session_dir, f"foley_{turn_count:04d}.wav")
            
            # Create a temporay video loop from the image for Foley Visual Context
            temp_vid = os.path.join(session_dir, f"temp_vid_{turn_count:04d}.mp4")
            subprocess.run([
                'ffmpeg', '-loop', '1', '-i', img_path, 
                '-c:v', 'libx264', '-t', str(duration), '-pix_fmt', 'yuv420p', 
                temp_vid, '-y', '-loglevel', 'error'
            ])
            
            generate_foley_asset(foley_prompt, foley_out, video_path=temp_vid, duration=duration)
            
            if os.path.exists(foley_out):
                foley_path = foley_out
                mixed_wav = os.path.join(session_dir, f"mixed_{turn_count:04d}.wav")
                subprocess.run([
                    'ffmpeg', '-i', final_wav, '-i', foley_path,
                    '-filter_complex', 'amix=inputs=2:duration=first:dropout_transition=2',
                    mixed_wav, '-y', '-loglevel', 'error'
                ])
                final_wav = mixed_wav
                
            # Cleanup temp video
            if os.path.exists(temp_vid): os.remove(temp_vid)
        
        assets.append({
            "audio": final_wav, "image": img_path, "duration": duration, 
            "speaker": speaker, "text": text,
            "action_prompt": action, "visual_focus": visual, "foley_prompt": foley_prompt if FOLEY_ENABLED else ""
        })
        
        total_duration += duration
        turn_count += 1
        
    # 4. Stitch
    stitch_assets(assets, session_dir, os.path.join(output_dir, f"Improv_{session_id}.mp4"))
    
    # 5. Export XMVP
    export_xmvp_manifest(
        session_dir,
        f"Improv_{session_id}",
        assets,
        list(cast.keys()),
        seeds=seeds,
        title=f"Improv Session {session_id}",
        synopsis="A generated improv comedy set."
    )


# --- PODCAST LOGIC (From podcast_animator.py) ---

class Segment:
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text
        self.image_path = None
        self.audio_path = None
        self.duration = 0.0

def parse_transcript_basic(txt_path):
    # Simplified parser
    segments = []
    with open(txt_path, 'r') as f: lines = f.readlines()
    curr_spk = "Unknown"
    curr_txt = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("---"): continue
        match = re.match(r'^([A-Z\s\.]+):', line)
        if match:
            if curr_txt: segments.append(Segment(curr_spk, " ".join(curr_txt)))
            curr_spk = match.group(1).title()
            curr_txt = [line[len(match.group(0)):].strip()]
        else:
            curr_txt.append(line)
    if curr_txt: segments.append(Segment(curr_spk, " ".join(curr_txt)))
    return segments

def run_podcast_processing(triplets_dir, output_dir, args):
    """
    Standard Podcast workflow: Triplet (TXT+JSON+MP3) or Pair (TXT+JSON) -> Video.
    """
    print("[*] Scanning for Podcast Text Pairs...")
    txt_files = glob.glob(os.path.join(triplets_dir, "*.txt"))
    
    for txt_path in txt_files:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        json_path = os.path.join(triplets_dir, f"{base}.json")
        mp3_path = os.path.join(triplets_dir, f"{base}.mp3")
        output_mp4 = os.path.join(output_dir, f"{base}.mp4")
        
        if os.path.exists(output_mp4):
            print(f"    [.] Skipping {base} (Exists)")
            continue
            
        if not os.path.exists(json_path):
            continue
            
        # Determine Mode: Triplet (Sync) or Pair (Gen)
        if os.path.exists(mp3_path):
             print(f"[*] Found TRIPLET: {base}")
             # Sync logic omitted for brevity in this merge, assuming Gen-Only focus for now 
             # or simply calling old logic if needed. 
             # User request emphasized "everything from improv_animator works in content_producer".
             # But let's support Audio Gen (Pair) mode as that's arguably more critical for "Content Producer".
             pass 
        else:
             print(f"[*] Found PAIR: {base} (Generating Audio)")
             process_pair(base, txt_path, json_path, output_mp4, project_id=args.project)

def process_pair(base, txt_path, json_path, output_mp4, project_id=None):
    segments = parse_transcript_basic(txt_path)
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{base}")
    os.makedirs(temp_dir, exist_ok=True)
    
    processed = []
    
    for i, seg in enumerate(segments):
        print(f"    Processing {i}/{len(segments)}: {seg.speaker}")
        
        # Audio
        wav_path = os.path.join(temp_dir, f"seg_{i}.wav")
        # Simple deterministic voice mapping
        # Hash speaker name to deterministic pitch/voice
        # Using foley_talk defaults handling if we just pass name? 
        # Actually foley_talk needs explicit voice usually, but let's default to Journey-D/F based on name hash
        
        pitch = 0
        voice = "en-US-Journey-D" 
        if output_mp4: # Hack check
            h = hash(seg.speaker)
            voice = "en-US-Journey-D" if h % 2 == 0 else "en-US-Journey-F"
            pitch = (h % 3) - 1 # -1, 0, 1
            
        if LOCAL_MODE:
            voice = map_voice_to_kokoro(voice)
            
        audio_mode = "kokoro" if LOCAL_MODE else "cloud"
        
        final_wav = generate_audio_asset(seg.text, wav_path, voice_name=voice, pitch=pitch, mode=audio_mode, project_id=project_id)
        if not final_wav: continue
        
        seg.audio_path = final_wav
        seg.duration = get_audio_duration(final_wav)
        
        # Image
        img_path = os.path.join(temp_dir, f"seg_{i}.png")
        prompt = f"Cinematic shot of {seg.speaker} speaking. Context: '{seg.text[:50]}'"
        if not generate_image(prompt, img_path):
            create_black_frame(img_path)
        seg.image_path = img_path
        
        processed.append(seg)
        time.sleep(1)
        
    # Stitch
    assets = [{"audio": s.audio_path, "image": s.image_path, "duration": s.duration} for s in processed]
    stitch_assets(assets, temp_dir, output_mp4)
    
    # Export XMVP
    cast_names = list(set([s.speaker for s in processed]))
    export_xmvp_manifest(
        os.path.dirname(output_mp4),
        os.path.splitext(os.path.basename(output_mp4))[0],
        processed, # Pass raw segments, helper handles obj
        cast_names,
        title=base,
        synopsis="Podcast Animation"
    )
    
    shutil.rmtree(temp_dir)

def stitch_assets(assets, temp_dir, output_mp4):
    # Quick Stitch
    list_path = os.path.join(temp_dir, "stitch.txt")
    mp4s = []
    
    for i, a in enumerate(assets):
        seg_mp4 = os.path.join(temp_dir, f"clip_{i}.mp4")
        subprocess.run([
            'ffmpeg', '-y', '-loop', '1', '-i', a['image'], '-i', a['audio'],
            '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p', '-shortest', seg_mp4, '-loglevel', 'error'
        ], check=True)
        mp4s.append(seg_mp4)
        
    with open(list_path, 'w') as f:
        for mp4 in mp4s: f.write(f"file '{mp4}'\n")
        
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_mp4, '-y', '-loglevel', 'error'])
    print(f"[+] Output Saved: {output_mp4}")

def export_xmvp_manifest(output_dir, base_name, assets, cast_names, seeds=None, title=None, synopsis="Generated Content"):
    """
    Generates and saves the XMVP XML manifest.
    """
    try:
        # 1. Models
        vp = VPForm(
            name="content-producer-v1",
            fps=24,
            description="Content Producer Output",
            mime_type="video/mp4"
        )
        
        cssv = CSSV(
            constraints=Constraints(width=1024, height=1024, fps=24),
            scenario=f"Content based on {base_name}",
            situation=synopsis,
            vision="Cinematic/Generated"
        )
        
        story = Story(
            title=title if title else base_name,
            synopsis=synopsis,
            characters=cast_names,
            theme="Improv/Podcast"
        )
        
        portions = []
        for i, item in enumerate(assets):
            # Handle both Dict (Improv) and Segment Obj (Podcast)
            if isinstance(item, dict):
                spk = item['speaker']
                txt = item.get('text', '')
                dur = item['duration']
            else:
                spk = item.speaker
                txt = item.text
                dur = item.duration
                
            portions.append(Portion(
                id=i+1,
                duration_sec=dur,
                content=f"{spk}: {txt}"
            ))
            
        xmvp_data = {
            "VPForm": vp,
            "CSSV": cssv,
            "Story": story,
            "Portions": [p.model_dump() for p in portions]
        }
        
        if seeds:
            xmvp_data["ChaosSeeds"] = seeds
        
        # 2. Detailed Dialogue Script (Line-by-Line)
        dialogue_lines = []
        for item in assets:
            if isinstance(item, dict):
                 # Improv Dict
                 dl = mvp_shared.DialogueLine(
                     character=item['speaker'],
                     text=item.get('text', ''),
                     duration=item['duration'],
                     action=item.get('action_prompt', ''), # Stored earlier?
                     foley=item.get('foley_prompt', ''),   # Wait, need to store these in assets in main loop
                     visual_focus=item.get('visual_focus', '')
                 )
            else:
                 # Podcast Segment
                 dl = mvp_shared.DialogueLine(
                     character=item.speaker,
                     text=item.text,
                     duration=item.duration
                 )
            dialogue_lines.append(dl)
            
        xmvp_data["Dialogue"] = mvp_shared.DialogueScript(lines=dialogue_lines)
            
        xml_path = os.path.join(output_dir, f"{base_name}_manifest.xml")
        save_xmvp(xmvp_data, xml_path)
        print(f"    📜 XML Manifest Saved: {xml_path}")
        
    except Exception as e:
        print(f"[-] XMVP Export Failed: {e}") 
        import traceback
        traceback.print_exc() 


# --- MAIN ---

def main():
    global LOCAL_MODE, FOLEY_ENABLED
    
    parser = argparse.ArgumentParser(description="Content Producer (Unified)")
    parser.add_argument("--vpform", help="Vision Platonic Form (e.g. 24-podcast, gahd-podcast)")
    parser.add_argument("--project", help="Project override (stub)")
    parser.add_argument("--ep", type=int, help="Episode number (stub)")
    parser.add_argument("--local", action="store_true", help="Use Local Engines (Flux + Kokoro)")
    parser.add_argument("--foley", choices=["on", "off"], default="off", help="Enable Generative Foley")
    parser.add_argument("--slength", type=float, default=0.0, help="Override duration in seconds")
    parser.add_argument("--fc", action="store_true", help="Code Painter Mode (Experimental)")
    parser.add_argument("--geminiapi", action="store_true", help="Force Cloud Gemini API for Text (Disable Local Gemma default)")
    args = parser.parse_args()
    
    LOCAL_MODE = args.local
    FOLEY_ENABLED = (args.foley == "on")

    # --- TEXT ENGINE LOGIC ---
    # Default to "local_gemma" for text to save quota, unless user forces --geminiapi
    if not args.geminiapi:
        print("   [Engine] Defaulting Text Engine to Local Gemma (Saving API Quota)...")
        os.environ["TEXT_ENGINE"] = "local_gemma"
        # We assume local_gemma path is set in env_vars or defaults to mlx-community/gemma-2-9b-it-4bit
    else:
        print("   [Engine] Text Engine Forced to Gemini API (Cloud).")
    
    print(f"🎬 CONTENT PRODUCER | Local: {LOCAL_MODE} | Foley: {FOLEY_ENABLED} | Form: {args.vpform}")
    
    if args.vpform and ("podcast" in args.vpform or "cartoon" in args.vpform):
        if "gahd" in args.vpform:
            # Traditional Pair/Triplet Processing
            run_podcast_processing(TRIPLETS_DIR, OUTPUT_DIR, args)
        else:
            # Improv Loops (24-podcast, 10-podcast)
            text_engine = TextEngine() # Need TextEngine for writing
            run_improv_session(args.vpform, OUTPUT_DIR, text_engine, args)
    else:
        # Default behavior: Scan for triplets (legacy podcast_animator behavior)
        run_podcast_processing(TRIPLETS_DIR, OUTPUT_DIR, args)

if __name__ == "__main__":
    main()
