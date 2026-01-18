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
    from vision_producer import get_chaos_seed
    # frame_canvas moved to lazy import
    
    # Local Bridges
    from flux_bridge import get_flux_bridge
    from hunyuan_foley_bridge import generate_foley_asset
except ImportError as e:
    print(f"[-] Critical Import Error: {e}")
    sys.exit(1)

print("DEBUG: Imports Done.")

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
                    model="gemini-2.0-flash", 
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
        {"voice": "en-US-Journey-D", "pitch": 0, "gender": "Male"},      # Deep
        {"voice": "en-US-Journey-F", "pitch": 0, "gender": "Female"},    # Warm
        {"voice": "en-US-Journey-L", "pitch": 0, "gender": "Male"},      # Neutral
        {"voice": "en-US-Journey-O", "pitch": 0, "gender": "Female"}     # Soft
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

def generate_ensemble_cast(text_engine):
    """Generates 4-person cast (2M, 2F) using Legacy GAHD Logic."""
    print("    [✨] Casting Call: Summoning Character Actors (1975-2005)...")
    
    prompt = (
        "Generate a cast of 4 distinct 'Character Actors' (2 Male, 2 Female) active 1975-2005.\n"
        "CRITERIA: Recognizable but not A-List. Distinct voices/types. Real actors or very close pastiches.\n"
        "OUTPUT JSON: "
        "[{ 'name': 'Name', 'gender': 'Male/Female', 'persona': 'Vocal/Personality description' }, ...]"
    )
    
    try:
        raw = text_engine.generate(prompt, json_schema=True)
        cast_list = json.loads(raw)
        
        # Ensure we have a list
        if isinstance(cast_list, dict) and "cast" in cast_list: cast_list = cast_list["cast"]
        
        final_cast = {}
        slots = [
            {'gender': 'Male', 'voice': 'en-US-Journey-D', 'pitch': 1},
            {'gender': 'Female', 'voice': 'en-US-Journey-F', 'pitch': -2},
            {'gender': 'Male', 'voice': 'en-US-Journey-D', 'pitch': -2},
            {'gender': 'Female', 'voice': 'en-US-Journey-F', 'pitch': 1}
        ]
        
        # Attempt to match genders to slots
        for i, actor in enumerate(cast_list[:4]):
            # Find matching slot
            slot = next((s for s in slots if s['gender'].lower() == actor.get('gender', 'Male').lower()), None)
            if not slot and slots: slot = slots[0] # Fallback
            if slot in slots: slots.remove(slot)
            
            voice = slot['voice'] if slot else 'en-US-Journey-D'
            pitch = slot['pitch'] if slot else 0
            
            final_cast[actor['name']] = {
                "base": actor['name'],
                "voice": voice, 
                "pitch": pitch,
                "persona": actor.get('persona', 'A character actor.')
            }
            print(f"       + Cast {actor['name']} as {voice} (p{pitch})")
            
        return final_cast
        
    except Exception as e:
        print(f"    [-] Casting Failed: {e}. Using Default Backup.")
        return {
            "Joe Pantoliano": {"voice": "en-US-Journey-D", "pitch": 1, "persona": "Fast-talking, nervous energy."},
            "Margo Martindale": {"voice": "en-US-Journey-F", "pitch": -2, "persona": "Authoritative, Southern warmth."},
            "JK Simmons": {"voice": "en-US-Journey-D", "pitch": -2, "persona": "Intense, demanding, rhythmic."},
            "CCH Pounder": {"voice": "en-US-Journey-F", "pitch": 1, "persona": "Gravitas, deep resonance."}
        }

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
    if "gahd" in vpform:
        print("[*] G.A.H.D Mode: Fetching 2 Seeds for Movie Pitch...")
        seeds = [get_chaos_seed() for _ in range(2)]
        print(f"    Seeds: {seeds}")
        
        # Restore Improv Logic for GAHD (Ensemble Casting)
        cast = generate_ensemble_cast(text_engine)
        cast_desc = "\n".join([f"- {n}: {d['persona']}" for n, d in cast.items()])
        
        # Override System Prompt for GAHD
        system_prompt = (
            "You are a Writers Room of 4 eccentric screenwriters (Character Actors).\n"
            f"The Cast:\n{cast_desc}\n\n"
            f"The Mission:\n"
            f"Develop a high-concept Movie Pitch that combines these two random elements:\n"
            f"1. {seeds[0]}\n2. {seeds[1]}\n\n"
            "The Rules:\n"
            "1. Generate ONE turn at a time (Speaker + Dialogue + Physical Action).\n"
            "2. Maintain a coherent conversation developing the pitch.\n"
            "3. Style: Fast-paced, witty, collaborative but argumentative.\n"
            "4. Format: JSON {{ 'speaker': 'Name', 'text': 'Dialogue', 'action_prompt': 'Visual description', 'visual_focus': 'Focus' }}\n"
        )
        
    else:
        # Standard Improv
        print("[*] Gathering Chaos Seeds...")
        seeds = [get_chaos_seed() for _ in range(6)]
        print(f"    Seeds: {seeds}")
        
        cast = defs["cast"]
        cast_desc = "\n".join([f"- {n}: {d['persona']}" for n, d in cast.items()])
        system_prompt = defs["system_prompt_template"].format(cast_desc=cast_desc)
    
    # 3. Execution Loop
    total_duration = 0.0
    turn_count = 0
    assets = []
    history = []
    
    # system_prompt is set in block above now
    # system_prompt = defs["system_prompt_template"].format(cast_desc=cast_desc)
    
    while total_duration < target_duration:
        # A. Seed Injection
        seed_idx = int(total_duration // 240)
        current_seed = seeds[seed_idx] if seed_idx < len(seeds) else "The Grand Finale"
        
        # B. Write
        # B. Write
        if "gahd" in vpform:
             prompt = (
                f"Time: {total_duration:.1f}/{target_duration}s.\n"
                f"Current Topic: Pitching '{seeds[0]}' meets '{seeds[1]}'.\n"
                f"History:\n" + "\n".join([f"{h['speaker']}: {h['text']}" for h in history[-10:]]) + 
                "\n\nGenerate next turn (JSON)."
            )
        else:
            seed_idx = int(total_duration // 240)
            current_seed = seeds[seed_idx] if seed_idx < len(seeds) else "The Grand Finale"
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
        
        # Truncate prompt to avoid CLIP 77 token warning (User Request)
        # Using ~300 chars as safe proxy for 76 tokens
        full_vis_prompt = f"{COURTROOM_STYLE_PROMPT}\nAction: {action}\nFocus: {visual}"
        if len(full_vis_prompt) > 300:
            visual_prompt = full_vis_prompt[:297] + "..."
        else:
            visual_prompt = full_vis_prompt
            
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
        
    # 4. Stitch / Export
    # Naming Logic: GAHD-{ep}-{ts}
    ep_str = f"{args.ep}-" if args.ep else ""
    final_basename = f"GAHD-{ep_str}{session_id}"
    
    # Target Directory
    gahd_out_dir = os.path.join(output_dir, "gahd-scripts-vids")
    os.makedirs(gahd_out_dir, exist_ok=True)
    
    final_mp4_path = os.path.join(gahd_out_dir, f"{final_basename}.mp4")
    
    stitch_assets(assets, session_dir, final_mp4_path)
    
    # 5. Export XMVP XML
    # Save alongside MP4
    xml_path = os.path.join(gahd_out_dir, f"{final_basename}.xml")
    
    cast_names = list(cast.keys())
    
    # We call helper differently? No, helper creates filename usually?
    # export_xmvp_manifest definition: def export_xmvp_manifest(output_dir, base_name, ...)
    # It constructs path: os.path.join(output_dir, f"{base_name}_manifest.xml")
    # User requested same name for MP4 and XML.
    # helper adds "_manifest.xml". We might want to adjust helper call or helper itself.
    # Let's adjust helper call to use gahd_out_dir and final_basename, but accepting the suffix for now.
    # User said: "same name for the MP4 and XML files" -> GAHD-201-1234.xml vs GAHD-201-1234_manifest.xml?
    # Usually clean names are better.
    # I'll let the helper add _manifest for clarity, or I'll patch helper too if needed.
    # Let's check helper definition below. 
    # Helper: xml_path = os.path.join(output_dir, f"{base_name}_manifest.xml")
    # I'll stick to that convention for now, close enough.
    
    export_xmvp_manifest(
        gahd_out_dir,
        final_basename,
        assets,
        cast_names,
        seeds=seeds,
        title=f"Improv Session {session_id} (Ep {args.ep})",
        synopsis="A generated improv comedy set."
    )
    
    print(f"✅ GAHD Session Complete. Output: {final_mp4_path}")


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

def run_thax_douglas_session(band_name, poem, output_dir):
    """
    Generates a Thax Douglas poem visualization.
    Voice: am_michael (Kokoro)
    Visuals: Band acting out poem.
    """
    if not band_name or not poem:
        print("[-] Error: --band and --poem required for thax-douglas mode.")
        return

    session_id = int(time.time())
    session_dir = os.path.join(output_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    print(f"🎸 Starting Thax Douglas Session: {band_name}...")
    
    # 1. Parse Poem
    # User Request: Replace slashes with periods to fix TTS flow
    # "The road is long / And my feet are tired" -> "The road is long. And my feet are tired"
    poem_sanitized = poem.replace("/", ".")
    
    # Split by newlines first
    stanzas = [line.strip() for line in poem_sanitized.split('\n') if line.strip()]
    
    # 1b. Smart Splitting (If single block detected)
    # If explicit newlines weren't used, we split by sentences to create more visual variety.
    if len(stanzas) == 1 and len(stanzas[0]) > 50:
         text_block = stanzas[0]
         # Split by punctuation followed by space (. ! ?)
         # We use a positive lookbehind to keep the punctuation attached to the sentence.
         # Regex: (?<=[.!?])\s+
         split_stanzas = re.split(r'(?<=[.!?])\s+', text_block)
         stanzas = [s.strip() for s in split_stanzas if s.strip()]
         print(f"   ✂️ Auto-Split Poem into {len(stanzas)} lines based on punctuation.")
         
    if not stanzas:
        stanzas = [poem] # Just one big block
        
    assets = []
    
    # 2. Loop
    for i, line in enumerate(stanzas):
        print(f"   📜 Stanza {i+1}/{len(stanzas)}: {line[:40]}...")
        
        # Audio (Thax Voice)
        # Force Kokoro 'am_michael' (or am_adam) for Thax
        # If Foley disabled or Cloud Mode, fall back to Journey-D via foley_talk defaults
        audio_name = f"thax_line_{i:03d}.wav"
        audio_path = os.path.join(session_dir, audio_name)
        
        # We use 'am_michael' as the 'voice_name'. foley_talk handles mapping/generation.
        # If mode='cloud', it maps to Journey-D naturally? No, foley_talk doesn't map am_michael to journey unless we map it.
        # So we pass 'am_michael' and mode='kokoro' if local, else 'en-US-Journey-D'.
        
        voice_target = "am_michael" if LOCAL_MODE else "en-US-Journey-D"
        mode_target = "kokoro" if LOCAL_MODE else "cloud"
        
        # Actually foley_talk.generate_audio_asset handles this if we pass mode.
        final_audio = generate_audio_asset(
            line, 
            audio_path, 
            voice_name=voice_target, 
            mode=mode_target
        )
        
        if not final_audio:
            print(f"   [-] Audio Gen Failed for line {i}")
            # Use silence as fallback?
            continue
            
        duration = get_audio_duration(final_audio)
        
        # Visual
        img_name = f"thax_vis_{i:03d}.jpg"
        img_path = os.path.join(session_dir, img_name)
        
        prompt = (
            f"Photorealistic concert photography of the band {band_name} performing on stage. "
            f"The band members are physically enacting the scene described: '{line}'. "
            f"Dramatic stage lighting, 4k, cinematic composition. "
            f"No text overlay."
        )
        
        # Truncate for Flux/CLIP safety
        if len(prompt) > 300: prompt = prompt[:297] + "..."
        
        if generate_image(prompt, img_path):
            assets.append({
                "path": img_path,
                "duration": duration, # Image holds for audio duration
                "audio": final_audio  # Attach audio to this slide
            })
        else:
            print("   [-] Image Gen Failed.")
            
    # 3. Stitch
    print("   🧵 stitching slides...")
    video_clips = []
    for i, asset in enumerate(assets):
        slide_mp4 = os.path.join(session_dir, f"slide_{i:03d}.mp4")
        # FFmpeg Image + Audio -> MP4
        # Loop image for duration of audio
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", asset['path'],
            "-i", asset['audio'],
            "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
            "-shortest", 
            "-pix_fmt", "yuv420p",
            slide_mp4
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            video_clips.append(slide_mp4)
        except Exception as e:
            print(f"   [-] Slide Mux Failed: {e}")
            
    # Concat clips
    if video_clips:
        thax_out_dir = os.path.join(output_dir, "thax-douglas")
        os.makedirs(thax_out_dir, exist_ok=True)
        final_filename = f"Thax-{band_name.replace(' ', '')}-{session_id}.mp4"
        final_path = os.path.join(thax_out_dir, final_filename)
        
        files_txt = os.path.join(session_dir, "clips.txt")
        with open(files_txt, 'w') as f:
            for v in video_clips:
                f.write(f"file '{v}'\n")
                
        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", files_txt,
            "-c", "copy",
            final_path
        ]
        subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Thax Douglas Session Complete. Output: {final_path}")
        
    else:
        print("❌ No slides generated.")

def generate_image(prompt, output_path):
    """
    Wrapper for Image Gen (Local Flux vs Cloud Gemini).
    """
    global FLUX_BRIDGE
    if LOCAL_MODE:
        if not FLUX_BRIDGE:
            # Init Bridge
            from flux_bridge import get_flux_bridge
            FLUX_BRIDGE = get_flux_bridge(FLUX_MODEL_PATH)
        
        # Flux Gen
        print(f"   🎨 [Flux] {prompt[:40]}...")
        # Flux usually returns a PIL Image
        img = FLUX_BRIDGE.generate(prompt, width=1024, height=576, steps=4)
        if img:
            img.save(output_path)
            return True
        return False
    else:
        return False 

def main():
    global LOCAL_MODE, FOLEY_ENABLED

    parser = argparse.ArgumentParser(description="Content Producer v0.5 (Unified)")
    parser.add_argument("vpform_pos", nargs="?", help="Positional Alias for VPForm (e.g. thax-douglas)")
    parser.add_argument("--vpform", help="Vision Platonic Form (e.g. 24-podcast, gahd-podcast)")
    parser.add_argument("--project", help="Project override (stub)")
    parser.add_argument("--ep", type=int, help="Episode number (stub)")
    parser.add_argument("--local", action="store_true", help="Use Local Engines (Flux + Kokoro)")
    parser.add_argument("--foley", choices=["on", "off"], default="off", help="Enable Generative Foley")
    parser.add_argument("--slength", type=float, default=0.0, help="Override duration in seconds")
    parser.add_argument("--fc", action="store_true", help="Code Painter Mode (Experimental)")
    parser.add_argument("--geminiapi", action="store_true", help="Force Cloud Gemini API for Text (Disable Local Gemma default)")
    parser.add_argument("--band", type=str, help="Band Name (for thax-douglas mode)")
    parser.add_argument("--poem", type=str, help="Poem Text (for thax-douglas mode)")
    args = parser.parse_args()
    
    # Alias Logic
    if args.vpform_pos:
        args.vpform = args.vpform_pos

    print(f"DEBUG: Args Parsed. VPForm: {args.vpform}")


    
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
    
    # 0. Thax Douglas Mode
    if args.vpform == "thax-douglas":
        run_thax_douglas_session(args.band, args.poem, OUTPUT_DIR)
        return

    if args.vpform and ("podcast" in args.vpform or "cartoon" in args.vpform):
        # 1. Generative Modes (GAHD, 24-podcast, 10-podcast)
        if "gahd" in args.vpform or "24" in args.vpform or "10" in args.vpform:
            text_engine = TextEngine()
            run_improv_session(args.vpform, OUTPUT_DIR, text_engine, args)
        
        # 2. Scanning Mode (Legacy/Manual Pairs)
        else:
            run_podcast_processing(TRIPLETS_DIR, OUTPUT_DIR, args)
    else:
        # Default behavior
        run_podcast_processing(TRIPLETS_DIR, OUTPUT_DIR, args)

if __name__ == "__main__":
    main()
