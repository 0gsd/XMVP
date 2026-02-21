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
# Fix for OMP Error #15 (Multiple OpenMP runtimes) - Critical for local MLL
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
    from mvp_shared import (
        save_xmvp, load_xmvp, CSSV, VPForm, Story, Portion, Constraints, 
        load_manifest, Manifest, Seg, DialogueScript, Indecision, DialogueLine
    )
    from text_engine import TextEngine
    from truth_safety import TruthSafety
    from foley_talk import generate_audio_asset
    from vision_producer import get_chaos_seed
    from vision_producer import get_chaos_seed
    from thax_audio import get_thax_engine
    from foley_talk import assign_kokoro_voice_deterministic
    from vision_producer import get_chaos_seed
    # frame_canvas moved to lazy import
    
    # Local Bridges
    from flux_bridge import get_flux_bridge
    # skyreels_bridge removed (deprecated audio-movie VPForm)
    from hunyuan_foley_bridge import generate_foley_asset
except ImportError as e:
    print(f"[-] Critical Import Error: {e}")
    sys.exit(1)

print("DEBUG: Imports Done.")

# --- CONFIG ---
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "z_test-outputs"))
TRIPLETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../z_podcast_triplets"))
MW_ROOT = "/Volumes/XMVPX/mw" # Standard mount point

# Resolve Flux Path via Definitions (GGUF First)
try:
    import definitions
    conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-klein")
    if not conf:
        conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-klein")
    FLUX_MODEL_PATH = conf.path if conf else os.path.join(MW_ROOT, "flux-root", "klein-9b")
except:
    FLUX_MODEL_PATH = os.path.join(MW_ROOT, "flux-root", "klein-9b")

print(f"    [🎨] Visual Engine: Flux GGUF-Ready (Path: {FLUX_MODEL_PATH})")

# --- GLOBAL STATE ---
LOCAL_MODE = False
FOLEY_ENABLED = False
FLUX_BRIDGE = None
GENERATION_DIMS = (512, 288) # Default 16:9

# --- CONSTANTS & FORMS ---
FORM_DEFS = {
    "24-cartoon": {
        "description": "4-Person Improv Comedy Special (24 Minutes)",
        "cast": {
            "William": {"base": "Billy Joel", "voice": "en-US-Journey-D", "pitch": 1, "persona": "The Piano Man. Working-class poet. Melodic, specific geographic references (Long Island), cynical but soulful."},
            "Maggie": {"base": "Margaret Thatcher", "voice": "en-US-Journey-F", "pitch": -2, "persona": "The Iron Lady. Stern, authoritative, uses 'Royal We'. Surprisingly willing to play high-status absurd characters."},
            "Francis": {"base": "Frank Sinatra", "voice": "en-US-Journey-L", "pitch": -2, "persona": "The Chairman. Cool, swaggering, mid-Atlantic accent. emotional, volatile, calls everyone 'baby'. Does it 'My Way'."},
            "Anne Tailored": {"base": "Taylor Swift", "voice": "en-US-Journey-O", "pitch": 1, "persona": "The Pop Icon. Earnest, confessional, detailed storytelling. Bridges scenes with emotional hooks. Avoids copyright infringement."}
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
    "10-podcast": {"alias": "24-cartoon", "duration_override": 600},
    "route66-podcast": {
        "description": "6-Person Improv Narrative (66 Minutes)",
        "duration_override": 3960,
        "cast": {
            "Amey": {"voice": "af_bella", "pitch": 0, "persona": "The Dreamer. Optimistic, sees signs everywhere, possibly psychic."},
            "Jessinny": {"voice": "af_sarah", "pitch": 0, "persona": "The Sceptic. Grounded, practical, calls out nonsense."},
            "Lorrey": {"voice": "af_nicole", "pitch": 0, "persona": "The Historian. Obsessed with the past, nostalgic, melancholic."},
            "Mercutio": {"voice": "am_michael", "pitch": 0, "persona": "The Trickster. Chaotic, plays devil's advocate, unpredictable."},
            "Rondio": {"voice": "am_adam", "pitch": 0, "persona": "The Driver. Focused, mission-oriented, protective."},
            "Totto": {"voice": "bm_george", "pitch": 0, "persona": "The Passenger. Along for the ride, observant, accidentally profound."}
        },
        "system_prompt_template": (
            "You are the Director of a 66-minute improvised audio drama featuring 6 travelers on a metaphysical road trip.\n"
            "The Cast:\n{cast_desc}\n\n"
            "The Mission:\n"
            "Weave a single coherent narrative incorporating 6 distinct Chaos Seeds over the journey.\n"
            "The Rules:\n"
            "1. Generate ONE turn at a time (Speaker + Dialogue + Physical Action).\n"
            "2. Maintain strict continuity of location and objective.\n"
            "3. Style: Atmospheric, somewhat surreal, character-driven.\n"
            "4. Format: JSON {{ 'speaker': 'Name', 'text': 'Dialogue', 'action_prompt': 'Visual description', 'visual_focus': 'Focus' }}\n"
        )
    }
}
FORM_DEFS["24-podcast"] = FORM_DEFS["24-cartoon"] # Simple alias ref

FORM_DEFS["black-box"] = {"alias": "fullmovie-still", "description": "Minimalist Black Box Theater Mode"}
FORM_DEFS["element-47"] = {"description": "Element 47 Audio Podcast", "mime_type": "video/mp4"}


# --- RVC CONFIG ---
RVC_MODELS_ROOT = os.path.join(os.path.dirname(__file__), "z_training_data")
# Note: 24-voices are in z_training_data/24_voices, Route 66 in z_training_data/route66_voices
# We need to handle sub-roots or just full relative paths in the map.
# Let's update RVC_MAP to include "24_voices/" or "route66_voices/" prefix or handle root logic.
# Implementation Plan assumed RVC_MAP values were subdir names.
# Let's make RVC_MODELS_ROOT generic to "z_training_data" and include parent dir in map values.

RVC_MAP = {
    # 24-Podcast
    "William": "24_voices/william-content",
    "Maggie": "24_voices/maggie-content",
    "Francis": "24_voices/francis-content",
    "Anne Tailored": "24_voices/annetailored-content",
    # Route 66
    "Amey": "route66_voices/amey-content",
    "Jessinny": "route66_voices/jessinny-content",
    "Lorrey": "route66_voices/lorrey-content",
    "Mercutio": "route66_voices/mercutio-content",
    "Rondio": "route66_voices/rondio-content",
    "Totto": "route66_voices/totto-content",
    # Element 47
    "Burn": "e47_voices/Burn",
    "Cruise": "e47_voices/Cruise",
    "Drip": "e47_voices/Drip",
    "Anchor": "e47_voices/Anchor"
}
RVC_PYTHON_BIN = os.path.join(os.path.expanduser("~"), "miniconda3/envs/rvc_env/bin/python")

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

def generate_location_context(text_engine, args, seeds=None):
    """
    Determines the visual setting for the session.
    1. Use --location if provided.
    2. Else, use Seeds to determine/extrapolate location.
    3. Else, generate a random, weird location.
    """
    if args.location:
        print(f"    [🌍] Location Override: {args.location}")
        return args.location
        
    print("    [🎲] Generating Location from Context...")
    
    seed_context = ""
    if seeds:
        seed_context = f"Context Seeds: {seeds}\n"
    
    prompt = (
        f"{seed_context}"
        "Based on the Context Seeds above, determine the 'Location' of this scene.\n"
        "1. If one of the seeds IS clearly a location (e.g. 'A haunted house'), USE IT.\n"
        "2. If multiple seeds are locations, choose the coolest/most visual one.\n"
        "3. If NO seeds are locations, extrapolate a logical setting where these seeds might meet (e.g. 'Golden Compass' -> 'A Steampunk Observatory').\n"
        "4. If no seeds provided, generate a specific, visually interesting, slightly weird location.\n\n"
        "Return ONLY the location description string."
    )
    
    try:
        loc = text_engine.generate(prompt).strip().strip('"')
        print(f"    [🌍] Location: {loc}")
        return loc
    except Exception as e:
        print(f"    [!] Location Gen Failed: {e}. using default.")
        return "An underground black box theater"

def generate_image(prompt, output_path, ts=None, init_image=None, strength=0.65):
    """
    Generates an image using either:
    1. Flux Klein (Local) if LOCAL_MODE is True
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
            # Pass init_image and strength for Img2Img/Context Passing
            
            # Use Global Dims
            w, h = GENERATION_DIMS
            
            img = FLUX_BRIDGE.generate(
                prompt, 
                width=w, height=h, 
                steps=12,  # Klein 12 steps
                image=init_image, 
                strength=strength
            )
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

def run_rvc_conversion(wav_path, character_name):
    """
    Polymorphic RVC Converter.
    1. Finds model for character in z_training_data/24_voices
    2. Runs inference via rvc_env
    3. Overwrites wav_path with converted audio
    """
    if character_name not in RVC_MAP:
        print(f"       [!] RVC: No mapping for {character_name}")
        return False
        
    model_subdir = RVC_MAP[character_name]
    model_dir = os.path.join(RVC_MODELS_ROOT, model_subdir)
    
    # Strict Filename Matching (User Sanitization Request)
    # Character Name key -> lowercase filename
    # e.g. "William" -> william-content -> william.pth
    
    # Extract clean name from character_name or model_subdir?
    # model_subdir is like "24_voices/william-content"
    # We want "william" from the basename "william-content" (split -)
    # OR simpler: map allows arbitrary keys, we should just assume the filename matches the key.lower()??
    # User said: "amey.pth ... under amey-content".
    # And "William" -> "william.pth".
    # Let's derive it from the key in RVC_MAP? 
    # Key is "William", "Anne Tailored", etc.
    # Filenames I just made: "william.pth", "annetailored.pth".
    # Logic: key.lower().replace(" ", "") ?
    # "Anne Tailored" -> "annetailored". Correct.
    # "William" -> "william". Correct.
    
    clean_name = character_name.lower().replace(" ", "")
    pth_file = os.path.join(model_dir, f"{clean_name}.pth")
    index_file = os.path.join(model_dir, f"{clean_name}.index")
    
    # Removed premature check
    # if not os.path.exists(pth_file): ...
        
    if not os.path.exists(index_file):
        index_file = "" # Optional

    # Check for generic model.pth if specific name not found
    if not os.path.exists(pth_file):
        generic_pth = os.path.join(model_dir, "model.pth")
        if os.path.exists(generic_pth):
            pth_file = generic_pth
            print(f"       [i] RVC: Using generic model.pth for {character_name}")
        else:
            print(f"       [!] RVC: Model file not found: {pth_file} or {generic_pth}")
            return False
        
    # Construct Temp Path
    # We write result to a temp file first, then move it
    temp_rvc_out = wav_path.replace(".wav", "_rvc.wav")
    
    rvc_cmd = [
        RVC_PYTHON_BIN, "-c",
        f"""
import os, sys, torch
# Monkeypatch PyTorch 2.6+
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

from rvc_python.infer import RVCInference
try:
    rvc = RVCInference(device="mps")
    try:
        rvc.load_model("{pth_file}", version="v2")
    except:
        rvc.load_model("{pth_file}")
        
    # Manual Single Inference
    wav_opt = rvc.vc.vc_single(
        0, "{wav_path}", 0, None, "rmvpe", 
        "{index_file if index_file else ''}", "", 0, 3, 0, 0.25, 0.33
    )
    
    if isinstance(wav_opt, tuple): tgt_sr, audio_data = wav_opt
    else: tgt_sr, audio_data = rvc.vc.tgt_sr, wav_opt

    from scipy.io import wavfile
    wavfile.write("{temp_rvc_out}", tgt_sr, audio_data)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""
    ]
    
    try:
        print(f"       mic... converting to {character_name} via RVC...")
        res = subprocess.run(rvc_cmd, capture_output=True, text=True)
        if "SUCCESS" in res.stdout:
             # Success
             os.replace(temp_rvc_out, wav_path)
             # print("       [+] RVC Converison Complete") # Optional verbose
             return True
        else:
             # Capture standard out too since our script prints ERROR there sometimes
             err_msg = res.stderr[:200] if res.stderr else res.stdout[:200]
             print(f"       [!] RVC Failed: {err_msg}")
             return False
    except Exception as e:
        print(f"       [!] RVC Exec Failed: {e}")
        return False


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

def validate_and_fill_seeds(seeds_input, text_engine, target_count=6):
    """
    Parses, validates, and fills a list of seeds.
    1. Parse string input (JSON or comma-separated).
    2. Validate each seed against Wikipedia.
    3. If invalid, ask Text Engine for a RELATED real entity.
    4. Fill remaining slots with random Chaos Seeds.
    """
    print(f"    [🌱] Processing Explicit Seeds: {seeds_input}")
    
    # 1. Parse
    parsed_seeds = []
    try:
        # Try JSON first
        clean_input = seeds_input.replace("'", '"') # Naive fix for single quotes
        parsed_seeds = json.loads(clean_input)
    except:
        # Fallback to simple split
        # Remove brackets and split
        clean = seeds_input.strip("[]").replace('"', '').replace("'", "")
        parsed_seeds = [s.strip() for s in clean.split(",") if s.strip()]
    
    if not isinstance(parsed_seeds, list):
        parsed_seeds = [str(parsed_seeds)]
        
    valid_seeds = []
    
    # 2. Validate & Fallback
    for seed in parsed_seeds:
        print(f"       Testing '{seed}'...")
        
        # Check Wikipedia
        # We use the same simple summary endpoint as vision_producer
        slug = seed.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        headers = {'User-Agent': 'VisionProducer/1.0 (seed_validator)'}
        
        try:
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                # Valid!
                canonical = resp.json().get('title', seed)
                print(f"       ✅ Valid: {canonical}")
                valid_seeds.append(canonical)
                continue
            else:
                print(f"       ❌ Invalid: {seed} (404/Error)")
        except:
             print(f"       ❌ Invalid: {seed} (Req Failed)")
             
        # Fallback: Intelligent Repair
        print(f"       🔧 Repairing '{seed}' via Text Engine...")
        prompt = (
            f"The user provided a Chaos Seed '{seed}' which is not a valid Wikipedia Page.\n"
            f"Identify the real-world concept they likely meant, or a closely related real entity.\n"
            f"Examples: 'Fear (Hubbard novella)' -> 'Fear (novella)'\n"
            f"Output ONLY the exact, correct Wikipedia Page Title. Nothing else."
        )
        try:
            suggestion = text_engine.generate(prompt).strip().strip('"').strip("'")
            # Verify suggestion?
            s_slug = suggestion.replace(" ", "_")
            s_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{s_slug}"
            s_resp = requests.get(s_url, headers=headers, timeout=3)
            if s_resp.status_code == 200:
                fixed = s_resp.json().get('title', suggestion)
                print(f"       ✨ Repaired: {fixed}")
                valid_seeds.append(fixed)
            else:
                print(f"       ⚠️ Repair Failed: {suggestion}. Using random seed.")
                valid_seeds.append(get_chaos_seed())
        except Exception as e:
            print(f"       ⚠️ Repair Error: {e}. Using random seed.")
            valid_seeds.append(get_chaos_seed())
            
    # 3. Fill
    while len(valid_seeds) < target_count:
        print("       ➕ Adding Random Chaos Seed...")
        valid_seeds.append(get_chaos_seed())
        
    return valid_seeds[:target_count]


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
        
        # Dynamic Location
        location = generate_location_context(text_engine, args, seeds)
        
        # Override System Prompt for GAHD
        system_prompt = (
            "You are a Writers Room of 4 eccentric screenwriters (Character Actors).\n"
            f"The Cast:\n{cast_desc}\n\n"
            f"The Setting: {location}\n"
            f"The Mission:\n"
            f"Develop a high-concept Movie Pitch that combines these two random elements:\n"
            f"1. {seeds[0]}\n2. {seeds[1]}\n\n"
            "The Rules:\n"
            "1. Generate ONE turn at a time (Speaker + Dialogue + Physical Action).\n"
            "2. Maintain a coherent conversation developing the pitch.\n"
            "3. Style: Fast-paced, witty, collaborative but argumentative.\n"
            "4. Format: JSON {{ 'speaker': 'Name', 'text': 'Dialogue', 'action_prompt': 'Visual description', 'visual_focus': 'Focus' }}\n"
        )
        
        # Dynamic Visual Prompt
        session_visual_style = (
            f"A photorealistic sketch of {location}. Drawn by a preternaturally talented artist. "
            "Hyperrealistic style, clean commercial art. NO WORDS, NO SPEECH BUBBLES. "
            "The artist captures the actors improvising. Dynamic composition."
        )
        

        
    elif "route66" in vpform:
        # Route 66 Logic
        if args.seeds:
            print("[*] Route 66 Mode: Processing Explicit Seeds...")
            seeds = validate_and_fill_seeds(args.seeds, text_engine, target_count=6)
        else:
            print("[*] Route 66 Mode: Fetching 6 Seeds for The Journey...")
            seeds = [get_chaos_seed() for _ in range(6)]
        
        print(f"    Seeds: {seeds}")
        
        cast = defs["cast"]
        cast_desc = "\n".join([f"- {n}: {d['persona']}" for n, d in cast.items()])
        system_prompt = defs["system_prompt_template"].format(cast_desc=cast_desc)
        
        # Random location generation logic same as others
        location = generate_location_context(text_engine, args, seeds)
        session_visual_style = (
            f"A cinematic shot of {location}. Wide aspect ratio. "
            "Atmospheric lighting, highly detailed 3D environment. "
            "Photorealistic."
        )

    else:
        # Standard Improv
        if args.seeds:
            print("[*] Standard Improv: Processing Explicit Seeds...")
            seeds = validate_and_fill_seeds(args.seeds, text_engine, target_count=6)
        else:
            print("[*] Gathering Chaos Seeds...")
            seeds = [get_chaos_seed() for _ in range(6)]
            
        print(f"    Seeds: {seeds}")
        
        cast = defs["cast"]
        cast_desc = "\n".join([f"- {n}: {d['persona']}" for n, d in cast.items()])
        system_prompt = defs["system_prompt_template"].format(cast_desc=cast_desc)
        
        # Dynamic Location (Default for standard improv too? Why not)
        location = generate_location_context(text_engine, args, seeds)
        session_visual_style = (
            f"A photorealistic sketch of {location}. Drawn by a preternaturally talented artist. "
            "Hyperrealistic style. NO WORDS. Dynamic composition."
        )
    
    # --- MLL INJECTION (Improv Mode) ---
    if args.mll == "on":
        print("    [🧬] MLL Requested for Improv Session. Generating Stub XML...")
        
        # 1. Create Stub Manifest
        # construct Story strings with anchors
        story_chars = []
        dummy_lines = []
        
        for name, info in cast.items():
            # "Name (Role) [Gender] {Anchor}"
            # Use Persona as anchor
            anchor = info.get('persona', 'A character')[:100].replace('"', '')
            gender = "Male" if info.get('pitch', 0) > -5 else "Female" # Rough guess or check defs?
            # Actually cast dict sometimes has gender, sometimes not.
            # Let's just use [?] if unknown, or default.
            
            s_str = f"{name} (Improviser) [?] {{{anchor}}}"
            story_chars.append(s_str)
            
            # Add dialogue line so it gets picked up
            dummy_lines.append(DialogueLine(
                character=name,
                text="Hello world.",
                action="standing",
                visual_focus=name
            ))
            
        # Construct & Save Stub
        stub_path = os.path.join(session_dir, "mll_stub.xml")
        
        # Determine Template Name
        tpl_name = "Improv_Template"
        if "gahd" in vpform: tpl_name = "GAHD_Template" 
        elif "route66" in vpform: tpl_name = "Route66_Template"
        elif "24-" in vpform: tpl_name = "24_Template"
        
        stub_data = {
            "Story": Story(
                title=f"Improv {session_id}",
                synopsis="Improv MLL Stub",
                characters=story_chars,
                theme="Improv",
                mll_template=tpl_name
            ),
            "Manifest": Manifest(
                segs=[Seg(id=0, start_frame=0, end_frame=0, prompt="Stub")],
                dialogue=DialogueScript(lines=dummy_lines)
            ),
            "Bible": CSSV(
                vision="Cinematic Improv",
                mll_template=tpl_name,
                scenario="Stub Scenario",
                situation="Stub", constraints=Constraints(width=1024, height=1024, fps=24)
            )
        }
        save_xmvp(stub_data, stub_path)
        
        # 2. Run Pipeline
        lora_path = run_mll_pipeline(stub_path, output_dir)
        if lora_path:
             print(f"    [💪] Activating MLL Adapter: {lora_path}")
             os.environ["LOCAL_ADAPTER_PATH"] = lora_path
    # -----------------------------------

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
        full_vis_prompt = f"{session_visual_style}\nAction: {action}\nFocus: {visual}"
        if len(full_vis_prompt) > 300:
            visual_prompt = full_vis_prompt[:297] + "..."
        else:
            visual_prompt = full_vis_prompt
            
        # Route 66 Carbonation Logic
        if "route66" in vpform:
            from sassprilla_carbonator import carbonate_prompt
            # Carbonate visual prompt with 3D space awareness
            # Priority: Character > Topic > Scene > Location
            # We map "Character + Action" to the 'Title' slot so Carbonator focuses on it.
            context_seed = current_seed if 'current_seed' in locals() else "The Journey"
            
            # Truncate location if it's too long (User Request: "don't start with 100 chars of location stuff")
            loc_str = location
            if len(loc_str) > 100:
                loc_str = loc_str[:97] + "..."

            carb_prompt = carbonate_prompt(
                title=f"{visual} :: {action}",  # "Song Title" acts as the core subject
                artist=f"Topic: {context_seed}", # "Artist" acts as the thematic lens
                extra_context=f"Location: {loc_str}. Style: {session_visual_style}"
            )
            if carb_prompt:
                # Smart Truncation to avoid FluxBridge naive chop (User Request)
                # Target ~300 chars (approx 75 tokens) for safety
                if len(carb_prompt) > 300:
                    short_prompt = carb_prompt[:300]
                    # Find last sentence-ending punctuation
                    last_punc = max(short_prompt.rfind('.'), short_prompt.rfind('!'), short_prompt.rfind('?'))
                    
                    if last_punc > 150: # Ensure we don't cut too much (keep at least half)
                        carb_prompt = short_prompt[:last_punc+1]
                    else:
                        # Fallback to last space if no punctuation found nearby
                        last_space = short_prompt.rfind(' ')
                        if last_space > 0:
                            carb_prompt = short_prompt[:last_space] + "..."
                        else:
                            carb_prompt = short_prompt # No spaces? Just use hard cut
                
                print(f"       🫧 Carbonated Visual: {carb_prompt[:50]}... ({len(carb_prompt)} chars)")
                visual_prompt = carb_prompt
            
        if not generate_image(visual_prompt, img_path):
             create_black_frame(img_path)
             
        # D. Audio (Speech)
        wav_path = os.path.join(session_dir, f"turn_{turn_count:04d}.wav")
        voice_info = cast.get(speaker, list(cast.values())[0])
        audio_mode = "kokoro" # User Request: Always use Kokoro
        
        target_voice = voice_info['voice']
        # Always map to Kokoro equivalents since we are forcing audio_mode="kokoro"
        target_voice = map_voice_to_kokoro(target_voice)
            
        final_wav = generate_audio_asset(
            text, wav_path, 
            voice_name=target_voice, 
            pitch=voice_info['pitch'],
            mode=audio_mode
        )
        
        # F. RVC Post-Process
        if args.rvc and final_wav and os.path.exists(final_wav):
            run_rvc_conversion(final_wav, speaker)
        
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
    
    # Determine Template Name based on vpform
    target_template = None
    if "gahd" in vpform: 
        target_template = "GAHD_Template"
    elif "route66" in vpform:
        target_template = "Route66_Template"
    elif "24-" in vpform:
        target_template = "24_Template"

    export_xmvp_manifest(
        gahd_out_dir,
        final_basename,
        assets,
        cast_names,
        seeds=seeds,
        title=f"Improv Session {session_id} (Ep {args.ep})",
        synopsis="A generated improv comedy set.",
        mll_template=target_template
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
            
        # FORCE KOKORO
        voice = map_voice_to_kokoro(voice)
        audio_mode = "kokoro"
        
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

def run_element47_production(manifest_path, output_dir, args):
    """
    Production loop for Element 47 Podcast.
    Features:
    - 2-Track Audio (Dialogue + SFX/Music)
    - RVC Voice Mapping (Performer)
    - Kokoro Character Mapping (Character)
    - Sequential Assembly suitable for simple podcast structure.
    """
    print("🎙️ ELEMENT 47 PRODUCTION")
    
    # 1. Load Manifest
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        print(f"[-] Manifest Load Failed: {e}")
        return

    base_name = os.path.splitext(os.path.basename(manifest_path))[0].replace("_manifest", "")
    session_dir = os.path.join(output_dir, f"e47_{base_name}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 2. Setup SFX
    from sfx_bridge import generate_sfx
    
    # 3. Processing Loop
    full_audio_assets = [] # List of {path, type, duration}
    
    # Track current page for visual if needed (optional)
    
    lines = manifest.dialogue.lines if manifest.dialogue else []
    print(f"[*] Processing {len(lines)} lines...")
    
    for i, line in enumerate(lines):
        line_type = "dialogue"
        if line.character in ["SFX", "MUSIC"]: line_type = line.character.lower()
        
        print(f"    [{i+1}/{len(lines)}] {line.character}: {line.text[:50]}...")
        
        asset_path = os.path.join(session_dir, f"line_{i:04d}_{line_type}.wav")
        
        if line_type == "dialogue":
            # Format: "PERFORMER::Character"
            if "::" in line.character:
                performer, character = line.character.split("::", 1)
            else:
                performer = line.character
                character = line.character # Fallback
            
            # Normalize Performer for RVC (Burn, Cruise, etc.)
            performer_key = performer.capitalize() # BURN -> Burn
                
            # A. Generate TTS (Kokoro)
            # Use Character name to determine Kokoro Voice/Pitch
            # This gives "Acting" variety while RVC gives "Voice" identity.
            
            # Map Performer to RVC Model
            # Map Character to Kokoro Base
            
            # We use assign_kokoro_voice_deterministic(character)
            voice_name, pitch = assign_kokoro_voice_deterministic(character, ["af_bella", "af_sarah", "am_michael", "am_adam"])
            
            # Clean Text: Remove (beat) and replace with pause
            clean_text = line.text
            if "(beat)" in clean_text.lower():
                import re
                clean_text = re.sub(r'\(beat\)', '...', clean_text, flags=re.IGNORECASE)
                print(f"       [✂️] Removed '(beat)', adjusted text: {clean_text[:50]}...")
            
            # Generate Base
            res_path = generate_audio_asset(
                clean_text, asset_path, 
                voice_name=voice_name, 
                pitch=pitch, 
                mode="kokoro"
            )
            
            if res_path and os.path.exists(res_path):
                # B. Apply RVC (Performer)
                # Check if Performer has RVC Model
                if performer_key in RVC_MAP:
                    run_rvc_conversion(res_path, performer_key)
                else:
                    print(f"       [!] No RVC model for {performer} ({performer_key}), keeping Kokoro base.")
                    
                dur = get_audio_duration(res_path)
                full_audio_assets.append({"path": res_path, "type": "dialogue", "duration": dur})
            else:
                print("       [-] TTS Failed.")
                
        elif line_type in ["sfx", "music"]:
            # SFX Generation
            duration = line.duration if line.duration > 0 else 5.0
            
            # If prompt implies length? 
            # "Theme music" -> 10s? "Crash" -> 2s?
            # Basic heuristics
            if "theme" in line.text.lower() or "music" in line.text.lower():
                duration = 10.0
            elif "sting" in line.text.lower():
                duration = 3.0
                
            success = generate_sfx(line.text, asset_path, duration=duration, type=line_type)
            if success:
                # Add gap/silence padding?
                # Just append
                full_audio_assets.append({"path": asset_path, "type": "sfx", "duration": duration})
            else:
                 print("       [-] SFX Failed.")

    # 4. Stitching
    # FIX: Normalize all assets to 44.1kHz 16-bit Stereo to prevent concat speed bugs (chipmunk effect)
    print("   🔧 Normalizing Audio Segments...")
    norm_dir = os.path.join(session_dir, "normalized")
    os.makedirs(norm_dir, exist_ok=True)
    
    normalization_failed = False
    
    # We essentially want to concat everything in order for now.
    print("   🧵 Stitching Audio Track...")
    concat_list = os.path.join(session_dir, "concat.txt")
    
    with open(concat_list, 'w') as f:
        for a in full_audio_assets:
            # Normalize
            try:
                base_seg = os.path.basename(a['path'])
                norm_path = os.path.join(norm_dir, base_seg)
                
                # Determine LUFS Target
                # Dialogue: -16 LUFS (Standard Podcast/Web)
                # SFX/Music: -20 LUFS (Slightly quieter context)
                lufs_target = "-16"
                if a.get("type", "dialogue") in ["sfx", "music"]:
                    lufs_target = "-20"
                
                # ffmpeg normalize + sample rate fix
                # -af loudnorm=I={lufs_target}:TP=-1.5:LRA=11
                subprocess.run([
                    "ffmpeg", "-y", "-i", a['path'],
                    "-af", f"loudnorm=I={lufs_target}:TP=-1.5:LRA=11",
                    "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                    "-loglevel", "error",
                    norm_path
                ], check=True)
                
                # Write to concat list (use normalized path)
                f.write(f"file '{os.path.abspath(norm_path)}'\n")
                
            except Exception as e:
                print(f"       [-] Normalization failed for {base_seg}: {e}")
                normalization_failed = True
                # Fallback to original (might cause issues but we try)
                f.write(f"file '{os.path.abspath(a['path'])}'\n")

    final_wav = os.path.join(session_dir, "final_mix.wav")
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', final_wav, '-y', '-loglevel', 'error'])
    
    # 4.5 Export MP3 (Audio Only)
    final_mp3 = os.path.join(output_dir, f"{base_name}.mp3")
    print(f"   🎵 Exporting MP3: {final_mp3}")
    subprocess.run([
        "ffmpeg", "-i", final_wav,
        "-b:a", "192k", "-y", "-loglevel", "error",
        final_mp3
    ])

    # 5. Visual (Static Black or Cover)
    final_mp4 = os.path.join(output_dir, f"{base_name}.mp4")
    
    # Generate Cover Image?
    cover_img = os.path.join(session_dir, "cover.jpg")
    create_black_frame(cover_img) # Placeholder
    
    print(f"   🎥 Rendering MP4: {final_mp4}")
    total_dur = get_audio_duration(final_wav)
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", cover_img,
        "-i", final_wav,
        "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
        "-t", str(total_dur),
        "-pix_fmt", "yuv420p",
        final_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("   ✅ Done.")

def stitch_assets(assets, temp_dir, output_mp4):
    # Quick Stitch
    list_path = os.path.join(temp_dir, "stitch.txt")
    mp4s = []
    
    print(f"   🧵 Stitching {len(assets)} segments...")
    
    for i, a in enumerate(assets):
        # Validate Assets
        img_path = a.get('image')
        aud_path = a.get('audio')
        
        if not img_path or not os.path.exists(img_path):
            print(f"      [!] Missing Image for Seg {i}: {img_path}")
            # Create black frame fallback?
            # For now, just skip to avoid crashing the whole render
            continue
            
        if not aud_path or not os.path.exists(aud_path):
            print(f"      [!] Missing Audio for Seg {i}: {aud_path}")
            continue

        seg_mp4 = os.path.join(temp_dir, f"clip_{i}.mp4")
        
        try:
            cmd = [
                'ffmpeg', '-y', '-loop', '1', '-i', img_path, '-i', aud_path,
                '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p'
            ]
            
            # Use explicit duration if available to prevent ffmpeg overshoot/desync (especially on Mac with AAC)
            dur = a.get('duration')
            if dur and float(dur) > 0:
                cmd.extend(['-t', str(dur)])
            else:
                cmd.append('-shortest')
                
            cmd.extend([seg_mp4, '-loglevel', 'error'])
            
            subprocess.run(cmd, check=True)
            mp4s.append(seg_mp4)
        except Exception as e:
             print(f"      [!] FFmpeg Failed for Seg {i}: {e}")
             
    if not mp4s:
        print("[-] No valid segments to stitch.")
        return

    with open(list_path, 'w') as f:
        for mp4 in mp4s: f.write(f"file '{mp4}'\n")
        
    try:
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_mp4, '-y', '-loglevel', 'error'], check=True)
        print(f"[+] Output Saved: {output_mp4}")
    except Exception as e:
        print(f"[-] Final Stitch Failed: {e}")

def export_xmvp_manifest(output_dir, base_name, assets, cast_names, seeds=None, title=None, synopsis="Generated Content", mll_template=None):
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
            vision="Cinematic/Generated",
            mll_template=mll_template
        )
        
        story = Story(
            title=title if title else base_name,
            synopsis=synopsis,
            characters=cast_names,
            theme="Improv/Podcast",
            mll_template=mll_template
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
        
        # Paging Logic (Improv/Podcast Emulation)
        # We simulate how many lines of "Script" this turn would take.
        current_page = 0
        lines_on_page = 0
        LINES_PER_PAGE = 55
        
        for item in assets:
            if isinstance(item, dict):
                 # Improv Dict
                 spk_len = 1 # Speaker Name
                 text_len = max(1, len(item.get('text', '')) // 60) # Approx 60 chars per line
                 act_len = 1 if item.get('action_prompt') else 0
                 
                 total_lines = spk_len + text_len + act_len
                 
                 # Update Page
                 lines_on_page += total_lines
                 if lines_on_page >= LINES_PER_PAGE:
                     current_page += 1
                     lines_on_page = 0
                 
                 dl = mvp_shared.DialogueLine(
                     character=item['speaker'],
                     text=item.get('text', ''),
                     duration=item['duration'],
                     action=item.get('action_prompt', ''), # Stored earlier?
                     foley=item.get('foley_prompt', ''),   # Wait, need to store these in assets in main loop
                     visual_focus=item.get('visual_focus', ''),
                     page_index=current_page
                 )
            else:
                 # Podcast Segment
                 spk_len = 1
                 text_len = max(1, len(item.text) // 60)
                 total_lines = spk_len + text_len
                 
                 lines_on_page += total_lines
                 if lines_on_page >= LINES_PER_PAGE:
                     current_page += 1
                     lines_on_page = 0

                 dl = mvp_shared.DialogueLine(
                     character=item.speaker,
                     text=item.text,
                     duration=item.duration,
                     page_index=current_page
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
        
    # 1.5 Prepend Band Name (Thax Introduction)
    # Thax says the band name first.
    stanzas.insert(0, f"{band_name}.")
    
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
        
        # Use Thax Engine (RVC) if possible, else standard
        thax_engine = get_thax_engine()
        success = thax_engine.generate(line, audio_path)
        final_audio = audio_path if success else None
        
        if not final_audio:
            print(f"   [-] Audio Gen Failed for line {i}")
            # Use silence as fallback?
            continue
            
        duration = get_audio_duration(final_audio)
        
        # Visual
        img_name = f"thax_vis_{i:03d}.jpg"
        img_path = os.path.join(session_dir, img_name)
        
        # Simplified Visual Prompt (User Request: "Exact line only")
        # Removing "Concert/Band" context.
        # Adding minimal style suffix for Flux quality.
        prompt = f"{line}, cinematic, photorealistic, 4k."
        
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

 

# --- FULLMOVIE STILL MODE ---

def get_segment_for_time(manifest: Manifest, time_sec: float, fps: float = 24.0):
    """Finds the visual segment active at the given time."""
    frame = int(time_sec * fps)
    # Simple linear search (manifests are small enough)
    for seg in manifest.segs:
        if seg.start_frame <= frame < seg.end_frame:
            return seg
    
    # Fallback: Closest or Last
    if manifest.segs:
        return manifest.segs[-1]
    return None

def is_valid_character(name):
    """
    Filters out metadata artifacts/garbage from character names.
    e.g. "SCENE_NUMBER", "LOCATION", "ACTION", or JSON-like keys.
    """
    if not name: return False
    
    # Pre-clean: Remove Anchor {} and Gender [] tags
    # "Stanley [Male] {Humphrey Bogart}" -> "Stanley"
    import re
    clean_n = re.sub(r'\{.*?\}', '', name)
    clean_n = re.sub(r'\[.*?\]', '', clean_n)
    clean_n = clean_n.strip().strip('"').strip("'")
    
    if len(clean_n) < 1: return False # Allow single letters like 'P'
    
    # 1. Check for JSON-like artifacts (contains quote-colon)
    # The user logs show '"CHARACTER": "Name",' often.
    if '":' in name or '": ' in name: return False
    
    # 2. Check for Metadata Keywords (Case-Insensitive)
    # Also filter widely known non-character tokens
    name_upper = clean_n.upper()
    invalid_tokens = [
        "SCENE_NUMBER", "LOCATION", "ACTION", "SCENE_HEADING", 
        "DIALOGUE", "LINE", "CHARACTERS", "EXT.", "INT.", 
        "CUT TO", "FADE IN", "FADE OUT", "TRANSITION",
        "CAST", "SCENE", "TIME", "SETTING", "DISSOLVE TO", "FADE TO"
    ]
    
    for token in invalid_tokens:
        if token in name_upper:
            # Only exact match or significant match?
            # "FADE TO:" contains "FADE TO". True.
            # "The Margin" does not contain "SCENE".
            return False

    # 3. Check for bracketed metadata or markdown
    if name.strip().startswith('"') and name.strip().endswith('":'): return False
    if name.strip().startswith('#'): return False
    if name.strip().startswith('**'): return False
    if name_upper.endswith(" TO:"): return False
    
    return True

    print(f"    [⚧️] Scan Results: {results}")
    return results

def scan_for_genders(manifest, known_chars):
    """
    Scans Manifest Segments (Action/Content) for pronoun usage to infer gender.
    Uses proximity/chunking to handle shared scenes.
    """
    print("    [⚧️] Scanning Script for Gender Pronouns...")
    scores = {c: {"m": 0, "f": 0} for c in known_chars}
    
    # Pronouns (Word boundaries)
    male_pr = [r"\bhe\b", r"\bhim\b", r"\bhis\b"]
    female_pr = [r"\bshe\b", r"\bher\b", r"\bhers\b"]
    
    for seg in manifest.segs:
        text = seg.prompt.lower() if seg.prompt else ""
        if not text: continue

        # Find all character occurrences
        occurrences = []
        for char in known_chars:
            c_low = char.lower()
            start = 0
            while True:
                idx = text.find(c_low, start)
                if idx == -1: break
                occurrences.append((idx, char))
                start = idx + 1
        
        if not occurrences: continue
        
        # Sort by position
        occurrences.sort(key=lambda x: x[0])
        
        # Process Chunks
        for i in range(len(occurrences)):
            start_idx, char_name = occurrences[i]
            # End is next char start or end of text
            end_idx = occurrences[i+1][0] if i+1 < len(occurrences) else len(text)
            
            chunk = text[start_idx:end_idx]
            
            # Scan chunk
            m_count = 0
            f_count = 0
            import re
            for p in male_pr: m_count += len(re.findall(p, chunk))
            for p in female_pr: f_count += len(re.findall(p, chunk))
            
            scores[char_name]["m"] += m_count
            scores[char_name]["f"] += f_count

    # Decide
    results = {}
    for char, counts in scores.items():
        m = counts["m"]
        f = counts["f"]
        total = m + f
        if total > 0:
            ratio = m / total
            if ratio > 0.6: results[char] = "Male"
            elif ratio < 0.4: results[char] = "Female"
            
    print(f"    [⚧️] Scan Results: {results}")
    return results

def construct_cinematic_prompt(segment_prompt, character, action, focus):
    # Clean inputs
    loc = segment_prompt.strip()
    char = character.strip()
    act = action.strip() if action else "standing"
    foc = focus.strip() if focus else char
    
    # Flux-friendly Template
    # "Cinematic film still of [Character] [Action], [Focus]. [Location]. [Style keywords]"
    
    prompt = f"Cinematic film still of {char} {act}, {foc}. {loc}. "
    prompt += "film grain, 4k, highly detailed, dramatic lighting, shallow depth of field, photorealistic, masterpiece."
    return prompt

def run_mll_pipeline(xml_path, output_dir):
    """
    Orchestrates the Movie-Level LoRA pipeline:
    1. prep_movie_assets.py -> Generate synthetic dataset from Manifest
    2. train_mll.py -> Train LoRA on dataset
    Returns: Path to trained LoRA or None
    """
    print("\n🧬 MLL PIPELINE: Initiating Movie-Level LoRA Training...")
    
    # Determine Persistent Paths
    base_dir = os.getcwd()
    mll_root = os.path.join(base_dir, "z_training_data", "movies")
    adapter_root = os.path.join(base_dir, "z_training_data", "adapters")
    os.makedirs(mll_root, exist_ok=True)
    os.makedirs(adapter_root, exist_ok=True)

    # 1. Prep Data
    try:
        prep_cmd = ["python3", "prep_movie_assets.py", "--xml", xml_path, "--out", mll_root]
        print(f"   [1/2] Generating Synthetic Assets... (Target: {mll_root})")
        subprocess.run(prep_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   [-] Asset Prep Failed: {e}")
        return None
        
    # Determine LoRA Name from XML to ensure we target the correct folder
    target_name = None
    try:
        raw_story = load_xmvp(xml_path, "Story")
        if raw_story:
            story = json.loads(raw_story)
            # Match prep_movie_assets logic
            target_name = story.get("title", "Unknown_Movie").replace(" ", "_")
            if story.get("mll_template"):
                target_name = story.get("mll_template")
                
        raw_bib = load_xmvp(xml_path, "Bible") 
        # Check Bible too
        if raw_bib:
             bible = json.loads(raw_bib)
             if bible.get("mll_template"):
                 target_name = bible.get("mll_template")
                 
    except Exception as e:
        print(f"   [-] Failed to parse XML for MLL target: {e}")

    if not target_name:
        # Fallback to naive (scan mll_root? might be dangerous if many movies exist)
        # But since we just ran prep, it likely touched one.
        # We really should have found it via XML.
        print("   [-] Could not determine MLL Target Name from XML.")
        return None
        
    print(f"   [i] Targeting MLL Dataset: {target_name}")
    
    dataset_path = os.path.join(mll_root, target_name, "dataset")
    
    if not os.path.exists(dataset_path):
        print(f"   [-] Dataset not found at {dataset_path}. Prep might have failed.")
        return None
        
    # CACHE CHECK: Check for existing adapter
    final_lora_path = os.path.join(adapter_root, f"{target_name}.safetensors")
    if os.path.exists(final_lora_path):
        print(f"   [⚡️] Cached Adapter Found: {final_lora_path}")
        print("   [.] Skipping Training.")
        return final_lora_path

    # Canonical Output
    base_dir = os.path.dirname(os.path.abspath(__file__))
    adapters_dir = os.path.join(base_dir, "adapters/movies")
    
    # 2. Train
    try:
        train_cmd = [
            "python3", "train_mll.py", 
            "--dataset", dataset_path, 
            "--name", target_name,
            "--output_dir", adapters_dir,
            "--steps", "100" # Fast tune
        ]
        print(f"   [2/2] Training LoRA... ({' '.join(train_cmd)})")
        subprocess.run(train_cmd, check=True)
        
        lora_path = os.path.join(adapters_dir, f"{target_name}.safetensors")
        if os.path.exists(lora_path):
            print(f"   ✅ MLL Training Complete: {lora_path}")
            return lora_path
            
    except subprocess.CalledProcessError as e:
        print(f"   [-] Training Failed: {e}")
        
    return None

def run_fullmovie_still_mode(xml_path, output_dir, text_engine, args):
    # Enforce Flux 2 Klein for Local Mode (User Request)
    global FLUX_MODEL_PATH
    
    if args.local:
        # Check standard Klein path
        klein_target = os.path.join(MW_ROOT, "flux-root/klein-9b")
        if os.path.exists(klein_target):
             if FLUX_MODEL_PATH != klein_target:
                 print(f"    [🎨] Enforcing Flux 2 Klein: {klein_target}")
                 FLUX_MODEL_PATH = klein_target
        else:
             print(f"    [⚠️] Flux 2 Klein requested but NOT FOUND at {klein_target}. Using default: {FLUX_MODEL_PATH}")
    """
    Ingests an existing XMVP XML and generates a slideshow movie.
    One frame per dialogue line.
    """
    print(f"🎬 FULLMOVIE-STILL ANIMATOR: {xml_path}")
    
    # 1. Load XMVP
    try:
        if xml_path.endswith('.xml'):
            print("    [📦] Loading XMVP XML Manifest...")
            
            # --- MLL INTEGRATION ---
            if args.mll == "on":
                 lora_path = run_mll_pipeline(xml_path, output_dir)
                 if lora_path:
                     print(f"    [💪] Activating MLL Adapter: {lora_path}")
                     os.environ["LOCAL_ADAPTER_PATH"] = lora_path
            # -----------------------

            raw_json = load_xmvp(xml_path, "Manifest")
            
            if not raw_json:
                print("    [!] No <Manifest> key. Checking for <Portions> (Script Mode)...")
                raw_portions = load_xmvp(xml_path, "Portions")
                if raw_portions:
                    # JIT Migration: Convert Portions -> Manifest
                    print("    [Build] Converting Portions to Manifest...")
                    portions = [Portion.model_validate(p) for p in json.loads(raw_portions)]
                    
                    # Flatten Dialogue
                    all_lines = []
                    # Create Segments from Portions as fallback
                    segs = []
                    
                    current_frame = 0
                    fps = 24
                    
                    for p in portions:
                       dur_frames = int(p.duration_sec * fps)
                       segs.append(Seg(
                           id=p.id, 
                           start_frame=current_frame, 
                           end_frame=current_frame + dur_frames,
                           prompt=p.content # Use portion content as visual prompt
                       ))
                       current_frame += dur_frames
                       
                       if p.dialogue:
                           for line in p.dialogue:
                               all_lines.append(line)
                       else:
                               # Fallback: Parse "Character: Text" from content
                           # Expected format: "Name: spoken text..."
                           if ":" in p.content:
                               parts = p.content.split(":", 1)
                               char_name = parts[0].strip()
                               spoken_text = parts[1].strip()
                               
                               if is_valid_character(char_name):
                                   # Create on-the-fly line
                                   fallback_line = DialogueLine(
                                       character=char_name,
                                       text=spoken_text,
                                       action="speaking", # Default action
                                       visual_focus=char_name
                                   )
                                   all_lines.append(fallback_line)
                               
                    manifest = Manifest(
                        segs=segs,
                        dialogue=DialogueScript(lines=all_lines)
                    )
                else:
                    print("[-] No <Manifest> or <Portions> found in XML.")
                    return
            else:
                manifest = Manifest.model_validate_json(raw_json)
        else:
            manifest = load_manifest(xml_path)
    except Exception as e:
        print(f"[-] Failed to load Manifest: {e}")
        return
            
    if not manifest.dialogue or not manifest.dialogue.lines:
        print("[-] XML has no dialogue lines.")
        return

    # 2. Setup Session
    session_id = int(time.time())
    session_dir = os.path.join(output_dir, f"fullmovie_s_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 3. Cast & Voices
    all_chars = set(line.character for line in manifest.dialogue.lines if is_valid_character(line.character))
    print(f"    [👥] Cast: {all_chars}")
    
    # Reuse Voice Logic from Audio Play (Simplified)
    cast_map = {}
    
    # Gender Map
    gender_map = {}
    try:
        raw_story = load_xmvp(xml_path, "Story")
        if raw_story:
            story_obj = Story.model_validate_json(raw_story)
            if story_obj.characters:
                for c_str in story_obj.characters:
                    gender = None
                    if "[Male]" in c_str: gender = "Male"
                    elif "[Female]" in c_str: gender = "Female"
                    
                    if gender:
                        name_part = c_str.split("(")[0].split("[")[0].strip().upper()
                        gender_map[name_part] = gender
    except: pass
    
    if len(gender_map) < len(all_chars):
        scanned_map = scan_for_genders(manifest, all_chars)
        for c, g in scanned_map.items():
            gender_map[c.upper()] = g

    # Assign Voices
    safe_voices_m = ["am_michael", "am_adam", "bm_george", "bm_fable", "bm_lewis", "am_eric"]
    safe_voices_f = ["af_bella", "af_sarah", "af_nicole", "af_sky", "af_jessica", "bf_emma"]
    safe_voices_all = safe_voices_m + safe_voices_f
    
    for char in all_chars:
         clean_char = char.strip().strip('"').strip("'").upper()
         forced_gender = gender_map.get(clean_char)
         subset = safe_voices_m if forced_gender == "Male" else (safe_voices_f if forced_gender == "Female" else safe_voices_all)
         v, p = assign_kokoro_voice_deterministic(char, subset)
         cast_map[char] = {"voice": v, "pitch": p}
         print(f"       + {char} [{forced_gender or '?'}] -> {v}")

    # 4. Generation Loop
    slides = []
    current_time = 0.0
    
    for i, line in enumerate(manifest.dialogue.lines):
        if not is_valid_character(line.character): continue
        print(f"    [{i+1}/{len(manifest.dialogue.lines)}] {line.character}: {line.text[:30]}...")
        
        # Audio
        wav_path = os.path.join(session_dir, f"line_{i:04d}.wav")
        voice_info = cast_map.get(line.character, {"voice": "af_bella", "pitch": 0})
        
        # Text Clean
        clean_text = re.sub(r'[\(\[].*?[\)\]]', '', line.text).strip()
        clean_text = re.sub(r'--+|—', '... ', clean_text) 
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        final_wav = None
        if not clean_text or not any(c.isalnum() for c in clean_text):
             create_silent_wav(wav_path, duration=1.0) # 1 sec silence for empty lines
             final_wav = wav_path
        else:
             final_wav = generate_audio_asset(
                clean_text, wav_path,
                voice_name=voice_info['voice'],
                pitch=voice_info['pitch'],
                mode="kokoro" if args.local else "cloud"
             )
        
        if not final_wav:
             print("       [-] Audio Gen Failed. Skipping.")
             continue
             
        import librosa
        dur = librosa.get_duration(path=final_wav)
        current_time += dur
        
        # Visual
        # Find active segment prompt
        seg = get_segment_for_time(manifest, current_time)
        base_prompt = seg.prompt if seg else f"{line.character} speaking."
        
        vis_prompt = construct_cinematic_prompt(base_prompt, line.character, line.action, line.visual_focus)
        if len(vis_prompt) > 300: vis_prompt = vis_prompt[:297] + "..."
        
        img_path = os.path.join(session_dir, f"frame_{i:04d}.jpg")
        
        # Flux Gen
        if generate_image(vis_prompt, img_path):
             # Stitch to Slide
             slide_mp4 = os.path.join(session_dir, f"slide_{i:04d}.mp4")
             cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-i", final_wav,
                "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-pix_fmt", "yuv420p",
                slide_mp4, "-loglevel", "error"
             ]
             subprocess.run(cmd, check=False)
             if os.path.exists(slide_mp4):
                 slides.append(slide_mp4)
        else:
             print("       [-] Visual Gen Failed.")
             
    # 5. Connect Slides
    if slides:
        final_file = os.path.join(output_dir, f"FullMovieStill-{session_id}.mp4")
        list_file = os.path.join(session_dir, "slides.txt")
        with open(list_file, 'w') as f:
            for s in slides:
                f.write(f"file '{s}'\n")
                
        print(f"   🧵 Stitching {len(slides)} slides...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", final_file, "-loglevel", "error"
        ], check=True)
        print(f"✅ FullMovie-Still Complete: {final_file}")
    else:
        print("❌ No slides generated.")

def stitch_audio_assets(assets, output_mp3):
    """
    Stitches a list of audio assets into a single MP3.
    assets: list of dicts {'audio': path} or paths.
    """
    print(f"   🧵 Stitching {len(assets)} audio segments...")
    
    list_path = output_mp3 + ".txt"
    with open(list_path, 'w') as f:
        for a in assets:
            path = a['audio'] if isinstance(a, dict) else a
            if path and os.path.exists(path):
                f.write(f"file '{path}'\n")
    
    try:
        # Concatenate audio
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:a", "libmp3lame", "-b:a", "192k", "-ac", "2", "-ar", "44100",
            output_mp3, "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        print(f"[+] Audio Play Saved: {output_mp3}")
    except Exception as e:
        print(f"[-] Audio Stitch Failed: {e}")

def run_audio_play_mode(xml_path, output_dir, args):
    """
    Generates an audio-only play from XMVP XML.
    """
    print(f"🎙️ AUDIO-PLAY PRODUCER: {xml_path}")

    # 1. Load XMVP (Reuse FullMovie Logic for Robustness)
    try:
        if xml_path.endswith('.xml'):
            print("    [📦] Loading XMVP XML Manifest...")
            raw_json = load_xmvp(xml_path, "Manifest")
            
            if not raw_json:
                print("    [!] No <Manifest> key. Checking for <Portions> (Script Mode)...")
                raw_portions = load_xmvp(xml_path, "Portions")
                if raw_portions:
                    # JIT Migration: Convert Portions -> Manifest (Simplified for Audio)
                    print("    [Build] Converting Portions to Manifest...")
                    portions = [Portion.model_validate(p) for p in json.loads(raw_portions)]
                    all_lines = []
                    segs = []
                    
                    for p in portions:
                       # Create Segment for Scanning
                       segs.append(Seg(
                           id=p.id,
                           start_frame=0, end_frame=0, # Dummy frames, we only need prompt
                           prompt=p.content
                       ))
                       
                       if p.dialogue:
                           for line in p.dialogue:
                               all_lines.append(line)
                       else:
                           # Fallback: Parse "Character: Text"
                           if ":" in p.content:
                               parts = p.content.split(":", 1)
                               char_name = parts[0].strip()
                               spoken_text = parts[1].strip()
                               
                               if is_valid_character(char_name):
                                   all_lines.append(DialogueLine(
                                       character=char_name,
                                       text=spoken_text,
                                       duration=p.duration_sec
                                   ))
                               
                    manifest = Manifest(segs=segs, dialogue=DialogueScript(lines=all_lines))
                else:
                    print("[-] No <Manifest> or <Portions> found in XML.")
                    return
            else:
                manifest = Manifest.model_validate_json(raw_json)
        else:
            manifest = load_manifest(xml_path)
    except Exception as e:
        print(f"[-] Failed to load Manifest: {e}")
        return

    if not manifest.dialogue or not manifest.dialogue.lines:
        print("[-] XML has no dialogue lines.")
        return

    # 2. Setup Session
    session_id = int(time.time())
    session_dir = os.path.join(output_dir, f"audioplay_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 3. Cast & Voices (Reuse Logic)
    all_chars = set(line.character for line in manifest.dialogue.lines if is_valid_character(line.character))
    print(f"    [👥] Cast: {all_chars}")
    
    # Simplified Voice Assignment (Standard + Route66 check)
    cast_map = {}
    
    # Check for specific templates (Route66)
    target_name = None
    try:
        raw_bib = load_xmvp(xml_path, "Bible") 
        if raw_bib:
            bib = json.loads(raw_bib)
            target_name = bib.get("mll_template")
    except: pass
    
    # Gender Map Logic
    gender_map = {}
    try:
        raw_story = load_xmvp(xml_path, "Story")
        if raw_story:
            story_obj = Story.model_validate_json(raw_story)
            if story_obj.characters:
                for c_str in story_obj.characters:
                    gender = None
                    if "[Male]" in c_str: gender = "Male"
                    elif "[Female]" in c_str: gender = "Female"
                    
                    if gender:
                        name_part = c_str.split("(")[0].split("[")[0].strip().upper()
                        gender_map[name_part] = gender
                        gender_map[f'"{name_part}"'] = gender
    except Exception:
        pass
        
    except Exception:
        pass
        
    print(f"    [⚧️] Gender Map (Metadata): {len(gender_map)} entries")
    
    # Fallback: Script Scan if Map is empty or partial
    if len(gender_map) < len(all_chars):
        scanned_map = scan_for_genders(manifest, all_chars)
        # Merge Scanned into Gender Map (Metadata takes precedence)
        for c, g in scanned_map.items():
            clean_c = c.upper()
            if clean_c not in gender_map and f'"{clean_c}"' not in gender_map:
                gender_map[clean_c] = g
                print(f"       + {c} inferred as {g}")

    if target_name and "Route66" in target_name:
         print("    [🎙️] Route 66 Cast Map (Canonical)...")
         r66_cast = {
            "Amey": {"voice": "af_bella", "pitch": 0},
            "Jessinny": {"voice": "af_sarah", "pitch": 0},
            "Lorrey": {"voice": "af_nicole", "pitch": 0},
            "Mercutio": {"voice": "am_michael", "pitch": 0},
            "Rondio": {"voice": "am_adam", "pitch": 0},
            "Totto": {"voice": "bm_george", "pitch": 0}
         }
         for char in all_chars:
             if char in r66_cast:
                 cast_map[char] = r66_cast[char]
             else:
                 v, p = assign_kokoro_voice_deterministic(char, ["af_bella", "am_adam"])
                 cast_map[char] = {"voice": v, "pitch": p}
    else:
        # Standard Generic with Gender Awareness
        safe_voices_m = ["am_michael", "am_adam", "bm_george", "bm_fable", "bm_lewis", "am_eric"]
        safe_voices_f = ["af_bella", "af_sarah", "af_nicole", "af_sky", "af_jessica", "bf_emma"]
        safe_voices_all = safe_voices_m + safe_voices_f
        
        for char in all_chars:
             clean_char = char.strip().strip('"').strip("'").upper()
             forced_gender = gender_map.get(clean_char)
             
             if forced_gender == "Male":
                 subset = safe_voices_m
             elif forced_gender == "Female":
                 subset = safe_voices_f
             else:
                 subset = safe_voices_all
                 
             v, p = assign_kokoro_voice_deterministic(char, subset)
             cast_map[char] = {"voice": v, "pitch": p}
             g_tag = f"[{forced_gender}]" if forced_gender else "[?]"
             print(f"       + {char} {g_tag} -> {v} (p{p})")

    # 4. Generation Loop
    assets = []
    
    for i, line in enumerate(manifest.dialogue.lines):
        if not is_valid_character(line.character): continue
        
        print(f"    [{i+1}] {line.character}: {line.text[:30]}...")
        
        wav_path = os.path.join(session_dir, f"line_{i:04d}.wav")
        voice_info = cast_map.get(line.character, {"voice": "af_bella", "pitch": 0})
        
        # Strip (...), [...]
        clean_text = re.sub(r'[\(\[].*?[\)\]]', '', line.text).strip()
        # Convert dashes to pauses (ellipsis)
        clean_text = re.sub(r'--+|—', '... ', clean_text)
        # Collapse spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        if not clean_text or not any(c.isalnum() for c in clean_text):
             print(f"       [🔇] Skipping TTS for empty/punctuation line. Adding silence.")
             create_silent_wav(wav_path, duration=0.75)
             final_wav = wav_path
        else:
             final_wav = generate_audio_asset(
                clean_text, wav_path,
                voice_name=voice_info['voice'],
                pitch=voice_info['pitch'],
                mode="kokoro" if args.local else "cloud"
             )
        
        if final_wav:
             assets.append({"audio": final_wav})
             
        # Add Pause (Pacing)
        # Default 0.5s silence between lines for flow
        pause_wav = os.path.join(session_dir, f"pause_{i:04d}.wav")
        create_silent_wav(pause_wav, duration=0.5)
        assets.append({"audio": pause_wav})
        
    # 5. Stitch
    final_mp3 = os.path.join(output_dir, f"AudioPlay-{session_id}.mp3")
    stitch_audio_assets(assets, final_mp3)
    return

# run_audio_movie_mode removed (deprecated — depended on skyreels_bridge which has been deleted)

def create_silent_wav(path, duration=1.0):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono", "-t", str(duration), path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def pad_audio_file(path, pad_duration):
    """Appends silence to the end of a wav file."""
    if not os.path.exists(path): return
    
    tmp_path = path.replace(".wav", "_padded.wav")
    try:
        # pad_dur implies adding duration. 
        # filter: apad=pad_dur=X adds X seconds.
        subprocess.run([
            "ffmpeg", "-y", "-i", path, "-af", f"apad=pad_dur={pad_duration}", 
            "-c:a", "pcm_s16le", tmp_path, "-loglevel", "error"
        ], check=True)
        
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[-] Failed to pad audio {path}: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)

def main():
    global LOCAL_MODE, FOLEY_ENABLED

    parser = argparse.ArgumentParser(description="Content Producer v0.5 (Unified)")
    
    # Global/Shared Positional Args
    import definitions
    definitions.add_global_vpform_args(parser)
    
    parser.add_argument("--vpform", default=None, help="Vision Platonic Form (e.g. 24-podcast, gahd-podcast)")
    parser.add_argument("--project", help="Project override (stub)")
    parser.add_argument("--ep", type=int, help="Episode number (stub)")
    parser.add_argument("--local", action="store_true", help="Use Local Engines (Flux + Kokoro)")
    parser.add_argument("--foley", choices=["on", "off"], default="off", help="Enable Generative Foley")
    parser.add_argument("--slength", type=float, default=0.0, help="Override duration in seconds")
    parser.add_argument("--fc", action="store_true", help="Code Painter Mode (Experimental)")
    parser.add_argument("--geminiapi", action="store_true", help="Force Cloud Gemini API for Text (Disable Local Gemma default)")
    parser.add_argument("--band", type=str, help="Band Name (for thax-douglas mode)")
    parser.add_argument("--poem", type=str, help="Poem Text (for thax-douglas mode)")
    parser.add_argument("--w", type=int, default=512, help="Width (Default: 512)")
    parser.add_argument("--h", type=int, default=288, help="Height (Default: 288)")
    parser.add_argument("--location", type=str, help="Override visual location (e.g. 'a haunted hotel')")
    parser.add_argument("--rvc", action="store_true", help="Enable RVC (Retrieval Voice Conversion) for Cast")
    parser.add_argument("--xml", type=str, help="Input XMVP XML path (for fullmovie-still mode)")
    parser.add_argument("--xb", type=str, help="Alias for --xml (XMVP Bible/Manifest)")
    parser.add_argument("--mu", type=str, help="Master Audio Input Path (for audio-movie mode)")
    parser.add_argument("--mll", choices=["on", "off"], default="off", help="Enable/Disable Movie-Level LoRA Training Step")
    parser.add_argument("--out", type=str, help="Output directory override")
    parser.add_argument("--seeds", type=str, help="Explicit list of Chaos Seeds (e.g. \"['Topic A', 'Topic B']\")")
    
    args, unknown = parser.parse_known_args()
    
    # Handle aliases via registry
    # Handle aliases via registry
    # If args.vpform is set (e.g. element-47), we try to resolve it. 
    # If definitions doesn't know it, it returns None, and we fall back to current_default.
    resolved = definitions.parse_global_vpform(args, current_default=None)
    if resolved:
        args.vpform = resolved
    elif args.vpform and args.vpform == "element-47":
        # Keep it if it was explicit but unknown to definitions (safety)
        pass
    else:
        # Only default if nothing resolved
        args.vpform = "24-podcast"
    
    print(f"DEBUG: Args Parsed. VPForm: {args.vpform}")
    
    if args.out:
        OUTPUT_DIR = os.path.abspath(args.out)
        print(f"    [📂] Output Dir Override: {OUTPUT_DIR}")
    else:
        # Re-assert default if needed, or rely on global (but we must assign to avoid UnboundLocalError if we assigned in 'if')
        # Actually simplest is just to set it to global default if not set.
        pass 
        # Wait, 'pass' doesn't help if Python sees an assignment later? 
        # The assignment 'OUTPUT_DIR = ...' makes it local for the whole function.
        # So we MUST assign it in the else branch to the default value.
        OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "z_test-outputs"))


    
    LOCAL_MODE = args.local
    FOLEY_ENABLED = (args.foley == "on")
    
    # Set Global Generation Dims
    global GENERATION_DIMS
    GENERATION_DIMS = (args.w, args.h)

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

        return


    # --- ELEMENT 47 ---
    if args.vpform == "element-47":
        if args.xb:
            manifest_path = args.xb
            if manifest_path.lower().endswith(".fountain"):
                print(f"    [🔄] Fountain Input Detected. Converting to XMVP XML...")
                import sys
                cmd = [sys.executable, "xmvp_converter.py", manifest_path, "--vpform", "element-47"]
                if args.local: cmd.append("--local")
                
                try:
                    subprocess.run(cmd, check=True)
                    # Predict Output Path
                    # xmvp_converter saves as {input}_manifest.xml by default adjacent to input
                    base_path = os.path.splitext(manifest_path)[0]
                    manifest_path = f"{base_path}_manifest.xml"
                    
                    if not os.path.exists(manifest_path):
                        print(f"[-] Conversion failed. Expected manifest not found: {manifest_path}")
                        return
                    print(f"    [✅] Conversion Complete. Using: {manifest_path}")
                except subprocess.CalledProcessError as e:
                    print(f"[-] Conversion crashed: {e}")
                    return

            run_element47_production(manifest_path, OUTPUT_DIR, args)
        else:
             print("[-] --xb (XMVP Manifest) required for element-47 production.")
        return

    # --- KOKORO SETUP (Audio Pacing) ---
    # Fix for macOS: Ensure espeak-ng library is found
    # Only if not already set
    if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
        candidates = [
            "/opt/homebrew/lib/libespeak-ng.dylib",
            "/usr/local/lib/libespeak-ng.dylib",
            "/usr/lib/libespeak-ng.dylib"
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"    [🔊] Configured Espeak Library: {c}")
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = c
                break

    # 0.5 FullMovie Still Mode (and Black Box)
    if args.vpform == "fullmovie-still" or args.vpform == "black-box":
        xml_input = args.xml or args.xb
        if not xml_input:
            print(f"[-] --xml (or --xb) argument required for {args.vpform} mode.")

            return
        run_fullmovie_still_mode(xml_input, OUTPUT_DIR, None, args)
        return

    if args.vpform == "audio-play":
        xml_input = args.xml or args.xb
        if not xml_input:
             print("[-] --xml (or --xb) required for audio-play.")
             return
        run_audio_play_mode(xml_input, OUTPUT_DIR, args)
        return

    if args.vpform == "audio-movie":
        print("[-] audio-movie VPForm has been deprecated (skyreels_bridge removed). Will return in a future version.")
        return

    if args.vpform and ("podcast" in args.vpform or "cartoon" in args.vpform):
        # 1. Generative Modes (GAHD, 24-podcast, 10-podcast, Route 66)
        if "gahd" in args.vpform or "24" in args.vpform or "10" in args.vpform or "route66" in args.vpform:
            text_engine = TextEngine()
            run_improv_session(args.vpform, OUTPUT_DIR, text_engine, args)
        
        else:
             print(f"[-] Unknown/Legacy VPForm '{args.vpform}'. Generative mode only.")
             # Removed run_podcast_processing call
    else:
        print("[-] No valid VPForm specified. Use --vpform [route66-podcast|gahd-podcast|24-podcast|10-podcast]")

if __name__ == "__main__":
    main()
