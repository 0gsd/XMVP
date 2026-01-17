#!/usr/bin/env python3
"""
improv_animator.py
------------------
The "Improv Animator": A self-contained, zero-touch improv comedy generator.
Feautures 4 recurring characters (William, Maggie, Francis, Anne Tailored)
who riff on random Wikipedia Chaos Seeds for 24 minutes.

Pipeline:
1. Chaos Seeds: Fetches 6 random Wikipedia titles.
2. Improv Loop (Streaming):
   - Writer: TextEngine generates Speaker/Dialogue/Action.
   - Painter: Gemini generates Black Box Theater scene.
   - Speaker: Cloud TTS (Journey) synthesizes audio.
3. Stitch: Combines assets into final 24-minute special.

Author: 0i0
Date: Jan 2026
"""

import os
import sys
import time
import json
import random
import logging
import re
import math
import shutil
import subprocess
import base64
import requests
import itertools
from datetime import datetime

import argparse

# MVP Shared Imports
try:
    from mvp_shared import get_client, setup_logging, get_project_id
    from mvp_shared import VPForm, CSSV, Constraints, Story, Portion, save_xmvp
    from text_engine import TextEngine
    from vision_producer import get_chaos_seed
    from truth_safety import TruthSafety
    import google.genai as genai
    from google.genai import types
    from foley_talk import generate_audio_asset # Integration!
    import frame_canvas # Support FC Mode
except ImportError as e:
    print(f"[-] Critical Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---

OUTPUT_DIR = os.path.join("z_test-outputs", "improvs")
TARGET_DURATION_SEC = 24 * 60  # 24 Minutes
TTS_MODEL_ID = "gemini-2.5-flash-tts" # Not used (Rest API preference)
IMAGE_MODEL_ID = "gemini-2.5-flash-image"

# --- FORM DEFINITIONS ---

FORM_DEFS = {
    "24-cartoon": {
        "description": "4-Person Improv Comedy Special (24 Minutes)",
        "cast": {
            "William": {
                "base": "Billy Joel",
                "voice": "en-US-Journey-D",
                "pitch": 1,
                "persona": "The Piano Man. Working-class poet. Melodic, specific geographic references (Long Island), cynical but soulful."
            },
            "Maggie": {
                "base": "Margaret Thatcher",
                "voice": "en-US-Journey-F",
                "pitch": -2,
                "persona": "The Iron Lady. Stern, authoritative, uses 'Royal We'. Surprisingly willing to play high-status absurd characters."
            },
            "Francis": {
                "base": "Frank Sinatra",
                "voice": "en-US-Journey-D",
                "pitch": -2,
                "persona": "The Chairman. Cool, swaggering, mid-Atlantic accent. emotional, volatile, calls everyone 'baby'. Does it 'My Way'."
            },
            "Anne Tailored": {
                "base": "Taylor Swift",
                "voice": "en-US-Journey-F",
                "pitch": 1,
                "persona": "The Pop Icon. Earnest, confessional, detailed storytelling. Bridges scenes with emotional hooks. Avoids copyright infringement."
            }
        },
        "system_prompt_template": (
            "You are the Director of a long-form Improv Comedy show in a Black Box Theater.\n"
            "The Cast:\n{cast_desc}\n\n"
            "The Rules:\n"
            "1. Generate ONE turn at a time (Speaker + Dialogue + Physical Action).\n"
            "2. Maintain a coherent 24-minute narrative arc weaving through the Chaos Seeds.\n"
            "3. Style: 'Yes, And', witty, character-driven, referencing the seeds.\n"
            "4. Format: JSON {{ 'speaker': 'Name', 'text': 'Dialogue', 'action_prompt': 'Visual description of action in theater', 'visual_focus': 'Focus of the shot' }}\n"
        )
    }
}

# Default Fallback

COURTROOM_STYLE_PROMPT = (
    "A photorealistic courtroom sketch. Drawn by a preternaturally talented artist "
    "in an underground black box theater. Hyperrealistic style, clean commercial art. "
    "NO WORDS, NO SPEECH BUBBLES, NO TEXT. "
    "The artist captures the performers improvising, surrounded by the imaginary objects "
    "they are describing, depicted as real physical things in the space. "
    "Dynamic composition, intense energy."
)

# Default Fallback
CAST = FORM_DEFS["24-cartoon"]["cast"]

def generate_dynamic_cast(text_engine, seeds):
    """Generates a dynamic 4-person cast based on Chaos Seeds."""
    print("    [✨] Casting Call: Generating Dynamic Personas...")
    
    # We need 4 personas. We have 6 seeds. Use first 4.
    selected_seeds = seeds[:4]
    
    # Fixed Voice Slots (2M, 2F)
    slots = [
        {"voice": "en-US-Journey-D", "pitch": 1, "gender": "Male"},
        {"voice": "en-US-Journey-F", "pitch": -2, "gender": "Female"},
        {"voice": "en-US-Journey-D", "pitch": -2, "gender": "Male"},
        {"voice": "en-US-Journey-F", "pitch": 1, "gender": "Female"}
    ]
    
    dynamic_cast = {}
    
    for i, seed in enumerate(selected_seeds):
        slot = slots[i]
        prompt = (
            f"Create a weird, specific improv character persona inspired by the concept: '{seed}'.\n"
            f"Gender: {slot['gender']}.\n"
            "Format JSON: { 'name': 'First Last', 'persona': 'Short pithy description (visuals + personality)' }"
        )
        
        try:
            raw = text_engine.generate(prompt, json_schema=True)
            data = json.loads(raw)
            name = data.get("name", f"Player {i+1}")
            persona = data.get("persona", f"Improviser inspired by {seed}")
            
            dynamic_cast[name] = {
                "base": name, # Metadata placeholder
                "voice": slot["voice"],
                "pitch": slot["pitch"],
                "persona": persona
            }
            print(f"       + Cast {name}: {persona[:50]}...")
        except Exception as e:
            print(f"       [-] Failed to cast slot {i}: {e}")
            name = f"Improviser {i+1}"
            dynamic_cast[name] = {
                "base": "Unknown", 
                "voice": slot["voice"], 
                "pitch": slot["pitch"], 
                "persona": "A mysterious stranger."
            }
            
    return dynamic_cast


# --- HELPERS ---

def get_access_token():
    """Fetches GCP Access Token via CLI."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"[-] Failed to get access token: {e}")
        return None

# Audio generation logic migrated to foley_talk.py
# Only get_audio_duration helper retained locally or imported if desired (but foley_talk has it too)
# We will rely on foley_talk's generate_audio_asset


def generate_image(prompt, output_path):
    """Generates an image using Gemini 2.5 Flash."""
    max_retries = 3
    for attempt in range(max_retries):
        client, key_used = get_client()
        try:
            # Prompt is now fully constructed by caller
            final_prompt = f"{prompt} --aspect_ratio 1:1"
            
            response = client.models.generate_content(
                model=IMAGE_MODEL_ID, 
                contents=final_prompt
            )
            
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        with open(output_path, "wb") as f:
                            f.write(part.inline_data.data)
                        return True
            print(f"   [-] No image data returned.")
            
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 * (attempt + 1))
            else:
                print(f"   [-] Image Gen Failed: {e}")
                
    # Fallback Black Frame
    subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=1024x1024:d=0.1', '-frames:v', '1', output_path, '-y', '-loglevel', 'error'])
    return False

def get_audio_duration(file_path):
    """Get precise duration."""
    try:
        cmd = ['ffprobe', '-i', file_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0.0

# --- CORE LOOP ---

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Improv Animator: The 24-Minute Special")
    parser.add_argument("--project", help="GCP Project ID for billing override (TTS)")
    parser.add_argument("--vpform", default="24-cartoon", help="Vision Platonic Form (e.g. 24-cartoon)")
    parser.add_argument("--slength", type=float, default=0.0, help="Override duration in seconds")
    parser.add_argument("--fc", action="store_true", help="Enable Frame & Canvas (Code Painter) Mode")
    args = parser.parse_args()

    print("🎭 IMPROV ANIMATOR: The 24-Minute Special")
    
    # 1. Setup Session
    session_id = int(time.time())
    session_dir = os.path.join(OUTPUT_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    # 2. Get Chaos Seeds
    print("[*] Fetching Chaos Seeds...")
    seeds = []
    for i in range(6):
        seed = get_chaos_seed()
        seeds.append(seed)
        time.sleep(0.5)
    print(f"    Seeds: {seeds}")
    
    # 3. Initialize Engines
    # TextEngine init is just TextEngine(config_path=...) or empty.
    # It does NOT take model_name in init, it uses .generate()
    text_engine = TextEngine() 
    print(f"    Loaded {len(text_engine.api_keys)} API Keys for Rotation.")
    
    access_token = get_access_token()
    
    # Project Priority: CLI Arg > Env Var > default
    project_id = args.project if args.project else get_project_id()
    
    # 4. Select Form
    form_key = args.vpform
    
    # dynamic duration override
    target_duration = TARGET_DURATION_SEC
    
    if form_key == "10-cartoon":
        print(f"    Selected Form: {form_key} (Dynamic)")
        print(f"    Description: 10-Minute Dynamic Improv Set")
        target_duration = 10 * 60
        
    if args.slength > 0:
        target_duration = args.slength
        print(f"[*] Duration overridden to {target_duration}s")
        
        # Generator
        cast = generate_dynamic_cast(text_engine, seeds)
        
        # Synthesize system prompt
        cast_desc = "\n".join([f"- {name}: {data['persona']}" for name, data in cast.items()])
        # Borrow template from 24-cartoon for now
        form_data = FORM_DEFS["24-cartoon"] # Use as base for template
        
    elif form_key in FORM_DEFS:
        form_data = FORM_DEFS[form_key]
        cast = form_data["cast"]
        cast_desc = "\n".join([f"- {name}: {data['persona']}" for name, data in cast.items()])
        print(f"    Selected Form: {form_key}")
        print(f"    Description: {form_data['description']}")
    
    else:
        print(f"[-] Unknown VPForm: {form_key}. Defaulting to 24-cartoon.")
        form_key = "24-cartoon"
        form_data = FORM_DEFS["24-cartoon"]
        cast = form_data["cast"]
        cast_desc = "\n".join([f"- {name}: {data['persona']}" for name, data in cast.items()])

    
    if not access_token:
        print("[-] Verification Failed: No GCP Access Token.")
        return
        
    print(f"    GCP Project: {project_id}")

    # 4. The Loop
    print(f"[*] Starting Improv Set (Target: {TARGET_DURATION_SEC}s)...")
    
    total_duration = 0.0
    turn_count = 0
    history = [] # List of {speaker, text}
    assets = [] # List of {audio, image, duration}
    
    # Pre-warm System Prompt
    # cast_desc already computed above
    system_prompt = form_data["system_prompt_template"].format(cast_desc=cast_desc)
    
    # Pre-warm Visual Context
    # "William: [Persona]..."
    cast_visual_context = f"CAST CONTEXT (The Performer Appearances):\n{cast_desc}"
    
    current_seed_idx = 0
    
    try:
        while total_duration < target_duration:
            # A. Manage Seeds
            # Distribute 6 seeds over 24 mins -> 1 seed every 4 mins (240s)
            target_seed_idx = int(total_duration // 240)
            if target_seed_idx < len(seeds):
                current_seed = seeds[target_seed_idx]
            else:
                current_seed = "The Grand Finale / Callbacks"
                
            # B. Write Line
            prompt = (
                f"Current Time: {total_duration:.1f}s / {target_duration}s.\n"
                f"Current Seed Inspiration: {current_seed}\n"
                f"Recent History:\n" + 
                "\n".join([f"{h['speaker']}: {h['text']}" for h in history[-10:]]) + 
                "\n\nGeneratre the next turn."
            )
            
            # Retry loop for JSON
            turn_data = None
            # Retry loop for JSON with 429 Backoff
            turn_data = None
            for attempt_idx in range(5): # Increased to 5
                try:
                    # TextEngine.generate() does not support system_instruction kwarg. 
                    # We must prepend it to the prompt manually.
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                    raw = text_engine.generate(full_prompt, json_schema=True)
                    
                    turn_data = json.loads(raw)
                    if turn_data['speaker'] not in cast:
                        print(f"    [!] Unknown speaker: {turn_data['speaker']}")
                        continue
                    break
                    
                except Exception as e:
                    # Check for 429 in error string
                    err_str = str(e)
                    if "429" in err_str:
                        # AGGRESSIVE BACKOFF
                        print(f"    [!] Rate Limit (429). Cooling down for 30s to preserve quality...")
                        time.sleep(30.0)
                        
                        sleep_time = (attempt_idx + 1) * 3  # Add standard backoff on top just in case
                        time.sleep(sleep_time)
                    else:
                        print(f"    [!] Writer Error (Attempt {attempt_idx}): {e}")
                        time.sleep(1)
                    continue
                    
                    if turn_data['speaker'] not in cast:
                        print(f"    [!] Unknown speaker: {turn_data['speaker']}")
                        continue
                        
                    break
                except Exception as e:
                    print(f"    [!] Writer Error (Attempt {attempt_idx}): {e}")
                    # print(f"    [!] Raw Output: {raw if 'raw' in locals() else 'None'}")
                    time.sleep(1)
                    continue
            
            if not turn_data:
                print("    [!] Writer blocked. Skipping turn.")
                continue
                
            speaker = turn_data['speaker']
            text = turn_data['text']
            action = turn_data.get('action_prompt', 'Standing in the black box theater.')
            visual = turn_data.get('visual_focus', speaker)
            
            print(f"    [{turn_count}] {speaker}: {text[:40]}...")
            
            # Update History
            history.append({'speaker': speaker, 'text': text})
            
            # C. Synthesize Audio
            if speaker not in cast:
                # Fallback for hallucinated speakers
                print(f"    [!] Unknown speaker: {speaker}. Assigning to random cast member.")
                speaker = random.choice(list(cast.keys()))
            
            c_data = cast[speaker]
            voice = c_data['voice']
            pitch = c_data['pitch']
            
            # Audio Gen
            wav_path = os.path.join(session_dir, f"turn_{turn_count:04d}.wav")
            # synthesize_text_rest returns audio_bytes directly in our impl, wait, checking def...
            # Yes: returns base64.b64decode...
            # Wait, I modified it to return (data, token) tuple? 
            # Let's check my implementation above... yes "return None, token" / "return data" ?? 
            # Ah, in the code block above I wrote "return None, token". 
            # But the success path returned "return base64..." -> SINGLE value.
            # I must fix this inconsistency in the function definition locally before running.
            
            # Fixing logic inline:
            # New Foley Talk API:
            # generate_audio_asset(text, output_path, voice_name=..., pitch=..., mode="cloud", project_id=...)
            
            # Using Cloud mode default
            final_wav = generate_audio_asset(
                text, 
                wav_path, # It handles returning the final path, potentially suffixed
                voice_name=voice, 
                pitch=pitch, 
                mode="cloud", 
                project_id=project_id
            )

            if not final_wav or not os.path.exists(final_wav):
                print("       [!] Audio failed.")
                continue
            
            # No manual pitch shift needed! foley_talk handles it.
            
            duration = get_audio_duration(final_wav)
            duration = get_audio_duration(final_wav)
            # Add small pauses
            duration += 0.5 
            
            # D. Paint Image
            img_path = os.path.join(session_dir, f"turn_{turn_count:04d}.jpg")
            
            # Construct Courtroom Prompt
            full_visual_prompt = (
                f"{COURTROOM_STYLE_PROMPT}\n\n"
                f"{cast_visual_context}\n\n"
                f"CURRENT ACTION:\n"
                f"Speaker: {speaker}\n"
                f"Action/Vibe: {action}\n"
                f"Visual Focus: {visual}"
            )
            
            if args.fc:
                 # CODE PAINTER MODE
                 print(f"       [FC] Painted via Code: {os.path.basename(img_path)}...")
                 
                 # Construct Context
                 # Global: Cast + Setup
                 # Local: Action + Visual
                 
                 ctx = {
                     "global_concept": f"Improv Comedy Show. Cast:\n{cast_desc}\nStyle: {COURTROOM_STYLE_PROMPT}",
                     "frame_index": turn_count + 1,
                     "current_speaker": speaker,
                     "current_action": action,
                     "visual_focus": visual
                 }
                 
                 # Hack: text_engine is in scope
                 model_inst = text_engine.get_model_instance()
                 
                 img = frame_canvas.generate_via_code(
                     prompt=full_visual_prompt,
                     width=1024, height=1024,
                     context=ctx,
                     model=model_inst
                 )
                 
                 if img:
                     img.save(img_path)
                 else:
                     print("       [!] FC Paint Failed. Fallback to Black Frame logic?")
                     # Fallback Black Frame
                     subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=1024x1024:d=0.1', '-frames:v', '1', img_path, '-y', '-loglevel', 'error'])
            
            elif not generate_image(full_visual_prompt, img_path):
                # generate_image creates a Black Frame on failure. 
                # We attempt to overwrite it with a Stutter Step (Clone Previous) if possible.
                if turn_count > 0:
                     prev_img = os.path.join(session_dir, f"turn_{turn_count-1:04d}.jpg")
                     if os.path.exists(prev_img):
                         try:
                             shutil.copy(prev_img, img_path)
                             print(f"       [!] Stuttered: Cloned {os.path.basename(prev_img)} -> {os.path.basename(img_path)}")
                         except Exception as e:
                             print(f"       [-] Stutter Clone Failed: {e}")
                
            # E. Save Asset Record
            assets.append({
                "audio": final_wav,
                "image": img_path,
                "duration": duration,
                "speaker": speaker,
                "text": text,
                "action": action
            })
            
            total_duration += duration
            turn_count += 1
            print(f"       [+] Duration: {duration:.1f}s | Total: {total_duration:.1f}s")
            
            # Sleep to respect quotas
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Stitching what we have...")
        
    # 5. Stitching
    print("[*] Stitching Final Video...")
    list_path = os.path.join(session_dir, "input_list.txt")
    with open(list_path, 'w') as f:
        for asset in assets:
            # FFMpeg Concat format
            # file 'image'
            # duration X
            # file 'audio' ??? FFMpeg concat demuxer is tricky with separate A/V.
            # Better approach: Create mini-video clips for each turn, then concat mp4s.
            pass
            
    # Revised Stitching Strategy: Create MP4s per turn
    mp4_files = []
    for i, asset in enumerate(assets):
        mp4_path = os.path.join(session_dir, f"clip_{i:04d}.mp4")
        # Stitch Image + Audio
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', asset['image'], '-i', asset['audio'],
            '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p', '-shortest', mp4_path, '-loglevel', 'error'
        ]
        subprocess.run(cmd)
        mp4_files.append(mp4_path)
        
    final_output = os.path.join(OUTPUT_DIR, f"Improv_Special_{session_id}.mp4")
    with open(list_path, 'w') as f:
        for mp4 in mp4_files:
            f.write(f"file '{os.path.abspath(mp4)}'\n")
            
    # Final Concat
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
        '-c', 'copy', final_output, '-loglevel', 'error'
    ])
    
    print(f"\n[+] SHOW COMPLETE: {final_output}")
    print(f"    Total Duration: {total_duration/60:.1f} minutes")
    
    # 6. XMVP Export
    print("\n[*] Generating XMVP Manifest (The Bible)...")
    
    # Retry Loop for Summary Generation (often hits 429 at end of run)
    story_data = {"title": f"Improv Special {session_id}", "synopsis": "A raw improv set.", "themes": ["Comedy"]}
    
    for attempt in range(3):
        try:
            # A. Summarize Story
            story_prompt = (
                "Analyze the improv transcript below and generate a Movie Metadata JSON.\n"
                "Title: A catchy title for this improv special.\n"
                "Synopsis: A summary of the chaotic narrative arc.\n"
                "Themes: 3 key themes.\n"
                f"Transcript:\n" + 
                "\n".join([f"{a['speaker']}: {a.get('text','')}" for a in assets[-50:]]) + # Limit to last 50 lines to save tokens
                "\n\nOutput JSON: {title, synopsis, themes}"
            )
            
            raw_story = text_engine.generate(story_prompt, json_schema=True)
            story_data = json.loads(raw_story)
            break
        except Exception as e:
            if "429" in str(e):
                print(f"    [!] 429 during Summary. Sleeping 30s... (Attempt {attempt+1})")
                time.sleep(30)
            else:
                print(f"    [!] Summary Gen Error: {e}")
                break

    try:
        # B. Construct Models
        vp = VPForm(
            name=form_key,
            fps=24,
            description="Improv Animator Session Export",
            mime_type="video/mp4"
        )
        
        cssv = CSSV(
            constraints=Constraints(width=1024, height=1024, fps=24, target_segment_length=4.0),
            scenario=f"Improv based on seeds: {', '.join(seeds)}",
            situation=story_data.get('synopsis', 'An improv comedy special.'),
            vision=COURTROOM_STYLE_PROMPT
        )
        
        story = Story(
            title=story_data.get('title', f"Improv Special {session_id}"),
            synopsis=story_data.get('synopsis', 'Unscripted chaos.'),
            characters=list(cast.keys()),
            theme=story_data.get('themes', ['Improv', 'Comedy'])[0]
        )
        
        portions = []
        for i, asset in enumerate(assets):
            portions.append(Portion(
                id=i+1,
                duration_sec=asset['duration'],
                content=f"{asset['speaker']}: {asset.get('text','')} [Action: {asset.get('action','')}]"
            ))
            
        xmvp_data = {
            "VPForm": vp,
            "CSSV": cssv,
            "Story": story,
            "Portions": [p.model_dump() for p in portions],
            "ChaosSeeds": seeds
        }
        
        xml_name = f"manifest_{session_id}.xml"
        xml_path = os.path.join(session_dir, xml_name)
        save_xmvp(xmvp_data, xml_path)
        print(f"    📜 Manifest Saved: {xml_path}")
        
    except Exception as e:
        print(f"[-] XMVP Export Failed: {e}")

    # 7. Cleanup (Preserve First Frame & Manifest)
    print("\n[🧹] Cleaning up temporary assets...")
    try:
        cleaned_count = 0
        for f in os.listdir(session_dir):
            f_path = os.path.join(session_dir, f)
            
            # Preserve Manifest
            if f.endswith(".xml"): continue
            
            # Preserve First Frame
            if "turn_0000" in f or "frame_0000" in f:
                continue
            
            # Delete Temp Media
            # Delete Temp Media
            # Keep Audio as requested
            if f.lower().endswith(('.jpg', '.png', '.mp4')):
                # SKIP if it is the Final Output (unlikely inside session dir but safe check)
                if "Improv_Special" in f: continue
                os.remove(f_path)
                cleaned_count += 1
                
        print(f"      - Removed {cleaned_count} temp files.")
        
    except Exception as e:
        print(f"[-] Cleanup Warning: {e}")

if __name__ == "__main__":
    main()
