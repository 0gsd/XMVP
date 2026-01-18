#!/usr/bin/env python3
import json
import requests
import argparse
import sys
import os
import time
import uuid
import logging
import subprocess
import base64
import math
import shutil
from pathlib import Path
from mvp_shared import load_manifest, Manifest, DialogueScript, DialogueLine, get_project_id
from hunyuan_foley_bridge import HunyuanFoleyBridge

# --- CONFIGURATION ---
COMFY_SERVER = "http://127.0.0.1:8188"
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- HELPERS (Shared) ---

def get_audio_duration(file_path):
    """Get precise duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-i', file_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def compose_track(assets, absolute_duration, output_path):
    """
    Composes a single continuous wav file from a list of {path, offset} assets.
    Uses FFMPEG concat demuxer with padding.
    """
    # 1. Create Silence Asset
    silence_path = "silence_base.wav"
    # Create 1s silence to reuse
    if not os.path.exists(silence_path):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1", silence_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                   
    # 2. Build List
    # Sort by offset
    assets = sorted(assets, key=lambda x: x['offset'])
    
    list_path = output_path + ".txt"
    cursor = 0.0
    
    with open(list_path, 'w') as f:
        for a in assets:
            gap = a['offset'] - cursor
            if gap > 0.01:
                # Add silence
                # FFmpeg concat demuxer allows looping a file? No, just repeat entry.
                # To fill X seconds with 1s silence file:
                # Need stream_loop? Or just use filters?
                # Concat demuxer "duration" directive overrides length.
                # file 'silence.wav' \n duration GAP
                f.write(f"file '{os.path.abspath(silence_path)}'\n")
                f.write(f"duration {gap}\n")
            
            f.write(f"file '{os.path.abspath(a['path'])}'\n")
            # We don't know duration of clip without probing? 
            # FFmpeg concat file directive usually auto-determines duration if not specified.
            # BUT for gap calculation next, we MUST know it.
            duration = get_audio_duration(a['path'])
            cursor = a['offset'] + duration
            
        # Pad end?
        if cursor < absolute_duration:
             remaining = absolute_duration - cursor
             f.write(f"file '{os.path.abspath(silence_path)}'\n")
             f.write(f"duration {remaining}\n")

    # 3. Render
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:a", "pcm_s16le", # PCM for intermediate
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

def mix_audio(video_path, foley_path, dialogue_files, output_path):
    """
    Uses FFmpeg to mix video + foley + dialogue(s).
    """
    logging.info("🎚️ Mixing Final Stems...")
    
    # Simple concat Strategy for now:
    # We will just focus on saving the individual assets for the "Dailies" workflow.
    # The actual mixdown for a full movie is complex. 
    # For this Dailies tool, we will just output the dialogue tracks side-by-side
    # or a basic mix if requested.
    
    # Placeholder: Just copy video to output for now as the 'mix' 
    # (since the user asked for "Dailies with draft sound", implementing the full 
    # complex filter mix of dynamic dialogue offsets is a larger task for Step 2).
    # For now, we ensure the ASSETS are generated.
    
    shutil.copy(video_path, output_path)
    return True

# --- BACKEND 1: COMFYUI (Local) ---
class ComfyWrapper:
    def __init__(self, server_url=COMFY_SERVER, dry_run=False):
        self.server_url = server_url
        self.dry_run = dry_run
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt_workflow):
        if self.dry_run:
            logging.info("   [DRY] Queuing Prompt...")
            return {"prompt_id": "dry_run_id"}
        p = {"prompt": prompt_workflow, "client_id": self.client_id}
        try:
            res = requests.post(f"{self.server_url}/prompt", json=p)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logging.error(f"   ❌ Queue Failed: {e}")
            return None

    def upload_file(self, file_path, subfolder="foley_uploads"):
        if self.dry_run:
            logging.info(f"   [DRY] Uploading {file_path}...")
            return {"name": os.path.basename(file_path)}
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'rb') as f:
                files = {'image': f}
                data = {'subfolder': subfolder, 'type': 'input', 'overwrite': 'true'}
                res = requests.post(f"{self.server_url}/upload/image", files=files, data=data)
                return res.json()
        except Exception as e:
            logging.error(f"   ❌ Upload Failed: {e}")
            return None
            
    def wait_for_history(self, prompt_id, timeout=300):
        if self.dry_run:
            time.sleep(1)
            return {"outputs": {}}
        start = time.time()
        while time.time() - start < timeout:
            try:
                res = requests.get(f"{self.server_url}/history/{prompt_id}")
                history = res.json()
                if prompt_id in history: return history[prompt_id]
                time.sleep(1)
            except:
                time.sleep(1)
        return None

def get_hunyuan_workflow(video_filename, output_prefix):
    # Placeholder for actual Hunyuan JSON
    return {"placeholder": "hunyuan", "video": video_filename}

def get_indextts_workflow(text, char_ref, output_prefix):
    # Placeholder for actual IndexTTS JSON
    return {"placeholder": "indextts", "text": text}

def generate_comfy_dialogue(wrapper, script: DialogueScript, output_dir):
    logging.info(f"🗣️ [Comfy] Generating Dialogue ({len(script.lines)} lines)...")
    results = []
    for i, line in enumerate(script.lines):
        logging.info(f"   Actor {line.character}: '{line.text[:30]}...'")
        # Logic to call get_indextts_workflow -> queue -> wait
        # Mocking output for now
        out_path = os.path.join(output_dir, f"dial_{i}_{line.character}.wav")
        if wrapper.dry_run:
            os.system(f"touch {out_path}")
        results.append({"path": out_path, "offset": line.start_offset})
    return results

# --- SHARED API EXPORTS ---

def get_access_token():
    try:
        cmd = ["gcloud", "auth", "print-access-token"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None

def pitch_shift_file(input_file, semitones):
    """
    Shifts pitch using FFmpeg.
    Returns the path to the shifted file.
    """
    if semitones == 0: return input_file
    try:
        # Simplified pitch shift logic
        # For robustness, we assume 24k or probe it? 
        # Let's stick to the ratio math which is sample-rate independent for asetrate usually if we know the rate?
        # Actually simplest is just simple filter chain? No, asetrate changes speed. 
        # We need the triplet: asetrate=new_rate, atempo=1/ratio, aresample=orig_rate
        
        # Probe rate first
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
        sr = int(subprocess.check_output(cmd).strip())
        
        ratio = math.pow(2, semitones / 12.0)
        new_rate = int(sr * ratio)
        tempo_corr = 1.0 / ratio
        
        output_file = input_file.replace(".wav", f"_p{semitones}.wav")
        filter_str = f"asetrate={new_rate},atempo={tempo_corr},aresample={sr}"
        
        subprocess.run(['ffmpeg', '-i', input_file, '-af', filter_str, output_file, '-y', '-loglevel', 'error'], check=True)
        return output_file
    except:
        return input_file

def synthesize_text_cloud(text, voice_name, output_path, project_id=None):
    """
    Synthesize text using Google Cloud TTS.
    Returns: path to file on success, None on failure.
    """
    token = get_access_token()
    if not token: 
        logging.error("❌ No GCP Token.")
        return None
        
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    lang_code = "-".join(voice_name.split("-")[:2])
    
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {"audioEncoding": "LINEAR16"}
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 200:
            content = base64.b64decode(res.json()['audioContent'])
            with open(output_path, 'wb') as f:
                f.write(content)
            return output_path
        else:
            logging.error(f"❌ Cloud TTS Error: {res.text}")
            return None
    except Exception as e:
        logging.error(f"❌ Cloud TTS Exception: {e}")
        return None

def generate_audio_asset(text, output_path, voice_name="en-US-Journey-D", pitch=0, mode="cloud", project_id=None, comfy_wrapper=None):
    """
    Unified entry point for generating a single audio asset (speech).
    Handles dispatch (Cloud vs Comfy) and Pitch Shifting.
    """
    # Sanitize Text (User Request: Kokoro reads asterisks literal)
    # Also strip hashtags used for markdown headers
    text = text.replace("*", "").replace("#", "")
    
    temp_raw = output_path.replace(".wav", "_raw.wav")
    
    final_path = None
    
    if mode == "cloud":
        # 1. Synthesize
        if synthesize_text_cloud(text, voice_name, temp_raw, project_id):
            # 2. Pitch Shift
            # pitch_shift_file writes to _pX.wav, we want it at output_path?
            # actually pitch_shift_file returns the new path. 
            # If pitch is 0, it returns input. 
            # We want assurance.
            shifted = pitch_shift_file(temp_raw, pitch)
            if shifted != output_path:
                shutil.move(shifted, output_path)
            final_path = output_path
            
            # Cleanup raw if it differs/exists
            if os.path.exists(temp_raw) and temp_raw != final_path:
                os.remove(temp_raw)
    
    elif mode == "comfy":
        pass
    
    elif mode == "kokoro":
        # Local Kokoro
        try:
            from kokoro_bridge import get_kokoro_bridge
            KOKORO_MODEL = "/Volumes/XMVPX/mw/kokoro-root/kokoro-v0_19.onnx"
            bridge = get_kokoro_bridge(KOKORO_MODEL)
            
            temp_raw = output_path.replace(".wav", "_raw.wav")
            
            if bridge.generate(text, temp_raw, voice_name=voice_name):
                # Apply Pitch Shift if needed
                shifted = pitch_shift_file(temp_raw, pitch)
                
                # Move if needed
                if shifted != output_path:
                    shutil.move(shifted, output_path)
                    
                # Cleanup raw
                if os.path.exists(temp_raw) and temp_raw != output_path:
                    os.remove(temp_raw)
                    
                return output_path
            
            return None
            
        except ImportError:
            logging.error("❌ Kokoro Bridge not found.")
            return None
        except Exception as e:
            logging.error(f"❌ Kokoro Error: {e}")
            return None
        
    return final_path

# --- BATCH GENERATORS ---

def generate_cloud_dialogue(script: DialogueScript, output_dir, project_id):
    logging.info(f"☁️ [Cloud] Generating Dialogue ({len(script.lines)} lines)...")
    results = []
    
    # Voice Matrix (Simple mapping for now)
    VOICE_MAP = {
        "William": ("en-US-Journey-D", 1),
        "Maggie": ("en-US-Journey-F", -2),
        "Francis": ("en-US-Journey-D", -2),
        "Anne Tailored": ("en-US-Journey-F", 1)
    }

    for i, line in enumerate(script.lines):
        logging.info(f"   Actor {line.character}: '{line.text[:30]}...'")
        
        config = VOICE_MAP.get(line.character, ("en-US-Journey-D", 0))
        voice_name, pitch = config
        
        out_path = os.path.join(output_dir, f"dial_{i}_{line.character}.wav")
        
        if generate_audio_asset(line.text, out_path, voice_name, pitch, mode="cloud", project_id=project_id):
             results.append({"path": out_path, "offset": line.start_offset})
            
    return results

# --- BACKEND 3: RVC (Legacy) ---
# Ported from voice_engine.py

class LegacyVoiceEngine:
    def __init__(self, weights_dir="/Volumes/XMVPX/mw/rvc-root/weights"):
        self.weights_dir = weights_dir
        
    def generate_base_audio(self, text, output_path, gender="MALE"):
        voice = "Alex" if gender == "MALE" else "Samantha"
        try:
            cmd = ['say', '-v', voice, '-o', output_path, '--data-format=LEF32@24000', text]
            subprocess.run(cmd, check=True)
            return True
        except:
            return False

    def assign_voice(self, actor_name):
        return "Unknown"

    def apply_rvc(self, input_path, output_path, model_name):
        logging.info(f"   [RVC] Mock Inference {model_name} -> {output_path}")
        shutil.copy(input_path, output_path)
        return True

def generate_hunyuan_batch(manifest: Manifest, output_dir):
    """
    Generates foley for every segment in the manifest that has a video file.
    """
    logging.info(f"🔊 [Hunyuan] Generating Foley for {len(manifest.files)} clips...")
    bridge = HunyuanFoleyBridge()
    results = [] # List of {path, offset}
    
    # Sort segs
    sorted_segs = sorted(manifest.segs, key=lambda s: s.id)
    
    # Calculate global offsets
    current_offset = 0.0
    
    for seg in sorted_segs:
        # Determine duration
        duration_frames = seg.end_frame - seg.start_frame
        duration_sec = duration_frames / 24.0
        
        if seg.id in manifest.files:
            vid_path = manifest.files[seg.id]
            foley_path = bridge.generate_foley(vid_path, seg.prompt, output_dir)
            
            if foley_path:
                results.append({"path": foley_path, "offset": current_offset})
                logging.info(f"   ✅ Seg {seg.id}: {foley_path} @ {current_offset:.2f}s")
            
        current_offset += duration_sec
        
    return results

def generate_rvc_dialogue(script: DialogueScript, output_dir):
    logging.info(f"💾 [Legacy] Generating Dialogue...")
    engine = LegacyVoiceEngine()
    results = []
    for i, line in enumerate(script.lines):
        base_path = os.path.join(output_dir, f"base_{i}.wav")
        final_path = os.path.join(output_dir, f"rvc_{i}.wav")
        
        engine.generate_base_audio(line.text, base_path)
        engine.apply_rvc(base_path, final_path, line.character)
        
        results.append({"path": final_path, "offset": line.start_offset})
    return results




def assign_kokoro_voice_deterministic(actor_name, available_voices):
    """
    Deterministically assigns a (voice_name, pitch_shift) tuple to an actor.
    Expansion Rule: neutral, +1, -2 for every voice.
    """
    if not available_voices:
        return "af_bella", 0 # Fallback
        
    # 1. Expand Palette
    palette = []
    for v in available_voices:
        palette.append((v, 0))
        palette.append((v, 1))
        palette.append((v, -2))
        
    # 2. Hash Actor Name
    # Simple hash based on sum of ordinals
    seed = sum(ord(c) for c in actor_name)
    idx = seed % len(palette)
    
    return palette[idx]

def generate_kokoro_dialogue(script: DialogueScript, output_dir):
    logging.info(f"🦜 [Kokoro] Generating Dialogue ({len(script.lines)} lines)...")
    results = []
    
    # Initialize Bridge & Fetch Voices
    try:
        from kokoro_bridge import get_kokoro_bridge
        KOKORO_MODEL = "/Volumes/XMVPX/mw/kokoro-root/kokoro-v0_19.onnx"
        bridge = get_kokoro_bridge(KOKORO_MODEL)
        
        # Ensure bridge loaded to get voices
        bridge.load()
        available_voices = bridge.get_voice_list()
        
        # Filter for quality if needed, currently take all
        # If none found (e.g. voices.json missing), fallback defaults
        if not available_voices:
            logging.warning("⚠️ No voices found in voices.json, using defaults.")
            available_voices = ["af_bella", "af_sarah", "am_michael", "am_adam"]
            
        logging.info(f"   🎤 Available Voices: {len(available_voices)} (Expanded to {len(available_voices)*3})")
        
    except Exception as e:
        logging.error(f"❌ Failed to init Kokoro for batch: {e}")
        return []

    # Cache assignments for this run
    actor_map = {}

    for i, line in enumerate(script.lines):
        logging.info(f"   Actor {line.character}: '{line.text[:30]}...'")
        
        # 1. Assign Voice/Pitch
        if line.character not in actor_map:
            voice_name, pitch = assign_kokoro_voice_deterministic(line.character, available_voices)
            actor_map[line.character] = (voice_name, pitch)
            logging.info(f"     -> Assigned: {voice_name} @ {pitch} semitones")
        else:
            voice_name, pitch = actor_map[line.character]

        out_path = os.path.join(output_dir, f"dial_{i}_{line.character}.wav")
        # generate_audio_asset handles the pitch shift if we pass pitch
        # but we need to pass mode="kokoro"
        
        # Update generate_audio_asset to support pitch for kokoro mode?
        # Current implementation of generate_audio_asset calls bridge.generate then returns.
        # It does NOT apply pitch shift for kokoro mode in the previous edit.
        # We need to ensure we call pitch_shift_file for Kokoro output too.
        
        # Let's call generate_audio_asset with pitch, and modify it to handle shifting for all modes.
        if generate_audio_asset(line.text, out_path, voice_name, pitch=pitch, mode="kokoro"):
             results.append({"path": out_path, "offset": line.start_offset})
             
    return results


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Foley Talk: Unified Audio Engine")
    parser.add_argument("--input", help="Input silent video path (Required for Mixing)")
    parser.add_argument("--xb", help="Input XMVP manifest (Source of Dialogue/Segments)")
    parser.add_argument("--out", default="final_mix.mp4", help="Output video path")
    parser.add_argument("--mode", choices=["cloud", "comfy", "rvc", "kokoro", "draft-mix"], default="cloud", help="Audio Backend")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution")
    
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    
    # --- DRAFT MIX MODE ---
    if args.mode == "draft-mix":
        if not args.xb or not args.input:
            logging.error("❌ 'draft-mix' requires both --xb (Manifest) and --input (Video).")
            sys.exit(1)
            
        logging.info("🍔 Enter Draft Mix Mode...")
        manifest = load_manifest(args.xb)
        
        # 1. Foley (Hunyuan)
        foley_assets = generate_hunyuan_batch(manifest, out_dir)
        
        # 2. Dialogue (Kokoro)
        dialogue_assets = []
        if manifest.dialogue and manifest.dialogue.lines:
             dialogue_assets = generate_kokoro_dialogue(manifest.dialogue, out_dir)
             
        # 3. Mux
        total_duration = get_audio_duration(args.input)
        if total_duration <= 0:
            logging.warning("⚠️ Could not determine video duration. Assuming 600s or Assets duration.")
            total_duration = 600.0
        
        # Compose Tracks
        foley_track = os.path.join(out_dir, "track_foley.wav")
        compose_track(foley_assets, total_duration, foley_track)
        
        dial_track = os.path.join(out_dir, "track_dialogue.wav")
        compose_track(dialogue_assets, total_duration, dial_track)
        
        # Final Mux
        logging.info("🎛️ Final Muxing...")
        cmd = [
            "ffmpeg", "-y",
            "-i", args.input,
            "-i", foley_track,
            "-i", dial_track,
            "-filter_complex", "[1:a]volume=0.6[a1];[2:a]volume=1.2[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            args.out
        ]
        try:
            subprocess.run(cmd, check=True)
            logging.info(f"✨ Done! Saved to {args.out}")
        except Exception as e:
            logging.error(f"❌ Mux Failed: {e}")
            
        return

    # --- LEGACY MODES ---
    if not args.input: 
        logging.error("❌ Legacy modes require --input.")
        sys.exit(1)
        
    # 1. Dialogue Generation
    dialogue_wavs = []
    if args.xb:
        try:
            manifest = load_manifest(args.xb)
            if manifest.dialogue:
                if args.mode == "cloud":
                    project_id = get_project_id()
                    dialogue_wavs = generate_cloud_dialogue(manifest.dialogue, out_dir, project_id)
                elif args.mode == "comfy":
                    wrapper = ComfyWrapper(dry_run=args.dry_run)
                    dialogue_wavs = generate_comfy_dialogue(wrapper, manifest.dialogue, out_dir)
                elif args.mode == "rvc":
                    dialogue_wavs = generate_rvc_dialogue(manifest.dialogue, out_dir)
                elif args.mode == "kokoro":
                    dialogue_wavs = generate_kokoro_dialogue(manifest.dialogue, out_dir)
        except Exception as e:
            logging.error(f"Failed to load dialogue: {e}")

    # 2. Foley (Enhancement)
    foley_wav = None
    if args.mode == "comfy":
         wrapper = ComfyWrapper(dry_run=args.dry_run)
         
    # 3. Mix
    mix_audio(args.input, foley_wav, dialogue_wavs, args.out)
    logging.info(f"✨ Done! Mode: {args.mode.upper()}")

if __name__ == "__main__":
    main()
