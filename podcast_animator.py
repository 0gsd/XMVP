
#!/usr/bin/env python3
# podcast_animator.py
# Version: 1.0
# Description: Animates podcast triplets (JSON, TXT, MP3) into MP4s using Gemini 2.5 Flash imagery.
# Usage: python3 podcast_animator.py

import os
import sys
import json
import argparse
import glob
import time
import re
import subprocess
import random
import yaml
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

import requests
import base64
import wave
import math

# --- CONFIG ---
TRIPLETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../z_podcast_triplets"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "z_test-outputs"))

# Import MVP Shared for Config Logic
try:
    import mvp_shared
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    import mvp_shared

MODEL_ID = "gemini-2.5-flash-image" # Verified L-Tier

# --- HELPERS (API & TTS) ---

KEYS = mvp_shared.load_api_keys() # Automatically finds env_vars.yaml
if not KEYS:
    print(f"[-] CRITICAL: No keys found. Check env_vars.yaml in tools/fmv/")
    # Don't exit immediately if only TTS is needed, but we need keys for Images anyway.
    sys.exit(1)

def get_client():
    """Returns a client with a random key rotation."""
    key = random.choice(KEYS)
    return genai.Client(api_key=key), key

def get_access_token(project_id=None):
    """Retrieves a Google Cloud Access Token via gcloud CLI."""
    try:
        cmd = ["gcloud", "auth", "print-access-token"]
        if project_id:
            cmd.append(f"--project={project_id}")
            
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"[-] Failed to get GCP access token: {e}")
        return None

def get_project_id():
    """Retrieves the current Google Cloud Project ID."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"[-] Failed to get GCP Project ID: {e}")
        return None

def detect_gender_rest(speakers, api_key):
    """Uses Gemini Flash (via REST) to detect predominant gender."""
    if not api_key: return "MALE"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    prompt = f"Given these names: {', '.join(speakers)}. Return ONLY the word 'MALE' if they are mostly male, or 'FEMALE' if they are mostly female. No other text."
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            txt = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
            if "FEMALE" in txt: return "FEMALE"
            return "MALE"
    except:
        pass
    return "MALE"

def synthesize_text_rest(text, voice_name, token, project_id):
    """Synthesizes speech using Google Cloud TTS REST API."""
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    lang_code = "-".join(voice_name.split("-")[:2])
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.0}
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "X-Goog-User-Project": project_id
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return base64.b64decode(response.json()['audioContent'])
        else:
            print(f"[-] TTS Error: {response.text}")
            return None
    except Exception as e:
        print(f"[-] TTS Request Failed: {e}")
        return None

def pitch_shift_file(input_file, semitones):
    """Shifts pitch using FFmpeg."""
    if semitones == 0: return input_file
    try:
        # Assumes 24kHz usually for Journey voices, but let's be robust
        # We need sample rate. 
        # Actually, simpler: just asetrate.
        # But we need to know the input rate to set the new rate.
        # Let's assume 24000 for Journey, or probe it.
        # probing...
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
        sr_str = subprocess.check_output(probe_cmd).strip()
        sample_rate = int(sr_str)
        
        ratio = math.pow(2, semitones / 12.0)
        new_rate = int(sample_rate * ratio)
        tempo_corr = 1.0 / ratio
        
        output_file = input_file.replace(".wav", f"_p{semitones}.wav")
        filter_str = f"asetrate={new_rate},atempo={tempo_corr}"
        
        subprocess.run(['ffmpeg', '-i', input_file, '-af', filter_str, output_file, '-y', '-loglevel', 'error'], check=True)
        return output_file
    except Exception as e:
        print(f"[-] Pitch shift error: {e}")
        return input_file

def get_audio_duration(file_path):
    """Get precise duration of audio using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-i', file_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[-] Error parsing audio duration ({file_path}): {e}")
        return 0.0

def generate_image(prompt, output_path):
    """Generates an image using Gemini 2.5 Flash via generate_content."""
    client, key_used = get_client()
    print(f"   [>] Getting Image (Key: ...{key_used[-4:]})")
    
    try:
        final_prompt = f"Generate an image of {prompt} --aspect_ratio 1:1"
        response = client.models.generate_content(model=MODEL_ID, contents=final_prompt)
        
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return True
        print("   [-] No image data returned.")
        return False
    except Exception as e:
        print(f"   [-] Generation Failed: {e}")
        return False

# --- PARSING AND PROCESSING ---

class Segment:
    def __init__(self, speaker, text, start_time=None, end_time=None):
        self.speaker = speaker
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.duration = 0.0
        self.image_path = None
        self.audio_path = None # For generated audio
        self.block_type = "banter"

def parse_transcript_basic(txt_path):
    """Simple block parser for when we don't need timestamp sync (Audio Gen mode)."""
    with open(txt_path, 'r') as f: lines = f.readlines()
    
    segments = []
    current_speaker = "Unknown"
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("---") or line.startswith("TITLE:"): continue
        if line.startswith("EPISODE:") or line.startswith("HOSTS:"): continue
        if line.startswith("GENERATED BY:") or line.startswith("PART"): continue
        
        # Remove timestamps if present [MM:SS]
        line = re.sub(r'\[\d{2}:\d{2}\]', '', line).strip()
        
        match = re.match(r'^([A-Z\s\.]+):', line) # "JOE PANTOLIANO:"
        if match:
            if current_text:
                segments.append(Segment(current_speaker, " ".join(current_text)))
                current_text = []
            
            current_speaker = match.group(1).title()
            content = line[len(match.group(0)):].strip()
            if content: current_text.append(content)
        else:
            current_text.append(line)
            
    if current_text:
         segments.append(Segment(current_speaker, " ".join(current_text)))
         
    return segments

def classify_segment(seg, meta):
    """Classifies segment as banter or movie based on metadata match."""
    is_movie = False
    raw_meta_movies = [
        meta.get('transcript_raw', {}).get('movie_a', ''),
        meta.get('transcript_raw', {}).get('movie_b', '')
    ]
    swatch = seg.text[:50]
    for m_txt in raw_meta_movies:
        if m_txt and swatch in m_txt:
            is_movie = True
            break
    seg.block_type = "movie" if is_movie else "banter"
    return seg

def get_image_prompt(seg, host_map):
    host_info = host_map.get(seg.speaker, {})
    
    if seg.block_type == "movie":
        prompt = (
            f"Cinematic still from a movie. "
            f"Scene description based on this narration: '{seg.text[:300]}'. "
            f"Atmospheric, detailed, 8k, photorealistic. "
            f"Style: {host_info.get('vibe', 'Cinematic')}."
        )
    else:
        vibe_desc = host_info.get('vibe', '')
        if not vibe_desc: vibe_desc = host_info.get('voice_persona', 'distinguished character actor')
        vibe_short = vibe_desc.split('.')[0]
        if len(vibe_short) > 100: vibe_short = vibe_short[:100]
        
        prompt = (
            f"A cinematic podcast studio shot of a general all-purpose {vibe_short}-type "
            f"impersonator from Hollywood vaguely resembling {seg.speaker}. "
            f"They are speaking with expression. "
            f"Lighting: Professional studio lighting. "
            f"Context: They are saying '{seg.text[:100]}...'"
        )
    return prompt

def process_triplet(base, txt_path, json_path, mp3_path, output_mp4):
    """Process existing audio triplet (Sync by proportional time)."""
    print(f"[*] Processing TRIPLET {base}...")
    
    # 1. Audio Duration
    dur = get_audio_duration(mp3_path)
    print(f"    Audio Duration: {dur:.2f}s")
    
    # 2. Parse (Reuse existing parser logic, imported or redefined)
    # Note: Refactoring parse_transcript to be locally available or method
    # For brevity in this edit, assuming the `parse_transcript` from old code 
    # (Proportional Time Allocation) is available or needs strictly re-included.
    # I will inline a simplified version for context if strict replacement used.
    # ... Wait, I'm replacing lines 21-435. I handled parsing in the previous file.
    # Since I'm multi-replacing, I need to make sure I don't lose the Triplet parsing logic.
    # I will rewrite `parse_transcript_triplet` below.
    
    segments, host_map = parse_transcript_triplet(txt_path, json_path, dur)
    print(f"    Segments: {len(segments)}")
    
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "input.txt")
    
    valid_segments = []
    
    try:
        for i, seg in enumerate(segments):
            img_filename = f"frame_{i:04d}.png"
            img_path = os.path.join(temp_dir, img_filename)
            
            prompt = get_image_prompt(seg, host_map)
            
            if not os.path.exists(img_path):
                if not generate_image(prompt, img_path):
                    Image.new('RGB', (1024, 1024), color='black').save(img_path)
            
            seg.image_path = img_path
            valid_segments.append(seg)
            
        # Stitch
        with open(input_list_path, 'w') as f:
            for seg in valid_segments:
                f.write(f"file '{seg.image_path}'\n")
                f.write(f"duration {seg.duration:.4f}\n")
                
        print("    Stitching Video...")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', input_list_path,
            '-i', mp3_path,
            '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k', '-shortest', output_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[+] Created {output_mp4}")
        
    finally:
        pass # shutil.rmtree(temp_dir) if desired

def process_pair(base, txt_path, json_path, output_mp4, project_id_override=None):
    """Process Pair (Text+JSON) by GENERATING Audio."""
    print(f"[*] Processing PAIR {base} (Generating Audio)...")
    
    # Auth
    if project_id_override:
        project_id = project_id_override
    else:
        project_id = get_project_id()
        
    token = get_access_token(project_id)
    
    if not token or not project_id:
        print("[-] Cannot generate audio: Missing GCP Auth.")
        return
    
    # Metadata & Config
    with open(json_path, 'r') as f: meta = json.load(f)
    
    # Hosts & Gender
    host_map = {}
    if 'meta' in meta and 'hosts' in meta['meta']:
        hosts = meta['meta']['hosts']
        if 'A' in hosts: host_map[hosts['A']['name']] = hosts['A']
        if 'B' in hosts: host_map[hosts['B']['name']] = hosts['B']
        
    speaker_names = list(host_map.keys())
    _, key = get_client() # just to get a key
    gender = detect_gender_rest(speaker_names, key)
    print(f"    Detected Gender Group: {gender}")
    
    base_voice = "en-US-Journey-D" if gender == "MALE" else "en-US-Journey-F"
    
    # Assign voices
    voice_config = {}
    sorted_speakers = sorted(speaker_names)
    for i, spk in enumerate(sorted_speakers):
        # Shift secondary speakers
        voice_config[spk] = {"name": base_voice, "pitch": 0.0 if i == 0 else -2.0}
        
    # Parse Basic
    segments = parse_transcript_basic(txt_path)
    print(f"    Segments to Synthesize: {len(segments)}")
    
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_gen_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_segs = []
    
    try:
        for i, seg in enumerate(segments):
            # 1. Classify
            seg = classify_segment(seg, meta)
            
            # 2. Audio Gen
            v_conf = voice_config.get(seg.speaker, {"name": base_voice, "pitch": 0.0})
            
            # Chunking handled? For now assume segments are sentences. 
            # If long, split? Let's assume MVP fit.
            # Clean text
            clean_txt = seg.text.replace("*", "").replace("_", "")
            
            print(f"    [{i+1}/{len(segments)}] {seg.speaker}: {clean_txt[:30]}...")
            
            # Refresh Token (Prevent 401 on long runs)
            current_token = get_access_token(project_id)
            if not current_token:
                print("       [!] Failed to refresh token, skipping.")
                continue

            audio_data = synthesize_text_rest(clean_txt[:4800], v_conf['name'], current_token, project_id)
            if not audio_data:
                print("       [!] Audio failed, skipping.")
                continue
                
            raw_wav = os.path.join(temp_dir, f"seg_{i:03d}_raw.wav")
            with open(raw_wav, "wb") as f: f.write(audio_data)
            
            # FX
            final_wav = raw_wav
            if v_conf['pitch'] != 0.0:
                final_wav = pitch_shift_file(raw_wav, v_conf['pitch'])
                
            seg.audio_path = final_wav
            seg.duration = get_audio_duration(final_wav)
            
            # 3. Image Gen
            img_path = os.path.join(temp_dir, f"seg_{i:03d}.png")
            prompt = get_image_prompt(seg, host_map)
            if not generate_image(prompt, img_path):
                Image.new('RGB', (1024, 1024), color='black').save(img_path)
            seg.image_path = img_path
            
            processed_segs.append(seg)
            
        if not processed_segs:
            print("[-] No content generated. Aborting.")
            return

        # 4. Stitch
        # Create audio concat list AND video concat list
        audio_list = os.path.join(temp_dir, "audio_list.txt")
        video_list = os.path.join(temp_dir, "video_list.txt")
        
        with open(audio_list, 'w') as fa, open(video_list, 'w') as fv:
            for seg in processed_segs:
                fa.write(f"file '{seg.audio_path}'\n")
                fv.write(f"file '{seg.image_path}'\n")
                fv.write(f"duration {seg.duration:.4f}\n")
        
        full_audio = os.path.join(temp_dir, "full_audio.wav")
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', audio_list, '-c', 'copy', full_audio, '-y', '-loglevel', 'error'], check=True)
        
        print("    Stitching Final Pair Video...")
        # Note: We need to ensure we use the generated audio
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', video_list,
            '-i', full_audio,
            '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k', '-shortest', output_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[+] Created {output_mp4}")
        
    finally:
        pass

def get_project_id():
    """Retrieves the current Google Cloud Project ID."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"[-] Failed to get GCP Project ID: {e}")
        return None

def detect_gender_rest(speakers, api_key):
    """Uses Gemini Flash (via REST) to detect predominant gender."""
    if not api_key: return "MALE"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    prompt = f"Here are the names of people in a podcast: {', '.join(speakers)}. Are they predominantly MALE or FEMALE? Reply with a single word."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
            return "FEMALE" if "FEMALE" in text else "MALE"
    except:
        pass
    return "MALE"

def synthesize_text_rest(text, voice_name, token, project_id):
    """Synthesizes speech using Google Cloud TTS REST API."""
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    lang_code = "-".join(voice_name.split("-")[:2])
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.0}
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "X-Goog-User-Project": project_id
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return base64.b64decode(response.json()['audioContent'])
        else:
            print(f"[-] TTS Error: {response.text}")
            return None
    except Exception as e:
        print(f"[-] TTS Request Failed: {e}")
        return None

def pitch_shift_file(input_file, semitones):
    """Shifts pitch using FFmpeg."""
    if semitones == 0: return input_file
    try:
        # Assumes 24kHz usually for Journey voices, but let's be robust
        # We need sample rate. 
        # Actually, simpler: just asetrate.
        # But we need to know the input rate to set the new rate.
        # Let's assume 24000 for Journey, or probe it.
        # probing...
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
        sr_str = subprocess.check_output(probe_cmd).strip()
        sample_rate = int(sr_str)
        
        ratio = math.pow(2, semitones / 12.0)
        new_rate = int(sample_rate * ratio)
        tempo_corr = 1.0 / ratio
        
        output_file = input_file.replace(".wav", f"_p{semitones}.wav")
        filter_str = f"asetrate={new_rate},atempo={tempo_corr}"
        
        subprocess.run(['ffmpeg', '-i', input_file, '-af', filter_str, output_file, '-y', '-loglevel', 'error'], check=True)
        return output_file
    except Exception as e:
        print(f"[-] Pitch shift error: {e}")
        return input_file

def get_audio_duration(file_path):
    """Get precise duration of audio using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-i', file_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[-] Error parsing audio duration ({file_path}): {e}")
        return 0.0

def generate_image(prompt, output_path):
    """Generates an image using Gemini 2.5 Flash via generate_content."""
    client, key_used = get_client()
    print(f"   [>] Getting Image (Key: ...{key_used[-4:]})")
    
    try:
        final_prompt = f"Generate an image of {prompt} --aspect_ratio 1:1"
        response = client.models.generate_content(model=MODEL_ID, contents=final_prompt)
        
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return True
        print("   [-] No image data returned.")
        return False
    except Exception as e:
        print(f"   [-] Generation Failed: {e}")
        return False

# --- PARSING AND PROCESSING ---

class Segment:
    def __init__(self, speaker, text, start_time=None, end_time=None):
        self.speaker = speaker
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.duration = 0.0
        self.image_path = None
        self.audio_path = None # For generated audio
        self.block_type = "banter"

def parse_transcript_basic(txt_path):
    """Simple block parser for when we don't need timestamp sync (Audio Gen mode)."""
    with open(txt_path, 'r') as f: lines = f.readlines()
    
    segments = []
    current_speaker = "Unknown"
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("---") or line.startswith("TITLE:"): continue
        if line.startswith("EPISODE:") or line.startswith("HOSTS:"): continue
        if line.startswith("GENERATED BY:") or line.startswith("PART"): continue
        
        # Remove timestamps if present [MM:SS]
        line = re.sub(r'\[\d{2}:\d{2}\]', '', line).strip()
        
        match = re.match(r'^([A-Z\s\.]+):', line) # "JOE PANTOLIANO:"
        if match:
            if current_text:
                segments.append(Segment(current_speaker, " ".join(current_text)))
                current_text = []
            
            current_speaker = match.group(1).title()
            content = line[len(match.group(0)):].strip()
            if content: current_text.append(content)
        else:
            current_text.append(line)
            
    if current_text:
         segments.append(Segment(current_speaker, " ".join(current_text)))
         
    return segments

def classify_segment(seg, meta):
    """Classifies segment as banter or movie based on metadata match."""
    is_movie = False
    raw_meta_movies = [
        meta.get('transcript_raw', {}).get('movie_a', ''),
        meta.get('transcript_raw', {}).get('movie_b', '')
    ]
    swatch = seg.text[:50]
    for m_txt in raw_meta_movies:
        if m_txt and swatch in m_txt:
            is_movie = True
            break
    seg.block_type = "movie" if is_movie else "banter"
    return seg

def get_image_prompt(seg, host_map):
    host_info = host_map.get(seg.speaker, {})
    
    if seg.block_type == "movie":
        prompt = (
            f"Cinematic still from a movie. "
            f"Scene description based on this narration: '{seg.text[:300]}'. "
            f"Atmospheric, detailed, 8k, photorealistic. "
            f"Style: {host_info.get('vibe', 'Cinematic')}."
        )
    else:
        vibe_desc = host_info.get('vibe', '')
        if not vibe_desc: vibe_desc = host_info.get('voice_persona', 'distinguished character actor')
        vibe_short = vibe_desc.split('.')[0]
        if len(vibe_short) > 100: vibe_short = vibe_short[:100]
        
        prompt = (
            f"A cinematic podcast studio shot of a general all-purpose {vibe_short}-type "
            f"impersonator from Hollywood vaguely resembling {seg.speaker}. "
            f"They are speaking with expression. "
            f"Lighting: Professional studio lighting. "
            f"Context: They are saying '{seg.text[:100]}...'"
        )
    return prompt

def parse_time(ts):
    parts = ts.split(':')
    return float(parts[0])*60 + float(parts[1])

def parse_transcript_triplet(txt_path, json_path, total_dur):
    # (Simplified re-implementation of the proportional logic from v1.0)
    with open(json_path, 'r') as f: meta = json.load(f)
    with open(txt_path, 'r') as f: lines = f.readlines()
    
    host_map = {}
    if 'meta' in meta and 'hosts' in meta['meta']:
        hosts = meta['meta']['hosts']
        if 'A' in hosts: host_map[hosts['A']['name']] = hosts['A']
        if 'B' in hosts: host_map[hosts['B']['name']] = hosts['B']

    raw_segs = []
    curr_spk = "Unknown"
    curr_txt = []
    curr_anchor = 0.0
    
    time_pat = re.compile(r'\[(\d{2}:\d{2})\]')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("---") or line.startswith("TITLE:"): continue
        
        ts_match = time_pat.search(line)
        ts_val = None
        if ts_match:
            ts_val = parse_time(ts_match.group(1))
            line = line.replace(ts_match.group(0), "").strip()
            
        spk_match = re.match(r'^([A-Z\s]+):', line)
        if spk_match:
            if curr_txt:
                raw_segs.append({"speaker": curr_spk, "text": "\n".join(curr_txt), "anchor": curr_anchor})
                curr_txt = []
            curr_spk = spk_match.group(1).title()
            content = line[len(spk_match.group(0)):].strip()
            if content: curr_txt.append(content)
            if ts_val is not None: curr_anchor = ts_val
            else: curr_anchor = None
        else:
            curr_txt.append(line)
            
    if curr_txt:
         raw_segs.append({"speaker": curr_spk, "text": "\n".join(curr_txt), "anchor": curr_anchor})
         
    # Proportional Alloc
    processed = []
    if raw_segs and raw_segs[0]['anchor'] is None: raw_segs[0]['anchor'] = 0.0
    
    i = 0
    while i < len(raw_segs):
        start = raw_segs[i]['anchor'] or 0.0
        # Find next
        next_i, end_t = len(raw_segs), total_dur
        for j in range(i+1, len(raw_segs)):
            if raw_segs[j]['anchor'] is not None:
                next_i, end_t = j, raw_segs[j]['anchor']
                break
        
        block = raw_segs[i:next_i]
        total_chars = sum(len(s['text']) for s in block)
        window = end_t - start
        
        curr_clock = start
        for b in block:
            dur = (len(b['text']) / total_chars * window) if total_chars else window
            seg = Segment(b['speaker'], b['text'], curr_clock, curr_clock+dur)
            seg.duration = dur
            seg = classify_segment(seg, meta)
            processed.append(seg)
            curr_clock += dur
        i = next_i
        
    return processed, host_map

def process_triplet(base, txt_path, json_path, mp3_path, output_mp4):
    """Process existing audio triplet (Sync by proportional time)."""
    print(f"[*] Processing TRIPLET {base}...")
    
    # 1. Audio Duration
    dur = get_audio_duration(mp3_path)
    print(f"    Audio Duration: {dur:.2f}s")
    
    # 2. Parse (Reuse existing parser logic, imported or redefined)
    # Note: Refactoring parse_transcript to be locally available or method
    # For brevity in this edit, assuming the `parse_transcript` from old code 
    # (Proportional Time Allocation) is available or needs strictly re-included.
    # I will inline a simplified version for context if strict replacement used.
    # ... Wait, I'm replacing lines 21-435. I handled parsing in the previous file.
    # Since I'm multi-replacing, I need to make sure I don't lose the Triplet parsing logic.
    # I will rewrite `parse_transcript_triplet` below.
    
    segments, host_map = parse_transcript_triplet(txt_path, json_path, dur)
    print(f"    Segments: {len(segments)}")
    
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "input.txt")
    
    valid_segments = []
    
    try:
        for i, seg in enumerate(segments):
            img_filename = f"frame_{i:04d}.png"
            img_path = os.path.join(temp_dir, img_filename)
            
            prompt = get_image_prompt(seg, host_map)
            
            if not os.path.exists(img_path):
                if not generate_image(prompt, img_path):
                    Image.new('RGB', (1024, 1024), color='black').save(img_path)
            
            seg.image_path = img_path
            valid_segments.append(seg)
            
        # Stitch
        with open(input_list_path, 'w') as f:
            for seg in valid_segments:
                f.write(f"file '{seg.image_path}'\n")
                f.write(f"duration {seg.duration:.4f}\n")
                
        print("    Stitching Video...")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', input_list_path,
            '-i', mp3_path,
            '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k', '-shortest', output_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[+] Created {output_mp4}")
        
    finally:
        pass # shutil.rmtree(temp_dir) if desired

def main():
    parser = argparse.ArgumentParser(description="Podcast Animator v1.1")
    parser.add_argument("--project", help="Google Cloud Project ID for TTS API", default=None)
    args = parser.parse_args()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("🎙️  Podcast Animator v1.1 (AudioGen Enabled)")
    if args.project:
        print(f"[*] Using Project Override: {args.project}")
    
    json_files = glob.glob(os.path.join(TRIPLETS_DIR, "*.json"))
    
    for j_path in json_files:
        base = os.path.splitext(os.path.basename(j_path))[0]
        txt_path = os.path.join(TRIPLETS_DIR, base + ".txt")
        mp3_path = os.path.join(TRIPLETS_DIR, base + ".mp3")
        
        if not os.path.exists(txt_path): continue
        
        final_mp4 = os.path.join(OUTPUT_DIR, base + ".mp4")
        if os.path.exists(final_mp4):
            print(f"[-] Skipping {base}, output exists.")
            continue
            
        if os.path.exists(mp3_path):
            process_triplet(base, txt_path, j_path, mp3_path, final_mp4)
        else:
            process_pair(base, txt_path, j_path, final_mp4, project_id_override=args.project)

if __name__ == "__main__":
    main()
