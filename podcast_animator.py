
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

import itertools

KEYS = mvp_shared.load_api_keys() # Automatically finds env_vars.yaml
if not KEYS:
    print(f"[-] CRITICAL: No keys found. Check env_vars.yaml in tools/fmv/")
    # Don't exit immediately if only TTS is needed, but we need keys for Images anyway.
    sys.exit(1)

random.shuffle(KEYS) # Shuffle once at start to avoid sync patterns with other scripts
KEY_CYCLE = itertools.cycle(KEYS)

def get_client():
    """Returns a client with a random key rotation."""
    key = next(KEY_CYCLE)
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

EISNER_STYLE = (
    "You are an Eisner-award winning comic book penciller, inker, and colorer (a triple threat). "
    "You draw in the style of the greats, like Jack Kirby, Rob Liefeld, Todd MacFarlane, Bill Sienkiwicz, "
    "and Sergio Aragones, but with a unique style all your own and extremely realistic "
    "(almost hyperrealistic) caricaturesque portraits of the impersonators hired to play each of our illustrious guests."
)

def split_text_into_chunks(text, max_chars=4000):
    """Splits text into chunks respecting sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current_chunk = []
    current_len = 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        if current_len + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(sentence)
        else:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def create_silence(duration_sec, output_path):
    """Creates a silent WAV file using ffmpeg."""
    subprocess.run([
        'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=r=24000:cl=mono:d={duration_sec}',
        '-c:a', 'pcm_s16le', str(output_path), '-y', '-loglevel', 'error'
    ], check=True)
    return output_path

def synthesize_text_rest(text, voice_name, token, project_id):
    """Synthesizes speech using Google Cloud TTS REST API."""
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    lang_code = "-".join(voice_name.split("-")[:2])
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "LINEAR16", 
            "speakingRate": 0.90 # SLOW DOWN TO 90%
        }
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
    
    # State tracking to skip metadata headers
    # We assume the actual script starts after we see the first timecode [00:00] OR 
    # if we encounter a clear SPEAKER: line after a separator.
    started_dialogue = False
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Skip metadata headers
        if line.startswith("---") or line.startswith("TITLE:"): continue
        if line.startswith("EPISODE:") or line.startswith("HOSTS:"): continue
        if line.startswith("GENERATED BY:") or line.startswith("PART"): continue
        if line.startswith("{") or line.startswith("}"): continue # Skip leaked JSON
        if '"voice_persona":' in line: continue
        
        # Check for start of dialogue via timestamp
        if not started_dialogue:
            if re.match(r'\[\d{2}:\d{2}\]', line):
                started_dialogue = True
            elif re.match(r'^[A-Z\s\.]+:', line) and "HOSTS" not in line and "EPISODE" not in line:
                # If we see "JOE:" and haven't seen [00:00], it might be the start, 
                # but let's be careful about metadata fields that look like speakers.
                # If we are strictly asked to wait for [00:00], we should. 
                # But let's support both for robustness.
                # User request: "actual text always starts from after the [00:00]"
                # So we will look for that magic [00:00] or similar.
                pass

        # If user explicitly said "starts from after [00:00]", we can enforce it.
        # But let's check if the file HAS [00:00].
        # For now, let's treat the [XX:XX] removal as a trigger too.
        
        has_timestamp = re.search(r'\[\d{2}:\d{2}\]', line)
        if has_timestamp: 
            started_dialogue = True
            
        if not started_dialogue:
            continue

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
            if current_speaker != "Unknown": # Only append if we have an active speaker
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
    
    # Scrub text for prompt (remove parentheticals and extra whitespace)
    # This prevents speech bubbles leaking into images
    prompt_text = re.sub(r'\(.*?\)', '', seg.text).replace("*","").replace("_","").strip()
    prompt_text = re.sub(r'\s+', ' ', prompt_text)
    
    if seg.block_type == "movie":
        prompt = (
            f"Cinematic still from a movie. "
            f"Scene description based on this narration: '{prompt_text[:300]}'. "
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
            f"Context: They are saying '{prompt_text[:100]}...'"
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
    """Process Pair (Text+JSON) by GENERATING Audio using Local Voice Engine."""
    print(f"[*] Processing PAIR {base} (Generating Audio)...")
    
    # Remove Local Voice Engine
    # try:
    #     from voice_engine import VoiceEngine
    # except ImportError:
    #     sys.path.append(os.path.dirname(__file__))
    #     from voice_engine import VoiceEngine
    # ve = VoiceEngine()

    # Need Access Token for Google Cloud TTS
    access_token = get_access_token()
    if not access_token:
        print("[-] Verification Failed: No GCP Access Token. Run 'gcloud auth print-access-token' to debug.")
        return

    # Use Project Override if provided
    gcp_project = project_id_override if project_id_override else get_project_id()
    print(f"    GCP Project for TTS: {gcp_project}")

    # Metadata & Config
    with open(json_path, 'r') as f: meta = json.load(f)
    
    # Hosts & Gender
    host_map = {}
    if 'meta' in meta and 'hosts' in meta['meta']:
        hosts = meta['meta']['hosts']
        if 'A' in hosts: host_map[hosts['A']['name']] = hosts['A']
        if 'B' in hosts: host_map[hosts['B']['name']] = hosts['B']
        
    speaker_names = list(host_map.keys())
    
    # Gender detection (still useful for base voice selection)
    # We can use a random key for this lightweight check if needed, or default to mix.
    # Since we are local now, we don't strictly need a key unless we want to use Gemini for gender.
    # We'll just define a fallback if no key.
    if KEYS:
        _, key = get_client()
        gender = detect_gender_rest(speaker_names, key)
    else:
        gender = "MALE" # Default
    
    print(f"    Detected Gender Group: {gender}")
    
    # Assign Base Voice
    # Journey Voices: en-US-Journey-D (Male), en-US-Journey-F (Female)
    base_voice_name = "en-US-Journey-D" if gender == "MALE" else "en-US-Journey-F"
    print(f"    Base Voice Selection: {base_voice_name}")
    
    # Parse Basic
    segments = parse_transcript_basic(txt_path)
    print(f"    Segments to Synthesize: {len(segments)}")
    
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_gen_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_segs = []
    
    try:
        # Create Silence Assets
        silence_short = os.path.join(temp_dir, "silence_0.5s.wav")
        silence_long = os.path.join(temp_dir, "silence_1.0s.wav")
        create_silence(0.5, silence_short)
        create_silence(1.0, silence_long)

        for i, seg in enumerate(segments):
            # 1. Classify
            seg = classify_segment(seg, meta)
            
            # Scrub parenthetical stage directions for TTS
            clean_txt = seg.text.replace("*", "").replace("_", "")
            clean_txt = re.sub(r'\(.*?\)', '', clean_txt).strip()
            clean_txt = re.sub(r'\s+', ' ', clean_txt) # Collapse spaces
            
            print(f"    [{i+1}/{len(segments)}] {seg.speaker}: {clean_txt[:30]}...")
            
            # Skip empty segments (if only contained stage directions)
            if not clean_txt:
                print("       [.] Segment was only stage directions. Skipping audio gen (using silence).")
                # We still need an image though? Or just skip entirely?
                # Probably safer to just use a short silence and still gen image to keep cadence.
                # Actually, easier to just generate a tiny silence.
                clean_txt = " " # TTS might fail on empty. give it a space or handle downstream.

            
            # Determine Pitch Shift for this Speaker
            # Robust Matching: Normalize to Upper Case
            curr_spk_upper = seg.speaker.strip().upper()
            known_hosts_upper = [h.upper() for h in speaker_names]
            sorted_hosts = sorted(known_hosts_upper)
            
            # Find index
            try:
                current_spk_idx = sorted_hosts.index(curr_spk_upper)
            except ValueError:
                # Fuzzy match? Or default.
                current_spk_idx = 0
            
            # Speaker 0: +1 semitone
            
            # Speaker 0: +1 semitone
            # Speaker 1: -2 semitones
            if current_spk_idx == 0:
                pitch_shift = 1
            else:
                pitch_shift = -2
                
            print(f"       [+] Voice: {base_voice_name} | Shift: {pitch_shift} semitones")

            chunks = split_text_into_chunks(clean_txt)
            chunk_files = []
            
            for c_idx, chunk in enumerate(chunks):
                base_wav = os.path.join(temp_dir, f"seg_{i:03d}_part_{c_idx:02d}_base.wav")
                final_chunk_wav = os.path.join(temp_dir, f"seg_{i:03d}_part_{c_idx:02d}.wav")
                
                # Generate Base (Google Cloud TTS)
                audio_data = synthesize_text_rest(chunk, base_voice_name, access_token, gcp_project)
                
                if audio_data:
                    with open(base_wav, 'wb') as f: f.write(audio_data)
                    
                    # Apply Pitch Shift
                    shifted_file = pitch_shift_file(base_wav, pitch_shift)
                    
                    # Rename to final if needed or just use shifted
                    # pitch_shift_file returns new filename like _p-2.wav
                    # We want to put it in valid list.
                    chunk_files.append(shifted_file)
                else:
                    print(f"       [!] Chunk {c_idx} failed generation.")

            if not chunk_files:
                print("       [!] Entire segment failed audio.")
                continue
                
            # Stitch Chunks + Silence
            pause_file = silence_long if (i % 5 == 0 or seg.block_type == 'movie') else silence_short
            seg_parts = chunk_files + [pause_file]
            
            merged_wav = os.path.join(temp_dir, f"seg_{i:03d}_full.wav")
            list_path = os.path.join(temp_dir, f"list_{i:03d}.txt")
            
            with open(list_path, 'w') as f:
                for p in seg_parts: f.write(f"file '{p}'\n")
                
            subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', merged_wav, '-y', '-loglevel', 'error'], check=True)
            
            seg.audio_path = merged_wav
            seg.duration = get_audio_duration(merged_wav)
            
            # 3. Image Gen
            img_path = os.path.join(temp_dir, f"seg_{i:03d}.png")
            prompt = get_image_prompt(seg, host_map)
            
            # Generate Image (Gemini)
            if not generate_image(prompt, img_path):
                Image.new('RGB', (1024, 1024), color='black').save(img_path)
            seg.image_path = img_path
            
            processed_segs.append(seg)
            
            # Rate Limit Protection
            print("       [...] Cooling down (0.2s)...")
            time.sleep(0.2)
            
        if not processed_segs:
            print("[-] No content generated. Aborting.")
            return

        # 4. Stitch Final Video
        audio_list = os.path.join(temp_dir, "audio_list.txt")
        video_list = os.path.join(temp_dir, "video_list.txt")
        
        with open(audio_list, 'w') as fa, open(video_list, 'w') as fv:
            for seg in processed_segs:
                fa.write(f"file '{seg.audio_path}'\n")
                fv.write(f"file '{seg.image_path}'\n")
                # Subtract small buffer to ensure Video <= Audio for -shortest
                safe_dur = max(0, seg.duration - 0.05) 
                fv.write(f"duration {safe_dur:.4f}\n")
        
        full_audio = os.path.join(temp_dir, "full_audio.wav")
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', audio_list, '-c', 'copy', full_audio, '-y', '-loglevel', 'error'], check=True)
        
        print("    Stitching Final Pair Video...")
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
        # CRITICAL FIX: Resample back to original rate to avoid concat drift
        filter_str = f"asetrate={new_rate},atempo={tempo_corr},aresample={sample_rate}"
        
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

MODEL_ID = "imagen-4.0-fast-generate-001" # Verified Imagen Fast

def generate_image(prompt, output_path):
    """Generates an image using Imagen 4.0 Fast via generate_images."""
    max_retries = 5
    base_wait = 2 # Reduced wait for Imagen Fast
    
    for attempt in range(max_retries):
        client, key_used = get_client() # Rotates key on each attempt
        if attempt > 0:
            print(f"   [>] Retry {attempt}/{max_retries} for Image (Key: ...{key_used[-4:]})")
        else:
            print(f"   [>] Getting Image (Key: ...{key_used[-4:]})")
        
        try:
            # Imagen API Call
            response = client.models.generate_images(
                model=MODEL_ID,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_LOW_AND_ABOVE",
                    person_generation="ALLOW_ADULT"
                )
            )
            
            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                return True
                
            print("   [-] No image data returned.")
            return False 
            
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1)
                print(f"   [-] 429 RESOURCE_EXHAUSTED. Sleeping {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"   [-] Generation Failed: {e}")
                wait_time = 1 * (attempt + 1)
                time.sleep(wait_time)
                
    print("   [!] Max retries reached. Using Black Frame.")
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
    
    # Scrub text for prompt (remove parentheticals and extra whitespace)
    # This prevents speech bubbles leaking into images
    prompt_text = re.sub(r'\(.*?\)', '', seg.text).replace("*","").replace("_","").strip()
    prompt_text = re.sub(r'\s+', ' ', prompt_text)
    
    # Base prompt with Eisner injection
    base = f"{EISNER_STYLE} "
    
    if seg.block_type == "movie":
        prompt = (
            f"{base}"
            f"Cinematic panel visualization. "
            f"Scene description based on this narration: '{prompt_text[:300]}'. "
            f"Atmospheric, violent, dynamic angles, 8k, photorealistic ink and color. "
            f"Style: {host_info.get('vibe', 'Cinematic')}."
        )
    else:
        vibe_desc = host_info.get('vibe', '')
        if not vibe_desc: vibe_desc = host_info.get('voice_persona', 'distinguished character actor')
        vibe_short = vibe_desc.split('.')[0]
        if len(vibe_short) > 100: vibe_short = vibe_short[:100]
        
        prompt = (
            f"{base}"
            f"A drawing of a {vibe_short}-type impersonator sitting in a featureless room talking about their life and dreams with no word bubbles. "
            f"Subject vaguely resembles {seg.speaker}. They are speaking with expression. "
            f"Lighting: Dramatic, high contrast. "
            f"Context: They are saying '{prompt_text[:100]}...'\n"
            f"CRITICAL INSTRUCTION: Absolutely no language transcription onto your drawing; strictly artistic representations of non-word things. "
            f"Paint a picture of theoretical things that come into your mind as you listen to the audio of the podcast you are drawing. "
            f"NO SPEECH BUBBLES. NO TEXT on the image. NO SIGNATURES."
        )
        
    # User Request: Suppress Signatures
    prompt += " You are an anonymous master artist because that leaves you with more free time and you can charge more money. Do not add a signature or attribution of any artists' names (including your own) to the drawing anywhere on it."
    
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
