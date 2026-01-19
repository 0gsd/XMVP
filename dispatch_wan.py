#!/usr/bin/env python3
"""
dispatch_wan.py
Orchestrator for Wan 2.1 Video Generation Pipeline.
Handles:
- Dialogue Audio Verification
- Keyframe Generation (Flux)
- Video Generation (Wan)
- Sequential Chain (Last Frame -> Next Keyframe)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import shutil
import subprocess

# Local Imports
import definitions
from definitions import Modality
from flux_bridge import get_flux_bridge
from wan_bridge import get_wan_bridge

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_wan_pipeline(manifest_path, out_path, staging_dir, args):
    """
    Main Loop for Wan Video Generation.
    """
    logging.info(f"📹 Starting Wan Dispatch: {manifest_path}")
    
    manifest = load_json(manifest_path)
    portions = manifest.get("portions", [])
    
    # Ensure Staging
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = staging_dir / "keyframes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = staging_dir / "clips"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Bridges
    # Flux for Keyframes
    flux_model = definitions.MODAL_REGISTRY[Modality.IMAGE].get("flux-schnell")
    flux = get_flux_bridge(flux_model.path if flux_model else None)
    
    # Wan for Video
    wan = get_wan_bridge()
    
    # State
    last_frame_path = None
    
    updated_portions = []
    
    for i, portion in enumerate(portions):
        pid = portion['id']
        text = portion['content']
        duration = portion.get('duration', 4.0)
        
        logging.info(f"\n🎬 Processing Clip {pid}: {text[:40]}...")
        
        # 1. Resolve Audio
        # Assuming writers_room/portion_control already generated audio?
        # If not, we might need to mock it or generate it here if TTS was skipped?
        # For now, let's assume 'audio_path' is in portion or we derive it.
        # Check standard pathes: staging/audio/portion_{pid}.wav?
        # Actually, manifest usually has 'audio_path' if portion_control ran.
        # But movie_producer flow: writers_room -> portion_control (generates manifest).
        # portion_control typically does NOT generate audio unless requested?
        # Wait, THAX mode does. Tech-movie mode... usually text only.
        # User said: "We will generate all dialogue lines BEFORE... even if we throw away"
        # So we expect Audio to exist.
        
        audio_path = portion.get('audio_path')
        if not audio_path or not os.path.exists(audio_path):
            logging.warning(f"   ⚠️ No Audio for Clip {pid}. Generating silence or skipping?")
            # TODO: Generate TTS here if missing?
            # For now, let's proceed without audio if Wan supports T2V, 
            # OR generate strict silence.
            # wan.generate likely needs audio for lip sync.
            # Fallback: Generate TTS using Kokoro Bridge if available
            try:
                from kokoro_bridge import get_kokoro_bridge
                kokoro = get_kokoro_bridge()
                
                # Default Voice? Bella, or random?
                # Let's use 'af_bella' as standard default
                temp_audio_path = staging_dir / f"audio_{pid}.wav"
                
                logging.info("   🎙️ Generating Fallback TTS (Kokoro)...")
                audio_path = kokoro.generate(text, voice="af_bella", output_path=str(temp_audio_path))
                
                if audio_path:
                    portion['audio_path'] = str(audio_path)
                    logging.info(f"   ✅ TTS Generated: {audio_path}")
            except Exception as e_tts:
                logging.warning(f"   ⚠️ TTS Generation Failed: {e_tts}")
                pass
            
        # 2. Resolve Keyframe
        keyframe_path = frames_dir / f"key_{pid:04d}.png"
        
        if i == 0 or not last_frame_path:
            # Generate Fresh Keyframe via Flux
            vis_prompt = f"Cinematic shot. {text}. High resolution, 8k." # Enrich this
            logging.info(f"   🖼️ Generating Start Keyframe: {vis_prompt[:30]}...")
            img = flux.generate(vis_prompt, width=1280, height=720) # Wan 14B usually likes 720p?
            if img:
                img.save(keyframe_path)
            else:
                logging.error("   ❌ Flux failed.")
                continue
        else:
            # Use Last Frame of Previous Clip
            logging.info(f"   🔗 Chaining Keyframe from previous clip...")
            shutil.copy(last_frame_path, keyframe_path)
            
        # 3. Generate Video (Wan)
        clip_name = f"clip_{pid:04d}.mp4"
        clip_path = video_dir / clip_name
        
        success = wan.generate(
            prompt=text,
            image_path=str(keyframe_path),
            audio_path=audio_path, # Might be None
            output_path=str(clip_path)
        )
        
        if success:
            portion['video_path'] = str(clip_path)
            
            # 4. Extract Last Frame for Next Chain
            # Use ffmpeg to extract last frame
            last_frame_path = frames_dir / f"last_{pid:04d}.png"
            try:
                cmd = [
                    "ffmpeg", "-y", 
                    "-sseof", "-3", # Look at last 3 seconds? No, last frame.
                    "-i", str(clip_path), 
                    "-vsync", "0", 
                    "-frame_pts", "1", 
                    "-update", "1", 
                    str(last_frame_path)
                ]
                # Better: -sseof -1 -i input -update 1 -q:v 1 out.png
                # Actually, getting exact last frame can be tricky.
                # Let's try:
                subprocess.run(
                    ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(clip_path), "-update", "1", "-q:v", "2", str(last_frame_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if not last_frame_path.exists():
                    logging.warning("   ⚠️ Could not extract last frame. Will Regen next keyframe.")
                    last_frame_path = None
            except Exception as e:
                logging.warning(f"   ⚠️ FFmpeg extract failed: {e}")
                last_frame_path = None
        
        updated_portions.append(portion)

    # Save Updated Manifest
    manifest['portions'] = updated_portions
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    return True

if __name__ == "__main__":
    # Test Stub
    pass
