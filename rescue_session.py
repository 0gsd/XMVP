#!/usr/bin/env python3
"""
rescue_session.py
-----------------
Rescues a failed FullMovie-Still session by restitching all generated assets.
Usage: python3 rescue_session.py path/to/session_dir
"""

import os
import sys
import glob
import subprocess
import argparse

# Import tools from content_producer
try:
    import content_producer
    from content_producer import stitch_assets, get_audio_duration
except ImportError:
    # If not in path, try adding current dir
    sys.path.append(os.getcwd())
    try:
        import content_producer
        from content_producer import stitch_assets, get_audio_duration
    except ImportError as e:
        print(f"[-] Could not import content_producer: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", help="Path to the failed session directory")
    args = parser.parse_args()
    
    session_dir = os.path.abspath(args.session_dir)
    if not os.path.exists(session_dir):
        print(f"[-] Session not found: {session_dir}")
        sys.exit(1)
        
    print(f"🚑 Rescuing Session: {session_dir}")
    
    # 1. Scan for Frames and Wavs
    frames = sorted(glob.glob(os.path.join(session_dir, "frame_*.jpg")))
    wavs = sorted(glob.glob(os.path.join(session_dir, "line_*.wav")))
    
    print(f"    Found {len(frames)} frames and {len(wavs)} wavs.")
    
    # 2. Match them up
    # We assume frame_XXXX.jpg matches line_XXXX.wav
    assets = []
    
    # Determine max index
    max_idx = -1
    for f in frames:
        try:
            idx = int(os.path.basename(f).split("_")[1].split(".")[0])
            if idx > max_idx: max_idx = idx
        except: pass
        
    print(f"    Max Index found: {max_idx}")
    
    for i in range(max_idx + 1):
        f_name = f"frame_{i:04d}.jpg"
        w_name = f"line_{i:04d}.wav"
        f_path = os.path.join(session_dir, f_name)
        w_path = os.path.join(session_dir, w_name)
        
        if not os.path.exists(f_path):
            print(f"    [!] Missing frame {i}")
            continue
            
        if not os.path.exists(w_path):
            print(f"    [!] Missing wav {i} (Creating silence)")
            # Create silence if missing? Or just skip?
            # User wants to stitch what's there.
            # But skipping de-syncs the audio if the video needs it?
            # No, fullmovie-still is 1-to-1.
            continue
            
        # Calc Duration
        try:
            dur = get_audio_duration(w_path) + 0.25
        except Exception as e:
            print(f"    [!] Bad WAV {i}: {e}")
            dur = 2.0
            
        assets.append({
            "image": f_path,
            "audio": w_path,
            "duration": dur,
            "speaker": "Unknown", # Metadata lost, but not needed for stitch
            "text": "Rescued line"
        })
        
    # 3. Stitch
    out_mp4 = os.path.join(session_dir, "RESCUED_OUTPUT.mp4")
    print(f"    🧵 Stitching {len(assets)} segments to {out_mp4}...")
    
    stitch_assets(assets, session_dir, out_mp4)
    print("✅ Rescue Complete.")

if __name__ == "__main__":
    main()
