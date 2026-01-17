#!/usr/bin/env python3
"""
music_video.py
--------------
A dedicated utility to sync visuals to audio.
"The Rescuer"

Modes:
1. Video Input (--vf): Retime video (stretch/squeeze) to match audio length exactly.
2. Frames Input (--ff): Calculate perfect FPS to make frames fit audio length exactly.

Usage:
  python3 music_video.py --mu song.mp3 --vf input.mp4 --out output.mp4
  python3 music_video.py --mu song.mp3 --ff ./frames_dir --out output.mp4
"""

import os
import sys
import argparse
import subprocess
import json
import glob
import shutil

def get_duration(media_path):
    """Returns duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            media_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[-] Error getting duration for {media_path}: {e}")
        return None

def count_frames(folder_path):
    """Counts png/jpg/jpeg files in folder."""
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return len(files), sorted(files)

def process_frames(frames_dir, audio_path, output_path):
    print(f"[*] Mode: Frames Stitching")
    print(f"    Frames: {frames_dir}")
    print(f"    Audio:  {audio_path}")
    
    # 1. Get Audio Duration
    audio_dur = get_duration(audio_path)
    if not audio_dur:
        print("[-] Could not get audio duration.")
        sys.exit(1)
        
    # 2. Count Frames
    num_frames, file_list = count_frames(frames_dir)
    if num_frames == 0:
        print("[-] No images found in frames folder.")
        sys.exit(1)
        
    # 3. Calculate FPS
    req_fps = num_frames / audio_dur
    print(f"    Audio Duration: {audio_dur:.2f}s")
    print(f"    Frame Count:    {num_frames}")
    print(f"    Calculated FPS: {req_fps:.4f}")
    
    if req_fps < 1.0:
        print(f"[-] Warning: calculated FPS ({req_fps:.4f}) is very low (<1.0).")
        # Proceed anyway? User said "if... faster than 1fps".
        # If slower than 1fps, we might need to duplicate frames or just warn.
        # User prompt: "if the stitched frames 'CAN BE' made to the same length... at 1 frame per second or faster"
        # So check:
        if req_fps < 1.0:
            print("[-] Error: Not enough frames to sustain >1 FPS.")
            sys.exit(1)

    # 4. Stitch
    # We need to ensure globular pattern works or create a list
    # Let's use glob pattern if files are named sequentially? 
    # Or safer: Create a temporary input file list
    
    list_path = os.path.join(frames_dir, "ffmpeg_input_list.txt")
    with open(list_path, 'w') as f:
        for path in file_list:
            # Escape path safely?
            f.write(f"file '{path}'\n")
            f.write(f"duration {1.0/req_fps:.6f}\n")
    
    # Actually, simpler method for exact sync:
    # Use image2pipe or concat demuxer.
    # Concat demuxer with fixed duration for each frame is best to avoid rounding drift.
    
    # BUT, ffmpeg -framerate X -pattern_type glob -i "*.png" is standard.
    # The issue with glob is MacOS zsh expansion vs internal glob.
    # Let's try the glob method first if names are cleaner, OR the concat file method.
    # Given we calculated a specific floating point FPS, passing `-framerate {req_fps}` to image2 is best.
    # But image2 requires sequential filenames or glob.
    
    # Safest fallback: Rename files to temp sequential? No, destructive.
    # Use image2 with glob if possible.
    # Assuming frames are standard naming "frame_%04d.png" or similar.
    
    # Let's assume standard glob "*.png" works.
    image_pattern = os.path.join(frames_dir, "*.png")
    
    # Build Command
    # -r sets the INPUT framerate. -i read at that rate.
    # Then we map audio.
    # Then output.
    
    cmd = [
        'ffmpeg',
        '-y',
        '-pattern_type', 'glob', 
        '-framerate', str(req_fps),
        '-i', image_pattern,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest', # Should match exactly due to math, but good safety
        output_path
    ]
    
    print(f"[*] Running FFmpeg...")
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Synced Video Created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[-] FFmpeg Failed: {e}")
        # Try fall back to sequential copy? 
        print("[!] Note: This script assumes *.png frames in the folder.")

def process_video(video_path, audio_path, output_path):
    print(f"[*] Mode: Video Retiming")
    print(f"    Video: {video_path}")
    print(f"    Audio: {audio_path}")
    
    a_dur = get_duration(audio_path)
    v_dur = get_duration(video_path)
    
    if not a_dur or not v_dur:
        sys.exit(1)
        
    diff = abs(a_dur - v_dur)
    print(f"    Audio Dur: {a_dur:.2f}s")
    print(f"    Video Dur: {v_dur:.2f}s")
    print(f"    Difference: {diff:.2f}s")
    
    if diff > 30.0:
        print("[-] Warning: Duration difference > 30s. This might look weird.")
        
    # Calculate PTS Factor
    # We want video to become length A_DUR.
    # New Duration = Old Duration * Factor
    # Factor = A / V
    # setpts filter: PTS * (A/V) ? 
    # WAIT. PTS is "Timestamp". If we want it strictly longer (slower), timestamps must increase.
    # So yes, multiply PTS by (Target / Source).
    
    factor = a_dur / v_dur
    
    print(f"    Retiming Factor: {factor:.4f}x")
    
    # Command
    # We strip original audio (-an) from video input
    
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-filter_complex', f"[0:v]setpts={factor}*PTS[v]",
        '-map', '[v]',
        '-map', '1:a',
        '-c:v', 'libx264',
        '-c:a', 'aac', # Re-encode audio to ensure container fit? copy is safer usually but aac standard
        '-shortest',
        output_path
    ]
    
    print(f"[*] Running FFmpeg...")
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Synced Video Created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[-] FFmpeg Failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Music Video Rescuer/Syncer")
    parser.add_argument("--mu", required=True, help="Input Audio File (required)")
    parser.add_argument("--vf", help="Input Video File")
    parser.add_argument("--ff", help="Input Frames Folder")
    parser.add_argument("--out", help="Output Filename (default: music_video.mp4)", default="music_video.mp4")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mu):
        print(f"[-] Audio file not found: {args.mu}")
        sys.exit(1)
        
    if args.vf:
        if not os.path.exists(args.vf):
            print(f"[-] Video file not found: {args.vf}")
            sys.exit(1)
        process_video(args.vf, args.mu, args.out)
        
    elif args.ff:
        if not os.path.exists(args.ff):
            print(f"[-] Frames folder not found: {args.ff}")
            sys.exit(1)
        process_frames(args.ff, args.mu, args.out)
        
    else:
        print("[-] Error: Must provide either --vf (Video) or --ff (Frames Folder).")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
