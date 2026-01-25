
import os
import sys
import random
import logging
import subprocess
import json
from pathlib import Path
import math

# Third Party (Dynamic Import to avoid crashes if missing)
try:
    import librosa
    import pyloudnorm as pyln
    import numpy as np
except ImportError:
    librosa = None
    pyln = None
    np = None

# Set up Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_audio_duration(file_path):
    try:
        cmd = ['ffprobe', '-i', file_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
        result = subprocess.run(cmd, capture_output=True, text=True)
        val = float(result.stdout.strip())
        return val
    except:
        return 0.0

def analyze_audio_profile(audio_path, duration):
    """
    Simplified Sonic Map (Ported/Adapted).
    Returns list of intensity values (0.0 to 1.0) per second (approx).
    """
    if not librosa or not pyln:
        logging.warning("   ⚠️ Librosa/Pyloudnorm missing. Using random intensity map.")
        return [random.random() for _ in range(int(duration))]
        
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate RMS energy over 1-second windows
        hop_length = sr # 1 second
        if len(y) < hop_length: hop_length = len(y)
        
        rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
        
        # Normalize RMS to 0-1
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
        
        # Smooth
        return rms_norm.tolist()
    except Exception as e:
        logging.error(f"   ❌ Audio Analysis Failed: {e}")
        return [0.5] * int(duration)

def scan_for_videos(folder_path):
    """Recursively finds video files."""
    exts = ['.mp4', '.mov', '.mkv', '.avi', '.m4v']
    videos = []
    folder = Path(folder_path)
    if not folder.exists(): return []
    
    for f in folder.rglob("*"):
        if f.suffix.lower() in exts and not f.name.startswith("._"):
            videos.append(f)
            
    return videos

def select_clip(source_video, duration):
    """
    Selects a random segment of `duration` from `source_video`.
    Returns (start_time, end_time) or None.
    """
    total_dur = get_audio_duration(str(source_video))
    if total_dur < duration: return None
    
    start = random.uniform(0, total_dur - duration)
    return start

def check_overlap(new_start, new_end, used_intervals):
    """
    Checks if [new_start, new_end] overlaps with any (start, end) in used_intervals.
    Returns True if overlap detected.
    """
    for u_start, u_end in used_intervals:
        # Standard overlap check: StartA < EndB and EndA > StartB
        if new_start < u_end and new_end > u_start:
            return True
    return False

def select_clip_smart(source_video, duration, usage_tracker):
    """
    Selects a non-overlapping random segment.
    usage_tracker: { 'intervals': [(s,e), ...], 'duration': total_sec }
    """
    total_dur = get_audio_duration(str(source_video))
    if total_dur < duration: return None
    
    # Try finding a free spot (Max Retries)
    max_retries = 20
    for _ in range(max_retries):
        start = random.uniform(0, total_dur - duration)
        end = start + duration
        
        # Check Tracker
        used = usage_tracker.get(source_video, [])
        if not check_overlap(start, end, used):
            return start, end
            
    # If we failed 20 times, the video is likely saturated.
    # User Policy: "not repeat... unless it has run out of options"
    # Fallback: Just return a random one (allow overlap) but Log it.
    start = random.uniform(0, total_dur - duration)
    return start, start + duration

def run_clip_video_pipeline(args):
    """
    Main Entry Point for Clip-Video Mode (v2: Smart Selection).
    """
    logging.info(f"🎬 Starting Clip-Video Pipeline (v2: Smart Beat Sheet)")
    logging.info(f"   📂 Source: {args.f}")
    logging.info(f"   🎵 Audio:  {args.mu}")
    
    if not args.f or not os.path.exists(args.f):
        logging.error("❌ Source folder (--f) not found.")
        return False
        
    if not args.mu or not os.path.exists(args.mu):
        logging.error("❌ Audio file (--mu) not found.")
        return False
        
    # 1. Setup Output
    out_dir = Path(args.out) if args.out else Path("z_test-outputs/clip_video")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Analyze Audio
    duration = get_audio_duration(args.mu)
    logging.info(f"   ⏱️  Target Duration: {duration:.1f}s")
    
    intensity_map = analyze_audio_profile(args.mu, duration)
    logging.info(f"   📊 Sonic Map generated ({len(intensity_map)} points).")
    
    # 3. Scan & Prep Videos
    videos = scan_for_videos(args.f)
    if not videos:
        logging.error("❌ No videos found in source folder.")
        return False
        
    logging.info(f"   🎞️  Found {len(videos)} source videos.")
    
    # 4. Generate Beat Sheet (The Schedule)
    logging.info("   📅 Generating Cut Schedule (The Beat Sheet)...")
    cut_schedule = [] # List of tuples: (start_time, duration, intensity, type)
    sched_time = 0.0
    
    # Tunables
    # Quiet (Low Energy): Long, contemplative shots.
    MAX_QUIET_LEN = 8.0
    MIN_QUIET_LEN = 4.0
    
    # Loud (High Energy): Fast, hectic cuts.
    MAX_LOUD_LEN = 2.0
    MIN_LOUD_LEN = 0.5 # Sub-second cuts for high intensity
    
    # Thresholds
    LOUD_THRESHOLD = 0.6 # Normalized RMS > 0.6 is "High Energy"
    
    while sched_time < duration:
        # Check Intensity
        map_idx = min(int(sched_time), len(intensity_map)-1)
        intensity = intensity_map[map_idx]
        
        # Determine Target Length
        if intensity > LOUD_THRESHOLD:
            # High Energy Zone
            # Stronger intensity -> Shorter clip
            # Map 0.6..1.0 to MAX_LOUD..MIN_LOUD
            norm_i = (intensity - LOUD_THRESHOLD) / (1.0 - LOUD_THRESHOLD)
            target_len = MAX_LOUD_LEN - (norm_i * (MAX_LOUD_LEN - MIN_LOUD_LEN))
            slot_type = "ACTION"
        else:
            # Low Energy Zone
            # Lower intensity -> Longer clip
            # Map 0.0..0.6 to MAX_QUIET..MIN_QUIET
            # Note: 0.0 is MAX_QUIET, 0.6 is MIN_QUIET
            norm_i = intensity / LOUD_THRESHOLD
            target_len = MAX_QUIET_LEN - (norm_i * (MAX_QUIET_LEN - MIN_QUIET_LEN))
            slot_type = "DRAMA"
            
        # Jitter
        target_len *= random.uniform(0.9, 1.1)
        
        # Clamp
        target_len = max(MIN_LOUD_LEN, min(target_len, MAX_QUIET_LEN))
        
        # Final Fit
        if sched_time + target_len > duration:
            target_len = duration - sched_time
            
        cut_schedule.append({
            "time": sched_time,
            "duration": target_len,
            "intensity": intensity,
            "type": slot_type
        })
        sched_time += target_len
        
    logging.info(f"   📝 Scheduled {len(cut_schedule)} cuts.")

    # 5. Metadata-Aware Scanning
    # Pre-fetch durations for deck to enable smart filtering
    logging.info("   🧠 Analyzing Source Library...")
    deck = []
    for v in videos:
        d = get_audio_duration(str(v))
        if d > 0.5:
             deck.append({
                 "path": v,
                 "duration": d,
                 "name": v.name
             })
    
    if not deck:
         logging.error("❌ No valid videos found/analyzed.")
         return False
         
    # Shuffle initially
    random.shuffle(deck)
    
    usage_map = { v["path"]: [] for v in deck }
    clips = []
    staging_dir = out_dir / "staging"
    staging_dir.mkdir(exist_ok=True)
    
    # 6. Fill the Slots
    for i, slot in enumerate(cut_schedule):
        target_dur = slot["duration"]
        slot_type = slot["type"]
        
        # SMART SELECTION STRATEGY
        # Filter deck for candidates that can support this duration
        # For DRAMA (Long) slots: Prefer longest available clips and least used?
        # For ACTION (Short) slots: Anything goes.
        
        candidates = [v for v in deck if v["duration"] >= target_dur]
        
        if not candidates:
             # Crisis: No source video is long enough for this slot.
             candidates = deck

        # Selection Weighting
        selected = None
        
        # Try to pick a candidate that hasn't been used much (Fairness)
        # Sort by usage count (len(usage_map[path]))
        candidates.sort(key=lambda x: len(usage_map[x["path"]]))
        
        # Take from the top tier of least used (first 25% or 5 items)
        tier_size = max(1, len(candidates) // 4)
        top_tier = candidates[:tier_size]
        
        # Random pick from top tier
        candidate = random.choice(top_tier)
        
        # Find Interval
        # Using existing smart selector
        res = select_clip_smart(candidate["path"], target_dur, usage_map)
        
        if not res:
             # Retry with others in top tier
             for alt in top_tier:
                 res = select_clip_smart(alt["path"], target_dur, usage_map)
                 if res:
                     candidate = alt
                     break
        
        # Last Resort: brute force any candidate
        if not res:
             random.shuffle(candidates)
             for alt in candidates[:10]: # Try 10 randoms
                 res = select_clip_smart(alt["path"], target_dur, usage_map)
                 if res:
                     candidate = alt
                     break
                     
        if res:
            start_t, end_t = res
            usage_map[candidate["path"]].append((start_t, end_t))
            
            # Extract
            source = candidate["path"]
            out_name = f"clip_{i:04d}.mp4"
            out_path = staging_dir / out_name
            
            vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_t),
                "-t", str(target_dur),
                "-i", str(source),
                "-vf", vf,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(out_path)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                clips.append(out_path)
                
                # Log with emoji based on type
                emoji = "🧘" if slot_type == "DRAMA" else "💥"
                logging.info(f"   {emoji} Clip {i}: {target_dur:.2f}s from {candidate['name']} (I:{slot['intensity']:.2f})")
            except Exception as e:
                logging.warning(f"Failed extract: {e}")
        else:
             logging.warning(f"   ⚠️ Skipping Slot {i} ({target_dur}s) - No valid interval found.")

    # 5. Stitch
    logging.info(f"   🧵 Stitching {len(clips)} clips...")
    list_path = out_dir / "stitch_list.txt"
    with open(list_path, 'w') as f:
        for c in clips:
            f.write(f"file '{c.resolve()}'\n")
            
    # Final Output Name
    clean_name = Path(args.mu).stem.replace(" ", "_")
    final_vid = out_dir / f"ClipVideo_{clean_name}_{random.randint(100,999)}.mp4"
    
    cmd_stitch = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(final_vid)
    ]
    subprocess.run(cmd_stitch, check=True)
    
    # 6. Add Music
    logging.info("   🎵 Mixing Audio...")
    final_muxed = out_dir / f"Final_{clean_name}.mp4"
    cmd_mix = [
        "ffmpeg", "-y",
        "-i", str(final_vid),
        "-i", str(args.mu),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-shortest",
        str(final_muxed)
    ]
    subprocess.run(cmd_mix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    logging.info(f"✅ DONE: {final_muxed}")
    return True
