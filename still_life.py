#!/usr/bin/env python3
"""
still_life.py
-------------
"Movie-fies" a static storyboard/podcast video by using Wan 2.1 (Speech-to-Video).

Workflow:
1. Inputs: MP4 Video + XMVP XML Manifest
2. Slices the source MP4 into per-scene Audio + Keyframe pairs.
3. Uses Local Gemma to generate a 'Sassprilla' prompt for the scene.
4. Uses Wan 2.1 to generate a video clip from Image + Audio + Text.
5. Stitches the result into a full motion movie.

Author: Antigravity
Date: Jan 2026
"""

import os
import sys
import argparse
import subprocess
import glob
import logging
import time
import gc

from pathlib import Path
import xml.etree.ElementTree as ET

# MVP Imports
try:
    import definitions
    from text_engine import TextEngine
    from hunyuan_video_bridge import HunyuanVideoBridge
except ImportError as e:
    print(f"[-] Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_xmvp_path(mp4_path):
    """Attempts to find the matching XML manifest."""
    base = os.path.splitext(mp4_path)[0]
    # 1. Exact match .xml
    if os.path.exists(f"{base}.xml"): return f"{base}.xml"
    # 2. _manifest.xml (Content Producer standard)
    if os.path.exists(f"{base}_manifest.xml"): return f"{base}_manifest.xml"
    # 3. Look in same dir
    dir_name = os.path.dirname(mp4_path)
    fname = os.path.basename(base)
    candidates = glob.glob(os.path.join(dir_name, "*.xml"))
    for c in candidates:
        if fname in c: return c
    return None

def parse_xmvp_portions(xml_path):
    """Extracts Portions (Scenes) with durations and dialogue."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    portions = []
    
    # Try Portions list
    # Structure: <Portions>[{"id":1, ...}]</Portions> (JSON) OR <Portions><Portion>...</Portion></Portions> (Legacy XML)
    p_node = root.find("Portions")
    if p_node is not None:
        # Check for XML children first (Legacy)
        xml_portions = p_node.findall("Portion")
        if xml_portions:
             for p in xml_portions:
                item = {
                    "id": int(p.find("id").text),
                    "duration": float(p.find("duration_sec").text),
                    "content": p.find("content").text
                }
                portions.append(item)
        else:
            # Try JSON parsing
            try:
                import json
                p_text = p_node.text.strip() if p_node.text else ""
                if p_text:
                    json_portions = json.loads(p_text)
                    # Normalize keys to match legacy structure
                    portions = []
                    for jp in json_portions:
                        portions.append({
                            "id": jp.get("id"),
                            "duration": jp.get("duration_sec", jp.get("duration", 5.0)),
                            "content": jp.get("content", "")
                        })
            except Exception as e:
                logging.warning(f"   ⚠️ Failed to parse Portions as JSON: {e}")

    # Enrich with Dialogue (Rich Actions/Visuals)
    d_node = root.find("Dialogue")
    if d_node is not None and len(portions) > 0:
        # Check XML (Legacy)
        xml_lines = d_node.findall("Line")
        
        dialogue_data = []
        if xml_lines:
             # Legacy XML parsing
             for line in xml_lines:
                 dialogue_data.append({
                     "action": line.find("action").text if line.find("action") is not None else "",
                     "visual_focus": line.find("visual_focus").text if line.find("visual_focus") is not None else "",
                     "character": line.find("character").text if line.find("character") is not None else "",
                     "text": line.find("text").text if line.find("text") is not None else ""
                 })
        else:
             # Try JSON
             try:
                 import json
                 d_text = d_node.text.strip() if d_node.text else ""
                 if d_text:
                     d_json = json.loads(d_text)
                     # Structure: {"lines": [...]}
                     dialogue_data = d_json.get("lines", [])
             except Exception as e:
                 logging.warning(f"   ⚠️ Failed to parse Dialogue as JSON: {e}")

        # Enrich
        if len(dialogue_data) == len(portions):
            logging.info("   ✅ Found matching Dialogue Script. Enriching prompts...")
            for i, line in enumerate(dialogue_data):
                # Extract extras (handle dict vs object/element access implicitly by dict get)
                action = line.get("action", "") or ""
                focus = line.get("visual_focus", "") or ""
                char = line.get("character", "") or ""
                text = line.get("text", "") or ""
                
                # Construct richer context
                # "Action: [Standing]. Focus: [Face]. Character [Name] says: '[Text]'"
                rich_content = []
                if action: rich_content.append(f"Action: {action}.")
                if focus: rich_content.append(f"Visual Focus: {focus}.")
                if char and text: rich_content.append(f"Character {char} says: '{text}'")
                elif portions[i].get("content"): rich_content.append(f"Context: {portions[i]['content']}")
                
                portions[i]["content"] = " ".join(rich_content)
        else:
            logging.warning(f"   ⚠️ Dialogue line count ({len(dialogue_data)}) != Portion count ({len(portions)}). Using basic Portions.")
        
    return portions

def slice_segment(input_mp4, start_time, duration, output_dir, idx):
    """
    Extracts Audio (WAV) and Start Frame (JPG) for the segment.
    """
    seg_audio = os.path.join(output_dir, f"seg_{idx:03d}.wav")
    seg_frame = os.path.join(output_dir, f"seg_{idx:03d}.jpg")
    
    # 1. Extract Audio
    cmd_audio = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", input_mp4,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        seg_audio,
        "-loglevel", "error"
    ]
    subprocess.run(cmd_audio)
    
    # 2. Extract First Frame (Visual Reference)
    # We grab the frame just slightly after start to avoid fade-in/black issues?
    # Content producer creates static images, so start_time + 0.1 is safe.
    cmd_frame = [
        "ffmpeg", "-y",
        "-ss", str(start_time + 0.1),
        "-i", input_mp4,
        "-frames:v", "1",
        "-q:v", "2",
        seg_frame,
        "-loglevel", "error"
    ]
    subprocess.run(cmd_frame)
    
    return seg_audio, seg_frame

def mux_audio_video(video_path, audio_path, output_path, duration):
    """
    Loops/Extends video to match audio duration and muxes them.
    """
    # 1. Loop video to match duration
    # stream_loop -1 loops indefinitely, -t cuts it.
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-t", str(duration),
        "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
        "-loglevel", "error"
    ]
    subprocess.run(cmd)
    return os.path.exists(output_path)

def main():
    parser = argparse.ArgumentParser(description="Still Life: Movie-fy Static Storyboards with Wan 2.1")
    parser.add_argument("--input", required=True, help="Input MP4 file (from content_producer)")
    parser.add_argument("--xml", help="Optional explicit path to XMVP XML")
    parser.add_argument("--local", action="store_true", help="Force Local Mode (Implied for Wan)")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--w", type=int, default=256, help="Width (Default: 256)")
    parser.add_argument("--h", type=int, default=144, help="Height (Default: 144)")
    
    args = parser.parse_args()
    
    # 1. Setup Inputs
    mp4_path = os.path.abspath(args.input)
    if not os.path.exists(mp4_path):
        logging.error(f"Input file not found: {mp4_path}")
        return
        
    xml_path = args.xml if args.xml else get_xmvp_path(mp4_path)
    if not xml_path or not os.path.exists(xml_path):
        logging.error("Could not find matching XMVP XML manifest. Please provide --xml.")
        return
        
    logging.info(f"🎨 Still Life: Animating {os.path.basename(mp4_path)}")
    logging.info(f"   📜 Manifest: {os.path.basename(xml_path)}")
    
    # 2. Setup Output
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    out_dir = args.out if args.out else os.path.join(os.path.dirname(mp4_path), f"still_life_{base_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    staging_dir = os.path.join(out_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    # 3. Parse Metadata
    portions = parse_xmvp_portions(xml_path)
    if not portions:
        logging.error("No portions found in XML.")
        return
        
    logging.info(f"   ✂️  Found {len(portions)} Scenes to Animate.")
    
    # 4. Engine Init
    # Text Engine for Sassprilla Prompts
    os.environ["TEXT_ENGINE"] = "local_gemma" # Force local for speed/privacy
    txt_engine = TextEngine()
    
    # Video Engine (Hunyuan via Comfy)
    video_bridge = HunyuanVideoBridge()
    if not video_bridge.comfy.check_server():
        logging.error("❌ ComfyUI Server not reachable. Please run 'sh start_comfyui.sh' first.")
        return
    
    # 5. Loop
    current_time = 0.0
    clips = []
    


    # 5. Loop
    current_time = 0.0
    clips = []
    
    for i, p in enumerate(portions):
        pid = p["id"]
        dur = p["duration"]
        content = p["content"] # Speaker: Text
        
        logging.info(f"\n🎬 Scene {pid}/{len(portions)} ({dur:.2f}s): {content[:40]}...")
        
        # A. Slice
        s_audio, s_frame = slice_segment(mp4_path, current_time, dur, staging_dir, i)
        
        if not os.path.exists(s_audio) or not os.path.exists(s_frame):
            logging.warning("   ⚠️ Slice failed. Skipping.")
            current_time += dur
            continue
            
        # B. Prompt Engineering (Sassprilla)
        prompt_request = (
            f"You are a film director. Convert this script context into a vivid, photorealistic AI Video Prompt.\n"
            f"Script Context: {content}\n"
            f"Requirements: Describe the visual action, lighting, and camera movement. No text overlays.\n"
            f"Output: A single concise prompt string."
        )
        
        try:
            wan_prompt = txt_engine.generate(prompt_request).strip().strip('"')
            if len(wan_prompt) < 10: raise Exception("Empty prompt")
        except:
            wan_prompt = f"Cinematic shot of {content}, photorealistic, detailed, 4k, slow motion."
            
        logging.info(f"   ✨ Prompt: {wan_prompt[:50]}...")
        
        # C. Generate Video (Silent Raw)
        raw_clip_name = f"raw_{i:03d}.mp4"
        raw_clip_path = os.path.join(staging_dir, raw_clip_name)
        
        success = video_bridge.generate(
            prompt=wan_prompt,
            # image_path=s_frame, # Hunyuan is T2V only for now
            output_dir=staging_dir,
            width=args.w,
            height=args.h
        )
        
        if success and os.path.exists(raw_clip_path):
            # D. Mux & Extend
            final_clip_name = f"clip_{i:03d}.mp4"
            final_clip_path = os.path.join(staging_dir, final_clip_name)
            
            mux_ok = mux_audio_video(raw_clip_path, s_audio, final_clip_path, dur)
            
            if mux_ok:
                clips.append(final_clip_path)
                logging.info(f"   ✅ Clip {i} Ready ({dur:.1f}s)")
            else:
                logging.error("   ❌ Mux Failed.")
        else:
            logging.error("   ❌ Wan Generation Failed.")
            
        current_time += dur
        
        # Safety save list
        with open(os.path.join(out_dir, "clips.txt"), 'w') as f:
            for c in clips: f.write(f"file '{c}'\n")
            
        # Memory Cleanup
        gc.collect()

    # 6. Concat
    if clips:
        logging.info("🧵 Stitching Final Movie...")
        final_out = os.path.join(out_dir, f"StillLife_{base_name}.mp4")
        
        # We need to concat. Ffmpeg concat demuxer.
        # Re-write clip list
        with open(os.path.join(out_dir, "clips.txt"), "w") as f:
             for c in clips: f.write(f"file '{c}'\n")
             
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", os.path.join(out_dir, "clips.txt"),
            "-c", "copy",
            final_out,
            "-loglevel", "error"
        ])
        
        if os.path.exists(final_out):
            logging.info(f"✅ STILL LIFE COMPLETE: {final_out}")
        else:
            logging.error("❌ Stitch failed.")
    


if __name__ == "__main__":
    main()
