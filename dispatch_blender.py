"""
dispatch_blender.py
Main entry point for rendering XMVP Manifests via Blender (Subprocess Mode).
"""

import os
import sys
import logging
import argparse
import subprocess
import json
from pathlib import Path

# Local
from mvp_shared import load_manifest, Manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_blender_binary():
    """Tries to find Blender executable."""
    # Common macOS paths
    paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender" # Environment path
    ]
    for p in paths:
        if os.path.exists(p):
            return p
        # Check Shell path
        try:
            subprocess.run([p, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return p
        except:
            continue
    return None

def run_blender_worker(args_list):
    """Executes blender_worker.py inside Blender with given args."""
    binary = get_blender_binary()
    if not binary:
        logging.error("❌ Blender binary not found. Please install Blender 4.3+.")
        return False
        
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_worker.py")
    
    cmd = [
        binary,
        "--background",
        "--python", worker_script,
        "--"
    ] + args_list
    
    logging.info(f"🚀 Spawning Blender: {' '.join(cmd)}")
    try:
        # Run and capture output to log? or just inherit stdout
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Blender Process Failed: {e}")
        return False

def run_dispatch(manifest_path, out_path, library_path):
    """
    Renders all shots in the manifest.
    """
    logging.info(f"🎬 Dispatching Blender Jobs for: {manifest_path}")
    
    manifest = load_manifest(manifest_path)
    output_dir = os.path.dirname(out_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rendered_files = {}

    import glob

    for i, seg in enumerate(manifest.segs):
        shot_name = f"shot_{seg.id:03d}.mp4"
        shot_out = os.path.join(output_dir, shot_name)
        
        # Temporary prefix for PNG sequence
        # Blender appends frame numbers (e.g. 0001.png) automatically to the path
        raw_prefix = os.path.join(output_dir, f"shot_{seg.id:03d}_raw_")
        
        logging.info(f"🎥 Rendering Shot {seg.id}: {shot_out}")

        # Calculate duration
        duration = 4.0
        
        # Subprocess Call (Blender renders PNGs)
        success = run_blender_worker([
            "render-shot",
            "--lib", library_path,
            "--out", raw_prefix,
            "--duration", str(duration)
        ])
        
        if success:
            # Check if PNGs were generated
            # Blender format is usually raw_prefix0001.png
            first_frame = f"{raw_prefix}0001.png"
            if os.path.exists(first_frame):
                logging.info(f"   🧵 Stitching PNGs to MP4...")
                
                # Check for Audio
                audio_input = []
                audio_map = []
                if hasattr(seg, 'audio_asset') and seg.audio_asset and os.path.exists(seg.audio_asset):
                    logging.info(f"   🎤 Muxing Audio: {os.path.basename(seg.audio_asset)}")
                    audio_input = ["-i", seg.audio_asset]
                    # Map loop video (0:v) and audio (1:a)
                    # -shortest ensures video matches audio duration if audio is shorter (or vice versa)
                    # changing to -c:a aac for compatibility
                    audio_map = ["-c:a", "aac", "-map", "0:v", "-map", "1:a", "-shortest"]

                # FFmpeg Stitching
                # -y overwrite
                # -framerate 24 (Match Blender)
                # -i raw_prefix%04d.png
                # [audio_inputs]
                # -c:v libx264 -pix_fmt yuv420p (Compatibility)
                # [audio_maps]
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate", "24",
                    "-i", f"{raw_prefix}%04d.png"
                ] + audio_input + [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p"
                ] + audio_map + [
                    "-loglevel", "error", # Quiet
                    shot_out
                ]
                
                try:
                    subprocess.run(ffmpeg_cmd, check=True)
                    if os.path.exists(shot_out):
                        rendered_files[seg.id] = shot_out
                        logging.info(f"   ✅ Stitched: {shot_out}")
                        
                        # Cleanup PNGs
                        for p in glob.glob(f"{raw_prefix}*.png"):
                            os.remove(p)
                except Exception as e:
                    logging.error(f"   ❌ FFmpeg Stitch Failed: {e}")
            else:
                 logging.warning(f"   ⚠️ No PNGs found for stitching: {first_frame}")

    # Update Manifest
    manifest.files = rendered_files
    with open(out_path, 'w') as f:
        f.write(manifest.model_dump_json(indent=2))
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lib", required=True, help="Path to library.blend")
    args = parser.parse_args()
    
    run_dispatch(args.manifest, args.out, args.lib)
