#!/usr/bin/env python3
import os
import json
import glob
import subprocess
import time
from pathlib import Path

# Config
TARGET_DIR = "/Users/0gs/METMcloud/METMroot/tools/fmv/mvp/v0.5/z_test-outputs/gahd-scripts-vids/ep_1768596120"
OUTPUT_DIR = "/Users/0gs/METMcloud/METMroot/tools/fmv/mvp/v0.5/z_test-outputs/gahd-scripts-vids"
TIMESTAMP = "1768596120"
BASE_NAME = f"gahd_episode_{TIMESTAMP}"

def get_audio_duration(path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except:
        return 2.0

def main():
    print(f"🚑 Rescuing Episode {TIMESTAMP}...")
    
    # 1. Scan Files
    wavs = sorted(glob.glob(os.path.join(TARGET_DIR, "line_*.wav")))
    pngs = sorted(glob.glob(os.path.join(TARGET_DIR, "line_*.png")))
    
    print(f"   Found {len(wavs)} wavs and {len(pngs)} pngs.")
    
    # Pair them up
    segments = []
    
    # Iterate based on index in filename
    # Assuming line_XXXX.wav
    
    count = min(len(wavs), len(pngs))
    
    history = []
    
    for i in range(count):
        wav = os.path.join(TARGET_DIR, f"line_{i:04d}.wav")
        png = os.path.join(TARGET_DIR, f"line_{i:04d}.png")
        
        if not os.path.exists(wav) or not os.path.exists(png):
            print(f"   [-] Gap at {i}, stopping.")
            break
            
        dur = get_audio_duration(wav)
        
        seg = {
            "speaker": "Rescued",
            "text": "(Rescued Audio - Text Unavailable)",
            "audio_path": wav,
            "image_path": png,
            "duration": dur
        }
        segments.append(seg)
        history.append({"speaker": "Rescued", "text": "(Rescued Audio)"})

    # 2. Export Metadata (JSON/TXT/XML)
    # JSON
    json_path = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.json")
    with open(json_path, "w") as f:
        json.dump({
            "meta": {"status": "rescued", "timestamp": TIMESTAMP},
            "history": history,
            "segments": segments
        }, f, indent=2)
    print(f"   [+] JSON: {json_path}")
    
    # TXT
    txt_path = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.txt")
    with open(txt_path, "w") as f:
        f.write("RESCUED EPISODE\n=================\n\n(Text data lost, audio preserved)\n")
    print(f"   [+] TXT: {txt_path}")
        
    # XML
    xml_path = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.xml")
    # Simple XML construction
    xml_content = f"""<XMVP version="0.5" status="rescued">
    <Manifest>
    """
    for i, s in enumerate(segments):
        xml_content += f"""    <Seg id="seg_{i:04d}">
            <Files audio="{os.path.basename(s['audio_path'])}" image="{os.path.basename(s['image_path'])}" />
            <Duration>{s['duration']}</Duration>
        </Seg>
    """
    xml_content += """    </Manifest>
</XMVP>"""
    with open(xml_path, "w") as f:
        f.write(xml_content)
    print(f"   [+] XML: {xml_path}")

    # 3. Stitch MP4
    print("   [+] Stitching MP4...")
    audio_list = os.path.join(TARGET_DIR, "rescue_audio.txt")
    video_list = os.path.join(TARGET_DIR, "rescue_video.txt")
    
    with open(audio_list, 'w') as fa, open(video_list, 'w') as fv:
        for s in segments:
            fa.write(f"file '{s['audio_path']}'\n")
            fv.write(f"file '{s['image_path']}'\n")
            fv.write(f"duration {s['duration']:.4f}\n")
            
    full_audio = os.path.join(TARGET_DIR, "rescue_full_audio.wav")
    output_mp4 = os.path.join(OUTPUT_DIR, f"{BASE_NAME}.mp4")
    
    # Concat Audio
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', audio_list, '-c', 'copy', full_audio, '-y', '-loglevel', 'error'], check=True)
    
    # Concat Video
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', video_list,
        '-i', full_audio,
        '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k', '-shortest', output_mp4
    ], check=True) # output shown for debug 
    
    print(f"   [+] MP4: {output_mp4}")

if __name__ == "__main__":
    main()
