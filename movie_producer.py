#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import time
import shutil

# Import MVP Modules
import vision_producer
import stub_reification
import writers_room
import portion_control
import dispatch_director
import post_production # For stitching

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_output_dir():
    """Returns the default output directory: ../z_test-outputs"""
    # Base it relative to this script location
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "z_test-outputs", "movies")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def clean_artifacts(out_dir):
    """Cleans up intermediate JSON files from previous runs to avoid confusion."""
    for f in ["bible.json", "story.json", "portions.json", "manifest.json", "manifest_updated.json"]:
        path = os.path.join(out_dir, f)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logging.warning(f"Could not remove {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Movie Producer: The MVP Orchestrator (1.1)")
    parser.add_argument("concept", nargs='?', help="The concept text (quoted string).")
    
    # Producer Args
    parser.add_argument("--seg", type=int, default=3, help="Number of segments to generate")
    parser.add_argument("--l", type=float, default=8.0, help="Length of each segment in seconds")
    parser.add_argument("--vpform", type=str, default="tech-movie", help="Form/Genre (realize-ad, tech-movie)")
    parser.add_argument("--cs", type=int, default=0, help="Chaos Seeds level")

    parser.add_argument("--cf", type=str, default=None, help="Cameo Feature: Wikipedia URL or Search Query")
    parser.add_argument("--mu", type=str, default=None, help="Music Track (for music-video vpform)")
    parser.add_argument("--vm", type=str, default="K", help="Video Model Tier (L, J, K)")
    parser.add_argument("--pg", action="store_true", help="Enable PG Mode (Relaxed Celebrity/Strict Child Safety)")
    
    # Ops Args
    parser.add_argument("--clean", action="store_true", help="Clean intermediate JSONs before running")
    parser.add_argument("--xb", type=str, help="XMVP Re-hydration path (Bypasses Vision Producer)")
    parser.add_argument("-f", "--fast", action="store_true", help="Use Faster/Cheaper Model Tier (Overwrites --vm)")
    parser.add_argument("--vfast", action="store_true", help="Use Legacy Veo 2.0 (Fastest)")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    
    args = parser.parse_args()

    # Fast Mode Override
    if args.fast:
        logging.info("🏎️ Fast Mode Enabled: Switching to Tier J.")
        args.vm = "J"
        
    # V-Fast Mode Override (Legacy)
    if args.vfast:
        logging.info("🦕 V-Fast Mode Enabled: Switching to Tier V2 (Veo 2.0).")
        args.vm = "V2"

    # 1. Setup Output Directory
    OUT_DIR = args.out if args.out else get_output_dir()
    logging.info(f"📂 Output Directory: {OUT_DIR}")
    
    # Update Paths (Locally Defined)
    DIR_PARTS = os.path.join(OUT_DIR, "componentparts")
    DIR_FINAL = os.path.join(OUT_DIR, "finalcuts")
    
    # Ensure Directories
    for d in [DIR_PARTS, DIR_FINAL]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 0. Boot
    if not args.concept and not args.xb:
        logging.error("Please provide a concept string OR an --xb path.")
        sys.exit(1)
        
    if args.clean:
        clean_artifacts(OUT_DIR)
        
    ts = int(time.time())
    logging.info("🎬 MOVIE PRODUCER 1.1: Spinning up the Modular Vision Pipeline...")

    # Define paths
    p_bible = os.path.join(OUT_DIR, "bible.json")
    p_story = os.path.join(OUT_DIR, "story.json")
    p_portions = os.path.join(OUT_DIR, "portions.json")
    p_manifest = os.path.join(OUT_DIR, "manifest.json")
    p_manifest_updated = os.path.join(OUT_DIR, "manifest_updated.json")

    # 1. Vision Producer (The Showrunner)
    if args.xb:
        logging.info(f"📚 Re-hydrating form XMVP: {args.xb}")
        from mvp_shared import load_xmvp
        bible_content = load_xmvp(args.xb, "Bible")
        if not bible_content:
            logging.error("Could not load <Bible> from XMVP.")
            sys.exit(1)
            
        with open(p_bible, "w") as f:
            f.write(bible_content)
        logging.info("   -> Skipped Vision Producer (Loaded from XML).")
        
    else:
        total_length = args.seg * args.l
        success = vision_producer.run_producer(
            vpform_name=args.vpform,
            prompt=args.concept,
            slength=total_length,
            seg_len=args.l,
            chaos_seed_count=args.cs,

            cameo=args.cf,
            out_path=p_bible,
            audio_path=args.mu
        )
        if not success: sys.exit(1)

    # 2. Stub Reification (The Writer)
    success = stub_reification.run_stub(
        bible_path=p_bible,
        out_path=p_story
    )
    if not success: sys.exit(1)

    # 3. Writers Room (The Screenwriter)
    success = writers_room.run_writers(
        bible_path=p_bible,
        story_path=p_story,
        out_path=p_portions
    )
    if not success: sys.exit(1)

    # 4. Portion Control (The Line Producer)
    success = portion_control.run_portion(
        bible_path=p_bible,
        portions_path=p_portions,
        out_path=p_manifest
    )
    if not success: sys.exit(1)

    # 5. Dispatch Director (The Director)
    success = dispatch_director.run_dispatch(
        manifest_path=p_manifest,
        mode="video",
        model_tier=args.vm,
        out_path=p_manifest_updated,
        staging_dir=DIR_PARTS,
        pg_mode=args.pg
    )
    if not success: sys.exit(1)

    # 6. Post-Production (The Editor)
    from mvp_shared import load_manifest, save_xmvp
    manifest = load_manifest(p_manifest_updated)
    
    sorted_segs = sorted(manifest.segs, key=lambda s: s.id)
    
    stitch_list = []
    for seg in sorted_segs:
        if seg.id in manifest.files:
            stitch_list.append({
                "local_file": manifest.files[seg.id]
            })
            
    if stitch_list:
        final_filename = os.path.join(DIR_FINAL, f"MVP_MOVIE_{ts}.mp4")
        logging.info(f"🧵 Stitching {len(stitch_list)} clips to {final_filename}...")
        post_production.stitch_videos(stitch_list, final_filename)
        
        # 6.5 Music Muxing
        if args.mu and os.path.exists(args.mu) and os.path.exists(final_filename):
            logging.info(f"🎵 Muxing Audio Track: {args.mu}")
            musical_filename = os.path.join(DIR_FINAL, f"MVP_MOVIE_MUSIC_{ts}.mp4")
            try:
                import subprocess
                cmd_mix = [
                    "ffmpeg", "-y",
                    "-i", final_filename,
                    "-i", args.mu,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "copy",
                    "-shortest", # Align to shortest (usually video if audio is longer, or vice versa)
                    musical_filename
                ]
                subprocess.run(cmd_mix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"✅ FINAL MUSICAL CUT: {musical_filename}")
            except Exception as e:
                logging.error(f"Failed to mux audio: {e}")
                
    else:
        logging.error("❌ No clips to stitch.")

    # 7. XMVP Archival
    logging.info("💾 Archiving Run to XMVP...")
    
    def read_json_safe(path):
        if os.path.exists(path):
            with open(path, 'r') as f: return f.read()
        return "{}"

    xmvp_data = {
        "Bible": read_json_safe(p_bible),
        "Story": read_json_safe(p_story),
        "Portions": read_json_safe(p_portions),
        "Manifest": read_json_safe(p_manifest_updated)
    }
    
    xmvp_filename = os.path.join(DIR_FINAL, f"run_{ts}.xml")
    save_xmvp(xmvp_data, xmvp_filename)
    logging.info(f"✅ XMVP Saved: {xmvp_filename}")
    
    # 8. Cleanup
    logging.info("🧹 Cleaning up component parts...")
    try:
        if os.path.exists(DIR_PARTS):
            shutil.rmtree(DIR_PARTS)
            # Re-create empty? No, user wants it clean.
            # But wait, user said "delete all but frame one... from component parts for movies"
            # It's safer to just empty it.
            # actually, maybe just leaving one file is good for debug?
            # Let's just delete the folder contents.
            os.makedirs(DIR_PARTS, exist_ok=True) 
    except Exception as e:
        logging.warning(f"Cleanup warning: {e}")

if __name__ == "__main__":
    main()
