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
import post_production 
from foley_talk import get_audio_duration
import math # For stitching

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
    
    # Global/Shared Positional Args
    import definitions
    definitions.add_global_vpform_args(parser)
    
    # Producer Args
    parser.add_argument("--seg", type=int, default=3, help="Number of segments to generate")
    parser.add_argument("--slength", type=float, default=0.0, help="Target Total Duration in Seconds")
    parser.add_argument("--l", type=float, default=8.0, help="Length of each segment in seconds")
    parser.add_argument("--vpform", type=str, default=None, help="Form/Genre (realize-ad, tech-movie)")
    parser.add_argument("--cs", type=int, default=0, help="Chaos Seeds level")

    parser.add_argument("--cf", type=str, default=None, help="Cameo Feature: Wikipedia URL or Search Query")
    parser.add_argument("--mu", type=str, default=None, help="Music Track (for music-video vpform)")
    parser.add_argument("--vm", type=str, default="L", help="Video Model Tier (L, J, K)")
    parser.add_argument("--pg", action="store_true", help="Enable PG Mode (Relaxed Celebrity/Strict Child Safety)")
    
    # Ops Args
    parser.add_argument("--xb", type=str, default="clean", help="XMVP Re-hydration path OR 'clean' (default)")
    parser.add_argument("--fast", action="store_true", help="Use Faster/Cheaper Model Tier (Overwrites --vm)") # Renamed from -f to avoid conflict
    parser.add_argument("--vfast", action="store_true", help="Use Legacy Veo 2.0 (Fastest)")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    parser.add_argument("--local", action="store_true", help="Run Locally (Gemma + LTX-Video)")
    parser.add_argument("--cloud", action="store_true", help="Force Cloud Mode (Gemini + Veo). Overrides --local.")
    
    # Clip Video Arg
    parser.add_argument("--f", type=str, help="Source Folder for Clip Video Mode")
    parser.add_argument("--res", type=str, default="720p", help="Resolution for Local Video (720p, 480p, 360p)")
    
    parser.add_argument("--retcon", action="store_true", help="Force Text-Only Expansion (Implies --local, Skips Video)")
    parser.add_argument("--prompt", type=str, help="Alias for concept (the prompt)")
    
    args, unknown = parser.parse_known_args()
    
    # Retcon Force Logic
    if args.retcon:
        logging.info("🔄 Retcon Mode Enabled: Forcing Local Mode + Text Only.")
        args.local = True

    # --- Smart Argument Resolution via Definitions ---
    # 1. Resolve VPForm from cli_args
    resolved_form = definitions.parse_global_vpform(args, current_default=args.vpform)
    args.vpform = resolved_form
    
    # Validation/Fallback
    if args.vpform is None:
        args.vpform = "movies-movie" # Default to standard movie trailer
        logging.info("ℹ️  No VPForm specified. Defaulting to 'movies-movie'.")
    
    # 2. Resolve Concept from leftover args
    # Concept is the first positional arg that is NOT the resolved form alias and NOT 'run'
    concept = None
    if args.cli_args:
        for val in args.cli_args:
            val_lower = val.lower()
            if val_lower == "run": continue
            
            # Check if this val IS the form alias
            form_match = definitions.resolve_vpform(val)
            if form_match and form_match.key == resolved_form:
                continue # Consumed as VPForm
            
            # If not consumed, it's the concept
            concept = val
            break # Take first non-form arg as concept
            
    args.concept = concept
    
    # Alias Support: If --prompt provided but no positional concept, use prompt
    if args.prompt and not args.concept:
        args.concept = args.prompt

    # Auto-Carbonation (Sassprilla)
    if args.concept:
        p_clean = args.concept.strip()
        if p_clean.istitle() and "." not in p_clean and len(p_clean) < 80:
            logging.info(f"🫧 Auto-Carbonating Title Prompt: '{p_clean}'...")
            try:
                 import sassprilla_carbonator
                 expanded = sassprilla_carbonator.carbonate_prompt(p_clean)
                 if expanded:
                     logging.info(f"   ✨ Expanded to {len(expanded)} chars.")
                     args.concept = expanded
            except Exception as e:
                 logging.warning(f"   ⚠️ Carbonation failed: {e}")

    # Default Override for Draft Animatic (10 mins default)
    if args.vpform == "draft-animatic" and args.seg == 3:
        logging.info("📜 Draft Animatic: Defaulting to 10 minutes (75 segments @ 8s).")
        args.seg = 75

    # Fast Mode Override
    if args.fast:
        logging.info("🏎️ Fast Mode Enabled: Switching to Tier J.")
        args.vm = "J"
        
    # V-Fast Mode Override (Legacy)
    if args.vfast:
        logging.info("🦕 V-Fast Mode Enabled: Switching to Tier V2 (Veo 2.0).")
        args.vm = "V2"

    # Cloud Override (Highest Priority)
    if args.cloud:
        logging.info("☁️  Cloud Mode Forced via --cloud.")
        args.local = False
        os.environ["TEXT_ENGINE"] = "gemini_cloud"
        # Ensure we don't accidentally auto-detect local later
    
    # Local Mode Override
    # Auto-Detect Local Preference from Active Profile (Only if not cloud and not already local)
    if not args.local and not args.cloud:
        try:
             import definitions
             # Force reload to get latest disk state
             definitions.load_active_profile()
             active_vid = definitions.ACTIVE_PROFILE.get(definitions.Modality.VIDEO)
             
             # Check if active video model is a known local backend config
             # We can check definitions registry
             if active_vid in definitions.MODAL_REGISTRY[definitions.Modality.VIDEO]:
                 conf = definitions.MODAL_REGISTRY[definitions.Modality.VIDEO][active_vid]
                 if conf.backend == definitions.BackendType.LOCAL:
                     logging.info(f"🏠 active_models.json requests Local Video ({active_vid}). Auto-enabling --local.")
                     args.local = True
        except Exception as e:
             logging.warning(f"Failed to check active profile for local pref: {e}")

    if args.local:
        if args.vpform == "tech-movie": # If still default
            logging.info("🏠 Local Mode: Defaulting vpform to 'music-video'")
            args.vpform = "music-video"
            
        if args.vpform == "full-movie":
             pass
        if args.vpform == "clip-video":
             # Special Mode: Clip Video (Montage)
             logging.info("🎬 Mode: Clip Video Montage (Source Folder -> Audio Sync)")
             # Verify Source Folder
             if not args.f: # Use -f for folder as defined in CLI
                 # CLI parser for movie_producer.py doesn't have explicit -f arg yet?
                 # Wait, line 62 in movie_producer defines -f as --fast. 
                 # User requested -f for folder?
                 # Argument conflict!
                 # User said: --f '/Volumes...'
                 # Parser says: parser.add_argument("-f", "--fast", ...)
                 # We need to change the Fast arg or add a new one.
                 # Let's check line 62.
                 pass

    # --- APPLY FORM DEFAULTS (Late Binding via Definitions) ---
    try:
        import definitions
        if args.vpform in definitions.FORM_REGISTRY:
            form_conf = definitions.FORM_REGISTRY[args.vpform]
            
            # Apply VM default if currently at global default "L"
            if "vm" in form_conf.default_args:
                target_vm = form_conf.default_args["vm"]
                if args.vm == "L": # Global Default
                    logging.info(f"✨ Auto-Switching VM to '{target_vm}' (VPForm Default)")
                    args.vm = target_vm
    except Exception as e:
        logging.warning(f"⚠️ Failed to apply form defaults: {e}")

    # Local Mode Configuration
    if args.local:
        logging.info("🏠 Local Mode Enabled: Switching models to Local Gemma (Text) and LTX (Video).")

        # Explicitly Enforce Local Text Engine
        os.environ["TEXT_ENGINE"] = "local_gemma"
        
        # ⚠️ CRITICAL: Reset TextEngine Singleton
        try:
             import text_engine
             text_engine._ENGINE = None
             logging.info("   🔄 TextEngine Singleton Reset for Local Mode.")
        except:
             pass

        # Resolve Local Model Path using Definitions (Global for ALL local modes)
        try:
            import definitions
            # Assuming 'gemma-2-9b-it-director' (GemmaW) is the target local model with adapter
            gemma_config = definitions.MODAL_REGISTRY[definitions.Modality.TEXT].get("gemma-2-9b-it-director")
            if not gemma_config: # Fallback to base
                 gemma_config = definitions.MODAL_REGISTRY[definitions.Modality.TEXT].get("gemma-2-9b-it")
            
            if gemma_config and gemma_config.path:
                 os.environ["LOCAL_MODEL_PATH"] = gemma_config.path
                 logging.info(f"   📍 Local Text Model Path: {gemma_config.path}")
                 
                 if gemma_config.adapter_path:
                      os.environ["LOCAL_ADAPTER_PATH"] = gemma_config.adapter_path
                      logging.info(f"   🧩 Local Adapter Path: {gemma_config.adapter_path}")
            else:
                 logging.warning("   ⚠️ Local Gemma path not found in definitions. Using default.")
        except ImportError:
             pass
        
    # Cloud Movie Overrides (Veo Constraints)
    if args.vpform in ["movies-movie", "parody-movie", "parody-video"]:
        if not args.local:
             logging.info(f"🌩️ Cloud Movie Mode ({args.vpform}): Enforcing Veo Constraints.")
             if args.l != 8.0:
                 logging.warning(f"   ⚠️ Overriding segment length {args.l}s -> 8.0s (Veo Requirement)")
                 args.l = 8.0
        else:
             # LOCAL MODE: Relax constraints
             # If using parody-video locally (Wan/LTX), we prefer variable pacing ~4s.
             # The default in parser is 8.0, so checking against that.
             if args.vpform == "parody-video" and args.l == 8.0:
                 logging.info(f"   🏠 Local Parody: Switching from strict 8s to flexible 4s pacing.")
                 args.l = 4.0
             
             # Default to K (Veo 3.1) unless Fast is specified (Legacy logic removed to allow L tier)
             # if not args.fast and not args.vfast and args.vm == "K": 
             #     pass
             # elif args.vm != "K" and not args.fast and not args.vfast:
             #     logging.info(f"   🎥 Switching Video Model to 'K' (Veo 3.1) for best results.")
             #     args.vm = "K"
        # Use Local Gemma (Director Adapter) for Hollywood Accuracy
        os.environ["TEXT_ENGINE"] = "local_gemma"
        
        # Resolve Local Model Path using Definitions (Same as Local Mode)
        try:
            import definitions
            gemma_config = definitions.MODAL_REGISTRY[definitions.Modality.TEXT].get("gemma-2-9b-it-director")
            if not gemma_config: 
                 gemma_config = definitions.MODAL_REGISTRY[definitions.Modality.TEXT].get("gemma-2-9b-it")
            
            if gemma_config and gemma_config.path:
                 os.environ["LOCAL_MODEL_PATH"] = gemma_config.path
                 logging.info(f"   📍 Local Text Model Path: {gemma_config.path}")
                 
                 if gemma_config.adapter_path:
                      os.environ["LOCAL_ADAPTER_PATH"] = gemma_config.adapter_path
                      logging.info(f"   🧩 Local Adapter Path: {gemma_config.adapter_path}")
        except ImportError:
             pass
        
        # Log Safety/Quality Status
        safety_status = "ON (Reasonable)" if args.pg else "OFF (Uncensored)"
        logging.info(f"   🛡️ Safety Filters: {safety_status}")
        logging.info(f"   ✨ Quality Refinement: ON (Hyper-Detailed Fattening)")
        
        import json
        am_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_models.json")
        try:
            with open(am_path, "w") as f:
                json.dump({
                    "text": "gemma-2-9b-it",
                    "image": "flux-klein",
                    "video": "ltx-video",
                    "spoken_tts": "kokoro-v1"
                }, f, indent=2)
            logging.info("   ✅ Active Models Updated.")
        except Exception as e:
            logging.error(f"   ❌ Failed to update active_models.json: {e}")
            sys.exit(1)

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
    is_clean_run = (args.xb == "clean")
    
    if not args.concept and is_clean_run and args.vpform != "clip-video":
        logging.error("Please provide a concept string OR an --xb path.")
        sys.exit(1)
        
    if is_clean_run:
        clean_artifacts(OUT_DIR)
        
    ts = int(time.time())
    logging.info("🎬 MOVIE PRODUCER 1.1: Spinning up the Modular Vision Pipeline...")
    
    # === SPECIAL ROUTING: CLIP VIDEO ===
    if args.vpform == "clip-video":
        import dispatch_clip_video
        success = dispatch_clip_video.run_clip_video_pipeline(args)
        if success:
            logging.info("✅ Clip Video Pipeline Complete.")
            sys.exit(0)
        else:
            logging.error("❌ Clip Video Pipeline Failed.")
            sys.exit(1)
    # ===================================

    # Auto-Carbonation for Short Titles (The "Sassprilla" Hook)
    # DISABLE for Cloud Movies (they are exact remakes)
    if args.vpform in ["movies-movie", "parody-movie", "parody-video"]:
        logging.info("🚫 Cloud Movie Mode: Auto-Carbonation Disabled (Preserving Exact Title).")
    elif args.concept and len(args.concept.split()) < 10 and "." not in args.concept and is_clean_run:
        logging.info(f"🫧 Auto-Carbonating detected Title: '{args.concept}'")
        try:
             import sassprilla_carbonator
             # Pass vpform as context (e.g. music-video implies certain lyrics/vibe)
             expanded = sassprilla_carbonator.carbonate_prompt(
                 args.concept, 
                 artist=None, # inferred
                 extra_context=args.vpform
             )
             if expanded:
                 logging.info(f"✨ Carbonated Prompt Injected ({len(expanded)} chars)")
                 # We replace the concept with the expanded version
                 args.concept = expanded
             else:
                 logging.warning("Carbonator returned empty.")
        except ImportError:
             logging.warning("sassprilla_carbonator module not found.")
        except Exception as e:
             logging.warning(f"Carbonation failed (continuing with raw prompt): {e}")

    # Define paths
    p_bible = os.path.join(OUT_DIR, "bible.json")
    p_story = os.path.join(OUT_DIR, "story.json")
    p_portions = os.path.join(OUT_DIR, "portions.json")
    p_manifest = os.path.join(OUT_DIR, "manifest.json")
    p_manifest_updated = os.path.join(OUT_DIR, "manifest_updated.json")

    # 1. Vision Producer (The Showrunner)
    if args.xb and args.xb != "clean":
        logging.info(f"📚 Re-hydrating form XMVP: {args.xb}")
        from mvp_shared import load_xmvp
        bible_content = load_xmvp(args.xb, "Bible")
        if not bible_content:
            logging.error("Could not load <Bible> from XMVP.")
            sys.exit(1)
            
        with open(p_bible, "w") as f:
            f.write(bible_content)
        logging.info("   -> Skipped Vision Producer (Loaded from XML).")

        # DURATION RETCON LOGIC
        # If the user asks for a new --slength while using --xb, we must:
        # 1. Update the Bible with the new constraints.
        # 2. Force Writers Room to re-run (by nuking downstream JSONs).
        if args.slength and args.slength > 0:
            logging.info(f"⏱️  Duration Retcon Detected: New Target {args.slength}s (Old XML ignored)")
            
            # Read loaded bible
            import json
            with open(p_bible, 'r') as f:
                bible_data = json.load(f)
            
            # Update Constraints
            if "constraints" in bible_data:
                bible_data["constraints"]["max_duration_sec"] = args.slength
                # Recalculate segments just for safety
                bible_data["constraints"]["max_segments"] = int(args.slength / args.l)
            
            # Save updated Bible
            with open(p_bible, 'w') as f:
                json.dump(bible_data, f, indent=2)
            logging.info("   ✅ Bible constraints updated.")
            
            # Invalidate downstream artifacts to force re-generation
            # We keep 'story.json' (The Plot) but nuke 'portions.json' (The Scenes)
            if os.path.exists(p_portions):
                logging.info("   💥 Invalidating old Portions (forcing writers room re-run)...")
                os.remove(p_portions)
            if os.path.exists(p_manifest):
                os.remove(p_manifest)
                
    else:

        # Logic: slength > 0 overrides seg
        if args.slength > 0:
            logging.info(f"⏱️  Target Duration (SLENGTH): {args.slength}s")
            args.seg = math.ceil(args.slength / args.l)
            total_length = args.slength
        else:
            # Calculate Total Length
            # Calculate Total Length
            # Generalize: If Audio is provided, it always drives duration (overriding --seg)
            if args.mu and os.path.exists(args.mu):
                audio_len = get_audio_duration(args.mu)
                if audio_len > 0:
                    logging.info(f"🎵 Audio Driven Duration: {audio_len:.1f}s")
                    args.seg = math.ceil(audio_len / args.l)
                    total_length = audio_len # Use exact audio length
                else:
                    total_length = args.seg * args.l
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
    if not success: sys.exit(1)
    logging.info(f"✅ Manifest ready: {p_manifest}")

    # 4.1 CHECKPOINT SAVE (The Safety Net)
    # Save partial XMVP now in case Video Generation crashes
    from mvp_shared import safe_save_xmvp
    
    meta_data = {
        "concept": args.concept, 
        "slength": args.slength, 
        "vpform": args.vpform,
        "local_mode": args.local
    }
    
    # Save to "SESSION_CHECKPOINT.xml"
    chk_path = os.path.join(OUT_DIR, "SESSION_CHECKPOINT.xml")
    safe_save_xmvp(chk_path, p_bible, p_story, p_manifest, extra_meta=meta_data)
    logging.info(f"💾 Checkpoint Saved: {chk_path}")

    # 4.5 MEMORY CLEANUP (Drop the Mic)
    # ... (existing cleanup code) ...
    if args.local:
         try:
             import text_engine
             logging.info("📉 Memory Optimization: Unloading Text Engine before Video Dispatch...")
             text_engine.get_engine().unload()
         except Exception as e:
             logging.warning(f"   ⚠️ Failed to unload Text Engine: {e}")

    # 5. Dispatch Director (The Director)
    if args.retcon:
        logging.info("🛑 Retcon Mode: Stopping before Video Dispatch (Text-Only Complete).")
        # Ensure we set success=True to proceed to cleanup/save if needed, though usually save happens after dispatch.
        # Actually, lines 456+ do final save. We want to skip dispatch but do final save.
        success = True
    elif args.vpform in ["draft-animatic", "music-video"]:
        # FLUX ANIMATIC ENGINE
        logging.info(f"🎬 Mode: {args.vpform} (Flux Animatic Engine)")
        import dispatch_animatic
        # Resolve Flux Path
        import definitions
        flux_conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-klein")
        flux_path = flux_conf.path if flux_conf else "/Volumes/XMVPX/mw/flux-root"
        
        success = dispatch_animatic.run_animatic(
            manifest_path=p_manifest,
            out_path=p_manifest_updated,
            staging_dir=DIR_PARTS,
            flux_path=flux_path
        )
        if args.local:
             logging.info(f"🎬 Mode: {args.vpform} (Local LTX-First - Video)")
        
        # Default Director (Handles LTX or Cloud Veo based on args.local)
        import dispatch_director
        success = dispatch_director.run_dispatch(
            manifest_path=p_manifest,
            mode="video",
            model_tier=args.vm, 
            out_path=p_manifest_updated,
            staging_dir=DIR_PARTS,
            pg_mode=args.pg,
            local_mode=args.local
        )
    
    if not success:
        # SALVAGE LOGIC: Check if we have enough clips to "rescue" the job
        logging.warning("⚠️ Producer reported failure. Checking for salvageable content...")
        
        # Reload Manifest to check actual file count
        files_present = 0
        total_segs = args.seg
        
        # Try to find refined count from updated manifest
        if os.path.exists(p_manifest_updated):
             try:
                 import mvp_shared
                 m_check = mvp_shared.load_manifest(p_manifest_updated)
                 if m_check.files: files_present = len(m_check.files)
                 if m_check.segs: total_segs = len(m_check.segs)
             except: pass
        
        # Fallback to direct file count if manifest invalid
        if files_present == 0 and os.path.exists(DIR_PARTS):
             files_present = len([f for f in os.listdir(DIR_PARTS) if f.endswith(".mp4")])
             
        missing_count = total_segs - files_present
        pct_missing = missing_count / total_segs if total_segs > 0 else 1.0
        
        if pct_missing < 0.25:
             logging.info(f"🚑 SALVAGE MODE ACTIVATED: Only {pct_missing*100:.1f}% clips missing (<25%).")
             logging.info("   -> Ignoring failure. Proceeding to Stitch & Stretch.")
        else:
             logging.error(f"❌ Critical Failure: {pct_missing*100:.1f}% missing ({files_present}/{total_segs}). Cannot salvage.")
             sys.exit(1)

    # 6. Post-Production (The Editor)
    from mvp_shared import load_manifest, safe_save_xmvp
    # Use the UPDATED manifest (with file paths) if available, otherwise original (Retcon mode)
    final_manifest_path = p_manifest_updated if os.path.exists(p_manifest_updated) else p_manifest
    manifest = load_manifest(final_manifest_path)
    
    # FINAL SAVE (The Golden Master)
    
    final_xmvp_path = os.path.join(DIR_FINAL, f"MVP_SESSION_{ts}.xml")
    safe_save_xmvp(final_xmvp_path, p_bible, p_story, final_manifest_path, extra_meta=meta_data)
    logging.info(f"🏆 Final XMVP Saved: {final_xmvp_path}")

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
        
        # --- STRETCH LOGIC (Salvage/Sync) ---
        # If we have a target length (Audio or explicit slength) and the stitched video 
        # is significantly shorter (due to missing clips), we stretch it.
        target_duration = 0
        if args.mu and os.path.exists(args.mu):
             target_duration = get_audio_duration(args.mu)
        elif args.slength > 0:
             target_duration = args.slength
             
        if target_duration > 0 and os.path.exists(final_filename):
             current_duration = get_audio_duration(final_filename) # Reuse function for video
             if current_duration > 0:
                 diff = target_duration - current_duration
                 # If missing clips, current < target. 
                 # If diff is significant (e.g. > 2 seconds or salvage triggered)
                 # User Rule: "fewer than 25% missing"... we already checked that to get here.
                 # If we are here and we missed clips, duration WILL be short.
                 if diff > 2.0:
                      logging.info(f"⏱️ Duration Mismatch (Target: {target_duration}s, Actual: {current_duration}s).")
                      logging.info("   🤸 Applying Time-Stretch (Changing Frame Rate) to match song...")
                      
                      # Rename check
                      shutil.move(final_filename, final_filename.replace(".mp4", "_raw.mp4"))
                      raw_input = final_filename.replace(".mp4", "_raw.mp4")
                      
                      # Calculate slowdown factor. 
                      # We want Current to become Target.
                      # PTS * (Target / Current)
                      factor = target_duration / current_duration
                      
                      # Use setpts to re-time. Keep audio from music track later.
                      # We just stretch video stream.
                      import subprocess
                      
                      # NOTE: We force standard 24fps output to ensure compatibility, 
                      # letting ffmpeg duplicate frames as needed to fill the time.
                      cmd_stretch = [
                          "ffmpeg", "-y",
                          "-i", raw_input,
                          "-filter:v", f"setpts={factor:.4f}*PTS",
                          "-c:a", "copy", # No audio usually in raw stitch
                          final_filename
                      ]
                      try:
                          subprocess.run(cmd_stretch, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                          logging.info(f"   ✅ Stretched Video Saved: {final_filename}")
                      except Exception as e:
                          logging.error(f"   ❌ Stretch failed: {e}. Reverting.")
                          shutil.copy(raw_input, final_filename)
        
        # 6.2 Draft Animatic Audio (The Whopper Integration)
        if args.vpform == "draft-animatic" and os.path.exists(final_filename):
            logging.info("🔊 Draft Animatic: Engaging Audio Pipeline (Draft Mix)...")
            draft_audio_filename = os.path.join(DIR_FINAL, f"MVP_DRAFT_AUDIO_{ts}.mp4")
            try:
                # Call foley_talk.py via subprocess to keep environment clean
                cmd_audio = [
                    sys.executable, "foley_talk.py",
                    "--input", final_filename,
                    "--xb", p_manifest_updated,
                    "--out", draft_audio_filename,
                    "--mode", "draft-mix"
                ]
                subprocess.run(cmd_audio, check=True)
                
                if os.path.exists(draft_audio_filename):
                    logging.info(f"✅ DRAFT AUDIO CUT: {draft_audio_filename}")
                    final_filename = draft_audio_filename # Promote to final
            except Exception as e:
                logging.error(f"❌ Failed to run Draft Audio Pipeline: {e}")

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

    # 7. XMVP Archival (Redundant but keeps legacy file structure if needed)
    # Actually, let's just rely on the GOLDEN MASTER above.
    logging.info("💾 (Legacy Archive step skipped, rely on Final XMVP above)")
    # We remove the failing save_xmvp call entirely since safe_save_xmvp already did it.
    
    # 8. Cleanup
    logging.info("🛡️ Component Cleanup Disabled (Preserving 'componentparts' for safety).")
    # if os.path.exists(DIR_PARTS):
    #    shutil.rmtree(DIR_PARTS)

if __name__ == "__main__":
    main()
