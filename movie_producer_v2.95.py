#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import time
import shutil
import subprocess

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
    parser.add_argument("--local", action="store_true", default=True, help="Run Locally (Gemma + LTX-Video). Default=True.")
    parser.add_argument("--cloud", action="store_true", help="Force Cloud Mode (Gemini + Veo). Overrides --local.")
    
    # Clip Video Arg
    parser.add_argument("--f", type=str, help="Source Folder for Clip Video Mode")
    parser.add_argument("--res", type=str, default=None, help="Resolution for Local/Cloud Video (720p, 360p, WxH)")
    
    parser.add_argument("--nico", type=str, default="off", choices=["on", "off"], help="Enable Nicotime Indexing (Default: off)")

    parser.add_argument("--retcon", action="store_true", help="Force Text-Only Expansion (Implies --local, Skips Video)")
    parser.add_argument("--reshoot", action="store_true", help="Force Re-shoot of Video (Ignore existing clips)")
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
    # Cloud Override (Highest Priority)
    if args.cloud:
        logging.info("☁️  Cloud Mode Forced via --cloud (Video Only).")
        args.local = False
        # Do NOT force gemini_cloud for text. User prefers local_gemma even for Cloud Video.
        # os.environ["TEXT_ENGINE"] = "gemini_cloud" 
        
        # Enforce Local Gemma by default if not set?
        if "TEXT_ENGINE" not in os.environ:
             os.environ["TEXT_ENGINE"] = "local_gemma"
             logging.info("   📝 Text Engine: Defaulting to Local Gemma (Hybrid Mode).")
    
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
             logging.info(f"🌩️ Cloud Movie Mode ({args.vpform}): Using LTX Cloud API.")
             # No 8s limit anymore!
             pass
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

    # --- AUDIO DRIVEN DURATION (Global Priority) ---
    # Determine target length from Audio if provided, BEFORE deciding to skip generation.
    audio_driven_length = 0.0
    if args.mu and os.path.exists(args.mu):
        logging.info(f"   🎵 Probing Audio Duration for: {args.mu}")
        audio_len = get_audio_duration(args.mu)
        
        # Fallback to librosa/basic if foley_talk failed
        if audio_len == 0.0:
             try:
                 import librosa
                 dur = librosa.get_duration(filename=args.mu)
                 if dur > 0: 
                     audio_len = dur
                     logging.info(f"   🎵 Librosa Duration Found: {audio_len:.2f}s")
             except: pass
        
        if audio_len > 0:
            logging.info(f"🎵 Audio Detected: {audio_len:.1f}s")
            audio_driven_length = audio_len
            
            # Update args.slength if not manually set (or override?)
            # Usually Audio is strict master.
            if args.slength == 0:
                logging.info(f"   ✨ Auto-setting --slength to match Audio: {audio_len:.1f}s")
                args.slength = audio_len
                
            # Update segment count assumption
            args.seg = math.ceil(audio_len / args.l)
        else:
            logging.warning("⚠️ Audio provided but duration could not be determined (0s).")

    # 1. Vision Producer (The Showrunner)
    if args.xb and args.xb != "clean":
        logging.info(f"📚 Re-hydrating form XMVP: {args.xb}")
        from mvp_shared import load_xmvp, load_cssv, CSSV
        
        # CLEANUP: Ensure we don't mix stale files with new XML data
        # This prevents the "waffle script" bug where stale manifest overrides new XML data
        clean_artifacts(OUT_DIR)
        
        # Full Rehydration
        try:
             bible_content = load_xmvp(args.xb, "Bible")
             story_content = load_xmvp(args.xb, "Story")
             manifest_content = load_xmvp(args.xb, "Manifest")
             portions_content = load_xmvp(args.xb, "Portions")
             
             if bible_content:
                  with open(p_bible, "w") as f: f.write(bible_content)
             if story_content:
                  with open(p_story, "w") as f: f.write(story_content)
             
             # Prioritize Manifest, but accept Portions if Manifest missing
             if manifest_content:
                  with open(p_manifest, "w") as f: f.write(manifest_content)
                  logging.info("   -> Skipped Vision Producer (Loaded Manifest from XML).")
             elif portions_content:
                  with open(p_portions, "w") as f: f.write(portions_content)
                  logging.info("   -> Loaded Portions from XML (Manifest will be generated).")
             
             logging.info("   📚 Rehydrated Bible, Story, and data from XML.")
        except Exception as e:
             logging.warning(f"   ⚠️ Rehydration Error: {e}")

        # DURATION RETCON LOGIC
        if args.slength and args.slength > 0:
            logging.info(f"⏱️  Duration Retcon Check: Target {args.slength}s")
            
            # Read loaded bible
            import json
            try:
                with open(p_bible, 'r') as f:
                    bible_data = json.load(f)
                
                old_max = 0
                if "constraints" in bible_data:
                     old_max = bible_data["constraints"].get("max_duration_sec", 0)
                
                # Tolerance check (e.g. 1 second diff)
                if abs(old_max - args.slength) > 2.0:
                    logging.info(f"   ♻️  New Duration ({args.slength}s) != old XML ({old_max}s). Triggering RE-PLAN.")
                    
                    # Update Constraints
                    if "constraints" in bible_data:
                        bible_data["constraints"]["max_duration_sec"] = args.slength
                        bible_data["constraints"]["max_segments"] = int(args.slength / args.l)
                    
                    # Save updated Bible
                    with open(p_bible, 'w') as f:
                        json.dump(bible_data, f, indent=2)
                    logging.info("   ✅ Bible constraints updated.")
                    
                    if os.path.exists(p_portions):
                        logging.info("   ♻️  Preserving existing Portions (Dialogue) for Re-Cut...")
                        # os.remove(p_portions) # DON'T DELETE DIALOGUE!
                    if os.path.exists(p_manifest):
                        os.remove(p_manifest)
                        # Remove manifest content variable to force downstream checks
                        manifest_content = None 
                else:
                    logging.info("   ✅ Duration matches XML. No Retcon needed.")
            except Exception as e:
                logging.warning(f"   ⚠️ Failed to parse Bible for Retcon: {e}")
                
    else:

        # Logic: slength > 0 overrides seg
        if args.slength > 0:
            logging.info(f"⏱️  Target Duration (SLENGTH): {args.slength}s")
            args.seg = math.ceil(args.slength / args.l)
            total_length = args.slength
        else:
             # Calculate Total Length based on Segs
             total_length = args.seg * args.l
             logging.info(f"⏱️  Manual Duration: {total_length}s ({args.seg} segs * {args.l}s)")

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

    # PIPELINE EXECUTION (Skip if Hydrated)
    if os.path.exists(p_manifest) and os.path.getsize(p_manifest) > 10:
        logging.info("⏩ Manifest present. Skipping Generation Pipeline (Writer/Director).")
    else:
        # 2. Stub Reification (The Writer)
        success = stub_reification.run_stub(
            bible_path=p_bible,
            out_path=p_story
        )
        if not success: sys.exit(1)
    
        # 3. Writers Room (The Screenwriter)
        if os.path.exists(p_portions) and os.path.getsize(p_portions) > 10:
             logging.info("⏩ Portions present (from XML). Skipping Writers Room to preserve script.")
        else:
            success = writers_room.run_writers(
                bible_path=p_bible,
                story_path=p_story,
                out_path=p_portions
            )
            if not success: sys.exit(1)
    
        # 4. Portion Control (The Line Producer)
        # Determine Max Segment Duration based on Video Model
        # If LTX is involved (Cloud Movie/Parody or Local + LTX), cap at 5.0s
        # If Veo (Cloud Tech) or Stills/Zoom (Black Box), no cap (0)
        
        max_dur = 0.0 # Default: No limit
        
        # Check for LTX usage
        is_ltx = False
        if args.local:
             # Local usually uses LTX unless configured otherwise
             # Check active_models or assume LTX for enabled video modes
             # If vpform is black-box (fullmovie-still), it uses Flux Stills -> No LTX -> No Cap.
             if args.vpform in ["fullmovie-still", "black-box"]:
                 is_ltx = False
             elif args.vpform in ["draft-animatic"]:
                 # Draft animatic might use simple tools or LTX?
                 is_ltx = True # Safer to cap
             else:
                 is_ltx = True # Default Local Video is LTX
        else:
             # Cloud Mode
             if args.vpform in ["movies-movie", "parody-movie", "parody-video"]:
                 # Cloud Movies use LTX now (as per earlier logic switch) or Veo?
                 # Earlier log: "Cloud Movie Mode ... Using LTX Cloud API."
                 # So yes, LTX.
                 is_ltx = True
        
        if is_ltx:
            max_dur = 5.0
            logging.info(f"✂️  Enforcing LTX Duration Limit: {max_dur}s per segment.")
        else:
            logging.info("🕊️  Relaxed Duration: No fixed segment limit (Veo/Stills).")

        success = portion_control.run_portion(
            bible_path=p_bible,
            portions_path=p_portions,
            out_path=p_manifest,
            max_seg_dur=max_dur
        )
        if not success: sys.exit(1)
        
    logging.info(f"✅ Manifest ready: {p_manifest}")

    # --- CSSV IMPORT & VALIDATION (User Request) ---
    try:
        from mvp_shared import load_cssv, CSSV
        # Explicitly load and validate the Bible (CSSV)
        if os.path.exists(p_bible):
            bible_obj = load_cssv(p_bible)
            logging.info(f"   👁️ CSSV Validated. Vision: {bible_obj.vision[:100]}...")
            if bible_obj.mll_template:
                logging.info(f"   🧬 MLL Template: {bible_obj.mll_template}")
        else:
             logging.warning("   ⚠️ CSSV Bible file not found.")
    except Exception as e:
        logging.warning(f"   ⚠️ CSSV Validation failed (non-critical): {e}")
    # -----------------------------------------------

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

    # 4.2 AUDIO PRE-PRODUCTION (Casting)
    # For narrative forms, we generate the dialogue audio NOW so Dispatch can use it for lip-sync/timing.
    # This supports the "Dialogue -> Flux -> LTX" workflow.
    NARRATIVE_FORMS = ["full-movie", "movies-movie", "parody-movie", "parody-video", "draft-animatic", "tech-movie", "realize-ad", "3d-movie"]
    if args.vpform in NARRATIVE_FORMS:
         logging.info("🎤 Audio Pre-Production: Generating Dialogue Assets...")
         try:
             import foley_talk
             # We use the 'p_manifest' as source and update it in-place
             foley_talk.run_audio_pipeline(p_manifest, OUT_DIR, mode="kokoro") # Defaulting to Kokoro for speed/quality
         except Exception as e:
             logging.error(f"❌ Audio Pre-Production Failed: {e}")
             # Non-fatal? If it fails, LTX gets no audio. Proceed.

    # 4.5 MEMORY CLEANUP (Drop the Mic)
    # Always attempt to unload TextEngine if it was used (Local or Cloud-with-Local-Gemma)
    try:
         import text_engine
         import torch
         if text_engine._ENGINE:
             logging.info("📉 Memory Optimization: Unloading Text Engine before Video Dispatch...")
             text_engine.get_engine().unload()
             text_engine._ENGINE = None # Hard reset
             
             import gc
             gc.collect()
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache()
    except Exception as e:
         logging.warning(f"   ⚠️ Failed to unload Text Engine: {e}")

    # 4.6 AUDIO PRE-PRODUCTION (The "Table Read")
    # For Audio-to-Video forms, we must generate dialogue NOW so LTX/Veo can hear it.
    TALKIE_FORMS = ["draft-animatic", "full-movie", "parody-movie", "parody-video", "tech-movie", "realize-ad", "3d-movie"]
    if args.vpform in TALKIE_FORMS and not args.retcon:
        logging.info("🎙️ Pre-Production: Generating Dialogue Assets (Kokoro)...")
        # We invoke foley_talk directly to generate and update Manifest
        try:
             import foley_talk
             from mvp_shared import load_manifest, save_manifest
             # We need to load manifest, generate wavs, update manifest object
             manifest = load_manifest(p_manifest)
             if manifest.dialogue:
                 # Generate Dialogue Batch
                 # Use Kokoro (Local) or Cloud depending on preference?
                 # User --cloud usually implies Cloud Veo, but Kokoro is lightweight local.
                 # Let's use 'kokoro' mode if available, else 'cloud'.
                 # foley_talk main logic isn't easily imported as a library function that updates manifest.
                 # We'll use the function 'generate_kokoro_dialogue' directly if we can import it.
                 
                 output_dir = os.path.join(DIR_PARTS, "audio")
                 os.makedirs(output_dir, exist_ok=True)
                 
                 # Determine Backend
                 backend = "kokoro"
                 
                 # Run Generation
                 assets = []
                 if backend == "kokoro":
                     assets = foley_talk.generate_kokoro_dialogue(manifest.dialogue, output_dir)
                 
                 # UPDATE MANIFEST SEGMENTS
                 # We need to map dialogue wavs to Segments.
                 # A segment might have multiple lines, or a line might span segments?
                 # Usually 1 line per segment in MVP architecture or close to it.
                 # Map based on timestamp/offset?
                 # Seg has start_frame/end_frame.
                 # DialogueLine has start_offset.
                 
                 # Brute force mapping: Find segment covering the start_offset of the line.
                 for asset in assets:
                     path = asset['path']
                     offset = asset['offset']
                     
                     # Find seg
                     fps = 24.0 # Default
                     for seg in manifest.segs:
                         start_time = seg.start_frame / fps
                         end_time = seg.end_frame / fps
                         
                         # If line starts within this segment
                         if start_time <= offset < end_time:
                             # Assign!
                             # Note: If multiple lines in one seg, we might overwrite?
                             # LTX only takes one audio file.
                             # We should MIX them if multiple.
                             # For now, First In Wins or Last In?
                             # Let's just assign.
                             seg.audio_asset = path
                             logging.info(f"   🔗 Linked Voice to Seg {seg.id}: {os.path.basename(path)}")
                             break
                 
                 # Save Manifest with audio links
                 from mvp_shared import save_manifest
                 save_manifest(manifest, p_manifest) # Overwrite
                 logging.info("   ✅ Manifest updated with Audio Assets.")
                 
        except ImportError:
             logging.warning("   ⚠️ foley_talk not found. Skipping Pre-Production.")
        except Exception as e:
             logging.error(f"   ❌ Audio Pre-Production Failed: {e}")

    # 5. Dispatch Director (The Director)
    if args.retcon:
        logging.info("🛑 Retcon Mode: Stopping before Video Dispatch (Text-Only Complete).")
        # Ensure we set success=True to proceed to cleanup/save if needed, though usually save happens after dispatch.
        # Actually, lines 456+ do final save. We want to skip dispatch but do final save.
        success = True
    elif args.vpform == "3d-movie":
        # 3D MOVIE PIPELINE (Blender)
        logging.info("🎬 Mode: 3d-movie (Blender Pipeline)")
        
        # 0. Index Concepts
        import nicotime_index
        # We assume XMVP (Manifest) has been re-saved to p_manifest above
        try:
            indexer = nicotime_index.NicotimeIndexer(output_dir_name="nicotime")
            
            if args.nico == "on":
                logging.info("   🧠 Indexing Nicotime Concepts from XMVP...")
                concepts = indexer.extract_concepts_from_xmvp(p_manifest, ignore_list=indexer.get_existing_indices())
                for c in concepts:
                    indexer.create_index(c)
            else:
                logging.info("   ⏩ Skipping Nicotime Indexing (Enable with --nico on)")

        except Exception as e:
            logging.warning(f"   ⚠️ Nicotime Indexing issue: {e}")
            
        # 1. Build Library
        nicotime_dir = os.path.join(OUT_DIR, "nicotime") # Or global? nicotime_index defaults to z_training_data
        # Actually nicotime_index defaults to relative to itself if we initialized it above?
        # Let's use the path from indexer
        lib_path = os.path.join(OUT_DIR, "library.blend")
        
        # 1. Build Library
        nicotime_dir = os.path.join(OUT_DIR, "nicotime") 
        lib_path = os.path.join(OUT_DIR, "library.blend")

        # Use dispatch_blender to spawn subprocess for 'build-lib' command
        import dispatch_blender
        try:
             # run_blender_worker expects list of args to pass to worker
             success = dispatch_blender.run_blender_worker([
                 "build-lib",
                 "--nicotime", str(indexer.target_dir),
                 "--out", lib_path
             ])
             if not success:
                 logging.error("   ❌ Failed to build library (Blender subprocess failed).")
        except Exception as e:
             logging.error(f"   ❌ Exception invoking Blender worker: {e}")
             
        # 2. Dispatch Render
        import dispatch_blender
        success = dispatch_blender.run_dispatch(
            manifest_path=p_manifest,
            out_path=p_manifest_updated,
            library_path=lib_path
        )
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
    else:
        # DEFAULT DIRECTOR (LTX / Veo / Parody)
        if args.local:
             logging.info(f"🎬 Mode: {args.vpform} (Local LTX-First - Video)")
        else:
             logging.info(f"🎬 Mode: {args.vpform} (Cloud Veo - Video)")
        
        # Default Director (Handles LTX or Cloud Veo based on args.local)
        # Default Director (Handles LTX or Cloud Veo based on args.local)
        import dispatch_director
        
        # Resolve Resolution from args.res
        vid_w, vid_h = 768, 512 # Fallback
        
        # Default Logic if not specified
        if not args.res:
            if args.local:
                args.res = "352x192" # Legacy Local Default
            else:
                args.res = "720p" # new Cloud Default (1280x720)
                logging.info("☁️  Cloud Mode: Defaulting to 720p (1280x720).")
        
        if args.res:
            res_map = {
                "1080p": (1920, 1080),
                "720p": (1280, 720),
                "480p": (854, 480), # 480p usually
                "360p": (640, 360),
                "240p": (426, 240)
            }
            if args.res in res_map:
                vid_w, vid_h = res_map[args.res]
            elif "x" in args.res:
                try:
                    parts = args.res.split("x")
                    vid_w = int(parts[0])
                    vid_h = int(parts[1])
                except:
                    logging.warning(f"⚠️ Failed to parse resolution '{args.res}'. Using default.")
            else:
                 logging.warning(f"⚠️ Unknown resolution '{args.res}'. Using default.")
                 
        # Snap to 32 for LTX safety here too? Dispatch does it, but good to be explicit.
        # Dispatch handles it.
        
        success = dispatch_director.run_dispatch(
            manifest_path=p_manifest,
            mode="video",
            model_tier=args.vm, 
            out_path=p_manifest_updated,
            staging_dir=DIR_PARTS,
            pg_mode=args.pg,
            local_mode=args.local,
            width=vid_w,
            height=vid_h,
            reshoot=args.reshoot
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
        
        # RELOAD FROM DISK (Double Check)
        # Often manifest isn't updated if a crash occurred mid-batch, but files exist.
        if os.path.exists(DIR_PARTS):
             disk_files = [f for f in os.listdir(DIR_PARTS) if f.endswith(".mp4")]
             disk_count = len(disk_files)
             if disk_count > files_present:
                 logging.info(f"   🔎 Found more files on disk ({disk_count}) than in manifest ({files_present}). Trusting disk.")
                 files_present = disk_count
                 
                 # We should also attempt to PATCH the manifest in memory so stitching works?
                 # If we proceed to Post-Prod, it uses 'manifest_updated' (line 801).
                 # We need to make sure 'manifest_updated' actually HAS these files.
                 # Reconstruct local_file map?
                 # Assuming filenames are 'seg_001_...mp4' we can map them back to IDs.
                 pass

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
        
        # 6.2 Narrative Audio (The Whopper Integration)
        TALKIE_FORMS = ["draft-animatic", "full-movie", "parody-movie", "parody-video", "tech-movie", "realize-ad"]
        if args.vpform in TALKIE_FORMS and os.path.exists(final_filename):
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
