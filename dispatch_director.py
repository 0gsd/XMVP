import argparse
import logging
import json
import sys
import os
import torch
import gc
from pathlib import Path
from diffusers import FluxPipeline
import time
from mvp_shared import Manifest, load_manifest, save_manifest, load_api_keys

import itertools
import random

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import Veo from action (Adapter pattern)
try:
    import action
except ImportError:
    logging.warning("⚠️ action.py not found. Video generation will be disabled.")
    action = None

import sanitizer

# --- MPS Memory Optimization (Crucial for M-Series) ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- Configuration ---
FLUX_CACHE = "/Volumes/ORICO/flux_models"
FLUX_REPO = "black-forest-labs/FLUX.1-schnell"

class FluxDirector:
    def __init__(self):
        self.pipe = None
        
    def load(self):
        if self.pipe: return
        
        logging.info(f"⏳ Loading Flux.1 Schnell from {FLUX_CACHE}...")
        try:
            if not os.path.exists(FLUX_CACHE):
                os.makedirs(FLUX_CACHE, exist_ok=True)
                
            self.pipe = FluxPipeline.from_pretrained(
                FLUX_REPO,
                cache_dir=FLUX_CACHE,
                torch_dtype=torch.bfloat16
            )
            
            if torch.backends.mps.is_available():
                self.pipe.enable_model_cpu_offload()
                logging.info("🚀 Flux loaded on MPS (CPU Offload enabled)")
            else:
                self.pipe.enable_model_cpu_offload()
                logging.warning("⚠️ MPS not available. Using CPU Offload.")
                
        except Exception as e:
            logging.error(f"❌ Failed to load Flux: {e}")
            sys.exit(1)

    def generate(self, prompt: str, width: int, height: int, output_path: str, seed: int = 42) -> bool:
        try:
            logging.info(f"   🎨 Painting: {prompt[:50]}...")
            
            # memory cleanup before acting
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            image = self.pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=512,
                generator=torch.Generator("mps").manual_seed(seed)
            ).images[0]
            
            image.save(output_path)
            return True
        except Exception as e:
            logging.error(f"   ❌ Generation failed: {e}")
            return False

class VideoDirectorAdapter:
    """
    Wraps action.VeoDirector for the MVP Dispatcher with Key Rotation.
    """
    def __init__(self, keys: list, model_name: str, pg_mode: bool = False):
        if not action:
            raise ImportError("action module missing")
        self.keys = keys
        self.model_name = model_name
        self.pg_mode = pg_mode
        random.shuffle(self.keys) # Shuffle once
        self.key_cycle = itertools.cycle(self.keys)
        
    def generate(self, prompt: str, output_path: str, context_uri: str = None) -> bool:
        logging.info(f"   🎥 Rolling Video: {prompt[:50]}...")
        
        # 0. Pre-emptive Sanitization (Proactive Safety)
        try:
            # Pick a key for sanitizer (peek at next without consuming, or just consume)
            # Sanitizer is cheap (Flash), let's just use one from the cycle or a random one?
            # User wants strict cycle. But sanitizer might burn a request. 
            # Let's simple use a random choice for sanitizer to avoid advancing the main video cycle "off beat"?
            # Or just advance it. It's fine.
            # Use round-robin key for sanitizer too
            sanitizer_key = next(self.key_cycle) 
            cleaner = sanitizer.Sanitizer(api_key=sanitizer_key)
            
            
            # This will replace "Nicolas Cage" with "impersonator" (or "Actor N.C." in PG mode), etc.
            prompt = cleaner.soften_prompt(prompt, pg_mode=self.pg_mode)
            
        except Exception as e:
            logging.warning(f"   ⚠️ Sanitizer unreachable: {e}. Proceeding with raw prompt.")

        max_retries = 3
        backoff = 10 
        
        for attempt in range(max_retries):
            # ROTATION: Round-Robin (Itertools)
            current_key = next(self.key_cycle)
            logging.info(f"   🔑 [Key Rotation] Action!")
            
            # Instantiate Director on the fly (lightweight) to swap key
            director = action.VeoDirector(api_key=current_key, model_name=self.model_name)
            
            # 1. Generate
            try:
                if attempt > 0:
                    logging.info(f"   🔄 Retry #{attempt} (Backoff {backoff}s)...")
                    time.sleep(backoff)
                    backoff *= 2 # Exponential backoff
                
                op_name = director.generate_segment(
                    prompt=prompt, 
                    context_uri=context_uri, 
                    context_type="video" if context_uri else "image" # Heuristic
                )
                
                if not op_name:
                    logging.warning("   ⚠️ Launch failed (no op_name). Retrying...")
                    continue
                    
                # 2. Wait
                result = director.wait_for_lro(op_name)
                if not result:
                    logging.warning("   ⚠️ LRO failed or timed out. Retrying...")
                    continue
                    
                # 3. Extract URI
                # action.VeoDirector logic is a bit messy with extraction validation, 
                # let's rely on finding 'uri' in the deep structure or 'video' key
                video_uri = None
                
                # Helper to dig for URI
                def find_uri(d):
                    if isinstance(d, dict):
                        if 'uri' in d and 'video' in str(d): # Simple heuristic check?
                            return d['uri']
                        for k, v in d.items():
                            if k == 'uri' and isinstance(v, str) and v.startswith('http'):
                                return v
                            res = find_uri(v)
                            if res: return res
                    elif isinstance(d, list):
                        for item in d:
                            res = find_uri(item)
                            if res: return res
                    return None
                    
                # Try specific paths first (matching action.py)
                if 'generateVideoResponse' in result:
                    samples = result['generateVideoResponse'].get('generatedSamples')
                    if samples:
                         video_uri = samples[0]['video']['uri']
                elif 'videos' in result:
                    video_uri = result['videos'][0]['uri']
                elif 'video' in result:
                    video_uri = result['video']['uri']
                
                if not video_uri:
                    # Fallback search
                    video_uri = find_uri(result)
                    
                if not video_uri:
                    logging.error("   ❌ URI not found in response.")
                    continue # Retry on weird response?
                    
                # 4. Download
                action.download_video(video_uri, output_path, current_key)
                return True
                
            except Exception as e:
                logging.error(f"   ❌ Video Director Error: {e}")
                # Retry on exception?
                continue

        logging.error("   ❌ All retries failed.")
        return False

def run_dispatch(manifest_path: str, mode: str = "image", model_tier: str = "J", out_path: str = "manifest_updated.json", staging_dir: str = "componentparts", pg_mode: bool = False) -> bool:
    """
    Executes the Dispatch pipeline.
    mode: "image" (Flux) or "video" (Veo)
    """
    # 1. Load Data
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        logging.error(f"Failed to load manifest: {manifest_path} -> {e}")
        return False
        
    # 2. Setup Staging
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True) # Ensure 'componentparts' exists
        
    # 3. Init Director
    director = None
    if mode == "image":
        director = FluxDirector()
        director.load()
    elif mode == "video":
        keys = load_api_keys()
        if not keys:
            logging.error("No API Keys for Video Dispatch.")
            return False
        
        # Resolve Model Name using action/definitions logic if possible, 
        # or hardcode fallback? 
        # We'll use definitions if available
        try:
            import definitions
            model_name = definitions.get_video_model(model_tier)
        except:
             model_name = "veo-2.0-generate-001"
             
        # Pass ALL keys to the adapter for rotation
        director = VideoDirectorAdapter(keys, model_name=model_name, pg_mode=pg_mode)
    else:
        logging.error(f"Unknown mode: {mode}")
        return False
    
    # 4. Action!
    logging.info(f"🎬 Director calling action on {len(manifest.segs)} segments (Mode: {mode})...")
    
    last_file = None
    
    # Sort segments by ID to ensure sequence (critical for video context)
    sorted_segs = sorted(manifest.segs, key=lambda s: s.id)
    
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    
    for seg in sorted_segs:
        # Check if done
        if seg.id in manifest.files:
            if os.path.exists(manifest.files[seg.id]):
                logging.info(f"   ⏩ Skipping Seg {seg.id} (Already wrapped).")
                last_file = manifest.files[seg.id]
                consecutive_failures = 0 # Reset on success/skip
                continue
                
        print(f"\n🎥 SEGMENT {seg.id}: {seg.prompt[:60]}...")
        
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logging.error(f"❌ Aborting Dispatch: {consecutive_failures} consecutive failures (Likely API Quota or Outage).")
            return False
        
        base_name = f"seg_{seg.id:03d}_{int(time.time())}"
        ext = ".mp4" if mode == "video" else ".png"
        filename = base_name + ext
        filepath = staging_path / filename
        
        success = False
        if mode == "image":
             # Flux Logic
             seed = 42 + seg.id
             success = director.generate(
                prompt=seg.prompt,
                width=768, # Fixed for MVP
                height=768,
                output_path=filepath,
                seed=seed
            )
        elif mode == "video":
            # Veo Logic
            # Context Logic: Use last wrapper file?
            # Issue: Veo context needs to be a specific URI (File API or GCS).
            # Local file path doesn't work directly for Veo unless we upload it or base64 it.
            # action.VeoDirector supports "context_uri" as a local path if it handles Base64.
            # Let's check action.VeoDirector implementation...
            # Yes, it checks: if context_uri and os.path.exists(context_uri) ... base64 encode.
            # So pass the local filepath of the previous segment.
            
            # BUT: Video-to-Video context? 
            # realize.py extracts the last frame to use as Image context.
            # Let's verify if we need to do that here or if VeoDirector handles it.
            # action.VeoDirector expects "image" context usually.
            
            context_arg = None
            if last_file:
                # Extract frame? 
                # action.extract_last_frame exists.
                if mode == "video" and action:
                     last_frame = action.extract_last_frame(str(last_file))
                     if last_frame:
                         context_arg = last_frame
                else:
                     context_arg = str(last_file) # For image-to-image/video?
            
            success = director.generate(
                prompt=seg.prompt,
                output_path=str(filepath),
                context_uri=context_arg
            )
            
        if success:
            manifest.files[seg.id] = str(filepath)
            last_file = filepath
            consecutive_failures = 0 # Reset
            logging.info(f"   ✅ Wrapped: {filepath}")
            
            # Rate Limit Protection (Veo 3 Preview is very strict)
            if mode == "video":
                cooldown = 30
                logging.info(f"   ⏳ Cooling down for {cooldown}s to protect Key/Project Quota...")
                time.sleep(cooldown)
        else:
            logging.warning(f"   ❌ Failed to shoot Seg {seg.id}")
            consecutive_failures += 1
            # If video fails, maybe we should stop? 
            # Or continue without context? 
            # For MVP, we continue.
            
    # 5. Wrap
    save_manifest(manifest, out_path)
    logging.info(f"🎉 Production Wrap! Updated manifest saved to {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Dispatch Director: The Director")
    parser.add_argument("--manifest", type=str, required=True, help="Path to input Manifest JSON")
    parser.add_argument("--out", type=str, default="manifest_updated.json", help="Output path for updated Manifest")
    parser.add_argument("--staging", type=str, default="componentparts", help="Directory to save assets")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"], help="Generation Mode")
    parser.add_argument("--vm", type=str, default="J", help="Video Model Tier (if mode=video)")
    parser.add_argument("--pg", action="store_true", help="Enable PG Mode (Relaxed Celebrity/Strict Child Safety)")
    
    args = parser.parse_args()
    
    success = run_dispatch(
        manifest_path=args.manifest,
        mode=args.mode,
        model_tier=args.vm,
        out_path=args.out,
        staging_dir=args.staging,
        pg_mode=args.pg
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
