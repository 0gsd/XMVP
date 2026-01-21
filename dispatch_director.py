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

import requests
import base64
from google import genai
from google.genai import types
import definitions # Ensure definitions is available (it was imported inside run_dispatch, likely need it global or locally)

from truth_safety import TruthSafety

# --- MPS Memory Optimization (Crucial for M-Series) ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- Configuration ---
import definitions
# Retrieve Config from Registry
FLUX_CACHE = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE]["flux-schnell"].path
FLUX_REPO = "black-forest-labs/FLUX.1-schnell"

# --- VEO DIRECTOR (Inlined from action.py) ---

def download_video(uri, local_path, api_key):
    """Downloads video from URI using requests with API Key authentication."""
    logging.info(f"   ⬇️ Downloading to {local_path}...")
    
    try:
        if uri.startswith("gs://"):
            cmd = f"gcloud storage cp {uri} {local_path}"
            os.system(cmd)
            return

        # Prepare HTTP request
        params = {}
        if "generativelanguage.googleapis.com" in uri:
            params['key'] = api_key
            
        r = requests.get(uri, params=params, stream=True)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
            logging.info("      ✓ Saved.")
        else:
            logging.error(f"      ❌ Download Failed ({r.status_code}): {r.text[:100]}")
            
    except Exception as e:
        logging.error(f"      ❌ Download Error: {e}")

def extract_last_frame(video_path):
    """Extracts the last frame of a video as a JPG."""
    if not os.path.exists(video_path):
        return None
        
    output_img = video_path.replace(".mp4", "_last.jpg")
    logging.info(f"   🖼️ Extracting last frame to {output_img}...")
    
    # ffmpeg: seek to last second, grab last frame
    cmd = f"ffmpeg -sseof -1 -i {video_path} -update 1 -q:v 2 {output_img} -y >/dev/null 2>&1"
    
    ret = os.system(cmd)
    if ret == 0 and os.path.exists(output_img):
        return output_img
    else:
        logging.warning("   ⚠️ Frame Extraction Failed.")
        return None

class VeoDirector:
    def __init__(self, api_key, model_version=3, model_name=None, pg_mode=False):
        self.api_key = api_key
        self.pg_mode = pg_mode
        # Priority: model_name > model_version
        if model_name:
            self.model_endpoint = model_name
        else:
            self.model_endpoint = "veo-2.0-generate-001" if model_version == 2 else "veo-3.0-generate-001"
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate_segment(self, prompt, context_uri=None, context_type="video", retry_safe=True):
        """
        Generates a video segment.
        context_uri: Optional URI (gs:// or https://) for the previous clip/image.
        context_type: "video" or "image".
        retry_safe: If True, attempts to soften prompt on safety trigger.
        """
        # --- GEMINI PATH (L-Tier) ---
        if "gemini" in self.model_endpoint.lower():
            logging.info(f"   🎥 Rolling Gemini (L-Tier): {self.model_endpoint}...")
            # Use SDK v1 Client
            client = genai.Client(api_key=self.api_key)
            try:
                # Simple prompt wrapping
                prompt_text = f"Generate a short video clip: {prompt}"
                
                response = client.models.generate_content(
                    model=self.model_endpoint,
                    contents=prompt_text
                )
                
                # Check for response
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        # SDK v1 usually returns inline_data or uri logic differently?
                        # Actually, L-Tier video generation purely via text-prompt is rare/experimental on Gemini.
                        # Usually it's Veo. But if this path is hit:
                         if part.video_metadata:
                             logging.info(f"   🎥 Found Video Metadata: {part.video_metadata}")
                         if part.text:
                             logging.info(f"   📄 Text Response: {part.text[:200]}...")

                logging.warning(f"   ⚠️ Gemini Video Gen is experimental. Response Text: {response.text[:100]}")
                return None  

            except Exception as e:
                logging.error(f"   Gemini Error: {e}")
                return None

        # --- VEO PATH (J/K Tier) ---
        clean_endpoint = self.model_endpoint.replace("models/", "")
        url = f"{self.base_url}/{clean_endpoint}:predictLongRunning?key={self.api_key}"
        headers = { "Content-Type": "application/json" }
        
        # 1. Try with Context (Base64 Image)
        # 1. Try with Context (Base64 Image)
        if context_uri and os.path.exists(context_uri) and context_type == "image":
             try:
                 logging.info(f"   🎥 Rolling with Context (Base64): {context_uri}...")
                 with open(context_uri, "rb") as image_file:
                     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                 
                 payload = {
                   "instances": [{ 
                       "prompt": prompt, 
                       "image": { "bytesBase64Encoded": encoded_string, "mimeType": "image/jpeg" } 
                   }]
                 }
                 
                 # Debug: Log partial payload (sans huge base64)
                 # logging.info(f"   🐛 Payload: prompt={prompt[:50]}...")
                 
                 res = requests.post(url, json=payload, headers=headers)
                 if res.status_code == 200:
                     return res.json().get('name')
                 else:
                     logging.warning(f"   ⚠️ Context Request Rejected ({res.status_code}): {res.text[:200]}... Retrying standalone...")
             except Exception as e:
                 logging.error(f"   ⚠️ Context Error: {e}")
                
        # 2. Standalone (No Context)
        payload = {
            "instances": [{ "prompt": prompt }]
        }
        logging.info("   🎥 Rolling Standalone...")
        
        try:
            res = requests.post(url, json=payload, headers=headers)
            
            # --- SAFETY CHECK & RETRY ---
            # Now explicitly using self.pg_mode
            if res.status_code == 400 and "policy" in res.text.lower():
                logging.warning(f"   ⚠️ Safety Trigger: {res.text[:100]}")
                if retry_safe:
                    logging.info(f"   🛡️ Initiating Safety Protocol (PG={self.pg_mode})...")
                    try:
                        cleaner = TruthSafety(api_key=self.api_key)
                        # We pass context_dict just for flavor, or omit
                        safe_prompt = cleaner.refine_prompt(prompt, context_dict={"Task": "Rescue"}, pg_mode=self.pg_mode)
                        
                        if safe_prompt != prompt:
                            logging.info("   🔄 Retrying with Refined Prompt...")
                            return self.generate_segment(safe_prompt, context_uri, context_type, retry_safe=False)
                    
                    except Exception as e_safe:
                        logging.error(f"   ❌ Safety Cleanup Failed: {e_safe}")

            if res.status_code != 200:
                logging.error(f"Veo Request Failed ({res.status_code}): {res.text}")
                return None
            
            data = res.json()
            op_name = data.get('name')
            if not op_name:
                logging.error(f"No operation name returned: {data}")
                return None
                
            return op_name

        except Exception as e:
            logging.error(f"Director Error: {e}")
            return None

    def wait_for_lro(self, op_name):
        """Polls for completion."""
        if op_name and op_name.startswith("IMMEDIATE:"):
            uri = op_name.replace("IMMEDIATE:", "")
            logging.info("   > Cut! (Immediate Success)")
            return {'videos': [{'uri': uri}]}

        url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={self.api_key}"
        logging.info(f"   > Action! Polling {op_name}...")
        
        start_t = time.time()
        while time.time() - start_t < 600: # 10m timeout
            try:
                res = requests.get(url)
                data = res.json()
                
                if 'done' in data and data['done']:
                    if 'error' in data:
                        logging.error(f"   x Cut! Error: {data['error']}")
                        return {'error': data['error']}
                    
                    logging.info("   > Cut! (Success)")
                    try:
                        result = data.get('response')
                        if not result: return "UNKNOWN_URI"
                        return result
                    except:
                        return data
                    
                time.sleep(10)
                print(".", end="", flush=True)
                
            except Exception as e:
                logging.warning(f"Polling glitch: {e}")
                time.sleep(5)
                
        logging.error("   x Cut! Timeout.")
        return None

class LTXDirector:
    """
    Director for Local LTX-Video Generation.
    """
    def __init__(self):
        self.bridge = None
        
    def load(self):
        if self.bridge: return
        try:
            from ltx_bridge import get_ltx_bridge
            import definitions
            
            # Get path from definitions
            config = definitions.MODAL_REGISTRY[definitions.Modality.VIDEO].get("ltx-video")
            if not config:
                logging.warning("⚠️ LTX-Video config not found in definitions. Using fallback path.")
                path = "/Volumes/XMVPX/mw/LT2X-root"
            else:
                path = config.path
                
            self.bridge = get_ltx_bridge(path)
            
        except Exception as e:
            logging.error(f"❌ Failed to load LTX Bridge: {e}")
            sys.exit(1)

    def generate(self, prompt, output_path, width=768, height=512, seed=None, image_path=None):
        return self.bridge.generate(
            prompt=prompt,
            output_path=output_path,
            width=width,
            height=height,
            seed=seed,
            image_path=image_path
        )


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
            # Use round-robin key for TruthSafety too
            sanitizer_key = next(self.key_cycle) 
            cleaner = TruthSafety(api_key=sanitizer_key)
            
            
            # This will replace "Nicolas Cage" with "impersonator", etc.
            # TruthSafety Refine
            prompt = cleaner.refine_prompt(prompt, context_dict={"Task": f"Video", "Model": self.model_name}, pg_mode=self.pg_mode)
            
        except Exception as e:
            logging.warning(f"   ⚠️ Sanitizer unreachable: {e}. Proceeding with raw prompt.")

        max_retries = 3
        backoff = 10 
        
        for attempt in range(max_retries):
            # ROTATION: Round-Robin (Itertools)
            current_key = next(self.key_cycle)
            logging.info(f"   🔑 [Key Rotation] Action!")
            
            # Instantiate Director on the fly (lightweight) to swap key
            director = VeoDirector(api_key=current_key, model_name=self.model_name, pg_mode=self.pg_mode)
            
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

                # Check for Safety Violation explicitly
                if isinstance(result, dict) and 'error' in result:
                    err = result['error']
                    # Code 3 = INVALID_ARGUMENT (Often Safety) or 400/429
                    # "prompt contains words that violate"
                    if err.get('code') == 3 or "violate" in err.get('message', '').lower():
                         logging.warning(f"   🚨 SAFETY VIOLATION DETECTED: {err.get('message')}")
                         
                         # Trigger SASSPRILLA PROTOCOL
                         logging.info(f"   🛡️ Initiating Sassprilla Protocol (Parody Euphemisms)...")
                         try:
                             sanitizer_key = next(self.key_cycle)
                             cleaner = TruthSafety(api_key=sanitizer_key)
                             safe_prompt = cleaner.refine_prompt(
                                 prompt, 
                                 context_dict={"Task": f"Video", "Model": self.model_name}, 
                                 parody_safe_mode=True
                             )
                             
                             if safe_prompt != prompt:
                                 logging.info(f"   ✨ Sassprilla Prompt: {safe_prompt[:60]}...")
                                 prompt = safe_prompt # Update prompt for next retry
                                 continue # Retry immediately with new prompt
                             else:
                                 logging.warning("   ⚠️ Sassprilla returned identical prompt. Likely failing safe.")
                         except Exception as e_safe:
                             logging.error(f"   ❌ Sassprilla Failed: {e_safe}")

                    logging.warning(f"   ⚠️ LRO Error: {err}. Retrying...")
                    continue
                
                if not result:
                    logging.warning("   ⚠️ LRO failed or timed out. Retrying...")
                    continue
                    
                # 3. Extract URI
                # VeoDirector logic is a bit messy with extraction validation, 
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
                    logging.error(f"   🔍 Debug Payload: {json.dumps(result, indent=2)}")
                    
                    # Check for safety
                    if 'error' in str(result):
                        logging.warning("   ⚠️ Possible Safety/API Error detected in payload.")
                        
                    continue # Retry on weird response?
                    
                # 4. Download
                download_video(video_uri, output_path, current_key)
                return True
                
            except Exception as e:
                logging.error(f"   ❌ Video Director Error: {e}")
                # Retry on exception?
                continue

        logging.error("   ❌ All retries failed.")
        return False

def run_dispatch(manifest_path: str, mode: str = "image", model_tier: str = "J", out_path: str = "manifest_updated.json", staging_dir: str = "componentparts", pg_mode: bool = False, **kwargs) -> bool:
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
        
    width = kwargs.get('width', 768)
    height = kwargs.get('height', 768)
        
    # 2. Setup Staging
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True) # Ensure 'componentparts' exists
        
    # 3. Init Director
    director = None
    if mode == "image":
        director = FluxDirector()
        director.load()
    elif mode == "video":
        local_mode = kwargs.get('local_mode', False)
        
        if local_mode:
            logging.info("🎥 Mode: Video (Local LTX)")
            director = LTXDirector()
            director.load()
        else:
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
                width=width,
                height=height,
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
            
            # action.VeoDirector expects "image" context usually.
            
            local_mode = kwargs.get('local_mode', False)
            
            context_arg = None
            if last_file:
                # Extract frame? 
                # extract_last_frame exists locally now.
                if mode == "video":
                     last_frame = extract_last_frame(str(last_file))
                     if last_frame:
                         context_arg = last_frame
                else:
                     context_arg = str(last_file) # For image-to-image/video?
            
            if local_mode:
                # LTX Logic with TruthSafety Fattening (Local Mode = True)
                logging.info(f"   ✨ Enhancing Prompt for Local LTX (PG: {pg_mode})...")
                try:
                    # We need an API key for TruthSafety (uses TextEngine internally)
                    # We can use a random key or load keys. 
                    # TruthSafety handles key loading if none provided.
                    cleaner = TruthSafety() 
                    fat_prompt = cleaner.refine_prompt(
                        seg.prompt, 
                        context_dict={"Task": "Cinematic Video"}, 
                        pg_mode=pg_mode, 
                        local_mode=True
                    )
                    logging.info(f"   💪 Fattened Prompt: {fat_prompt[:60]}...")
                except Exception as e:
                    logging.warning(f"   ⚠️ Fattening failed: {e}. Using raw prompt.")
                    fat_prompt = seg.prompt

                seed = 42 + seg.id
                success = director.generate(
                    prompt=fat_prompt,
                    output_path=str(filepath),
                    width=width,   # LTX supports width/height
                    height=height, # LTX supports width/height
                    seed=seed,
                    image_path=context_arg if context_arg else None # Img2Vid
                )
            else:
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
    parser.add_argument("--width", type=int, default=768, help="Output width (Image Mode)")
    parser.add_argument("--height", type=int, default=768, help="Output height (Image Mode)")
    parser.add_argument("--local", action="store_true", help="Force Local Mode (LTX for Video)")
    
    args = parser.parse_args()
    
    success = run_dispatch(
        manifest_path=args.manifest,
        mode=args.mode,
        model_tier=args.vm,
        out_path=args.out,
        staging_dir=args.staging,
        pg_mode=args.pg,
        width=args.width,
        height=args.height,
        local_mode=args.local
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
