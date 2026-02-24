import argparse
import logging
import json
import sys
import os
import torch
import gc
from pathlib import Path
from diffusers import FluxPipeline
try:
    from flux_bridge import get_flux_bridge
except ImportError:
    logging.warning("FluxBridge unavailable. Image generation will fail.")
import time
from mvp_shared import Manifest, load_manifest, save_manifest, load_api_keys

import itertools
import random
import shutil # Added for file moving

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import requests
import base64
import io
from PIL import Image
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
import definitions # Ensure definitions is available (it was imported inside run_dispatch, likely need it global or locally)

from truth_safety import TruthSafety

# --- MPS Memory Optimization (Crucial for M-Series) ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- Configuration ---
import definitions
# Retrieve Config from Registry
def get_flux_cache_path():
    try:
        conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-dev")
        if not conf:
            conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-dev")
        return conf.path if conf else "/Volumes/XMVPX/mw/flux-root/dev"
    except:
        return "/Volumes/XMVPX/mw/flux-root/dev"

FLUX_CACHE = get_flux_cache_path()
FLUX_REPO = "black-forest-labs/FLUX.1-schnell" # Updated repo for cloud fallback

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

    def generate(self, prompt, output_path, width=768, height=512, seed=None, image_path=None, num_frames=121):
        return self.bridge.generate(
            prompt=prompt,
            output_path=output_path,
            width=width,
            height=height,
            seed=seed,
            image_path=image_path,
            num_frames=num_frames
        )




# FluxDirector removed. Replaced by flux_bridge usage.

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
        # ONLY if PG Mode is active. If not, we trust the prompt (or let Veo filter it natively).
        if self.pg_mode:
            try:
                # Pick a key for sanitizer
                sanitizer_key = next(self.key_cycle) 
                cleaner = TruthSafety(api_key=sanitizer_key)
                
                # TruthSafety Refine
                prompt = cleaner.refine_prompt(prompt, context_dict={"Task": f"Video", "Model": self.model_name}, pg_mode=self.pg_mode)
                logging.info(f"   🛡️ Sanitized Prompt: {prompt[:60]}...")
            except Exception as e:
                logging.warning(f"   ⚠️ Safety Check failed: {e}. Proceeding with raw prompt.")
        else:
             logging.info("   🛡️ Safety Filters: OFF (Sending Raw Prompt)")

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

class LTXCloudDirector:
    """
    Cloud-based LTX Director using HF InferenceClient (fal-ai provider).
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Use fal-ai provider for LTX-Video (Image-to-Video optimized)
        self.client = InferenceClient(provider="fal-ai", api_key=self.api_key)
        
    def generate(self, prompt, output_path, num_frames=121, fps=24, image_path=None, audio_path=None):
        logging.info(f"   ☁️ Rolling LTX Cloud (Hybrid): {prompt[:50]}... ({num_frames} frames) Audio: {bool(audio_path)}")
        
        if not image_path or not os.path.exists(image_path):
            logging.error("      ❌ Hybrid Mode requires an input image!")
            return False
            
        try:
             # Load Image Bytes
             with open(image_path, "rb") as f:
                 img_bytes = f.read()
                 
             # Call Image-to-Video
             # Note: client.image_to_video returns bytes of the video
             # We pass 'num_frames' and 'fps' as kwargs which InferenceClient forwards to the provider.
             
             # Audio Logic (Audio-to-Video)
             extra_params = {
                 "num_frames": num_frames,
                 "fps": fps,
                 "num_inference_steps": 30
             }
             
             if audio_path and os.path.exists(audio_path):
                 logging.info(f"      🎤 Attaching Audio: {os.path.basename(audio_path)}")
                 with open(audio_path, "rb") as af:
                     audio_b64 = base64.b64encode(af.read()).decode('utf-8')
                     # Use Data URI for Fal
                     extra_params["audio_url"] = f"data:audio/wav;base64,{audio_b64}"
             
             video_bytes = self.client.image_to_video(
                 image=img_bytes,
                 prompt=prompt,
                 model="Lightricks/LTX-Video",
                 **extra_params
             )
             
             if video_bytes:
                 with open(output_path, "wb") as f:
                     f.write(video_bytes)
                 logging.info("      ✓ Saved from Cloud (fal-ai).")
                 return True
             else:
                 logging.error("      ❌ Cloud Gen returned empty bytes.")
                 return False
                 
        except Exception as e:
             logging.error(f"      ❌ Cloud Gen Exception: {e}")
             return False

class FluxCloudDirector:
    """
    Cloud-based Flux Director using HF InferenceClient (fal-ai).
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Use fal-ai provider for Flux.1-dev or similar standard endpoint
        self.client = InferenceClient(provider="fal-ai", api_key=self.api_key)
        
        # Cloud Mapping: If local is dev-based, map to well-supported dev model on cloud if necessary.
        target_model = FLUX_REPO
        if "dev" in target_model.lower():
             logging.info(f"   ☁️ Mapping Local '{target_model}' -> Cloud 'black-forest-labs/FLUX.1-dev'")
             target_model = "black-forest-labs/FLUX.1-dev"
             
        self.model = target_model
        
    def generate(self, prompt, width=1280, height=720, seed=None):
        logging.info(f"   ☁️ Flux Cloud: Generating '{prompt[:40]}...' ({width}x{height})")
        
        try:
            # InferenceClient text_to_image returns a PIL Image by default
            image = self.client.text_to_image(
                prompt=prompt,
                model=self.model,
                width=width,
                height=height,
                num_inference_steps=28, # Standard dev steps
                guidance_scale=3.5,
                seed=seed
            )
            return image
        except Exception as e:
            logging.error(f"   ❌ Flux Cloud Error: {e}")
            return None

def run_dispatch(manifest_path: str, mode: str = "image", model_tier: str = "J", out_path: str = "manifest_updated.json", staging_dir: str = "componentparts", pg_mode: bool = False, **kwargs) -> bool:
    """
    Executes the Dispatch pipeline.
    mode: "image" (Flux) or "video" (Veo/LTX)
    """
    # 1. Load Data
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        logging.error(f"Failed to load manifest: {manifest_path} -> {e}")
        return False
        
    width = kwargs.get('width', 768)
    height = kwargs.get('height', 512)
    strength = kwargs.get('strength', 0.50)
        
    # 2. Setup Staging
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True) # Ensure 'componentparts' exists
        
    # 3. Init Director
    director = None
    local_mode = kwargs.get('local_mode', False)
    
    if mode == "image":
        # FLUX
        if local_mode:
            logging.info(f"   🌊 Initializing Flux Bridge from {FLUX_CACHE}...")
            try:
                director = get_flux_bridge(FLUX_CACHE) 
            except Exception as e:
                logging.error(f"Failed to load Flux Bridge: {e}")
                return False
        else:
             logging.info("   ☁️  Mode: Image (Cloud Flux)")
             _ = load_api_keys()
             hf_token = os.environ.get("HF_TOKEN")
             if not hf_token:
                 logging.error("❌ HF_TOKEN missing for Cloud Flux.")
                 return False
             director = FluxCloudDirector(api_key=hf_token)
            
    elif mode == "video":
        
        if local_mode:
            logging.info("🎥 Mode: Video (Local LTX-First)")
            director = LTXDirector()
            director.load()
            
            # Local Hybrid: Connect Flux for Keyframes (JIT MODE - DO NOT LOAD YET)
            logging.info("   🔌 JIT Hybrid Mode: Flux will be loaded on-demand.")
            director.aux_director = "JIT_FLUX_TOKEN" # Marker to trigger JIT logic
        else:
            logging.info("☁️ Mode: Video (Cloud LTX-Video)")
            
            # Prime Environment Variables from YAML
            _ = load_api_keys() 
            ltx_api_key = os.environ.get("HF_TOKEN")
            if not ltx_api_key:
                logging.error("❌ HF_TOKEN not found in env_vars.yaml or environment.")
                return False
                
            director = LTXCloudDirector(api_key=ltx_api_key)
            
            # HYBRID MODE: We ALSO need Flux for Keyframes
            logging.info("   🔌 Connecting Hybrid Link (Flux Keyframes)...")
            try:
                if local_mode:
                    aux_director = get_flux_bridge(FLUX_CACHE)
                else:
                    # Cloud Hybrid -> NOW USING LOCAL FLUX per user request (Hybrid Fal/Local)
                    logging.info("   🏠 Hybrid Override: Using Local Flux Bridge for Cloud LTX Keyframes.")
                    aux_director = get_flux_bridge(FLUX_CACHE)
                    # aux_director = FluxCloudDirector(api_key=ltx_api_key) # Users same token
                    
                director.aux_director = aux_director
            except Exception as e:
                logging.warning(f"   ⚠️ Could not load Flux for Hybrid Keyframes: {e}")
                director.aux_director = None

    else:
        logging.error(f"Unknown mode: {mode}")
        return False
    
    # 4. Action!
    logging.info(f"🎬 Director calling action on {len(manifest.segs)} segments (Mode: {mode})...")
    
    last_file = None
    sorted_segs = sorted(manifest.segs, key=lambda s: s.id)
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    
    for seg in sorted_segs:
        # Check if done (unless Reshoot forced)
        reshoot_mode = kwargs.get('reshoot', False)
        
        if not reshoot_mode and seg.id in manifest.files and os.path.exists(manifest.files[seg.id]):
             logging.info(f"   ⏩ Skipping Seg {seg.id} (Already wrapped).")
             last_file = manifest.files[seg.id]
             consecutive_failures = 0 
             continue
                
        print(f"\n🎥 SEGMENT {seg.id} {'(RESHOOT)' if reshoot_mode else ''}: {seg.prompt[:60]}...")
        
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logging.error(f"❌ Aborting Dispatch: {consecutive_failures} consecutive failures.")
            return False
        
        base_name = f"seg_{seg.id:03d}_{int(time.time())}"
        ext = ".mp4" if mode == "video" else ".png"
        filename = base_name + ext
        filepath = staging_path / filename
        
        success = False
        
        if mode == "image":
             # Flux Logic
             seed = 42 + seg.id
             
             # Extract Context Image
             context_arg = str(last_file) if last_file and os.path.exists(str(last_file)) else None
             if context_arg and context_arg.endswith((".mp4", ".mov")):
                  context_arg = extract_last_frame(context_arg)
                  
             img = director.generate(
                prompt=seg.prompt,
                width=width,
                height=height,
                seed=seed,
                strength=strength,
                image_path=context_arg
            )
             if img:
                 img.save(filepath)
                 success = True
        
        elif mode == "video":
            # LTX Logic (Local OR Cloud)
            
            # Use Argument Dimensions (Default 1280x720)
            # LTX prefers multiples of 32, but Flux (often used for keyframes) 
            # requires multiples of 128 (escalated from 64) on MPS to prevent buffer size issues.
            t_width = (width // 128) * 128
            t_height = (height // 128) * 128
            
            # Calculate Target Frames
            target_frames = seg.end_frame - seg.start_frame
            if target_frames < 33: target_frames = 33 # Min ~1.5s
            
            # Snap to 8k+1 logic (LTX Preference)
            remainder = (target_frames - 1) % 8
            if remainder != 0:
                target_frames = target_frames - remainder
                
            # Context Logic (Img2Vid)
            context_arg = str(last_file) if last_file and os.path.exists(str(last_file)) else None
            
            if local_mode:
                # Local LTX Logic (with Fattening + Context)
                ltx_context_image = None
                if context_arg:
                    if context_arg.endswith(".mp4") or context_arg.endswith(".mov"):
                         frame = extract_last_frame(context_arg)
                         if frame and os.path.exists(frame): ltx_context_image = frame
                    else:
                         ltx_context_image = context_arg

                # IF NO CONTEXT, GENERATE KEYFRAME (Local Hybrid JIT)
                if not ltx_context_image and hasattr(director, 'aux_director') and director.aux_director == "JIT_FLUX_TOKEN":
                     logging.info(f"   🎨 Local Hybrid (JIT): Flux -> Keyframe...")
                     keyframe_path = str(filepath).replace(".mp4", "_key.png")
                     cinematic_prompt = f"Cinematic wide shot, high quality, {seg.prompt}"
                     
                     try:
                         # 1. Unload LTX (Free VRAM)
                         # We can't easily unload 'director' itself as it holds the state, but we can tell bridge to unload pipelines?
                         # ltx_bridge has free_memory() but pipelines are persistent.
                         # We need to manually nuke pipes if we need space.
                         # But wait, LTXDirector wraps the bridge.
                         
                         # Hack: Use deep private access or just rely on OS paging? 
                         # No, 96GB fails. We must be aggressive.
                         # LTX Bridge holds 'txt2vid_pipe' and 'img2vid_pipe'.
                         if director.bridge:
                              # Manually delete pipes
                              if director.bridge.txt2vid_pipe: 
                                  del director.bridge.txt2vid_pipe
                                  director.bridge.txt2vid_pipe = None
                              if director.bridge.img2vid_pipe:
                                  del director.bridge.img2vid_pipe
                                  director.bridge.img2vid_pipe = None
                              director.bridge.free_memory()
                              logging.info("      📉 JIT: LTX Unloaded.")

                         # 2. Load Flux
                         aux_director = get_flux_bridge(FLUX_CACHE)
                         
                         # 3. Generate
                         kf_img = aux_director.generate(
                             prompt=cinematic_prompt,
                             width=t_width,
                             height=t_height,
                             seed=42 + seg.id
                         )
                         
                         if kf_img:
                             kf_img.save(keyframe_path)
                             ltx_context_image = keyframe_path
                             logging.info(f"      ✓ JIT: Keyframe Ready: {keyframe_path}")
                             
                         # 4. Unload Flux
                         aux_director.unload()
                         logging.info("      📉 JIT: Flux Unloaded.")

                     except Exception as e:
                         logging.error(f"      ❌ Flux JIT Failed: {e}")

                     # 5. Reload LTX (Will happen automatically in director.generate via lazy load or we trigger it?)
                     # LTXDirector.generate calls bridge.generate -> load_img2vid()
                     # So it handles reload. Perfect.

                # Fatten Prompt (DISABLED per user request for continuity/speed)
                # "the 'fattening' seems to break continuity from clip to clip... and makes everything take longer."
                fat_prompt = seg.prompt 
                # try:
                #     cleaner = TruthSafety() 
                #     fat_prompt = cleaner.refine_prompt(...)
                # except:
                #     fat_prompt = seg.prompt

                # Fix dims (Already calculated above as t_width, t_height)
                
                success = director.generate(
                    prompt=fat_prompt,
                    output_path=str(filepath),
                    width=t_width, 
                    height=t_height,
                    num_frames=target_frames,
                    seed=42 + seg.id,
                    image_path=ltx_context_image 
                )
            else:
                # Cloud LTX Live (Hybrid)
                # 1. Generate Keyframe (Flux)
                keyframe_path = str(filepath).replace(".mp4", "_key.png")
                
                if hasattr(director, 'aux_director') and director.aux_director:
                     logging.info(f"   🎨 Hybrid: Generating Keyframe (Flux)...")
                     # Enhance Prompt for Cinematic Quality (User Request)
                     cinematic_prompt = f"Cinematic wide shot, high quality, {seg.prompt}"
                     
                     # Use Flux Bridge
                     seed = 42 + seg.id
                     try:
                         kf_img = director.aux_director.generate(
                             prompt=cinematic_prompt,
                             width=t_width,
                             height=t_height, 
                             seed=seed
                         )
                         if kf_img:
                             kf_img.save(keyframe_path)
                             logging.info(f"      ✓ Keyframe Ready: {keyframe_path}")
                         else:
                             logging.error("      ❌ Flux returned no image.")
                             keyframe_path = None
                     except Exception as e:
                         logging.error(f"      ❌ Flux Keyframe Gen Failed: {e}")
                         keyframe_path = None
                else:
                    logging.warning("   ⚠️ No Flux Bridge available for keyframe. LTX might fail if text-to-video is unsupported.")
                    keyframe_path = None

                # 2. Animate (LTX Cloud)
                success = director.generate(
                    prompt=seg.prompt,
                    output_path=str(filepath),
                    num_frames=target_frames,
                    fps=24,
                    image_path=keyframe_path,
                    audio_path=seg.audio_asset # Pass the pre-generated audio!
                )
            
        if success:
            manifest.files[seg.id] = str(filepath)
            last_file = filepath
            consecutive_failures = 0
            logging.info(f"   ✅ Wrapped: {filepath}")
        else:
            logging.warning(f"   ❌ Failed to shoot Seg {seg.id}")
            consecutive_failures += 1
            
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
    parser.add_argument("--width", type=int, default=1280, help="Output width (Image Mode)")
    parser.add_argument("--height", type=int, default=720, help="Output height (Image Mode)")
    parser.add_argument("--local", action="store_true", help="Force Local Mode (LTX for Video)")
    parser.add_argument("--reshoot", action="store_true", help="Force Re-shoot (Ignore existing files)")
    
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
        local_mode=args.local,
        reshoot=args.reshoot
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
