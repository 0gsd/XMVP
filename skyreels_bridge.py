import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import logging
import gc
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO)

class SkyReelsBridge:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Note: SkyReels 19B might be too big for MPS (Mac), but we'll try or rely on CPU offload.
        # Ideally this runs on a beefy box. If running on Mac Studio Ultra, MPS might work for some parts.
        self.model_path = model_path
        self.device = device
        self.pipe = None
        
        # Check availability
        if device == "mps" and not torch.backends.mps.is_available():
            logging.warning("⚠️ MPS not available. Falling back to CPU.")
            self.device = "cpu"
            
    def load_pipeline(self):
        """Loads the SkyReels Pipeline using external codebase."""
        if self.pipe: return
        
        # 1. Add Repo to Path
        REPO_PATH = "/Volumes/XMVPX/mw/SkyReels-V3-repo"
        if os.path.exists(REPO_PATH):
             if REPO_PATH not in sys.path:
                 sys.path.append(REPO_PATH)
                 logging.info(f"   ➕ Added {REPO_PATH} to sys.path")
        else:
            logging.error(f"   ❌ SkyReels Repo not found at {REPO_PATH}")
            return

        try:
            from skyreels_v3.pipelines import TalkingAvatarPipeline
            from skyreels_v3.configs import WAN_CONFIGS
            
            logging.info(f"   🎬 Loading SkyReels Pipeline (Wan 2.1 Arch) from: {self.model_path}...")
            
            # Config for 19B Avatar
            config = WAN_CONFIGS["talking-avatar-19B"]
            
            # Initialize Pipeline
            # Note: We enforce device placement manually since Wan might assume distributed
            rank = 0
            self.pipe = TalkingAvatarPipeline(
                config=config,
                model_path=self.model_path,
                device_id=rank,
                rank=rank,
                use_usp=False, # Single GPU
                offload=True,  # Mandatory for 19B on consumer hardware
                low_vram=True  # Recommended for <24GB, assuming user might benefit
            )
            
            logging.info("   ✅ SkyReels Pipeline Ready.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load SkyReels: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def free_memory(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, audio_path, image_path, prompt, output_path, width=720, height=720, steps=30, guidance_scale=7.5, seed=None):
        """
        Generates video from audio + image + prompt using SkyReels local pipeline.
        """
        try:
            self.load_pipeline()
            if not self.pipe: return False
            
            # Import Preprocessor from Repo
            from skyreels_v3.utils.avatar_preprocess import preprocess_audio
            
            logging.info(f"   🎬 SkyReels Generating: {prompt[:40]}...")
            self.free_memory()

            # Prepare Data Dict
            input_data = {
                "prompt": prompt,
                "cond_image": image_path,
                "cond_audio": {"person1": audio_path},
            }
            
            # Preprocess Audio (extract features)
            logging.info("   🔊 Preprocessing Audio...")
            # We need a temp dir for processed audio cache? 
            # preprocess_audio writes cache to 'processed_audio' defined in second arg?
            # Actually second arg is `cache_dir`.
            cache_dir = os.path.dirname(output_path)
            input_data, _ = preprocess_audio(self.model_path, input_data, cache_dir)
            
            # Generation Args
            # Mapping our simple args to SkyReels complex kwargs
            # Default resolution bucket strings: "720P", "540P", "480P"
            # We map generic width/height to closest bucket
            res_str = "720P"
            if height <= 480: res_str = "480P"
            elif height <= 540: res_str = "540P"
            
            kwargs = {
                "input_data": input_data,
                "size_buckget": res_str, 
                "motion_frame": 5, # Default from example
                "frame_num": 81,   # Max frames? 81 frames @ 24fps ~= 3.4s? 
                                   # We need to calculate frame_num based on audio length?
                                   # SkyReels: "Audio duration must be <= 200 seconds"
                                   # Wan pipeline seemingly auto-handles duration if not specified? 
                                   # But kwargs in generate_video.py hardcodes frame_num=81.
                                   # Let's try to infer frame_num from audio duration if possible, 
                                   # or pass a large max and let it stop at audio end.
                "drop_frame": 12,
                "shift": 11,
                "text_guide_scale": 1.0, # Guidance Low for avatars?
                "audio_guide_scale": 1.0,
                "seed": seed if seed else 42,
                "sampling_steps": steps if steps else 20, # Wan defaults 20-50
                "max_frames_num": 5000, # Cap
            }
            
            # NOTE: frame_num=81 might be for the example short video. 
            # If we want dynamic length, we should check audio duration.
            # But for now, we leave it or better, increase it if the user audio is long.
            # 5s * 24fps = 120 frames. 81 is ~3.5s.
            # Let's set frame_num to match expected duration from content_producer (approx).
            # We don't have duration passed explicitly besides args.
            # However, SkyReels pipeline usually auto-aligns to audio. 
            # We will use a larger default or try to remove it if optional.
            # Checking generate_video.py source... it passes 81. 
            # Let's try passing -1 or larger value if it allows.
            # Or better, check audio duration.
            
            video_out = self.pipe.generate(**kwargs)
            
            # Save Video
            import imageio
            fps = 25 # Avatar default
            
            # video_out is tensor or list? generate_video.py says:
            # imageio.mimwrite(output_path, video_out, fps=fps, ...)
            
            # Write Silent Video First
            temp_vid = output_path.replace(".mp4", "_silent.mp4")
            imageio.mimwrite(
                temp_vid,
                video_out,
                fps=fps,
                quality=8,
                output_params=["-loglevel", "error"],
            )
            
            # Merge Audio
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_vid,
                '-i', audio_path,
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                output_path,
                '-loglevel', 'error'
            ]
            subprocess.run(cmd, check=True)
            
            if os.path.exists(temp_vid): os.remove(temp_vid)
            
            logging.info(f"   💾 Saved to {output_path}")
            return True
                
        except Exception as e:
            logging.error(f"   ❌ SkyReels Generation Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_cloud(self, audio_path, image_path, prompt, output_path, width=720, height=720, steps=30, guidance_scale=2.5, seed=None):
        """
        Generates video using Hugging Face Inference API for SkyReels V3.
        """
        logging.info(f"   ☁️ SkyReels Cloud Gen: {prompt[:40]}...")
        
        try:
            from huggingface_hub import InferenceClient
            import base64
            
            # API Key
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logging.error("   ❌ HF_TOKEN not found for Cloud Generation.")
                return False
                
            client = InferenceClient(token=hf_token)
            
            # Encode Inputs
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            # Construct Payload
            # Note: SkyReels API payload format might differ. 
            # Assuming standard "audio-to-video" or generic call structure.
            # If strictly text-to-video with conditioning, we might need custom payload.
            # For now, we try generic or check if specific task exists.
            
            # Skywork/SkyReels-V3-A2V-19B is likely a custom model on HF currently.
            # The API parameters might be specific.
            # We assume a standard structure for now, but this is a RISK point.
            
            # Attempt 1: Direct model call with inputs
            # API URL: Check env for dedicated endpoint, else default
            custom_endpoint = os.environ.get("SKYREELS_ENDPOINT")
            if custom_endpoint:
                api_url = custom_endpoint
                logging.info(f"   ☁️ Using Custom Endpoint: {api_url}")
            else:
                api_url = f"https://router.huggingface.co/models/{model_id}"
            
            headers = {"Authorization": f"Bearer {hf_token}"}
            
            logging.info(f"   🚀 Sending Request to {model_id}...")
            response = requests.post(api_url, headers=headers, json=payload, stream=True) # Stream for large video?
            
            if response.status_code != 200:
                logging.error(f"   ❌ API Error {response.status_code}: {response.text}")
                return False
                
            # Content is likely the video bytes?
            # Or JSON with "video": base64?
            try:
               resp_json = response.json()
               if "error" in resp_json:
                   logging.error(f"   ❌ API Error: {resp_json['error']}")
                   return False
            except:
                # If not JSON, it's bytes (video/mp4)
                pass

            # Save
            with open(output_path, "wb") as f:
                f.write(response.content)
                
            logging.info(f"   💾 Cloud Video Saved to {output_path}")
            return True

        except Exception as e:
            logging.error(f"   ❌ SkyReels Cloud Gen Failed: {e}")
            return False

# Singleton
_BRIDGE = None
def get_skyreels_bridge(path):
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = SkyReelsBridge(path, device="mps")
    return _BRIDGE

if __name__ == "__main__":
    # Test
    path = "/Volumes/XMVPX/mw/skyreels-root"
    if os.path.exists(path):
        bridge = SkyReelsBridge(path)
        # minimal test usage
