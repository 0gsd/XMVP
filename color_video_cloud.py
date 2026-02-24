import os
import io
import time
import logging
import subprocess
import random
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class ColorVideoCloudBridge:
    def __init__(self, api_keys=None):
        self.api_keys = api_keys if isinstance(api_keys, list) else ([api_keys] if api_keys else [])
        if not self.api_keys:
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key:
                self.api_keys = [env_key]
        
        self.key_index = 0
        self.client = None
        self._rotate_client()

    def _rotate_client(self):
        if not self.api_keys:
            logging.warning("⚠️ No Gemini API keys provided for Cloud Colorization.")
            return False
        
        current_key = self.api_keys[self.key_index % len(self.api_keys)]
        self.client = genai.Client(api_key=current_key)
        self.key_index += 1
        return True

    def colorize_frame(self, image_path, prompt, attempt=0):
        """
        Sends a single B&W frame to Gemini for colorization.
        """
        if not self.client:
            return None

        try:
            # 1. Prepare Image
            img = Image.open(image_path).convert("RGB")
            # Gemini 2.5 Flash Image works well around 1K, but we'll maintain input aspect
            img.thumbnail((1024, 1024)) 
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            
            contents = [
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                f"SOURCE: The attached image is a black and white film frame.\n"
                f"TASK: Colorize this image with high fidelity and natural, cinematic tones. {prompt}\n"
                f"CRITICAL: Maintain the exact structure and lighting of the source. No text or artifacts."
            ]

            # 2. Call Gemini
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(image_size="1K")
                )
            )

            # 3. Extract Result
            for part in response.parts:
                if part.inline_data:
                    return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                elif image_wrapper := part.as_image():
                    return Image.open(io.BytesIO(image_wrapper.data)).convert("RGB")
            
            return None

        except Exception as e:
            if "429" in str(e):
                self._rotate_client()
                wait = (2 ** attempt) + random.random()
                logging.warning(f"   ⏳ Rate limited (429). Rotated Key. Waiting {wait:.1f}s...")
                time.sleep(wait)
                if attempt < 5:
                    return self.colorize_frame(image_path, prompt, attempt + 1)
            else:
                logging.error(f"   ❌ Gemini Error: {e}")
            return None

    def process(self, args, output_root):
        """
        Main entry point for color-video VPForm using Gemini Cloud.
        """
        if not args.mu or not os.path.exists(args.mu):
             logging.error("❌ color-video requires input video (pass as arg or --mu)")
             return False

        logging.info(f"   ☁️  Gemini Cloud Colorize: Processing {Path(args.mu).name}...")
        
        # 0. Setup Dirs
        project_name = Path(args.mu).stem
        project_dir = Path(output_root) / f"{project_name}_GeminiColor"
        frames_dir = project_dir / "frames"
        source_dir = project_dir / "source_frames"
        ensure_dir(frames_dir)
        ensure_dir(source_dir)

        # 1. Extraction (Audio)
        audio_path = project_dir / "extracted_audio.wav"
        if not audio_path.exists():
            logging.info("   🔊 Extracting audio...")
            cmd_a = ['ffmpeg', '-y', '-i', args.mu, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)]
            subprocess.run(cmd_a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Extraction (Frames)
        project_fps = args.fps if args.fps else 4
        logging.info(f"   🎞️  Extracting source frames @ {project_fps} FPS...")
        cmd_f = ['ffmpeg', '-y', '-i', args.mu, '-vf', f'fps={project_fps}', str(source_dir / 'source_%04d.png')]
        subprocess.run(cmd_f, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Resolution Probing
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', args.mu]
            probe_out = subprocess.check_output(probe_cmd, text=True).strip()
            vid_w, vid_h = [int(x) for x in probe_out.split(',')]
            logging.info(f"   📐 Input video: {vid_w}×{vid_h}")
        except:
            vid_w, vid_h = 1280, 720

        target_w = args.w if args.w else vid_w
        target_h = args.h if args.h else vid_h

        # 4. Colorization Loop
        source_frames = sorted(list(source_dir.glob("source_*.png")))
        if getattr(args, 'limit', None):
            source_frames = source_frames[:args.limit]

        prompt = args.prompt if args.prompt else "natural skin tones, period accurate colors"

        for i, src_path in enumerate(source_frames):
            idx = i + 1
            dst = frames_dir / f"frame_{idx:04d}.png"
            if dst.exists(): continue

            logging.info(f"   ✨ Colorizing {idx}/{len(source_frames)} via Gemini...")
            colorized = self.colorize_frame(src_path, prompt)
            
            if colorized:
                if colorized.size != (target_w, target_h):
                    colorized = colorized.resize((target_w, target_h), Image.Resampling.LANCZOS)
                colorized.save(dst)
                # Small delay to keep quota stable
                time.sleep(1.0) 
            else:
                logging.warning(f"   ⚠️ Frame {idx} failed. Using grayscale fallback.")
                grayscale = Image.open(src_path).convert("RGB")
                if grayscale.size != (target_w, target_h):
                    grayscale = grayscale.resize((target_w, target_h), Image.Resampling.LANCZOS)
                grayscale.save(dst)

        # 5. Assembly
        logging.info("   🧵 stitching...")
        out_vid = project_dir / "final_cloud_output.mp4"
        cmd_s = [
            'ffmpeg', '-y', '-framerate', str(project_fps),
            '-i', str(frames_dir / 'frame_%04d.png'),
            '-i', str(audio_path),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k', '-shortest',
            str(out_vid)
        ]
        subprocess.run(cmd_s, check=True)
        logging.info(f"   ✅ Done: {out_vid}")
        return True
