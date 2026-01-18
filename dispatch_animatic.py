#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import time
import torch
import gc
from pathlib import Path
from PIL import Image

# MVP Shared Logic
from mvp_shared import Manifest, load_manifest, save_manifest

# MLX for Director
try:
    import mlx_lm
except ImportError:
    logging.error("❌ mlx_lm not found. Install with: pip install mlx-lm")
    sys.exit(1)

# Flux Bridge
try:
    from flux_bridge import get_flux_bridge
except ImportError:
    logging.error("❌ flux_bridge not found.")
    sys.exit(1)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
# Resolution: 512x288 (16:9)
WIDTH = 512
HEIGHT = 288
DIRECTOR_MODEL_PATH = "mlx-community/gemma-2-9b-it-4bit"
ADAPTER_PATH = "adapters/director_v1"

class DirectorEngine:
    """Manages the Director Model (Gemma) via MLX."""
    def __init__(self, model_path, adapter_path):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load(self):
        if self.model: return
        
        logging.info(f"   🎬 Loading Director Model: {self.model_path}")
        if os.path.exists(self.adapter_path):
            logging.info(f"   ➕ With Adapter: {self.adapter_path}")
            # MLX Load with Adapter
            # Note: mlx_lm.load returns (model, tokenizer)
            self.model, self.tokenizer = mlx_lm.load(
                self.model_path, 
                adapter_path=self.adapter_path,
                tokenizer_config={"trust_remote_code": True}
            )
        else:
            logging.warning(f"   ⚠️ Adapter not found at {self.adapter_path}. Loading Base Model.")
            self.model, self.tokenizer = mlx_lm.load(self.model_path)
    
    def direct_shot(self, script_line, previous_shot_desc=None):
        """Generates a visual description from the script."""
        if not self.model: self.load()
        
        system_prompt = "You are a specialized screenplay visualization assistant. Your goal is to translate screenplay text into vivid, 'Fattened' visual descriptions aka SASSPRILLA Prompts."
        
        context = ""
        if previous_shot_desc:
            context = f"Previous Shot: {previous_shot_desc}\n"
            
        user_prompt = f"{context}Visualize this scene:\n\n{script_line}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Try standard template
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            # Fallback for models without System Role support (e.g. Gemma 2 via certain tokenizers)
            # Manual formatting:
            logging.warning(f"   ⚠️ Chat Template Error ({e}). Falling back to manual formatting.")
            full_prompt = (
                f"<start_of_turn>user\n"
                f"{system_prompt}\n\n{user_prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        text = mlx_lm.generate(
            self.model, 
            self.tokenizer, 
            prompt=full_prompt,
            max_tokens=256,
            verbose=False
        )
        
        # Parse? The model is trained to output "Visual: ... Action: ... Shot: ..."
        # But we just want the raw text for Flux mainly.
        return text.strip()

def run_animatic(manifest_path, out_path, staging_dir, flux_path):
    # 1. Load Manifest
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        logging.error(f"Failed to load manifest: {e}")
        return False

    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Init Flux
    logging.info(f"   🌊 Initializing Flux Bridge from {flux_path}...")
    flux = get_flux_bridge(flux_path)
    
    # 3. Init Director
    director = DirectorEngine(DIRECTOR_MODEL_PATH, ADAPTER_PATH)
    director.load()
    
    # 4. Loop
    logging.info(f"   🎞️ Starting Animatic Loop: {len(manifest.segs)} segments @ {WIDTH}x{HEIGHT}...")
    
    sorted_segs = sorted(manifest.segs, key=lambda s: s.id)
    last_visual = None
    last_image = None # For potential Img2Img flow
    
    for seg in sorted_segs:
        # Skip if done
        if seg.id in manifest.files and os.path.exists(manifest.files[seg.id]):
            logging.info(f"   ⏩ Seg {seg.id} done.")
            continue
            
        print(f"\n📍 SEGMENT {seg.id}")
        
        # Determine Duration
        # Writers Room typically sets duration_sec in Portion, but Seg might not have it strictly mapped?
        # mvp_shared.Seg doesn't have duration_sec explicit, but we can infer from frames or pass it.
        # Wait, Portion has duration_sec. Seg has start_frame/end_frame.
        # We can trust start/end frame if Portion Control calculated it correctly.
        duration_frames = seg.end_frame - seg.start_frame
        duration_sec = duration_frames / 24.0 # Assuming 24fps base
        
        print(f"   📜 Script: \"{seg.prompt[:100]}...\" ({duration_sec:.1f}s)")
        
        # A. Direct (Gemma)
        # We assume seg.prompt holds the SCRIPT LINE
        visual_prompt = director.direct_shot(seg.prompt, last_visual)
        print(f"   🎬 Director: \"{visual_prompt[:100]}...\"")
        
        # Update context
        last_visual = visual_prompt
        
        # B. Visualize (Flux Loop)
        # We will generate a sequence of frames at a "Draft FPS"
        DRAFT_FPS = 2 # Generate 2 images per second (Real-Time Render roughly)
        num_gen_frames = max(1, int(duration_sec * DRAFT_FPS))
        
        seg_staging = staging_path / f"seg_{seg.id:03d}"
        seg_staging.mkdir(exist_ok=True)
        
        frame_paths = []
        
        for i in range(num_gen_frames):
            frame_filename = f"frame_{i:04d}.png"
            frame_path = seg_staging / frame_filename
            
            # TODO: Img2Img for temporal coherence?
            # For now, pure T2I with same prompt = Jittery "Boiling" effect (Animatic style)
            # Maybe mix in a bit of last_image if available?
            
            steps = 4
            # Seed strategy: stable or varying? varying = boiling. stable = frozen.
            # Animatic usually wants frozen or slight boil.
            # Let's try varying seed every frame for "Alive" feel.
            
            img = flux.generate(
                prompt=visual_prompt,
                width=WIDTH,
                height=HEIGHT,
                steps=steps,
                seed=None # Random seed for boiling
            )
            
            if img:
                img.save(frame_path)
                frame_paths.append(str(frame_path))
            
            print(f"      Frame {i+1}/{num_gen_frames}", end="\r")
            
        print("")
        
        # C. Stitch into Video (mp4)
        # Use ffmpeg to hold each frame for (24/DRAFT_FPS) frames? or set output fps to DRAFT_FPS?
        # We want standard 24fps output file.
        
        out_video_name = f"animatic_seg_{seg.id:03d}.mp4"
        out_video_path = staging_path / out_video_name
        
        # FFMPEG concat
        # Create list file
        list_txt_path = seg_staging / "list.txt"
        with open(list_txt_path, 'w') as f:
            for fp in frame_paths:
                # Each image plays for 1/DRAFT_FPS seconds
                f.write(f"file '{fp}'\n")
                f.write(f"duration {1.0/DRAFT_FPS}\n")
            # Repeat last for safety?
            f.write(f"file '{frame_paths[-1]}'\n")
            
        import subprocess
        # Concat demuxer
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_txt_path),
            "-vf", "fps=24", # Force standard output
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_video_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"   ✅ Saved {out_video_name} ({duration_sec:.1f}s)")
            manifest.files[seg.id] = str(out_video_path)
        except Exception as e:
            logging.error(f"   ❌ FFMPEG Error: {e}")

            
    # 5. Save Manifest
    save_manifest(manifest, out_path)
    logging.info(f"🎉 Animatic Wrap! Saved to {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Dispatch Animatic: High-Speed Visualization Engine")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="manifest_updated.json")
    parser.add_argument("--staging", default="componentparts")
    
    # Resolving Flux Path (Hardcoded fallback or env?)
    # We'll try to get it from definitions if possible, or use the standard path
    # But definitions is in the same dir
    import definitions
    flux_conf = definitions.MODAL_REGISTRY[definitions.Modality.IMAGE].get("flux-schnell")
    fallback_path = flux_conf.path if flux_conf else "/Volumes/XMVPX/mw/flux-root"
    
    parser.add_argument("--flux_path", default=fallback_path)
    
    args = parser.parse_args()
    
    success = run_animatic(args.manifest, args.out, args.staging, args.flux_path)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
