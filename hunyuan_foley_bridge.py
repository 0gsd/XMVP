#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
from pathlib import Path

# Config
REPO_PATH = "/Volumes/XMVPX/mw/hunyuan-foley-code"
MODEL_PATH = "/Volumes/XMVPX/mw/hunyuan-foley" # Assuming weights here? Check listing.

class HunyuanFoleyBridge:
    def __init__(self, repo_path=REPO_PATH, model_path=MODEL_PATH):
        self.repo_path = Path(repo_path)
        self.model_path = Path(model_path)
        
        if not self.repo_path.exists():
            logging.error(f"❌ Hunyuan Repo not found: {self.repo_path}")
            
    def generate_foley(self, video_path, prompt, output_dir, model_size="xxl"):
        """
        Generates foley audio for a video file.
        Returns path to generated .wav file.
        """
        if not self.repo_path.exists(): return None
        
        # Output naming (infer.py likely creates default names)
        # We need to capture what it made.
        # "video_filename_foley.wav" usually?
        
        cmd = [
            sys.executable, 
            str(self.repo_path / "infer.py"),
            "--model_path", str(self.model_path),
            "--single_video", str(video_path),
            "--single_prompt", prompt,
            "--output_dir", str(output_dir),
            "--model_size", model_size
            # "--enable_offload" ?
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_path) + ":" + env.get("PYTHONPATH", "")
        
        logging.info(f"🔊 Generating Foley: {prompt[:30]}...")
        try:
            subprocess.run(cmd, check=True, env=env, cwd=str(self.repo_path), stdout=subprocess.DEVNULL)
            
            # Find output
            # Usually stem + "_foley.wav" or just in the dir?
            # Let's verify output.
            vid_stem = Path(video_path).stem
            # infer.py behavior: saves to output_dir / {vid_stem}.wav ?
            
            expected_output = Path(output_dir) / f"{vid_stem}.wav" 
            # Or checks infer.py code...
            
            if expected_output.exists():
                return str(expected_output)
                
            # Fallback check
            for f in Path(output_dir).glob("*.wav"):
                if vid_stem in f.name:
                    return str(f)
                    
            return None
            
        except Exception as e:
            logging.error(f"❌ Foley Generation Failed: {e}")
            return None

def generate_foley_asset(prompt, output_wav, video_path=None, duration=None):
    """
    Wrapper for Content Producer compatibility.
    """
    # Create Bridge
    bridge = HunyuanFoleyBridge()
    
    # Needs video path. If none, we can't do Foley (It's video-to-audio).
    if not video_path or not os.path.exists(video_path):
        logging.warning("Foley Engine requires video_path.")
        return None
        
    output_dir = os.path.dirname(output_wav)
    
    result = bridge.generate_foley(video_path, prompt, output_dir)
    
    if result and os.path.exists(result):
        # Rename to target output_wav if needed
        if os.path.abspath(result) != os.path.abspath(output_wav):
            if os.path.exists(output_wav): os.remove(output_wav)
            os.rename(result, output_wav)
        return output_wav
        
    return None

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    bridge = HunyuanFoleyBridge()
    # Dummy test if args provided?
