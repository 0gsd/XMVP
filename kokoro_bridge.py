#!/usr/bin/env python3
import os
import json
import logging
import soundfile as sf
import numpy as np
from pathlib import Path

# Try to import kokoro_onnx, but don't crash if missing (for now)
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logging.warning("⚠️ kokoro-onnx not installed. Please run: pip install kokoro-onnx soundfile")

class KokoroBridge:
    def __init__(self, model_path, voices_path=None):
        self.model_path = model_path
        # If voices_path not provided, assume voices.json is in same dir as model
        if not voices_path:
            # Prefer .npz if exists
            npz_path = Path(model_path).parent / "voices.npz"
            if npz_path.exists():
                self.voices_path = str(npz_path)
            else:
                self.voices_path = str(Path(model_path).parent / "voices.json")
        else:
            self.voices_path = voices_path
            
        self.kokoro = None
        self.voices = {}
        
    def load(self):
        if not KOKORO_AVAILABLE:
            raise ImportError("kokoro-onnx library not found.")
            
        if self.kokoro: return

        if not os.path.exists(self.model_path):
             raise FileNotFoundError(f"Kokoro model not found at: {self.model_path}")

        if not os.path.exists(self.voices_path):
             raise FileNotFoundError(f"Kokoro voices not found at: {self.voices_path}")

        logging.info(f"   🦜 Loading Kokoro TTS from {self.model_path}...")
        try:
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            
            # Load voice definitions for introspection if needed
            if self.voices_path.endswith(".json"):
                with open(self.voices_path, 'r') as f:
                    self.voices = json.load(f)
            elif self.voices_path.endswith(".npz"):
                # Load keys from npz
                data = np.load(self.voices_path)
                self.voices = {k: "vector" for k in data.files}
                data.close()
                
            logging.info(f"   ✅ Kokoro Ready. Voices: {list(self.voices.keys())}")
        except Exception as e:
            logging.error(f"   ❌ Failed to load Kokoro: {e}")
            raise e

    def generate(self, text, output_path, voice_name="af_bella", speed=1.0):
        if not self.kokoro: self.load()
        
        # Fallback for unknown voice
        if voice_name not in self.voices:
            # Try to match gender prefix?
            if voice_name.startswith("af"): voice_name = "af_bella"
            elif voice_name.startswith("am"): voice_name = "am_michael"
            elif voice_name.startswith("bf"): voice_name = "bf_emma"
            elif voice_name.startswith("bm"): voice_name = "bm_george"
            else:
                logging.warning(f"   ⚠️ Voice '{voice_name}' not found. Using default 'af_bella'.")
                voice_name = "af_bella"

        logging.info(f"   🗣️ Kokoro Speaking ({voice_name}): '{text[:30]}...'")
        
        try:
            # Generate audio (returns numpy array, sample_rate)
            samples, sample_rate = self.kokoro.create(
                text, 
                voice=voice_name, 
                speed=speed, 
                lang="en-us"
            )
            
            # Save to file
            sf.write(output_path, samples, sample_rate)
            return True
            
        except Exception as e:
            logging.error(f"   ❌ Kokoro Generation Error: {e}")
            return False

    def get_voice_list(self):
        if not self.voices and os.path.exists(self.voices_path):
             if self.voices_path.endswith(".json"):
                 with open(self.voices_path, 'r') as f:
                    self.voices = json.load(f)
             elif self.voices_path.endswith(".npz"):
                 data = np.load(self.voices_path)
                 self.voices = {k: "vector" for k in data.files}
                 data.close()
        return list(self.voices.keys())

# Singleton
_BRIDGE = None
def get_kokoro_bridge(model_path):
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = KokoroBridge(model_path)
    return _BRIDGE

if __name__ == "__main__":
    # Test
    # Assuming standard install location
    path = "/Volumes/XMVPX/mw/kokoro-root/kokoro-v0_19.onnx"
    if os.path.exists(path):
        bridge = KokoroBridge(path)
        bridge.generate("Hello world, this is a test of the local Emergency Broadcast System.", "test_kokoro.wav")
