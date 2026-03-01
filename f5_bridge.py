#!/usr/bin/env python3
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

# Placeholder for F5-TTS library imports
# The user hasn't installed f5-tts yet, so we'll wrap imports in try-except
F5_AVAILABLE = False
try:
    # In the pip-installed version (1.1.x), the F5TTS class is the way to go
    from f5_tts.api import F5TTS
    F5_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ F5-TTS libraries not found. Please install via pip: pip install f5-tts")

class F5TTSBridge:
    def __init__(self, model_dir, device=None):
        self.model_dir = Path(model_dir)
        self.ckpt_path = self.model_dir / "F5TTS_v1_Base" / "model_1250000.safetensors"
        self.vocab_path = self.model_dir / "F5TTS_v1_Base" / "vocab.txt"
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.f5_api = None
        
    def load(self):
        if not F5_AVAILABLE:
            raise ImportError("F5-TTS libraries not found.")
            
        if self.f5_api: return

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"F5-TTS checkpoint not found at: {self.ckpt_path}")

        logging.info(f"   🚀 Loading F5-TTS from {self.ckpt_path} on {self.device}...")
        try:
            self.f5_api = F5TTS(
                ckpt_file=str(self.ckpt_path),
                vocab_file=str(self.vocab_path),
                device=self.device
            )
            logging.info(f"   ✅ F5-TTS Ready.")
            
        except Exception as e:
            logging.error(f"   ❌ Failed to load F5-TTS: {e}")
            raise e

    def generate(self, text, output_path, ref_audio=None, ref_text="", speed=1.0):
        """
        Generates audio using F5-TTS.
        ref_audio: path to a short (~5-15s) reference audio file for voice cloning.
        ref_text: the transcript of the reference audio.
        """
        if not self.f5_api: self.load()
        
        logging.info(f"   🗣️ F5-TTS Speaking: '{text[:30]}...'")
        
        try:
            if not ref_audio:
                ref_audio = str(self.model_dir / "sample.wav")
                if not os.path.exists(ref_audio):
                    # If no sample.wav, F5TTS might still work if it has internal defaults or we need to provide one.
                    # The F5TTS.infer method requires ref_audio.
                    logging.warning("   ⚠️ No reference audio provided and default sample.wav not found.")
            
            # F5TTS.infer returns (pcm, sr, spectrogram)
            audio, sr, _ = self.f5_api.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                speed=speed
            )
            
            # Save to file
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            sf.write(output_path, audio, sr)
            return True
            
        except Exception as e:
            logging.error(f"   ❌ F5-TTS Generation Error: {e}")
            return False

# Singleton
_BRIDGE = None
def get_f5_bridge(model_dir):
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = F5TTSBridge(model_dir)
    return _BRIDGE
