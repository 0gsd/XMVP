#!/usr/bin/env python3
"""
sfx_bridge.py
-------------
Bridge for generating Sound Effects (SFX) and Music.
Tries to use AudioCraft (MusicGen/AudioGen) first.
Falls back to Stable Audio Open (via Diffusers) if AudioCraft is missing.
"""

import os
import sys
import logging
import torch
import soundfile as sf
import random
import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class SFXGenerator:
    def __init__(self):
        self.backend = None
        self.model = None
        self.processor = None
        self._init_backend()

        return
            
    def _init_backend(self):
        """Attempts to load Stable Audio (Diffusers) first, then AudioCraft."""
        
        # 1. Try Stable Audio Open (Diffusers) - User Preferred
        try:
            from diffusers import StableAudioPipeline
            self.backend = "stable_audio"
            logging.info("   🎵 SFX Bridge: Using Stable Audio Open (Diffusers).")
            return
        except ImportError as e:
            logging.warning(f"   ⚠️ SFX Bridge: Diffusers not found/error: {e}. Trying AudioCraft...")

        # 2. Try AudioCraft
        try:
            import audiocraft
            from audiocraft.models import MusicGen, AudioGen
            self.backend = "audiocraft"
            logging.info("   🎵 SFX Bridge: AudioCraft detected.")
            # We load models lazily to save vram
            return
        except ImportError:
            logging.error("   ❌ SFX Bridge: AudioCraft not found.")
            self.backend = "none"

    def _load_audiocraft(self, model_type="facebook/musicgen-small"):
        """Loads AudioCraft model on demand."""
        from audiocraft.models import MusicGen, AudioGen
        
        if self.model: return

        logging.info(f"   LOADING AUDIOCRAFT: {model_type}...")
        try:
            if "musicgen" in model_type:
                self.model = MusicGen.get_pretrained(model_type)
            else:
                self.model = AudioGen.get_pretrained(model_type)
            
            self.model.set_generation_params(duration=10) # Default, override later
        except Exception as e:
            logging.error(f"   [-] Failed to load AudioCraft: {e}")
            self.backend = "stable_audio" # Fallback immediately
            self._load_stable_audio()

    def _load_stable_audio(self):
        """Loads Stable Audio Pipeline."""
        if self.model and self.backend == "stable_audio": return
        
        from diffusers import StableAudioPipeline, EulerDiscreteScheduler
        
        model_id = "stabilityai/stable-audio-open-1.0"
        logging.info(f"   LOADING STABLE AUDIO: {model_id}...")
        
        # FIX: Force CPU to avoid MPS Contention and Recursion Errors
        # User requested to always process SFX on CPU to free up GPU for other tasks.
        self.sfx_device = "cpu"
        
        try:
            self.model = StableAudioPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float32 # CPU uses float32
            )
            # SCHEDULER SWAP: Avoid torchsde RecursionError on MPS by using EulerDiscrete (ODE)
            self.model.scheduler = EulerDiscreteScheduler.from_config(self.model.scheduler.config)
            
            self.model.to(self.sfx_device)
        except Exception as e:
            logging.error(f"   [-] Stable Audio Load Failed: {e}")
            self.backend = "none"

    def generate(self, prompt, output_path, duration=5.0, type="sfx"):
        """
        Generates audio based on prompt.
        type: 'sfx' or 'music' (hints for AudioCraft model selection)
        """
        if self.backend == "none":
            logging.warning("   [-] No SFX backend available (using dummy).")
            return self._generate_dummy(duration, output_path)

        logging.info(f"   🎧 Generating {type.upper()}: '{prompt}' ({duration}s)...")

        # --- AUDIOCRAFT ---
        if self.backend == "audiocraft":
            try:
                # Select Model
                target_model = "facebook/audiogen-medium" if type == "sfx" else "facebook/musicgen-small"
                # If we have a model loaded but it's the wrong one, we might need to swap?
                # For simplicity, let's stick to MusicGen Small for everything if we want speed,
                # or AudioGen for SFX. Implementation complexity: Swapping.
                # Let's try to use MusicGen for everything for now as it's more versatile, 
                # or just failover to whatever is loaded.
                
                # Logic: If type is SFX, use AudioGen. If Music, MusicGen.
                self._load_audiocraft(target_model)
                
                self.model.set_generation_params(duration=duration)
                wav = self.model.generate([prompt], progress=True)
                
                # wav is [batch, channels, time]
                # Save
                import torchaudio
                torchaudio.save(output_path, wav[0].cpu(), self.model.sample_rate)
                return True

            except Exception as e:
                logging.error(f"   [-] AudioCraft Gen Failed: {e}. Trying fallback...")
                self.backend = "stable_audio"
                self.model = None # Force reload
                # Fallthrough to Stable Audio

        # --- STABLE AUDIO ---
        if self.backend == "stable_audio":
            try:
                self._load_stable_audio()
                if not self.model: return False
                
                # Seed
                seed = random.randint(0, 2**32-1)
                generator = torch.Generator(self.sfx_device).manual_seed(seed)
                
                # Generate directly on DEVICE (MPS/CUDA) using Euler scheduler
                audio_tuple = self.model(
                    prompt,
                    negative_prompt="low quality, silence, distorted",
                    num_inference_steps=50, 
                    audio_end_in_s=duration,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                    return_dict=False
                )
                
                # Tuple is (audios,)
                audio = audio_tuple[0]
                
                # Save
                output = audio[0].float().cpu().numpy().T
                sf.write(output_path, output, self.model.vae.sampling_rate)
                return True
                
            except Exception as e:
                logging.error(f"   [-] Stable Audio Gen Failed: {e}")
                # Fallback to Dummy
                return self._generate_dummy(duration, output_path)

        # --- DUMMY/NONE ---
        logging.warning(f"   ⚠️ SFX Backend logic exhausted. generating silent placeholder.")
        return self._generate_dummy(duration, output_path)

    def _generate_dummy(self, duration, output_path):
        """Generates 1 second of silence/metadata to satisfy pipeline."""
        try:
            sr = 32000
            # Generate silence
            data = np.zeros(int(duration * sr))
            sf.write(output_path, data, sr)
            logging.info(f"   ⚠️ Generated Dummy SFX (Silence): {output_path}")
            return True
        except Exception as e:
            logging.error(f"   ❌ Dummy Gen Failed: {e}")
            return False

# Singleton
_GENERATOR = None

def get_sfx_generator():
    global _GENERATOR
    if not _GENERATOR:
        _GENERATOR = SFXGenerator()
    return _GENERATOR

def generate_sfx(prompt, output_path, duration=5.0, type="sfx"):
    """Global helper."""
    gen = get_sfx_generator()
    return gen.generate(prompt, output_path, duration, type)

if __name__ == "__main__":
    # Test
    if len(sys.argv) > 2:
        prompt = sys.argv[1]
        out = sys.argv[2]
        generate_sfx(prompt, out, 5.0)
