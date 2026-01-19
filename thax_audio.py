
import os
import logging
import subprocess
from pathlib import Path
from kokoro_bridge import get_kokoro_bridge

# Configuration
# Path to the Thax RVC Model (User must train this)
# 1. Env Var -> 2. Relative ./models/thax -> 3. Hardcoded Local Fallback
DEFAULT_MODEL_DIR = Path("z_training_data/thax_voice/model")
MODEL_DIR = Path(os.getenv("THAX_MODEL_DIR", DEFAULT_MODEL_DIR))
MODEL_NAME = "thax.pth"
INDEX_NAME = "thax.index"

# Path to Kokoro (Base TTS)
# 1. Env Var -> 2. Hardcoded Local Fallback
DEFAULT_KOKORO = "/Volumes/XMVPX/mw/kokoro-root/kokoro-v0_19.onnx"
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", DEFAULT_KOKORO)

# RVC Python Path (In separate environment)
# 1. Env Var -> 2. Fallback to standard miniconda location
DEFAULT_RVC_BIN = Path.home() / "miniconda3/envs/rvc_env/bin/python"
RVC_PYTHON_BIN = Path(os.getenv("RVC_PYTHON_BIN", DEFAULT_RVC_BIN))

class ThaxVoiceEngine:
    def __init__(self):
        self.bridge = get_kokoro_bridge(KOKORO_MODEL_PATH)
        self.model_path = MODEL_DIR / MODEL_NAME
        self.index_path = MODEL_DIR / INDEX_NAME
        
    def generate(self, text, output_path):
        """
        Generates Thax Douglas audio:
        1. Kokoro TTS (af_bella or am_michael) -> Temp WAV
        2. RVC Inference -> Final WAV
        """
        # 1. Base Audio (Kokoro)
        temp_path = str(Path(output_path).parent / f"temp_{Path(output_path).name}")
        
        # Use a neutral/male voice. am_michael is standard.
        success = self.bridge.generate(text, temp_path, voice_name="am_michael", speed=0.9) # Slightly slower for poetic effect
        
        if not success:
            logging.error("   ❌ Kokoro Base Gen Failed.")
            return False
            
        # 2. Check if RVC Model Exists
        if not self.model_path.exists():
            logging.warning(f"   ⚠️ Thax Model not found at {self.model_path}. Using untransformed base audio.")
            # Rename temp to final
            os.rename(temp_path, output_path)
            return True
            
        # 3. RVC Inference (Shell out to rvc_env)
        # We use a python one-liner or simple script wrapper
        logging.info(f"   🎤 Converting to Thax Voice (RVC)...")
        
        rvc_cmd = [
            str(RVC_PYTHON_BIN),
            "-c",
            f"""
import os
import sys
import torch

# MONKEYPATCH: Fix PyTorch 2.6+ 'weights_only=True' breaking legacy checkpoints
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

from rvc_python.infer import RVCInference

try:
    rvc = RVCInference(device="mps") # Force MPS for Mac Studio
    # Try explicit V2 load first (Most common for Applio)
    try:
        rvc.load_model("{str(self.model_path)}", version="v2")
    except Exception as e_v2:
        print(f"WARN: V2 Load failed ({{e_v2}}). Trying auto/v1...")
        rvc.load_model("{str(self.model_path)}")

    # MANUAL INFERENCE (Bypassing infer_file wrapper which crashes on tuple)
    print("   running Manual Inference Logic...")
    try:
        # Disable Index for Debugging (Empty strings)
        # Also print args
        print(f"      Calling vc_single with temp_path={temp_path}")
        wav_opt = rvc.vc.vc_single(
            0, # sid
            "{temp_path}", # input path
            0, # f0_up_key
            None, # f0_file
            "rmvpe", # f0_method / algo
            "", # file_index (DISABLED)
            "", # file_index2
            0, # index_rate (DISABLED)
            3, # filter_radius
            0, # resample_sr
            0.25, # rms_mix_rate
            0.33 # protect
        )
    except Exception as e_inf:
        print(f"      ❌ vc_single Crashed: {{e_inf}}")
        print(f"      Traceback:")
        import traceback
        traceback.print_exc()
        raise e_inf
    
    # Handle Tuple Return (sr, data) or (data)
    if isinstance(wav_opt, tuple):
        tgt_sr, audio_data = wav_opt
    else:
        # Assuming defaults?
        tgt_sr = rvc.vc.tgt_sr
        audio_data = wav_opt

    from scipy.io import wavfile
    wavfile.write("{output_path}", tgt_sr, audio_data)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
            """
        ]
        
        try:
            result = subprocess.run(rvc_cmd, capture_output=True, text=True)
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                 logging.info("   ✅ Thax RVC Conversion Complete.")
                 # Cleanup temp
                 if os.path.exists(temp_path): os.remove(temp_path)
                 return True
            else:
                 logging.error(f"   ❌ RVC Conversion Failed: {result.stderr}")
                 logging.info("   ⚠️ Falling back to base audio.")
                 if os.path.exists(temp_path):
                     if os.path.exists(output_path): os.remove(output_path)
                     os.rename(temp_path, output_path)
                 return True # Return true as we have *some* audio
                 
        except Exception as e:
            logging.error(f"   ❌ RVC Subprocess Error: {e}")
            return False

# Singleton
_THAX_ENGINE = None
def get_thax_engine():
    global _THAX_ENGINE
    if _THAX_ENGINE is None:
        _THAX_ENGINE = ThaxVoiceEngine()
    return _THAX_ENGINE
