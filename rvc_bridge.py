import os
import subprocess
import logging
from pathlib import Path

# Path to the isolated RVC Python environment
DEFAULT_RVC_BIN = str(Path.home() / "miniconda3/envs/rvc_env/bin/python")

class RVCBridge:
    def __init__(self, rvc_python_bin=DEFAULT_RVC_BIN):
        self.rvc_bin = rvc_python_bin

    def generate(self, input_wav, model_dir, output_wav=None, pitch_shift=0):
        """
        Runs RVC Inference over an existing WAV file.
        `model_dir` is expected to contain a .pth file and optionally an .index file.
        """
        if not output_wav:
             output_wav = input_wav
             
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            logging.error(f"   ❌ RVC Model Directory not found: {model_dir}")
            return False
            
        # Find .pth and .index
        pth_files = list(model_dir_path.glob("*.pth"))
        index_files = list(model_dir_path.glob("*.index"))
        
        if not pth_files:
            logging.error(f"   ❌ No .pth model found in {model_dir}")
            return False
            
        pth_file = str(pth_files[0])
        index_file = str(index_files[0]) if index_files else ""
        
        temp_out = input_wav.replace(".wav", "_rvc_temp.wav")
        
        logging.info(f"   🎤 Converting Voice via RVC (Model: {Path(pth_file).name})...")
        
        rvc_cmd = [
            self.rvc_bin, "-c",
            f"""
import os, sys, torch
import traceback

# Monkeypatch PyTorch 2.6+ to allow loading older checkpoints
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

from rvc_python.infer import RVCInference
try:
    rvc = RVCInference(device="mps")
    try:
        rvc.load_model("{pth_file}", version="v2")
    except Exception as e1:
        print(f"WARN: V2 load failed ({{e1}}). Trying V1...")
        rvc.load_model("{pth_file}")
        
    wav_opt = rvc.vc.vc_single(
        0, "{input_wav}", {pitch_shift}, None, "rmvpe", 
        "{index_file}", "", 0, 3, 0, 0.25, 0.33
    )
    
    if isinstance(wav_opt, tuple):
        tgt_sr, audio_data = wav_opt
    else: 
        tgt_sr, audio_data = rvc.vc.tgt_sr, wav_opt

    from scipy.io import wavfile
    wavfile.write("{temp_out}", tgt_sr, audio_data)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        ]
        
        try:
            res = subprocess.run(rvc_cmd, capture_output=True, text=True)
            if "SUCCESS" in res.stdout:
                if os.path.exists(temp_out):
                    os.replace(temp_out, output_wav)
                logging.info("   ✅ RVC Conversion Complete.")
                return True
            else:
                err_msg = res.stderr.strip() if res.stderr else res.stdout.strip()
                logging.error(f"   ❌ RVC Failed: {err_msg}")
                # Clean up temp
                if os.path.exists(temp_out):
                     os.remove(temp_out)
                return False
        except Exception as e:
            logging.error(f"   ❌ RVC Exec Failed: {e}")
            return False

# Singleton
_BRIDGE = None
def get_rvc_bridge():
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = RVCBridge()
    return _BRIDGE
