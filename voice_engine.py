#!/usr/bin/env python3
# voice_engine.py
# Abstraction for TTS and RVC-based Voice Conversion.
# Uses macOS 'say' command for base audio (Free) and RVC for skinning.

import os
import subprocess
import random
import time

class VoiceEngine:
    def __init__(self, weights_dir="/Volumes/ORICO/weightsquared/weights"):
        self.weights_dir = weights_dir
        self.system_voices = ["Alex", "Fred", "Samantha", "Victoria"] # Basic macOS voices
        self.has_warned_missing_lib = False
        
    def generate_base_audio(self, text, output_path, gender="MALE"):
        """Generates base audio using macOS 'say' command."""
        # Simple mapping for base voice
        voice = "Alex" if gender == "MALE" else "Samantha"
        
        # We can add variety if needed, but consistency is better for RVC source.
        # Ideally, we want a flat, clean voice.
        
        try:
            cmd = ['say', '-v', voice, '-o', output_path, '--data-format=LEF32@24000', text]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[-] TTS Error (macOS say): {e}")
            return False
        except Exception as e:
             print(f"[-] TTS exception: {e}")
             return False

    def apply_rvc(self, input_audio, output_audio, model_name, pitch_shift=0):
        """
        Applies RVC voice conversion using 'rvc-python' if available.
        """
        # Search for model path in all potential dirs
        potential_dirs = [
            self.weights_dir, 
            os.path.join(self.weights_dir, "male"), 
            os.path.join(self.weights_dir, "female"), 
            os.path.join(self.weights_dir, "_unsorted")
        ]
        
        model_path = None
        index_path = None
        
        for p_dir in potential_dirs:
            # Check Folder-based model
            m_dir = os.path.join(p_dir, model_name)
            if os.path.exists(m_dir) and os.path.isdir(m_dir):
                # Found directory, look for pth inside
                pths = [f for f in os.listdir(m_dir) if f.endswith(".pth") and not f.startswith("G_") and not f.startswith("D_")]
                if pths:
                    model_path = os.path.join(m_dir, pths[0])
                    # Look for index in same dir
                    idxs = [f for f in os.listdir(m_dir) if f.endswith(".index")]
                    if idxs: index_path = os.path.join(m_dir, idxs[0])
                    break
                    
            # Check Flat file match
            m_file = os.path.join(p_dir, f"{model_name}.pth")
            if os.path.exists(m_file):
                model_path = m_file
                # Look for index next to it (maybe named similarly?)
                # Try simple wildcard search in that dir
                idxs = [f for f in os.listdir(p_dir) if f.endswith(".index") and model_name in f]
                if idxs: index_path = os.path.join(p_dir, idxs[0])
                break
        
        if not model_path:
            # print(f"[-] RVC Model missing: {model_name}")
            subprocess.run(['cp', input_audio, output_audio], check=True)
            return True

        try:
            # 1. Try using rvc-python library
            from rvc_python.infer import RVCInference
            
            print(f"    [RVC] Inferring with {model_name}...")
            rvc = RVCInference(device="mps:0" if subprocess.run(["uname", "-m"], capture_output=True).stdout.strip() == b"arm64" else "cpu")
            
            rvc.infer_file(
                input_path=input_audio,
                output_path=output_audio,
                model_path=model_path,
                index_path=index_path,
                f0_method="rmvpe", # Best quality usually
                f0_up_key=pitch_shift,
                index_rate=0.75,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.25,
                protect=0.33
            )
            return True
            
        except ImportError:
            if not self.has_warned_missing_lib:
                print("[-] 'rvc-python' lib not found. Using Mock (Passthrough).") 
                print("    -> pip install rvc-python")
                self.has_warned_missing_lib = True
                
            subprocess.run(['cp', input_audio, output_audio], check=True)
            return True
        except Exception as e:
            print(f"[-] RVC Inference Failed: {e}")
            # Fallback
            subprocess.run(['cp', input_audio, output_audio], check=True)
            return True

    def assign_voice(self, actor_name, gender="MALE"):
        """
        Scans weights directory for a matching model.
        Strategy:
        1. Exact Name Match (e.g. "Joe_Pantoliano" folder or file)
        2. Gender-Based Pool: Look in `male` or `female` subfolders.
        3. Consistent Hash: Pick from the pool.
        """
        # 1. Normalize name for filesystem
        safe_name = actor_name.replace(" ", "_").replace(".", "").strip()
        
        # Check direct match (Folder or File) anywhere in root or subfolders?
        # Let's check root first, then gender folders.
        potential_paths = [
            self.weights_dir,
            os.path.join(self.weights_dir, "male"),
            os.path.join(self.weights_dir, "female"),
            os.path.join(self.weights_dir, "_unsorted")
        ]
        
        for p_dir in potential_paths:
            if not os.path.exists(p_dir): continue
            
            # Folder match
            c_dir = os.path.join(p_dir, safe_name)
            if os.path.exists(c_dir) and os.path.isdir(c_dir):
                return safe_name 
            
            # File match
            c_file = os.path.join(p_dir, f"{safe_name}.pth")
            if os.path.exists(c_file):
                return safe_name

        # 2. Build Pool based on Gender
        # If gender is specified, prefer that folder.
        # Fallback to _unsorted if pool is empty?
        
        target_subdirs = []
        if gender and gender.upper() == "FEMALE":
            target_subdirs = ["female", "_unsorted"]
        else:
            target_subdirs = ["male", "_unsorted"] # Default to male/unsorted
            
        available_models = []
        system_prefixes = ["G", "D", "f0", "hubert", "rmvpe", "fcpe", "rvc", "UVR"]
        
        for sub in target_subdirs:
            search_path = os.path.join(self.weights_dir, sub)
            if not os.path.exists(search_path): continue
            
            try:
                raw_list = os.listdir(search_path)
                for f in raw_list:
                    if any(f.startswith(p) for p in system_prefixes): continue
                    
                    full_p = os.path.join(search_path, f)
                    if os.path.isdir(full_p):
                        if not f.startswith("."): available_models.append(f)
                    elif f.endswith(".pth"):
                        available_models.append(os.path.splitext(f)[0])
            except:
                continue
                
        # Deduplicate
        available_models = sorted(list(set(available_models)))
        
        if not available_models:
            # Try ALL folders if specific gender failed
            # ... implementation omitted for brevity, usually _unsorted covers it.
            return None
            
        # 3. Consistent Hash
        chk = sum(ord(c) for c in actor_name)
        idx = chk % len(available_models)
        assigned_model = available_models[idx]
        
        print(f"       [+] VoiceEngine Assigned: {actor_name} ({gender}) -> {assigned_model}")
        return assigned_model

if __name__ == "__main__":
    # Test
    ve = VoiceEngine()
    ve.generate_base_audio("Hello, this is a test of the emergency broadcast system.", "test_base.wav", "MALE")
    ve.apply_rvc("test_base.wav", "test_rvc.wav", "SomeModel")
