#!/usr/bin/env python3
"""
populate_models_xmvp.py
-----------------------
Downloads selected local models for the XMVP pipeline.
Requires:
- --f: Root folder where all subfolders go.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def ensure_library():
    try:
        import huggingface_hub
    except ImportError:
        print("[-] Installing huggingface_hub, mflux, and hf_transfer...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "mflux", "hf_transfer"], check=True)

def main():
    parser = argparse.ArgumentParser(description="XMVP Model Populator")
    parser.add_argument("--f", type=str, required=True, help="Target root directory for models")
    args = parser.parse_args()

    MW_ROOT = Path(args.f).resolve()
    HF_CACHE = MW_ROOT / "huggingface-root"

    # Ensure Environment for Cache
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    MODELS = {
        "Gemma 4": {
            "repo": "google/gemma-4-E4B-it",
            "type": "snapshot",
            "target": MW_ROOT / "gemma-root"
        },
        "Gemma 3": {
            "repo": "mlx-community/gemma-2-9b-it-4bit",
            "type": "snapshot",
            "target": MW_ROOT / "gemma-3-root"
        },
        "Flux-Dev": {
            "repo": "black-forest-labs/FLUX.2-dev",
            "type": "file",
            "filename": "flux2-dev.safetensors",
            "target": MW_ROOT / "flux-root" / "dev"
        },
        "MFLUX-Klein-4B": {
            "repo": "black-forest-labs/FLUX.2-klein-4B",
            "type": "snapshot",
            "target": MW_ROOT / "mflux-root" / "klein-4b"
        },
        "F5-TTS": {
            "repo": "SWivid/F5-TTS",
            "type": "files",
            "filenames": [
                "F5TTS_v1_Base/model_1250000.safetensors",
                "F5TTS_v1_Base/vocab.txt"
            ],
            "target": MW_ROOT / "f5tts-root"
        },
        "Kokoro": {
            "repo": "hexgrad/Kokoro-82M",
            "type": "snapshot",
            "target": MW_ROOT / "kokoro-root"
        },
        "RVC": {
            "repo": "lj1995/VoiceConversionWebUI",
            "type": "files",
            "filenames": ["hubert_base.pt", "rmvpe.pt"],
            "target": MW_ROOT / "rvc-root"
        },
        "Stable-Audio-Open": {
            "repo": "stabilityai/stable-audio-open-1.0",
            "type": "snapshot",
            "target": MW_ROOT / "stable-audio-root"
        }
    }

    print(f"🚀 XMVP Model Populator")
    print(f"   Root: {MW_ROOT}")
    print(f"   Cache: {HF_CACHE}")

    print("\n--- Installation Plan ---")
    print(f"This will install the following models' weights files in subfolders in {MW_ROOT}:")
    for name, conf in MODELS.items():
        print(f"  - {name} ({conf['repo']}) -> {conf['target']}")
    
    confirm = input("\ncontinue? y/n: ").strip().lower()
    if confirm != 'y':
        print("Aborted by user.")
        sys.exit(0)

    ensure_library()
    from huggingface_hub import hf_hub_download, snapshot_download, login, get_token

    # AUTH CHECK
    print("\n🔐 Checking Hugging Face Authentication...")
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = get_token()
        except:
            token = None
            
        if token:
            print("   -> Found cached HF token.")
        else:
            print("   -> No HF_TOKEN env var or cached token found. If you hit 401 errors, you need to login.")
            print("      To login now, enter your User Access Token (Text) below. Press Enter to skip.")
            token = input("      HF Token > ").strip()
            if token: 
                login(token=token)
                print("   ✅ Logged in.")
            else:
                token = None
                print("   ⚠️ Proceeding without explicit token (Public only).")

    # HF Models
    for name, conf in MODELS.items():
        print(f"\n[*] Processing {name} ({conf['repo']}) -> {conf['target']}")
        
        conf['target'].mkdir(parents=True, exist_ok=True)
        
        try:
            dl_kwargs = {
                "repo_id": conf['repo'],
                "revision": conf.get('revision', None),
                "local_dir": str(conf['target']),
                "local_dir_use_symlinks": True, # CRITICAL: Uses hardlinks on same drive, saves OS storage
                "token": token
            }

            if conf['type'] == 'snapshot':
                snapshot_download(**dl_kwargs)
                
            elif conf['type'] == 'file':
                hf_hub_download(
                    repo_id=conf['repo'],
                    filename=conf['filename'],
                    revision=conf.get('revision', None),
                    local_dir=str(conf['target']),
                    local_dir_use_symlinks=True,
                    token=token
                )
            elif conf['type'] == 'files':
                for fname in conf['filenames']:
                    print(f"    -> Fetching {fname}...")
                    hf_hub_download(
                        repo_id=conf['repo'],
                        filename=fname,
                        revision=conf.get('revision', None),
                        local_dir=str(conf['target']),
                        local_dir_use_symlinks=True,
                        token=token
                    )

            print(f"    ✅ Done.")
            
        except Exception as e:
            print(f"    ❌ Download Failed: {e}")

    print("\n✨ All operations complete.")

if __name__ == "__main__":
    main()
