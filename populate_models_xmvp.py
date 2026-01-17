#!/usr/bin/env python3
"""
populate_models_xmvp.py
-----------------------
Downloads all required local models for the XMVP v0.5 pipeline to:
/Volumes/XMVPX/mw

Targets:
1. LT2X (Lightricks/LTX-Video)
2. Flux (black-forest-labs/FLUX.1-schnell)
3. IndexTTS (IndexTeam/IndexTTS-2)
4. Hunyuan-Foley (city96/HunyuanVideo-gguf)
5. RVC Base Assets (lj1995/VoiceConversionWebUI)
6. ComfyUI (Git Clone)
7. Gemma 3 (google/gemma-3-27b-it)
"""

import os
import sys
import subprocess
from pathlib import Path

# --- CONFIG ---
MW_ROOT = Path("/Volumes/XMVPX/mw")
HF_CACHE = MW_ROOT / "huggingface-root"

# Ensure Environment for Cache
os.environ["HF_HOME"] = str(HF_CACHE)

MODELS = {
    "LT2X": {
        "repo": "Lightricks/LTX-Video",
        "type": "snapshot",
        "target": MW_ROOT / "LT2X-root"
    },
    "Flux": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "type": "snapshot",
        "target": MW_ROOT / "flux-root"
    },
    "IndexTTS": {
        "repo": "IndexTeam/IndexTTS-2",
        "type": "snapshot",
        "target": MW_ROOT / "indextts-root"
    },
    "Hunyuan": {
        "repo": "city96/HunyuanVideo-gguf",
        "type": "file",
        # Verified filename format from repo listing (Q8_0 is standard)
        # Often these are named 'hunyuan-video-t2v-720p-Q8_0.gguf' or just 'Q8_0.gguf'
        # Let's try the full name with correct casing.
        "filename": "hunyuan-video-t2v-720p-Q8_0.gguf",
        "target": MW_ROOT / "hunyuan-root"
    },
    "RVC": {
        "repo": "lj1995/VoiceConversionWebUI",
        "type": "files",
        "filenames": ["hubert_base.pt", "rmvpe.pt"],
        "target": MW_ROOT / "rvc-root"
    },
    "Gemma": {
        "repo": "google/gemma-3-27b-it",
        "type": "snapshot",
        "target": MW_ROOT / "gemma-root"
    },
    "T5": {
        "repo": "city96/t5-v1_1-xxl-encoder-bf16",
        "type": "snapshot",
        "target": MW_ROOT / "t5weights-root"
    }
}

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
COMFY_TARGET = MW_ROOT / "comfyui-root"

def ensure_library():
    try:
        import huggingface_hub
    except ImportError:
        print("[-] Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)

def git_clone_comfy():
    print(f"\n[*] Processing ComfyUI -> {COMFY_TARGET}")
    if (COMFY_TARGET / "main.py").exists():
        print("    -> ComfyUI seems already installed (main.py found). Skipping clone.")
        return

    if not COMFY_TARGET.exists():
        COMFY_TARGET.mkdir(parents=True, exist_ok=True)
    
    # Check if empty
    if any(COMFY_TARGET.iterdir()):
        print("    [!] Warning: Target directory not empty. Attempting clone anyway (git might fail)...")
    
    try:
        subprocess.run(["git", "clone", COMFY_REPO, "."], cwd=COMFY_TARGET, check=True)
        print("    ✅ Cloned ComfyUI.")
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Git Clone Failed: {e}")

def main():
    print(f"🚀 XMVP Model Populator")
    print(f"   Root: {MW_ROOT}")
    print(f"   Cache: {HF_CACHE}")
    
    ensure_library()
    from huggingface_hub import hf_hub_download, snapshot_download, login
    
    # AUTH CHECK
    print("\n🔐 Checking Hugging Face Authentication...")
    # Check if token exists in env or cache
    token = os.environ.get("HF_TOKEN")
    if token:
        print("   -> Found HF_TOKEN in environment. Logging in...")
        login(token=token)
    else:
        # Try to see if we are already logged in via cached token
        # There isn't a simple public API to check 'is_logged_in' without side effects, 
        # but running a simple whoami check via shell or just trying download works.
        # We'll just prompt if we suspect it might fail, or better, always give user a chance to login if they want.
        print("   -> No HF_TOKEN env var found. If you hit 401 errors, you need to login.")
        print("      To login now, enter your User Access Token (Text) below. Press Enter to skip.")
        user_token = input("      HF Token > ").strip()
        if user_token:
            login(token=user_token)
        else:
            print("      Skipping login (assuming cached credentials).")

    # 1. ComfyUI
    git_clone_comfy()

    # 2. HF Models
    for name, conf in MODELS.items():
        print(f"\n[*] Processing {name} ({conf['repo']}) -> {conf['target']}")
        conf['target'].mkdir(parents=True, exist_ok=True)
        
        try:
            if conf['type'] == 'snapshot':
                snapshot_download(
                    repo_id=conf['repo'],
                    local_dir=str(conf['target']),
                    local_dir_use_symlinks=False
                )
            elif conf['type'] == 'file':
                hf_hub_download(
                    repo_id=conf['repo'],
                    filename=conf['filename'],
                    local_dir=str(conf['target']),
                    local_dir_use_symlinks=False
                )
            elif conf['type'] == 'files':
                for fname in conf['filenames']:
                    print(f"    -> Fetching {fname}...")
                    hf_hub_download(
                        repo_id=conf['repo'],
                        filename=fname,
                        local_dir=str(conf['target']),
                        local_dir_use_symlinks=False
                    )
            print(f"    ✅ Done.")
            
        except Exception as e:
            print(f"    ❌ Download Failed: {e}")

    print("\n✨ All operations complete.")

if __name__ == "__main__":
    main()
