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
    "Flux-Dev": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "type": "snapshot",
        "target": MW_ROOT / "flux-root" / "dev"
    },
    # GGUF entries removed — stable-diffusion-cpp doesn't support Flux 2.
    # Use Diffusers at flux-root/dev instead.
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
    },
    "Kokoro": {
        "repo": "Kijai/Kokoro-82M-ONNX",
        "type": "snapshot",
        "target": MW_ROOT / "kokoro-root"
    },
    "hunyuan-foley": {
        "repo": "tencent/HunyuanVideo-Foley", # Corrected Repo ID
        "type": "snapshot",
        "target": MW_ROOT / "hunyuan-foley"
    },
    "Wan-2.1": {
        "repo": "Wan-AI/Wan2.1-I2V-14B-720P",
        "type": "snapshot", 
        "target": MW_ROOT / "wan-root"
    },
    "Hunyuan-CLIP": {
        "repo": "Comfy-Org/HunyuanVideo_repackaged",
        "type": "files",
        "filenames": [
            "split_files/text_encoders/clip_l.safetensors" 
        ],
        "target": MW_ROOT / "comfyui-root" / "models" / "clip"
    },
    "Hunyuan-T5": {
        "repo": "city96/t5-v1_1-xxl-encoder-bf16",
        "type": "files",
        "filenames": ["model.safetensors"], # It's usually named model.safetensors in this repo
        # We need to rename it to t5xxl_fp8_e4m3fn.safetensors on arrival? 
        # Actually, let's stick to Comfy-Org if possible, but it wasn't in the list?
        # Let's try downloading from the MAIN city96 repo but saving to models/clip
        "target": MW_ROOT / "comfyui-root" / "models" / "clip"
    },
    "Hunyuan-VAE": {
        "repo": "Comfy-Org/HunyuanVideo_repackaged",
        "type": "files",
        "filenames": ["split_files/vae/hunyuan_video_vae_bf16.safetensors"],
        "target": MW_ROOT / "comfyui-root" / "models" / "vae"
    },
    "SkyReels": {
        "repo": "Skywork/SkyReels-V3-A2V-19B",
        "type": "snapshot",
        "target": MW_ROOT / "skyreels-root"
    },
    # --- 3D Model Collections ---
    "glTF-Sample-Assets": {
        "repo": "KhronosGroup/glTF-Sample-Assets",
        "type": "git_clone",
        "url": "https://github.com/KhronosGroup/glTF-Sample-Assets.git",
        "target": MW_ROOT / "3D-objects" / "glTF-Sample-Assets"
    },
    "ThreeJS-Models": {
        "repo": "mrdoob/three.js",
        "type": "git_clone",
        "url": "https://github.com/mrdoob/three.js.git",
        "target": MW_ROOT / "3D-objects" / "three.js" 
        # We only really want examples/models but a full clone is safest/easiest for now. 
        # It's big but useful.
    },
    "Kenney-Assets-Space": {
        "repo": "KenneyNL/assets-space-kit", 
        "type": "placeholder", 
        "target": MW_ROOT / "3D-objects"
    },
    "ColorizeDiffusion": {
        "repo": "tellurion/ColorizeDiffusion",
        "type": "snapshot",
        "target": MW_ROOT / "colorize-diffusion-root"
    },
    "F5-TTS": {
        "repo": "SWivid/F5-TTS",
        "type": "files",
        "filenames": [
            "F5TTS_v1_Base/model_1250000.safetensors",
            "F5TTS_v1_Base/vocab.txt"
        ],
        "target": MW_ROOT / "f5tts-root"
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

WAN_REPO = "https://github.com/Wan-Video/Wan2.1.git"
WAN_TARGET = MW_ROOT / "Wan2.1-main"

def git_clone_wan():
    print(f"\n[*] Processing Wan2.1 Code -> {WAN_TARGET}")
    if (WAN_TARGET / "generate.py").exists():
        print("    -> Wan2.1 code seems already present (generate.py found). Skipping clone.")
        return

    if not WAN_TARGET.exists():
        WAN_TARGET.mkdir(parents=True, exist_ok=True)
    
    # Check if empty
    if any(WAN_TARGET.iterdir()):
        print("    [!] Warning: Target directory not empty. Attempting clone anyway (git might fail)...")
    
    try:
        subprocess.run(["git", "clone", WAN_REPO, "."], cwd=WAN_TARGET, check=True)
        print("    ✅ Cloned Wan2.1 Code.")
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Git Clone Failed: {e}")

def main():
    print(f"🚀 XMVP Model Populator")
    print(f"   Root: {MW_ROOT}")
    print(f"   Cache: {HF_CACHE}")
    
    ensure_library()
    from huggingface_hub import hf_hub_download, snapshot_download, login, get_token
    
    # AUTH CHECK
    print("\n🔐 Checking Hugging Face Authentication...")
    # Check if token exists in env or cache
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try to find cached token using modern utility
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

    # 1. ComfyUI & Wan2.1 Source
    git_clone_comfy()
    git_clone_wan()

    # HF Models
    for name, conf in MODELS.items():
        print(f"\n[*] Processing {name} ({conf['repo']}) -> {conf['target']}")
        
        # FIX: Do not blindly create directory for git_clone if we want clone to work naturally
        # unless it's not git_clone.
        if conf.get('type') != 'git_clone':
            conf['target'].mkdir(parents=True, exist_ok=True)
        else:
            # Check if parent exists
            if not conf['target'].parent.exists():
                conf['target'].parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Common patterns to ignore for large snapshots (e.g. redundant split weights if we want consolidated)
            # Or vice-versa. Default: enable symlinks to save 50% storage immediately.
            dl_kwargs = {
                "repo_id": conf['repo'],
                "revision": conf.get('revision', None),
                "local_dir": str(conf['target']),
                "local_dir_use_symlinks": True, # CRITICAL: Uses hardlinks on same drive, saves 100s of GBs
                "token": token
            }

            if conf['type'] == 'snapshot':
                # Optimization for Flux/Gemma to avoid double-weight downloads
                if "FLUX" in name:
                    # If we only want the consolidated file (best for ComfyUI/Local Single-Load)
                    # We ignore the split transformer/vae/text_encoder component folders
                    dl_kwargs["ignore_patterns"] = ["transformer/*", "text_encoder/*", "text_encoder_2/*", "vae/*"]
                    print("    -> Opt: Ignoring split component folders (using consolidated .safetensors only)")
                
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
            elif conf['type'] == 'git_clone':
                target_dir = conf['target']
                
                # Check if directory exists and is NOT a git repo and is empty/junk
                if target_dir.exists() and not (target_dir / ".git").exists():
                    # If empty, we can clone into it? Git usually wants it to not exist or be empty.
                    # If it has files but no .git, it's a broken state or user created it.
                    if not any(target_dir.iterdir()):
                        print("    -> Directory exists but is empty. Cloning into it...")
                    else:
                        print("    [!] Directory exists, is not empty, and not a git repo. Attempting to clone anyway (git might fail)...")
                
                if not target_dir.exists() or (target_dir.exists() and not any(target_dir.iterdir())):
                    print(f"    -> Cloning {conf['url']}...")
                    subprocess.run(["git", "clone", conf['url'], str(target_dir)], check=True)
                else:
                    print(f"    -> Directory exists and not empty. Checking if it is a repo...")
                    if (target_dir / ".git").exists():
                         print(f"    -> It is a git repo. Pulling latest...")
                         try:
                             subprocess.run(["git", "pull"], cwd=target_dir, check=True)
                         except Exception as e:
                             print(f"    [!] Pull failed (might be dirty): {e}")
                    else:
                        print(f"    [!] Target {target_dir} is not a git repo. Skipping.")

            print(f"    ✅ Done.")
            
        except Exception as e:
            print(f"    ❌ Download/Clone Failed: {e}")

    print("\n✨ All operations complete.")

if __name__ == "__main__":
    main()
