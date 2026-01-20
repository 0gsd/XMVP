import os
import glob

TARGET_DIR = "/Volumes/XMVPX/mw/Wan2.1-main/wan"
SHIM_IMPORT = "from wan.modules.mps_fix import amp"
DEVICE_REPLACE = "torch.device('mps' if torch.backends.mps.is_available() else 'cpu')"

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    orig_content = content
    
    # 1. Replace amp import
    # Matches: import torch.cuda.amp as amp
    if "import torch.cuda.amp as amp" in content:
        content = content.replace("import torch.cuda.amp as amp", SHIM_IMPORT)
    
    # 2. Replace device initialization
    # Matches: torch.device(f"cuda:{device_id}")
    if 'torch.device(f"cuda:{device_id}")' in content:
        content = content.replace('torch.device(f"cuda:{device_id}")', DEVICE_REPLACE)
    if "torch.device(f'cuda:{device_id}')" in content:
        content = content.replace("torch.device(f'cuda:{device_id}')", DEVICE_REPLACE)
    
    # Matches simple: device="cuda" (in vae.py)
    content = content.replace('device="cuda"', 'device="mps"')
    content = content.replace("device='cuda'", "device='mps'")
    
    # 3. Replace empty_cache
    content = content.replace("torch.cuda.empty_cache()", "if torch.cuda.is_available(): torch.cuda.empty_cache()")
    
    # 4. Replace synchronize
    content = content.replace("torch.cuda.synchronize()", "if torch.cuda.is_available(): torch.cuda.synchronize()")
    
    # 5. T5 current_device
    content = content.replace("torch.cuda.current_device()", "0")
    
    # 6. flash_attention hard assert in attention.py
    if "attention.py" in filepath:
        content = content.replace("assert q.device.type == 'cuda'", "# assert q.device.type == 'cuda'")
        
    if content != orig_content:
        print(f"Patching {filepath}")
        with open(filepath, 'w') as f:
            f.write(content)
    else:
        print(f"Skipping {filepath} (no changes needed)")

def main():
    files = glob.glob(f"{TARGET_DIR}/**/*.py", recursive=True)
    for f in files:
        patch_file(f)

if __name__ == "__main__":
    main()
