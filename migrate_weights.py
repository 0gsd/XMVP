#!/usr/bin/env python3
# migrate_weights.py
# Scans for Replay/RVC models and copies them to the central weights library.

import os
import shutil
import glob
from pathlib import Path

SOURCE_SEARCH_PATHS = [
    os.path.expanduser("~/Library/Application Support"),
    os.path.expanduser("~/Library/Application Support/Weights"),
    os.path.expanduser("~/Library/Application Support/Replay"),
    os.path.expanduser("~/Downloads"), # sometimes people leave them here
]

DEST_DIR = "/Volumes/ORICO/weightsquared"
EXTENSIONS = ["*.pth", "*.index"]

def find_files(base_path):
    matches = []
    for ext in EXTENSIONS:
        # Recursive search max depth 4 to avoid scanning entire drive
        # We manually walk to control depth
        for root, dirs, files in os.walk(base_path):
            depth = root[len(base_path):].count(os.sep)
            if depth > 4:
                del dirs[:] # Don't go deeper
                continue
            for f in files:
                if f.endswith(ext.replace("*", "")):
                    matches.append(os.path.join(root, f))
    return matches

def main():
    print(f"[*] Starting Weight Migration to {DEST_DIR}")
    
    if not os.path.exists(DEST_DIR):
        try:
            os.makedirs(DEST_DIR, exist_ok=True)
            print(f"[+] Created destination: {DEST_DIR}")
        except PermissionError:
            print(f"[-] ERROR: Cannot create {DEST_DIR}. Check permissions or drive mount.")
            return

    found_files = []
    print("[*] Scanning common locations (this may take a moment)...")
    
    for base in SOURCE_SEARCH_PATHS:
        if os.path.exists(base):
            print(f"    Scanning {base}...")
            found_files.extend(find_files(base))
            
    if not found_files:
        print("[-] No .pth or .index files found in standard locations.")
        print("    Please manually drag your 'Weights' or 'Replay' models folder to /Volumes/ORICO/weightsquared")
        return

    print(f"[+] Found {len(found_files)} candidates.")
    
    success_count = 0
    for src in found_files:
        filename = os.path.basename(src)
        # Check if it looks like a model (usually > 10MB for pth, or strictly named)
        # Actually RVC models can be small. Let's just copy everything that looks like a weight.
        
        # Create a clean folder name based on the parent dir if possible?
        # Usually Replay stores them in folders by name.
        parent_name = os.path.basename(os.path.dirname(src))
        
        # If the parent is just "models" or "weights", maybe go up one level?
        if parent_name.lower() in ["models", "weights", "rvc"]:
             parent_name = os.path.basename(os.path.dirname(os.path.dirname(src)))
             
        # target structure: /weightsquared/VoiceName/model.pth
        target_folder = os.path.join(DEST_DIR, parent_name)
        os.makedirs(target_folder, exist_ok=True)
        
        dest_path = os.path.join(target_folder, filename)
        
        if os.path.exists(dest_path):
            print(f"    [.] Skipping {filename} (exists)")
        else:
            print(f"    [>] Copying {parent_name}/{filename}...")
            shutil.copy2(src, dest_path)
            success_count += 1
            
    print(f"[*] Migration Complete. New files: {success_count}")

if __name__ == "__main__":
    main()
