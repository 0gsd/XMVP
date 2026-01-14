#!/usr/bin/env python3
# unzip_models.py
import os
import zipfile
import shutil
import re

ZIP_DIR = "/Volumes/ORICO/weightsquared/zips"
DEST_ROOT = "/Volumes/ORICO/weightsquared/weights"
UNSORTED_DIR = os.path.join(DEST_ROOT, "_unsorted")

def clean_name(name):
    # Remove URL encoded chars and parens
    name = name.replace("%28", "(").replace("%29", ")")
    name = re.sub(r'[^\w\s\-\.]', '', name)
    return name.strip()

def main():
    if not os.path.exists(UNSORTED_DIR):
        os.makedirs(UNSORTED_DIR, exist_ok=True)
        
    zips = [f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")]
    print(f"[*] Found {len(zips)} zips.")
    
    for zf in zips:
        zip_path = os.path.join(ZIP_DIR, zf)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Inspect contents
                # We want to extract .pth and .index
                # We should create a folder for each model based on zip name if it's named humanly
                # If zip name is UUID (e.g. 07ba1...), we need to look inside for finding the name.
                
                # Check for .pth file inside
                pth_files = [n for n in z.namelist() if n.endswith(".pth") and not n.startswith("__MACOSX")]
                
                if not pth_files:
                    print(f"[-] Skipped {zf}: No .pth found.")
                    continue
                
                # Naming Logic
                # If zip name is human (Ariana Grande), use it.
                # If zip name is UUID, use the .pth filename.
                
                model_name = os.path.splitext(zf)[0]
                if "-" in model_name and any(c.isdigit() for c in model_name) and len(model_name) > 20:
                     # Likely UUID, verify if pth name is better
                     pth_name = os.path.basename(pth_files[0])
                     possible_name = os.path.splitext(pth_name)[0]
                     # Clean up pth noise like "G_" or "D_" if purely that
                     if possible_name not in ["G", "D", "model"]:
                         model_name = clean_name(possible_name)
                else:
                    model_name = clean_name(model_name)
                    
                target_dir = os.path.join(UNSORTED_DIR, model_name)
                os.makedirs(target_dir, exist_ok=True)
                
                print(f"    [>] Extracting {model_name}...")
                
                for file_info in z.infolist():
                    if file_info.filename.startswith("__MACOSX"): continue
                    if file_info.filename.endswith(".pth") or file_info.filename.endswith(".index"):
                        # Extract flatly to target_dir
                        file_info.filename = os.path.basename(file_info.filename)
                        z.extract(file_info, target_dir)
                        
        except Exception as e:
            print(f"[-] Error unzipping {zf}: {e}")

    # Create Category Folders
    os.makedirs(os.path.join(DEST_ROOT, "male"), exist_ok=True)
    os.makedirs(os.path.join(DEST_ROOT, "female"), exist_ok=True)
    
    print("\n[*] Extraction Complete.")
    print(f"[*] Models are in: {UNSORTED_DIR}")
    print("[*] Please manually drag them into 'male' or 'female' folders in:")
    print(f"    {DEST_ROOT}")

if __name__ == "__main__":
    main()
