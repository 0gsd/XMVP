
import os
import torch
import numpy as np
import glob
import logging

logging.basicConfig(level=logging.INFO)

ROOT_DIR = "/Volumes/XMVPX/mw/kokoro-root"
VOICES_DIR = os.path.join(ROOT_DIR, "voices")
OUTPUT_PATH = os.path.join(ROOT_DIR, "voices.npz")

def convert():
    if not os.path.exists(VOICES_DIR):
        print(f"Voices dir not found: {VOICES_DIR}")
        return

    pt_files = glob.glob(os.path.join(VOICES_DIR, "*.pt"))
    print(f"Found {len(pt_files)} voice files.")
    
    voice_dict = {}
    
    for pt in pt_files:
        name = os.path.splitext(os.path.basename(pt))[0]
        try:
            # Load torch tensor
            # weights_only=True is safer, but might fail if old format. 
            # Trying default first.
            tensor = torch.load(pt, weights_only=False)
            
            # Convert to numpy
            # Handle if it's not a tensor but a list/dict? (Unlikely for .pt)
            if isinstance(tensor, torch.Tensor):
                voice_dict[name] = tensor.numpy()
            else:
                print(f"Skipping {name}: Not a tensor ({type(tensor)})")
                
        except Exception as e:
            print(f"Error loading {name}: {e}")

    print(f"Saving {len(voice_dict)} voices to {OUTPUT_PATH}...")
    np.savez(OUTPUT_PATH, **voice_dict)
    print("Done.")

if __name__ == "__main__":
    convert()
