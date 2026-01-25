#!/usr/bin/env python3
"""
prep_movie_assets.py
--------------------
Extracts characters, locations, and style from an XMVP file and generates a synthetic dataset
for training a Movie-Level LoRA (MLL) using Flux.

Features:
- Characters: 20 images each.
- Locations: 10 images each (extracted from INT./EXT.).
- Style: 10 images (abstract/concept art for visual style).
"""

import os
import sys
import json
import re
import argparse
import random
import glob
from pathlib import Path
from PIL import Image

# MVP Imports
try:
    import mvp_shared
    from mvp_shared import load_xmvp, Manifest, CSSV
    from flux_bridge import get_flux_bridge
except ImportError as e:
    print(f"[-] Critical Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
STYLE_PREFIX = "maximum quality 4K cinematic CGI movie render of"

VIEWS = [
    "close-up portrait", "medium shot", "wide full-body shot", "extreme close-up of eyes",
    "side profile portrait", "3/4 angle portrait", "low angle heroic shot", "high angle dramatic shot",
    "walking towards camera", "sitting at a table"
]

EXPRESSIONS = [
    "neutral expression", "slight smile", "serious intense look", "concerned expression",
    "laughing", "surprised expression", "angry shouting", "thoughtful looking away",
    "speaking energetically", "calm and relaxed"
]

LIGHTING = [
    "soft studio lighting", "dramatic cinematic lighting", "natural window light", "hard rim lighting"
]

def generate_char_prompts(char_name, movie_style, count=20, anchor=None):
    """Generates prompts for characters."""
    prompts = []
    combined = []
    for view in VIEWS:
        for expr in EXPRESSIONS:
             combined.append((view, expr))
    
    random.shuffle(combined)
    selected = combined[:count]
    while len(selected) < count:
        selected.append(random.choice(combined))
        
    for i, (view, expr) in enumerate(selected):
        light = random.choice(LIGHTING)
        desc = char_name
        if anchor:
            # Mix Name + Anchor
            # "A portrait of Stanley, resembles Humphrey Bogart at age 45..."
            desc = f"{char_name} (resembling {anchor})"
            
        p = f"A {view} of {desc}, {expr}, {light}. {STYLE_PREFIX} {movie_style}"
        prompts.append(p)
    return prompts

def generate_loc_prompts(loc_name, movie_style, count=10):
    """Generates prompts for locations."""
    prompts = []
    angles = ["wide shot", "establishing shot", "detail shot", "low angle", "overhead shot"]
    times = ["daylight", "night", "golden hour", "foggy morning"]
    
    for i in range(count):
        angle = random.choice(angles)
        time = random.choice(times)
        p = f"Empty set of {loc_name}, {angle}, {time}. {STYLE_PREFIX} {movie_style}"
        prompts.append(p)
    return prompts

def generate_style_prompts(movie_style, count=10):
    """Generates pure style concept prompts."""
    prompts = []
    subjects = ["abstract cinematic composition", "character silhouette", "landscape", "object detail", "light and shadow study"]
    
    for i in range(count):
        subj = random.choice(subjects)
        p = f"{STYLE_PREFIX} {movie_style}. {subj}"
        prompts.append(p)
    return prompts

def extract_locations(manifest):
    """Extracts INT./EXT. locations from segments/prompts."""
    locs = set()
    # Robust Regex: Handles "INT.", "EXT.", "INT/EXT" with optional quotes
    # Capture group 1 is the main location name
    regex = r"[\"']?(?:INT\.|EXT\.|INT/EXT|I/E)\s+([A-Z0-9\s\-\'.]+?)(?:\s+\-|\s*$|\s*[\"'])"
    
    if manifest.segs:
        for seg in manifest.segs:
            if seg.prompt:
                # Iterate matches to catch multiple locations if present
                for match in re.finditer(regex, seg.prompt, re.IGNORECASE):
                     raw_loc = match.group(1).strip()
                     # Cleanup
                     clean_loc = raw_loc.strip('"').strip("'").strip()
                     if len(clean_loc) > 2:
                         locs.add(clean_loc.upper())
    return list(locs)

def is_valid_clean_char(c):
    """Validates character names, filtering out metadata keys and syntax artifacts."""
    if not c: return False
    # Strip anchor tags first
    c = re.sub(r'\{.*?\}', '', c)
    c = c.split('[')[0] # Strip gender tags too if present
    c = c.strip().strip('"').strip("'")
    if len(c) < 2: return False
    if len(c) > 30: return False
    
    # Reject syntactical junk (JSON artifacts often leave these)
    if any(x in c for x in [":", "{", "}", "[", "]", "SCRIPT", "SCENE"]): return False
    
    # Blacklist common false positives from JSON dumps
    blacklist = [
        "ACTION", "DIALOGUE", "CUT TO", "FADE IN", "FADE OUT", 
        "INT.", "EXT.", "CAMERA", "SFX", "MUSIC", 
        "TIME_OF_DAY", "SETTING", "CHARACTER", "CHARACTERS", 
        "TITLE", "SCENE_NUMBER", "SCENE_CONTENT", "LINES",
        "SCENE", "SHEET"
    ]
    if c.upper() in blacklist: return False
    return True

def main():
    parser = argparse.ArgumentParser(description="XMVP Movie Asset Prepper")
    parser.add_argument("--xml", required=True, help="Input XMVP XML file")
    parser.add_argument("--out", default="z_training_data/movies", help="Output Root for Datasets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")
    args = parser.parse_args()

    xml_path = args.xml
    if not os.path.exists(xml_path):
        print(f"[-] XML not found: {xml_path}")
        sys.exit(1)
        
    print(f"📦 Loading XMVP: {xml_path}")
    
    # 1. Load Data
    try:
        raw_man = load_xmvp(xml_path, "Manifest")
        if raw_man:
            manifest = Manifest.model_validate_json(raw_man)
        else:
            print("    [!] No <Manifest>. Checking <Portions>...")
            raw_portions = load_xmvp(xml_path, "Portions")
            if raw_portions:
                from mvp_shared import Portion, Seg, DialogueScript, DialogueLine
                portions = [Portion.model_validate(p) for p in json.loads(raw_portions)]
                
                all_lines = []
                segs = []
                current_frame = 0
                fps = 24
                
                for p in portions:
                     dur_frames = int(p.duration_sec * fps)
                     segs.append(Seg(
                         id=p.id, start_frame=current_frame, end_frame=current_frame + dur_frames, prompt=p.content
                     ))
                     current_frame += dur_frames
                     
                     if p.dialogue:
                         all_lines.extend(p.dialogue)
                     elif ":" in p.content:
                         # Fallback parse
                         parts = p.content.split(":", 1)
                         raw_char = parts[0].strip()
                         
                         # Sanitization
                         # Remove quotes
                         clean_char = raw_char.strip('"').strip("'").strip().upper()
                         
                         if is_valid_clean_char(clean_char):
                             all_lines.append(DialogueLine(
                                 character=clean_char, text=parts[1].strip(), action="speaking", visual_focus=clean_char
                             ))
                         
                manifest = Manifest(segs=segs, dialogue=DialogueScript(lines=all_lines))
            else:
                 raise ValueError("No Manifest or Portions found.")

        raw_bib = load_xmvp(xml_path, "Bible")
        bible = CSSV.model_validate_json(raw_bib) if raw_bib else None
        
    except Exception as e:
        print(f"[-] Failed to parse XMVP: {e}")
        sys.exit(1)
        
    # Extract Title
    movie_title = "Unknown_Movie"
    anchor_map = {}
    try:
        raw_story = load_xmvp(xml_path, "Story")
        if raw_story:
            story = json.loads(raw_story)
            movie_title = story.get("title", "Unknown_Movie").replace(" ", "_")
            
            # Build Anchor Map: "NAME" -> "Anchor Prompt"
            # Chars are likely "Name (Role) [Gender] {Anchor}"
            if story.get("characters"):
                for c_str in story["characters"]:
                    # check for {Anchor}
                    anchor_match = re.search(r'\{(.*?)\}', c_str)
                    if anchor_match:
                        # Extract Name
                        name_part = c_str.split("(")[0].split("[")[0].strip().upper()
                        name_part = name_part.replace('"', '').replace("'", "")
                        anchor_map[name_part] = anchor_match.group(1)
    except:
        pass
        
    # Extract Chars
    chars = set()
    if manifest.dialogue and manifest.dialogue.lines:
        for line in manifest.dialogue.lines:
            # Clean and Validate
            clean = line.character.strip().strip('"').strip("'").upper()
            if is_valid_clean_char(clean):
                chars.add(clean)
            
    # Extract Locations
    locs = extract_locations(manifest)
    
    # Extract Style
    style_core = "Cinematic"
    if bible and bible.vision:
        style_core = bible.vision.replace("STYLE:", "").strip()
        
    print(f"    🎬 Movie: {movie_title}")
    
    # MLL Template Logic
    target_name = movie_title
    
    # Check Story first (since we just loaded it as dict)
    if isinstance(story, dict) and story.get("mll_template"):
        target_name = story.get("mll_template")
        
    # Check Bible (Override or Primary?)
    if bible and bible.mll_template:
        target_name = bible.mll_template
        
    if target_name != movie_title:
        print(f"    🏷️  MLL Template Active: {target_name}")
        
    print(f"    🎨 Style Core: {style_core[:50]}...")
    print(f"    👥 Characters ({len(chars)}): {list(chars)}")
    print(f"    📍 Locations ({len(locs)}): {locs}")

    # Check Key LoRA Existence (If template)
    # If we are using a template, and it exists, we might normally skip. 
    # But prep_movie_assets is explicitly called to PREP. 
    # We should let the caller decide to skip?
    # Or strict check here?
    # User said: "defaults to creating the MLLTemplate folder's version first since the XMVP it has to fulfill needs one and none exists"
    # So if it DOES exist, we probably shouldn't overwrite it with this random episode's data unless force.
    
    # MLL Template Logic
    # If LoRA exists, we usually skip. BUT we want "Additive Casting".
    # So we proceed to generation. The generation helper `generate_and_save` 
    # already checks `if os.path.exists(fpath) and not args.force`.
    # This means existing characters are safely skipped, and NEW characters are added.
    # The only catch is we need to signal to the Trainer that we have NEW data.
    # We can trust `content_producer` to always run training? Or we can output a status.
    
    lora_check = os.path.join("adapters/movies", f"{target_name}.safetensors")
    if os.path.exists(lora_check): # and not args.force:
        print(f"    ℹ️  Template Adapter Found: {lora_check}")
        print("    [+] Proceeding in ADDITIVE MODE (Scanning for new Cast members)...")
        # Do NOT exit. Allow generation loop to run.
        # It will only generate missing (new) files.

    # 2. Setup Output
    dataset_dir = os.path.join(args.out, target_name, "dataset")
    if os.path.exists(dataset_dir): # and not args.force:
        print(f"    [.] Dataset dir exists: {dataset_dir}. Checking for updates...")
        # sys.exit(0) # Logic: if forced, fine.
        
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 3. Init Bridge
    print("    🌊 Initializing Flux Bridge...")
    
    # Priority 1: Flux.2 [klein] 9B
    klein_path = "/Volumes/XMVPX/mw/flux-root/klein-9b"
    flux_path = klein_path
    
    # Priority 2: Standard Flux Schnell Root (Symlink?)
    if not os.path.exists(flux_path):
        flux_path = "/Volumes/XMVPX/mw/flux-root"
        
    # Priority 3: Old Safetensors fallback
    if not os.path.exists(flux_path):
        flux_path = "/Volumes/XMVPX/mw/flux-schnell.safetensors"
        
    print(f"       Target Model: {flux_path}")
    bridge = get_flux_bridge(flux_path)
    if not bridge:
        print("[-] Bridges burnt. Aborting.")
        sys.exit(1)
        
    # Determine Guidance based on Model
    # Klein 9B typically needs guidance ~3.5. Schnell needs 0.0.
    # Logic: if path contains "klein", use 3.5. Else 0.0.
    guidance = 3.5 if "klein" in str(flux_path).lower() else 0.0
    print(f"       Guidance Scale: {guidance}")

    # 4. Generate Assets
    metadata = []
    
    def generate_and_save(prompt, fname):
        fpath = os.path.join(dataset_dir, fname)
        if os.path.exists(fpath) and not args.force:
            print(f"       . skipping {fname}")
            return
            
        if len(prompt) > 76:
            prompt = prompt[:76]
            
        print(f"       + {fname} | {prompt[:40]}...")
        # 512x512 for Speed + LoRA Standard
        img = bridge.generate(prompt, width=512, height=512, steps=4, guidance_scale=guidance)
        if img:
            img.save(fpath)
        else:
            print("       [-] Gen failed.")

    # A. Characters (20 each)
    # Load CLIP for Consistency Checking
    try:
        from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
        import torch
        print("    👁️ Loading CLIP for Visual Consistency Check...")
        clip_model_name = "openai/clip-vit-base-patch32"
        # Use simple model for speed
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_valid = True
    except Exception as e:
        print(f"    [-] CLIP not available ({e}). Skipping consistency check.")
        clip_valid = False

    def cull_outliers(image_paths, keep_count=10):
        if not clip_valid or len(image_paths) <= keep_count: 
            return image_paths
            
        print(f"       ⚖️ Culling {len(image_paths)} -> {keep_count} based on visual consistency...")
        
        try:
            # Load Images
            images = []
            valid_paths = []
            for p in image_paths:
                try:
                    images.append(Image.open(p))
                    valid_paths.append(p)
                except: pass
            
            if not images: return []

            # Compute Embeddings
            inputs = clip_processor(images=images, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
            
            # Normalize
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Compute Centroid
            centroid = embeddings.mean(dim=0)
            
            # Compute Cosine Similarity to Centroid
            # Sim = (A . B) / (|A| |B|) -> Since normalized, just A . B
            scores = (embeddings @ centroid.unsqueeze(1)).squeeze()
            
            # Sort indices by score (descending)
            # scores is a 1D tensor of length N
            top_k_indices = torch.topk(scores, k=keep_count).indices.tolist()
            
            kept_paths = []
            for i in range(len(valid_paths)):
                if i in top_k_indices:
                    kept_paths.append(valid_paths[i])
                else:
                    # Delete outlier
                    print(f"       🗑️ Pruning outlier: {os.path.basename(valid_paths[i])} (Score: {scores[i]:.4f})")
                    try:
                        os.remove(valid_paths[i])
                    except: pass
            
            return kept_paths
            
        except Exception as e:
            print(f"       [-] Culling failed: {e}")
            return image_paths

    for char in chars:
        print(f"    📸 Shooting Character: {char}")
        clean_name = char.replace(" ", "_").lower()
        
        # --- PERSISTENCE CHECK ---
        # Check for existing images
        existing_pattern = os.path.join(dataset_dir, f"char_{clean_name}_*.jpg")
        existing_files = glob.glob(existing_pattern)
        
        if len(existing_files) >= 6 and not args.force:
             print(f"       ✨ canon found ({len(existing_files)} images). Skipping generation.")
             # Add existing to metadata
             for ef in existing_files:
                 ename = os.path.basename(ef)
                 # We don't have the original prompt cleanly mapped unless we read old metadata 
                 # or just use a generic one. For now, empty prompt or reconstruct?
                 # content_producer/training usually needs a caption. 
                 # If we skip, we might miss adding them to the NEW metadata file if we are rewriting it.
                 # The script rewrites metadata.jsonl at the end.
                 # So we MUST add them to `metadata` list here.
                 p_text = f"A portrait of {char}" # Fallback
                 metadata.append({"file_name": ename, "text": p_text})
             continue
        # -------------------------
 
        # Lookup Anchor
        anchor_text = anchor_map.get(char, None)
        if anchor_text:
            print(f"       ⚓ Anchor Found: {anchor_text}")
        
        prompts = generate_char_prompts(char, style_core, count=10, anchor=anchor_text) # Reduced from 20 -> 10
        
        generated_files = []
        
        for i, p in enumerate(prompts):
            fname = f"char_{clean_name}_{i:02d}.jpg"
            generate_and_save(p, fname)
            fpath = os.path.join(dataset_dir, fname)
            if os.path.exists(fpath):
                 generated_files.append(fpath)
            
        # Cull
        if generated_files:
            kept = cull_outliers(generated_files, keep_count=6) # Reduced from 10 -> 6
            
            # Map fname -> prompt locally
            local_map = {f"char_{clean_name}_{i:02d}.jpg": p for i, p in enumerate(prompts)}
            
            for kpath in kept:
                kname = os.path.basename(kpath)
                text = local_map.get(kname, "")
                metadata.append({"file_name": kname, "text": text})

    # B. Locations (10 each)
    # (Leaving Locations uncullled for now or reuse function?)
    # User said "check each group of 20 images" (implied characters).
    for loc in locs:
        print(f"    📸 Shooting Location: {loc}")
        prompts = generate_loc_prompts(loc, style_core, count=10)
        clean_loc = loc.replace(" ", "_").lower()
        for i, p in enumerate(prompts):
            fname = f"loc_{clean_loc}_{i:02d}.jpg"
            generate_and_save(p, fname)
            metadata.append({"file_name": fname, "text": p})
            
    # C. Style Anchors (10 total)
    print(f"    📸 Shooting Style Anchors...")
    prompts = generate_style_prompts(style_core, count=10)
    for i, p in enumerate(prompts):
        fname = f"style_anchor_{i:02d}.jpg"
        generate_and_save(p, fname)
        metadata.append({"file_name": fname, "text": p})

    # 5. Write Metadata
    meta_path = os.path.join(dataset_dir, "metadata.jsonl")
    with open(meta_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Dataset Ready: {dataset_dir}")
    print(f"   Images: {len(metadata)}")
    print(f"   Metadata: {meta_path}")

if __name__ == "__main__":
    main()
