#!/usr/bin/env python3
"""
xmvp_converter.py
-----------------
Universal Script Converter for XMVP.
Ingests text-based scripts (TXT, MD, etc.) and hydrates them into
a full 24-Beat XMVP XML structure.

Features:
- "HJ24" Smart Chunking: Splits script into 24 narrative beats.
- Parody Mode: Rewrites content using GemmaW (Simulacrum).
- Retcon Support: Adjusts duration constraints on the fly.
"""

import os
import sys
import argparse
import logging
import json
import re
import csv
import math
import random
from pathlib import Path

# MVP Import Setup
MV_ROOT = Path(__file__).resolve().parent
sys.path.append(str(MV_ROOT))

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from text_engine import get_engine, TextEngine
    import mvp_shared
    from mvp_shared import save_xmvp, CSSV, VPForm, Story, Portion, Constraints, Manifest, Seg
    import definitions
except ImportError as e:
    logging.error(f"MVP Import Error: {e}")
    sys.exit(1)

# Training Data Paths
HJ24_PATH = MV_ROOT / "z_training_data" / "hj24.csv"

# --- HELPERS ---

def load_hj24():
    """Matches the 24 lines in hj24.csv to a list of beat definitions."""
    beats = []
    if not HJ24_PATH.exists():
        logging.warning(f"HJ24 CSV not found at {HJ24_PATH}. Using generic beats.")
        return [{"id": i+1, "name": f"Beat {i+1}", "desc": "Generic Narrative Beat"} for i in range(24)]
    
    try:
        with open(HJ24_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter out Act headers or empty lines if they exist in CSV
                if row.get("Beat") and str(row.get("Beat")).strip().isdigit():
                    beats.append({
                        "id": int(row["Beat"]),
                        "name": row.get("Campbellian Stage", f"Beat {row['Beat']}"),
                        "desc": row.get("Narrative Function (Production Detail)", "")
                    })
    except Exception as e:
        logging.error(f"Failed to load HJ24: {e}")
        # Fallback
        return [{"id": i+1, "name": f"Beat {i+1}", "desc": "Generic Narrative Beat"} for i in range(24)]
    
    return beats

def count_words(text):
    return len(re.findall(r'\w+', text))

def smart_chunk_script(text, num_chunks=24):
    """
    Splits text into `num_chunks` roughly equal parts, 
    respecting scene boundaries (INT./EXT. or Hyphens) where possible.
    """
    # 1. Split into Scenes
    # Robust Regex: 
    # - Standard Headers: INT., EXT., I/E.
    # - Hyphen separaters: ^\s*-\s*$
    # - Dashed lines: ^-{2,}$
    
    scene_pattern = r'(?:^|\n)(\s*(?:INT\.|EXT\.|I/E\.|INT |EXT |I/E |^\s*-\s*$|^-{2,}$)(?:.*))(?:\n|$)'
    raw_scenes = re.split(scene_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    
    scenes = []
    if len(raw_scenes) > 1:
        # Reconstruct scenes
        current_scene = raw_scenes[0] 
        # raw_scenes format: [pre_content, header1, body1, header2, body2, ...]
        # Actually re.split with capture group returns [part, sep, part, sep...]
        # Wait, if capture group is used, separators are included in result.
        
        # My previous logic was a bit naive expecting split to behave differently or just iterating?
        # Let's iterate and check if part is a header.
        
        # Correct approach for re.split with capture:
        # result = [text0, sep1, text1, sep2, text2...]
        
        # But here I want to include the separator with the following text.
        
        # Let's simplify: just match all lines.
        lines = text.split('\n')
        current_scene_lines = []
        
        for line in lines:
            is_sep = False
            # Check for header
            if re.match(r'^\s*(?:INT\.|EXT\.|I/E\.|INT |EXT |I/E )', line, re.IGNORECASE):
                is_sep = True
            elif re.match(r'^\s*-\s*$', line): # Single hyphen
                is_sep = True
            elif re.match(r'^-{2,}$', line): # Dashes
                is_sep = True
                
            if is_sep and current_scene_lines:
                # Flush
                scenes.append("\n".join(current_scene_lines))
                current_scene_lines = [line]
            else:
                current_scene_lines.append(line)
        
        if current_scene_lines:
            scenes.append("\n".join(current_scene_lines))
            
    else:
        # Fallback: Split by double newline (paragraphs)
        scenes = text.split('\n\n')

    # Filter empty
    scenes = [s for s in scenes if len(s.strip()) > 50]
    
    if not scenes:
        return [""] * num_chunks

    # 2. Distribute Scenes into Chunks
    # We enforce exactly num_chunks if possible.
    total_words = sum(count_words(s) for s in scenes)
    target_words_per_chunk = math.ceil(total_words / num_chunks)
    
    chunks = []
    current_chunk = ""
    current_count = 0
    
    # If we have very few scenes (e.g. 1 big one), we must split by length
    if len(scenes) < num_chunks:
        # Split purely by word count
        all_text = "\n\n".join(scenes)
        words = all_text.split()
        total = len(words)
        chunk_size = math.ceil(total / num_chunks)
        for i in range(0, total, chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
            
    else:
        # Scene-based distribution
        for scene in scenes:
            s_words = count_words(scene)
            
            # If adding preserves balance better than starting new, add.
            # Logic: Try to fill bucket.
            if current_count + s_words > target_words_per_chunk:
                # Decide whether to overflow or break
                # If current bucket is empty, we must take it (or split it?)
                if not current_chunk:
                    chunks.append(scene)
                    continue
                
                # If we are close to target, break
                # (Simple greedy packing)
                chunks.append(current_chunk)
                current_chunk = scene
                current_count = s_words
            else:
                current_chunk += "\n\n" + scene
                current_count += s_words
        
        if current_chunk:
            chunks.append(current_chunk)
    
    # Resize to exactly num_chunks (Force fit)
    if len(chunks) < num_chunks:
        # Pad with empty
        while len(chunks) < num_chunks:
            chunks.append("")
    elif len(chunks) > num_chunks:
        # Merge tail
        base_chunks = chunks[:num_chunks-1]
        tail = "\n\n".join(chunks[num_chunks-1:])
        base_chunks.append(tail)
        chunks = base_chunks
        
    return chunks

# --- CORE CONVERTER ---

def setup_gemma_director():
    """Updates TextEngine to use Local Gemma Director Adapter."""
    # Force Local Mode logic similar to movie_producer
    os.environ["TEXT_ENGINE"] = "local_gemma"
    
    # Reset singleton to ensure clean slate
    import text_engine
    text_engine._ENGINE = None
    
    engine = TextEngine()
    engine.backend = "local_gemma"
    
    # Resolve Model Path
    gemma_conf = definitions.MODAL_REGISTRY[definitions.Modality.TEXT].get("gemma-2-9b-it-director")
    if gemma_conf:
        if gemma_conf.path: engine.local_model_path = gemma_conf.path
        if gemma_conf.adapter_path: engine.local_adapter_path = gemma_conf.adapter_path
        logging.info(f"   🎬 Director Adapter Loaded: {gemma_conf.adapter_path}")
    
    engine._init_local_model()
    return engine

# --- GENESIS HELPERS ---

def load_random_inspiration():
    """Returns (ATU_Theme, TLP_Axiom)"""
    atu_theme = "ATU 300: The Dragon Slayer"
    tlp_axiom = "The world is everything that is the case."
    
    # Load ATU
    atu_path = MV_ROOT / "atui_235.md"
    if atu_path.exists():
        try:
            full = atu_path.read_text(encoding='utf-8')
            lines = [l.strip() for l in full.split('\n') if "* **ATU" in l]
            if lines: atu_theme = random.choice(lines).replace("* **", "").replace("**", "")
        except: pass
        
    # Load TLP
    tlp_path = MV_ROOT / "tlp.md"
    if tlp_path.exists():
        try:
            full = tlp_path.read_text(encoding='utf-8')
            # Paragraphs in TLP are separated by blank lines. We want a juicy one.
            paras = [p.strip() for p in full.split('\n\n') if len(p) > 20 and not p[0].isdigit()]
            if paras: tlp_axiom = random.choice(paras)
        except: pass
            
    return atu_theme, tlp_axiom

def process_file(input_path, args):
    # GENESIS MODE CHECK
    genesis_mode = False
    content = ""
    filename = "Ex_Nihilo_Project"
    
    if not input_path or input_path == "GENERATE":
        genesis_mode = True
        logging.info("✨✨✨ GENESIS MODE ACTIVATED ✨✨✨")
        atu, tlp = load_random_inspiration()
        logging.info(f"   📚 Theme: {atu}")
        logging.info(f"   🧠 Philosophy: {tlp}")
        
        # Default duration 1h 40m if not set
        if not args.slength: args.slength = 6000
    else:
        filename = Path(input_path).name
        logging.info(f"📄 Ingesting: {filename}")
        try:
            content = Path(input_path).read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logging.error(f"Failed to read file: {e}")
            return

    # 2. Config Engines
    if args.vpform in ["parody-movie", "comedy", "thriller", "movies-movie"] or args.local or genesis_mode:
        logging.info("   🧠 Initializing GemmaW (Director Mode)...")
        engine = setup_gemma_director()
    else:
        engine = get_engine()

    # 3. Analyze / Genre Concept
    rewrite_target = None
    if args.vpform in ["parody-movie", "comedy", "thriller"]:
        rewrite_target = args.vpform
    # In Genesis mode, if no vpform specified, default to "Drama" or use args.vpform
    if genesis_mode and not rewrite_target:
        rewrite_target = args.vpform if args.vpform != "standard" else "Cinematic Drama"
        
    mapping = {}
    title = filename.replace(".txt","").replace("_", " ").title()
    logline = f"The story of {title}."
    
    if genesis_mode:
        # GENESIS CONCEPT
        title = "Untitled Genesis Project"
        genre_display = rewrite_target.replace("-", " ").title()
        
        logging.info(f"   ⚛️  Synthesizing Concept ({genre_display})...")
        prompt = f"""
        CREATE A MOVIE CONCEPT from scratch.
        Genre: {genre_display}
        Core Theme (Aarne-Thompson-Uther): {atu}
        Philosophical Undertone (Wittgenstein): "{tlp}"
        
        Task:
        1. Invent a Title.
        2. Write a Logline.
        3. Create a Cast of Characters (Names & Roles).
        
        OUTPUT JSON: {{ "new_title": "...", "logline": "...", "characters": ["Name (Role)", ...] }}
        """
        try:
            raw = engine.generate(prompt, json_schema=True)
            data = json.loads(engine._clean_json_output(raw))
            if isinstance(data, list) and data: data = data[0]
            
            title = data.get("new_title", "Genesis Movie")
            logline = data.get("logline", "A movie created from nothing.")
            mapping = {c.split('(')[0].strip(): c for c in data.get("characters", [])}
            logging.info(f"      📽️  Title: {title}")
            logging.info(f"      📝 Logline: {logline}")
        except Exception as e:
            logging.warning(f"   ⚠️ Genesis Concept Failed: {e}")
            
    elif rewrite_target:
        # REWRITE CONCEPT (Existing Logic)
        genre_display = rewrite_target.replace("-", " ").title()
        if rewrite_target == "parody-movie": genre_display = "Comedic Parody"
        
        logging.info(f"   🎭 Generating {genre_display} Concept...")
        
        rename_instruction = "Create unique, creative names for ALL characters and Major Locations to fit the new genre."
        
        prompt = f"""
        Analyze this script snippet and create a concept for a {genre_display} version.
        Original Title: {title}
        Snippet: {content[:4000]}
        
        Task: 
        1. Create a new Title.
        2. Create a Logline that shifts the genre to {genre_display}.
        3. {rename_instruction}
        
        OUTPUT JSON: {{ "new_title": "...", "logline": "...", "mapping": {{ "OldName": "NewName", "OldLocation": "NewLocation" }} }}
        """
        try:
            raw = engine.generate(prompt, json_schema=True)
            data = json.loads(engine._clean_json_output(raw))
            # Handle list wrapper if any
            if isinstance(data, list) and data: data = data[0]
            
            title = data.get("new_title", data.get("parody_title", f"{genre_display} of {title}"))
            logline = data.get("logline", logline)
            mapping = data.get("mapping", {})
            logging.info(f"      start -> {title}")
        except Exception as e:
            logging.warning(f"   ⚠️ Concept Gen Failed: {e}")

    # 4. Chunk into 24 Beats
    logging.info("   🔪 Chunking into 24 Hero's Journey Beats...")
    beats_def = load_hj24()
    
    if genesis_mode:
        chunks = [""] * 24 # Empty chunks, we will generate content
    else:
        chunks = smart_chunk_script(content, num_chunks=24)
    
    processed_segs = []
    
    # 5. Process Beats
    for i, chunk in enumerate(chunks):
        beat_info = beats_def[i] if i < len(beats_def) else {"name": f"Beat {i+1}", "desc": ""}
        
        if genesis_mode:
            # GENESIS WRITING
            logging.info(f"      ✍️  Writing Beat {i+1}: {beat_info['name']}...")
            
            # Context for continuity (simplified)
            prev_context = ""
            if processed_segs:
                prev_context = f"PREVIOUSLY: {processed_segs[-1]['script_content'][-500:]}"
            
            prompt = f"""
            WRITE A SCENE for the movie "{title}".
            Genre: {genre_display if 'genre_display' in locals() else args.vpform}
            Beat Function: {beat_info['name']} ({beat_info['desc']})
            Characters: {list(mapping.values())}
            
            Philosophical Constraint: Influence the dialogue/structure with: "{tlp}"
            
            {prev_context}
            
            Action: Write the script for this scene. 
            Format: Standard Screenplay.
            """
            script_text = engine.generate(prompt, temperature=0.85)
            
        elif not chunk.strip():
            logging.warning(f"      Beat {i+1} empty. Padding.")
            script_text = "(Action)\nTime passes..."
        else:
            if rewrite_target:
                # Rewrite
                logging.info(f"      Rewriting Beat {i+1}: {beat_info['name']}...")
                
                # Dynamic Instruction
                if rewrite_target == "comedy":
                    mode_instr = "Rewrite this scene to be a HILARIOUS COMEDY. If it's already funny, make it funnier."
                elif rewrite_target == "thriller":
                    mode_instr = "Rewrite this scene to be a HIGH-STAKES THRILLER. Increase tension, suspense, and danger."
                else:
                    mode_instr = "Rewrite this scene to be a COMEDIC PARODY."

                prompt = f"""
                {mode_instr}
                New Title: {title}
                Function: {beat_info['name']} ({beat_info['desc']})
                Mappings: {mapping} (Use these new names/locations consistently)
                
                Original Content:
                {chunk[:12000]}
                
                OUTPUT: Standard Screenplay Format Only.
                """
                script_text = engine.generate(prompt, temperature=0.9 if rewrite_target == "comedy" else 0.8)
            else:
                # Standard formatting/cleanup could go here, but raw is fine for 'ingest' 
                # unless we want to convert purely to dialogue lines?
                # User said "full line-by-line... XMVP-twin".
                # For basic ingestion, preserving text in the segment is mostly enough, 
                # but let's do a light cleanup pass if we want high quality.
                # For now, pass raw chunk as the script content.
                script_text = chunk

        # MEMORY OPTIMIZATION: Aggressive Cleanup
        if hasattr(engine, "clear_cache"):
             # Double Tap: Clear Metal Cache then Python GC
             engine.clear_cache()
             
        import gc
        gc.collect()

        processed_segs.append({
            "id": i+1,
            "prompt": f"{title} - Beat {i+1}: {beat_info['name']}",
            "action": "cinematic",
            "script_content": script_text
        })

    # 6. Assembly
    logging.info("   📦 Assembling XMVP...")
    
    # Default duration: if slength provided, use it. Else default to 2hrs (7200s) or based on word count?
    # User said "if slength... retcon that bad-boy".
    target_duration = args.slength if args.slength else 7200
    
    bible = {
        "constraints": {"max_duration_sec": target_duration, "fps": 24},
        "scenario": f"The world of {title}",
        "situation": logline,
        "vision": "Cinematic, High Production Value"
    }
    
    story = {
        "title": title,
        "synopsis": logline,
        "characters": list(mapping.values()) if (rewrite_target or genesis_mode) else ["Protagonist"],
        "theme": "Hero's Journey"
    }
    
    segs_manifest = []
    for p in processed_segs:
        segs_manifest.append({
            "id": p["id"],
            "start_frame": 0,
            "end_frame": 0, # Calculated later or irrelevent for text-first
            "prompt": p["prompt"],
            "action": p["action"],
            "script_content": p["script_content"]
        })
        
    manifest = {
        "segs": segs_manifest,
        "files": {},
        # Ideally parsing dialogue lines here would be great but complex.
        # We store the script_content in Segs for now. content_producer might need an upgrade to read Seg script_content if DialogueScript is missing.
        # But 'run_fullmovie_still_mode' expects manifest.dialogue.lines.
        # TODO: A parser that turns script_content into manifest.dialogue lines is needed for full operability.
        # I'll add a quick regex parser for that.
    }
    
    # 6b. Rough Dialogue Parser (for FullMovie compat)
    dialogue_lines = []
    total_offset = 0.0
    
    for seg in segs_manifest:
        raw_lines = seg["script_content"].split('\n')
        last_char = None
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            
            # Simple heuristic: Character name is UPPERCASE, short, no periods (usually)
            # Or formatted like "Character Name:"
            if line.isupper() and len(line) < 50 and not line.endswith('.'):
                last_char = line
            elif line.strip().endswith(':') and len(line) < 50:
                last_char = line.strip()[:-1].upper()
            elif last_char:
                # This is dialogue
                dialogue_lines.append({
                    "character": last_char,
                    "text": line,
                    "start_offset": total_offset,
                    "visual_focus": last_char,
                    "action": "Speaking"
                })
                # Estimate duration
                dur = len(line.split()) * 0.4 # approx 0.4s per word
                total_offset += dur
                last_char = None # Reset
                
    if dialogue_lines:
        manifest["dialogue"] = {"lines": dialogue_lines}
    
    # Meta
    meta = {
        "source_file": str(filename),
        "hj24_map": True,
        "parody": bool(rewrite_target),
        "genesis_mode": genesis_mode,
        "atu_theme": atu if 'atu' in locals() else None,
        "tlp_axiom": tlp if 'tlp' in locals() else None
    }
    
    # 7. Save
    safe_name = "".join([c for c in title if c.isalnum() or c==' ']).replace(" ", "_")
    output_path = Path(args.out) / f"{safe_name}.xml" if args.out else Path(".").resolve() / f"{safe_name}.xml" # Default to current dir if no input path/parent
    
    if args.input_file and args.input_file != "GENERATE":
         output_path = Path(args.out) / f"{safe_name}.xml" if args.out else Path(input_path).parent / f"{safe_name}.xml"
    
    full_data = {
        "Bible": bible,
        "Story": story,
        "Manifest": manifest,
        "Meta": meta
    }
    
    save_xmvp(full_data, output_path)
    logging.info(f"✅ XMVP Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="XMVP Universal Script Converter")
    parser.add_argument("input_file", nargs='?', help="Path to input script (txt, md). Optional for Genesis Mode.")
    parser.add_argument("--vpform", type=str, default="standard", help="Form (parody-movie, comedy, thriller, standard)")
    parser.add_argument("--slength", type=float, help="Target Duration (Retcon)")
    parser.add_argument("--out", type=str, help="Output Directory")
    parser.add_argument("--local", action="store_true", help="Force Local Models")
    
    args = parser.parse_args()
    
    if not args.input_file:
         # Implicit Genesis Mode
         args.input_file = "GENERATE"
    
    if args.input_file != "GENERATE" and not os.path.exists(args.input_file):
        logging.error(f"File not found: {args.input_file}")
        return
        
    process_file(args.input_file, args)

if __name__ == "__main__":
    main()
