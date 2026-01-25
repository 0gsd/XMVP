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

def get_best_inspiration(engine=None, content_snippet=None):
    """
    Returns (ATU_Theme, TLP_Axiom)
    If engine & content_snippet provided, asks LLM to pick best ATU.
    Else, random.
    """
    atu_theme = "ATU 300: The Dragon Slayer"
    tlp_axiom = "The world is everything that is the case."
    
    # Load ATU
    atu_options = []
    atu_path = MV_ROOT / "z_training_data" / "atui_235.md"
    if atu_path.exists():
        try:
            full = atu_path.read_text(encoding='utf-8')
            atu_options = [l.replace("*", "").strip() for l in full.split('\n') if "**ATU" in l]
        except: pass

    # Load TLP
    tlp_options = []
    tlp_path = MV_ROOT / "z_training_data" / "tlp.md"
    if tlp_path.exists():
        try:
            full = tlp_path.read_text(encoding='utf-8')
            tlp_options = [p.strip() for p in full.split('\n\n') if len(p) > 20 and not p[0].isdigit()]
        except: pass

    # Smart Selection (if Engine available and Content exists)
    if engine and content_snippet and atu_options:
        logging.info("   🧠 Analyzing Script for Best ATU Theme...")
        prompt = f"""
        ANALYZE this story summary and SELECT the single most appropriate Aarne-Thompson-Uther (ATU) Tale Type from the list below.
        
        Story Snippet:
        {content_snippet[:3000]}
        
        Available Themes:
        {json.dumps(atu_options)} 
        (Pick best match)
        
        Task: Return ONLY the selected ATU string (e.g. "ATU 300: The Dragon Slayer").
        """
        try:
            # Quick cheaper call
            selected = engine.generate(prompt, temperature=0.3)
            if "ATU" in selected:
                # Cleanup
                for opt in atu_options:
                    if opt in selected:
                        atu_theme = opt
                        break
                # Fallback if fuzzy match failed but string looks right
                if "ATU" in selected and len(selected) < 100:
                     atu_theme = selected.strip().strip('"')
            logging.info(f"   🎯 Targeted Theme: {atu_theme}")
        except Exception as e:
            logging.warning(f"   ⚠️ Smart Theme Select Failed: {e}")
    else:
        # Random (Genesis or Fallback)
        import time
        random.seed(time.time())
        if atu_options: atu_theme = random.choice(atu_options)
        
    if tlp_options:
        tlp_axiom = random.choice(tlp_options)
            
    return atu_theme, tlp_axiom

    return atu_theme, tlp_axiom

def get_celebrity_anchor(gender):
    """Returns a visual anchor (deceased historical actor) for consistent facial generation."""
    mou = [
        "Humphrey Bogart", "Cary Grant", "James Stewart", "Marlon Brando", "Alec Guinness",
        "Richard Burton", "Peter O'Toole", "Montgomery Clift", "John Cazale", "Philip Seymour Hoffman",
        "Oliver Reed", "James Mason", "Buster Keaton", "Rudolph Valentino", "Orson Welles",
        "Leslie Howard", "Fredric March", "Charles Laughton", "Robert Mitchum", "Henry Fonda"
    ]
    fem = [
        "Katharine Hepburn", "Bette Davis", "Audrey Hepburn", "Ingrid Bergman", "Grace Kelly",
        "Marilyn Monroe", "Vivien Leigh", "Elizabeth Taylor", "Judy Garland", "Lauren Bacall",
        "Joan Crawford", "Barbara Stanwyck", "Greta Garbo", "Ava Gardner", "Hedy Lamarr",
        "Gene Tierney", "Rita Hayworth", "Mochelle Pfeiffer", "Gena Rowlands", "Jeanne Moreau"
    ]
    pool = mou if gender == "Male" else fem
    anchor = random.choice(pool)
    # Add random age for variety 
    age = random.choice(["25", "35", "45", "55", "65"])
    return f"{anchor} at age {age}"

def process_file(input_path, args):
    # GENESIS MODE CHECK
    genesis_mode = False
    content = ""
    filename = "Ex_Nihilo_Project"
    
    if not input_path or input_path == "GENERATE":
        genesis_mode = True
        logging.info("✨✨✨ GENESIS MODE ACTIVATED ✨✨✨")
        
        # Genesis: No content yet, random inspiration
        # For Painter mode, we do individual assignment later, so skip global assignment logging to avoid confusion
        if args.vpform != "painter":
            atu, tlp = get_best_inspiration()
            logging.info(f"   📚 Theme: {atu}")
            logging.info(f"   🧠 Philosophy: {tlp}")
        else:
            # Placeholder to be overwritten
            atu = "Multifaceted Painter Themes"
            tlp = "Multifaceted Wittgenstein Propositions"
        
        # Default duration 1h 40m if not set
        if not args.slength: args.slength = 6000
    else:
        path_obj = Path(input_path)
        filename = path_obj.name
        logging.info(f"📄 Ingesting: {filename}")
        
        # --- XML RE-INGESTION SUPPORT ---
        if path_obj.suffix.lower() == '.xml':
            logging.info("   🔄 Detected XMVP XML input. Flattening to text for re-processing...")
            try:
                # Load the XML using shared loader or manual extraction
                # Since we want to use xmvp_converter logic on the CONTENT, we extract prompts.
                from mvp_shared import load_xmvp
                # load_xmvp returns a dict structure
                # We need to use load_xmvp carefully or just regex/json parse plain text to avoid class issues
                # Let's try to just read it as text and extract the json blocks
                raw_xml = path_obj.read_text(encoding='utf-8', errors='ignore')
                
                # Extract Manifest JSON
                import re
                m_match = re.search(r'<Manifest>(.*?)</Manifest>', raw_xml, re.DOTALL)
                if m_match:
                     m_json = json.loads(m_match.group(1))
                     segs = m_json.get('segs', [])
                     # Reconstruct script from prompts or script_content
                     lines = []
                     for s in segs:
                         # Use script_content if available and robust, else prompt
                         txt = s.get('script_content', '')
                         if not txt or len(txt) < 5:
                             txt = s.get('prompt', '')
                         lines.append(txt)
                     
                     content = "\n\n".join(lines)
                     logging.info(f"   ✅ Extracted {len(segs)} segments as source text.")
                else:
                     logging.error("   ❌ Could not parse <Manifest> in XML.")
                     return
                     
            except Exception as e:
                logging.error(f"   ❌ XML Extraction Failed: {e}")
                return
        # --------------------------------
        else:
            try:
                content = path_obj.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logging.error(f"Failed to read file: {e}")
                return
            
        # Standard: Inspiration happens LATER after engine init so we can pass content
        atu = None
        tlp = None

    # 2. Config Engines
    if args.vpform in ["parody-movie", "comedy", "thriller", "movies-movie", "painter"] or args.local or genesis_mode:
        logging.info("   🧠 Initializing GemmaW (Director Mode)...")
        engine = setup_gemma_director()
    else:
        engine = get_engine()

    # 2b. Smart Inspiration (Standard Mode)
    if not genesis_mode and not atu:
         atu, tlp = get_best_inspiration(engine, content)
         logging.info(f"   📚 Smart-Selected Theme: {atu}")
         logging.info(f"   🧠 Philosophy: {tlp}")

    # 3. Analyze / Genre Concept
    rewrite_target = None
    if args.vpform in ["parody-movie", "comedy", "thriller", "painter"]:
        rewrite_target = args.vpform
    # In Genesis mode, if no vpform specified, default to "Drama" or use args.vpform
    if genesis_mode and not rewrite_target:
        rewrite_target = args.vpform if args.vpform != "standard" else "Cinematic Drama"
        
    mapping = {}
    title = filename.replace(".txt","").replace("_", " ").title()
    logline = f"The story of {title}."
    
    if genesis_mode:
        if rewrite_target == "painter":
             # PAINTER GENRE HANDLING (Genesis)
             logging.info("   🎨 Synthesizing PAINTER Concept (Pinter-esque)...")
             
             # 1. Characters: 2 to 6, equal split
             num_chars = random.choice([2, 4, 6]) 
             half = num_chars // 2
             genders = ["Male"] * half + ["Female"] * half
             random.shuffle(genders)
             
             # 2. Assign Invisibles (ATU + TLP)
             all_atus = []
             atu_path = MV_ROOT / "z_training_data" / "atui_235.md"
             if atu_path.exists():
                 try: 
                     all_atus = [l.replace("*", "").strip() for l in atu_path.read_text(encoding='utf-8').split('\n') if "**ATU" in l]
                 except: pass
             
             all_tlps = []
             tlp_path = MV_ROOT / "z_training_data" / "tlp.md"
             if tlp_path.exists():
                try:
                    all_tlps = [p.strip() for p in tlp_path.read_text(encoding='utf-8').split('\n\n') if len(p) > 20 and not p[0].isdigit()]
                except: pass
                
             # Safe sampling
             if len(all_atus) < num_chars: all_atus = ["Generic Theme"] * num_chars
             if len(all_tlps) < num_chars: all_tlps = ["Generic Axiom"] * num_chars
             
             assigned_atus = random.sample(all_atus, num_chars)
             assigned_tlps = random.sample(all_tlps, num_chars)
             
             painter_chars = []
             for i in range(num_chars):
                 painter_chars.append({
                     "gender": genders[i],
                     "id": i,
                     "atu": assigned_atus[i],
                     "tlp": assigned_tlps[i]
                 })
                 
             # 3. Prompt for Names
             char_prompt = f"""
             Create {num_chars} characters for a Harold Pinter style play.
             Format: JSON list of objects {{ "name": "...", "role": "...", "id": ... }}
             Constraints:
             - {half} Men, {half} Women.
             - Roles should be vague (e.g. "The Lodger", "The Wife", "The Visitor").
             - Names should be simple, British (e.g. Stanley, Meg, Goldberg).
             """
             try:
                 raw_c = engine.generate(char_prompt, json_schema=True)
                 char_data = json.loads(engine._clean_json_output(raw_c))
                 if isinstance(char_data, dict) and "characters" in char_data: char_data = char_data["characters"]
                 
                 final_chars = []
                 for i, c_def in enumerate(char_data[:num_chars]):
                     p_char = painter_chars[i]
                     # Assign Visual Anchor
                     anchor = get_celebrity_anchor(p_char["gender"])
                     final_chars.append({
                         "name": c_def.get("name", f"Person {i}"),
                         "role": c_def.get("role", "Participant"),
                         "gender": p_char["gender"],
                         "anchor": anchor,
                         "secret_atu": p_char["atu"],
                         "secret_tlp": p_char["tlp"]
                     })
                 
                 # {Name (Role) [Gender] {Anchor}}
                 mapping = {c["name"]: f"{c['name']} ({c['role']}) [{c['gender']}] {{{c['anchor']}}}" for c in final_chars}
                 painter_context = final_chars 
                 
                 # Generate a unique Pinter-esque title
                 title_prompt = "Invent a title for a Harold Pinter style play. It should be obscure, mundane, or ominous. Do NOT use existing Pinter titles (e.g. No 'Homecoming', 'Caretaker', 'Birthday Party'). Examples: 'The Dumb Waiter', 'Celebration', 'The Hothouse', 'Family Voices', 'A Kind of Alaska'. Output ONLY the title."
                 try:
                     raw_title = engine.generate(title_prompt, json_schema=False).strip().strip('"').strip("'")
                     if len(raw_title) > 5:
                         title = raw_title
                     else:
                         title = f"The {random.choice(['Room', 'Party', 'Homecoming', 'Caretaker', 'Collection', 'Lover', 'Tea Party'])} at No. {random.randint(1,99)}"
                 except:
                      title = f"The {random.choice(['Room', 'Party', 'Homecoming', 'Caretaker', 'Collection', 'Lover', 'Tea Party'])} at No. {random.randint(1,99)}"
                 
                 logline = f"A group of {num_chars} people in a room, avoiding the truth."
                 
                 logging.info(f"      🎭 Pinter Cast Generated:")
                 for c in final_chars:
                     logging.info(f"         👤 {c['name']} ({c['role']}) [{c['gender']}]")
                     logging.info(f"            SECRET ATU: {c['secret_atu']}")
                     logging.info(f"            SECRET TLP: {c['secret_tlp']}")
                 
             except Exception as e:
                 logging.warning(f"Failed to generate painter characters: {e}")
                 mapping = {"Person A": "Person A (Unknown)", "Person B": "Person B (Unknown)"}
                 painter_context = [
                    {"name": "Person A", "gender": "Male", "secret_atu": "Generic", "secret_tlp": "Generic"},
                    {"name": "Person B", "gender": "Female", "secret_atu": "Generic", "secret_tlp": "Generic"}
                 ]
        else:
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
        # PAINTER GENRE HANDLING
        if rewrite_target == "painter":
             logging.info("   🎨 Synthesizing PAINTER Concept (Pinter-esque)...")
             
             # 1. Characters: 2 to 6, equal split
             num_chars = random.choice([2, 4, 6]) # user requested 2, 4, or 6
             # Logic: "equal number of men and women if the number is even" - 
             # Since we only chose even numbers (2,4,6), we always split equally.
             half = num_chars // 2
             genders = ["Male"] * half + ["Female"] * half
             random.shuffle(genders)
             
             # 2. Assign Invisibles (ATU + TLP)
             # We need to load ALL TLP/ATU options to sample unique ones
             # Re-using the logic from get_best_inspiration but forced bulk load
             all_atus = []
             atu_path = MV_ROOT / "z_training_data" / "atui_235.md"
             if atu_path.exists():
                 try: 
                     all_atus = [l.replace("*", "").strip() for l in atu_path.read_text(encoding='utf-8').split('\n') if "**ATU" in l]
                 except: pass
             
             all_tlps = []
             tlp_path = MV_ROOT / "z_training_data" / "tlp.md"
             if tlp_path.exists():
                try:
                    all_tlps = [p.strip() for p in tlp_path.read_text(encoding='utf-8').split('\n\n') if len(p) > 20 and not p[0].isdigit()]
                except: pass
                
             # Safe sampling
             if len(all_atus) < num_chars: all_atus = ["Generic Theme"] * num_chars
             if len(all_tlps) < num_chars: all_tlps = ["Generic Axiom"] * num_chars
             
             assigned_atus = random.sample(all_atus, num_chars)
             assigned_tlps = random.sample(all_tlps, num_chars)
             
             painter_chars = []
             for i in range(num_chars):
                 painter_chars.append({
                     "gender": genders[i],
                     "id": i,
                     "atu": assigned_atus[i],
                     "tlp": assigned_tlps[i]
                 })
                 
             # 3. Prompt for Names/Roles
             char_prompt = f"""
             Create {num_chars} characters for a Harold Pinter style play.
             Format: JSON list of objects {{ "name": "...", "role": "...", "id": ... }}
             Constraints:
             - {half} Men, {half} Women.
             - Roles should be vague (e.g. "The Lodger", "The Wife", "The Visitor").
             - Names should be simple, British (e.g. Stanley, Meg, Goldberg).
             """
             try:
                 raw_c = engine.generate(char_prompt, json_schema=True)
                 char_data = json.loads(engine._clean_json_output(raw_c))
                 if isinstance(char_data, dict) and "characters" in char_data: char_data = char_data["characters"]
                 
                 # Merge generated names with assigned invisibles
                 final_chars = []
                 for i, c_def in enumerate(char_data[:num_chars]):
                     # Find matching internal struct
                     p_char = painter_chars[i]
                     # Assign Visual Anchor
                     anchor = get_celebrity_anchor(p_char["gender"])
                     final_chars.append({
                         "name": c_def.get("name", f"Person {i}"),
                         "role": c_def.get("role", "Participant"),
                         "gender": p_char["gender"],
                         "anchor": anchor,
                         "secret_atu": p_char["atu"],
                         "secret_tlp": p_char["tlp"]
                     })
                 
                 # {Name (Role) [Gender] {Anchor}}
                 mapping = {c["name"]: f"{c['name']} ({c['role']}) [{c['gender']}] {{{c['anchor']}}}" for c in final_chars}
                 painter_context = final_chars # Save for beat loop
                 title = f"The {random.choice(['Room', 'Party', 'Homecoming', 'Caretaker', 'Collection', 'Lover', 'Tea Party'])} at No. {random.randint(1,99)}"
                 logline = f"A group of {num_chars} people in a room, avoiding the truth."
                 
                 logging.info(f"      🎭 Pinter Cast Generated:")
                 for c in final_chars:
                     logging.info(f"         👤 {c['name']} ({c['role']}) [{c['gender']}]")
                     logging.info(f"            SECRET ATU: {c['secret_atu']}")
                     logging.info(f"            SECRET TLP: {c['secret_tlp']}")
                 
             except Exception as e:
                 logging.warning(f"Failed to generate painter characters: {e}")
                 mapping = {"Person A": "Person A (Unknown)", "Person B": "Person B (Unknown)"}
                 painter_context = [
                    {"name": "Person A", "gender": "Male", "secret_atu": "Generic", "secret_tlp": "Generic"},
                    {"name": "Person B", "gender": "Female", "secret_atu": "Generic", "secret_tlp": "Generic"}
                 ]

        # REWRITE CONCEPT (Existing Logic)
        elif rewrite_target:
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
            
            if rewrite_target == "painter":
                 # PAINTER SPECIFIC GENERATION
                 # Pacing: 1 minute per page.
                 # Total Pages = slength / 60.
                 # Pages per beat (24 beats) = (slength/60) / 24.
                 total_pages = (args.slength if args.slength else 6000) / 60.0
                 pages_per_beat = max(0.5, total_pages / 24.0)
                 
                 # Location Logic: "one or at most two interior locations"
                 # Let's say beats 1-12 in Room A, 13-24 in Room B (if 2 rooms) or random?
                 # Pinter usually stays in one room. Let's flip a coin for a "Location Switch" at Midpoint (Beat 12).
                 loc_name = "The Living Room (Bland, Featureless)"
                 if i > 12 and random.random() > 0.5:
                      loc_name = "The Other Room (Equally Bland)"
                      
                 # Character Selection: "2 to 6 characters"
                 # In a beat, maybe not ALL are present?
                 # Let's pick a subset (at least 2) for the argument.
                 present_chars = random.sample(painter_context, k=random.randint(2, len(painter_context)))
                 
                 char_context_str = "\n".join([f"- {c['name']} ({c['gender']}): Driven by \"{c['secret_atu']}\" and believes \"{c['secret_tlp']}\"" for c in present_chars])
                 
                 prompt = f"""
                 WRITE A SCENE for a Harold Pinter style play.
                 Scene Length: EXACTLY {pages_per_beat:.1f} PAGES.
                 Setting: {loc_name}.
                 
                 Characters Present:
                 {char_context_str}
                 
                 Narrative Beat: {beat_info['name']} ({beat_info['desc']})
                 
                 STYLE GUIDE:
                 - The room is bland and featureless. Describe NOTHING about the decor.
                 - Structure: {pages_per_beat:.1f} pages of dialogue.
                 - Content: An inscrutable, witty, clever, complication argument.
                 - They are arguing about something trivial, but neither addresses the REAL issue directly.
                 - Use the characters' HIDDEN MOTIVATIONS (ATU/TLP) to guide their subtext, but NEVER state them allowed.
                 
                 {prev_context}
                 
                 Action: Write the script. Standard Screenplay Format.
                 """
                 script_text = engine.generate(prompt, temperature=0.88)
            
            else:
                # STANDARD GENESIS
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
                
                Philosophical Context (Brain Logic): "{tlp}"
                Thematic Undercurrent: "{atu}"

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

    # MEMORY OPTIMIZATION: Final Unload
    if hasattr(engine, "unload"):
        engine.unload()

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
    
    # Page Tracking State
    current_page = 0
    lines_on_page = 0
    LINES_PER_PAGE = 55
    
    for seg in segs_manifest:
        # Try JSON parsing first (New Logic)
        try:
             # Clean potential markdown fences
             clean_content = seg["script_content"]
             if "```json" in clean_content:
                 clean_content = clean_content.split("```json")[1].split("```")[0]
             elif "```" in clean_content:
                 clean_content = clean_content.split("```")[1].split("```")[0]
             
             data_block = json.loads(clean_content)
             
             # Locate Dialogue List
             dialogue_list_json = [] # Renamed to avoid conflict with outer dialogue_lines
             if "SCENE" in data_block:
                 s = data_block["SCENE"]
                 if isinstance(s, list) and len(s) > 0: s = s[0] # Handle [{...}]
                 if "DIALOGUE" in s:
                     dialogue_list_json = s["DIALOGUE"]
             elif "DIALOGUE" in data_block:
                 dialogue_list_json = data_block["DIALOGUE"]
             elif isinstance(data_block, list):
                 # Maybe it's just a list of lines?
                 dialogue_list_json = data_block

             if dialogue_list_json:
                 for d_item in dialogue_list_json:
                     char_name = d_item.get("CHARACTER", "Unknown").upper().strip()
                     line_text = d_item.get("LINE", "")
                     if char_name and line_text:
                         # Paging Logic (Rate Limiter: count every line of the script)
                         lines_on_page += 1
                         if lines_on_page >= LINES_PER_PAGE:
                             current_page += 1
                             lines_on_page = 0

                         dialogue_lines.append({
                             "character": char_name,
                             "text": line_text,
                             "start_offset": total_offset,
                             "visual_focus": char_name,
                             "action": "Speaking",
                             "page_index": current_page
                         })
                         # Estimate duration
                         dur = len(line_text.split()) * 0.4 # approx 0.4s per word
                         total_offset += dur
                 continue # Successfully parsed as JSON, skip fallback

        except Exception as e:
             # logging.debug(f"JSON Parse failed for segment {seg.get('id')}, falling back to text split: {e}")
             pass

        # Fallback: Plain Text Parsing (Legacy)
        raw_lines = seg["script_content"].split('\n')
        last_char = None
        for line in raw_lines:
            # Paging Logic (Rate Limiter: count every line of the script)
            lines_on_page += 1
            if lines_on_page >= LINES_PER_PAGE:
                current_page += 1
                lines_on_page = 0
                
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
                    "action": "Speaking",
                    "page_index": current_page
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
    # Append random suffix to avoid collisions (especially in batch/painter mode)
    safe_name = f"{safe_name}_{random.randint(1000, 9999)}"
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
