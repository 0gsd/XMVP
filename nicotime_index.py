#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime

# Local Imports
try:
    from text_engine import get_engine
    from mvp_shared import load_text_keys
except ImportError:
    # Allow running from root if needed?
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from text_engine import get_engine
    from mvp_shared import load_text_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sanitize_filename(name):
    """Sanitize prompt for filename."""
    return re.sub(r'[^\w\-_]', '_', name)[:50]

class NicotimeIndexer:
    def __init__(self, output_dir_name="nicotime"):
        self.engine = get_engine()
        # Default target: subfolder of z_training_data/nicotime
        # Find project root (assume we are in tools/fmv/mvp/v0.5)
        # We want to go up to METMroot? No, user said "subfolder under z_training_data".
        # Assuming z_training_data is relative to execution or project root.
        
        # Let's find z_training_data or create it in standard location relative to this script
        # Likely ../../../../../z_training_data if following structure?
        # Or usually <Root>/z_training_data
        
        # Heuristic: Look for "z_training_data" in cwd, parent, etc.
        self.output_root = None
        candidates = [Path("z_training_data"), Path("../z_training_data"), Path("../../z_training_data")]
        for c in candidates:
            if c.exists():
                self.output_root = c
                break
        
        if not self.output_root:
            # Create in cwd by default if not found
            self.output_root = Path("z_training_data")
            self.output_root.mkdir(exist_ok=True)
            
        self.target_dir = self.output_root / output_dir_name
        self.target_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"📂 NICOTIME Index Location: {self.target_dir.resolve()}")

    def distill_entities(self, prompt):
        """Phase 1: Distill prompt into Noospheric Entities."""
        logging.info(f"⚗️  Distilling: '{prompt}'...")
        
        system_prompt = (
            f"You are a Cultural Anthropologist and Noospheric Analyst. "
            f"Your task is to deconstruct the concept '{prompt}' into 3 to 5 atomic 'Noospheric Entities'. "
            f"An Entity is a fundamental unit of meaning, sensory vibe, or cultural baggage associated with the concept. "
            f"Return JSON strictly: {{ 'entities': [ {{ 'name': 'Entity Name', 'type': 'Vibe|Object|Social|Abstract' }} ] }}"
        )
        
        json_str = self.engine.generate(system_prompt, temperature=0.7, json_schema=True)
        try:
            data = json.loads(json_str)
            return data.get("entities", [])
        except Exception as e:
            logging.error(f"❌ Distillation Failed: {e}")
            return []

    def expand_entity(self, entity_name, context_prompt):
        """Phase 2: Expand specific entity with Zeitgeist/Skuddlebutt."""
        logging.info(f"   🔍 Expanding Entity: {entity_name}...")
        
        system_prompt = (
            f"Context: We are analyzing '{context_prompt}'.\n"
            f"Target Entity: '{entity_name}'.\n"
            f"Task: Provide deep semantic details for this entity.\n"
            f"Return JSON strictly: {{ "
            f"'definition': 'Standard definition', "
            f"'visual_semiotics': 'Visual/Sensory descriptions (colors, textures, lighting)', "
            f"'skuddlebutt': 'Rumors, urban legends, or colloquial associations', "
            f"'zeitgeist': 'How this concept is viewed historically vs now' "
            f"}}"
        )
         
        json_str = self.engine.generate(system_prompt, temperature=0.8, json_schema=True)
        try:
            return json.loads(json_str)
        except:
             return None

    def get_existing_indices(self):
        """Returns a list of concept names (sanitized) that already exist."""
        if not self.target_dir.exists():
            return []
        # Return filenames without extension
        return [f.stem for f in self.target_dir.glob("*.xml")]

    def extract_concepts_from_xmvp(self, xmvp_path, ignore_list=None):
        """Extracts list of key concepts from an XMVP file, avoiding existing ones."""
        logging.info(f"📂 Analyzing XMVP for concepts: {xmvp_path}")
        if ignore_list:
             logging.info(f"   🚫 Ignoring {len(ignore_list)} existing concepts.")

        try:
            # Simple Text Load
            with open(xmvp_path, 'r') as f:
                content = f.read()
                
            # Parse strictly for Story/Manifest content to avoid noise
            story_content = ""
            try:
                tree = ET.parse(xmvp_path)
                root_xml = tree.getroot()
                
                # Story
                story = root_xml.find("Story")
                if story is not None:
                     story_content += f"TITLE: {story.findtext('title', '')}\n"
                     story_content += f"SYNOPSIS: {story.findtext('synopsis', '')}\n"
            except:
                 pass
                 
            if not story_content:
                story_content = content[:8000] # Truncate if raw

            # Format ignore list for prompt
            ignore_prompt = ""
            if ignore_list:
                # Top 50 to avoid token overflow if list is huge
                ignore_prompt = f"- DO NOT include these concepts (already indexed): {', '.join(ignore_list[:50])}...\n"

            system_prompt = (
                f"Analyze the following movie metadata (Story/Synopsis).\n"
                f"Identify 50 to 100 unique, distinct NOUNS, LOCATIONS, or CONCEPTS that are central to this specific universe and require a definition.\n"
                f"Rules:\n"
                f"- Ignore generic words like 'Man', 'Room', 'Day'.\n"
                f"- Focus on proper nouns, specific locations (e.g. ' The Mall'), or thematic concepts (e.g. 'Teen Ennui').\n"
                f"{ignore_prompt}"
                f"- Return strictly a JSON object: {{ 'concepts': ['Concept A', 'Concept B', ...] }}\n\n"
                f"METADATA:\n{story_content}"
            )
            
            json_str = self.engine.generate(system_prompt, temperature=0.7, json_schema=True)
            data = json.loads(json_str)
            return data.get("concepts", [])
            
        except Exception as e:
            logging.error(f"❌ Failed to extract concepts from XMVP: {e}")
            return []

    def create_index(self, prompt):
        """Main workflow."""
        safe_name = sanitize_filename(prompt)
        
        # Double check existence here in case of race condition or manual run
        out_path = self.target_dir / f"{safe_name}.xml"
        if out_path.exists():
             logging.info(f"   ⏩ Skipping '{prompt}' (Already indexed at {safe_name}.xml)")
             return str(out_path)

        logging.info(f"🚀 Indexing: '{prompt}' -> {safe_name}")
        entities = self.distill_entities(prompt)
        
        if not entities:
            logging.error(f"Sketchy extraction. Skipping '{prompt}'.")
            return None

        # Build XML
        root = ET.Element("NicotimeIndex")
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "Prompt").text = prompt
        ET.SubElement(header, "Timestamp").text = datetime.now().isoformat()
        ET.SubElement(header, "Engine").text = self.engine.backend
        
        noosphere = ET.SubElement(root, "Noosphere")
        
        for ent in entities:
            name = ent.get('name')
            etype = ent.get('type', 'Unknown')
            
            details = self.expand_entity(name, prompt)
            
            e_node = ET.SubElement(noosphere, "Entity", type=etype)
            ET.SubElement(e_node, "Name").text = name
            
            if details:
                ET.SubElement(e_node, "Definition").text = details.get('definition', '')
                ET.SubElement(e_node, "VisualSemiotics").text = details.get('visual_semiotics', '')
                ET.SubElement(e_node, "Skuddlebutt").text = details.get('skuddlebutt', '')
                ET.SubElement(e_node, "Zeitgeist").text = details.get('zeitgeist', '')

        # Save
        filename = f"{safe_name}.xml"
        out_path = self.target_dir / filename
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        
        logging.info(f"✅ NICOTIME Index Saved: {out_path}")
        return str(out_path)

def main():
    parser = argparse.ArgumentParser(description="NICOTIME Indexer")
    parser.add_argument("prompt", nargs="?", help="Concept/Word/Phrase to index (Optional if --xb used)")
    parser.add_argument("--out", default="nicotime", help="Subfolder name")
    parser.add_argument("--xb", help="Path to XMVP XML file for batch processing")
    
    args = parser.parse_args()
    
    indexer = NicotimeIndexer(output_dir_name=args.out)
    
    if args.xb:
        if not os.path.exists(args.xb):
            logging.error(f"File not found: {args.xb}")
            sys.exit(1)
            
        existing = indexer.get_existing_indices()
        concepts = indexer.extract_concepts_from_xmvp(args.xb, ignore_list=existing)
        
        if not concepts:
             logging.info("🤷 No new concepts found to index.")
             sys.exit(0)
             
        logging.info(f"📜 Found {len(concepts)} NEW concepts to index: {concepts}")
        
        for c in concepts:
            indexer.create_index(c)
            
    elif args.prompt:
        indexer.create_index(args.prompt)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
