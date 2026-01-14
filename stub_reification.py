import argparse
import logging
import json
import random
import sys
from pathlib import Path
from mvp_shared import CSSV, Story, load_cssv, load_api_keys
from text_engine import get_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synthesize_story(cssv: CSSV) -> Story:
    """
    Synthesizes a Story object from a CSSV bible using the configured Text Engine.
    """
    logging.info(f"📖 Synthesizing Story for: {cssv.situation[:50]}...")

    prompt = f"""
    You are a Screenwriter. Convert this CSSV Bible into a linear Story JSON.
    
    CSSV DATA:
    - SCENARIO: {cssv.scenario}
    - SITUATION: {cssv.situation}
    - VISION: {cssv.vision}
    - CONSTRAINTS: {cssv.constraints.model_dump_json()}
    
    OUTPUT SCHEMA (JSON Only):
    {{
      "title": "String",
      "logline": "String",
      "acts": [
        {{ "name": "Act 1", "summary": "..." }}
      ],
      "characters": [ ... ]
    }}
    """
    
    logging.info("🧠 Sends Prompt to Text Engine...")
    engine = get_engine()
    response_text = engine.generate(prompt, temperature=0.7)
    
    if not response_text:
        logging.error("[-] Text Engine returned empty response.")
        # Return fallback/stub story
        return Story(
            title="Generation Failed",
            synopsis=cssv.situation,
            characters=["Unknown"],
            theme="Error"
        )
        
    # Clean and Parse JSON
    try:
        # Remove markdown fences
        clean = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        
        # Handle list wrapping
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                 pass

        return Story(**data)
        
    except Exception as e:
        logging.error(f"[-] JSON Parse Error from Engine: {e}")
        logging.error(f"Raw Output: {response_text[:200]}...")
        return Story(
            title="Parse Error",
            synopsis=cssv.situation,
            characters=["Unknown"],
            theme="Error"
        )

def run_stub(bible_path: str, out_path: str = "story.json", request: str = None) -> bool:
    """
    Executes the Stub Reification pipeline.
    """
    # 1. Load Bible
    try:
        cssv = load_cssv(bible_path)
    except Exception as e:
        logging.error(f"Failed to load bible: {bible_path}")
        return False

    # 2. (Keys loaded internally by TextEngine if needed)
    
    # 3. Reify
    logging.info(f"✍️  Reifying Story from: {cssv.situation[:50]}...")
    
    # Apply request modifier if present (simple append to vision for now)
    if request:
        cssv.vision += f" [NOTE: {request}]"

    story = synthesize_story(cssv)
    
    # 4. Save
    with open(out_path, 'w') as f:
        f.write(story.model_dump_json(indent=2))
        
    logging.info(f"✅ Story Created: {story.title}")
    logging.info(f"   Synopsis: {story.synopsis}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Stub Reification: The Writer")
    parser.add_argument("--bible", type=str, default="bible.json", help="Path to input CSSV JSON")
    parser.add_argument("--out", type=str, default="story.json", help="Output path for Story JSON")
    parser.add_argument("--req", type=str, help="Optional extra request/notes to modify the bible on the fly")
    parser.add_argument("--xb", type=str, help="Path to XMVP XML file (Overrides --bible)")
    
    args = parser.parse_args()
    
    # Logic: If --xb is present, we must extract the Bible from it first.
    bible_path = args.bible
    
    if args.xb:
        logging.info(f"📚 Loading Bible from XMVP: {args.xb}")
        from mvp_shared import load_xmvp, save_cssv
        bible_json = load_xmvp(args.xb, "Bible")
        if bible_json:
             # Save to temp bible.json so existing logic works smoothly?
             # Or modify 'run_stub' to take an object.
             # Ideally run_stub stays file based for simplicity in pipeline.
             # We overwrite the 'args.bible' path or a temp one.
             if args.bible == "bible.json": 
                 # use default
                 pass
             
             with open(args.bible, "w") as f:
                 f.write(bible_json)
             logging.info(f"   -> Extracted to: {args.bible}")
             bible_path = args.bible
        else:
             logging.error("Failed to extract <Bible> from XMVP.")
             sys.exit(1)
    
    success = run_stub(bible_path, args.out, args.req)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
