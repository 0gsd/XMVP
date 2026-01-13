import argparse
import logging
import json
import random
import sys
from pathlib import Path
from google import genai
from google.genai import types
from mvp_shared import CSSV, Story, load_cssv, load_api_keys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synthesize_story(cssv: CSSV, api_key: str) -> Story:
    """
    Uses Gemini to verify/flesh out the arc.
    """
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    CONTEXT: You are a Master Story Architect. We are entering pre-production.
    
    VISION BIBLE (CSSV):
    - SCENARIO: {cssv.scenario}
    - SITUATION: {cssv.situation}
    - VISION: {cssv.vision}
    - FORM constraints: {cssv.constraints.model_dump_json()}
    
    TASK: 
    Crystallize this information into a concrete Narrative Arc ("The Story").
    Even for a 30s commercial, we need a "Story" (Beginning/Hook, Middle/Solution, End/CallToAction).
    
    OUTPUT JSON (Strict):
    {{
        "title": "A catchy working title",
        "synopsis": "A 2-3 sentence summary of the arc.",
        "characters": ["List of key visual subjects/characters"],
        "theme": "The underlying emotional or selling theme"
    }}
    """
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        text = resp.text.strip()
        # Basic cleanup just in case
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0]
        
        data = json.loads(text)
        
        # Handle list wrapping (sometimes Gemini returns [{}])
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
               # Try to find a dict inside?
               pass 

        return Story(**data)
        
    except Exception as e:
        logging.error(f"Story synthesis failed: {e}")
        # Fallback stub
        return Story(
            title="Untitled Production",
            synopsis=cssv.situation,
            characters=["Unknown"],
            theme="General"
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

    # 2. Load Keys
    keys = load_api_keys()
    if not keys:
        logging.error("No API keys found in env_vars.yaml")
        return False
    
    # 3. Reify
    logging.info(f"✍️  Reifying Story from: {cssv.situation[:50]}...")
    # TODO: Add request modifier logic if needed
    story = synthesize_story(cssv, random.choice(keys))
    
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
