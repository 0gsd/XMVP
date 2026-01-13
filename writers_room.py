import argparse
import logging
import json
import random
import math
import sys
import re
from pathlib import Path
from google import genai
from google.genai import types
from mvp_shared import CSSV, Story, Portion, load_cssv, load_api_keys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_story(path: str) -> Story:
    with open(path, 'r') as f:
        return Story.model_validate_json(f.read())

def break_story(story: Story, cssv: CSSV, api_key: str) -> list[Portion]:
    """
    Splits the story into portions based on constraints.
    """
    total_duration = cssv.constraints.max_duration_sec
    seg_length = cssv.constraints.target_segment_length
    
    # Calculate required segments
    num_portions = math.ceil(total_duration / seg_length)
    
    logging.info(f"🧮 Calculation: {total_duration}s total / {seg_length}s seg = {num_portions} portions.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    CONTEXT: You are a Screenwriter breaking a story into scenes.
    
    STORY:
    Title: {story.title}
    Synopsis: {story.synopsis}
    Characters: {", ".join(story.characters)}
    
    CONSTRAINTS:
    - Total Duration: {total_duration}s
    - We need exactly {num_portions} sequential segments.
    - Each segment represents approx {seg_length} seconds of screen time.
    
    TASK:
    Write a specific visual description for each of the {num_portions} segments.
    They must flow continuously to tell the Story.
    
    OUTPUT FORMAT (JSON List):
    [
        {{ "id": 1, "content": "Description of segment 1..." }},
        {{ "id": 2, "content": "Description of segment 2..." }},
        ...
    ]
    """
    
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            text = resp.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0]
            
            data = json.loads(text)
            
            portions = []
            for item in data:
                portions.append(Portion(
                    id=item['id'],
                    duration_sec=seg_length, 
                    content=item['content']
                ))
            
            # Validation
            if len(portions) != num_portions:
                logging.warning(f"⚠️ Generated {len(portions)} portions, expected {num_portions}. Adjusting...")
                # Truncate or Pad? 
                # For now, just accept it, but warn.
            
            return portions
            
        except Exception as e:
            logging.warning(f"attempt {attempt+1} failed: {e}")
            
    return []

def run_writers(bible_path: str, story_path: str, out_path: str = "portions.json") -> bool:
    """
    Executes the Writers Room pipeline.
    """
    # 1. Load Data
    try:
        cssv = load_cssv(bible_path)
        story = load_story(story_path)
    except Exception as e:
        logging.error(f"Failed to load inputs: {e}")
        return False

    # 2. Load Keys
    keys = load_api_keys()
    if not keys:
        logging.error("No API keys found")
        return False
        
    # 3. Break Story
    logging.info(f"✍️  Breaking story '{story.title}' into scenes...")
    portions = break_story(story, cssv, random.choice(keys))
    
    if not portions:
        logging.error("Failed to generate portions.")
        return False
        
    # 4. Save
    output_data = {"portions": [p.model_dump() for p in portions]}
    
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    logging.info(f"✅ Script Written: {len(portions)} portions saved to {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Writers Room: The Screenwriter")
    parser.add_argument("--bible", type=str, required=True, help="Path to input CSSV JSON")
    parser.add_argument("--story", type=str, required=True, help="Path to input Story JSON")
    parser.add_argument("--out", type=str, default="portions.json", help="Output path for Portions JSON")
    
    args = parser.parse_args()
    
    success = run_writers(args.bible, args.story, args.out)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
