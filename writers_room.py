import argparse
import logging
import json
import random
import math
import sys
import re
from pathlib import Path
from text_engine import get_engine
from mvp_shared import CSSV, Story, Portion, load_cssv, load_xmvp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_story(path: str) -> Story:
    with open(path, 'r') as f:
        return Story.model_validate_json(f.read())

def break_story(story: Story, cssv: CSSV) -> list[Portion]:
    """
    Splits the story into portions based on constraints using Text Engine.
    """
    total_duration = cssv.constraints.max_duration_sec
    seg_length = cssv.constraints.target_segment_length
    
    # Calculate required segments
    num_portions = math.ceil(total_duration / seg_length)
    
    logging.info(f"🧮 Calculation: {total_duration}s total / {seg_length}s seg = {num_portions} portions.")
    
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
    
    engine = get_engine()
    
    for attempt in range(3):
        try:
            response_text = engine.generate(prompt, temperature=0.7)
            if not response_text:
                raise ValueError("Empty response from Text Engine")

            # Clean Markdown
            text = response_text.replace("```json", "").replace("```", "").strip()
            # Find list start
            if "[" in text and "]" in text:
                text = text[text.find("["):text.rfind("]")+1]
            
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

    # 2. (Keys loaded internally by TextEngine if needed)
        
    # 3. Break Story
    logging.info(f"✍️  Breaking story '{story.title}' into scenes...")
    portions = break_story(story, cssv)
    
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
    parser.add_argument("--bible", type=str, default="bible.json", help="Path to input CSSV JSON")
    parser.add_argument("--story", type=str, default="story.json", help="Path to input Story JSON")
    parser.add_argument("--out", type=str, default="portions.json", help="Output path for Portions JSON")
    parser.add_argument("--xb", type=str, help="Path to XMVP XML file (Overrides inputs)")
    
    args = parser.parse_args()
    
    bible_in = args.bible
    story_in = args.story
    
    if args.xb:
        logging.info(f"📚 Loading Context from XMVP: {args.xb}")
        b_raw = load_xmvp(args.xb, "Bible")
        s_raw = load_xmvp(args.xb, "Story")
        
        if b_raw and s_raw:
             # Create temp files or just overwrite args?
             # Let's overwrite specific inputs if they are default
             with open("bible.json", "w") as f: f.write(b_raw)
             with open("story.json", "w") as f: f.write(s_raw)
             bible_in = "bible.json"
             story_in = "story.json"
        else:
             logging.error("Failed to extract Bible/Story from XMVP.")
             sys.exit(1)
    
    success = run_writers(bible_in, story_in, args.out)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
