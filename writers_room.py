import argparse
import logging
import json
import random
import math
import sys
import re
from pathlib import Path
from text_engine import get_engine
from mvp_shared import CSSV, Story, Portion, DialogueLine, load_cssv, load_xmvp

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
    
    # Calculate required segments (Estimation only)
    # We want variable durations between 4s and 20s.
    # We will ask the LLM to determine scene cuts naturally.
    # But we constrain the total duration.
    
    # num_portions = math.ceil(total_duration / seg_length) 
    # logging.info(f"🧮 Calculation: {total_duration}s total / {seg_length}s seg = {num_portions} portions.")
    logging.info(f"🧮 Variable Duration Mode: Total {total_duration}s. Scenes allowed 4.0s - 20.0s.")
    
    prompt = f"""
    CONTEXT: You are a Screenwriter breaking a story into scenes.
    
    STORY:
    Title: {story.title}
    Synopsis: {story.synopsis}
    Characters: {", ".join(story.characters)}
    
    CONSTRAINTS:
    - Total Run Time: Approximately {total_duration} seconds.
    - Break the story into separate Visual Scenes/Shots.
    - Each Scene MUST have a duration between 1.0 and 120.0 seconds.
    - Variable pacing is encouraged (e.g. fast 1s cuts, or long 60s dialogue takes).
    - Ensure the sum of durations is close to {total_duration}s (+/- 10s).
    
    TASK:
    Write a list of Scenes/Portions to tell this story.
    
    OUTPUT FORMAT (JSON List):
    [
        {{ 
            "id": 1, 
            "duration": 5.5, 
            "content": "Description of scene 1...",
            "dialogue": [
                {{ "character": "Hero", "text": "Let's go!", "emotion": "urgent" }}
            ]
        }},
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
            calculated_total = 0.0
            for item in data:
                # Fallback if duration missing (should rarely happen with good prompt)
                dur = float(item.get('duration', seg_length))
                
                # Clamp Duration
                if dur < 1.0: dur = 1.0
                if dur > 120.0: dur = 120.0 # Relaxed for Animatic / Long forms
                
                # Parse Dialogue
                dialogue_list = []
                if 'dialogue' in item:
                    for d in item['dialogue']:
                        dialogue_list.append(DialogueLine(
                            character=d.get('character', 'Unknown'),
                            text=d.get('text', ''),
                            emotion=d.get('emotion', 'neutral')
                        ))

                portions.append(Portion(
                    id=item['id'],
                    duration_sec=dur, 
                    content=item['content'],
                    dialogue=dialogue_list
                ))
                calculated_total += dur
            
            # Validation
            logging.info(f"   Generated {len(portions)} scenes. Total Duration: {calculated_total:.2f}s (Target: {total_duration}s)")
            
            # Basic Check
            if abs(calculated_total - total_duration) > 30.0:
                 logging.warning(f"⚠️ Duration mismatch > 30s. Target: {total_duration}, Got: {calculated_total}")

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
