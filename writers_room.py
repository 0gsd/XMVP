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
    Splits the story into portions using an Iterative Batching approach.
    Allows for unlimited duration by generating in chunks (Micro-Batches).
    """
    total_duration_target = cssv.constraints.max_duration_sec
    
    # Batch Configuration
    BATCH_DURATION = 180.0  # Generate 3 minutes at a time
    
    # Defaults
    MIN_SCENE_DUR = 2.0
    MAX_SCENE_DUR = 15.0
    
    # Strict Length Logic (e.g. for Parody Video / Music Video where beats matter)
    target_seg_len = cssv.constraints.target_segment_length
    is_strict_pacing = False
    
    # Heuristic: If target is significantly different from default 4.0 (e.g. 8.0 for Veo), 
    # OR if we are in a mode that implies structure, we assume user wants that specific pacing.
    if target_seg_len and target_seg_len > 0 and target_seg_len != 4.0:
        logging.info(f"   📏 Strict Pacing Detected: Target {target_seg_len}s per scene.")
        MIN_SCENE_DUR = max(1.0, target_seg_len - 0.5)
        MAX_SCENE_DUR = target_seg_len + 0.5
        is_strict_pacing = True
    
    current_total_duration = 0.0
    all_portions = []
    
    chunk_index = 0
    engine = get_engine()
    
    logging.info(f"🔄 Starting Batched Generation. Target: {total_duration_target}s found. Batch Size: {BATCH_DURATION}s")
    
    while current_total_duration < total_duration_target:
        chunk_index += 1
        remaining_time = total_duration_target - current_total_duration
        current_batch_target = min(BATCH_DURATION, remaining_time)
        
        # Stop if we are close enough (within 10s margin or less than a minimal scene)
        if remaining_time < MIN_SCENE_DUR:
            logging.info("   🛑 Reached target duration.")
            break
            
        logging.info(f"\n   📑 Generative Batch {chunk_index}: Target {current_batch_target}s (Progress: {current_total_duration:.1f}/{total_duration_target}s)")
        
        # Context Management
        last_scenes_context = "START OF MOVIE"
        if all_portions:
            # Get last 3 scenes for continuity
            tail = all_portions[-3:]
            last_scenes_context = "\n".join([f"- Scene {p.id}: {p.content} ({p.duration_sec}s)" for p in tail])
        
        pacing_instruction = f"- SCENE DURATION: Keep scenes short ({MIN_SCENE_DUR}s - {MAX_SCENE_DUR}s) for dynamic pacing."
        if is_strict_pacing:
             pacing_instruction = f"- STRICT PACING: You MUST aim for exactly {target_seg_len}s per scene. We are syncing to a beat."

        prompt = f"""
        CONTEXT: You are a Screenwriter writing a long-form movie script, chunk by chunk.
        
        STORY:
        Title: {story.title}
        Synopsis: {story.synopsis}
        
        CURRENT STATE:
        - Time Elapsed: {current_total_duration:.1f}s / {total_duration_target}s
        - PREVIOUS SCENES (CONTINUITY):
        {last_scenes_context}
        
        TASK:
        Write the NEXT {current_batch_target} seconds of the movie.
        - Create a sequence of "Micro-Scenes" (Fast cuts, dialogue bits, action beats).
        - Focus ONLY on the immediate next segment of the plot. Do not rush to the ending unless we are near {total_duration_target}s.
        {pacing_instruction}
        - TARGET QUANTITY: You need roughly {int(current_batch_target / (target_seg_len if is_strict_pacing else 5))} scenes for this batch.
        
        OUTPUT FORMAT (JSON List):
        [
            {{ 
                "duration": 5.0, 
                "content": "Visual description...",
                "dialogue": [ {{ "character": "Name", "text": "Line" }} ]
            }},
            ...
        ]
        """
        
        success_batch = False
        for attempt in range(3):
            try:
                response_text = engine.generate(prompt, temperature=0.7, json_schema=True)
                if not response_text: continue
                
                # Loose JSON parsing
                data = json.loads(response_text)
                if not isinstance(data, list): data = [data] # Handle single obj
                
                batch_portions = []
                batch_dur = 0.0
                
                start_id = len(all_portions) + 1
                
                for i, item in enumerate(data):
                    dur = float(item.get('duration', 5.0))
                    
                    if is_strict_pacing:
                        dur = target_seg_len # FORCE alignment
                    else:
                        # Clamp
                        dur = max(MIN_SCENE_DUR, min(MAX_SCENE_DUR, dur))
                    
                    # Prevent overshooting Total Duration aggressively
                    if (current_total_duration + batch_dur + dur) > (total_duration_target + (MAX_SCENE_DUR * 0.5)):
                         logging.info("      ✂️ Truncating batch to prevent overshoot.")
                         break
                    
                    dialogue_list = []
                    if 'dialogue' in item:
                        for d in item['dialogue']:
                            dialogue_list.append(DialogueLine(
                                character=d.get('character', 'Unknown'),
                                text=d.get('text', ''),
                                emotion=d.get('emotion', 'neutral')
                            ))
                            
                    p = Portion(
                        id=start_id + i,
                        duration_sec=dur,
                        content=item.get('content', 'Scene...'),
                        dialogue=dialogue_list
                    )
                    batch_portions.append(p)
                    batch_dur += dur
                
                if batch_portions:
                    all_portions.extend(batch_portions)
                    current_total_duration += batch_dur
                    logging.info(f"      ✅ Batch {chunk_index} added {len(batch_portions)} scenes ({batch_dur:.1f}s). New Total: {current_total_duration:.1f}s")
                    success_batch = True
                    break
                    
            except Exception as e:
                logging.warning(f"      ⚠️ Batch Attempt {attempt+1} failed: {e}")
        
        if not success_batch:
            logging.error("      ❌ Failed to generate batch after retries. Aborting/Ending early.")
            break
            
    logging.info(f"✅ Final Script: {len(all_portions)} scenes. Total Duration: {current_total_duration:.1f}s")
    return all_portions

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
