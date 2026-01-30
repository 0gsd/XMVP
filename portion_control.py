import argparse
import logging
import json
import sys
from pathlib import Path
from mvp_shared import CSSV, Seg, Indecision, Manifest, DialogueScript, DialogueLine, load_cssv, load_api_keys, load_xmvp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_portions(path: str) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)
        # Handle both wrapped dict (legacy) and direct list (XML)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
             return data.get("portions", [])
        return []

def run_portion(bible_path: str, portions_path: str, out_path: str = "manifest.json", max_seg_dur: float = 5.0) -> bool:
    """
    Executes the Portion Control pipeline.
    max_seg_dur: Maximum duration per segment (default 5.0 for LTX safety). Set to 0 to disable splitting.
    """
    # 1. Load Data
    try:
        cssv = load_cssv(bible_path)
        portions_data = load_portions(portions_path)
    except Exception as e:
        logging.error(f"Failed to load inputs: {e}")
        return False

    fps = cssv.constraints.fps
    logging.info(f"📐 Calculating segments @ {fps} FPS... (Max Dur: {max_seg_dur}s)")
    
    segs = []
    all_dialogue_lines = []
    current_frame = 0
    
    for p_data in portions_data:
        # p_data is a dict from portions.json
        p_id = p_data['id']
        duration = p_data['duration_sec']
        content = p_data['content']
        
        # Calculate frames
        frame_count = int(duration * fps)
        
        # Track portion start for dialogue offsets
        portion_start_frame = current_frame

        # LTX SAFETY SPLIT
        # If duration > max_seg_dur, split into chunks.
        # Check if splitting is enabled (max_seg_dur > 0)
        should_split = (max_seg_dur > 0 and duration > max_seg_dur)
        
        if should_split:
            # Number of chunks
            import math
            num_chunks = math.ceil(duration / max_seg_dur)
            chunk_frames = frame_count // num_chunks
            
            for k in range(num_chunks):
                # Sub-segment
                s_start = current_frame
                s_end = current_frame + chunk_frames
                if k == num_chunks - 1: s_end = current_frame + frame_count - (chunk_frames * (num_chunks-1)) # Remainder
                
                # Suffix ID for uniqueness (e.g. 101, 102...) or just increment?
                # Portions usually 1..N. Segs can be 1..M.
                # Let's just append to segs list.
                seg = Seg(
                    id=p_id * 100 + k, # Pseudo-ID: Portion 5 -> Seg 500, 501
                    start_frame=s_start,
                    end_frame=s_end,
                    prompt=content + f" (Part {k+1})",
                    action="static"
                )
                segs.append(seg)
                current_frame = s_end
        else:
            # Standard 1:1
            start = current_frame
            end = current_frame + frame_count
            seg = Seg(
                id=p_id,
                start_frame=start,
                end_frame=end,
                prompt=content,
                action="static" 
            )
            segs.append(seg)
            current_frame = end

        if 'dialogue' in p_data:
            # Use portion_start_frame (which equals start of first segment)
            scene_start_time = float(portion_start_frame) / float(fps)
            line_offset_accumulator = 0.5 # Start 0.5s into scene
            
            for d in p_data['dialogue']:
                # Basic Timing: assume 3 words per second?
                # Or just stack them with slight gaps.
                # Ideally, Writers Room provides duration. If not, estimate.
                est_duration = len(d.get('text', '').split()) * 0.4 # Rough estimate
                
                dl = DialogueLine(
                    character=d.get('character', 'Unknown'),
                    text=d.get('text', ''),
                    emotion=d.get('emotion', 'neutral'),
                    start_offset=scene_start_time + line_offset_accumulator
                )
                all_dialogue_lines.append(dl)
                line_offset_accumulator += (est_duration + 0.5)

        # Advance cursor
        # Advance cursor - already done in if/else blocks above.

        
    logging.info(f"✅ Generated {len(segs)} segments. Total Frames: {current_frame}. Dialogue Lines: {len(all_dialogue_lines)}")
    
    # Construct Manifest
    manifest = Manifest(
        segs=segs,
        files={},
        indecisions=[],
        dialogue=DialogueScript(lines=all_dialogue_lines) if all_dialogue_lines else None
    )
    
    # Save
    with open(out_path, 'w') as f:
        f.write(manifest.model_dump_json(indent=2))
        
    logging.info(f"📋 Manifest saved: {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Portion Control: The Line Producer")
    parser.add_argument("--bible", type=str, default="bible.json", help="Path to input CSSV JSON")
    parser.add_argument("--portions", type=str, default="portions.json", help="Path to input Portions JSON")
    parser.add_argument("--out", type=str, default="manifest.json", help="Output path for Manifest JSON")
    parser.add_argument("--xb", type=str, help="Path to XMVP XML file (Overrides inputs)")
    
    args = parser.parse_args()
    
    bible_in = args.bible
    portions_in = args.portions
    
    if args.xb:
        logging.info(f"📚 Loading Context from XMVP: {args.xb}")
        b_raw = load_xmvp(args.xb, "Bible")
        p_raw = load_xmvp(args.xb, "Portions")
        
        if b_raw and p_raw:
             with open("bible.json", "w") as f: f.write(b_raw)
             with open("portions.json", "w") as f: f.write(p_raw)
             bible_in = "bible.json"
             portions_in = "portions.json"
        else:
             logging.error("Failed to extract Bible/Portions from XMVP.")
             sys.exit(1)
    
    success = run_portion(bible_in, portions_in, args.out)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
