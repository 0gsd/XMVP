import argparse
import logging
import json
import sys
from pathlib import Path
from mvp_shared import CSSV, Seg, Indecision, Manifest, load_cssv, load_api_keys, load_xmvp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_portions(path: str) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)
        return data.get("portions", [])

def run_portion(bible_path: str, portions_path: str, out_path: str = "manifest.json") -> bool:
    """
    Executes the Portion Control pipeline.
    """
    # 1. Load Data
    try:
        cssv = load_cssv(bible_path)
        portions_data = load_portions(portions_path)
    except Exception as e:
        logging.error(f"Failed to load inputs: {e}")
        return False

    fps = cssv.constraints.fps
    logging.info(f"📐 Calculating segments @ {fps} FPS...")
    
    segs = []
    current_frame = 0
    
    for p_data in portions_data:
        # p_data is a dict from portions.json
        p_id = p_data['id']
        duration = p_data['duration_sec']
        content = p_data['content']
        
        # Calculate frames
        frame_count = int(duration * fps)
        start = current_frame
        end = current_frame + frame_count
        
        # Construct Seg
        seg = Seg(
            id=p_id,
            start_frame=start,
            end_frame=end,
            prompt=content,
            action="static" # Default for now, could be inferred later
        )
        segs.append(seg)
        
        # Advance cursor
        current_frame = end
        
    logging.info(f"✅ Generated {len(segs)} segments. Total Frames: {current_frame}")
    
    # Construct Manifest
    manifest = Manifest(
        segs=segs,
        files={},
        indecisions=[] 
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
