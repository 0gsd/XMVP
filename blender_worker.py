"""
blender_worker.py
Executed Internal to Blender via --python.
Handles:
1. Building Library (build-lib)
2. Rendering Shot (render-shot)
"""

import sys
import os
import argparse
import logging

# Setup Logging inside Blender
logging.basicConfig(level=logging.INFO, format="%(asctime)s - WORKER - %(levelname)s - %(message)s")

def get_args():
    # Blender arguments are after "--"
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []

# Add current dir to path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import blender_converter
    import bpy
except ImportError as e:
    logging.error(f"Failed to import modules: {e}")
    sys.exit(1)

def main():
    argv = get_args()
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    # CMD: build-lib
    p_lib = subparsers.add_parser("build-lib")
    p_lib.add_argument("--nicotime", required=True)
    p_lib.add_argument("--out", required=True)
    
    # CMD: render-shot
    p_shot = subparsers.add_parser("render-shot")
    p_shot.add_argument("--lib", required=True)
    p_shot.add_argument("--out", required=True)
    p_shot.add_argument("--duration", type=float, default=2.0)
    p_shot.add_argument("--seed", type=int, default=42)
    # We could pass more shot specific metadata here (camera moves etc)
    
    args = parser.parse_args(argv)
    
    if args.command == "build-lib":
        logging.info(f"🧱 Task: Build Library from {args.nicotime}")
        blender_converter.build_library(args.nicotime, args.out)
        logging.info("✅ Build Complete")
        
    elif args.command == "render-shot":
        logging.info(f"🎥 Task: Render Shot to {args.out}")
        
        # 1. Setup
        # We Mock a 'Seg' object since we don't want to parse full manifest inside blender if we can avoid it. 
        # Or better, we just pass the raw data needed.
        # blender_converter.setup_scene_for_shot expects a Seg object, let's fix that or mock it.
        
        # Let's adjust setup_scene_for_shot in blender_converter to be more flexible?
        # Or just mock it here.
        class MockSeg:
            pass
        seg = MockSeg()
        
        blender_converter.setup_scene_for_shot(seg, args.lib)
        
        # 2. Render
        scene = bpy.context.scene
        scene.render.filepath = args.out
        # Scene setup might have set camera already
        
        # Frame Range
        fps = 24
        scene.render.fps = fps
        scene.frame_start = 1
        scene.frame_end = int(args.duration * fps)
        
        # Output Format: PNG Sequence
        scene.render.image_settings.file_format = 'PNG'
        
        # Execute
        bpy.ops.render.render(animation=True)
        logging.info("✅ Render Complete")
        
    else:
        logging.error("Unknown command")
        sys.exit(1)

if __name__ == "__main__":
    main()
