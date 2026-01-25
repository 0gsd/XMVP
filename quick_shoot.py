import argparse
import logging
import os
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MAC OPENMP FIX ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Quick Shoot: Instant Local Video Generation")
    parser.add_argument("prompt", help="The prompt to film (e.g. 'A dog eating a pizza')")
    parser.add_argument("--out", default=None, help="Output filename (optional override)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=3.0)
    
    args = parser.parse_args()
    
    # Setup Output
    if args.out:
        final_path = os.path.abspath(args.out)
    else:
        out_dir = os.path.join("z_test-outputs", "quick_shoot")
        os.makedirs(out_dir, exist_ok=True)
        
        # Clean prompt for filename
        clean_slug = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '_')).replace(' ', '_')[:40]
        timestamp = int(time.time())
        final_path = os.path.join(out_dir, f"shoot_{timestamp}_{clean_slug}.mp4")

    print(f"🎬 Quick Shoot: '{args.prompt}'")
    print(f"   📐 Resolution: {args.width}x{args.height}")
    print(f"   ⚙️  Steps: {args.steps}, CFG: {args.guidance}")
    
    # 1. Import Director
    try:
        from dispatch_director import LTXDirector
    except ImportError:
        print("❌ Could not import LTXDirector. Are you in the right folder?")
        sys.exit(1)
        
    # 2. Init
    director = LTXDirector()
    director.load()
    
    # 3. Action
    start_t = time.time()
    
    # Check if director supports bridge access for params
    # dispatch_director.LTXDirector.generate signature is fixed, but we can pass kwargs if we modify it
    # OR we can access director.bridge directly here since we know it's LTXDirector
    
    success = False
    try:
        # direct bridge access for advanced tuning
        if hasattr(director, 'bridge') and director.bridge:
             success = director.bridge.generate(
                prompt=args.prompt,
                output_path=final_path,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance
             )
    except Exception as e:
        print(f"⚠️ Direct bridge call failed ({e}). Falling back to standard generate.")
        success = director.generate(
            prompt=args.prompt,
            output_path=final_path,
            width=args.width,
            height=args.height
        )
    
    if success:
        print(f"✅ Cut! Video saved to: {final_path}")
        print(f"⏱️ Time: {time.time() - start_t:.1f}s")
    else:
        print("❌ Action Failed.")

if __name__ == "__main__":
    main()
