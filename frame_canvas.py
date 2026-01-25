import os
import sys
import argparse
import time
import argparse
import time
import gc # Garbage Collection
import numpy as np
import io
import io
try:
    # V2 SDK (Safe)
    from google.genai import types 
    GEMINI_AVAILABLE = True
except ImportError:
    pass
from PIL import Image
try:
    import scipy.ndimage as ndimage
except ImportError:
    ndimage = None
import definitions
from definitions import Modality, BackendType
try:
    from flux_bridge import get_flux_bridge
except ImportError:
    pass # Flux might not be installed, handle gracefully

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------

def get_gemini_logic(model, stage, context, current_res, text_engine=None):
    """
    Constructs the prompt and gets the code from text_engine (Local or Cloud).
    """
    
    # SYSTEM PROMPT (Simplified for Local LLMs like Gemma)
    prompt = f"""
    You are a Python Expert.
    Task: Write a Python function using `numpy` (as np) and `scipy.ndimage` (as ndimage) to process an image.
    
    CONTEXT: {context}
    TARGET RESOLUTION: {current_res} (This is a label, use img.shape dynamically!)
    
    RULES:
    1. Output ONLY raw Python code. No markdown. No comments.
    2. Input `img` is (H, W, 3) float32 array (0.0 to 1.0).
    3. Return `img` of same shape and dtype.
    4. NO hallucinated functions. Use standard numpy operations.
    5. Handle shapes dynamically: `h, w = img.shape[:2]`
    """

    if stage == "pixel_pass":
        prompt += """
        Function: `def logic_pixel_redraw(img):`
        Goal: Analog Pixel Drift.
        Logic: 
        1. Add slight noise: `img += np.random.normal(0, 0.02, img.shape)`
        2. Shift channels slightly if Level > 50.
        3. Clamp to [0,1].
        """
        
    elif stage == "refine_pass":
        prompt += """
        Function: `def logic_refine_block(block, avg_color):`
        Goal: Coherence.
        Logic: 
        1. Blend block towards avg_color by 0.1 factor.
        2. If variance is high, sharpen slightly using `block + (block - ndimage.gaussian_filter(block, 1)) * 0.2`.
        """
        
    elif stage == "degrade_pass":
        prompt += """
        Function: `def logic_degrade(img):`
        Goal: VHS/Retro Effect.
        Logic:
        1. Blur slightly: `img = ndimage.gaussian_filter(img, sigma=0.5)`
        2. Add noise: `img += np.random.uniform(-0.05, 0.05, img.shape)`
        """
    
    elif stage == "pixel_painter":
        h = context.get('h', 64) if isinstance(context, dict) else 64
        w = context.get('w', 64) if isinstance(context, dict) else 64
        prompt += f"""
        Function: `def paint_base(img):`
        Goal: Draw the prompt description from scratch.
        Canvas: {h}x{w} pixels.
        Logic:
        1. Initialize `canvas` as zeros ({h}, {w}, 3).
        2. Draw simple shapes (rectangles/gradients) based on prompt context using numpy slicing.
        3. Example: `canvas[10:50, 10:50] = [1.0, 0.0, 0.0]` (Red Box).
        4. Return `canvas`.
        """

    elif stage == "detail_pass":
        prompt += """
        Function: `def logic_detail_pass(img):`
        Goal: Enhance Details.
        Logic:
        1. Unsharp Mask: `detail = img - ndimage.gaussian_filter(img, 2)`
        2. Add detail back: `img += detail * 0.3`
        3. Add grain: `img += np.random.choice([-0.02, 0.02], img.shape)`
        """

    # GENERATION
    # Prioritize TextEngine (Handles Local/Cloud/Keys)
    if text_engine:
        # Lower temp for code accuracy
        text = text_engine.generate(prompt, temperature=0.2)
        if text:
            # Clean Markdown
            if "```python" in text:
                text = text.split("```python")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return text.strip()
            
    return None

def compile_ai_code(code_str, func_name):
    """
    Compiles the AI-generated code into a callable Python function.
    """
    if not code_str:
        return None

    # We import scipy here to ensure it's available to the exec scope
    import scipy

    # Safe execution scope
    local_scope = {}
    global_scope = {
        'np': np, 
        'ndimage': ndimage,
        'scipy': scipy,
        'Image': Image
    }
    # Add ImageDraw explicitly
    from PIL import ImageDraw
    global_scope['ImageDraw'] = ImageDraw
    

    try:
        exec(code_str, global_scope, local_scope)
        if func_name in local_scope:
            return local_scope[func_name]
    except Exception as e:
        print(f"(!) Generated code failed to compile: {e}")
        # print(code_str) # Uncomment to debug bad code
    return None

# -----------------------------------------------------------------------------
# 2. Execution Stages
# -----------------------------------------------------------------------------

def generate_via_code(prompt, width=1024, height=1024, context=None, model=None, text_engine=None):
    """
    Generates an image from scratch using code.
    """
    if not model and not text_engine:
        # Assuming we can't easily init model here without API key if not passed
        # But usually this is called from main or producer where model exists
        return None
        
    print(f"\n--- Code Painter: {prompt[:50]}... ---")
    
    full_ctx = f"Prompt: {prompt}\nContext: {context}"
    # Note: text_engine isn't passed to generate_via_code in current usage, but we can add it if needed later.
    code = get_gemini_logic(model, "painter", full_ctx, width, text_engine)
    func = compile_ai_code(code, "draw_scene")
    
    if func:
        try:
            pil_img = func(width, height)
            return pil_img
        except Exception as e:
            print(f"(!) Runtime Error in Painter Code: {e}")
            return None
    return None

def tween_frames(img1, img2, blend=0.5):
    """
    Simple Linear Interpolation (Lerp) between two images.
    Returns: PIL Image
    """
    if not img1 or not img2: return None
    
    # Ensure same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
        
    # Blend
    # Image.blend is efficient for linear interpolation
    return Image.blend(img1, img2, blend)

def refine_tween(img_tween, prompt, model=None, width=1024, height=1024, text_engine=None):
    """
    Refines a blended 'ghost' image into a solid frame using the detail pass.
    """
    if not img_tween: return None
    if not model and not text_engine: return None
    
    # Helper
    def get_model():
        if text_engine: return text_engine.get_model_instance()
        return model
    
    print(f"   [Tween] Refining ghost frame ({width}x{height}px)...")
    
    # Convert to array
    arr = np.array(img_tween).astype(np.float32) / 255.0
    
    # Detail Pass
    # Use max dim for resolution label if needed, or better, pass dimensions in prompt
    res_label = max(width, height)
    code = get_gemini_logic(get_model(), "detail_pass", f"Resolution: {width}x{height}. {prompt}", res_label, text_engine)
    func = compile_ai_code(code, "logic_detail_pass")
    
    if func:
        try:
            # Add detail
            arr = func(arr)
            arr = np.clip(arr, 0.0, 1.0)
            return Image.fromarray((arr * 255).astype(np.uint8))
        except Exception as e:
            print(f"(!) Tween Refine Failed: {e}")
            return img_tween # Return original blend if fail
            
    return img_tween

def generate_seed_image(prompt, text_engine, init_dim=1024):
    """
    Generates a high-quality seed image using Imagen 3 via Gemini Client.
    Used for the first frame or recovery from void.
    Respects init_dim for generation size (e.g. 1024 for better Flux composition).
    """
    if not text_engine: return None
    
    print(f"   [Seed] 🎨 Requesting Seed for: {prompt[:40]}...")
    
    # CHECK REGISTRY: Are we Local?
    img_conf = definitions.get_active_model(Modality.IMAGE)
    
    if img_conf.backend == BackendType.LOCAL:
        # Local Flux Path
        print(f"   [Seed] 🔌 Using Local Flux: {img_conf.path}")
        try:
             bridge = get_flux_bridge(img_conf.path)
             if bridge:
                 # FLUX GENERATION AT INIT_DIM
                 return bridge.generate(prompt, width=init_dim, height=init_dim)
             else:
                 print("   (!) Flux Bridge not initialized.")
                 return None
        except Exception as e:
             print(f"   (!) Flux Generation Failed: {e}")
             return None

    # CLOUD PATH (Gemini/Imagen)
    client = text_engine.get_gemini_client()
    if not client: return None
    
    # Fallback Cascade: Verified via model_scout.py
    candidate_models = [
        'gemini-2.5-flash-image',               # Native Content Gen
        'gemini-2.0-flash',                     # Native Content Gen (Stable)
        'imagen-4.0-generate-001',              # Imagen Native
        'imagen-4.0-fast-generate-001',         # Speed Fallback
        'gemini-2.0-flash'                      # Last Resort
    ]
    
    for model_name in candidate_models:
        try:
            # print(f"   [Seed] Trying {model_name}...")
            
            # BRANCH A: Gemini Native (generate_content)
            if "gemini" in model_name.lower():
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    )
                )
                # Parse Gemini Response
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            img = Image.open(io.BytesIO(part.inline_data.data))
                            return img
                        elif part.executable_code: continue # Skip code
                        
            # BRANCH B: Imagen (generate_images)
            else:
                response = client.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="1:1",
                        output_mime_type="image/png"
                    )
                )
                if response.generated_images:
                     img_blob = response.generated_images[0]
                     return img_blob.image
                     
        except Exception as e:
            # print(f"   (!) {model_name} Failed: {e}")
            continue
            
    print("   (!) All Seed Models Failed.")
    return None

def generate_recursive(prompt, width=1024, height=1024, context=None, model=None, text_engine=None, prev_img=None, init_dim=1024):
    """
    Multi-stage generation:
    1. 64x64 Pixel Art Base (Structure & Composition)
       - Uses prev_img (downscaled) as seed if provided.
       - Uses KID (init_dim) for fresh seeds (e.g. 1024px -> 64px) for better composition.
    2. Upscale to 128x128 (Nearest Neighbor)
    ...
    """

    if not model and not text_engine: return None
    
    # Helper to clean/get model
    def get_model():
        if text_engine: return text_engine.get_model_instance()
        return model

    # STAGE 1: Low-Res Pixel Art Base
    # -------------------------------------
    # Calculate Base aspect-correct resolution (aiming for ~64px on shortest side)
    aspect = width / height
    if aspect > 1:
        base_h = 64
        base_w = int(64 * aspect)
    else:
        base_w = 64
        base_h = int(64 / aspect)
        
    print(f"   [Base] Dreaming {base_w}x{base_h} Pixel Art...")
    
    # Pass dimensions via context dict instead of string
    ctx_dict = {'w': base_w, 'h': base_h}
    full_ctx_str = f"Prompt: {prompt}\nContext: {context}" # For the LLM text prompt
    
    # We pass the dict as 'context' to get_gemini_logic for string formatting, 
    # but the function expects a string for the 'Context' field in the prompt.
    # We need to overload get_gemini_logic or just format the prompt manually?
    # Actually get_gemini_logic takes 'context' string.
    # Let's modify get_gemini_logic's signature? No, let's pass a tuple/dict if stage is pixel_painter.
    
    # Hack: Passing dict to get_gemini_logic via 'context' arg is messy if it expects str.
    # Let's update get_gemini_logic to handle dicts in 'context' OR simply pass dimensions explicitly?
    # Let's update get_gemini_logic to accept **kwargs or check type.
    
    # Re-calling logic from clean slate:
    # 1. Update get_gemini_logic matches above replacement chunk which uses `context.get('h')`.
    #    So we must pass a dict as 'context'.
    
    code_px = get_gemini_logic(get_model(), "pixel_painter", {'h': base_h, 'w': base_w, 'text': full_ctx_str}, 100, text_engine)
    func_px = compile_ai_code(code_px, "paint_base")
    
    current_arr = np.zeros((base_h, base_w, 3), dtype=np.float32)
    
    # 1A. Seed Strategy
    if prev_img:
        try:
             small_prev = prev_img.resize((base_w, base_h), Image.Resampling.BILINEAR).convert('RGB')
             current_arr = np.array(small_prev).astype(np.float32) / 255.0
             print(f"   [Base] 🌱 Evolving from previous frame...")
        except: pass
    
    elif not prev_img and text_engine:
         seed_img = generate_seed_image(prompt, text_engine, init_dim=init_dim)
         if seed_img:
             small_seed = seed_img.resize((base_w, base_h), Image.Resampling.LANCZOS).convert('RGB')
             current_arr = np.array(small_seed).astype(np.float32) / 255.0
             print(f"   [Base] 📸 Seeded with Image API (KID: {init_dim}px -> {base_w}x{base_h}).")

    if func_px:
        try:
             current_arr = func_px(current_arr)
             current_arr = np.clip(current_arr, 0.0, 1.0)
             
             if np.mean(current_arr) < 0.02:
                 print(f"   (!) Void Output Detected. Attempting Image API Recovery...")
                 seed_img = generate_seed_image(prompt, text_engine, init_dim=init_dim)
                 if seed_img:
                     small_seed = seed_img.resize((base_w, base_h), Image.Resampling.LANCZOS).convert('RGB')
                     current_arr = np.array(small_seed).astype(np.float32) / 255.0
                 else:
                     print(f"   (!) Recovery Failed. Using Noise.")
                     current_arr = np.random.uniform(0.1, 0.4, (base_h, base_w,3)).astype(np.float32)

             mean_val = np.mean(current_arr)
             if mean_val < 0.35:
                 print(f"   [Base] 💡 Adjusting Exposure (Mean: {mean_val:.2f})")
                 current_arr = np.clip(current_arr * 1.4, 0, 1.0)
                 
        except Exception as e:
             print(f"(!) Pixel Painter Failed: {e}")
             current_arr = np.random.uniform(0, 1, (base_h, base_w, 3)).astype(np.float32)
    else:
        current_arr = np.random.uniform(0, 1, (base_h, base_w, 3)).astype(np.float32)

    # STAGES: Upscale Loop (Dynamic)
    # Target is width/height.
    # We want to double until we hit target.
    
    # Current size
    cur_w, cur_h = base_w, base_h
    
    while cur_w < width or cur_h < height:
        # Determine next step (Double or Cap at target)
        next_w = min(width, cur_w * 2)
        next_h = min(height, cur_h * 2)
        
        # Upscale
        print(f"   [Zoom] Upscaling to {next_w}x{next_h}...")
        
        # NDImage Zoom takes factors
        fac_h = next_h / cur_h
        fac_w = next_w / cur_w
        
        current_arr = ndimage.zoom(current_arr, (fac_h, fac_w, 1), order=1) # Linear
        
        # Update trackers
        cur_w, cur_h = next_w, next_h
        res_label = max(cur_w, cur_h) # Label for prompt
        
        # Detail Pass
        current_arr = perform_stage_pass(current_arr, "detail_pass", prompt, get_model(), res_label, text_engine)
        gc.collect()

    # Final Convert
    final_img = (np.clip(current_arr, 0, 1) * 255).astype(np.uint8)
    del current_arr # Free massive array
    gc.collect()
    return Image.fromarray(final_img)

def perform_stage_pass(img_arr, stage_name, prompt, model, res, text_engine=None):
    """ Helper to run a logic pass on an array """
    print(f"   [Refine] Adding details for {res}px...")
    full_ctx = f"Prompt: {prompt}\nResolution: {res}px"
    code = get_gemini_logic(model, stage_name, full_ctx, 0, text_engine)
    func = compile_ai_code(code, f"logic_{stage_name}")
    
    if func:
        try:
            # We copy to avoid mutation bugs
            in_arr = img_arr.copy()
            out_arr = func(in_arr)
            # HARD SHAPE SAFETY CHECK
            if out_arr.shape != img_arr.shape:
                print(f"(!) Shape Mismatch: {out_arr.shape} vs {img_arr.shape}. Reverting.")
                return img_arr
            return np.clip(out_arr, 0.0, 1.0)
        except Exception as e:
             print(f"(!) Detail Pass Failed at {res}px: {e}")
             return img_arr
    return img_arr

def stage_1_pixels(img_array, model, degrade, text_engine=None):
    """
    Pass 1: Pixel Redraw.
    We use vectorization to simulate the "one by one" process efficiently.
    """
    print(f"\n--- Pass 1: Pixel Redraw (Level {degrade}) ---")
    print("Asking Gemini for pixel logic...")
    
    code = get_gemini_logic(model, "pixel_pass", "Simulate analog pixel placement", degrade, text_engine)
    print(f"--- [DEBUG] Generated Pixel Logic ---\n{code}\n-------------------------------------")
    func = compile_ai_code(code, "logic_pixel_redraw")

    if func:
        try:
            # Apply the logic to the array
            # This is mathematically equivalent to looping every pixel but takes 0.1s instead of 10m
            result = func(img_array.copy())
            return np.clip(result, 0.0, 1.0)
        except Exception as e:
            print(f"Runtime Error in AI Code: {e}")
            
    # Fallback if AI fails
    print("Using fallback pixel logic.")
    return np.clip(img_array + np.random.normal(0, 0.02, img_array.shape), 0, 1)

def stage_2_refinement(img_array, original_array, model, degrade, text_engine=None):
    """
    Pass 2: Recursive Group Refinement.
    We iterate through block sizes and PHYSICALLY process the image in chunks.
    """
    print(f"\n--- Pass 2: Recursive Refinement ---")
    
    h, w, c = img_array.shape
    canvas = img_array.copy()
    
    # The refinement steps (Block sizes)
    steps = [4, 10, 32] 
    
    for size in steps:
        if size > h or size > w: break
        
        print(f"Refining {size}x{size} pixel groups...")
        context = f"Processing image in {size}x{size} chunks."
        
        # 1. Get Logic for this specific scale
        code = get_gemini_logic(model, "refine_pass", context, degrade, text_engine)
        print(f"--- [DEBUG] Generated Refine Logic ({size}x{size}) ---\n{code}\n-------------------------------------")
        func = compile_ai_code(code, "logic_refine_block")
        
        if not func:
            continue

        # 2. Execute Logic on every block (The "Hard in Practice" part)
        # using tqdm for progress because this can take a few seconds
        for y in tqdm(range(0, h, size), desc=f"Grid {size}", unit="blk", leave=False):
            for x in range(0, w, size):
                # Define block boundaries
                y_end, x_end = min(h, y+size), min(w, x+size)
                
                # Extract
                block = canvas[y:y_end, x:x_end]
                orig_block = original_array[y:y_end, x:x_end]
                
                # Calculate Context (Average Color of the original area)
                # This helps the AI decide how to "fix" the block
                avg_color = np.mean(orig_block, axis=(0, 1))
                
                try:
                    # Apply AI function to this specific block
                    new_block = func(block, avg_color)
                    
                    # Safety shape check
                    if new_block.shape == block.shape:
                        canvas[y:y_end, x:x_end] = new_block
                except:
                    pass
                    
    return np.clip(canvas, 0.0, 1.0)

def stage_3_flourishes(img_array, model, degrade, text_engine=None):
    """
    Pass 3: Final Degradation & Perspective Errors.
    """
    print(f"\n--- Pass 3: Flourishes & Degradation ---")
    
    # 1. Image Processing Effects (Blur, Aberration) via Gemini
    code = get_gemini_logic(model, "degrade_pass", "Final artistic touches", degrade, text_engine)
    print(f"--- [DEBUG] Generated Degrade Logic ---\n{code}\n-------------------------------------")
    func = compile_ai_code(code, "logic_degrade")
    
    if func:
        try:
            img_array = func(img_array)
        except Exception as e:
            print(f"Error in degradation pass: {e}")
            
    img_array = np.clip(img_array, 0.0, 1.0)
    
    # 2. Perspective / Lens Errors (Breaking the illusion)
    # DISABLED: User feedback indicated this caused unwanted "keystone" distortion.
    # if degrade > 0:
    #     print("Applying perspective errors...")
    #     pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
    #     w, h = pil_img.size
        
    #     # Calculate distortion intensity (0.0 to 0.3)
    #     intensity = (degrade / 100.0) * 0.2
        
    #     if intensity > 0.01:
    #         # Create a subtle Perspective Warp (Quad Transform)
    #         # We skew the "source" rectangle slightly
    #         dx = int(w * intensity)
    #         dy = int(h * intensity)
            
    #         # Source Quad: Top-Left, Bottom-Left, Bottom-Right, Top-Right
    #         # We squeeze the corners to simulate a lens warp or paper bend
    #         quad = (
    #             -dx, -dy,         # TL
    #             0, h + dy,        # BL
    #             w, h + dy,        # BR
    #             w + dx, -dy       # TR
    #         )
            
    #         try:
    #             # Transform: We map the distorted quad back to the original rectangle
    #             pil_img = pil_img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
    #             img_array = np.array(pil_img).astype(np.float32) / 255.0
    #         except Exception as e:
    #             print(f"Perspective warp skipped: {e}")

    return img_array

# -----------------------------------------------------------------------------
# 3. Main
# -----------------------------------------------------------------------------

try:
    from text_engine import get_engine
except ImportError:
    pass

# -----------------------------------------------------------------------------
# 3. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemini-Powered Recursive Image Reconstruction")
    # Flexible Input Args
    parser.add_argument("--img", dest="input_image", help="Path to input image")
    parser.add_argument("input_image_positional", nargs="?", help="Path to input image (Positional)")
    parser.add_argument("--prompt", help="Text prompt for Code Painter mode (No input image needed)")
    
    parser.add_argument("--degrade", type=int, default=0, choices=range(0, 100), help="Degradation level (0-99)")
    parser.add_argument("--api_key", default=os.environ.get("GOOGLE_API_KEY"), help="Google AI Studio API Key")
    
    # Flexible Output Args
    parser.add_argument("--out", dest="output", help="Output filename")
    parser.add_argument("--output", dest="output_legacy", help="Output filename (Legacy)")
    
    args = parser.parse_args()
    
    # Setup
    t0 = time.time()
    
    # NEW: Use TextEngine
    text_engine = None
    try:
        text_engine = get_engine()
        print("🧠 Text Engine Connected.")
    except Exception as e:
        print(f"(!) Failed to connect Text Engine: {e}")
        # Soft fail if just testing logic without AI?
        
    model = None # Legacy model object is no longer used directly

    # Mode 1: Code Painter (Generate from scratch)
    if args.prompt:
        print(f"--- Code Painter Mode ---")
        img_arr = None
        pil_img = generate_via_code(args.prompt, text_engine=text_engine)
        
        if pil_img:
            # Convert to array for pipeline consistency
            img_arr = np.array(pil_img).astype(np.float32) / 255.0
            original_arr = img_arr.copy()
        else:
            print("(!) Code Painter failed to generate image.")
            sys.exit(1)

    # Mode 2: Image Reconstruction (Input image required)
    else:
        # Resolve Args
        input_path = args.input_image or args.input_image_positional
        if not input_path:
            parser.error("Must provide an input image via positional arg or --img, OR use --prompt")
            
        # Load Image
        try:
            pil_img = Image.open(input_path).convert('RGB')
            original_dpi = pil_img.info.get('dpi', (72, 72))
            print(f"Loaded {input_path} | {pil_img.size} | DPI: {original_dpi}")
        except Exception as e:
            print(f"Failed to load image: {e}")
            sys.exit(1)

        # Normalize to Float32 (0.0 - 1.0)
        img_arr = np.array(pil_img).astype(np.float32) / 255.0
        original_arr = img_arr.copy()
        
        # Pipeline Stage 1 (only for reconstruction)
        # Note: We pass model as None, relying on text_engine internally
        img_arr = stage_1_pixels(img_arr, model=None, degrade=args.degrade, text_engine=text_engine)

    # Common Pipeline (Refinement)
    # Even generated images can benefit from refinement if we want, or we skip?
    # For now, let's skip stage 2/3 for Painter unless degrade > 0?
    # User said: "have python math workers actually draw it out and refine it"
    # So maybe we SHOULD run refinement? Let's run it.
    
    # Must update function signatures in main loop to accept text_engine explicitly if they don't default it
    # Currently stage_1_pixels calls get_gemini_logic(model, ...)
    # We need to make sure stage_1_pixels etc pass text_engine down.
    
    # I need to update the wrapper functions (stage_1, stage_2, stage_3) to accept/pass text_engine.
    img_arr = stage_2_refinement(img_arr, original_arr, model=None, degrade=args.degrade, text_engine=text_engine)
    
    if args.degrade > 0:
        img_arr = stage_3_flourishes(img_arr, model=None, degrade=args.degrade, text_engine=text_engine)
    else:
        print("\n--- Pass 3: Skipped (Degrade = 0) ---")

    # Save
    if args.output or args.output_legacy:
        out_path = args.output or args.output_legacy
    else:
        if args.prompt:
            clean_prompt = "".join([c for c in args.prompt if c.isalnum() or c in (' ', '_')]).rstrip()[:20].replace(" ", "_")
            out_path = f"painter_{clean_prompt}_level{args.degrade}.png"
        else:
            fn, ext = os.path.splitext(input_path)
            out_path = f"{fn}_redraw_level{args.degrade}.png"

    out_pil = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
    out_pil.save(out_path, dpi=original_dpi)
    
    print(f"\nSuccess! Processed in {time.time() - t0:.2f}s")
    print(f"Saved to: {out_path}")

    out_pil = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
    out_pil.save(out_path, dpi=original_dpi)
    
    print(f"\nSuccess! Processed in {time.time() - t0:.2f}s")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()