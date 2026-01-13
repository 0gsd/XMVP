import argparse
import json
import logging
import sys
from pathlib import Path
from mvp_shared import CSSV, Constraints, VPForm, save_cssv

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Form Definitions (Registry) ---
# In the future, these could be loaded from external YAML/JSON files.

VP_FORMS = {
    "realize-ad": VPForm(
        name="realize-ad",
        fps=24,
        mime_type="video/mp4",
        description="A commercial advertisement video generated from a high-level concept."
    ),
    "tech-movie": VPForm(
        name="tech-movie",
        fps=24, 
        mime_type="video/mp4",
        description="A narrative short film about technology."
    ),
    "podcast-interview": VPForm(
        name="podcast-interview",
        fps=1, # Audio focused
        mime_type="audio/mp3", 
        description="A two-person interview podcast."
    ),
    "movies-movie": VPForm(
        name="movies-movie",
        fps=24,
        mime_type="video/mp4",
        description="A condensed Hollywood Blockbuster remake (1979-2001 era)."
    ),
    "studio-movie": VPForm(
        name="studio-movie",
        fps=24,
        mime_type="video/mp4",
        description="A chaotic 'Making Of' mockumentary about a film production."
    )
}

def get_default_vision(form_name: str) -> str:
    """Returns the default 'Edict' or 'Vision' for a given form."""
    if form_name == "realize-ad":
        return (
            "STYLE: High-end TV Commercial. "
            "AESTHETIC: Polished, Cinematic, persuasive lighting. "
            "PACING: Fast, energetic, engaging. "
            "PURPOSE: To sell or promote the concept provided in the Situation."
        )
    elif form_name == "tech-movie":
        return (
            "STYLE: Modern Sci-Fi Drama. "
            "AESTHETIC: Cold blues, warm oranges, lens flares, high-tech interfaces. "
            "PACING: Narrative-driven, suspenseful."
        )
    elif form_name == "movies-movie":
        return (
            "STYLE: Hollywood Blockbuster (1979-2001). "
            "AESTHETIC: 35mm film grain, high-budget action/drama, cinematic anamorphic lens. "
            "PACING: Condensed, urgent, 'Movie Trailer' energy. "
            "CRITICAL: Must look like a real movie from that era."
        )
    elif form_name == "studio-movie":
        return (
            "STYLE: Behind-The-Scenes Mockumentary / Found Footage. "
            "AESTHETIC: Handheld camera, raw lighting, visible film equipment (boom mics, light stands, craft services). "
            "CONTENT: Actors out of character, crew members struggling, set disasters. "
            "VIBE: Chaotic, funny, disastrous production."
        )
    else:
        return "Standard Production."

# --- Chaos Engine ---
import requests
def get_chaos_seed() -> str:
    """Fetches a random Wikipedia title."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        headers = {'User-Agent': 'VisionProducer/1.0 (chaos_engine)'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            title = resp.json().get('title', 'Entropy')
            logging.info(f"🎲 Chaos Seed: {title}")
            return title
    except Exception as e:
        logging.warning(f"Chaos engine failed: {e}")
    return "The Unknown" # Fallback

def get_specific_seed(query: str) -> str:
    """
    Fetches a specific Wikipedia summary based on query.
    Handles standard URL or search terms.
    """
    slug = query
    
    # 1. URL parsing
    if "wikipedia.org/wiki/" in query:
        try:
            # Extract everything after /wiki/
            slug = query.split("/wiki/")[-1].split("?")[0].split("#")[0]
        except:
            slug = query
            
    logging.info(f"🔎 Looking up Cameo: '{slug}'...")

    try:
        # Use simple summary endpoint
        # https://en.wikipedia.org/api/rest_v1/page/summary/{title}
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        headers = {'User-Agent': 'VisionProducer/1.0 (cameo_engine)'}
        
        resp = requests.get(url, headers=headers, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            title = data.get('title')
            extract = data.get('extract', '')[:200]  # First 200 chars
            result = f"{title} ({extract}...)"
            logging.info(f"   ✨ Cameo Found: {title}")
            return result
        elif resp.status_code == 404:
             # Try search?
             logging.warning(f"   ⚠️ Direct lookup failed for '{slug}'. Trying search...")
             # Search endpoint: https://en.wikipedia.org/w/api.php?action=opensearch...
             # For MVP, just return the query itself if lookup fails, 
             # forcing the LLM to hallucinate/knowledge-retrieve it.
             return f"{query} (Concept)"
    except Exception as e:
        logging.warning(f"   ❌ Cameo lookup error: {e}")
        
    return query

def run_producer(vpform_name: str, prompt: str, slength: float = 60.0, flength: int = 0, seg_len: float = 4.0, chaos_seed_count: int = 0, cameo: str = None, out_path: str = "bible.json") -> bool:
    """
    Executes the Vision Producer pipeline.
    """
    # 1. Validate Form
    if vpform_name not in VP_FORMS:
        logging.error(f"Unknown VPForm: {vpform_name}")
        logging.info(f"Available Forms: {list(VP_FORMS.keys())}")
        return False
        
    form = VP_FORMS[vpform_name]
    logging.info(f"🎬 Vision Producer running form: {form.name}")
    
    # 2. Chaos Seeds & Cameo Logic
    situation_text = prompt
    
    # Chaos
    if chaos_seed_count > 0:
        logging.info(f"🎲 Rolling {chaos_seed_count} Chaos Seeds...")
        seeds = [get_chaos_seed() for _ in range(chaos_seed_count)]
        
        if chaos_seed_count == 2:
            situation_text = f"CONCEPT: {prompt}. CHAOS INFLUENCE (Merge these into the core concept): 1. {seeds[0]} 2. {seeds[1]}"
        else:
            core_seeds = seeds[:2]
            secondary_seeds = seeds[2:]
            
            situation_text = (
                f"CONCEPT: {prompt}.\n"
                f"PRIMARY CHAOS (The Core Premise): {', '.join(core_seeds)}.\n"
                f"SECONDARY CHAOS (The 'B-Plot' or background flavor): {', '.join(secondary_seeds)}."
            )
            
        logging.info(f"🎲 Chaos Integrated. New Situation Length: {len(situation_text)}")

    # Cameo
    if cameo:
        cameo_content = get_specific_seed(cameo)
        situation_text += f"\nCAMEO FEATURE (Minor Appearance / Easter Egg): {cameo_content}"
        logging.info(f"   🌟 Cameo Injected: {cameo_content[:50]}...")

    # 3. Calculate Constraints
    fps = form.fps
    
    # Duration Logic
    if flength > 0:
        total_frames = flength
        duration_sec = total_frames / fps
        logging.info(f"   Frame-based duration: {total_frames} frames @ {fps}fps = {duration_sec:.2f}s")
    else:
        duration_sec = slength
        total_frames = int(duration_sec * fps)
        logging.info(f"   Time-based duration: {duration_sec}s @ {fps}fps = {total_frames} frames")
        
    # 4. Construct CSSV
    cssv = CSSV(
        constraints=Constraints(
            width=768, 
            height=768,
            fps=fps,
            max_duration_sec=duration_sec,
            target_segment_length=seg_len,
            black_and_white=False,
            silent=False if "audio" in form.mime_type or "video" in form.mime_type else True
        ),
        scenario=f"A {duration_sec:.1f}-second {form.description}",
        situation=situation_text,
        vision=get_default_vision(form.name)
    )

    # 5. Save
    out_path_obj = Path(out_path)
    save_cssv(cssv, out_path_obj)
    logging.info(f"✅ Bible printed: {out_path_obj}")
    logging.info(f"   Situation: {cssv.situation[:100]}...")
    logging.info(f"   Vision: {cssv.vision[:50]}...")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Vision Producer: The Showrunner")
    parser.add_argument("--vpform", type=str, required=True, help="The VPForm to use (e.g. realize-ad)")
    parser.add_argument("--prompt", type=str, required=True, help="The core request/concept (The 'Situation')")
    parser.add_argument("--slength", type=float, default=60.0, help="Total Duration in Seconds")
    parser.add_argument("--flength", type=int, default=0, help="Total Duration in Frames (Overrides slength if set)")
    parser.add_argument("--seg_len", type=float, default=4.0, help="Target Segment Length in Seconds")
    parser.add_argument("--cs", type=int, default=0, choices=[0, 2, 3, 4, 5, 6], help="Chaos Seeds: 0=Off. 2-6=Wikipedia Injection.")
    parser.add_argument("--out", type=str, default="bible.json", help="Output path for the CSSV Bible")
    
    args = parser.parse_args()

    run_producer(
        vpform_name=args.vpform,
        prompt=args.prompt,
        slength=args.slength,
        flength=args.flength,
        seg_len=args.seg_len,
        chaos_seed_count=args.cs,
        out_path=args.out
    )

if __name__ == "__main__":
    main()
