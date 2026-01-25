import argparse
import json
import logging
import sys
import subprocess
from typing import Optional
from pathlib import Path
from mvp_shared import CSSV, Constraints, VPForm, save_cssv
import librosa
import numpy as np
# Monkeypatch for Scipy 1.13+ vs Librosa < 0.10 compatibility
try:
    import scipy.signal
    if not hasattr(scipy.signal, "hann"):
        if hasattr(scipy.signal.windows, "hann"):
             scipy.signal.hann = scipy.signal.windows.hann
except ImportError:
    pass

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
    ),
    "parody-movie": VPForm(
        name="parody-movie",
        fps=24,
        mime_type="video/mp4",
        description="A direct parody or pastiche spoof of the concept."
    ),
    "music-video": VPForm(
        name="music-video",
        fps=24,
        mime_type="video/mp4",
        description="A music video synchronized to an audio track."
    ),
    "parody-video": VPForm(
        name="parody-video",
        fps=24,
        mime_type="video/mp4",
        description="A music-synced parody video (Veo)."
    ),
    "draft-animatic": VPForm(
        name="draft-animatic",
        fps=24,
        mime_type="video/mp4", # Sequence of images really
        description="High-speed 512x288 animatic for storyboard visualization."
    ),
    "full-movie": VPForm(
        name="full-movie",
        fps=24,
        mime_type="video/mp4",
        description="A full-length feature film animatic."
    ),
    "gahd-podcast": VPForm(
        name="gahd-podcast",
        fps=1,
        mime_type="audio/mp3",
        description="Great Moments in History Podcast."
    ),
    "24-podcast": VPForm(
        name="24-podcast",
        fps=1,
        mime_type="audio/mp3",
        description="24 Hours Real-Time Podcast."
    ),
    "route66-podcast": VPForm(
        name="route66-podcast",
        fps=1,
        mime_type="audio/mp3",
        description="Route 66 Travelogue Podcast."
    )
}

def get_default_vision(form_name: str, seg_count: int = 0) -> str:
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
            "VIBE: Chaotic, funny, disastrous production."
        )
    elif form_name == "parody-movie":
        if seg_count > 4:
            # Hypercompressed Feature Parody
            return (
                "STYLE: ZAZ-style Spoof / Feature-Length Parody (Hypercompressed). "
                "AESTHETIC: High production value (matching the source material perfectly) but filled with visual non-sequiturs, background gags, and literal interpretations of metaphors. "
                "PACING: Relentless, joke-a-minute, slapstick mixed with deadpan seriousness. "
                "INSPIRATION: Airplane!, The Naked Gun, Hot Shots!, Spaceballs. "
                "AUDIO: Pun-heavy dialogue, serious delivery of absurd lines, surreal sound effects (Firesign Theater influence). "
                "STRUCTURE: A full three-act movie condensed into the allocated runtime, following the exact plot beats of the original but making them ridiculous."
            )
        else:
            # Pastiche Segment
            return (
                "STYLE: Pastiche Parody Segment / Sketch Spoof. "
                "AESTHETIC: Exaggerated caricature of the source material. 'Scary Movie' style direct mockery. "
                "CONTENT: A specific famous scene from the movie turned into a sketch. Characters make bad decisions, break the fourth wall, or are exaggerated stereotypes. "
                "VIBE: Irreverent, gross-out humor, meta-commentary, pop-culture mashups. "
                "INSPIRATION: Scary Movie, Don't Be a Menace, Spy Hard. "
                "GOAL: To relentlessly mock the specific scene or concept provided."
            )
    elif form_name == "parody-video":
         return (
                "STYLE: Music Video Parody / Movie Montage set to Music. "
                "AESTHETIC: High-end cinematic parody. Visually matching the prompt (movie title) but edited to the beat of the music. "
                "CONTENT: A sequence of iconic scenes from the movie, but slightly 'off' or exaggerated."
                "PACING: Rhythmic, musical, montage-style."
         )
    elif form_name == "music-video":
        return (
            "CONTENT: Visual metaphors, performance shots (if band mentioned), or pure narrative storytelling."
        )
    elif form_name == "full-movie":
        return (
            "STYLE: Feature Film Animatic. "
            "AESTHETIC: Cinematic composition, 512x288 aspect, storyboard style or rough render. "
            "PACING: Narrative-driven, scene-by-scene. "
            "STRUCTURE: Coherent long-form narrative. "
            "GOAL: To visualize a complete feature film story within the duration."
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

def get_audio_duration(file_path):
    """Get precise duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-i', str(file_path),
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return 60.0

def analyze_audio(audio_path):
    """
    Analyzes audio for Duration and BPM.
    Returns (duration, bpm)
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        # duration = get_audio_duration(audio_path) # Fallback to ffprobe if librosa fails? Librosa is better for exact sample count.
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # tempo is usually a scalar, but can be array.
        if isinstance(tempo, np.ndarray):
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)
            
        logging.info(f"   🎵 Librosa Analysis: {duration:.2f}s @ {bpm:.2f} BPM")
        return duration, bpm
    except Exception as e:
        logging.error(f"Audio analysis failed: {e}")
        return 60.0, 120.0

def run_producer(vpform_name: str, prompt: str, slength: float = 60.0, flength: int = 0, seg_len: float = 4.0, chaos_seed_count: int = 0, cameo: str = None, out_path: str = "bible.json", audio_path: str = None) -> Optional[CSSV]:
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
    # OVERRIDE: Music Video Duration (and Parody Video)
    if (vpform_name == "music-video" or vpform_name == "parody-video") and audio_path and Path(audio_path).exists():
        logging.info(f"   🎵 Analyzing Audio: {audio_path}")
        duration_sec, bpm = analyze_audio(audio_path)
        slength = duration_sec # Override the argument
        logging.info(f"   🎵 Music Detected: {duration_sec:.1f}s @ {bpm:.1f} BPM")
        situation_text += f"\n   MUSIC CONTEXT: {bpm} BPM. Track Duration: {duration_sec:.1f}s."
    
    if flength > 0:
        total_frames = flength
        duration_sec = total_frames / fps
        logging.info(f"   Frame-based duration: {total_frames} frames @ {fps}fps = {duration_sec:.2f}s")
    else:
        duration_sec = slength
        total_frames = int(duration_sec * fps)
        logging.info(f"   Time-based duration: {duration_sec}s @ {fps}fps = {total_frames} frames")
        
    # 4. Construct CSSV
    seg_count = int(total_frames / (fps * seg_len)) if flength > 0 else int(slength / seg_len)
    
    # MLL Template Logic
    template_id = None
    if vpform_name == "gahd-podcast":
         template_id = "GAHD_Template"
    elif vpform_name == "24-podcast":
         template_id = "24_Template"
    elif vpform_name == "route66-podcast":
         template_id = "Route66_Template"

    cssv = CSSV(
        constraints=Constraints(
            width=768, 
            height=768,
            fps=fps,
            max_duration_sec=duration_sec,
            target_segment_length=seg_len if form.name != "draft-animatic" else 15.0, # Default higher for animatic
            black_and_white=False,
            silent=False if "audio" in form.mime_type or "video" in form.mime_type else True
        ),
        scenario=f"A {duration_sec:.1f}-second {form.description}",
        situation=situation_text,
        vision=get_default_vision(form.name, seg_count=seg_count),
        mll_template=template_id
    )

    # DRAFT ANIMATIC OVERRIDE
    if form.name == "draft-animatic":
        # We rely on the Writers Room to be variable.
        # We implicitly signal this by the form name check downstream
        pass

    # 5. Save
    out_path_obj = Path(out_path)
    save_cssv(cssv, out_path_obj)
    logging.info(f"✅ Bible printed: {out_path_obj}")
    logging.info(f"   Situation: {cssv.situation[:100]}...")
    logging.info(f"   Vision: {cssv.vision[:50]}...")
    
    return cssv

def main():
    parser = argparse.ArgumentParser(description="Vision Producer: The Showrunner")
    
    # Smart Positional Args
    parser.add_argument("pos_arg1", nargs='?', help="VPForm OR Prompt")
    parser.add_argument("pos_arg2", nargs='?', help="Prompt (if arg1 was VPForm)")
    
    parser.add_argument("--vpform", type=str, help="The VPForm to use (e.g. realize-ad)")
    parser.add_argument("--prompt", type=str, help="The core request/concept (The 'Situation')")
    parser.add_argument("--slength", type=float, default=60.0, help="Total Duration in Seconds")
    parser.add_argument("--flength", type=int, default=0, help="Total Duration in Frames (Overrides slength if set)")
    parser.add_argument("--seg_len", type=float, default=4.0, help="Target Segment Length in Seconds (Default: 4.0 = Loose/Variable)")
    parser.add_argument("--cs", type=int, default=0, choices=[0, 2, 3, 4, 5, 6], help="Chaos Seeds: 0=Off. 2-6=Wikipedia Injection.")
    parser.add_argument("--out", type=str, default="bible.json", help="Output path for the CSSV Bible")
    
    args = parser.parse_args()
    
    # --- Smart Resolution ---
    potential_form = args.pos_arg1
    resolved_prompt = None
    
    # Check if pos_arg1 looks like a VPForm (dashed, short)
    if potential_form and ("-" in potential_form) and (" " not in potential_form) and (len(potential_form) < 30):
         if not args.vpform:
             args.vpform = potential_form
             logging.info(f"🔎 Detected Positional VPForm: {args.vpform}")
         resolved_prompt = args.pos_arg2
    else:
         # arg1 is likely the prompt (if vpform provided via flag, or missing)
         if potential_form:
             resolved_prompt = potential_form
             
    # Assign Prompt
    if resolved_prompt and not args.prompt:
        args.prompt = resolved_prompt
        
    # Validation
    if not args.vpform or not args.prompt:
        parser.error("VPForm and Prompt are required (either via flags or positional args 'form prompt')")

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
