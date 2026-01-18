# XMVP: The Modular Vision Pipeline (v2.5: The "Local-First" Era)

XMVP is a complete "Value Chain" for automated audiovisual content production. It decomposes the creative process into specialist modules (Producers, Directors, Writers, Editors) that can execute in isolation or as a unified pipeline.

**v2.5 Major Update:** Full "Strict Local" support for privacy, cost-saving, and uncensored creativity using Apple Silicon (MPS).

## 🚀 The Three Producers

### 1. `movie_producer.py` (The Showrunner)
Orchestrates the creation of structured video content (Ads, Music Videos, Movie Parodies) from a single prompt.
**New in v2.4:**
- **Local Mode (`--local`)**: Runs 100% offline using `Gemma-2-9b-it` (Text) and `LTX-Video` (Video) on your Mac.
- **Variable Scenes**: Scenes can now vary in length (4s-20s) based on narrative needs.
- **Uncensored Creativity**: Local mode bypasses safety filters unless `--pg` is requested, and applies "Cinematic Fattening" to prompts.

**Usage:**
```bash
# Classic Cloud Mode (Veo + Gemini) -> Tier K (Pro)
python3 movie_producer.py "A sci-fi film about a robot" --vm K

# Local Mode (Gemma + LTX + Flux) -> Uncensored
python3 movie_producer.py "A cyberpunk documentary" --local --seg 10
```

### 2. `cartoon_producer.py` (The Animator)
Specialized pipeline for "Frame-By-Frame" animation and audio-reactive visuals.
**Modes:**
- **Creative Agency**: Prompt -> Story -> Animation.
- **Music Agency**: Audio Track -> Narrative Arc -> Animation (synced to beats).
- **Music Visualizer**: Audio Track -> Abstract/Fractal Evolution.

**Usage:**
```bash
# Music Video with Synced Animation
python3 cartoon_producer.py --vpform music-agency --mu song.mp3 --prompt "Cyberpunk chase"
```

### 3. `post_production.py` (The Editor)
**Now merges `music_video.py` functionality.**
Handles upscaling, interpolation, retiming, and audio sticking.
**New in v2.4:**
- **Flux-Based Processing (`--local`)**: Uses `Flux.1-schnell` for Img2Img upscaling and "Tweening" (interpolation).
- **Audio Sync (`--mu`)**: Auto-calculates Frame Interpolation factors to stretch/squeeze video to match an audio track.
- **4x Upscale (`--more`)**: Chained upscale/interpolation for ultra-smooth output.

**Usage:**
```bash
# Classic Cloud Upscale
python3 post_production.py input.mp4 --scale 2.0

# Local Flux Upscale + 2x Interpolation
python3 post_production.py input.mp4 --local --scale 2.0 -x 2

# Audio Sync (Fit video to audio duration)
python3 post_production.py input.mp4 --mu soundtrack.mp3 --local
```

---

## ⚙️  The Pipeline Modules

1.  **`vision_producer.py`**: Creates the "Bible" (Concept, Characters, Style).
2.  **`stub_reification.py`**: Expands the Bible into a full linear "Story".
3.  **`writers_room.py`**: Breaks Story into timed "Portions" (Scenes). *Now supports Variable Durations.*
4.  **`portion_control.py`**: Calculator for frame ranges and segment logic.
5.  **`dispatch_director.py`**: The rendering engine.
    *   **Cloud**: Uses Veo (Video) or Imagen/Gemini (Image).
    *   **Local**: Uses `LTX-Video` (Video) or `Flux.1-schnell` (Image).
6.  **`truth_safety.py`**: Prompt Refinement Engine.
    *   **Cloud/PG**: Enforces safety and consistency.
    *   **Local**: Applies "Cinematic Fattening" (enriching prompts) and optionally skips/applies safety filters.

## 📁 Configuration & Models (`definitions.py`)
Centralized registry for all models.
*   **Text**: `gemini-2.0-flash` (Cloud), `gemma-2-9b-it` (Local /Volumes/XMVPX).
*   **Image**: `gemini-2.5-flash-image` (Cloud), `flux-schnell` (Local /Volumes/XMVPX).
*   **Video**: `veo-3.1` (Cloud), `ltx-video` (Local /Volumes/XMVPX).
*   **Audio**: `google-journey` (Cloud), `kokoro-v0_19` (Local).

## ⚠️ Deprecations
*   `music_video.py`: **DEPRECATED**. Use `post_production.py --mu`.