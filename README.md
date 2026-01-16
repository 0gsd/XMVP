# XMVP: The Modular Vision Pipeline

This folder contains a complete "Value Chain" for audiovisual motion content production, decomposed into specialist modules which can be used in isolation or in combination. 

It can accept, via movie_producer.py or cartoon_producer.py, an XML "input vision" in the XMVP format and execute based on the XMVP file's contents, *or* it can auto-generate a full vision along with its execution chain and the inputs needed for its output, given only a prompt "idea" and a set of "constraints" -- shorthanded via the "Vision Platonic Form" or --vpform paradigm.

## 🚀 Creative Engines (Start Process Modules)

These modules are the "Big Red Buttons" that kick off the entire generation process.

### `movie_producer.py`
The "Showrunner" that orchestrates the entire 7-stage pipeline to create structured video content (Ads, Movie Trailers) from a simple prompt.
**Usage:**
```bash
python3 movie_producer.py "A sci-fi film about a robot learning to love" [ARGS]
```
See arguments below for details.

### `cartoon_producer.py` (Creative Agency)
Specialized pipeline for "Frame-By-Frame" animation, Music Video syncing, and Creative Agency work.
**Usage:**
```bash
# Creative Agency Mode (Prompt -> Story -> Animation)
./cartoon_producer.py --prompt "A sad toaster finds love"

# Music Video Mode (Syncs animation length to song)
./cartoon_producer.py --prompt "Rave" --mu song.mp3

# Re-Render Mode (Ingest XMVP Manifest)
./cartoon_producer.py --xb manifest.xml
```

**Arguments:**
- `--vpform [STR]`: Mode. `creative-agency` (Default), `fbf-cartoon` (Legacy), `music-visualizer` (Audio-reactive abstract animation), `music-agency` (Audio-reactive narrative story).
- `--prompt [STR]`: The creative concept for the agency to visualize.
- `--mu [PATH]`: Path to an audio file (MP3/WAV/AIFF). Locks video duration to track length and muxes audio.
- `--xb [PATH]`: Path to an existing XMVP XML Manifest to ingest and re-render.
- `--style [STR]`: Aesthetic style description (e.g., "Pixel Art", "Oil Painting"). Default: "Indie Graphic Novel".
- `--slength [FLOAT]`: Target duration in seconds (if no music or XML provided). Default: 60s.
- `--fps [INT]`: Output FPS. Default: `4`.
- `--tf`, `--vf`: Arguments for Transcript/Video folder paths.

### `improv_animator.py`
The "Improv Troupe". Generates endless, streaming improv comedy specials with dynamic casts and reliable XMVP exports.
**Usage:**
```bash
./improv_animator.py --vpform 10-cartoon
```
**Arguments:**
- `--vpform`: `10-cartoon` (10-min Dynamic Cast), `24-cartoon` (24-min Fixed Cast).
- `--slength`: Override duration in seconds (e.g. `60` for testing).
- `--project [STR]`: Google Cloud Project ID for billing overrides.

### `podcast_animator.py` (The VJ)
Visualizes audio podcasts (or generates them) into animated video pairs/triplets.
**Usage:**
```bash
python3 podcast_animator.py --project [ID]
```


### `action.py` (The Video Director)
The "Action!" module. A dedicated Veo/Gemini Video orchestration tool that generates scripts via a "Writers Room" simulation and executes them using Veo (Tier J/K) or Gemini (Tier L) models. It integrates `truth_safety.py` to ensure prompts are coherent and safe before generation.
**Usage:**
```bash
python3 action.py --cut --genre movies --seg 4 --vm J
```
**Arguments:**
- `--cut`: Execute the video generation (without this, it only generates the script).
- `--genre`: `movies` (Remake), `studio` (Mockumentary), `tech` (Sci-Fi).
- `--seg`: Number of segments/shots to generate.
- `--vm`: Video Model Tier (`L`: Light/Gemini, `J`: Just/Veo-2, `K`: Killer/Veo-3).
---

## ⚙️  The Value Chain (Internal Modules)
*These are typically called by `movie_producer.py`, but can be run individually for debugging.*

### 1. `vision_producer.py` (The Visionary)
Creates the "Bible" (CSSV) from a prompt.
- `--vpform`, `--prompt`, `--slength` (Duration Sec), `--flength` (Duration Frames), `--seg_len`, `--cs` (Chaos), `--out`.

### 2. `stub_reification.py` (The Writer)
Expands the Bible into a "Story" (Narrative Arc).
- `--bible` (Input CSSV), `--out`, `--req` (Requests), `--xb` (Load from XML).

### 3. `writers_room.py` (The Screenwriter)
Breaks the Story into temporal "Portions" (Scenes).
- `--bible`, `--story`, `--out`.

### 4. `portion_control.py` (The Line Producer)
Calculates exact frame ranges for each Portion.
- `--bible`, `--portions`, `--out` (Manifest).

### 5. `dispatch_director.py` (The Director)
Executes the Manifest to generate assets.
- `--manifest`, `--staging`, `--out` (Updated Manifest).
- `--mode`: `image` (Flux) or `video` (Veo).
- `--vm`: Video Model Tier (`J`/`K`).
- `--pg`: PG Mode flag.

### 6. `truth_safety.py` (The Guardian)
*Replaces legacy `sanitizer.py`.*
The central "Truth & Safety" engine.
- **Truth**: Enforces physical coherence, logic, and style consistency using `refine_prompt`.
- **Safety**: Applies PG or Safe-Mode constraints to protect against policy violations.
- **Efficiency**: Uses dedicated `TEXT_KEYS_LIST` (High Quota) to avoid burning expensive Action keys on prompt checks.

---

## 🛠️ Utility Tools

### `model_scout.py`
Scans available Gemini/Veo models and checks if `definitions.py` is up to date.
**Usage:**
```bash
python3 model_scout.py [--probe MODEL_NAME]
```


### `post_production.py` (The Redoer)
The "Svelte 2x-ing Machine". Handles visual upscaling, obsessive repainting (detail injection), and frame interpolation (tweening).
**Usage:**
```bash
python3 post_production.py input.mp4 --output /path/to/out -x 2 --scale 2.0
```
**Arguments:**
- `input`: Video file or directory of frames.
- `--output`: Output directory.
- `-x [INT]`: Frame Expansion/Tweening factor (e.g., `2` generates 1 tween per frame).
- `--scale [FLOAT]`: Spatial upscaling factor (e.g., `2.0`).
- `--restyle [STR]`: Apply intermediate style processing. Options: `ascii` (Overlays ASCII art at 33% opacity).

**Narrative Aware Frame Interpolation (NAFI):**
If a matching XML file exists for the input (e.g., `input.xml` next to `input.mp4`), the script automatically loads it. It extracts the CSSV (Vision, Scenario) and Story (Characters) to inject "Project Context" into the tweening prompt, dramatically improving visual coherence during frame interpolation.
### `run_improv_batch.sh`
Orchestrates sequential runs of `improv_animator.py` with cooldowns to respect API quotas.
**Usage:**
```bash
./run_improv_batch.sh [COUNT] [VPFORM]
```
Example: `./run_improv_batch.sh 5 10-cartoon` (Runs 5 episodes of 10-cartoon).

---

## 📁 Environment
Ensure `env_vars.yaml` is populated with the appropriate API keys you need to use any cloud services, in this format: 
```yaml
ACTION_KEYS_LIST: "key1,key2,key3" # For high-cost video generation (Veo)
TEXT_KEYS_LIST: "key4,key5,key6"   # For high-volume text/safety checks (Flash)
```
Your env_vars.yaml can be in a different location (the code references an off-root directory), but you'll have to figure out how to route the calls there.

Even among two Gemini APIs, using the same key, the way requests have to be constructed and massaged, and you'll always need to be keeping everything up to date. I can't save you from that, though model_scout.py IS included which can refer to definitions.py for your preferred models and probe the various APIs using test_gen_capabilities.py. 