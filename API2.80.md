# XMVP API Reference v2.80
## Complete Command-Line Interface Documentation

---

# Table of Contents

1. [Creative Engines (Start Process Modules)](#creative-engines)
   - [movie_producer.py](#movie_producerpy)
   - [cartoon_producer.py](#cartoon_producerpy)
   - [content_producer.py](#content_producerpy)
   - [post_production.py](#post_productionpy)
   - [xmvp_converter.py](#xmvp_converterpy)
2. [Pipeline Modules (Internal)](#pipeline-modules)
   - [vision_producer.py](#vision_producerpy)
   - [stub_reification.py](#stub_reificationpy)
   - [writers_room.py](#writers_roompy)
   - [portion_control.py](#portion_controlpy)
   - [dispatch_director.py](#dispatch_directorpy)
   - [dispatch_animatic.py](#dispatch_animaticpy)
   - [dispatch_wan.py](#dispatch_wanpy)
3. [Audio & Speech Modules](#audio--speech-modules)
   - [foley_talk.py](#foley_talkpy)
   - [thax_audio.py](#thax_audiopy)
4. [Utility & Management Modules](#utility--management-modules)
   - [model_scout.py](#model_scoutpy)
   - [populate_models_xmvp.py](#populate_models_xmvppy)
   - [sassprilla_carbonator.py](#sassprilla_carbonatorpy)
   - [dialogue_critic.py](#dialogue_criticpy)
   - [convert_voices.py](#convert_voicespy)
   - [rescue_session.py](#rescue_sessionpy)
   - [prep_movie_assets.py](#prep_movie_assetspy)
   - [still_life.py](#still_lifepy)
   - [test_gen_capabilities.py](#test_gen_capabilitiespy)
5. [Bridge Modules (Local Inference)](#bridge-modules)
   - [flux_bridge.py](#flux_bridgepy)
   - [ltx_bridge.py](#ltx_bridgepy)
   - [kokoro_bridge.py](#kokoro_bridgepy)
   - [hunyuan_foley_bridge.py](#hunyuan_foley_bridgepy)
   - [wan_bridge.py](#wan_bridgepy)
6. [Core Libraries](#core-libraries)
   - [text_engine.py](#text_enginepy)
   - [truth_safety.py](#truth_safetypy)
   - [definitions.py](#definitionspy)
   - [mvp_shared.py](#mvp_sharedpy)
   - [frame_canvas.py](#frame_canvaspy)
7. [Data Models & Schemas](#data-models--schemas)
8. [Configuration Files](#configuration-files)
9. [Training & Voice Models](#training--voice-models)

---

# Creative Engines

## movie_producer.py

**The Showrunner** — Orchestrates the entire 7-stage pipeline to create structured video content from a single prompt. Features auto-carbonation for title-style prompts.

### Usage
```bash
python3 movie_producer.py "Your concept here" [OPTIONS]
# Or using positional VPForm syntax:
python3 movie_producer.py tech-movie "Your concept here"
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `concept` | positional | - | The concept text (quoted string). Required unless `--xb` provided. |
| `cli_args` | positional | - | Global positional args (VPForm alias, commands) |

### Options

#### Producer Options
| Option | Type | Default | Values/Range | Description |
|--------|------|---------|--------------|-------------|
| `--seg` | int | `3` | 1-∞ | Number of video segments to generate |
| `--slength` | float | `0.0` | 1-∞ | Target total duration in seconds (overrides --seg) |
| `--l` | float | `8.0` | 1.0-60.0 | Length of each segment in seconds |
| `--vpform` | str | `None` | See VPForm Registry | Vision Platonic Form (genre template) |
| `--cs` | int | `0` | 0, 2-6 | Chaos Seeds level (Wikipedia concept injection) |
| `--cf` | str | `None` | URL or query | Cameo Feature: Inject specific concept |
| `--mu` | str | `None` | Path | Music track for music-video mode |
| `--vm` | str | `"L"` | `L`, `J`, `K`, `D`, `V2` | Video Model Tier |
| `--pg` | flag | `False` | - | Enable PG Mode (relaxed celebrity, strict child safety) |
| `--prompt` | str | `None` | - | Alias for concept |

#### Operations Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--xb` | str | `"clean"` | XMVP re-hydration path OR 'clean' (default) |
| `-f`, `--fast` | flag | `False` | Use faster/cheaper model tier (sets --vm to J) |
| `--vfast` | flag | `False` | Use legacy Veo 2.0 (sets --vm to V2) |
| `--out` | str | `None` | Override output directory |
| `--local` | flag | `False` | Run 100% locally (Gemma + LTX-Video) |
| `--retcon` | flag | `False` | Force text-only expansion (implies --local, skips video) |

### Video Model Tiers

| Tier | Model | Description |
|------|-------|-------------|
| `L` | veo-2.0-generate-001 | Light (fast, lower quality) |
| `J` | veo-3.1-fast-generate-preview | Just Right (balanced) |
| `K` | veo-3.1-generate-preview | Killer (cinematic 4K) |
| `D` | veo-2.0-generate-001 | Default legacy |
| `V2` | veo-2.0-generate-001 | Legacy Veo 2.0 |

### Examples
```bash
# Basic cloud generation
python3 movie_producer.py "A noir detective story" --seg 5 --vm K

# Local mode (offline, uncensored)
python3 movie_producer.py "Cyberpunk rebellion" --local --seg 8

# Music video with audio sync
python3 movie_producer.py "Electronic dance" --vpform music-video --mu track.mp3

# Long-form movie (Micro-Batching)
python3 movie_producer.py "Epic Space Opera" --vpform full-movie --local --slength 3000

# Re-render from existing XMVP manifest
python3 movie_producer.py --xb previous_run.xml --vm J

# Using positional VPForm alias
python3 movie_producer.py tech-movie "AI Awakening"
```

---

## cartoon_producer.py

**The Animator** — Specialized pipeline for frame-by-frame animation, music video syncing, and creative agency work.

### Usage
```bash
python3 cartoon_producer.py [OPTIONS]
```

### Options

#### Core Options
| Option | Type | Default | Values | Description |
|--------|------|---------|--------|-------------|
| `--vpform` | str | `"creative-agency"` | `creative-agency`, `fbf-cartoon`, `music-visualizer`, `music-agency` | Vision Platonic Form |
| `--prompt` | str | `None` | Any text | Creative prompt for agency mode |
| `--style` | str | See default | Any text | Visual style definition |
| `--slength` | float | `60.0` | 1.0-∞ | Target length in seconds (if no music) |
| `--fps` | int | `4` | 1-60 | Output FPS / expansion factor |

**Default style**: `"Indie graphic novel artwork. Precise, uniform, dead-weight linework. Highly stylized, elegantly sophisticated, and with an explosive, highly saturated pop-color palette."`

#### Source Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tf` | Path | Default path | Transcript folder (source) |
| `--vf` | Path | `/Volumes/XMVPX/fmv_corpus` | Video folder (corpus) |
| `--xb` | str | `None` | Path to XMVP XML manifest for re-rendering |
| `--mu` | str | `None` | Path to music/audio file for sync |
| `--project` | str | `None` | Specific project name to process |

#### Processing Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--delay` | float | `5.0` | Delay between API requests in seconds |
| `--limit` | int | `0` | Limit number of frames (0 = unlimited) |
| `--smin` | float | `0.0` | Minimum duration filter in seconds |
| `--smax` | float | `None` | Maximum duration filter in seconds |
| `--shuffle` | flag | `False` | Shuffle projects before processing |
| `--cs` | int | `0` | Chaos Seeds level (0-3) |
| `--bpm` | float | `None` | Manual BPM override (bypasses detection) |
| `--pg` | flag | `False` | Enable PG Mode |
| `--vspeed` | float | `8.0` | Visualizer speed (FPS) for music-agency |
| `--fc` | flag | `False` | Enable Frame & Canvas (Code Painter) mode |

### Examples
```bash
# Creative agency mode
python3 cartoon_producer.py --prompt "A sad robot finds purpose" --style "Pixel Art"

# Music video synced to track
python3 cartoon_producer.py --vpform music-agency --mu song.mp3 --prompt "Neon dreams"

# Abstract visualizer
python3 cartoon_producer.py --vpform music-visualizer --mu ambient.wav
```

---

## content_producer.py

**The Podcast Factory** — Unified generator for podcast and improv content with RVC voice support.

### Usage
```bash
python3 content_producer.py [OPTIONS]
```

### Options

| Option | Type | Default | Values | Description |
|--------|------|---------|--------|-------------|
| `--vpform` | str | `None` | `24-podcast`, `24-cartoon`, `10-podcast`, `gahd-podcast`, `thax-douglas`, `route66-podcast`, `fullmovie-still` | Vision Platonic Form |
| `--project` | str | `None` | - | Project override |
| `--ep` | int | `None` | - | Episode number (format: SSE for Season S, Episode E) |
| `--local` | flag | `False` | - | Use local engines (Flux + Kokoro) |
| `--foley` | str | `"off"` | `on`, `off` | Enable generative foley audio |
| `--slength` | float | `0.0` | - | Override duration in seconds |
| `--fc` | flag | `False` | - | Code Painter Mode |
| `--geminiapi` | flag | `False` | - | Force Cloud Gemini API for text |
| `--band` | str | `None` | - | Band name (thax-douglas mode) |
| `--poem` | str | `None` | - | Poem text (thax-douglas mode) |
| `--w` | int | `1024` | - | Width |
| `--h` | int | `576` | - | Height |
| `--location` | str | `None` | - | Override visual location |
| `--rvc` | flag | `False` | - | Enable RVC voice conversion |
| `--xml` / `--xb` | str | `None` | - | Input XMVP XML path |

### VP Forms for Content

| Form | Cast Size | Default Duration | Description |
|------|-----------|------------------|-------------|
| `24-podcast` | 4 | 24 min (1440s) | 4-person improv comedy special |
| `10-podcast` | 4 | 10 min (600s) | Topical tech podcast |
| `route66-podcast` | 6 | 66 min (3960s) | Road trip narrative |
| `gahd-podcast` | Variable | Variable | Great Moments in History |
| `thax-douglas` | 1 | Variable | Spoken word poetry |

### Examples
```bash
# 24-minute improv podcast
python3 content_producer.py --vpform 24-podcast --local --slength 1440

# Route 66 with RVC voices
python3 content_producer.py --vpform route66-podcast --rvc --local --slength 3960 --ep 301 --location "The Roadside Diner"

# GAHD podcast
python3 content_producer.py --vpform gahd-podcast --slength 3200 --ep 207 --local --location "Ancient Rome"

# Thax Douglas spoken word
python3 content_producer.py --vpform thax-douglas --band "The Mountain Goats"
```

---

## post_production.py

**The Editor** — Upscaling, frame interpolation, restyling, and audio stitching.

### Usage
```bash
python3 post_production.py <input> [OPTIONS]
# Input can be: video file, folder of frames (numbered), or folder of videos (numbered)
```

### Input Types
1. **Single video file**: `video.mp4`
2. **Folder of frame images**: Directory with `frame_00001.png`, `frame_00002.png`, etc.
3. **Folder of video segments**: Directory with `seg_001.mp4`, `seg_002.mp4`, etc.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input` | positional/flag | - | Input video file or directory (use `--input` or positional) |
| `--output` | str | `None` | Output directory |
| `-x` | int | `2` | Frame expansion factor (tweening). Default: 2 |
| `--scale` | float | `2.0` | Upscale factor. Default: 2.0 |
| `--restyle` | str | `None` | Restyle mode (e.g., 'ascii') |
| `--local` | flag | `False` | Run locally (Flux Img2Img) |
| `--more` | flag | `False` | Enable secondary pass (4x total) |
| `--mu` | str | `None` | Audio file for music video sync |
| `--stitch-audio` | flag | `False` | Force-stitch frames to match audio duration |

### Processing Pipeline
1. **Extract** — Extract frames from video (if video input)
2. **Interpolate** — Generate in-between frames (controlled by `-x`)
3. **Upscale** — Increase resolution (controlled by `--scale`)
4. **Restyle** — Apply style filters (if `--restyle` set)
5. **Stitch** — Combine frames into final video with audio sync

### Examples
```bash
# 2x upscale a single video
python3 post_production.py video.mp4 --local --scale 2.0

# Frame interpolation (2x smoother)
python3 post_production.py video.mp4 --local -x 2

# 4x processing (interpolation + upscale)
python3 post_production.py video.mp4 --local --more

# Sync video frames to audio duration
python3 post_production.py /path/to/frames/ --mu soundtrack.mp3 --stitch-audio

# Process folder of video segments with audio
python3 post_production.py --input /path/to/segments/ --mu /path/to/audio.aif --stitch-audio

# ASCII restyle
python3 post_production.py video.mp4 --restyle ascii
```

---

## xmvp_converter.py

**The Converter** — Converts external scripts/text into XMVP format for rendering.

### Usage
```bash
python3 xmvp_converter.py <input_file> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input_file` | positional | - | Path to input script (txt, md) |
| `--vpform` | str | `"standard"` | Form (parody-movie, standard) |
| `--slength` | float | `None` | Target duration for retcon |
| `--out` | str | `None` | Output directory |
| `--local` | flag | `False` | Force local models |

### Examples
```bash
# Convert a script to parody movie format
python3 xmvp_converter.py /Volumes/XMVPX/mw/your-project/processed_text/Script.txt \
    --vpform parody-movie \
    --slength 5820
```

---

# Pipeline Modules

## vision_producer.py

**The Visionary** — Creates the "Bible" (CSSV) from a prompt.

### Usage
```bash
python3 vision_producer.py [VPForm] "prompt" [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vpform` | str | `None` | The VPForm to use |
| `--prompt` | str | `None` | The core concept |
| `--slength` | float | `60.0` | Total duration in seconds |
| `--flength` | int | `0` | Total duration in frames (overrides slength) |
| `--seg_len` | float | `8.0` | Target segment length in seconds |
| `--cs` | int | `0` | Chaos Seeds (0=Off, 2-6=Wikipedia injection) |
| `--out` | str | `"bible.json"` | Output path |

---

## stub_reification.py

**The Story Architect** — Expands a Bible into a full Story structure.

### Usage
```bash
python3 stub_reification.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bible` | str | `"bible.json"` | Path to input CSSV JSON |
| `--out` | str | `"story.json"` | Output path for Story JSON |
| `--req` | str | `None` | Optional extra request/notes |
| `--xb` | str | `None` | Path to XMVP XML (overrides --bible) |

---

## writers_room.py

**The Writers** — Breaks a Story into timed Portions (scenes).

### Usage
```bash
python3 writers_room.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bible` | str | `"bible.json"` | Path to input CSSV JSON |
| `--story` | str | `"story.json"` | Path to input Story JSON |
| `--out` | str | `"portions.json"` | Output path for Portions JSON |
| `--xb` | str | `None` | Path to XMVP XML (overrides inputs) |

---

## portion_control.py

**The Calculator** — Converts Portions into frame-accurate Segments.

### Usage
```bash
python3 portion_control.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bible` | str | `"bible.json"` | Path to input CSSV JSON |
| `--portions` | str | `"portions.json"` | Path to input Portions JSON |
| `--out` | str | `"manifest.json"` | Output path for Manifest JSON |
| `--xb` | str | `None` | Path to XMVP XML (overrides inputs) |

---

## dispatch_director.py

**The Director** — Generates video/image assets from a Manifest.

### Usage
```bash
python3 dispatch_director.py --manifest <path> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--manifest` | str | Required | Path to input Manifest JSON |
| `--out` | str | `"manifest_updated.json"` | Output path for updated Manifest |
| `--staging` | str | `"componentparts"` | Directory to save assets |
| `--mode` | str | `"image"` | Generation mode: `image` or `video` |
| `--vm` | str | `"J"` | Video Model Tier |
| `--pg` | flag | `False` | Enable PG Mode |
| `--width` | int | `768` | Output width (Image Mode) |
| `--height` | int | `768` | Output height (Image Mode) |
| `--local` | flag | `False` | Force Local Mode (LTX for Video) |

---

## dispatch_animatic.py

**The Storyboarder** — High-speed storyboard generation using Gemma + Flux.

### Usage
```bash
python3 dispatch_animatic.py --manifest <path> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--manifest` | str | Required | Path to Manifest JSON |
| `--out` | str | `"manifest_updated.json"` | Output path |
| `--staging` | str | `"componentparts"` | Directory to save assets |
| `--flux_path` | str | Fallback path | Path to Flux model |

---

## dispatch_wan.py

**The Wan Dispatcher** — Handles Wan 2.1 video generation for long-form content.

### Classes
- Handles I2V (Image-to-Video) generation using Wan 2.1

---

# Audio & Speech Modules

## foley_talk.py

**The Sound Designer** — Generates dialogue audio and foley effects.

### Usage
```bash
python3 foley_talk.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | str | `None` | Input silent video path |
| `--xb` | str | `None` | Input XMVP manifest |
| `--out` | str | `"final_mix.mp4"` | Output video path |
| `--mode` | str | `"cloud"` | Audio backend: `cloud`, `comfy`, `rvc`, `kokoro`, `draft-mix` |
| `--dry-run` | flag | `False` | Simulate execution |

### Classes
- `ComfyWrapper` — ComfyUI integration for advanced audio
- `LegacyVoiceEngine` — Fallback TTS engine

---

## thax_audio.py

**The Poet's Voice** — Thax Douglas voice generation using Kokoro + RVC.

### Classes
- `ThaxVoiceEngine` — Generates spoken word audio in Thax Douglas's voice

---

# Utility & Management Modules

## model_scout.py

**The Scout** — Manages and inspects model configurations.

### Usage
```bash
python3 model_scout.py [OPTIONS]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--status` | flag | Show current active models |
| `--list` | str | List models for modality (or ALL) |
| `--switch` | MODALITY MODEL_ID | Switch active model |
| `--scan` | flag | Scan cloud for available models |
| `--probe` | str | Probe a specific model for rate limits |
| `--pull` | str | Download HF model via MLX |

### Examples
```bash
# Check current configuration
python3 model_scout.py --status

# Switch to local Flux for images
python3 model_scout.py --switch image flux-schnell

# List all video models
python3 model_scout.py --list video
```

---

## sassprilla_carbonator.py

**The Carbonator** — Auto-expands title-style prompts into rich visual concepts.

### Usage
```bash
python3 sassprilla_carbonator.py "Title" [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title` | positional | Required | Song/Movie title |
| `--artist` | str | `None` | Artist name |
| `--context` | str | `None` | Additional context (e.g., 'Cyberpunk', 'Slow') |
| `--run` | flag | `False` | Execute movie_producer automatically |

### Examples
```bash
python3 sassprilla_carbonator.py "Purple Rain" --artist "Prince"
python3 sassprilla_carbonator.py "Midnight Train To Georgia" --context "Melancholy"
```

---

## rescue_session.py

**The Rescuer** — Recovers and resumes failed sessions.

### Usage
```bash
python3 rescue_session.py <session_dir>
```

---

## prep_movie_assets.py

**The Prep** — Prepares training datasets from XMVP XML.

### Usage
```bash
python3 prep_movie_assets.py --xml <path> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--xml` | str | Required | Input XMVP XML file |
| `--out` | str | `"z_training_data/movies"` | Output root |
| `--force` | flag | `False` | Overwrite existing |

---

## still_life.py

**The Still** — Generates frame+audio slideshows from XMVP XML.

### Usage
```bash
python3 still_life.py --input <video> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | str | Required | Input MP4 file |
| `--xml` | str | `None` | Explicit path to XMVP XML |
| `--local` | flag | `False` | Force Local Mode |
| `--out` | str | `None` | Output directory |
| `--w` | int | `256` | Width |
| `--h` | int | `144` | Height |

---

# Bridge Modules

## flux_bridge.py

**FluxBridge** — Local image generation using Flux.1-schnell.

### Class: `FluxBridge`
- `generate(prompt, width, height, seed)` → PIL.Image
- `img2img(image, prompt, strength)` → PIL.Image

### Factory
```python
from flux_bridge import get_flux_bridge
bridge = get_flux_bridge()  # Singleton
```

---

## ltx_bridge.py

**LTXBridge** — Local video generation using LTX-Video.

### Class: `LTXBridge`
- `generate(prompt, num_frames, fps)` → Video path

### Factory
```python
from ltx_bridge import get_ltx_bridge
bridge = get_ltx_bridge()  # Singleton, lazy-loaded
```

---

## wan_bridge.py

**WanVideoBridge** — Local video generation using Wan 2.1 14B.

### Class: `WanVideoBridge`
- `generate(prompt, image, num_frames)` → Video path
- Supports keyframe chaining for long-form content

---

## kokoro_bridge.py

**KokoroBridge** — Local TTS using Kokoro ONNX.

### Class: `KokoroBridge`
- `generate(text, voice, output_path)` → Audio path
- Voices: `af_bella`, `af_sarah`, `af_nicole`, `am_michael`, `am_adam`, `bm_george`, etc.

---

## hunyuan_foley_bridge.py

**HunyuanFoleyBridge** — Local foley/sound effect generation.

### Class: `HunyuanFoleyBridge`
- `generate(prompt, duration)` → Audio path

---

# Core Libraries

## text_engine.py

**TextEngine** — Unified text generation interface (Cloud/Local).

### Class: `TextEngine`
- `generate(prompt, system_prompt, temperature)` → str
- `generate_json(prompt, schema)` → dict

### Factory
```python
from text_engine import get_engine
engine = get_engine()  # Uses active profile
```

---

## truth_safety.py

**TruthSafety** — Content moderation and PG filtering.

### Class: `TruthSafety`
- `sanitize(prompt, pg_mode)` → str
- `check_safety(content)` → bool

---

## definitions.py

**Definitions** — Model registry and VP Form configurations.

### Enums
- `BackendType`: `CLOUD`, `LOCAL`
- `Modality`: `TEXT`, `IMAGE`, `VIDEO`, `FOLEY`, `SPOKEN_TTS`, `CLONED_TTS`

### Key Functions
| Function | Description |
|----------|-------------|
| `get_video_model(key)` | Legacy accessor for video models |
| `get_active_model(modality)` | Get currently active model config |
| `set_active_model(modality, model_id)` | Set and persist active model |
| `resolve_vpform(input_string)` | Resolve form key or alias |

### VP Form Registry

| Key | Aliases | Description |
|-----|---------|-------------|
| `music-video` | `mv`, `music-agency` | Music Video (Story/Agency Mode) |
| `music-visualizer` | `viz`, `visualizer`, `audio-reactive` | Abstract Music Visualizer |
| `creative-agency` | `ca`, `commercial`, `ad`, `agency` | Commercial/Creative Agency |
| `tech-movie` | `tech`, `tm` | Tech/Code Movie |
| `draft-animatic` | `animatic`, `draft`, `storyboard` | Static Storyboard |
| `full-movie` | `feature`, `movie` | Full-length feature |
| `movies-movie` | `mm`, `remake`, `blockbuster` | Condensed Blockbuster Remake |
| `parody-movie` | `pm`, `spoof`, `parody` | Direct Parody/Spoof |
| `parody-video` | `pv`, `music-parody` | Music-Synced Parody |
| `thax-douglas` | `thax`, `td` | Thax Douglas Spoken Word |
| `gahd-podcast` | `gahd`, `god`, `history` | Great Moments in History |
| `24-podcast` | `24`, `news` | 24-minute 4-person improv |
| `10-podcast` | `10`, `tech-news` | 10-minute topical podcast |
| `route66-podcast` | `r66`, `route66` | 6-Person Road Trip Narrative |
| `fullmovie-still` | `fms`, `slideshow` | Frame+Audio Slideshow |

### Registered Models

#### Text
| ID | Backend | Path/Endpoint |
|----|---------|---------------|
| `gemini-2.0-flash` | cloud | - |
| `gemini-1.5-pro` | cloud | - |
| `gemma-2-9b-it` | local | `/Volumes/XMVPX/mw/gemma-root` |
| `gemma-2-9b-it-director` | local | `/Volumes/XMVPX/mw/gemma-root` + adapter |

#### Image
| ID | Backend | Path/Endpoint |
|----|---------|---------------|
| `gemini-2.5-flash-image` | cloud | - |
| `imagen-3` | cloud | - |
| `flux-schnell` | local | `/Volumes/XMVPX/mw/flux-root` |

#### Video
| ID | Backend | Path/Endpoint |
|----|---------|---------------|
| `veo-3.1-fast` | cloud | veo-3.1-fast-generate-preview |
| `veo-3.1-4k` | cloud | veo-3.1-generate-preview |
| `ltx-video` | local | `/Volumes/XMVPX/mw/LT2X-root` |

#### TTS
| ID | Backend | Path/Endpoint |
|----|---------|---------------|
| `google-journey` | cloud | en-US-Journey-F |
| `kokoro-v1` | local | `/Volumes/XMVPX/mw/kokoro-root/kokoro-v0_19.onnx` |

---

## mvp_shared.py

**Shared Data Models & Utilities**

### Data Models (Pydantic)

| Model | Description |
|-------|-------------|
| `VPForm` | Genre and output mechanics |
| `Constraints` | Technical limits (resolution, FPS, duration) |
| `CSSV` | The "Bible" — Constraints, Scenario, Situation, Vision |
| `Story` | Narrative backbone |
| `Portion` | High-level narrative chunk |
| `Seg` | Executable technical segment |
| `Indecision` | A/B test choice |
| `DialogueLine` | Single line of dialogue |
| `DialogueScript` | Full dialogue script |
| `Manifest` | Segment-to-file mapping |

### I/O Functions

| Function | Description |
|----------|-------------|
| `load_cssv(path)` | Load CSSV from JSON |
| `save_cssv(cssv, path)` | Save CSSV to JSON |
| `load_manifest(path)` | Load Manifest from JSON |
| `save_manifest(manifest, path)` | Save Manifest to JSON |
| `load_api_keys(env_path)` | Load API keys from YAML |
| `load_text_keys(env_path)` | Load TEXT_KEYS_LIST |
| `save_xmvp(data_models, path)` | Save to XMVP XML format |
| `load_xmvp(path, key)` | Load specific key from XMVP XML |
| `get_client()` | Get rotated genai.Client |
| `get_project_id()` | Get GCP project ID |

---

## frame_canvas.py

**Code Painter** — Procedural image generation using Gemini-generated code.

### Features
- Multi-stage procedural generation (pixel, refine, degrade passes)
- Gemini-powered NumPy/SciPy code generation
- Safe execution with confidence scoring

### Usage
Typically invoked via `--fc` flag:
```bash
python3 cartoon_producer.py --prompt "Abstract art" --fc
```

---

# Data Models & Schemas

## CSSV (Bible) Structure

```json
{
  "constraints": {
    "width": 768,
    "height": 768,
    "fps": 24,
    "max_duration_sec": 60.0,
    "target_segment_length": 8.0,
    "black_and_white": false,
    "silent": false,
    "style_bans": []
  },
  "scenario": "A 60.0-second tech-movie",
  "situation": "CONCEPT: Your concept here",
  "vision": "STYLE: ... AESTHETIC: ... PACING: ..."
}
```

## Manifest Structure

```json
{
  "segs": [
    {
      "id": 1,
      "start_frame": 0,
      "end_frame": 96,
      "prompt": "Scene description...",
      "action": "static",
      "model_overrides": {}
    }
  ],
  "files": {},
  "indecisions": [],
  "dialogue": null
}
```

## XMVP XML Format

```xml
<?xml version='1.0' encoding='utf-8'?>
<XMVP version="2.80">
  <Bible>{JSON}</Bible>
  <Story>{JSON}</Story>
  <Manifest>{JSON}</Manifest>
</XMVP>
```

---

# Configuration Files

## env_vars.yaml

```yaml
# Text Engine Selection
TEXT_ENGINE: "gemini_api"  # or "local_gemma"
LOCAL_MODEL_PATH: ""       # HuggingFace ID or path

# API Keys
GEMINI_API_KEY: "YOUR_KEY"
ACTION_KEYS_LIST: "key1,key2,key3,..."  # 16 keys for video/image (rotated)
TEXT_KEYS_LIST: "key4,key5,..."         # 8 keys for text operations
```

## active_models.json

Auto-generated profile tracking current active models:

```json
{
  "text": "gemini-2.0-flash",
  "image": "gemini-2.5-flash-image",
  "video": "veo-3.1-fast",
  "spoken_tts": "google-journey"
}
```

---

# Training & Voice Models

## Directory Structure

```
z_training_data/
├── parsed_scripts/          # Parsed screenplay JSONs for DialogueCritic
│   └── *.json
├── thax_voice/              # Thax Douglas voice model
│   └── model/
│       ├── thax.pth         # RVC model weights
│       └── thax.index       # RVC index file
├── 24_voices/               # 24-podcast character voices
└── route66_voices/          # Route 66 character voices
```

## Using the Thax Douglas Voice

1. Ensure files are in `z_training_data/thax_voice/model/`
2. Set up RVC environment: `conda create -n rvc_env python=3.10 && pip install rvc-python`
3. Run: `python3 content_producer.py --vpform thax-douglas`

---

*Generated for XMVP v2.80 — January 2026*
