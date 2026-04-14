# XMVP: The Modular Vision Pipeline

**Frame-by-frame animation, beat-synced visualizers, and audio for your scripts — all from a single command, locally or via the cloud. Own your creative pipeline.**

XMVP is an open-source creative production toolkit that turns text prompts into animated shorts, music videos, audio-reactive visualizations, and multi-character spoken-word content. Everything runs on your Mac, with optional cloud acceleration. No subscriptions, no per-generation fees, no content filters (unless you want them).

---

## Why XMVP?

Generative video models are getting good at producing short clips of photorealistic footage. XMVP does something different: it builds *structured productions* where an LLM directs every frame, scene, and cut. The pipeline decomposes creative work into specialist stages — a Vision Producer writes the brief, a Writers Room breaks it into scenes, a Director calls the shots, and a Post house stitches the result — so you get narrative coherence instead of a single incoherent clip.

Version 3.00 focuses the toolkit around three production modes:

**Cartoons** — Frame-by-frame image generation guided by GemmaW (a fine-tuned Gemma director model with included adapter weights), with beat-synced shot planning, Flux/Gemini rendering, img2img coherence, and Wan 2.1 keyframe animation. This is the core of XMVP: structured, LLM-directed animation from prompt to final cut.

**Visualizers** — Two flavors. The *procedural* path (ANSI and Unicode visualizers) uses Demucs stem separation to drive per-instrument character animations at 24–30 FPS with no AI inference at all — pure math on the audio signal. The *LLM-directed* path has the model "draw" each frame in block characters or Unicode art, then renders to video. Both sync to music.

**Content** — Multi-character podcast and spoken-word generation with Kokoro TTS (local) or Google Journey (cloud), optional RVC voice cloning, generative foley via Hunyuan, and a per-line visual illustration mode. Formats include 4-person improv comedy, historical radio drama, road-trip narratives, Thax Douglas spoken word (with his voice model, shared with permission), and Element 47 audio plays.

---

## Quick Start (Cloud Mode)

```bash
# 1. Clone and install
git clone https://github.com/0gsd/xmvp.git && cd xmvp
pip install -r requirements.txt

# 2. Configure
cp env_vars.example.yaml env_vars.yaml
# Edit env_vars.yaml — add your Gemini API key(s)

# 3. Make a cartoon
python3 cartoon_producer.py --prompt "A melancholy astronaut drifts through a neon city" --style "Pixel Art"
```

Output lands in `z_test-outputs/cartoons/`.

---

## The "Local First" Setup

With the right hardware and a big external drive, you can run the entire pipeline offline — no API keys, no per-generation costs, no content filters.

### What You'll Need

- **Mac with Apple Silicon** (M1/M2/M3/M4, 16GB+ RAM recommended)
- **External SSD** (1TB+, named `XMVPX`)
- **~100GB** disk space for model weights
- **Python 3.10+** with Miniforge

### Step 1: Prepare Your External Drive

Format an external SSD and name it **`XMVPX`**. Create the weights directory:

```bash
mkdir -p /Users/m3u/METMcloud/METMroot/tools/fmv/weights
```

Final structure:

```
/Users/m3u/METMcloud/METMroot/tools/fmv/weights/
├── flux-root/           # Flux.1-schnell + Klein 9B (image generation)
├── gemma-root/          # Gemma 3 (text/direction)
├── t5weights-root/      # T5 encoder for Flux
├── kokoro-root/         # Kokoro TTS (speech)
├── hunyuan-foley/       # Hunyuan Foley (sound effects)
├── LT2X-root/           # LTX-Video (video clips, optional)
├── wan-root/            # Wan 2.1 (keyframe animation, optional)
├── skyreels-root/       # SkyReels A2V (audio-to-video, optional)
├── flux-gguf-root/      # Flux GGUF quantized (low-memory option)
└── rvc-root/            # RVC base assets (voice cloning, optional)
```

### Step 2: Set Up Python

```bash
brew install miniforge
conda create -n xmvp python=3.10
conda activate xmvp

pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install mlx mlx-lm mflux            # Gemma (text) & MFLUX (Native Apple Silicon Image Gen)
pip install diffusers transformers        # LTX Server Fallback
pip install kokoro-onnx soundfile         # Kokoro TTS
pip install librosa pyloudnorm demucs     # Audio analysis + stem splitting
```

### Step 3: Download Models

```bash
conda activate xmvp
python3 populate_models_xmvp.py
```

This prompts for a HuggingFace token and downloads everything to `/Users/m3u/METMcloud/METMroot/tools/fmv/weights/`. Expect ~400GB.

Manual download reference:

| Model | HuggingFace Repo | Target Folder |
|-------|------------------|---------------|
| Flux Schnell | `black-forest-labs/FLUX.1-schnell` | `flux-root/` |
| Flux 2 Klein 9B | (via populate script) | `flux-root/klein-9b/` |
| Gemma 3 | `google/gemma-3-27b-it` | `gemma-root/` |
| T5 Encoder | `city96/t5-v1_1-xxl-encoder-bf16` | `t5weights-root/` |
| Kokoro TTS | `Kijai/Kokoro-82M-ONNX` | `kokoro-root/` |

### Step 4: Verify

```bash
python3 model_scout.py --status
```

---

## The Three Producers

### 🎨 `cartoon_producer.py` — The Animator

The primary creative engine. Generates frame-by-frame animation from prompts using an LLM director (GemmaW locally, Gemini in the cloud) to write per-frame image prompts, then renders with Flux or Gemini image generation.

```bash
# Prompt-driven animation (creative agency mode)
python3 cartoon_producer.py --prompt "A sad robot finds purpose" --style "Pixel Art"

# Music video with beat-synced narrative
python3 cartoon_producer.py --vpform music-video --mu song.mp3 --prompt "Neon dreams"

# Beat-synced procedural visualizer (no AI, pure signal processing)
python3 cartoon_producer.py --vpform music-visualizer --mu ambient.wav

# Frame-by-frame video restyling (img2img)
python3 cartoon_producer.py --vpform cartoon-video --mu input.mp4 --style "Oil painting"

# Full-length feature animatic
python3 cartoon_producer.py --vpform full-movie --prompt "The Odyssey" --local --slength 600

# Wan 2.1 keyframe animation (local only)
python3 cartoon_producer.py --prompt "Dancing in the rain" --wan --local

# Beat-matched clip montage from a video folder
python3 cartoon_producer.py --vpform clip-video --mu track.mp3 --f /path/to/clips/
```

### 🎙️ `content_producer.py` — The Podcast Factory

Generates scripted or improvised multi-character audio content with per-line visual illustration.

```bash
# 24-minute 4-person improv comedy
python3 content_producer.py --vpform 24-podcast --local

# Great Moments in History (dramatized radio format)
python3 content_producer.py --vpform gahd-podcast --ep 207 --local --location "The Colosseum"

# 6-person road trip narrative (66 minutes)
python3 content_producer.py --vpform route66-podcast --rvc --local --slength 3960

# Thax Douglas spoken word (included voice model)
python3 content_producer.py --vpform thax-douglas

# Element 47 audio play (from Fountain script)
python3 content_producer.py --vpform element-47 --xb script.fountain --local

# Full-movie slideshow (XMVP XML → frame+audio)
python3 content_producer.py --vpform fullmovie-still --xb manifest.xml

# Audio-only play (MP3 output)
python3 content_producer.py --vpform audio-play --xb script.xml --local

# Audio-to-video (SkyReels A2V)
python3 content_producer.py --vpform audio-movie --xb manifest.xml --mu master.wav --local
```

### 🎞️ `post_production.py` — The Editor

Upscaling, frame interpolation, retiming, and audio stitching.

```bash
# 2x upscale with Flux img2img
python3 post_production.py video.mp4 --scale 2.0

# Frame interpolation (2x smoother via AI tweening)
python3 post_production.py video.mp4 -x 2

# Sync video to audio duration
python3 post_production.py video.mp4 --mu soundtrack.mp3 --stitch-audio

# Retime to specific framerate
python3 post_production.py video.mp4 --framerate 24.0

# VDJ blend mode (two video layers + audio)
python3 post_production.py --vvaudio --bottomvideo base.mp4 --topvideo overlay.mp4 --mu mix.mp3
```

---

## Standalone Visualizers

These run independently of the producer pipeline — point them at an audio file and get a video.

### `ansi_visualizer.py` — Procedural ANSI Animation

Splits audio into four stems via Demucs (drums, bass, keys, other), generates per-track ASCII animations driven by loudness and spectral character, composites the layers with opacity blending, and muxes synced audio into a final MP4.

```bash
python3 ansi_visualizer.py --mu song.mp3 --fps 24
python3 ansi_visualizer.py --mu song.wav --fps 30 --width 120 --height 40
```

### `unicode_visualizer.py` — Extended Unicode Animation

Same stem-splitting pipeline with 140K+ Unicode characters, themed character pools (Matrix, Emoji, Braille, Geometric, etc.), and per-section theme randomization.

```bash
python3 unicode_visualizer.py --mu song.mp3 --fps 24 --theme matrix
python3 unicode_visualizer.py --mu song.wav --theme emoji
python3 unicode_visualizer.py --mu song.mp3 --theme random
```

---

## Understanding the Pipeline

When you run `cartoon_producer.py` in creative-agency mode, the internal sequence is:

```
1. VISION PRODUCER    → Creates the "Bible" (concept, style, constraints)
2. STUB REIFICATION   → Expands into a Story (characters, arc, theme)
3. WRITERS ROOM       → Breaks into timed Portions (scenes)
4. PORTION CONTROL    → Calculates frame ranges
5. SHOT PLANNING      → Beat-synced cut points (if music provided)
6. DISPATCH DIRECTOR  → Generates image/video assets per frame
7. POST PRODUCTION    → Stitches, interpolates, and finalizes
8. XMVP EXPORT       → Saves everything to XML
```

Each module can run independently for debugging or custom workflows:

```bash
python3 vision_producer.py --vpform creative-agency --prompt "AI rebellion" --out bible.json
python3 stub_reification.py --bible bible.json --out story.json
python3 writers_room.py --bible bible.json --story story.json --out portions.json
python3 portion_control.py --bible bible.json --portions portions.json --out manifest.json
```

---

## VP Forms Reference

| Form | Aliases | Producer | Description |
|------|---------|----------|-------------|
| `creative-agency` | `ca`, `commercial`, `ad`, `agency` | cartoon | LLM-directed animation from prompt |
| `music-video` | `mv`, `music-agency` | cartoon | Beat-synced narrative animation to audio |
| `music-visualizer` | `viz`, `visualizer`, `audio-reactive` | cartoon | Procedural stem-reactive visualizer |
| `cartoon-video` | `cv`, `vid2vid`, `rotoscope` | cartoon | Frame-by-frame video restyling |
| `clip-video` | (via `--f` flag) | cartoon | Beat-matched clip montage |
| `full-movie` | `feature`, `movie` | cartoon | Full-length feature animatic |
| `tech-movie` | `tech`, `tm` | cartoon | Tech/code themed animation |
| `draft-animatic` | `animatic`, `draft`, `storyboard` | cartoon | Static storyboard mode |
| `3d-movie` | `3d`, `blender`, `cgi` | cartoon | 3D via Blender/bpy |
| `ansi-video` | `ansi`, `ascii`, `pixel-art`, `blocks` | (registered) | LLM-drawn ANSI block animation |
| `ansi-redraw` | `ansi-trace`, `ascii-redraw`, `block-trace` | (registered) | LLM redraws video as block art |
| `24-podcast` | `24`, `news` | content | 4-person improv comedy (24 min) |
| `10-podcast` | `10`, `tech-news` | content | Topical tech podcast (10 min) |
| `route66-podcast` | `r66`, `route66` | content | 6-person road trip narrative (66 min) |
| `gahd-podcast` | `gahd`, `god`, `history` | content | Great Moments in History |
| `thax-douglas` | `thax`, `td` | content | Spoken word (included voice model) |
| `element-47` | `e47`, `element47` | content | Element 47 audio play |
| `fullmovie-still` | `fms`, `slideshow` | content | Frame+audio slideshow from XML |
| `audio-play` | `ap`, `audioplay`, `play` | content | Audio-only play (MP3) |
| `audio-movie` | `am`, `a2v`, `audiomovie` | content | Audio-to-video via SkyReels |
| `black-box` | `bb`, `theater`, `stage`, `min` | content | Minimalist theater mode |
| `parody-movie` | `pm`, `spoof`, `parody` | (legacy) | Direct parody/spoof |
| `parody-video` | `pv`, `music-parody` | (legacy) | Music-synced parody |
| `movies-movie` | `mm`, `remake`, `blockbuster` | (legacy) | Condensed blockbuster remake |

---

## Cloud vs Local

| Feature | Cloud | Local |
|---------|-------|-------|
| Text / Direction | Gemini 2.0 Flash | Gemma 3 27B (MLX) |
| Image Rendering | Gemini Flash / Imagen 3 | Flux Schnell / Klein 9B |
| Video Clips | (not used in v3) | LTX-Video / Wan 2.1 |
| Speech | Google Journey TTS | Kokoro ONNX |
| Sound Effects | — | Hunyuan Foley / SFX Bridge |
| Cost | Per-generation API fees | Free after setup |
| Content Filters | Google safety filters | None (unless `--pg`) |

### PG Mode

When `--pg` is enabled: children are replaced with adults in prompts, celebrities become "impersonator performing as [Name]", violence/gore/nudity removed. Works in both cloud and local modes.

Without `--pg` in local mode: no filters. Full artistic freedom.

---

## XMVP XML Format

Every production exports to an open XML format that captures the full creative state:

```xml
<?xml version='1.0' encoding='utf-8'?>
<XMVP version="3.00">
  <Bible>{"constraints": {...}, "scenario": "...", "situation": "...", "vision": "..."}</Bible>
  <Story>{"title": "...", "synopsis": "...", "characters": [...]}</Story>
  <Manifest>{"segs": [...], "files": {...}}</Manifest>
</XMVP>
```

Re-render any XMVP file with different settings. With v3.00+, passing `--xb` to an XML that contains a `GeneratedFrames` array will automatically bypass the LLM director and perform a **1:1 Frame-accurate re-render** with the exact original prompts (e.g., pumping a local sequence up to Gemini on the cloud).

```bash
# General Re-render or 1:1 Prompt Mapping
python3 cartoon_producer.py --xb previous_run.xml --local

# E.g. Send local frames directly to the cloud
python3 cartoon_producer.py --xb previous_local_run.xml --cloud --model gemini-2.5-flash
```

---

## Included Adapters & Training Data

XMVP ships with fine-tuned adapter weights and training data:

- **GemmaW Director** (`adapters/director_v1/`) — Gemma adapter trained to write cinematic frame prompts
- **Movie-Level LoRA Templates** (`adapters/movies/`) — Pre-trained Flux LoRA templates for consistent style
- **Thax Douglas Voice** (`z_training_data/thax_voice/`) — RVC model, shared with permission
- **Element 47 Voices** (`z_training_data/e47_voices/`) — 4-character voice reference audio
- **NICOTIME Index** (`z_training_data/nicotime/`) — Noospheric entity research documents
- **Example Parodies** (`z_training_data/example_parodies/`) — Reference scripts

---

## Troubleshooting

**"No API Keys found"** → Ensure `env_vars.yaml` exists with valid keys.

**"Local model not found"** → Verify `/Users/m3u/METMcloud/METMroot/tools/fmv/weights/` contains model folders. Run `python3 model_scout.py --status`.

**"MPS not available"** → Requires macOS 12.3+ on Apple Silicon. Falls back to CPU (slow).

**"Out of memory"** → Close other apps, try smaller `--slength`, 16GB+ recommended.

**Rate limits (429)** → Add more keys to `ACTION_KEYS_LIST`, use `--local`, or increase `--delay`.

**"RVC conversion failed"** → Set up RVC environment: `conda create -n rvc_env python=3.10 && pip install rvc-python`.

---

## Tips

1. **Start small**: `--slength 30` or `--limit 10` for quick tests
2. **Check models**: `python3 model_scout.py --status`
3. **Local = uncensored**: `--local` has no content filters unless you add `--pg`
4. **Output locations**: Cartoons in `z_test-outputs/cartoons/`, content in `z_test-outputs/`
5. **Auto-Carbonation**: Short, title-case prompts get auto-expanded by SASSPRILLA into rich visual concepts
6. **Chaos Seeds**: `--cs 2` injects random Wikipedia concepts for creative serendipity

---

## Contributing

XMVP is a personal project shared because the "modular vision pipeline" concept is useful. Issues and PRs welcome, but no promises on response time.

## License

Free and open for use by all. You'll need your own API keys for cloud mode, or your own hardware for local mode. The included Thax Douglas voice model is shared with permission for creative use.

---

*"A reasoning, bureaucratic chain of simulated production specialists."*