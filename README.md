# XMVP: The Modular Vision Pipeline

**Make movies from prompts. Run locally. Own your creative pipeline.**

XMVP is an open-source "film studio in a box" that decomposes video production into specialist modules—Producers, Writers, Directors, Editors—that can run in the cloud, entirely on your Mac, or any combination you choose.

---

## What Can XMVP Do?

- **Generate full video sequences** from a single text prompt
- **Create music videos** synced to any audio track
- **Animate frame-by-frame** with consistent style
- **Run 100% locally** on Apple Silicon (M1/M2/M3/M4) for privacy and uncensored creativity
- **Export everything** to the open XMVP XML format for editing, re-rendering, or sharing
- **Auto-expand titles** into rich visual concepts with the SASSPRILLA Carbonator
- **Generate spoken word content** with cloned voices (Thax Douglas model included!)

---

## Quick Start (Cloud Mode)

If you just want to try XMVP with cloud APIs:

```bash
# 1. Clone the repo
git clone https://github.com/0gsd/xmvp.git
cd xmvp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your config
cp env_vars.example.yaml env_vars.yaml
# Edit env_vars.yaml and add your Gemini API key(s)

# 4. Make a movie!
python3 movie_producer.py "A cyberpunk detective story" --seg 4 --vm K
```

Your video will appear in `z_test-outputs/movies/finalcuts/`.

---

## The "Local First" Setup

This is where XMVP really shines. With the right hardware and a big external drive, you can run the entire pipeline offline—no API keys, no per-generation costs, no content filters (unless you want them).

### What You'll Need

- **Mac with Apple Silicon** (M1/M2/M3/M4 with 16GB+ RAM recommended)
- **External SSD** (1TB+ recommended, named `XMVPX`)
- **About 100GB** of disk space for model weights
- **Python 3.10+** with Miniconda/Miniforge

### Step 1: Prepare Your External Drive

Format an external SSD and name it **`XMVPX`**. This is the standard mount point XMVP expects.

Create the model weights directory:

```bash
mkdir -p /Volumes/XMVPX/mw
```

The `mw/` folder will contain all local model weights. Here's what the final structure should look like:

```
/Volumes/XMVPX/
└── mw/
    ├── flux-root/           # Flux.1-schnell (Image generation)
    ├── LT2X-root/           # LTX-Video (Video generation)
    ├── gemma-root/          # Gemma 3 (Text generation)
    ├── t5weights-root/      # T5 encoder for Flux
    ├── kokoro-root/         # Kokoro TTS (Speech)
    ├── hunyuan-foley/       # Hunyuan Foley (Sound effects)
    ├── wan-root/            # Wan 2.1 (Speech-to-video, optional)
    ├── comfyui-root/        # ComfyUI (optional, for advanced workflows)
    ├── indextts-root/       # IndexTTS (optional, for cloned voices)
    └── rvc-root/            # RVC base assets (optional)
```

### Step 2: Set Up Your Python Environment

We recommend using Miniforge (conda for Apple Silicon):

```bash
# Install Miniforge if you haven't
brew install miniforge

# Create the XMVP environment
conda create -n xmvp python=3.10
conda activate xmvp

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install core dependencies
pip install -r requirements.txt

# Install local inference packages
pip install mlx mlx-lm                    # For Gemma (text)
pip install diffusers transformers        # For Flux & LTX
pip install kokoro-onnx soundfile         # For Kokoro TTS
pip install librosa pyloudnorm            # For audio analysis
```

### Step 3: Download Model Weights

XMVP includes a helper script that downloads everything you need:

```bash
conda activate xmvp
python3 populate_models_xmvp.py
```

This will:
1. Prompt you for a HuggingFace token (needed for gated models like Gemma)
2. Download all models to `/Volumes/XMVPX/mw/`
3. Clone ComfyUI for advanced workflows

**⚠️ This downloads ~400GB of model weights. Go get coffee.**

If you prefer to download models manually or already have some:

| Model | HuggingFace Repo | Target Folder |
|-------|------------------|---------------|
| Flux Schnell | `black-forest-labs/FLUX.1-schnell` | `flux-root/` |
| LTX-Video | `Lightricks/LTX-Video` | `LT2X-root/` |
| Gemma 3 | `google/gemma-3-27b-it` | `gemma-root/` |
| T5 Encoder | `city96/t5-v1_1-xxl-encoder-bf16` | `t5weights-root/` |
| Kokoro TTS | `Kijai/Kokoro-82M-ONNX` | `kokoro-root/` |

### Step 4: Test Your Setup

```bash
# Check model registry status
python3 model_scout.py --status

# Run a local test
python3 movie_producer.py "A robot painting a sunset" --local --seg 2
```

If everything is configured correctly, you'll see:
```
🏠 Local Mode Enabled: Switching models to Local Gemma (Text) and LTX (Video).
   📍 Local Text Model Path: /Volumes/XMVPX/mw/gemma-root
   🛡️ Safety Filters: OFF (Uncensored)
   ✨ Quality Refinement: ON (Hyper-Detailed Fattening)
```

For **Long-Form** content (e.g. `full-movie`), XMVP now uses **Wan 2.1** and **Micro-Batching**:

```bash
python3 movie_producer.py "The Odyssey" --vpform full-movie --local --slength 600
```

This will trigger the iterative writer, generating scenes in 180s chunks until the 600s target is reached.

---

## The Three Producers

XMVP has three main entry points, each suited to different workflows:

### 🎬 `movie_producer.py` — The Showrunner

Creates structured video content (ads, trailers, music videos) from prompts.

```bash
# Cloud mode (Veo 3.1 + Gemini)
python3 movie_producer.py "A noir detective story" --seg 6 --vm K

# Local mode (LTX + Gemma) — uncensored!
python3 movie_producer.py "Underground rave documentary" --local --seg 8

# Music video synced to a track
python3 movie_producer.py "Abstract visuals" --vpform music-video --mu song.mp3

# Long-form movie (Micro-Batching enabled!)
# Generates 50 minutes of content in 3-minute iterative batches
python3 movie_producer.py "Epic Space Opera" --vpform full-movie --local --slength 3000

# Auto-carbonation: just give it a title!
python3 movie_producer.py "Midnight Train To Georgia"

# Using VPForm aliases
python3 movie_producer.py tech-movie "AI Awakening"
python3 movie_producer.py draft-animatic "Space Opera Epic"
```

### 🎨 `cartoon_producer.py` — The Animator

Frame-by-frame animation and audio-reactive visuals.

```bash
# Creative agency mode (prompt → story → animation)
python3 cartoon_producer.py --prompt "A melancholy astronaut" --style "Pixel Art"

# Music video with beat-synced narrative
python3 cartoon_producer.py --vpform music-agency --mu track.mp3 --prompt "Cyberpunk chase"

# Abstract visualizer
python3 cartoon_producer.py --vpform music-visualizer --mu ambient.wav
```

### 🎙️ `content_producer.py` — The Podcast Factory

Generates improv comedy and spoken word content.

```bash
# 24-minute 4-person improv special
python3 content_producer.py --vpform 24-podcast

# Thax Douglas spoken word (uses included voice model!)
python3 content_producer.py --vpform thax-douglas

# Local mode with Kokoro TTS
python3 content_producer.py --vpform 24-podcast --local
```

### 🎞️ `post_production.py` — The Editor

Upscaling, interpolation, and audio stitching.

```bash
# 2x upscale with Flux
python3 post_production.py video.mp4 --local --scale 2.0

# Frame interpolation (2x smoother)
python3 post_production.py video.mp4 --local -x 2

# Sync video to audio duration
python3 post_production.py video.mp4 --mu soundtrack.mp3 --stitch-audio
```

---

## Understanding the Pipeline

When you run `movie_producer.py`, it orchestrates this sequence:

```
1. VISION PRODUCER    → Creates the "Bible" (concept, style, constraints)
2. STUB REIFICATION   → Expands into a full Story (characters, arc)
3. WRITERS ROOM       → Breaks into timed Portions (scenes)
4. PORTION CONTROL    → Calculates frame ranges
5. DISPATCH DIRECTOR  → Generates video/image assets
6. POST PRODUCTION    → Stitches and finalizes
7. XMVP EXPORT        → Saves everything to XML
```

Each module can be run independently for debugging or custom workflows:

```bash
# Generate just the "Bible"
python3 vision_producer.py --vpform tech-movie --prompt "AI rebellion" --out bible.json

# Expand to story
python3 stub_reification.py --bible bible.json --out story.json

# Continue the chain...
```

---

## New in v2.80

### 🫧 SASSPRILLA Carbonator
Auto-expands title-style prompts into dense, genre-appropriate visual concepts:
```bash
python3 sassprilla_carbonator.py "Purple Rain" --artist "Prince"
```

### 🎤 Thax Douglas Voice Model
Included in `z_training_data/thax_voice/` — a trained RVC model of Chicago poet Thax Douglas, shared with his blessing. Use with:
```bash
python3 content_producer.py --vpform thax-douglas
```

### 🎭 Dialogue Critic (Gemma Wittgenstein)
Refines generated dialogue against a corpus of professional screenplays for more natural, cinematic lines.

### 📹 Wan 2.1 Bridge
New local video generation option using Wan 2.1 14B with keyframe chaining.

### 🎬 Dispatch Animatic
High-speed storyboard generation using Gemma + Flux for rapid visualization.

### 📋 VP Form Registry
Unified form system with aliases — use `tech-movie` or `tm`, `music-video` or `mv`:
```bash
python3 movie_producer.py mv "Neon Dreams"
```

---

## Configuration

### env_vars.yaml

Copy `env_vars.example.yaml` to `env_vars.yaml` and configure:

```yaml
# Engine Selection
TEXT_ENGINE: "gemini_api"      # "gemini_api" or "local_gemma"
LOCAL_MODEL_PATH: ""           # Only needed if using non-standard path

# API Keys (for cloud mode)
GEMINI_API_KEY: "YOUR_KEY_HERE"
ACTION_KEYS_LIST: "key1,key2,key3"   # For video generation (rotated)
TEXT_KEYS_LIST: "key4,key5"          # For text operations (fallback)
```

**Pro tip:** Use multiple API keys in `ACTION_KEYS_LIST` to avoid rate limits during batch generation.

### Model Switching

Use `model_scout.py` to manage which models are active:

```bash
# See current config
python3 model_scout.py --status

# Switch to local Flux for images
python3 model_scout.py --switch image flux-schnell

# Switch to cloud Veo for video
python3 model_scout.py --switch video veo-3.1-fast
```

---

## The XMVP Format

Every run exports to the open XMVP XML format:

```xml
<?xml version='1.0' encoding='utf-8'?>
<XMVP version="2.80">
  <Bible>{"constraints": {...}, "scenario": "...", "situation": "...", "vision": "..."}</Bible>
  <Story>{"title": "...", "synopsis": "...", "characters": [...]}</Story>
  <Manifest>{"segs": [...], "files": {...}}</Manifest>
</XMVP>
```

You can re-render any XMVP file:

```bash
# Re-render with different settings
python3 movie_producer.py --xb previous_run.xml --vm K --local
```

---

## Modes Explained

### Cloud vs Local

| Feature | Cloud | Local |
|---------|-------|-------|
| Text Generation | Gemini 2.0 Flash | Gemma 3 27B |
| Image Generation | Gemini Flash / Imagen 3 | Flux.1-schnell |
| Video Generation | Veo 3.1 | Wan 2.1 (full-movie) / LTX-Video (clips) |
| Speech | Google Journey TTS | Kokoro ONNX |
| Cost | Per-generation API fees | Free after setup |
| Content Filters | Google's safety filters | None (unless `--pg`) |
| Speed | Fast (cloud GPUs) | Depends on your Mac |

### PG Mode

When `--pg` is enabled:
- Children are replaced with adults in prompts
- Celebrities become "impersonator performing as [Name]"
- Violence/gore/nudity removed
- Works in both cloud and local modes

Without `--pg` in local mode: **No filters applied.** Full artistic freedom.

### Video Model Tiers

| Tier | Model | Use Case |
|------|-------|----------|
| `K` | veo-3.1-generate-preview | Cinematic 4K (highest quality) |
| `J` | veo-3.1-fast-generate-preview | Balanced speed/quality |
| `L` | veo-2.0-generate-001 | Light/fast |
| `D` | veo-2.0-generate-001 | Legacy Veo 2.0 |

---

## Troubleshooting

### "No API Keys found"
→ Make sure `env_vars.yaml` exists and has valid keys

### "Local model not found"
→ Check that `/Volumes/XMVPX/mw/` exists and contains model folders
→ Run `python3 model_scout.py --status` to verify paths

### "MPS not available"
→ You need macOS 12.3+ and an Apple Silicon Mac
→ Falls back to CPU (very slow)

### "Out of memory"
→ Close other apps, especially browsers
→ Try smaller `--seg` count
→ Local models are memory-hungry; 16GB+ recommended

### Rate limits (429 errors)
→ Add more keys to `ACTION_KEYS_LIST`
→ Use `--fast` for cheaper model tiers
→ Switch to `--local` mode

### "RVC conversion failed" (Thax mode)
→ Set up RVC environment: `conda create -n rvc_env python=3.10 && pip install rvc-python`
→ Check `RVC_PYTHON_BIN` environment variable points to correct Python

---

## Training Data & Voice Models

XMVP v2.80 includes:

- **Thax Douglas Voice Model** (`z_training_data/thax_voice/`) — RVC model for the Chicago poet, shared with his permission
- **Screenplay Corpus** (`z_training_data/parsed_scripts/`) — Parsed scripts for dialogue refinement (not included in public repo)

To use the Thax voice:
1. Ensure files are in `z_training_data/thax_voice/model/`
2. Set up RVC environment
3. Run `python3 content_producer.py --vpform thax-douglas`

---

## Contributing

XMVP is a personal project that I'm sharing because I think the "modular vision pipeline" concept is useful. Issues and PRs welcome, but no promises on response time.

---

## License

Free and open for use by all. You'll need your own API keys for cloud mode, or your own hardware for local mode.

The included Thax Douglas voice model is shared with permission for creative use.

---

## Getting Started

Once you have XMVP installed, your external drive (`/Volumes/XMVPX/mw/`) populated with model weights, and your `env_vars.yaml` configured with API keys (16 `ACTION_KEYS_LIST` for video/image generation, 8 `TEXT_KEYS_LIST` for text operations), you're ready to start creating.

Below are example commands for common workflows. Replace paths and parameters as needed for your setup.

---

### Converting Text to Video

**To create a parody movie from a text file you already have:**

```bash
python3 xmvp_converter.py /Volumes/XMVPX/mw/your-project/processed_text/Your_Script.txt \
    --vpform parody-movie \
    --slength 5820
```

This converts a pre-processed text file into a ~97-minute parody-format video.

---

### Post-Production: Stitching Audio to Video

**To add music or narration to an existing video sequence (folder of segments):**

```bash
python3 post_production.py \
    --input /path/to/your/video-segments-folder \
    --mu /path/to/your/audio/soundtrack.aif \
    --stitch-audio
```

The `--stitch-audio` flag syncs and combines your audio track with the video output.

**To process a single video file:**

```bash
python3 post_production.py video.mp4 --mu soundtrack.mp3 --stitch-audio
```

**To process a folder of numbered frame images:**

```bash
python3 post_production.py /path/to/frames/ --mu audio.aif --stitch-audio
```

---

### Cloud Mode: Quick Movie from a Prompt

**To create a segmented movie using cloud APIs (Gemini + Veo):**

```bash
python3 movie_producer.py "Your Movie Title (Year)" \
    --vpform parody-movie \
    --pg \
    --vm L \
    --seg 12
```

- `--pg` enables PG-safe content filtering
- `--vm L` selects the "Light/Fast" video model tier
- `--seg 12` creates 12 segments

---

### Local Mode: Full-Length Movie (Uncensored)

**To create a full-length movie running entirely on your Mac:**

```bash
python3 movie_producer.py "Your Creative Movie Title" \
    --vpform full-movie \
    --local \
    --slength 3000
```

- `--local` uses Gemma for text and Wan 2.1/LTX for video (no API costs, no content filters)
- `--slength 3000` targets a 50-minute runtime

---

### Podcast Content: Route 66 Format

**To create a Route 66-style podcast episode with RVC voice conversion:**

```bash
python3 content_producer.py \
    --vpform route66-podcast \
    --rvc \
    --local \
    --slength 3960 \
    --ep 301 \
    --location "The Roadside Diner"
```

- `--rvc` enables Real Voice Cloning
- `--ep 301` sets episode number (Season 3, Episode 1)
- `--location` sets the narrative location

---

### Podcast Content: GAHD Format

**To create a Great Moments in History podcast episode:**

```bash
python3 content_producer.py \
    --vpform gahd-podcast \
    --slength 3200 \
    --ep 207 \
    --local \
    --location "The Colosseum at Dawn"
```

Replace the location with your preferred setting.

---

### Podcast Content: 24-Minute Improv Special

**To create a 24-minute 4-person improv podcast:**

```bash
python3 content_producer.py \
    --vpform 24-podcast \
    --local \
    --slength 1440
```

The `--slength 1440` sets the target duration to 24 minutes (1440 seconds).

---

### Quick Reference: VP Forms

| Form | Alias | Description |
|------|-------|-------------|
| `parody-movie` | `pm` | Parody-style movie content |
| `full-movie` | `fm`, `feature`, `movie` | Full-length feature (uses micro-batching) |
| `music-video` | `mv`, `music-agency` | Music video synced to audio |
| `tech-movie` | `tm`, `tech` | Technology-themed content |
| `route66-podcast` | `r66`, `route66` | Route 66 travel podcast format |
| `gahd-podcast` | `gahd`, `god`, `history` | Great Moments in History podcast |
| `24-podcast` | `24`, `news` | 24-minute improv special |
| `thax-douglas` | `thax`, `td` | Thax Douglas spoken word |
| `draft-animatic` | `animatic`, `draft` | Static storyboard mode |

---

### Tips for New Users

1. **Start small**: Try `--seg 2` or `--slength 120` for quick test runs
2. **Check model status**: Run `python3 model_scout.py --status` to verify your local models are configured
3. **Watch your keys**: Rotate through `ACTION_KEYS_LIST` keys to avoid rate limits on cloud mode
4. **Local = uncensored**: `--local` mode has no content filters unless you add `--pg`
5. **Output location**: Videos appear in `z_test-outputs/movies/finalcuts/` by default

---

*"A reasoning, bureaucratic chain of simulated movie and video production specialists."*
