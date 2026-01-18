# XMVP: The Modular Vision Pipeline

**Make movies from prompts. Run locally. Own your creative pipeline.**

XMVP is an open-source "film studio in a box" that decomposes video production into specialist modules—Producers, Writers, Directors, Editors—that can run in the cloud, entirely on your Mac, or any combination you choose.

---

## What Can XMVP Do?

- **Generate full video sequences** from a single text prompt
- **Create music videos** synced to any audio track
- **Animate frame-by-frame** with consistent style
- **Run 100% locally** on Apple Silicon (M1/M2/M3) for privacy and uncensored creativity
- **Export everything** to the open XMVP XML format for editing, re-rendering, or sharing

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

- **Mac with Apple Silicon** (M1/M2/M3 with 16GB+ RAM recommended)
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
🏠 Local Mode Enabled: Switching models to Gemma (Text) and LTX (Video).
   📍 Local Text Model Path: /Volumes/XMVPX/mw/gemma-root
   🛡️ Safety Filters: OFF (Uncensored)
   ✨ Quality Refinement: ON (Hyper-Detailed Fattening)
```

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
<XMVP version="2.4">
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
| Video Generation | Veo 3.1 | LTX-Video |
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
| `L` | veo-3.1-fast-generate-preview | Light/fast |
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

---

## Full API Reference

For complete documentation of every argument and option, see:
**[docs/API2.4.md](docs/API2.4.md)**

---

## Contributing

XMVP is a personal project that I'm sharing because I think the "modular vision pipeline" concept is useful. Issues and PRs welcome, but no promises on response time.

---

## License

Free and open for use by all. You'll need your own API keys for cloud mode, or your own hardware for local mode.

---

*"A reasoning, bureaucratic chain of simulated movie and video production specialists."*