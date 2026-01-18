# XMVP API Reference v2.5
## Complete Command-Line Interface Documentation

---

# Table of Contents

1. [Creative Engines (Start Process Modules)](#creative-engines)
   - [movie_producer.py](#movie_producerpy)
   - [cartoon_producer.py](#cartoon_producerpy)
   - [content_producer.py](#content_producerpy)
   - [post_production.py](#post_productionpy)
2. [Pipeline Modules (Internal)](#pipeline-modules)
   - [vision_producer.py](#vision_producerpy)
   - [stub_reification.py](#stub_reificationpy)
   - [writers_room.py](#writers_roompy)
   - [portion_control.py](#portion_controlpy)
   - [dispatch_director.py](#dispatch_directorpy)
3. [Audio & Speech Modules](#audio--speech-modules)
   - [foley_talk.py](#foley_talkpy)
4. [Utility & Management Modules](#utility--management-modules)
   - [model_scout.py](#model_scoutpy)
   - [populate_models_xmvp.py](#populate_models_xmvppy)
5. [Bridge Modules (Local Inference)](#bridge-modules)
   - [flux_bridge.py](#flux_bridgepy)
   - [ltx_bridge.py](#ltx_bridgepy)
   - [kokoro_bridge.py](#kokoro_bridgepy)
   - [hunyuan_foley_bridge.py](#hunyuan_foley_bridgepy)
6. [Core Libraries](#core-libraries)
   - [text_engine.py](#text_enginepy)
   - [truth_safety.py](#truth_safetypy)
   - [definitions.py](#definitionspy)
   - [mvp_shared.py](#mvp_sharedpy)
7. [Data Models & Schemas](#data-models--schemas)

---

# Creative Engines

## movie_producer.py

**The Showrunner** - Orchestrates the entire 7-stage pipeline to create structured video content from a single prompt.

### Usage
```bash
python3 movie_producer.py "Your concept here" [OPTIONS]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `concept` | positional (optional) | - | The concept text (quoted string). Required unless `--xb` is provided. |

### Options

#### Producer Options
| Option | Type | Default | Values/Range | Description |
|--------|------|---------|--------------|-------------|
| `--seg` | int | `3` | 1-∞ | Number of video segments to generate |
| `--l` | float | `8.0` | 1.0-60.0 | Length of each segment in seconds |
| `--vpform` | str | `"tech-movie"` | `realize-ad`, `tech-movie`, `movies-movie`, `studio-movie`, `parody-movie`, `music-video` | Vision Platonic Form (genre template) |
| `--cs` | int | `0` | 0, 2, 3, 4, 5, 6 | Chaos Seeds level (Wikipedia concept injection) |
| `--cf` | str | `None` | Wikipedia URL or search query | Cameo Feature: Inject specific concept |
| `--mu` | str | `None` | Path to audio file | Music track for music-video mode |
| `--vm` | str | `"K"` | `L`, `J`, `K`, `V2`, `D` | Video Model Tier |
| `--pg` | flag | `False` | - | Enable PG Mode (relaxed celebrity, strict child safety) |

#### Operations Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--clean` | flag | `False` | Clean intermediate JSON files before running |
| `--xb` | str | `None` | XMVP re-hydration path (bypass Vision Producer, load from XML) |
| `-f`, `--fast` | flag | `False` | Use faster/cheaper model tier (overrides `--vm` to J) |
| `--vfast` | flag | `False` | Use legacy Veo 2.0 (fastest, overrides `--vm` to V2) |
| `--out` | str | `None` | Override output directory |
| `--local` | flag | `False` | Run 100% locally (Gemma + LTX-Video) |

### Video Model Tiers

| Tier | Model | Description |
|------|-------|-------------|
| `L` | veo-3.1-fast-generate-preview | Light (fast, lower quality) |
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

# Re-render from existing XMVP manifest
python3 movie_producer.py --xb previous_run.xml --vm J
```

---

## cartoon_producer.py

**The Animator** - Specialized pipeline for frame-by-frame animation, music video syncing, and creative agency work.

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
| `--tf` | Path | `/Users/0gs/METMcloud/METMroot/tools/fmv/fbf_data` | Transcript folder (source) |
| `--vf` | Path | `/Volumes/XMVPX/fmv_corpus` | Video folder (corpus) |
| `--xb` | str | `None` | Path to XMVP XML manifest for re-rendering |
| `--mu` | str | `None` | Path to music/audio file for sync |
| `--project` | str | `None` | Specific project name to process |

#### Processing Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--delay` | float | `5.0` | Delay between API requests in seconds |
| `--limit` | int | `0` | Limit number of frames per project (0 = unlimited) |
| `--smin` | float | `0.0` | Minimum duration filter in seconds |
| `--smax` | float | `None` | Maximum duration filter in seconds |
| `--shuffle` | flag | `False` | Shuffle projects before processing |
| `--cs` | int | `0` | 0-3 | Chaos Seeds level |
| `--bpm` | float | `None` | Manual BPM override (bypasses detection) |
| `--pg` | flag | `False` | Enable PG Mode |

#### Advanced Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vspeed` | float | `8.0` | Visualizer speed (FPS) for music-agency. Supports 2, 4, 8, 16 |
| `--fc` | flag | `False` | Enable Frame & Canvas (Code Painter) mode |
| `--kid` | int | `512` | Keyframe init dimension (higher = better composition) |
| `--local` | flag | `False` | Local mode (Gemma + Flux) |
| `--w` | int | `None` | Override width (local only) |
| `--h` | int | `None` | Override height (local only) |

### VP Forms Explained

| Form | Description |
|------|-------------|
| `creative-agency` | Prompt → Story → Animation (default) |
| `fbf-cartoon` | Legacy frame-by-frame animation |
| `music-visualizer` | Audio-reactive abstract animation |
| `music-agency` | Audio-reactive narrative story |

### Examples
```bash
# Creative agency mode
python3 cartoon_producer.py --prompt "A sad robot finds purpose" --style "Pixel Art"

# Music video synced to track
python3 cartoon_producer.py --vpform music-agency --mu song.mp3 --prompt "Neon dreams"

# Abstract visualizer
python3 cartoon_producer.py --vpform music-visualizer --mu ambient.wav

# Local mode with custom dimensions
python3 cartoon_producer.py --prompt "Space opera" --local --w 1920 --h 1080
```

---

## content_producer.py

**Unified Generator** - Merges podcast animation and improv comedy generation.

### Usage
```bash
python3 content_producer.py [OPTIONS]
```

### Options

| Option | Type | Default | Values | Description |
|--------|------|---------|--------|-------------|
| `--vpform` | str | `None` | `24-podcast`, `24-cartoon`, `10-podcast`, `gahd-podcast` | Vision Platonic Form |
| `--project` | str | `None` | - | Project override (stub) |
| `--ep` | int | `None` | - | Episode number (stub) |
| `--local` | flag | `False` | - | Use local engines (Flux + Kokoro) |
| `--foley` | str | `"off"` | `on`, `off` | Enable generative foley audio |
| `--slength` | float | `0.0` | - | Override duration in seconds |
| `--fc` | flag | `False` | - | Code Painter mode (experimental) |
| `--geminiapi` | flag | `False` | - | Force cloud Gemini API for text (disables local Gemma default) |

### VP Forms

| Form | Description |
|------|-------------|
| `24-podcast` / `24-cartoon` | 24-minute 4-person improv comedy special |
| `10-podcast` | 10-minute version with duration override |
| `gahd-podcast` | Traditional pair/triplet processing mode |

---

## post_production.py

**The Editor / Svelte 2x Machine** - Handles upscaling, frame interpolation, and audio stitching.

### Usage
```bash
python3 post_production.py INPUT [OPTIONS]
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | positional (optional) | Input video file or directory of frames |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | str | `None` | Output directory |
| `-x` | int | `2` | Frame expansion factor (tweening). 1 = no interpolation |
| `--scale` | float | `2.0` | Upscale factor |
| `--restyle` | str | `None` | Restyle mode. Values: `ascii` |
| `--local` | flag | `False` | Run locally (Flux Img2Img) |
| `--more` | flag | `False` | Enable secondary interpolation/upscale (4x total) |
| `--mu` | str | `None` | Audio file for music video sync |
| `--stitch-audio` | flag | `False` | Force-stitch frames to match audio duration (ignore delta threshold) |

### Processing Modes

| Mode | Description |
|------|-------------|
| Default | Cloud-based Gemini upscaling |
| `--local` | Flux-based Img2Img processing |
| `--more` | 4x frames AND 4x resolution |
| `--restyle ascii` | ASCII art overlay at 33% opacity |

### Audio Sync Logic
When `--mu` is provided:
1. Measures audio duration
2. Counts video frames
3. If delta ≤ 10s (or `--stitch-audio`): auto-adjusts FPS to match
4. If delta > 10s: keeps original FPS with warning

### Examples
```bash
# Basic 2x upscale
python3 post_production.py input.mp4 --scale 2.0

# Local Flux processing with interpolation
python3 post_production.py input.mp4 --local --scale 2.0 -x 2

# Music video sync
python3 post_production.py input.mp4 --mu soundtrack.mp3 --local --stitch-audio

# Maximum quality (4x everything)
python3 post_production.py input.mp4 --local --more
```

---

# Pipeline Modules

## vision_producer.py

**The Visionary** - Creates the "Bible" (CSSV) from a prompt.

### Usage
```bash
python3 vision_producer.py --vpform FORM --prompt "CONCEPT" [OPTIONS]
```

### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--vpform` | str | - | **Yes** | VPForm to use |
| `--prompt` | str | - | **Yes** | Core concept (the "Situation") |
| `--slength` | float | `60.0` | No | Total duration in seconds |
| `--flength` | int | `0` | No | Total duration in frames (overrides slength) |
| `--seg_len` | float | `8.0` | No | Target segment length in seconds |
| `--cs` | int | `0` | No | Chaos seeds (0, 2, 3, 4, 5, 6) |
| `--out` | str | `"bible.json"` | No | Output path for CSSV Bible |

### Available VPForms

| Form | FPS | Description |
|------|-----|-------------|
| `realize-ad` | 24 | Commercial advertisement |
| `podcast-interview` | 1 | Two-person interview (audio focused) |
| `movies-movie` | 24 | Hollywood blockbuster remake (1979-2001 era) |
| `studio-movie` | 24 | Behind-the-scenes mockumentary |
| `parody-movie` | 24 | Direct parody/spoof |
| `music-video` | 24 | Music video synced to audio |

---

## stub_reification.py

**The Writer** - Expands the Bible into a full Story (narrative arc).

### Usage
```bash
python3 stub_reification.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bible` | str | `"bible.json"` | Path to input CSSV JSON |
| `--out` | str | `"story.json"` | Output path for Story JSON |
| `--req` | str | `None` | Optional extra request/notes to modify the bible |
| `--xb` | str | `None` | Path to XMVP XML file (overrides --bible) |

---

## writers_room.py

**The Screenwriter** - Breaks Story into temporal Portions (scenes).

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
| `--xb` | str | `None` | Path to XMVP XML file (overrides inputs) |

### Scene Duration Rules
- Minimum: 4.0 seconds
- Maximum: 20.0 seconds
- Variable pacing encouraged based on narrative needs

---

## portion_control.py

**The Line Producer** - Calculates exact frame ranges for each Portion.

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
| `--xb` | str | `None` | Path to XMVP XML file (overrides inputs) |

---

## dispatch_director.py

**The Director** - Executes the Manifest to generate visual assets.

### Usage
```bash
python3 dispatch_director.py --manifest PATH [OPTIONS]
```

### Options

| Option | Type | Default | Values | Description |
|--------|------|---------|--------|-------------|
| `--manifest` | str | - | **Required** | Path to input Manifest JSON |
| `--out` | str | `"manifest_updated.json"` | - | Output path for updated Manifest |
| `--staging` | str | `"componentparts"` | - | Directory to save generated assets |
| `--mode` | str | `"image"` | `image`, `video` | Generation mode |
| `--vm` | str | `"J"` | L, J, K | Video model tier (if mode=video) |
| `--pg` | flag | `False` | - | Enable PG mode |
| `--width` | int | `768` | - | Output width (image mode) |
| `--height` | int | `768` | - | Output height (image mode) |
| `--local` | flag | `False` | - | Force local mode (LTX for video, Flux for image) |

---

# Audio & Speech Modules

## foley_talk.py

**Unified Audio Engine** - Handles dialogue generation and audio processing.

### Usage
```bash
python3 foley_talk.py --input VIDEO [OPTIONS]
```

### Options

| Option | Type | Default | Values | Description |
|--------|------|---------|--------|-------------|
| `--input` | str | - | **Required** | Input silent video path |
| `--xb` | str | `None` | - | Input XMVP manifest (source of dialogue) |
| `--out` | str | `"final_mix.mp4"` | - | Output video path |
| `--mode` | str | `"cloud"` | `cloud`, `comfy`, `rvc`, `kokoro` | Audio backend |
| `--dry-run` | flag | `False` | - | Simulate execution without generating |

### Audio Backends

| Backend | Description |
|---------|-------------|
| `cloud` | Google Cloud TTS (Journey voices) |
| `comfy` | ComfyUI local workflow (IndexTTS) |
| `rvc` | Legacy RVC voice conversion |
| `kokoro` | Local Kokoro ONNX TTS |

---

# Utility & Management Modules

## model_scout.py

**Registry Manager** - Scans available models and manages the active profile.

### Usage
```bash
python3 model_scout.py [OPTIONS]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--status` | flag | Show current active models |
| `--list` | str (optional) | List models for modality (or ALL). Values: `text`, `image`, `video`, `foley`, `spoken_tts`, `cloned_tts`, `ALL` |
| `--switch` | 2 args | Switch active model. Format: `MODALITY MODEL_ID` |
| `--scan` | flag | Scan cloud for available models |
| `--probe` | str | Probe a specific model for rate limits |
| `--pull` | str | Download HuggingFace model via MLX |

### Examples
```bash
# Show current configuration
python3 model_scout.py --status

# List all text models
python3 model_scout.py --list text

# Switch to local Flux for images
python3 model_scout.py --switch image flux-schnell

# Probe Veo model
python3 model_scout.py --probe veo-3.1-generate-preview

# Download and convert model
python3 model_scout.py --pull google/gemma-2-9b-it
```

---

## populate_models_xmvp.py

**Model Downloader** - Downloads all required local models to `/Volumes/XMVPX/mw`.

### Usage
```bash
python3 populate_models_xmvp.py
```

### No command-line arguments - runs interactively with optional HuggingFace token prompt.

### Models Downloaded

| Model | Repository | Target Directory |
|-------|------------|------------------|
| LTX-Video | Lightricks/LTX-Video | `/Volumes/XMVPX/mw/LT2X-root` |
| Flux Schnell | black-forest-labs/FLUX.1-schnell | `/Volumes/XMVPX/mw/flux-root` |
| IndexTTS | IndexTeam/IndexTTS-2 | `/Volumes/XMVPX/mw/indextts-root` |
| Hunyuan-Foley | tencent/HunyuanVideo-Foley | `/Volumes/XMVPX/mw/hunyuan-foley` |
| RVC Base | lj1995/VoiceConversionWebUI | `/Volumes/XMVPX/mw/rvc-root` |
| Gemma 3 | google/gemma-3-27b-it | `/Volumes/XMVPX/mw/gemma-root` |
| T5 Weights | city96/t5-v1_1-xxl-encoder-bf16 | `/Volumes/XMVPX/mw/t5weights-root` |
| Kokoro TTS | Kijai/Kokoro-82M-ONNX | `/Volumes/XMVPX/mw/kokoro-root` |
| ComfyUI | github.com/comfyanonymous/ComfyUI | `/Volumes/XMVPX/mw/comfyui-root` |

---

# Bridge Modules

These modules provide interfaces to local inference engines. They are typically not called directly but are used internally by other modules.

## flux_bridge.py

**Flux Interface** - Text-to-Image and Image-to-Image generation.

### Class: `FluxBridge`

```python
bridge = FluxBridge(model_path, device="mps")
```

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate()` | `prompt`, `width=1024`, `height=1024`, `steps=4`, `seed=None` | PIL Image | Text-to-image generation |
| `generate_img2img()` | `prompt`, `image`, `strength=0.6`, `width=1024`, `height=1024`, `steps=4`, `seed=None` | PIL Image | Image-to-image transformation |

### Singleton Helper
```python
from flux_bridge import get_flux_bridge
bridge = get_flux_bridge("/Volumes/XMVPX/mw/flux-root")
```

---

## ltx_bridge.py

**LTX-Video Interface** - Text-to-Video and Image-to-Video generation.

### Class: `LTXBridge`

```python
bridge = LTXBridge(model_path, device="mps")
```

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate()` | `prompt`, `output_path`, `width=768`, `height=512`, `num_frames=121`, `fps=24`, `seed=None`, `image_path=None` | bool | Video generation. If `image_path` provided, uses Img2Vid |

---

## kokoro_bridge.py

**Kokoro TTS Interface** - Local text-to-speech.

### Class: `KokoroBridge`

```python
bridge = KokoroBridge(model_path, voices_path=None)
```

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `load()` | - | - | Load model into memory |
| `generate()` | `text`, `output_path`, `voice_name="af_bella"`, `speed=1.0` | bool | Generate speech |
| `get_voice_list()` | - | List[str] | Get available voice names |

### Available Voices (Default)
- `af_bella`, `af_sarah` (American Female)
- `am_michael`, `am_adam` (American Male)
- `bf_emma` (British Female)
- `bm_george` (British Male)

---

## hunyuan_foley_bridge.py

**Hunyuan Foley Interface** - Video-to-Audio foley generation.

### Class: `HunyuanFoleyBridge`

```python
bridge = HunyuanFoleyBridge(model_path="/Volumes/XMVPX/mw/hunyuan-foley", device="auto")
```

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_foley()` | `text_prompt`, `video_path`, `output_path`, `duration=None`, `guidance_scale=4.5`, `steps=30` | bool | Generate foley audio for video |

### Helper Function
```python
from hunyuan_foley_bridge import generate_foley_asset
result = generate_foley_asset(prompt, output_path, video_path=None, duration=4.0)
```

---

# Core Libraries

## text_engine.py

**Central Text Generation** - Unified interface for cloud and local LLM inference.

### Class: `TextEngine`

```python
engine = TextEngine(config_path=None)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `backend` | str | Current backend: `"gemini_api"` or `"local_gemma"` |
| `local_model_path` | str | Path to local model |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate()` | `prompt`, `temperature=0.7`, `json_schema=None` | str | Generate text response |
| `get_gemini_client()` | - | genai.Client | Get configured Gemini client |
| `get_model_instance()` | - | GenerativeModel | Get V1 SDK model object |

### Singleton Helper
```python
from text_engine import get_engine
engine = get_engine()
```

### Environment Override
Set `TEXT_ENGINE=local_gemma` to force local mode.

---

## truth_safety.py

**Prompt Refinement Engine** - Ensures coherence, safety, and quality.

### Class: `TruthSafety`

```python
ts = TruthSafety(api_key=None)
```

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `refine_prompt()` | `prompt`, `context_dict=None`, `pg_mode=False`, `local_mode=False` | str | Full refinement pipeline |
| `soften_prompt()` | `prompt`, `pg_mode=False` | str | Legacy wrapper for refine_prompt |
| `describe_image()` | `image_path` | str | Get dense description of image |
| `wash_image()` | `image_path` | str | Sanitize image (describe → regenerate) |
| `critique_dialogue()` | `draft_line`, `character="Unknown"`, `context=None` | str | Refine dialogue via DialogueCritic |

### Refinement Phases
1. **TRUTH**: Physical coherence, logic, style consistency
2. **CONTEXT ALIGNMENT**: Weave in style/character context
3. **FATTENING** (local_mode only): Expand to 100-150 words of cinematic detail
4. **SAFETY**: Apply PG or Standard safety guidelines (skipped if local_mode AND NOT pg_mode)

---

## definitions.py

**Model Registry** - Centralized configuration for all models.

### Enums

```python
class BackendType(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    FOLEY = "foley"
    SPOKEN_TTS = "spoken_tts"
    CLONED_TTS = "cloned_tts"
```

### Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `get_video_model()` | `key` | str | Legacy accessor for VIDEO_MODELS |
| `get_active_model()` | `modality: Modality` | ModelConfig | Get config for active model |
| `set_active_model()` | `modality: Modality`, `model_id: str` | - | Set and persist active model |

### Registered Models

#### Text
| ID | Backend | Path/Endpoint |
|----|---------|---------------|
| `gemini-2.0-flash` | cloud | - |
| `gemini-1.5-pro` | cloud | - |
| `gemma-2-9b-it` | local | `/Volumes/XMVPX/mw/gemma-root` |

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
| `CSSV` | The "Bible" - Constraints, Scenario, Situation, Vision |
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
| `load_text_keys(env_path)` | Load TEXT_KEYS_LIST fallback |
| `save_xmvp(data_models, path)` | Save to XMVP XML format |
| `load_xmvp(path, key)` | Load specific key from XMVP XML |
| `get_client()` | Get rotated genai.Client |

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

## Story Structure

```json
{
  "title": "Title Here",
  "synopsis": "Synopsis text...",
  "characters": ["Character 1", "Character 2"],
  "theme": "Theme description"
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
<XMVP version="2.4">
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
ACTION_KEYS_LIST: "key1,key2,key3"  # High-cost operations (Veo)
TEXT_KEYS_LIST: "key4,key5"         # High-volume text operations
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

*Generated for XMVP v2.4 - January 2026*