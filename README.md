# XMVP: The Modular Vision Pipeline

This folder contains the complete "Value Chain" for AI content production, decomposed into specialized specialist modules.

## 🎬 Core Orchestrators

### `movie_producer.py`
The "Showrunner" that orchestrates the entire 7-stage pipeline to create video content from a simple prompt.

**Usage:**
```bash
python3 movie_producer.py "A sci-fi film about a robot learning to love" [ARGS]
```

**Arguments:**
- `concept` (Positional): The core prompt/logline.
- `--seg [INT]`: Number of segments to generate (default: `3`).
- `--l [FLOAT]`: Length of each segment in seconds (default: `4.0`).
- `--vpform [STR]`: The Genre/Form to use. Options: `realize-ad`, `tech-movie` (default: `tech-movie`).
- `--cs [0-6]`: Chaos Seeds level (Entropy injection). `0`=Off.
- `--cf [URL/Query]`: Cameo Feature. Injects a specific Wikipedia topic or search result as a "Minor Appearance".
- `--vm [L/J/K]`: Video Model Tier. `J`=Veo 2, `K`=Veo 3 (default: `K`).
- `--pg`: **PG Mode**. Enables "Actor N.C." obfuscation for celebrities and strict child safety cleaning.
- `--clean`: Deletes intermediate JSON artifacts before running.
- `--xb [PATH]`: Re-hydrates the pipeline from an existing XMVP XML file (skips Vision Producer).
- `-f`, `--fast`: Shortcut for Tier `J` (Legacy Veo 2).
- `--vfast`: Shortcut for Tier `V2` (Legacy).
- `--out [PATH]`: Override output directory.

---

### `cartoon_producer.py`
Specialized pipeline for "Frame-By-Frame" (FBF) animation and legacy interpolation.

**Usage:**
```bash
python3 cartoon_producer.py --vpform fbf-cartoon --tf /path/to/transcripts --vf /path/to/videos
```

**Arguments:**
- `--vpform [STR]`: Mode. `fbf-cartoon` (Frame-by-Frame) or `legacy` (Interpolation). Default: `fbf-cartoon`.
- `--tf [PATH]`: Transcript Folder containing sub-project folders.
- `--vf [PATH]`: Video Folder containing original source videos (for audio muxing).
- `--fps [INT]`:
    - In **FBF Mode**: Expansion Factor (1 = 1 frame per line, 2 = 2 frames per line).
    - In **Legacy Mode**: Output FPS.
- `--delay [FLOAT]`: Seconds to wait between API requests to avoid rate limits (default: `3.0`).
- `--limit [INT]`: Test limit (stop after N frames). `0`=No limit.
- `--project [STR]`: specific project name to process (skips others).
- `--smin [FLOAT]`: Minimum project duration to process.
- `--smax [FLOAT]`: Maximum project duration to process.
- `--shuffle`: Randomize project processing order.

---

### `podcast_animator.py`
Visualizes audio podcasts (or generates them) into animated video pairs/triplets.

**Usage:**
```bash
python3 podcast_animator.py --project [ID]
```

**Arguments:**
- `--project [ID]`: Google Cloud Project ID to use for TTS (Text-to-Speech) billing. Prevents `403 PERMISSION_DENIED` if the default project lacks TTS enablement.

---

## 🛠️ utility Tools

### `model_scout.py`
Scans available Gemini/Veo models and checks if `definitions.py` is up to date.

**Usage:**
```bash
python3 model_scout.py [--probe MODEL_NAME]
```

**Arguments:**
- `--probe [MODEL_NAME]`: Runs a stress test (5 attempts) against a specific model to check for rate limits/latency.

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

---

## 📁 Environment
Ensure `tools/fmv/env_vars.yaml` is populated with your keys:
```yaml
GEMINI_API_KEY: "..."
ACTION_KEYS_LIST: "key1,key2,key3" # For rotation
```
