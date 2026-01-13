# XMVP
> A reasoning, bureaucratic chain of simulated movie and video production specialists; your own creative technostructure. Free and open for use by all (you'll need your own API keys) in the interest of creating more coherent, universally interesting, and entertainment-forward audiovisual outputs via the models you trust, in the formats you want, from nothing or anything.

**Status**: Beta
**Date**: Jan 13, 2026

## Setup & Configuration
1.  **Install Requirements**: `pip install -r requirements.txt`
2.  **API Keys**: 
    - The pipeline shares a centralized configuration with the legacy suite.
    - Copy `env_vars.example.yaml` to `../../env_vars.yaml` (root of `tools/fmv/`).
    - Add your Gemini/Veo API keys to `ACTION_KEYS_LIST` in `tools/fmv/env_vars.yaml`.
    - **Note**: `env_vars.yaml` is gitignored to protect your secrets.

The Modular Vision Pipeline (MVP) breaks video generation into a 7-stage Value Chain, orchestrated by `movie_producer.py`. Each module isolates a specific creative or technical decision, passing standardized Data Contracts (JSON) to the next stage.

## 1. Orchestration
### `movie_producer.py`
The Master Controller.
- **Role**: Sequential execution, argument parsing, error handling, and artifact persistence (XMVP).
- **Inputs**: CLI Arguments (`--concept`, `--seg`, `--vpform`, `--cf`, `--vm`, `--xb`).
- **Outputs**:
    - `bible.json`
    - `story.json`
    - `portions.json`
    - `manifest.json`
    - `finalcuts/*.mp4`
    - `finalcuts/run_*.xml` (XMVP Archive)

## 2. Creative Modules (The "Above the Line" Team)

### `vision_producer.py` (The Showrunner)
- **Role**: Defines the static "Bible" for the production.
- **Features**:
    - **Chaos Seeds (`--cs`)**: Injects Wikipedia entropy into the core concept.
    - **Cameo (`--cf`)**: Injects specific topics/people via Wikipedia lookup logic.
- **Output**: `bible.json` (CSSV: Constraints, Scenario, Situation, Vision).

### `stub_reification.py` (The Writer)
- **Role**: Synthesizes a concrete narrative arc from the high-level Bible.
- **Logic**: Uses Gemini to invent Characters, Theme, and Synopsis.
- **Output**: `story.json`.

### `writers_room.py` (The Screenwriter)
- **Role**: Breaks the Story into sequential Scenes (Portions).
- **Output**: `portions.json` (List of narrative blocks with duration).

## 3. Technical Modules (The "Below the Line" Team)

### `portion_control.py` (The Line Producer)
- **Role**: Converts narrative Portions into executable Specs (Segs).
- **Logic**: constant frame rate calculation, prompt formatting.
- **Output**: `manifest.json` (List of `Seg` objects).

### `dispatch_director.py` (The Director)
- **Role**: Generates actual assets (Video/Image) from the Manifest.
- **Features**:
    - **VideoDirectorAdapter**: Wraps `action.py` (Veo).
    - **Retry Logic**: Handles 'Model Overloaded' (429/503) with exponential backoff.
    - **Context Passing**: Feeds the last frame of Seg N to Seg N+1.
- **Output**: `componentparts/*.mp4`, `manifest_updated.json`.

## 4. Safety & Sanitation

### `sanitizer.py`
- **Role**: Ensures generated content meets safety guidelines without blocking the pipeline.
- **Method**:
    - **Softening**: Rewrites prompts to change "children" to "adults", "celebrities" to "impersonators".
    - **Washing**: Re-generates unsafe reference images using "dazzle camouflage" techniques (describe -> re-render).
- **Integration**: Called automatically by `action.py` on 400 Policy Errors.

## 5. Persistence

### `mvp_shared.py` (The Library)
- **Role**: Shared Pydantic models (`CSSV`, `Seg`, etc.) and XMVP Utils.
- **XMVP**: `<Bible>`, `<Story>`, `<Portions>`, `<Manifest>` wrapped in `<XMVP>` XML tags.
- **Re-hydration**: Logic to load `bible.json` from `run.xml` via `--xb`.

## 7. Specialized Producers (v1.1)

### `cartoon_producer.py`
The Animation Studio.
- **Role**: Generates Frame-by-Frame (FBF) animation sequences and XMVP Storyboards.
- **Modes**:
    - **Flipbook (FBF)** (`--vpform fbf-cartoon`): Generates "Keyframes" from audio transcripts, then "Expands" them into smooth animation sequences using frame interpolation logic.
    - **Storyboard Export**: Automatically hallucinates a "Movie Bible" (Title/Synopsis/Vibe) from the visual frames and exports a valid XMVP XML file. This allows you to generate a cartoon, and then immediately "remake" it as a live-action movie using `movie_producer.py --xb`.
- **Key Flags**:
    - `--fps N`: Controls **Expansion Factor** in FBF mode (e.g., `--fps 4` expands 1 row to 4 frames).
    - `--vpform`: `fbf-cartoon` (default).

### `post_production.py` (Post)
The VFX Suite.
- **Role**: Enhances video output via hallucinated tweening and detail injection.
- **Pipeline**: Extract Frames -> Tween (Gemini) -> Upscale (Obsessive/Lancaster) -> Stitch.
- **Key Flags**:
    - `-x N`: Tweening factor (e.g., `-x 2` doubles framerate).
    - `--scale N`: Upscaling factor (e.g., `--scale 2` doubles resolution).

### `model_scout.py`
The Casting Director for Models.
- **Role**: Audits available Google Cloud models (Gemini/Veo/Imagen).
- **Features**:
    - `--probe`: Stress-tests model endpoints to determine empirical rate limits and quota costs.

## 8. CLI Reference (Movie Producer)

| Flag | Description | Tier / Model |
| :--- | :--- | :--- |
| `--concept "..."` | The high-level prompt | N/A |
| `--seg N` | Number of segments (shots) | N/A |
| `--l N.N` | Length per segment (seconds) | N/A |
| `--vpform FORM` | Genre/Structure (`realize-ad`, `tech-movie`, `fairy-tale-movie`) | N/A |
| `--vm TIER` | Video Model Tier (`K`=Veo 3, `J`=Veo 3 Fast, `V2`=Veo 2) | K, J, V2 |
| `--fast` | Shortcut for `--vm J` (Cheaper/Faster) | J |
| `--vfast` | Shortcut for `--vm V2` (Legacy Veo 2.0 - No Audio) | V2 |
| `--xb PATH` | **Re-hydrate** from XMVP XML file (Bypass creation) | N/A |

## 9. Deprecated / Merged
- **`realize.py`**: **DEPRECATED**. Replaced by `movie_producer.py`.
- **`intake_outgive`**: **MERGED**. Functionality subsumed by `dispatch_director.py` (Context Passing).
- **`editor_decider`**: **INLINE**. Simple stitching logic currently lives in `movie_producer.py` (Step 6). Future expansion planned for `Indecision` resolution.
