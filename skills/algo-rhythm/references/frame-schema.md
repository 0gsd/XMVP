# Frame Schema Reference

The complete JSON schema for algo-rhythmic-narrative output. All frame data follows this structure.

## Master Envelope

The top-level output file (or the index file if chunked):

```json
{
  "algo_rhythmic_narrative": "1.0",
  "meta": { ... },
  "sections": [ ... ],
  "palettes": { ... },
  "scenes": [ ... ],
  "frames": [ ... ]
}
```

### `meta` Object

```json
{
  "meta": {
    "bpm": 120,
    "time_signature": [4, 4],
    "fps": 4,
    "duration_sec": 210.5,
    "total_frames": 842,
    "grid_w": 64,
    "grid_h": 64,
    "color_mode": "palette-index",
    "encoding": "delta",
    "source_audio": "my_track.mp3",
    "source_audio_hash": "sha256:abc123...",
    "generated_at": "2026-02-24T12:00:00Z",

    "computed": {
      "frames_per_beat": 2.0,
      "frames_per_bar": 8.0,
      "total_beats": 421,
      "total_bars": 105.25
    }
  }
}
```

**Fields**:
- `bpm`: Beats per minute (float). Detected from audio or specified manually.
- `time_signature`: `[numerator, denominator]`. Default `[4, 4]`.
- `fps`: Frames per second of the output sequence. Must match rendering pipeline.
- `duration_sec`: Total duration in seconds.
- `total_frames`: `ceil(duration_sec × fps)`. This is the authoritative frame count.
- `grid_w`, `grid_h`: Cell grid dimensions.
- `color_mode`: One of `"hex"`, `"palette-index"`, `"density"`.
  - `"hex"`: Each cell is a 7-char string like `"#FF00AA"`.
  - `"palette-index"`: Each cell is an integer (0-based) into the active scene's palette.
  - `"density"`: Each cell is a float 0.0–1.0 (for character-density renderers like Unicode visualizer).
- `encoding`: `"full"` (every frame has complete grid) or `"delta"` (keyframes + diffs).
- `source_audio`: Filename of the input audio (informational).
- `source_audio_hash`: SHA-256 of the audio file for provenance.
- `computed`: Derived values for convenience. Renderers can recompute these but shouldn't have to.

### `sections` Array

Corresponds to the musical structure of the song.

```json
{
  "sections": [
    {
      "id": "intro",
      "name": "Intro",
      "start_frame": 0,
      "end_frame": 60,
      "start_time": 0.0,
      "end_time": 15.0,
      "energy_profile": "building",
      "energy_avg": 0.25,
      "energy_peak": 0.45
    },
    {
      "id": "verse-1",
      "name": "Verse 1",
      "start_frame": 60,
      "end_frame": 248,
      ...
    }
  ]
}
```

**Fields**:
- `id`: Machine-readable slug. Used as key in scene mappings.
- `name`: Human-readable label.
- `start_frame`, `end_frame`: Inclusive start, exclusive end. `section[n].end_frame == section[n+1].start_frame`.
- `start_time`, `end_time`: Seconds (for display/debugging; frames are authoritative).
- `energy_profile`: One of `"quiet"`, `"building"`, `"mid"`, `"high"`, `"climax"`, `"falling"`, `"breakdown"`.
- `energy_avg`, `energy_peak`: Normalized 0.0–1.0 from RMS analysis.

### `palettes` Object

Named color palettes referenced by scenes.

```json
{
  "palettes": {
    "cold-blue": {
      "colors": ["#0A0A2E", "#1B1B4B", "#2D2D8A", "#4444CC", "#6666FF", "#99AAFF", "#CCDDFF", "#FFFFFF"],
      "bg": "#0A0A2E",
      "fg": "#6666FF",
      "accent1": "#99AAFF",
      "accent2": "#CCDDFF",
      "flash": "#FFFFFF"
    },
    "neon-city": {
      "colors": ["#000000", "#1A001A", "#FF00FF", "#00FFFF", "#FF6600", "#FFFF00", "#FFFFFF", "#330033"],
      "bg": "#000000",
      "fg": "#FF00FF",
      "accent1": "#00FFFF",
      "accent2": "#FF6600",
      "flash": "#FFFFFF"
    }
  }
}
```

**Rules**:
- Every palette must have exactly `bg`, `fg`, `accent1`, `accent2`, and `flash` named entries.
- The `colors` array is the full ordered palette (index 0 = bg, index 1 = darkest non-bg, ... index N = brightest/flash).
- Palette-index mode references indices into `colors`.
- Palette size: 4–12 colors. 8 is the sweet spot.

### `scenes` Array

Maps narrative scenes to sections, palettes, and visual rules.

```json
{
  "scenes": [
    {
      "id": "hands-typing",
      "section_ids": ["verse-1", "verse-2"],
      "palette_id": "cold-blue",
      "composition": {
        "template": "center-subject",
        "params": {
          "subject_rect": [0.25, 0.3, 0.75, 0.8],
          "subject_palette_indices": [3, 4, 5],
          "bg_palette_indices": [0, 1]
        }
      },
      "beat_rules": [
        {
          "trigger": "downbeat",
          "effect": "brightness-pulse",
          "intensity": 0.3,
          "decay_frames": 3
        },
        {
          "trigger": "every-bar",
          "effect": "palette-rotate",
          "shift": 1
        }
      ],
      "energy_rules": {
        "brightness_floor": 0.2,
        "brightness_ceiling": 1.0,
        "saturation_tracks_energy": true
      },
      "narrative_directive": "Close-up of hands on keyboard. Blue-purple tones. Slight pulse on each beat. Keys occasionally flash white."
    }
  ]
}
```

#### Composition Templates

Built-in templates and their parameters:

| Template | Description | Params |
|----------|-------------|--------|
| `center-subject` | Dominant color block in center, bg elsewhere | `subject_rect` [x1,y1,x2,y2] normalized, `subject_palette_indices`, `bg_palette_indices` |
| `horizon` | Horizontal split | `split_y` (0.0–1.0), `top_palette_indices`, `bottom_palette_indices` |
| `vertical-split` | Vertical split | `split_x`, `left_palette_indices`, `right_palette_indices` |
| `radial` | Concentric rings from center | `center` [x,y], `ring_widths` [float], `ring_palette_indices` [[int]] |
| `diagonal-sweep` | Gradient at an angle | `angle_deg`, `palette_indices` (ordered bright→dark) |
| `grid-scatter` | Regular pattern of accent dots on bg | `spacing` [x,y], `dot_palette_index`, `bg_palette_index` |
| `noise-field` | Perlin-like noise mapped to palette | `scale`, `octaves`, `seed` |
| `bands` | Horizontal or vertical stripes | `direction` ("h"/"v"), `band_palette_indices` [[int]], `band_heights` [float] |
| `vignette` | Darkens toward edges | `inner_palette_indices`, `outer_palette_indices`, `radius` |
| `solid` | Entire grid one color | `palette_index` |
| `custom` | User-defined function | `fn_name` (references a custom function in the rendering pipeline) |

#### Beat Rule Triggers

| Trigger | Description |
|---------|-------------|
| `downbeat` | Beat 1 of every bar |
| `every-beat` | Every beat |
| `snare` | Beats 2 and 4 (in 4/4 time) |
| `every-bar` | First frame of every bar |
| `every-n-bars` | Every N bars (param: `n`) |
| `section-start` | First frame of the scene's section |
| `energy-threshold` | When energy exceeds a threshold (param: `threshold`) |
| `manual` | Specific frame indices (param: `frames` [int]) |

#### Beat Rule Effects

| Effect | Description | Params |
|--------|-------------|--------|
| `brightness-pulse` | All cells brighten then decay | `intensity` (0.0–1.0), `decay_frames` |
| `flash-color` | All cells snap to flash color then decay | `decay_frames` |
| `flash-cells` | Random subset of cells flash | `fraction` (0.0–1.0), `decay_frames` |
| `ripple` | Expanding ring from a point | `origin` [x,y] or `"center"`, `speed` (cells/frame), `width`, `palette_index` |
| `palette-rotate` | Shift all palette indices by N | `shift` |
| `palette-invert` | Swap bg↔fg for one frame | (none) |
| `shake` | Offset entire grid by random [dx,dy] | `magnitude` (cells) |
| `scanline` | Horizontal bright line sweeps down | `speed`, `palette_index` |
| `column-bars` | Vertical bars whose height tracks energy per frequency band | `num_bands`, `palette_indices` |
| `wipe` | Progressive reveal of next scene | `direction` ("left"/"right"/"up"/"down"), `frames` |

### `frames` Array

The main payload. Each element is one frame.

#### Full Encoding (`encoding: "full"`)

```json
{
  "i": 142,
  "t": 35.5,
  "beat": 71,
  "bar": 17,
  "beat_phase": 0.75,
  "section": "chorus-1",
  "scene": "city-night",
  "energy": 0.82,
  "transition": null,
  "directive": "Neon city wide shot. High energy. Flash on this beat.",
  "grid": [
    [0, 0, 0, 1, 1, 2, 2, 1, 0, 0, ...],
    [0, 0, 1, 2, 3, 4, 4, 3, 2, 1, ...],
    ...
  ]
}
```

**Fields**:
- `i`: Frame index (0-based). Authoritative identifier.
- `t`: Time in seconds (informational; `i / fps`).
- `beat`: Beat number (0-based from start of song).
- `bar`: Bar number (0-based).
- `beat_phase`: Position within the current beat, 0.0 = on the beat, 0.5 = halfway to next.
- `section`: Section ID this frame belongs to.
- `scene`: Scene ID this frame belongs to.
- `energy`: Normalized energy at this frame (0.0–1.0). From RMS analysis or estimated from section profile.
- `transition`: `null` if not in transition, or `{ "from": "scene-a", "to": "scene-b", "progress": 0.35 }`.
- `directive`: Human-readable description of what this frame should depict. Used by LLM-based renderers (like frame_canvas.py's text_engine) as a prompt. Concise — one sentence.
- `grid`: 2D array, `grid_h` rows × `grid_w` columns. Values depend on `color_mode`.

#### Delta Encoding (`encoding: "delta"`)

To reduce file size, delta encoding stores only changes from the previous frame.

**Keyframes** (first frame of each scene, or every N frames):
```json
{
  "i": 142,
  "keyframe": true,
  "grid": [[0, 0, 1, ...], ...]
  // ... all other fields same as full encoding
}
```

**Delta frames** (everything else):
```json
{
  "i": 143,
  "keyframe": false,
  "delta": [
    [12, 5, 4],
    [12, 6, 5],
    [30, 20, 7],
    [30, 21, 7]
  ]
  // delta entries: [row, col, new_value]
  // ... other fields same as full
}
```

If `delta` is an empty array `[]`, the frame is identical to the previous frame (common in low-FPS, static scenes). The renderer copies the previous grid.

## XMVP XML Envelope

For integration with the XMVP pipeline, the frame data can be wrapped in an XMVP XML document:

```xml
<?xml version='1.0' encoding='utf-8'?>
<XMVP version="3.03">
  <Bible>{
    "constraints": { "width": 64, "height": 64, "fps": 4, "max_duration_sec": 210.5 },
    "scenario": "Music video for my_track.mp3",
    "vision": "STYLE: Algo-rhythmic cell grid. AESTHETIC: Beat-synced color matrices.",
    "situation": "CONCEPT: Frame-by-frame pixel grid narrative score."
  }</Bible>
  <Story>{
    "title": "Narrative Score for my_track",
    "synopsis": "...",
    "characters": [],
    "theme": "Rhythmic Visual Narrative"
  }</Story>
  <Manifest>{
    "segs": [
      { "id": 1, "start_frame": 0, "end_frame": 60, "prompt": "Intro: fade from black", "action": "static" },
      ...
    ]
  }</Manifest>
  <FrameGrid encoding="delta" color_mode="palette-index">
    <!-- Base64-encoded JSON of the frames array, or path to external .json file -->
    { "palettes": {...}, "frames": [...] }
  </FrameGrid>
</XMVP>
```

The `<FrameGrid>` section is a custom XMVP extension. Renderers that don't understand it ignore it; renderers that do get frame-level precision.

## Compact Notation for Small Grids

For very small grids (≤16×16), frames can use a compact string notation instead of nested arrays:

```json
{
  "i": 0,
  "grid_compact": "00001122110000\n00012345432100\n00123456543210\n..."
}
```

Each character is a palette index (0–9 for palettes up to 10 colors, or hex a-f for 11–16). Rows separated by `\n`. This is ~4× more compact than the array notation for small grids.

## Energy Envelope Format

When extracted from audio, the energy envelope is stored separately (since it applies globally, not per-scene):

```json
{
  "energy_envelope": {
    "source": "rms_analysis",
    "sample_rate_hz": 4,
    "values": [0.05, 0.06, 0.08, 0.12, 0.15, 0.22, ...]
  }
}
```

If the sample rate doesn't match FPS, the renderer interpolates. If no audio is available, the skill generates a synthetic energy envelope based on section profiles:
- `"quiet"`: 0.1–0.2
- `"building"`: linear ramp from 0.2 to section's `energy_avg`
- `"mid"`: 0.4–0.6
- `"high"`: 0.7–0.9
- `"climax"`: 0.9–1.0
- `"falling"`: linear ramp down
- `"breakdown"`: 0.2–0.4

Plus sinusoidal micro-variation at the beat frequency to simulate rhythmic pulse.
