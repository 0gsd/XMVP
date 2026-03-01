---
name: algo-rhythmic-narrative
description: "Translate a music video 'shooting schedule' into frame-by-frame color-coded pixel grids for XMVP's cell-based painting pipeline. Given a song's BPM, section map, and scene descriptions, this skill generates per-frame JSON instruction files where each frame is a small pixel grid (e.g. 64x64) with per-cell color and annotation data, synced to musical beats and sections. Use whenever the user wants to plan a music video as pixel grids, create a frame-by-frame color map for animation, convert a storyboard or shooting schedule into cell-painting instructions, generate beat-synced frame data for XMVP's frame_canvas.py, or bridge narrative scene descriptions to per-pixel color grids. Also trigger for 'algo-rhythmic', 'rhythmic narrative', 'shooting schedule to frames', 'beat-synced pixel grids', 'music video frame map', 'cell painting instructions', or 'narrative-to-grid'. Works with cartoon_producer.py's music-video and music-agency modes."
---

# Algo-Rhythmic Narrative

Translate a music video shooting schedule into frame-by-frame, cell-by-cell color-coded pixel grids — one grid per frame of a long narrative sequence, synced to a song's BPM and sections.

## Before Starting: Read References

Load these before generating any frame data:

1. `references/pipeline-integration.md` — How outputs connect to XMVP's frame_canvas.py, cartoon_producer.py, and unicode_visualizer.py. Data formats, file conventions, and the handoff protocol.
2. `references/grid-language.md` — The "Grid Language" specification: how to encode narrative meaning, mood, motion, and emphasis into color values and cell annotations at the pixel level.

These files contain the essential data contracts. Do not proceed without reading them.

## Core Concept

A music video is a temporal narrative synchronized to musical structure. This skill is the **translation layer** between:

- **Input**: A "shooting schedule" — scene descriptions, timings, moods, camera directions, character positions — organized by musical section (intro, verse, chorus, bridge, etc.)
- **Output**: A sequence of small pixel grids (JSON), one per frame, where every cell carries color data and optional annotation data that downstream painting/generation tools can interpret.

The grids are NOT the final visual output. They are **instruction maps** — structured blueprints that tell frame_canvas.py, an LLM image generator, or any cell-based renderer *what to paint where and when*. Think of them as the musical score to the orchestra's performance.

## The Three Clocks

Every frame in an algo-rhythmic narrative is governed by three simultaneous temporal systems:

1. **Musical Clock** — BPM, beats, bars, sections. The rhythmic skeleton.
2. **Narrative Clock** — Scene progression, character arcs, emotional trajectory. The story.
3. **Visual Clock** — Camera moves, color transitions, composition shifts. The eye.

The skill's job is to **synchronize all three clocks** into a single grid-per-frame output stream.

## User Input

The user provides:

### Required
1. **Song metadata**: BPM, duration, and a section map (timestamps for intro/verse/chorus/bridge/outro or equivalent).
2. **Shooting schedule**: A list of scenes with descriptions. Each scene specifies:
   - Time range (or musical section reference)
   - Visual description (what we see)
   - Mood/energy level
   - Optional: camera direction, character positions, color palette hints

### Optional
3. **Grid dimensions**: Default 64x64. Can be any power-of-2 square or rectangular grid.
4. **FPS**: Default derived from BPM (4 frames per beat). Can be overridden.
5. **Audio file**: If provided, run `analyze_audio()` and `analyze_audio_profile()` for automatic BPM/section detection.
6. **XMVP manifest** (`.xml`): If provided, extract Bible/Story/Portions/Manifest data directly.
7. **Style vocabulary**: Named palette or color-coding convention (see Grid Language reference).

## Pipeline Overview

```
INPUT                               PHASE 1                 PHASE 2
┌──────────────┐                   Temporal                Scene
│ Song Metadata │──┐               Scaffold               Decomposition
│ BPM/sections  │  │    ┌──────────────────┐    ┌──────────────────────┐
└──────────────┘  ├───▶│ Beat grid         │───▶│ Zones per scene      │
┌──────────────┐  │    │ Section boundaries│    │ Color palettes       │
│ Shooting     │──┤    │ Scene boundaries  │    │ Motion vectors       │
│ Schedule     │  │    │ Frame count       │    │ Evolution curves     │
└──────────────┘  │    └──────────────────┘    └──────────┬───────────┘
┌──────────────┐  │                                       │
│ Audio File   │──┘                                       ▼
│ (optional)   │                                 PHASE 3: Grid Synthesis
└──────────────┘                                 ┌──────────────────────┐
                                                 │ For each frame:      │
                                                 │  Musical position    │
                                                 │  Active scene(s)     │
                                                 │  Per-cell RGB        │
                                                 │  Beat modulation     │
                                                 │  Transition blend    │
                                                 └──────────┬───────────┘
                                                            │
                                                            ▼
                                                 PHASE 4: Export
                                                 ┌──────────────────────┐
                                                 │ manifest.json        │
                                                 │ frames/*.json        │
                                                 │ preview.html         │
                                                 │ XMVP .xml (optional) │
                                                 └──────────────────────┘
```

## Phase 1: Temporal Scaffold

The temporal scaffold is the rhythmic skeleton that everything else hangs on.

### Building the Scaffold

```python
# Core calculation
fps = (bpm / 60.0) * frames_per_beat  # e.g. 120 BPM * 4 fpb = 8 fps
total_frames = int(duration * fps)

# Beat grid
frames_per_beat_exact = fps * (60.0 / bpm)
beat_boundaries = [int(i * frames_per_beat_exact) for i in range(total_beats)]

# Bar grid (assuming 4/4 time unless specified)
beats_per_bar = time_signature[0]  # default 4
bar_boundaries = beat_boundaries[::beats_per_bar]
```

### Scaffold JSON Structure

```json
{
  "song": {
    "title": "Track Name",
    "bpm": 120,
    "duration_sec": 210.5,
    "time_signature": [4, 4],
    "fps": 8.0,
    "frames_per_beat": 4,
    "total_frames": 1684,
    "total_beats": 421,
    "total_bars": 105
  },
  "sections": [
    {
      "name": "intro",
      "start_sec": 0.0,
      "end_sec": 16.0,
      "start_frame": 0,
      "end_frame": 128,
      "start_beat": 0,
      "end_beat": 32,
      "energy": "low",
      "mood": "mysterious"
    }
  ],
  "scenes": [
    {
      "id": 1,
      "description": "Wide shot: empty highway at dawn, heat shimmer",
      "section": "intro",
      "start_frame": 0,
      "end_frame": 64,
      "palette": "dawn_warm",
      "camera": "slow_push",
      "zones": {
        "sky": { "y_range": [0.0, 0.4], "base_color": "#2B1B4E" },
        "horizon": { "y_range": [0.4, 0.5], "base_color": "#FF6B35" },
        "road": { "y_range": [0.5, 1.0], "base_color": "#1A1A2E" }
      }
    }
  ]
}
```

## Phase 2: Scene Decomposition

Each scene from the shooting schedule gets decomposed into **spatial zones** and **temporal curves**.

### Spatial Zones

A zone is a rectangular region of the grid that carries consistent semantic meaning:

- **Background zones**: sky, ground, walls, environment — large areas, slow-changing
- **Subject zones**: characters, objects — smaller areas, more dynamic
- **Accent zones**: highlights, effects, overlays — sparse, beat-reactive

Zones are defined as fractional y-ranges (and optionally x-ranges) of the grid. They can overlap (later zones composite over earlier ones).

### Temporal Curves

Each zone has evolution curves that define how its properties change frame-to-frame:

- **Color curve**: Base color to target color over the scene's duration (linear, ease-in/out, step)
- **Brightness curve**: Modulated by beat phase (pulse on downbeats)
- **Position curve**: For subject zones that move (character walks, camera pans)
- **Opacity curve**: For fade-in/fade-out, crossfades between scenes

### Beat Modulation

The Grid Language (see reference) defines how beats affect the grid:

- **Downbeat (beat 1 of bar)**: +15% brightness across all zones, optional color shift
- **Backbeat (beats 2, 4)**: +8% brightness in accent zones
- **Off-beat**: Neutral
- **Fill/syncopation**: Brief complementary color flash in accent zones

## Phase 3: Grid Synthesis

This is the core generation loop. For each frame:

1. **Determine musical position**: Which beat, which bar, which section, beat phase (0.0-1.0 within the beat)
2. **Determine active scene**: Which scene(s) are active, and if transitioning, the blend factor
3. **For each cell (row, col)**:
   - Determine which zone(s) it belongs to
   - Sample the zone's color at this frame's temporal position
   - Apply beat modulation
   - Apply transition blending if between scenes
   - Write the final RGB value

### Frame JSON Structure

Each frame outputs a JSON file:

```json
{
  "frame_index": 42,
  "time_sec": 5.25,
  "beat": 10,
  "beat_phase": 0.5,
  "bar": 2,
  "bar_beat": 2,
  "section": "intro",
  "scene_id": 1,
  "transition": null,
  "grid": {
    "width": 64,
    "height": 64,
    "cells": [
      [{"r": 43, "g": 27, "b": 78}, {"r": 44, "g": 28, "b": 79}],
      "..."
    ]
  },
  "annotations": {
    "zones_active": ["sky", "horizon", "road"],
    "beat_emphasis": 0.0,
    "narrative_weight": 0.2,
    "camera_offset": [0.0, -0.02]
  }
}
```

### Compact Grid Format (for large frame counts)

For videos with thousands of frames, use the compact format — flat hex strings instead of nested objects:

```json
{
  "f": 42,
  "t": 5.25,
  "b": 10,
  "bp": 0.5,
  "s": "intro",
  "sc": 1,
  "px": "2B1B4E2C1C4F2D1D50..."
}
```

Where `px` is a hex string of concatenated RGB values, row-major order. This reduces file size by ~80% vs. verbose format.

## Phase 4: Export

### Output Files

```
algo-rhythmic-output/
├── narrative_grid_manifest.json    # Master index
├── temporal_scaffold.json          # Beat/section map
├── scene_plans.json                # Decomposed scenes
├── frames/
│   ├── frame_00000.json            # Per-frame grids (compact)
│   ├── frame_00001.json
│   └── ...
├── palettes.json                   # Named color palettes used
└── preview.html                    # Animated grid viewer
```

### Manifest Structure

```json
{
  "version": "1.0",
  "generator": "algo-rhythmic-narrative",
  "song": { "...from scaffold..." },
  "grid_dimensions": [64, 64],
  "total_frames": 1684,
  "format": "compact",
  "frame_pattern": "frames/frame_{:05d}.json",
  "xmvp_compat": {
    "bible_style": "cell-painting instruction grids",
    "portions_count": 8,
    "manifest_segments": 1684
  }
}
```

### XMVP Integration

To feed this into XMVP directly, generate an XMVP XML manifest where each segment references the corresponding frame grid:

```xml
<XMVP version="3.03">
  <Bible>{"constraints": {"width": 64, "height": 64, "fps": 8, ...}, ...}</Bible>
  <Story>{"title": "...", "synopsis": "...", ...}</Story>
  <Manifest>{"segs": [
    {"id": 0, "start_frame": 0, "end_frame": 1,
     "prompt": "GRID_INSTRUCTION: Paint per frames/frame_00000.json RGB values.",
     "grid_ref": "frames/frame_00000.json"},
    ...
  ]}</Manifest>
</XMVP>
```

The `grid_ref` field is a non-standard extension. XMVP's `cartoon_producer.py` can be patched to recognize it, or the prompt text itself can encode the instruction for LLM-based painters.

## Implementation Strategy

### For Short Sequences (< 500 frames)

Generate all frame JSONs directly in Claude. Write a Python script that:
1. Parses the shooting schedule
2. Builds the temporal scaffold
3. Synthesizes each grid frame-by-frame
4. Writes to `/mnt/user-data/outputs/`

### For Long Sequences (500+ frames)

Generate a **standalone Python generator script** that the user runs locally:

1. Generate `temporal_scaffold.json` and `scene_plans.json` in full
2. Generate `grid_synthesizer.py` — a standalone script that reads the scaffold + plans and outputs all frames
3. The synthesizer script should:
   - Accept `--start-frame` and `--end-frame` for partial generation
   - Support `--preview` mode (renders every Nth frame)
   - Output to a configurable directory
   - Include a progress bar
   - Require only numpy and standard library (no XMVP dependencies)

### Preview HTML

Always generate a preview HTML artifact that:
- Loads the manifest and a subset of frames (or inline data for short sequences)
- Animates them at the correct FPS
- Shows beat markers and section labels
- Allows scrubbing through the timeline
- Displays the current frame's grid as colored cells on a canvas
- Can be opened standalone in a browser

## Quality Checklist

Before outputting, verify:
- [ ] Total frames matches `duration * fps` (within +/- 1 frame)
- [ ] Beat boundaries are correctly calculated from BPM
- [ ] Scene transitions don't leave gaps or overlaps in the timeline
- [ ] Every cell in every frame has a valid RGB value
- [ ] Compact hex strings are exactly `width * height * 6` characters
- [ ] Downbeat frames show measurable brightness increase vs. off-beat frames
- [ ] Section changes align with shooting schedule timings
- [ ] Manifest frame count matches actual frame file count
- [ ] Preview HTML animates at correct speed

## Anti-Patterns

- Do not generate identical grids for consecutive frames (even subtle evolution should be visible)
- Do not ignore beat phase (grids should breathe with the music)
- Do not hard-cut between scenes without transition blending unless the schedule explicitly specifies jump cuts
- Do not produce monochrome grids with no spatial zoning (every frame should have compositional structure)
- Do not generate frame files larger than 100KB each (use compact format for grids > 32x32)
- Do not generate thousands of files directly in Claude — use the generator script pattern for long videos
