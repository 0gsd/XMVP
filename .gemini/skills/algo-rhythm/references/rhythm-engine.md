# Rhythm Engine Reference

The math, logic, and procedures for syncing visual frames to musical time.

## BPM Sync Fundamentals

### The Beat Grid

Everything in algo-rhythmic-narrative is quantized to a **beat grid**. The grid is defined by:

```
frames_per_beat = fps × (60 / bpm)
frames_per_bar  = frames_per_beat × beats_per_bar
```

**Example** (120 BPM, 4/4 time, 24 FPS):
```
frames_per_beat = 24 × (60 / 120) = 12.0
frames_per_bar  = 12.0 × 4 = 48.0
```

**Example** (90 BPM, 4/4 time, 4 FPS):
```
frames_per_beat = 4 × (60 / 90) = 2.667
frames_per_bar  = 2.667 × 4 = 10.667
```

Note: `frames_per_beat` is often fractional. We do NOT round it. We compute beat position per frame using floating-point arithmetic:

```python
beat_number = floor(frame_index / frames_per_beat)
beat_phase  = (frame_index / frames_per_beat) - beat_number   # 0.0–0.999
bar_number  = floor(frame_index / frames_per_bar)
bar_phase   = (frame_index / frames_per_bar) - bar_number
is_on_beat  = beat_phase < (1.0 / frames_per_beat)  # True for the first frame of each beat
is_downbeat = is_on_beat and (beat_number % beats_per_bar == 0)
```

### Low FPS Considerations

At 4 FPS / 120 BPM, we get 2 frames per beat. This means:
- Every beat gets only 2 frames of visual expression
- A "flash + decay" effect is literally: frame 1 = flash, frame 2 = decay, done
- Subtlety comes from palette variation, not animation smoothness

At 4 FPS / 180 BPM, we get 1.33 frames per beat. Some beats don't get their own frame at all. The beat grid is still computed correctly — it just means some frames represent the "between" of two beats, and the renderer must interpolate.

**Rule**: Never skip a beat in the metadata. Even if a beat doesn't land exactly on a frame, the nearest frame's `beat_number` and `beat_phase` still reflect the correct musical position.

### Swing and Shuffle

For songs with swing/shuffle feel, the beat grid can be modified:

```json
{
  "meta": {
    "swing": 0.6
  }
}
```

A swing value of 0.5 = straight time (default). 0.6 = light swing. 0.67 = hard triplet swing.

When swing is active, even-numbered eighth notes are delayed:
```
straight_position = beat_number × frames_per_beat
swing_offset = (beat_number % 2 == 1) ? (swing - 0.5) × frames_per_beat : 0
actual_position = straight_position + swing_offset
```

## Audio Analysis Pipeline

When the user provides an audio file, extract temporal features before generating any frames.

### Step 1: Basic Info

```bash
ffprobe -v quiet -print_format json -show_format -show_streams <input_file>
```

Extract: duration, sample rate, channels.

### Step 2: BPM Detection

If BPM not manually specified:

```python
import librosa
y, sr = librosa.load(audio_path)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# tempo is estimated BPM
# beat_frames are sample indices of detected beats
```

Or via ffmpeg + onset detection if librosa is unavailable. If the user provides BPM manually, skip detection.

### Step 3: Energy Envelope (RMS)

```bash
ffmpeg -i <input_file> \
  -af "astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=rms.txt" \
  -f null -
```

Parse `rms.txt`:
```python
import re
times, levels = [], []
with open('rms.txt') as f:
    t = None
    for line in f:
        m = re.search(r'pts_time:([\d.]+)', line)
        if m: t = float(m.group(1))
        m = re.search(r'RMS_level=([-\d.]+)', line)
        if m and t is not None:
            times.append(t)
            levels.append(float(m.group(1)))
```

RMS values are in dB (negative). Normalize to 0.0–1.0:
```python
import numpy as np
levels_arr = np.array(levels)
# Typical range: -60 dB (silence) to -3 dB (loud)
normalized = np.clip((levels_arr + 60) / 57, 0.0, 1.0)
```

Resample to target FPS using linear interpolation:
```python
frame_times = np.arange(total_frames) / fps
energy_per_frame = np.interp(frame_times, times, normalized)
```

### Step 4: Section Detection

**Automatic** (if using librosa):
```python
# Spectral clustering for section boundaries
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
boundaries = librosa.segment.agglomerative(mfcc, k=8)
boundary_times = librosa.frames_to_time(boundaries, sr=sr)
```

**Manual** (preferred — user provides section map):
```
0:00-0:15  Intro
0:15-1:02  Verse 1
1:02-1:30  Chorus 1
```

Parse into section objects with start/end frames.

**Hybrid**: User provides rough sections, RMS analysis confirms/adjusts boundaries to nearest bar line.

### Step 5: Section Energy Profiling

For each section, compute:
```python
section_energy = energy_per_frame[start_frame:end_frame]
energy_avg = np.mean(section_energy)
energy_peak = np.max(section_energy)
energy_variance = np.var(section_energy)

# Classify
if energy_avg < 0.25:
    profile = "quiet"
elif energy_avg < 0.45 and energy_variance > 0.02:
    profile = "building"
elif energy_avg < 0.55:
    profile = "mid"
elif energy_avg < 0.75:
    profile = "high"
else:
    profile = "climax"

# Override: if energy trends strongly up/down within section
gradient = np.polyfit(range(len(section_energy)), section_energy, 1)[0]
if gradient > 0.001:
    profile = "building"
elif gradient < -0.001:
    profile = "falling"
```

## Scene Mapping Logic

### Assigning Scenes to Sections

The user's shooting schedule maps scenes to musical sections. One scene can span multiple sections, and one section can contain multiple scenes (cuts within a verse, for example).

**Simple case**: 1:1 mapping. "Verse 1 → hands-typing, Chorus 1 → city-night."

**Complex case**: The user specifies sub-cuts. "Verse 1: first 8 bars = hands-typing, last 4 bars = face-closeup."

In both cases, the skill resolves to frame ranges:
```python
for scene in scenes:
    for section_id in scene.section_ids:
        section = sections_by_id[section_id]
        scene.frame_ranges.append((section.start_frame, section.end_frame))
```

### Transition Zones

Between consecutive scenes, allocate a **transition zone** — frames that belong to both the outgoing and incoming scene.

**Default duration**: 1 bar (frames_per_bar frames).

**Transition types**:
- `"crossfade"`: Linear blend. `progress` goes 0→1 over the transition. Renderer blends grids.
- `"cut"`: No transition. Last frame of scene A, first frame of scene B, hard switch. Transition zone is 0 frames.
- `"wipe"`: Directional reveal. `progress` controls how far the wipe has swept.
- `"flash"`: All cells go to `flash` color at midpoint, then resolve to new scene.
- `"beat-cut"`: Cut happens precisely on the next downbeat.

For "beat-cut" (the most musical transition):
```python
# Find the first downbeat at or after the section boundary
boundary_frame = section.end_frame
beat_at_boundary = boundary_frame / frames_per_beat
next_downbeat = ceil(beat_at_boundary / beats_per_bar) * beats_per_bar
cut_frame = int(next_downbeat * frames_per_beat)
```

## Grid Generation Logic

### Composition Template Execution

Each composition template is a function that takes (grid_w, grid_h, palette, params) and returns a 2D array of palette indices.

**center-subject**:
```python
def center_subject(w, h, palette, params):
    grid = np.full((h, w), params.bg_palette_indices[0])
    x1 = int(params.subject_rect[0] * w)
    y1 = int(params.subject_rect[1] * h)
    x2 = int(params.subject_rect[2] * w)
    y2 = int(params.subject_rect[3] * h)
    # Fill subject area with subject palette indices
    for y in range(y1, y2):
        for x in range(x1, x2):
            # Use inner palette indices with some variation
            dist_from_center = sqrt(((x - (x1+x2)/2) / (x2-x1))**2 +
                                    ((y - (y1+y2)/2) / (y2-y1))**2)
            idx = int(dist_from_center * len(params.subject_palette_indices))
            idx = min(idx, len(params.subject_palette_indices) - 1)
            grid[y][x] = params.subject_palette_indices[idx]
    return grid
```

**noise-field** (for organic, non-geometric scenes):
```python
def noise_field(w, h, palette, params):
    grid = np.zeros((h, w), dtype=int)
    scale = params.get('scale', 0.1)
    seed = params.get('seed', 42)
    rng = np.random.default_rng(seed)
    # Simple Perlin-like noise approximation
    for y in range(h):
        for x in range(w):
            # Value noise with interpolation
            val = noise_2d(x * scale, y * scale, seed=seed)  # 0.0–1.0
            palette_idx = int(val * (len(palette.colors) - 1))
            grid[y][x] = palette_idx
    return grid
```

### Beat Rule Execution

Beat rules modify the base grid on specific frames. They are applied in order after the base composition.

**brightness-pulse**:
```python
def apply_brightness_pulse(grid, palette, intensity, decay_frames, frames_since_trigger):
    if frames_since_trigger > decay_frames:
        return grid  # Effect expired
    # Decay curve (exponential)
    strength = intensity * (1.0 - frames_since_trigger / decay_frames) ** 2
    # Shift all palette indices toward brighter end
    max_idx = len(palette.colors) - 1
    shift = int(strength * max_idx * 0.5)
    return np.clip(grid + shift, 0, max_idx)
```

**ripple**:
```python
def apply_ripple(grid, w, h, palette, origin, speed, width, palette_index, frames_since_trigger):
    radius = speed * frames_since_trigger
    cx, cy = origin if origin != "center" else (w // 2, h // 2)
    for y in range(h):
        for x in range(w):
            dist = sqrt((x - cx)**2 + (y - cy)**2)
            if abs(dist - radius) < width:
                grid[y][x] = palette_index
    return grid
```

### Energy Modulation

Applied after beat rules. Scales the effective palette range based on current energy:

```python
def apply_energy(grid, palette, energy, rules):
    floor = rules.brightness_floor  # e.g., 0.2
    ceiling = rules.brightness_ceiling  # e.g., 1.0
    # Map energy to effective palette range
    effective_max = floor + (ceiling - floor) * energy
    max_idx = len(palette.colors) - 1
    # Scale all indices
    scaled = (grid.astype(float) / max_idx) * effective_max * max_idx
    return np.clip(scaled.astype(int), 0, max_idx)
```

### Transition Blending

During transition zones, blend two grids:

```python
def blend_grids(grid_out, grid_in, progress, mode="crossfade"):
    if mode == "crossfade":
        # Probabilistic blend: each cell picks from outgoing or incoming
        # based on progress (stochastic dither for integer palette indices)
        mask = np.random.random(grid_out.shape) < progress
        return np.where(mask, grid_in, grid_out)
    elif mode == "wipe":
        # Left-to-right wipe
        split_col = int(progress * grid_out.shape[1])
        result = grid_out.copy()
        result[:, :split_col] = grid_in[:, :split_col]
        return result
    elif mode == "flash":
        if progress < 0.5:
            # Brightening phase
            flash_idx = len(palette.colors) - 1  # Flash color
            amount = progress * 2  # 0→1 in first half
            mask = np.random.random(grid_out.shape) < amount
            result = grid_out.copy()
            result[mask] = flash_idx
            return result
        else:
            # Resolving phase
            amount = (progress - 0.5) * 2  # 0→1 in second half
            flash_idx = len(palette.colors) - 1
            mask = np.random.random(grid_in.shape) < (1.0 - amount)
            result = grid_in.copy()
            result[mask] = flash_idx
            return result
```

## Frame Generation Pseudocode

The complete per-frame generation loop:

```python
def generate_all_frames(meta, sections, palettes, scenes, energy_envelope):
    frames = []
    prev_grid = None

    for f in range(meta.total_frames):
        # 1. Rhythm metadata
        beat = floor(f / meta.frames_per_beat)
        bar = floor(f / meta.frames_per_bar)
        beat_phase = (f / meta.frames_per_beat) - beat
        section = find_section(f, sections)
        energy = energy_envelope[f] if energy_envelope else estimate_energy(f, section)

        # 2. Scene lookup
        scene, transition = find_scene_and_transition(f, scenes, sections, meta)

        # 3. Generate base grid
        palette = palettes[scene.palette_id]
        grid = apply_composition(scene.composition, meta.grid_w, meta.grid_h, palette)

        # 4. Beat rules
        for rule in scene.beat_rules:
            if is_triggered(rule, beat, bar, beat_phase, energy, f):
                frames_since = f - last_trigger_frame(rule, f, meta)
                grid = apply_effect(grid, rule.effect, palette, rule.params, frames_since)

        # 5. Energy modulation
        grid = apply_energy(grid, palette, energy, scene.energy_rules)

        # 6. Transition blending
        if transition:
            other_scene = scenes_by_id[transition.to_scene if transition.progress < 0.5 else transition.from_scene]
            other_palette = palettes[other_scene.palette_id]
            other_grid = apply_composition(other_scene.composition, meta.grid_w, meta.grid_h, other_palette)
            grid = blend_grids(
                grid if transition.progress < 0.5 else other_grid,
                other_grid if transition.progress < 0.5 else grid,
                transition.progress, transition.mode
            )

        # 7. Emit
        frame = {
            "i": f,
            "t": round(f / meta.fps, 3),
            "beat": beat,
            "bar": bar,
            "beat_phase": round(beat_phase, 3),
            "section": section.id,
            "scene": scene.id,
            "energy": round(energy, 3),
            "transition": transition,
            "directive": generate_directive(scene, beat, bar, energy, transition),
            "grid": grid.tolist()
        }

        # Delta encoding
        if meta.encoding == "delta" and prev_grid is not None:
            if not frame.get("keyframe"):
                delta = compute_delta(prev_grid, grid)
                frame["delta"] = delta
                del frame["grid"]

        frames.append(frame)
        prev_grid = grid.copy()

    return frames

def compute_delta(prev, current):
    """Returns list of [row, col, new_value] for changed cells."""
    changes = []
    for y in range(prev.shape[0]):
        for x in range(prev.shape[1]):
            if prev[y][x] != current[y][x]:
                changes.append([y, x, int(current[y][x])])
    return changes

def generate_directive(scene, beat, bar, energy, transition):
    """One-sentence human-readable description of this frame."""
    parts = [scene.narrative_directive]
    if energy > 0.8:
        parts.append("High energy.")
    if beat % 4 == 0:
        parts.append("Downbeat.")
    if transition:
        parts.append(f"Transition {int(transition.progress * 100)}%.")
    return " ".join(parts)
```

## Quantization Helpers

### Snap to Beat

When a scene boundary or event should land on a beat:
```python
def snap_to_beat(frame_index, frames_per_beat):
    return round(frame_index / frames_per_beat) * frames_per_beat

def snap_to_bar(frame_index, frames_per_bar):
    return round(frame_index / frames_per_bar) * frames_per_bar

def next_downbeat_after(frame_index, frames_per_bar):
    return ceil(frame_index / frames_per_bar) * frames_per_bar
```

### Time ↔ Frame Conversion

```python
def time_to_frame(seconds, fps):
    return int(seconds * fps)

def frame_to_time(frame_index, fps):
    return frame_index / fps

def timecode_to_seconds(tc):
    """Parse 'M:SS' or 'H:MM:SS' to seconds."""
    parts = tc.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
```

## Practical Sizing Guide

| Duration | FPS | BPM | Total Frames | Grid 64×64 palette-idx | Grid 120×40 palette-idx |
|----------|-----|-----|-------------|----------------------|------------------------|
| 30s      | 4   | 120 | 120         | ~31 KB full / ~8 KB delta | ~19 KB full / ~5 KB delta |
| 3:30     | 4   | 120 | 840         | ~215 KB / ~55 KB | ~134 KB / ~35 KB |
| 3:30     | 24  | 120 | 5,040       | ~1.3 MB / ~320 KB | ~800 KB / ~200 KB |
| 5:00     | 24  | 140 | 7,200       | ~1.8 MB / ~450 KB | ~1.1 MB / ~280 KB |

Delta encoding assumes ~25% of cells change per frame on average (highly scene-dependent).

For songs > 5 minutes at 24fps, use section-chunked output. Each chunk ≤ 50MB.
