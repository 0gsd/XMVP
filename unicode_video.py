#!/usr/bin/env python3
"""
unicode_video.py - Direct Text-to-Unicode-Art Video Pipeline
============================================================
Generates music videos as unicode block art.  The LLM acts as "art director",
providing compact scene descriptions once per detected section.  Procedural
Python code then renders full-coverage pixel grids for every frame, driven
by real-time audio features (loudness, onset, chroma, beat).

Key design:
  - librosa section detection splits the song into natural segments
  - One LLM call per section (~200 tokens) instead of per frame (~12K)
  - Procedural renderer guarantees full-coverage colourful frames
  - Audio features from TrackAnalyzer drive per-frame animation

Usage:
    python unicode_video.py \
        --prompt "Dr. Music visits the CD-ROOM..." \
        --mu /path/to/audio.aif \
        --style "abstract distorted 1990s collage" \
        --w 576 \
        [--h HEIGHT] [--fsync 0.5] [--bpm 120]
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import subprocess
import colorsys
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from text_engine import TextEngine
from unicode_visualizer import (
    grid_to_image, load_multi_font, TrackAnalyzer,
    CELL_W, CELL_H,
)
from cartoon_producer import analyze_audio, analyze_audio_profile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Density gradient for overlay shading (sparse -> dense)
DENSITY_CHARS = " \u00b7.,:;!|{}[]()#@\u2588"

# Logical grid size limits
MAX_LOG_W = 64
MAX_LOG_H = 48

# ── Fallback palettes (used when LLM call fails) ────────────────────────────
FALLBACK_PALETTES = [
    ["#FF6B35", "#004E89", "#1A936F", "#C6DABF", "#88D498"],
    ["#E63946", "#457B9D", "#1D3557", "#F1FAEE", "#A8DADC"],
    ["#FFD166", "#06D6A0", "#118AB2", "#073B4C", "#EF476F"],
    ["#7400B8", "#6930C3", "#5390D9", "#4EA8DE", "#48BFE3"],
    ["#FF006E", "#FB5607", "#FFBE0B", "#3A86FF", "#8338EC"],
    ["#2D00F7", "#6A00F4", "#8900F2", "#A100F2", "#B100E8"],
    ["#F72585", "#B5179E", "#7209B7", "#560BAD", "#480CA8"],
    ["#606C38", "#283618", "#FEFAE0", "#DDA15E", "#BC6C25"],
]

# Pattern types the zone renderer supports
ZONE_PATTERNS = [
    "gradient", "noise", "blocks", "stripes",
    "scatter", "solid", "vignette", "plasma",
    "checkerboard", "radial", "diagonal", "wave",
]

# Fallback zone templates for deterministic fallback
FALLBACK_ZONE_TEMPLATES = [
    # horizon/landscape
    [{"name": "sky",    "y": [0.0, 0.45], "pattern": "gradient"},
     {"name": "horizon","y": [0.40, 0.60], "pattern": "plasma"},
     {"name": "ground", "y": [0.55, 1.0],  "pattern": "noise"}],
    # center-subject
    [{"name": "bg",     "y": [0.0, 1.0],  "pattern": "noise"},
     {"name": "subject","y": [0.2, 0.8],  "x": [0.2, 0.8], "pattern": "blocks"}],
    # full-field
    [{"name": "field",  "y": [0.0, 1.0],  "pattern": "plasma"},
     {"name": "accent", "y": [0.7, 1.0],  "pattern": "stripes"}],
    # vertical split
    [{"name": "left",   "y": [0.0, 1.0],  "x": [0.0, 0.5], "pattern": "gradient"},
     {"name": "right",  "y": [0.0, 1.0],  "x": [0.5, 1.0], "pattern": "blocks"}],
    # vignette
    [{"name": "outer",  "y": [0.0, 1.0],  "pattern": "solid"},
     {"name": "inner",  "y": [0.15, 0.85],"x": [0.15, 0.85], "pattern": "plasma"}],
    # bands
    [{"name": "top",    "y": [0.0, 0.33], "pattern": "stripes"},
     {"name": "mid",    "y": [0.33, 0.66],"pattern": "scatter"},
     {"name": "bot",    "y": [0.66, 1.0], "pattern": "blocks"}],
]


# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(hex_color):
    """Convert '#RRGGBB' or 'RRGGBB' to (r, g, b) tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) >= 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (128, 128, 128)  # grey fallback, not black


def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB tuples."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def palette_sample(palette_rgb, t):
    """Sample a colour along a palette gradient (t in 0..1)."""
    t = max(0.0, min(1.0, t))
    n = len(palette_rgb)
    if n == 0:
        return (128, 128, 128)
    if n == 1:
        return palette_rgb[0]
    idx = t * (n - 1)
    i = int(idx)
    frac = idx - i
    if i >= n - 1:
        return palette_rgb[-1]
    return lerp_color(palette_rgb[i], palette_rgb[i + 1], frac)


def ensure_vibrant(r, g, b):
    """Enforce minimum saturation and brightness to prevent desaturation."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    s = max(0.4, s)   # Never drop below 40% saturation
    v = max(0.25, v)   # Never drop below 25% brightness
    rf, gf, bf = colorsys.hsv_to_rgb(h, s, v)
    return int(rf * 255), int(gf * 255), int(bf * 255)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE DESCRIPTION (LLM) — Zone-based
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scene_description(text_engine, section_prompt, section_idx, retries=2, prev_mood=None):
    """Ask the LLM for a zone-based scene description for one section.

    Returns a dict with zones (each having its own palette and pattern),
    accent effects, mood, and motion type.
    Falls back to deterministic defaults on failure.
    """
    # Rotate through diverse layout examples so the LLM doesn't always copy sky/horizon/ground
    _EXAMPLES = [
        # A — Diagonal / x-range split
        '{\n'
        '  "zones": [\n'
        '    {"name": "left_field", "y": [0.0, 1.0], "x": [0.0, 0.55], "palette": ["#2B1B4E","#5533AA","#8855CC"], "pattern": "plasma"},\n'
        '    {"name": "right_field", "y": [0.0, 1.0], "x": [0.45, 1.0], "palette": ["#FF6B35","#FFD700","#FF3366"], "pattern": "noise"},\n'
        '    {"name": "accent_band", "y": [0.35, 0.65], "palette": ["#00FFAA","#00AAFF"], "pattern": "stripes"}\n'
        '  ],\n'
        '  "accents": ["flash", "shake"],\n'
        '  "mood": "electric",\n'
        '  "motion": "cascade"\n'
        '}',
        # B — Full-field with centered inset subject
        '{\n'
        '  "zones": [\n'
        '    {"name": "atmosphere", "y": [0.0, 1.0], "palette": ["#0D0221","#150734","#261447"], "pattern": "plasma"},\n'
        '    {"name": "subject", "y": [0.15, 0.85], "x": [0.2, 0.8], "palette": ["#FF006E","#FB5607","#FFBE0B"], "pattern": "radial"},\n'
        '    {"name": "floor_glow", "y": [0.75, 1.0], "palette": ["#3A86FF","#8338EC"], "pattern": "wave"}\n'
        '  ],\n'
        '  "accents": ["ripple", "column_bars"],\n'
        '  "mood": "psychedelic",\n'
        '  "motion": "breathe"\n'
        '}',
        # C — Vertical columns / totally different composition
        '{\n'
        '  "zones": [\n'
        '    {"name": "col_left", "y": [0.0, 1.0], "x": [0.0, 0.33], "palette": ["#F72585","#B5179E","#7209B7"], "pattern": "checkerboard"},\n'
        '    {"name": "col_center", "y": [0.0, 1.0], "x": [0.28, 0.72], "palette": ["#06D6A0","#118AB2","#073B4C"], "pattern": "diagonal"},\n'
        '    {"name": "col_right", "y": [0.0, 1.0], "x": [0.67, 1.0], "palette": ["#FFD166","#EF476F","#26547C"], "pattern": "blocks"}\n'
        '  ],\n'
        '  "accents": ["scanline"],\n'
        '  "mood": "chaotic",\n'
        '  "motion": "shimmer"\n'
        '}',
    ]
    example = _EXAMPLES[section_idx % len(_EXAMPLES)]

    # Build contrast instruction from previous section
    contrast_hint = ""
    if prev_mood:
        contrast_hint = (
            f"\nIMPORTANT: The previous section used a '{prev_mood}' mood. "
            "Make THIS section visually DIFFERENT — contrasting composition, palette family, and energy level.\n"
        )

    prompt = (
        f"Design a VISUAL SCENE for section {section_idx + 1} of an animated music video.\n"
        f"Scene concept: {section_prompt}\n\n"
        "The scene is built from 2-4 SPATIAL ZONES — distinct visual regions that make this scene unique.\n"
        "Each zone has its own palette and pattern. Zones can overlap (later zones paint over earlier ones).\n\n"
        f"Return ONLY a JSON object (example for inspiration — DO NOT copy this layout):\n{example}\n\n"
        "CRITICAL: Do NOT always use horizontal bands (sky/horizon/ground). Vary the spatial composition:\n"
        "- Try vertical columns (using x ranges), centered inset subjects, diagonal splits, full-field patterns\n"
        "- Use x ranges [0.0, 0.5] and [0.5, 1.0] for left/right splits\n"
        "- Use overlapping zones where a smaller zone sits inside a larger background\n"
        "- Make each section look structurally DIFFERENT from the others\n\n"
        f"{contrast_hint}"
        "ZONE RULES:\n"
        "- Each zone needs: name, y (vertical range 0-1), palette (2-4 HEX colours), pattern\n"
        "- Optional: x (horizontal range 0-1, defaults to [0,1])\n"
        "- Patterns: gradient, noise, blocks, stripes, scatter, solid, vignette, plasma, checkerboard, radial, diagonal, wave\n"
        "- Make zones VISUALLY DISTINCT — different patterns, different palettes\n"
        "- Use VIVID, SATURATED colours — no greys, no pastels\n"
        "- NEVER make all zones 'solid' or 'gradient' — mix pattern types!\n\n"
        "ACCENTS (1-2): ripple, scanline, flash, column_bars, shake\n"
        "MOOD: calm, dreamy, energetic, chaotic, dark, warm, electric, psychedelic\n"
        "MOTION: pulse, drift, rotate, ripple, bounce, shimmer, cascade, breathe\n"
    )

    for attempt in range(retries + 1):
        try:
            raw = text_engine.generate(prompt, json_schema=True)
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            zones = data.get("zones", [])
            if not zones or not isinstance(zones, list):
                raise ValueError("Missing zones")
            # Validate each zone
            valid_zones = []
            for z in zones:
                if not isinstance(z, dict):
                    continue
                pal = z.get("palette", [])
                pal = [p for p in pal if isinstance(p, str) and len(p.lstrip('#')) >= 6]
                if len(pal) < 2:
                    continue
                z["palette"] = pal
                # Validate y range
                y_range = z.get("y", [0.0, 1.0])
                if not isinstance(y_range, list) or len(y_range) < 2:
                    z["y"] = [0.0, 1.0]
                # Validate pattern
                pattern = z.get("pattern", "gradient")
                if pattern not in ZONE_PATTERNS:
                    z["pattern"] = "gradient"
                valid_zones.append(z)
            if len(valid_zones) < 1:
                raise ValueError("No valid zones")
            # Anti-flat guard: reject if all zones are solid/gradient covering full canvas
            patterns_used = set(z.get("pattern", "gradient") for z in valid_zones)
            if patterns_used <= {"solid", "gradient"}:
                # Force at least one zone to use a more interesting pattern
                valid_zones[0]["pattern"] = "plasma"
                if len(valid_zones) > 1:
                    valid_zones[-1]["pattern"] = "noise"
                logging.info(f"   Anti-flat guard triggered for section {section_idx+1}, upgraded patterns")
            data["zones"] = valid_zones
            zone_summary = " | ".join(f"{z.get('name','?')}:{z['pattern']}" for z in valid_zones)
            logging.info(f"   Section {section_idx+1}: {data.get('mood','?')} — [{zone_summary}]")
            return data
        except Exception as e:
            logging.warning(f"   Scene description attempt {attempt+1} failed: {e}")
            if attempt < retries:
                time.sleep(0.5)

    # Deterministic fallback
    template_idx = section_idx % len(FALLBACK_ZONE_TEMPLATES)
    pal_idx = section_idx % len(FALLBACK_PALETTES)
    fallback_zones = []
    for z in FALLBACK_ZONE_TEMPLATES[template_idx]:
        zone = dict(z)
        # Rotate palette colours for variety
        pal = FALLBACK_PALETTES[(pal_idx + len(fallback_zones)) % len(FALLBACK_PALETTES)]
        zone["palette"] = pal[:3]
        fallback_zones.append(zone)
    logging.warning(f"   Using fallback zones for section {section_idx+1}")
    return {
        "zones": fallback_zones,
        "accents": ["ripple"],
        "mood": "energetic",
        "motion": "pulse",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN FUNCTIONS — each produces genuinely different visuals
# ═══════════════════════════════════════════════════════════════════════════════

def _pattern_gradient(nx, ny, palette_rgb, anim, phase, rng):
    """Smooth palette sweep."""
    t = (ny + anim * 0.3) % 1.0
    return palette_sample(palette_rgb, t)


def _pattern_noise(nx, ny, palette_rgb, anim, phase, rng):
    """Perlin-like noise field — organic, textured."""
    # Multi-octave value noise approximation using sin
    val = 0.0
    val += math.sin(nx * 7.3 + anim * 2.1) * 0.35
    val += math.sin(ny * 5.7 - anim * 1.8) * 0.35
    val += math.sin((nx + ny) * 4.1 + phase * 0.5) * 0.2
    val += math.sin((nx * 3.3 - ny * 2.7) + anim * 3.0) * 0.1
    t = (val + 1.0) * 0.5  # normalise to 0..1
    return palette_sample(palette_rgb, max(0, min(1, t)))


def _pattern_blocks(nx, ny, palette_rgb, anim, phase, rng):
    """Chunky mosaic of palette colours — very distinct from gradients."""
    block_size = 0.12 + math.sin(phase * 0.5) * 0.03
    bx = int((nx + anim * 0.15) / block_size) % 17
    by = int((ny + anim * 0.1) / block_size) % 13
    idx = (bx * 7 + by * 3) % len(palette_rgb)
    return palette_rgb[idx]


def _pattern_stripes(nx, ny, palette_rgb, anim, phase, rng):
    """Hard-edged alternating bands — not smooth gradients."""
    stripe_width = 0.08
    pos = ny + anim * 0.2
    stripe_idx = int(pos / stripe_width) % len(palette_rgb)
    return palette_rgb[stripe_idx]


def _pattern_scatter(nx, ny, palette_rgb, anim, phase, rng):
    """Sparse random dots on a solid background."""
    bg = palette_rgb[0]
    # Deterministic scatter based on position
    h = hash((int(nx * 20), int(ny * 15), int(anim * 10))) % 100
    if h < 20:
        return palette_rgb[h % len(palette_rgb)]
    return bg


def _pattern_solid(nx, ny, palette_rgb, anim, phase, rng):
    """Single colour fill — for negative space and contrast."""
    idx = int(anim * len(palette_rgb)) % len(palette_rgb)
    return palette_rgb[idx]


def _pattern_vignette(nx, ny, palette_rgb, anim, phase, rng):
    """Bright center, dark edges."""
    cx, cy = 0.5, 0.5
    dist = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2) * 1.4
    dist = min(1.0, dist)
    inner = palette_rgb[len(palette_rgb) // 2]
    outer = palette_rgb[0]
    return lerp_color(inner, outer, dist)


def _pattern_plasma(nx, ny, palette_rgb, anim, phase, rng):
    """Multi-frequency sine interference — classic demo scene look."""
    v = 0.0
    v += math.sin(nx * 10.0 + phase * 1.5)
    v += math.sin(ny * 8.0 - phase * 1.2)
    v += math.sin((nx + ny) * 6.0 + anim * 4.0)
    v += math.sin(math.sqrt(((nx - 0.5) * 10) ** 2 + ((ny - 0.5) * 8) ** 2) + phase)
    t = (v / 4.0 + 1.0) * 0.5  # normalise
    return palette_sample(palette_rgb, max(0, min(1, t)))


def _pattern_checkerboard(nx, ny, palette_rgb, anim, phase, rng):
    """Alternating palette squares — grid-like mosaic."""
    cell_size = 0.1 + math.sin(phase * 0.3) * 0.02
    cx = int((nx + anim * 0.1) / cell_size)
    cy = int((ny + anim * 0.05) / cell_size)
    idx = (cx + cy) % len(palette_rgb)
    return palette_rgb[idx]


def _pattern_radial(nx, ny, palette_rgb, anim, phase, rng):
    """Concentric rings emanating from center."""
    cx, cy = 0.5 + math.sin(phase) * 0.1, 0.5 + math.cos(phase) * 0.1
    dist = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
    ring_val = (dist * 8.0 + anim * 3.0) % 1.0
    return palette_sample(palette_rgb, ring_val)


def _pattern_diagonal(nx, ny, palette_rgb, anim, phase, rng):
    """Angled bands — diagonal stripes."""
    angle = 0.7 + math.sin(phase * 0.5) * 0.3  # ~45° with wobble
    pos = nx * math.cos(angle) + ny * math.sin(angle)
    band_width = 0.1
    idx = int((pos + anim * 0.2) / band_width) % len(palette_rgb)
    return palette_rgb[idx]


def _pattern_wave(nx, ny, palette_rgb, anim, phase, rng):
    """Sinusoidal colour waves — flowing organic look."""
    wave = math.sin(nx * 8.0 + phase * 2.0) * 0.15
    wave += math.sin(ny * 6.0 - phase * 1.5) * 0.1
    t = (ny + wave + anim * 0.3) % 1.0
    return palette_sample(palette_rgb, max(0, min(1, t)))


PATTERN_FUNCS = {
    "gradient": _pattern_gradient,
    "noise": _pattern_noise,
    "blocks": _pattern_blocks,
    "stripes": _pattern_stripes,
    "scatter": _pattern_scatter,
    "solid": _pattern_solid,
    "vignette": _pattern_vignette,
    "plasma": _pattern_plasma,
    "checkerboard": _pattern_checkerboard,
    "radial": _pattern_radial,
    "diagonal": _pattern_diagonal,
    "wave": _pattern_wave,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ACCENT EFFECTS — high-impact beat-reactive overlays
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_accents(grid, log_w, log_h, accents, audio_feat, phase, anim, palette_rgb):
    """Apply accent effects in-place on the grid."""
    loud = audio_feat.get("loudness", 0.5)
    onset = audio_feat.get("onset", 0.0)
    is_beat = audio_feat.get("is_beat", False)

    for accent in accents:
        if accent == "ripple" and is_beat:
            # Expanding ring from center on beats
            radius = anim * 0.5
            ring_w = 0.04
            accent_col = palette_rgb[-1] if palette_rgb else (255, 255, 255)
            for row in range(log_h):
                ny = row / max(1, log_h - 1)
                for col in range(log_w):
                    nx = col / max(1, log_w - 1)
                    dist = math.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
                    if abs(dist - radius) < ring_w:
                        grid[row][col] = lerp_color(grid[row][col], accent_col, 0.7)

        elif accent == "scanline":
            # Bright horizontal line sweeping down
            scan_y = int((anim * 1.5) % 1.0 * log_h)
            if 0 <= scan_y < log_h:
                accent_col = palette_rgb[-1] if palette_rgb else (255, 255, 255)
                for col in range(log_w):
                    grid[scan_y][col] = lerp_color(grid[scan_y][col], accent_col, 0.6 + loud * 0.3)

        elif accent == "flash" and is_beat and onset > 0.5:
            # Brief palette inversion on strong beats
            for row in range(log_h):
                for col in range(log_w):
                    r, g, b = grid[row][col]
                    grid[row][col] = (255 - r, 255 - g, 255 - b)

        elif accent == "column_bars":
            # Vertical bars rising from bottom, height tracks loudness
            n_bars = min(log_w, 16)
            bar_w = max(1, log_w // n_bars)
            bar_h = int(loud * log_h * 0.8)
            accent_col = palette_rgb[len(palette_rgb) // 2] if palette_rgb else (200, 200, 200)
            for bi in range(n_bars):
                cx = bi * bar_w + bar_w // 2
                for row in range(max(0, log_h - bar_h), log_h):
                    for col in range(max(0, cx - bar_w // 2), min(log_w, cx + bar_w // 2)):
                        grid[row][col] = lerp_color(grid[row][col], accent_col, 0.5)

        elif accent == "shake" and onset > 0.6:
            # Random offset on onset spikes — shift entire grid
            dx = int(onset * 2)
            shifted = [row[:] for row in grid]
            for row in range(log_h):
                for col in range(log_w):
                    src_col = (col - dx) % log_w
                    grid[row][col] = shifted[row][src_col]


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE-BASED FRAME RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_procedural_frame(scene, log_w, log_h, frame_progress, audio_feat, section_seed):
    """Render a full-coverage RGB grid using spatial zones.

    Each zone has its own palette and pattern function, producing
    structurally distinct visuals per section.

    Args:
        scene: dict from generate_scene_description (zone-based)
        log_w, log_h: logical grid dimensions
        frame_progress: 0.0 → 1.0 within this section
        audio_feat: dict from TrackAnalyzer.get_frame()
        section_seed: int for deterministic randomness per section

    Returns:
        2D list of (r, g, b) tuples, shape [log_h][log_w]
    """
    rng = np.random.default_rng(section_seed + int(frame_progress * 10000))
    motion = scene.get("motion", "pulse")
    zones = scene.get("zones", [])
    accents = scene.get("accents", [])

    # Collect a "master palette" for accents (union of first colours from each zone)
    master_palette = []
    for z in zones:
        pal = [hex_to_rgb(h) for h in z.get("palette", ["#FF6B35"])]
        master_palette.extend(pal)
    if not master_palette:
        master_palette = [(255, 107, 53)]

    # Audio features
    loud = audio_feat.get("loudness", 0.5)
    onset = audio_feat.get("onset", 0.0)
    bright = audio_feat.get("brightness", 0.5)
    chroma = audio_feat.get("chroma", 0)
    flux = audio_feat.get("flux", 0.0)
    is_beat = audio_feat.get("is_beat", False)

    # Hue shift from chroma (0-11 → 0-1 rotation)
    hue_shift = chroma / 12.0

    # Phase for animation
    phase = frame_progress * math.pi * 2
    if motion == "pulse":
        anim = math.sin(phase * 2) * 0.5 + 0.5
    elif motion == "drift":
        anim = frame_progress
    elif motion == "rotate":
        anim = (frame_progress * 3) % 1.0
    elif motion == "ripple":
        anim = math.sin(phase * 4) * 0.5 + 0.5
    elif motion == "bounce":
        anim = abs(math.sin(phase * 3))
    elif motion == "shimmer":
        anim = (math.sin(phase * 6) * 0.3 + math.sin(phase * 2.7) * 0.7) * 0.5 + 0.5
    elif motion == "cascade":
        anim = (frame_progress * 5) % 1.0
    elif motion == "breathe":
        anim = (math.sin(phase) * 0.5 + 0.5) ** 2
    else:
        anim = frame_progress

    # Beat flash
    beat_flash = 1.3 if is_beat else 1.0
    onset_boost = 1.0 + onset * 0.3

    # Pre-parse zones: palette RGBs, y/x ranges, pattern functions
    parsed_zones = []
    for z in zones:
        pal_rgb = [hex_to_rgb(h) for h in z.get("palette", ["#FF6B35", "#004E89"])]
        y_range = z.get("y", [0.0, 1.0])
        x_range = z.get("x", [0.0, 1.0])
        pattern_name = z.get("pattern", "gradient")
        pattern_fn = PATTERN_FUNCS.get(pattern_name, _pattern_gradient)
        parsed_zones.append((pal_rgb, y_range, x_range, pattern_fn))

    # Render grid — zones paint back-to-front (later zones override earlier)
    # Start with first zone's base colour as background
    bg_col = parsed_zones[0][0][0] if parsed_zones else (64, 64, 64)
    grid = [[bg_col for _ in range(log_w)] for _ in range(log_h)]

    for zi, (pal_rgb, y_range, x_range, pattern_fn) in enumerate(parsed_zones):
        y0, y1 = y_range[0], y_range[1]
        x0, x1 = x_range[0], x_range[1]

        # Zone breathing — shift zone boundaries over time for organic movement
        breath_amount = 0.06 * (0.5 + loud * 0.5)  # scale by loudness
        breath_offset = math.sin(phase + zi * 1.7) * breath_amount
        y0 = max(0.0, min(1.0, y0 + breath_offset))
        y1 = max(0.0, min(1.0, y1 + breath_offset))
        x_breath = math.cos(phase + zi * 2.3) * breath_amount * 0.5
        x0 = max(0.0, min(1.0, x0 + x_breath))
        x1 = max(0.0, min(1.0, x1 + x_breath))

        # Pixel range for this zone
        row_start = max(0, int(y0 * log_h))
        row_end = min(log_h, int(y1 * log_h))
        col_start = max(0, int(x0 * log_w))
        col_end = min(log_w, int(x1 * log_w))

        for row in range(row_start, row_end):
            # Normalised position WITHIN the zone (0..1)
            zone_ny = (row - row_start) / max(1, row_end - row_start - 1)
            for col in range(col_start, col_end):
                zone_nx = (col - col_start) / max(1, col_end - col_start - 1)
                grid[row][col] = pattern_fn(zone_nx, zone_ny, pal_rgb, anim, phase, rng)

    # ── Apply accent effects ─────────────────────────────────────────────
    _apply_accents(grid, log_w, log_h, accents, audio_feat, phase, anim, master_palette)

    # ── Post-processing: hue shift + audio reactivity + vibrance floor ───
    for row in range(log_h):
        for col in range(log_w):
            r, g, b = grid[row][col]

            # Single HSV pass: hue shift + brightness + saturation floor
            h_val, s_val, v_val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h_val = (h_val + hue_shift * 0.12) % 1.0
            v_val = min(1.0, v_val * (0.6 + loud * 0.4) * beat_flash * onset_boost)
            s_val = min(1.0, s_val * (0.8 + bright * 0.2))
            # Vibrance floors — prevent desaturation
            s_val = max(0.4, s_val)
            v_val = max(0.25, v_val)
            rf, gf, bf = colorsys.hsv_to_rgb(h_val, s_val, v_val)
            r, g, b = int(rf * 255), int(gf * 255), int(bf * 255)

            # Flux-driven noise
            if flux > 0.4:
                noise = rng.integers(-15, 15, size=3)
                r = max(0, min(255, r + int(noise[0] * flux)))
                g = max(0, min(255, g + int(noise[1] * flux)))
                b = max(0, min(255, b + int(noise[2] * flux)))

            grid[row][col] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# CHARACTER GRID CONVERSION (kept from original)
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_char_cell(r, g, b):
    """Convert an RGB pixel to a 7-tuple unicode character cell.
    Returns (char, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b)
    """
    bg_r, bg_g, bg_b = r, g, b

    brightness = (r + g + b) / (3.0 * 255.0)
    char_idx = int(brightness * (len(DENSITY_CHARS) - 1))
    char = DENSITY_CHARS[char_idx]

    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if s < 0.05:
        v_offset = 0.3 * (1.0 - v) if v < 0.5 else -0.3 * v
        v = max(0.0, min(1.0, v + v_offset))
    else:
        v_offset = 0.3 * (1.0 - v) - 0.2 * v
        s_offset = 0.2 * (1.0 - s)
        v = max(0.0, min(1.0, v + v_offset))
        s = max(0.0, min(1.0, s + s_offset))

    fg_r_f, fg_g_f, fg_b_f = colorsys.hsv_to_rgb(h, s, v)
    fg_r = int(fg_r_f * 255)
    fg_g = int(fg_g_f * 255)
    fg_b = int(fg_b_f * 255)

    return (char, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b)


def logical_to_char_grid(rgb_grid, log_w, log_h, char_w, char_h):
    """Upscale a logical RGB grid to a full character grid with unicode cells."""
    grid = []
    for row in range(char_h):
        line = []
        ly = min(int(row * log_h / char_h), log_h - 1)
        for col in range(char_w):
            lx = min(int(col * log_w / char_w), log_w - 1)
            r, g, b = rgb_grid[ly][lx]
            line.append(rgb_to_char_cell(r, g, b))
        grid.append(line)
    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# STORY BEATS (LLM screenplay — maps to sections)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_story_beats(text_engine, prompt_concept, bpm, duration, sonic_map, num_sections):
    """Generate story beats via TextEngine, one per detected section."""
    story_req = (
        f"Create a VISUAL SCREENPLAY for a {duration:.0f}s music video (Animated).\n"
        f"Concept: {prompt_concept}\n"
        f"Music Vibe: {bpm:.0f} BPM.\n"
        f"Audio Profile (Energy/Mood over time): {sonic_map}\n"
        f"Constraints: We need exactly {num_sections} distinct visual scenes, one per musical section.\n"
        f"Critical: Match the visual intensity to the Audio Profile.\n"
        'Output JSON: { "title": "...", "synopsis": "...", "beats": ["Scene 1 description", "Scene 2", ...] }'
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logging.info(f"   Calling Writer (Attempt {attempt+1}/{max_retries})...")
            raw = text_engine.generate(story_req, json_schema=True)
            story_data = json.loads(raw)
            if isinstance(story_data, list):
                story_data = story_data[0]
            if 'beats' not in story_data:
                raise ValueError("Missing 'beats' in JSON.")
            # Pad or trim to match num_sections
            beats = story_data['beats']
            while len(beats) < num_sections:
                beats.append(beats[-1] if beats else "Abstract colourful animation")
            beats = beats[:num_sections]
            story_data['beats'] = beats
            logging.info(f"   Generated {len(beats)} story beats for {num_sections} sections.")
            return story_data
        except Exception as e:
            logging.warning(f"   Writer Attempt {attempt+1} Failed: {e}")
            if attempt == max_retries - 1:
                # Return generic beats
                logging.warning("   Using generic fallback beats")
                return {
                    "title": "Untitled",
                    "synopsis": prompt_concept,
                    "beats": [f"Scene {i+1}: {prompt_concept}" for i in range(num_sections)]
                }
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Direct Text-to-Unicode-Art Video Pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Creative concept for the video")
    parser.add_argument("--mu", type=str, required=True, help="Path to music/audio file")
    parser.add_argument("--style", type=str, default="abstract animated artwork",
                        help="Visual style description")
    parser.add_argument("--w", type=int, default=576,
                        help="Output width in character cells")
    parser.add_argument("--h", type=int, default=None,
                        help="Output height in character cells (auto from aspect if omitted)")
    parser.add_argument("--fsync", type=float, default=0.5,
                        help="BPM sync multiplier (lower = fewer frames)")
    parser.add_argument("--bpm", type=float, default=None, help="Manual BPM override")
    args = parser.parse_args()

    if not os.path.exists(args.mu):
        logging.error(f"Audio file not found: {args.mu}")
        return

    audio_path = args.mu
    audio_stem = Path(audio_path).stem

    # Project directory (mirrors cartoon_producer's z_test-outputs/ convention)
    project_dir = Path("z_test-outputs") / "unicodes" / audio_stem
    frames_dir = project_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Audio analysis via TrackAnalyzer ──────────────────────────────────
    logging.info("Step 1: Analyzing audio with TrackAnalyzer...")

    if args.bpm:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        bpm = args.bpm
        bps = bpm / 60.0
        fps = (bps * 4) * args.fsync
    else:
        bpm, duration, _fpb, fps = analyze_audio(audio_path, fsync=args.fsync)

    target_frames = int(duration * fps)
    logging.info(f"   BPM: {bpm:.0f} | Duration: {duration:.1f}s | FPS: {fps:.2f} | Frames: {target_frames}")

    # TrackAnalyzer for section detection + per-frame audio features
    analyzer = TrackAnalyzer(audio_path, fps=int(round(fps)))
    sections = analyzer.section_boundaries
    num_sections = len(sections)
    logging.info(f"   Sections detected: {num_sections}")

    # Build section ranges: [(start_frame, end_frame), ...]
    section_ranges = []
    for i in range(num_sections):
        start = int(sections[i])
        end = int(sections[i + 1]) if i + 1 < num_sections else target_frames
        section_ranges.append((start, end))
    logging.info(f"   Section ranges: {[(s, e) for s, e in section_ranges]}")

    # Character grid dimensions
    char_w = args.w
    if args.h:
        char_h = args.h
    else:
        aspect = 9.0 / 16.0
        char_h = max(20, int(char_w * aspect * (CELL_W / CELL_H)))

    # Logical grid (for procedural rendering)
    log_w = min(MAX_LOG_W, char_w)
    log_h = min(MAX_LOG_H, char_h)
    while log_w * log_h > 3072:  # Keep grid manageable for rendering speed
        log_w = max(16, log_w - 4)
        log_h = max(12, log_h - 3)

    logging.info(f"   Canvas: {char_w}x{char_h} chars -> {char_w * CELL_W}x{char_h * CELL_H}px output")
    logging.info(f"   Logical grid: {log_w}x{log_h} ({log_w * log_h} pixels)")

    # ── 2. Story beats (one per section) ─────────────────────────────────────
    logging.info("Step 2: Generating story beats...")
    text_engine = TextEngine()

    # Audio profile for story context
    try:
        sonic_map = analyze_audio_profile(audio_path, duration)
    except Exception:
        sonic_map = "Audio profile unavailable"

    story_data = generate_story_beats(text_engine, args.prompt, bpm, duration, sonic_map, num_sections)
    beats = story_data.get('beats', [])
    logging.info(f"   Title: {story_data.get('title', 'Untitled')}")
    for i, beat in enumerate(beats):
        logging.info(f"   Beat {i+1}: {beat[:80]}{'...' if len(beat) > 80 else ''}")

    # ── 3. Scene descriptions (one LLM call per section) ─────────────────────
    logging.info("Step 3: Generating scene descriptions...")
    scene_descriptions = []
    prev_mood = None
    for i, beat in enumerate(beats):
        scene_prompt = f"Style: {args.style}. Scene: {beat}"
        scene = generate_scene_description(text_engine, scene_prompt, i, prev_mood=prev_mood)
        scene_descriptions.append(scene)
        prev_mood = scene.get("mood", None)

    logging.info(f"   Generated {len(scene_descriptions)} scene descriptions")

    # ── 4. Load fonts ────────────────────────────────────────────────────────
    logging.info("Step 4: Loading fonts...")
    fonts = load_multi_font()

    # ── 5. Generate frames ───────────────────────────────────────────────────
    logging.info(f"Step 5: Generating {target_frames} frames...")
    frames_generated = 0
    frames_skipped = 0

    for sec_idx, (sec_start, sec_end) in enumerate(section_ranges):
        scene = scene_descriptions[sec_idx] if sec_idx < len(scene_descriptions) else scene_descriptions[-1]
        sec_len = max(1, sec_end - sec_start)
        section_seed = int(hashlib.md5(f"{args.prompt}:{sec_idx}".encode()).hexdigest()[:8], 16)

        for frame_idx in range(sec_start, min(sec_end, target_frames)):
            frame_num = frame_idx + 1
            dst = frames_dir / f"frame_{frame_num:04d}.png"

            # Resume support
            if dst.exists():
                frames_skipped += 1
                continue

            # Frame progress within this section (0..1)
            frame_progress = (frame_idx - sec_start) / max(1, sec_len - 1)

            # Audio features for this frame
            audio_feat = analyzer.get_frame(frame_idx)

            # Procedural render
            rgb_grid = render_procedural_frame(
                scene, log_w, log_h, frame_progress, audio_feat, section_seed
            )

            # Convert to unicode character grid (upscale)
            char_grid = logical_to_char_grid(rgb_grid, log_w, log_h, char_w, char_h)

            # Render to image
            out_img = grid_to_image(char_grid, char_w, char_h, fonts)
            out_img.save(dst)
            frames_generated += 1

            if frame_num == 1 or frame_num % 25 == 0 or frame_num == target_frames:
                pct = frame_num / target_frames * 100
                logging.info(f"   [{sec_idx+1}/{num_sections}] Frame {frame_num}/{target_frames} ({pct:.0f}%)")

    if frames_skipped > 0:
        logging.info(f"   Skipped {frames_skipped} existing frames, generated {frames_generated} new frames")

    # ── 6. Stitch video ─────────────────────────────────────────────────────
    logging.info("Step 6: Stitching video with ffmpeg...")
    out_vid = project_dir / f"{audio_stem}_unicode_video.mp4"
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-i', audio_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        str(out_vid)
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"   Done: {out_vid}")


if __name__ == "__main__":
    main()
