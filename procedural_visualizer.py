#!/usr/bin/env python3
"""
procedural_visualizer.py — Audio-Reactive Procedural Visualizer
================================================================
Renders abstract, shapes-and-patterns music visualizations at full pixel
resolution, driven by real-time spectral audio features.  Adapted from
unicode_video.py's zone/pattern/accent system, but outputs directly to
PIL Images instead of character grids.

Used by cartoon_producer.py when vpform=music-visualizer and --local.

Key differences from Flux-based rendering:
  - Pure procedural math (numpy/scipy) — no ML model, no GPU
  - Genuinely abstract: plasma, interference, spirals, radial bursts
  - Audio-reactive: loudness, onset, beat, chroma, spectral flux drive
    every pixel every frame
  - Fast: ~0.1s per frame vs ~8s for Flux
  - LLM scene descriptions (1 per section) guide palette/pattern/mood
"""

import os
import math
import time
import logging
import hashlib
import colorsys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_W = 512
DEFAULT_H = 288

# Fallback palettes (vivid, saturated)
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

ZONE_PATTERNS = [
    "gradient", "noise", "blocks", "stripes",
    "scatter", "solid", "vignette", "plasma",
    "checkerboard", "radial", "diagonal", "wave",
    "spiral", "interference", "particlefield",
]

FALLBACK_ZONE_TEMPLATES = [
    [{"name": "sky",    "y": [0.0, 0.45], "pattern": "plasma"},
     {"name": "horizon","y": [0.40, 0.60], "pattern": "interference"},
     {"name": "ground", "y": [0.55, 1.0],  "pattern": "noise"}],
    [{"name": "bg",     "y": [0.0, 1.0],  "pattern": "spiral"},
     {"name": "subject","y": [0.2, 0.8],  "x": [0.2, 0.8], "pattern": "radial"}],
    [{"name": "field",  "y": [0.0, 1.0],  "pattern": "plasma"},
     {"name": "accent", "y": [0.7, 1.0],  "pattern": "particlefield"}],
    [{"name": "left",   "y": [0.0, 1.0],  "x": [0.0, 0.5], "pattern": "interference"},
     {"name": "right",  "y": [0.0, 1.0],  "x": [0.5, 1.0], "pattern": "spiral"}],
]


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) >= 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (128, 128, 128)


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def palette_sample(palette_rgb, t):
    t = max(0.0, min(1.0, t))
    n = len(palette_rgb)
    if n == 0: return (128, 128, 128)
    if n == 1: return palette_rgb[0]
    idx = t * (n - 1)
    i = int(idx)
    frac = idx - i
    if i >= n - 1: return palette_rgb[-1]
    return lerp_color(palette_rgb[i], palette_rgb[i + 1], frac)


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN FUNCTIONS — vectorized for full-resolution rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _pattern_gradient(nx, ny, palette_rgb, anim, phase, rng):
    t = (ny + anim * 0.3) % 1.0
    return palette_sample(palette_rgb, t)

def _pattern_noise(nx, ny, palette_rgb, anim, phase, rng):
    val = 0.0
    val += math.sin(nx * 7.3 + anim * 2.1) * 0.35
    val += math.sin(ny * 5.7 - anim * 1.8) * 0.35
    val += math.sin((nx + ny) * 4.1 + phase * 0.5) * 0.2
    val += math.sin((nx * 3.3 - ny * 2.7) + anim * 3.0) * 0.1
    t = (val + 1.0) * 0.5
    return palette_sample(palette_rgb, max(0, min(1, t)))

def _pattern_blocks(nx, ny, palette_rgb, anim, phase, rng):
    block_size = 0.12 + math.sin(phase * 0.5) * 0.03
    bx = int((nx + anim * 0.15) / block_size) % 17
    by = int((ny + anim * 0.1) / block_size) % 13
    idx = (bx * 7 + by * 3) % len(palette_rgb)
    return palette_rgb[idx]

def _pattern_stripes(nx, ny, palette_rgb, anim, phase, rng):
    stripe_width = 0.08
    pos = ny + anim * 0.2
    stripe_idx = int(pos / stripe_width) % len(palette_rgb)
    return palette_rgb[stripe_idx]

def _pattern_scatter(nx, ny, palette_rgb, anim, phase, rng):
    bg = palette_rgb[0]
    h = hash((int(nx * 20), int(ny * 15), int(anim * 10))) % 100
    if h < 20:
        return palette_rgb[h % len(palette_rgb)]
    return bg

def _pattern_solid(nx, ny, palette_rgb, anim, phase, rng):
    idx = int(anim * len(palette_rgb)) % len(palette_rgb)
    return palette_rgb[idx]

def _pattern_vignette(nx, ny, palette_rgb, anim, phase, rng):
    cx, cy = 0.5, 0.5
    dist = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2) * 1.4
    dist = min(1.0, dist)
    inner = palette_rgb[len(palette_rgb) // 2]
    outer = palette_rgb[0]
    return lerp_color(inner, outer, dist)

def _pattern_plasma(nx, ny, palette_rgb, anim, phase, rng):
    v = 0.0
    v += math.sin(nx * 10.0 + phase * 1.5)
    v += math.sin(ny * 8.0 - phase * 1.2)
    v += math.sin((nx + ny) * 6.0 + anim * 4.0)
    v += math.sin(math.sqrt(((nx - 0.5) * 10) ** 2 + ((ny - 0.5) * 8) ** 2) + phase)
    t = (v / 4.0 + 1.0) * 0.5
    return palette_sample(palette_rgb, max(0, min(1, t)))

def _pattern_checkerboard(nx, ny, palette_rgb, anim, phase, rng):
    cell_size = 0.1 + math.sin(phase * 0.3) * 0.02
    cx = int((nx + anim * 0.1) / cell_size)
    cy = int((ny + anim * 0.05) / cell_size)
    idx = (cx + cy) % len(palette_rgb)
    return palette_rgb[idx]

def _pattern_radial(nx, ny, palette_rgb, anim, phase, rng):
    cx = 0.5 + math.sin(phase) * 0.1
    cy = 0.5 + math.cos(phase) * 0.1
    dist = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
    ring_val = (dist * 8.0 + anim * 3.0) % 1.0
    return palette_sample(palette_rgb, ring_val)

def _pattern_diagonal(nx, ny, palette_rgb, anim, phase, rng):
    angle = 0.7 + math.sin(phase * 0.5) * 0.3
    pos = nx * math.cos(angle) + ny * math.sin(angle)
    band_width = 0.1
    idx = int((pos + anim * 0.2) / band_width) % len(palette_rgb)
    return palette_rgb[idx]

def _pattern_wave(nx, ny, palette_rgb, anim, phase, rng):
    wave = math.sin(nx * 8.0 + phase * 2.0) * 0.15
    wave += math.sin(ny * 6.0 - phase * 1.5) * 0.1
    t = (ny + wave + anim * 0.3) % 1.0
    return palette_sample(palette_rgb, max(0, min(1, t)))

# --- NEW PATTERNS for music-visualizer ---

def _pattern_spiral(nx, ny, palette_rgb, anim, phase, rng):
    """Logarithmic spiral with palette sweep — classic visualizer look."""
    cx, cy = 0.5, 0.5
    dx, dy = nx - cx, ny - cy
    angle = math.atan2(dy, dx)
    dist = math.sqrt(dx*dx + dy*dy)
    # Log spiral: r = a * e^(b*theta)
    spiral_val = (angle / (2 * math.pi) + dist * 5.0 + anim * 2.0) % 1.0
    # Add twist
    twist = math.sin(dist * 12.0 - phase * 3.0) * 0.2
    t = (spiral_val + twist + 1.0) * 0.5
    return palette_sample(palette_rgb, max(0, min(1, t % 1.0)))

def _pattern_interference(nx, ny, palette_rgb, anim, phase, rng):
    """Moiré interference — overlapping concentric wave sources."""
    # Three wave sources at different positions
    sources = [
        (0.3 + math.sin(phase * 0.7) * 0.15, 0.3 + math.cos(phase * 0.5) * 0.1),
        (0.7 + math.cos(phase * 0.6) * 0.12, 0.5 + math.sin(phase * 0.8) * 0.15),
        (0.5 + math.sin(phase * 0.9) * 0.1, 0.7 + math.cos(phase * 0.4) * 0.12),
    ]
    val = 0.0
    for sx, sy in sources:
        dist = math.sqrt((nx - sx)**2 + (ny - sy)**2)
        val += math.sin(dist * 30.0 - phase * 2.5)
    t = (val / 3.0 + 1.0) * 0.5
    return palette_sample(palette_rgb, max(0, min(1, t)))

def _pattern_particlefield(nx, ny, palette_rgb, anim, phase, rng):
    """Simulated particle positions — dots/blobs moving through space."""
    # Generate deterministic particle positions based on anim phase
    val = 0.0
    n_particles = 8
    for i in range(n_particles):
        # Each particle orbits with unique freq
        px = 0.5 + math.sin(phase * (0.5 + i * 0.3) + i * 1.7) * 0.4
        py = 0.5 + math.cos(phase * (0.4 + i * 0.2) + i * 2.3) * 0.4
        dist = math.sqrt((nx - px)**2 + (ny - py)**2)
        # Soft blob falloff
        blob = max(0.0, 1.0 - dist * 8.0)
        blob = blob * blob  # Quadratic falloff
        val += blob * (1.0 + math.sin(i * 0.7 + anim * 2.0) * 0.3)
    t = min(1.0, val)
    return palette_sample(palette_rgb, t)


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
    "spiral": _pattern_spiral,
    "interference": _pattern_interference,
    "particlefield": _pattern_particlefield,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ACCENT EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_accents_array(img_arr, accents, audio_feat, phase, anim, palette_rgb):
    """Apply accent effects to a numpy image array (H, W, 3) in uint8."""
    h, w = img_arr.shape[:2]
    loud = audio_feat.get("loudness", 0.5)
    onset = audio_feat.get("onset", 0.0)
    is_beat = audio_feat.get("is_beat", False)

    for accent in accents:
        if accent == "ripple" and is_beat:
            radius = (anim * 0.5) % 0.5
            ring_w_px = max(2, int(min(h, w) * 0.02))
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / max(h, w)
            mask = np.abs(dist - radius) < (ring_w_px / max(h, w))
            if palette_rgb:
                col = np.array(palette_rgb[-1], dtype=np.uint8)
            else:
                col = np.array([255, 255, 255], dtype=np.uint8)
            img_arr[mask] = (img_arr[mask].astype(np.float32) * 0.3 + col.astype(np.float32) * 0.7).astype(np.uint8)

        elif accent == "scanline":
            scan_y = int((anim * 1.5) % 1.0 * h)
            if 0 <= scan_y < h:
                blend = 0.6 + loud * 0.3
                if palette_rgb:
                    col = np.array(palette_rgb[-1], dtype=np.float32)
                else:
                    col = np.array([255, 255, 255], dtype=np.float32)
                img_arr[scan_y] = (img_arr[scan_y].astype(np.float32) * (1 - blend) + col * blend).astype(np.uint8)

        elif accent == "flash" and is_beat and onset > 0.5:
            img_arr[:] = 255 - img_arr

        elif accent == "column_bars":
            n_bars = min(w, 16)
            bar_w_px = max(1, w // n_bars)
            bar_h_px = int(loud * h * 0.8)
            if palette_rgb:
                col = np.array(palette_rgb[len(palette_rgb) // 2], dtype=np.float32)
            else:
                col = np.array([200, 200, 200], dtype=np.float32)
            for bi in range(n_bars):
                cx = bi * bar_w_px + bar_w_px // 2
                x0 = max(0, cx - bar_w_px // 2)
                x1 = min(w, cx + bar_w_px // 2)
                y0 = max(0, h - bar_h_px)
                region = img_arr[y0:h, x0:x1].astype(np.float32)
                img_arr[y0:h, x0:x1] = (region * 0.5 + col * 0.5).astype(np.uint8)

        elif accent == "shake" and onset > 0.6:
            dx = int(onset * 4)
            img_arr[:] = np.roll(img_arr, dx, axis=1)

    return img_arr


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE DESCRIPTION (LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scene_description(text_engine, section_prompt, section_idx, retries=2, prev_mood=None):
    """Ask the LLM for a zone-based scene description for one section.
    Falls back to deterministic defaults on failure.
    """
    _EXAMPLES = [
        '{\n'
        '  "zones": [\n'
        '    {"name": "left_field", "y": [0.0, 1.0], "x": [0.0, 0.55], "palette": ["#2B1B4E","#5533AA","#8855CC"], "pattern": "plasma"},\n'
        '    {"name": "right_field", "y": [0.0, 1.0], "x": [0.45, 1.0], "palette": ["#FF6B35","#FFD700","#FF3366"], "pattern": "spiral"},\n'
        '    {"name": "accent_band", "y": [0.35, 0.65], "palette": ["#00FFAA","#00AAFF"], "pattern": "interference"}\n'
        '  ],\n'
        '  "accents": ["flash", "shake"],\n'
        '  "mood": "electric",\n'
        '  "motion": "cascade"\n'
        '}',
        '{\n'
        '  "zones": [\n'
        '    {"name": "atmosphere", "y": [0.0, 1.0], "palette": ["#0D0221","#150734","#261447"], "pattern": "interference"},\n'
        '    {"name": "subject", "y": [0.15, 0.85], "x": [0.2, 0.8], "palette": ["#FF006E","#FB5607","#FFBE0B"], "pattern": "radial"},\n'
        '    {"name": "floor_glow", "y": [0.75, 1.0], "palette": ["#3A86FF","#8338EC"], "pattern": "particlefield"}\n'
        '  ],\n'
        '  "accents": ["ripple", "column_bars"],\n'
        '  "mood": "psychedelic",\n'
        '  "motion": "breathe"\n'
        '}',
        '{\n'
        '  "zones": [\n'
        '    {"name": "col_left", "y": [0.0, 1.0], "x": [0.0, 0.33], "palette": ["#F72585","#B5179E","#7209B7"], "pattern": "spiral"},\n'
        '    {"name": "col_center", "y": [0.0, 1.0], "x": [0.28, 0.72], "palette": ["#06D6A0","#118AB2","#073B4C"], "pattern": "interference"},\n'
        '    {"name": "col_right", "y": [0.0, 1.0], "x": [0.67, 1.0], "palette": ["#FFD166","#EF476F","#26547C"], "pattern": "particlefield"}\n'
        '  ],\n'
        '  "accents": ["scanline"],\n'
        '  "mood": "chaotic",\n'
        '  "motion": "shimmer"\n'
        '}',
    ]
    example = _EXAMPLES[section_idx % len(_EXAMPLES)]

    contrast_hint = ""
    if prev_mood:
        contrast_hint = (
            f"\nIMPORTANT: The previous section used a '{prev_mood}' mood. "
            "Make THIS section visually DIFFERENT — contrasting composition, palette family, and energy level.\n"
        )

    prompt = (
        f"Design an ABSTRACT VISUAL SCENE for section {section_idx + 1} of a MUSIC VISUALIZER.\n"
        f"Scene concept: {section_prompt}\n\n"
        "The scene is built from 2-4 SPATIAL ZONES — distinct visual regions.\n"
        "Each zone has its own palette and pattern. Zones can overlap.\n\n"
        f"Return ONLY a JSON object (example for inspiration — DO NOT copy this layout):\n{example}\n\n"
        "CRITICAL: Vary the spatial composition.\n"
        "- Try vertical columns (using x ranges), centered inset subjects, diagonal splits, full-field patterns\n"
        "- Use overlapping zones where a smaller zone sits inside a larger background\n\n"
        f"{contrast_hint}"
        "ZONE RULES:\n"
        "- Each zone needs: name, y (vertical range 0-1), palette (2-4 HEX colours), pattern\n"
        "- Optional: x (horizontal range 0-1, defaults to [0,1])\n"
        "- Patterns: gradient, noise, blocks, stripes, scatter, solid, vignette, plasma, "
        "checkerboard, radial, diagonal, wave, spiral, interference, particlefield\n"
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
            valid_zones = []
            for z in zones:
                if not isinstance(z, dict): continue
                pal = z.get("palette", [])
                pal = [p for p in pal if isinstance(p, str) and len(p.lstrip('#')) >= 6]
                if len(pal) < 2: continue
                z["palette"] = pal
                y_range = z.get("y", [0.0, 1.0])
                if not isinstance(y_range, list) or len(y_range) < 2:
                    z["y"] = [0.0, 1.0]
                pattern = z.get("pattern", "gradient")
                if pattern not in PATTERN_FUNCS:
                    z["pattern"] = "plasma"
                valid_zones.append(z)
            if len(valid_zones) < 1:
                raise ValueError("No valid zones")
            patterns_used = set(z.get("pattern", "gradient") for z in valid_zones)
            if patterns_used <= {"solid", "gradient"}:
                valid_zones[0]["pattern"] = "plasma"
                if len(valid_zones) > 1:
                    valid_zones[-1]["pattern"] = "spiral"
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
# HIGH-RESOLUTION FRAME RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_hires_frame(scene, width, height, frame_progress, audio_feat, section_seed):
    """Render a full-resolution procedural frame as a PIL Image.

    Uses 2x supersampling for anti-aliasing, then downsamples + gaussian blur
    for a smooth, organic look (vs. the blocky character-grid in unicode_video).

    Args:
        scene: dict from generate_scene_description (zone-based)
        width, height: output pixel dimensions
        frame_progress: 0.0 → 1.0 within this section
        audio_feat: dict from TrackAnalyzer.get_frame()
        section_seed: int for deterministic randomness per section

    Returns:
        PIL Image (RGB) at width × height
    """
    # Supersampling factor — render at 2x then downsample
    ss = 2
    rw, rh = width * ss, height * ss

    rng = np.random.default_rng(section_seed + int(frame_progress * 10000))
    motion = scene.get("motion", "pulse")
    zones = scene.get("zones", [])
    accents = scene.get("accents", [])

    # Master palette for accents
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

    beat_flash = 1.3 if is_beat else 1.0
    onset_boost = 1.0 + onset * 0.3

    # Pre-parse zones
    parsed_zones = []
    for z in zones:
        pal_rgb = [hex_to_rgb(h) for h in z.get("palette", ["#FF6B35", "#004E89"])]
        y_range = z.get("y", [0.0, 1.0])
        x_range = z.get("x", [0.0, 1.0])
        pattern_name = z.get("pattern", "gradient")
        pattern_fn = PATTERN_FUNCS.get(pattern_name, _pattern_plasma)
        parsed_zones.append((pal_rgb, y_range, x_range, pattern_fn))

    # Allocate render buffer
    img_arr = np.zeros((rh, rw, 3), dtype=np.uint8)

    # Fill with first zone's base colour as background
    if parsed_zones:
        bg_col = parsed_zones[0][0][0]
        img_arr[:, :] = bg_col

    # Render zones — back to front
    for zi, (pal_rgb, y_range, x_range, pattern_fn) in enumerate(parsed_zones):
        y0, y1 = y_range[0], y_range[1]
        x0, x1 = x_range[0], x_range[1]

        # Zone breathing
        breath_amount = 0.06 * (0.5 + loud * 0.5)
        breath_offset = math.sin(phase + zi * 1.7) * breath_amount
        y0 = max(0.0, min(1.0, y0 + breath_offset))
        y1 = max(0.0, min(1.0, y1 + breath_offset))
        x_breath = math.cos(phase + zi * 2.3) * breath_amount * 0.5
        x0 = max(0.0, min(1.0, x0 + x_breath))
        x1 = max(0.0, min(1.0, x1 + x_breath))

        row_start = max(0, int(y0 * rh))
        row_end = min(rh, int(y1 * rh))
        col_start = max(0, int(x0 * rw))
        col_end = min(rw, int(x1 * rw))

        zone_h = max(1, row_end - row_start)
        zone_w = max(1, col_end - col_start)

        for row in range(row_start, row_end):
            zone_ny = (row - row_start) / max(1, zone_h - 1)
            for col in range(col_start, col_end):
                zone_nx = (col - col_start) / max(1, zone_w - 1)
                r, g, b = pattern_fn(zone_nx, zone_ny, pal_rgb, anim, phase, rng)
                img_arr[row, col] = (r, g, b)

    # Post-processing: hue shift + audio reactivity
    # Convert to float for HSV manipulation
    float_arr = img_arr.astype(np.float32) / 255.0

    # Vectorized HSV manipulation using numpy
    # Flatten for colorsys-like operations
    r_ch = float_arr[:, :, 0].ravel()
    g_ch = float_arr[:, :, 1].ravel()
    b_ch = float_arr[:, :, 2].ravel()

    # Simple brightness/saturation adjustment (avoiding per-pixel colorsys calls for speed)
    # Brightness scale: audio-reactive
    brightness_scale = (0.6 + loud * 0.4) * beat_flash * onset_boost
    float_arr = np.clip(float_arr * brightness_scale, 0.0, 1.0)

    # Hue rotation via channel mixing (approximate)
    if hue_shift > 0.01:
        angle = hue_shift * 0.12 * 2 * math.pi
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        r_new = float_arr[:,:,0] * cos_a - float_arr[:,:,1] * sin_a
        g_new = float_arr[:,:,0] * sin_a + float_arr[:,:,1] * cos_a
        float_arr[:,:,0] = np.clip(r_new, 0, 1)
        float_arr[:,:,1] = np.clip(g_new, 0, 1)

    # Flux-driven noise
    if flux > 0.4:
        noise = rng.normal(0, flux * 0.06, float_arr.shape).astype(np.float32)
        float_arr = np.clip(float_arr + noise, 0, 1)

    img_arr = (float_arr * 255).astype(np.uint8)

    # Apply accent effects
    img_arr = _apply_accents_array(img_arr, accents, audio_feat, phase, anim, master_palette)

    # Create PIL Image from supersampled buffer
    img = Image.fromarray(img_arr, 'RGB')

    # Downsample from 2x to target resolution with anti-aliasing
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Light gaussian blur for organic smoothness
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    return img


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — called from cartoon_producer.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_procedural_visualizer(args, project_dir, text_engine, key_cycle=None):
    """Full procedural visualizer pipeline for music-visualizer --local.

    Generates all frames procedurally (no Flux), then returns so
    cartoon_producer can handle stitching.

    Args:
        args: argparse namespace (needs .mu, .prompt, .style, .w, .h, .bpm, .fsync)
        project_dir: Path to project output directory
        text_engine: TextEngine instance for LLM scene descriptions
        key_cycle: API key cycle (unused for local, but may be used for cloud LLM)

    Returns:
        tuple: (frames_dir, project_fps, target_duration, target_frames)
            or None on failure
    """
    from unicode_visualizer import TrackAnalyzer
    from cartoon_producer import analyze_audio, analyze_audio_profile

    audio_path = args.mu
    if not audio_path or not os.path.exists(audio_path):
        logging.error("❌ Procedural Visualizer requires --mu [audio_path]")
        return None

    # --- Resolution ---
    width = args.w if args.w else DEFAULT_W
    height = args.h if args.h else DEFAULT_H
    logging.info(f"   📐 Procedural Visualizer Resolution: {width}×{height}")

    # --- Audio Analysis ---
    if args.bpm and args.bpm > 0:
        bpm = float(args.bpm)
        try:
            import soundfile as sf
            f = sf.SoundFile(audio_path)
            duration = len(f) / f.samplerate
        except:
            import librosa
            duration = librosa.get_duration(path=audio_path)

        bps = bpm / 60.0
        fps = (bps * 4) * (getattr(args, 'fsync', None) or 1.0)
    else:
        bpm, duration, _fpb, fps = analyze_audio(audio_path, fsync=(getattr(args, 'fsync', None) or 1.0))

    target_frames = int(duration * fps)
    project_fps = fps
    target_duration = duration

    logging.info(f"   🎹 BPM: {bpm:.0f} | Duration: {duration:.1f}s | FPS: {fps:.2f} | Target Frames: {target_frames}")

    # --- TrackAnalyzer for spectral features + section detection ---
    logging.info("   🔬 Analyzing audio spectrum (TrackAnalyzer)...")
    analyzer = TrackAnalyzer(audio_path, fps=int(round(fps)))
    sections = analyzer.section_boundaries
    num_sections = len(sections)
    logging.info(f"   📊 Sections detected: {num_sections}")

    section_ranges = []
    for i in range(num_sections):
        start = int(sections[i])
        end = int(sections[i + 1]) if i + 1 < num_sections else target_frames
        section_ranges.append((start, end))

    # --- Audio Profile for story context ---
    try:
        sonic_map = analyze_audio_profile(audio_path, duration)
    except:
        sonic_map = "Audio profile unavailable"

    # --- Story Beats (one per section) ---
    logging.info("   ✍️  Generating story beats for scene descriptions...")
    prompt_concept = args.prompt if args.prompt else "Abstract cosmic energy"

    story_req = (
        f"Create a VISUAL CONCEPT for a {duration:.0f}s abstract music visualizer.\n"
        f"Concept: {prompt_concept}\n"
        f"Music Vibe: {bpm:.0f} BPM.\n"
        f"Audio Profile: {sonic_map}\n"
        f"Constraints: We need exactly {num_sections} distinct abstract visual scenes.\n"
        f"These are NOT narrative — they are ABSTRACT VISUAL MOODS (shapes, colors, patterns, energy).\n"
        'Output JSON: { "title": "...", "synopsis": "...", "beats": ["Abstract scene 1", "Abstract scene 2", ...] }'
    )

    beats = []
    try:
        raw = text_engine.generate(story_req, json_schema=True)
        story_data = json.loads(raw)
        if isinstance(story_data, list):
            story_data = story_data[0]
        beats = story_data.get('beats', [])
    except Exception as e:
        logging.warning(f"   ⚠️ Story generation failed: {e}")

    while len(beats) < num_sections:
        beats.append(f"Abstract energy, section {len(beats)+1}")
    beats = beats[:num_sections]
    logging.info(f"   📜 Generated {len(beats)} abstract scene beats")

    # --- Scene Descriptions (one LLM call per section) ---
    logging.info("   🎨 Generating zone-based scene descriptions...")
    scene_descriptions = []
    prev_mood = None
    for i, beat in enumerate(beats):
        style_hint = args.style if hasattr(args, 'style') and args.style else "abstract animated artwork"
        scene_prompt = f"Style: {style_hint}. Scene: {beat}"
        scene = generate_scene_description(text_engine, scene_prompt, i, prev_mood=prev_mood)
        scene_descriptions.append(scene)
        prev_mood = scene.get("mood", None)

    # --- Setup Output Directory ---
    frames_dir = project_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Render Frames ---
    logging.info(f"   🎬 Rendering {target_frames} procedural frames at {width}×{height}...")
    frames_generated = 0
    frames_skipped = 0
    t_start = time.time()

    for sec_idx, (sec_start, sec_end) in enumerate(section_ranges):
        scene = scene_descriptions[sec_idx] if sec_idx < len(scene_descriptions) else scene_descriptions[-1]
        sec_len = max(1, sec_end - sec_start)
        section_seed = int(hashlib.md5(f"{prompt_concept}:{sec_idx}".encode()).hexdigest()[:8], 16)

        for frame_idx in range(sec_start, min(sec_end, target_frames)):
            frame_num = frame_idx + 1
            dst = frames_dir / f"frame_{frame_num:04d}.png"

            # Resume support
            if dst.exists():
                frames_skipped += 1
                continue

            frame_progress = (frame_idx - sec_start) / max(1, sec_len - 1)
            audio_feat = analyzer.get_frame(frame_idx)

            img = render_hires_frame(
                scene, width, height, frame_progress, audio_feat, section_seed
            )
            img.save(dst)
            frames_generated += 1

            if frame_num == 1 or frame_num % 50 == 0 or frame_num == target_frames:
                elapsed = time.time() - t_start
                fps_actual = frames_generated / max(0.1, elapsed)
                pct = frame_num / target_frames * 100
                logging.info(
                    f"   [{sec_idx+1}/{num_sections}] Frame {frame_num}/{target_frames} "
                    f"({pct:.0f}%) — {fps_actual:.1f} frames/sec"
                )

    elapsed = time.time() - t_start
    logging.info(f"   ✅ Rendered {frames_generated} frames in {elapsed:.1f}s "
                 f"(skipped {frames_skipped} existing)")

    return frames_dir, project_fps, target_duration, target_frames
