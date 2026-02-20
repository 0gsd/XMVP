#!/usr/bin/env python3
"""
ANSI Audio Visualizer — 4-Track Demucs Stem Splitter + Procedural ASCII Animation
==================================================================================
Splits audio into drums/bass/keys/other via Demucs, generates per-track
colorized ASCII animations driven by loudness & spectral character, then
composites the four layers (with opacity blending) and muxes synced audio
into a final MP4.

Dependencies (pip):
    librosa, demucs, torch, numpy, Pillow

System deps:
    ffmpeg  (for final video mux)

Usage:
    python ansi_visualizer.py --mu song.mp3 --fps 24
    python ansi_visualizer.py --mu song.wav --fps 30 --width 120 --height 40
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import math
import hashlib
import colorsys
from pathlib import Path

import numpy as np
import librosa
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────────────────────

# Canvas dimensions in characters
DEFAULT_WIDTH = 120
DEFAULT_HEIGHT = 40

# Monospace character cell size in pixels (for rendering)
CELL_W = 10
CELL_H = 18

# Character palettes ordered by visual density (sparse → dense)
DENSITY_CHARS = " .·:;=+*#%@█"
# Specialty palettes per track type
DRUM_CHARS   = " ·.oO0@█▓▒░╳╬"
BASS_CHARS   = " ._~≈≋∼∽║▐█▓▒"  # Using ∼∽ instead of ∿⌇ (better font support)
KEYS_CHARS   = " .·°•◦○◎●♪♫♬★"
OTHER_CHARS  = " .:░▒▓╱╲╳◇◆▲△"
SPIRAL_CHARS = " ·∘○◯◎●◉⊙⊚⊛★"
WAVE_CHARS   = " ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
RAIN_CHARS   = " ·ｦｱｳｴｵｶｷｸｹｺ01"
STAR_CHARS   = " ·∗✦✧★☆✴✵✶✷"

# ── Dynamic Color Palettes ───────────────────────────────────────────────────
# Each palette is a list of (R, G, B) tuples for gradient interpolation
PALETTES = {
    "vaporwave": [(255, 113, 206), (1, 205, 254), (5, 255, 161), (185, 103, 255)],
    "fire": [(255, 0, 0), (255, 154, 0), (255, 255, 0), (255, 100, 50)],
    "ocean": [(0, 119, 190), (0, 180, 216), (144, 224, 239), (72, 202, 228)],
    "neon": [(57, 255, 20), (255, 20, 147), (0, 255, 255), (255, 255, 0)],
    "sunset": [(255, 94, 77), (255, 154, 139), (255, 206, 86), (255, 123, 84)],
    "cyber": [(0, 255, 159), (159, 0, 255), (255, 159, 0), (0, 159, 255)],
    "psychedelic": [(255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 128), (0, 255, 128)],
    "aurora": [(0, 255, 128), (0, 200, 255), (128, 0, 255), (255, 0, 128)],
    "blood": [(139, 0, 0), (178, 34, 34), (220, 20, 60), (255, 69, 0)],
    "toxic": [(0, 255, 0), (127, 255, 0), (50, 205, 50), (0, 128, 0)],
    "ice": [(200, 230, 255), (135, 206, 250), (70, 130, 180), (100, 149, 237)],
    "monochrome": [(255, 255, 255), (180, 180, 180), (120, 120, 120), (60, 60, 60)],
}

# Keywords in prompt that trigger specific palettes
THEME_KEYWORDS = {
    "space": "cyber", "cosmic": "cyber", "stars": "cyber", "galaxy": "aurora",
    "fire": "fire", "flame": "fire", "burn": "fire", "hot": "fire",
    "water": "ocean", "ocean": "ocean", "sea": "ocean", "wave": "ocean",
    "neon": "neon", "electric": "neon", "synth": "neon", "retro": "neon",
    "sunset": "sunset", "dawn": "sunset", "dusk": "sunset",
    "psychedelic": "psychedelic", "trippy": "psychedelic", "acid": "psychedelic", "dream": "psychedelic",
    "aurora": "aurora", "northern": "aurora", "lights": "aurora",
    "blood": "blood", "dark": "blood", "evil": "blood", "death": "blood",
    "toxic": "toxic", "poison": "toxic", "matrix": "toxic", "hack": "toxic",
    "ice": "ice", "cold": "ice", "frozen": "ice", "winter": "ice",
    "vapor": "vaporwave", "wave": "vaporwave", "aesthetic": "vaporwave",
}

# Base hue per track (0-360) - now used as fallback
TRACK_HUES = {
    "drums": 15,    # orange/red
    "bass":  260,   # purple/blue
    "keys":  130,   # green/cyan
    "other": 45,    # gold/yellow
}

# Layer compositing opacity (bottom to top) - increased for more visible layers
LAYER_OPACITY = {
    "drums": 1.0,
    "bass":  0.65,
    "keys":  0.50,
    "other": 0.40,
}

# Layer order (bottom first)
LAYER_ORDER = ["drums", "bass", "keys", "other"]


# ── Audio Analysis ───────────────────────────────────────────────────────────

class TrackAnalyzer:
    """Extracts per-frame audio features for driving visuals, including section detection."""

    def __init__(self, audio_path: str, fps: int, detect_sections: bool = True):
        self.fps = fps
        self.y, self.sr = librosa.load(audio_path, sr=22050, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.n_frames = int(self.duration * fps)
        self.hop = int(self.sr / fps)
        self.detect_sections = detect_sections

        # Pre-compute features
        self._compute_features()
        
        # Section detection
        if detect_sections:
            self._detect_sections()
        else:
            self.section_boundaries = []
            self.section_frames = []
            
        # Beat detection
        self._detect_beats()

    def _compute_features(self):
        """Compute loudness, spectral centroid, onset strength, spectral bandwidth."""
        # RMS loudness per frame
        frame_length = max(self.hop * 2, 2048)
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=self.hop)[0]
        self.loudness = self._normalize(rms[:self.n_frames])

        # Spectral centroid (brightness)
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, hop_length=self.hop)[0]
        self.brightness = self._normalize(cent[:self.n_frames])

        # Onset strength (percussiveness / transients)
        onset = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop)
        self.onset = self._normalize(onset[:self.n_frames])

        # Spectral bandwidth (timbral spread)
        bw = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr, hop_length=self.hop)[0]
        self.bandwidth = self._normalize(bw[:self.n_frames])

        # Chromagram summary (for harmonic color shifts) — average chroma energy
        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=self.hop)
        # Dominant chroma bin per frame → hue shift
        self.chroma_peak = np.argmax(self.chroma[:, :self.n_frames], axis=0)  # 0-11

        # Spectral flux for motion intensity
        S = np.abs(librosa.stft(self.y, hop_length=self.hop))
        flux = np.sqrt(np.mean(np.diff(S, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])
        self.flux = self._normalize(flux[:self.n_frames])

    def _detect_sections(self):
        """Detect song sections using spectral clustering on chroma features."""
        try:
            # Use chroma for section segmentation
            # Compute segment boundaries using recurrence matrix
            R = librosa.segment.recurrence_matrix(self.chroma, mode='affinity', 
                                                   sym=True, sparse=True)
            
            # Use k-means style agglomerative clustering
            # Aim for ~6-10 sections for a typical song
            n_sections = max(4, min(12, int(self.duration / 15)))  # ~1 section per 15s
            
            # Get segment boundaries (in frames relative to chroma, need to convert)
            bounds = librosa.segment.agglomerative(self.chroma, n_sections)
            
            # Convert chroma frame indices to audio frame indices
            # Chroma hop is self.hop, so chroma frame i corresponds to time i * self.hop / self.sr
            chroma_times = librosa.frames_to_time(bounds, sr=self.sr, hop_length=self.hop)
            self.section_boundaries = (chroma_times * self.fps).astype(int)
            
            # Ensure boundaries are within valid range
            self.section_boundaries = self.section_boundaries[self.section_boundaries < self.n_frames]
            self.section_boundaries = np.unique(np.concatenate([[0], self.section_boundaries]))
            
            # Create per-frame section assignment
            self.section_frames = np.zeros(self.n_frames, dtype=int)
            for i, boundary in enumerate(self.section_boundaries[:-1]):
                next_boundary = self.section_boundaries[i + 1] if i + 1 < len(self.section_boundaries) else self.n_frames
                self.section_frames[boundary:next_boundary] = i
            if len(self.section_boundaries) > 0:
                self.section_frames[self.section_boundaries[-1]:] = len(self.section_boundaries) - 1
                
            print(f"   Detected {len(self.section_boundaries)} sections")
        except Exception as e:
            print(f"   ⚠ Section detection failed: {e}")
            self.section_boundaries = np.array([0])
            self.section_frames = np.zeros(self.n_frames, dtype=int)
    
    def _detect_beats(self):
        """Detect beats for synchronized visual effects."""
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr, hop_length=self.hop)
            # Convert to video frames
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.hop)
            self.beat_frames = set((beat_times * self.fps).astype(int))
            # Handle librosa returning array for tempo
            self.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]) if len(tempo) > 0 else 120.0
            print(f"   Detected {len(self.beat_frames)} beats, tempo ~{self.tempo:.0f} BPM")
        except Exception as e:
            print(f"   ⚠ Beat detection failed: {e}")
            self.beat_frames = set()
            self.tempo = 120.0

    @staticmethod
    def _normalize(arr):
        """Normalize array to 0-1 range."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def get_frame(self, frame_idx: int) -> dict:
        """Return feature dict for a single frame, including section info."""
        i = min(frame_idx, self.n_frames - 1)
        
        # Check if this is a section boundary (within 2 frames)
        is_section_boundary = any(abs(i - b) <= 2 for b in self.section_boundaries)
        
        # Check if this is a beat frame
        is_beat = i in self.beat_frames
        
        return {
            "loudness": float(self.loudness[i]),
            "brightness": float(self.brightness[i]),
            "onset": float(self.onset[i]),
            "bandwidth": float(self.bandwidth[i]),
            "chroma": int(self.chroma_peak[i]),
            "flux": float(self.flux[i]),
            "time": frame_idx / self.fps,
            "frame": frame_idx,
            # New fields
            "section": int(self.section_frames[i]) if len(self.section_frames) > i else 0,
            "is_section_boundary": is_section_boundary,
            "is_beat": is_beat,
            "tempo": self.tempo,
        }


# ── Animation Renderers (per-track) ─────────────────────────────────────────

class BaseRenderer:
    """Base class for track-specific ASCII animation with palette support."""

    def __init__(self, width: int, height: int, track_name: str, seed_prompt: str = None, palette_name: str = None):
        self.w = width
        self.h = height
        self.track = track_name
        self.seed_prompt = seed_prompt or "Entropy"
        self.base_hue = TRACK_HUES.get(track_name, 180)
        self.chars = DENSITY_CHARS
        self.rng = np.random.default_rng(
            int(hashlib.md5((track_name + self.seed_prompt).encode()).hexdigest()[:8], 16)
        )
        # Shared grid for optimization
        self.grid_x, self.grid_y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2*height/width, 2*height/width, height))
        
        # Dynamic palette
        self.palette_name = palette_name or self._detect_palette_from_prompt()
        self.palette = PALETTES.get(self.palette_name, PALETTES["psychedelic"])
        
        # Section tracking for resets
        self.current_section = -1
        
    def _detect_palette_from_prompt(self) -> str:
        """Check prompt for theme keywords to pick a palette."""
        prompt_lower = self.seed_prompt.lower()
        for keyword, palette in THEME_KEYWORDS.items():
            if keyword in prompt_lower:
                return palette
        # Fallback: hash-based selection for variety
        phash = int(hashlib.md5(self.seed_prompt.encode()).hexdigest(), 16)
        palette_names = list(PALETTES.keys())
        return palette_names[phash % len(palette_names)]
    
    def set_palette(self, palette_name: str):
        """Switch to a different palette (e.g., at section boundaries)."""
        self.palette_name = palette_name
        self.palette = PALETTES.get(palette_name, PALETTES["psychedelic"])
        
    def reset(self, section_idx: int):
        """Called at section boundaries - subclasses can override to reset state."""
        self.current_section = section_idx

    def render_frame(self, feat: dict) -> list:
        raise NotImplementedError

    def _hue_for_frame(self, feat: dict) -> float:
        chroma_shift = (feat["chroma"] / 12.0) * 30
        hue = (self.base_hue + chroma_shift) % 360
        return hue / 360.0

    def _color(self, hue: float, saturation: float, value: float) -> tuple:
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _palette_color(self, t: float, brightness: float = 1.0) -> tuple:
        """Get color from current palette at position t (0-1), with brightness modulation."""
        t = max(0, min(1, t))
        n = len(self.palette)
        idx = t * (n - 1)
        i = int(idx)
        frac = idx - i
        
        if i >= n - 1:
            c = self.palette[-1]
        else:
            c1, c2 = self.palette[i], self.palette[i + 1]
            c = tuple(int(c1[j] + frac * (c2[j] - c1[j])) for j in range(3))
        
        # Apply brightness
        return tuple(int(min(255, v * brightness)) for v in c)


def get_text_mask(text, w, h):
    """Rasterizes text into a w x h boolean grid (1.0 where text is)."""
    if not text: return np.zeros((h, w))
    
    # Use PIL
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    
    # Try to fit text
    fontsize = int(h * 0.5)
    try:
        font = load_font() # Re-use our safe loader
        # But we need to scale size
        # load_font returns object with set size. We might need a raw loader.
        # Quick hack: standard load_font is fine, we just want *something*
        # Actually let's try to load a bigger one if possible
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", fontsize)
    except:
        font = ImageFont.load_default()
        
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    dx = (w - tw) // 2
    dy = (h - th) // 2
    
    draw.text((dx, dy), text, font=font, fill=255)
    
    # Convert to explicit numpy mask
    arr = np.array(img) / 255.0
    return arr


class JuliaRenderer(BaseRenderer):
    """(Drums) Julia Set Fractal that warps with Onsets."""

    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "drums", seed_prompt)
        self.chars = DRUM_CHARS
        
        # Seed C from prompt
        phash = int(hashlib.md5(self.seed_prompt.encode()).hexdigest(), 16)
        r1 = (phash % 100) / 100.0 - 0.5
        r2 = ((phash >> 8) % 100) / 100.0 - 0.5
        
        self.c = complex(-0.8 + r1*0.4, 0.156 + r2*0.4)
        self.zoom = 1.0

    def render_frame(self, feat):
        t = feat["time"]
        loud = feat["loudness"]
        onset = feat["onset"]
        hue = self._hue_for_frame(feat)

        if onset > 0.4:
            # Warp the constant C on drum hits
            self.c += complex((self.rng.random()-0.5)*0.1, (self.rng.random()-0.5)*0.1)
        
        # Drift C back to base to prevent total chaos
        base_c = complex(-0.8, 0.156)
        self.c = self.c * 0.95 + base_c * 0.05
        
        # Zoom breathe
        target_zoom = 1.0 + loud * 0.5
        self.zoom = self.zoom * 0.9 + target_zoom * 0.1
        
        # Vectorized Julia Iteration
        # Prepare Z
        Z = (self.grid_x + 1j * self.grid_y) * (1.0 / self.zoom)
        # Rotate Z over time
        rot = t * 0.1
        Z = Z * (math.cos(rot) + 1j * math.sin(rot))

        # Iterations
        M = np.zeros(Z.shape, dtype=float)
        for i in range(12): # Low iter count for speed + smoothness
            Z = Z**2 + self.c
            mask = np.abs(Z) < 2
            M[mask] += 1
            
        M = M / 12.0 # Normalize 0-1
        # Guard against NaN from overflow
        M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)
        
        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                val = M[row, col]
                # Modulate with loudness
                val = max(0.0, min(1.0, val + loud*0.2))
                
                char_idx = int(val * (len(self.chars) - 1))
                char = self.chars[char_idx]

                sat = 0.6 + onset * 0.4
                bright = min(1.0, val * 0.9 + loud * 0.2)
                h_shift = hue + val * 0.1
                color = self._color(h_shift % 1.0, sat, bright)
                line.append((char, *color))
            grid.append(line)
        return grid


class PlasmaRenderer(BaseRenderer):
    """(Bass) Old-school plasma interference."""

    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "bass", seed_prompt)
        self.chars = BASS_CHARS
        
        # Jitter phase based on prompt
        phash = int(hashlib.md5(self.seed_prompt.encode()).hexdigest(), 16)
        self.phase_shift = (phash % 1000) / 100.0

    def render_frame(self, feat):
        t = feat["time"] * 2.0 + self.phase_shift # Faster + Unique Phase
        loud = feat["loudness"]
        bw = feat["bandwidth"]
        hue = self._hue_for_frame(feat)

        # Vectorized plasma
        X = self.grid_x
        Y = self.grid_y
        
        v1 = np.sin(X * 3.0 + t)
        v2 = np.sin(10 * (X * np.sin(t / 2) + Y * np.cos(t / 3)) + t)
        cx = X + 0.5 * np.sin(t / 5)
        cy = Y + 0.5 * np.cos(t / 3)
        v3 = np.sin(np.sqrt(100 * (cx**2 + cy**2) + 1) + t)
        
        V = (v1 + v2 + v3) / 3.0
        V = (V + 1) / 2.0 # 0-1 range
        
        # Bass warping
        V = np.power(V, 1.0 - loud * 0.5)

        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                val = V[row, col]
                char_idx = int(val * (len(self.chars) - 1))
                char = self.chars[char_idx]

                sat = 0.5 + val * 0.5
                bright = val * 0.8 + loud * 0.3
                h = (hue + val * 0.2) % 1.0
                color = self._color(h, sat, bright)
                line.append((char, *color))
            grid.append(line)
        return grid


class LifeRenderer(BaseRenderer):
    """(Keys) Conway's Game of Life."""

    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "keys", seed_prompt)
        self.chars = KEYS_CHARS
        
        # Random noise seeding (no text - cleaner look)
        self.cells = np.random.choice([0, 1], size=(h, w), p=[0.85, 0.15])
        
        self.accumulator = np.zeros((h, w)) # Trail
        
    def step(self):
        # Count neighbors
        N = (
            np.roll(self.cells, 1, axis=0) + np.roll(self.cells, -1, axis=0) +
            np.roll(self.cells, 1, axis=1) + np.roll(self.cells, -1, axis=1) +
            np.roll(np.roll(self.cells, 1, axis=0), 1, axis=1) +
            np.roll(np.roll(self.cells, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(self.cells, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(self.cells, -1, axis=0), -1, axis=1)
        )
        # Rules: Birth on 3, Survive on 2 or 3
        birth = (N == 3) & (self.cells == 0)
        survive = ((N == 2) | (N == 3)) & (self.cells == 1)
        self.cells[...] = 0
        self.cells[birth | survive] = 1

    def render_frame(self, feat):
        loud = feat["loudness"]
        bright = feat["brightness"]
        flux = feat["flux"]
        hue = self._hue_for_frame(feat)

        # Evolve
        self.step()
        
        # Inject new life based on notes/flux
        if flux > 0.2:
            num_spawns = int(flux * 10)
            for _ in range(num_spawns):
                rx = self.rng.integers(0, self.w)
                ry = self.rng.integers(0, self.h)
                # Glider-ish
                self.cells[ry, rx] = 1
                if ry+1 < self.h and rx+1 < self.w: self.cells[ry+1, rx+1] = 1
                
        # Accumulate trails
        self.accumulator = self.accumulator * 0.85 + self.cells * 1.0
        
        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                val = max(0, min(1.0, self.accumulator[row, col]))
                
                if val < 0.1:
                    char = ' '
                else:
                    char_idx = int(val * (len(self.chars) - 1))
                    char = self.chars[char_idx]

                sat = 0.3 + val * 0.7
                bri = val * (0.5 + loud)
                h = (hue + col/self.w * 0.1) % 1.0
                color = self._color(h, sat, bri)
                line.append((char, *color))
            grid.append(line)
        return grid


class ReactionRenderer(BaseRenderer):
    """(Other) Gray-Scott Reaction Diffusion."""

    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "other", seed_prompt)
        self.chars = OTHER_CHARS
        # A and B chemicals
        self.A = np.ones((h, w))
        self.B = np.zeros((h, w))
        
        # Random scatter seeding (no text - cleaner look)
        # Use prompt hash to determine seed pattern
        phash = int(hashlib.md5((seed_prompt or "").encode()).hexdigest(), 16)
        n_seeds = 5 + (phash % 8)  # 5-12 seed points
        rng = np.random.default_rng(phash)
        for _ in range(n_seeds):
            cx = rng.integers(5, w - 5)
            cy = rng.integers(5, h - 5)
            r = 3 + rng.integers(0, 4)
            y1, y2 = max(0, cy - r), min(h, cy + r)
            x1, x2 = max(0, cx - r), min(w, cx + r)
            self.B[y1:y2, x1:x2] = 1.0
        
        # Parameters (Gray-Scott spots/corals)
        self.Du, self.Dv = 0.16, 0.08
        self.f, self.k = 0.035, 0.060 # Standard coral

    def render_frame(self, feat):
        loud = feat["loudness"]
        flux = feat["flux"]
        hue = self._hue_for_frame(feat)
        
        # Modulate params with music
        # More flux = more chaos (lower k)
        # More loud = more feed (higher f)
        f_mod = self.f + loud * 0.02
        k_mod = self.k - flux * 0.01

        # Laplacian
        def laplacian(M):
            return (
                -4 * M +
                np.roll(M, 1, axis=0) + np.roll(M, -1, axis=0) +
                np.roll(M, 1, axis=1) + np.roll(M, -1, axis=1)
            )

        # 4 steps per frame for stability/speed
        for _ in range(4):
            Lu = laplacian(self.A)
            Lv = laplacian(self.B)
            
            uvv = self.A * self.B * self.B
            self.A += (self.Du * Lu - uvv + f_mod * (1 - self.A))
            self.B += (self.Dv * Lv + uvv - (f_mod + k_mod) * self.B)
        
        # Clamp to prevent NaN/overflow
        self.A = np.clip(self.A, 0, 1)
        self.B = np.clip(self.B, 0, 1)
        np.nan_to_num(self.A, copy=False, nan=0.5)
        np.nan_to_num(self.B, copy=False, nan=0.0)

        # Visualize B concentration
        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                val = self.B[row, col]
                # Guard against NaN/Inf
                if not np.isfinite(val):
                    val = 0.0
                val = max(0.0, min(1.0, val))
                
                # Nonlinear boost for contrast
                val = val ** 0.5
                
                char_idx = int(val * (len(self.chars) - 1))
                char = self.chars[char_idx]

                sat = val
                bri = val * 2.0 # Boost brightness
                h = (hue + val * 0.3) % 1.0
                color = self._color(h, sat, min(1.0, bri))
                line.append((char, *color))
            grid.append(line)
        return grid



RENDERERS = {
    "drums": JuliaRenderer,
    "bass":  PlasmaRenderer,
    "keys":  LifeRenderer,
    "other": ReactionRenderer,
}


# ── NEW RENDERERS ─────────────────────────────────────────────────────────────

class SpiralRenderer(BaseRenderer):
    """Logarithmic spirals that pulse with audio."""
    
    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "keys", seed_prompt)
        self.chars = SPIRAL_CHARS
        phash = int(hashlib.md5(self.seed_prompt.encode()).hexdigest(), 16)
        self.arm_count = 3 + (phash % 5)  # 3-7 spiral arms
        self.rotation = 0.0
        
    def reset(self, section_idx: int):
        super().reset(section_idx)
        # Randomize arm count on section change
        self.arm_count = 3 + (section_idx % 5)
        
    def render_frame(self, feat):
        t = feat["time"]
        loud = feat["loudness"]
        onset = feat["onset"]
        flux = feat["flux"]
        
        # Rotate with beat
        self.rotation += 0.02 + onset * 0.15
        
        # Breathing zoom
        zoom = 1.0 + loud * 0.5 + math.sin(t * 2) * 0.2
        
        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                x = self.grid_x[row, col]
                y = self.grid_y[row, col]
                
                # Convert to polar
                r = math.sqrt(x*x + y*y) + 0.001
                theta = math.atan2(y, x) + self.rotation
                
                # Logarithmic spiral formula
                spiral_val = (theta + math.log(r * zoom) * 3) * self.arm_count / (2 * math.pi)
                spiral_val = (math.sin(spiral_val * math.pi * 2) + 1) / 2
                
                # Add radial pulse
                pulse = math.sin(r * 10 - t * 5) * 0.3 * loud
                val = max(0, min(1, spiral_val + pulse))
                
                char_idx = int(val * (len(self.chars) - 1))
                char = self.chars[char_idx]
                
                # Palette color based on angle
                t_color = (theta / (2 * math.pi) + 0.5) % 1.0
                color = self._palette_color(t_color, 0.5 + val * 0.5 + loud * 0.3)
                line.append((char, *color))
            grid.append(line)
        return grid


class WaveformRenderer(BaseRenderer):
    """Oscilloscope-style waveform visualization."""
    
    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "bass", seed_prompt)
        self.chars = WAVE_CHARS
        self.history = np.zeros((h, w))  # Decay trail
        
    def reset(self, section_idx: int):
        super().reset(section_idx)
        self.history *= 0.2  # Partial reset on section change
        
    def render_frame(self, feat):
        loud = feat["loudness"]
        bright = feat["brightness"]
        flux = feat["flux"]
        onset = feat["onset"]
        t = feat["time"]
        
        # Generate waveform line at center
        center_y = self.h // 2
        amplitude = int(self.h * 0.4 * loud)
        
        # Decay history
        self.history *= 0.85
        
        # Draw multiple wave lines
        for wave_i in range(3):
            freq = 2 + wave_i + bright * 3
            phase = t * (5 + wave_i * 2) + wave_i * math.pi / 3
            for col in range(self.w):
                x_norm = col / self.w
                y_offset = math.sin(x_norm * freq * math.pi * 2 + phase) * amplitude
                y_offset += math.sin(x_norm * freq * 2 * math.pi + phase * 1.5) * amplitude * 0.3
                y = int(center_y + y_offset)
                if 0 <= y < self.h:
                    self.history[y, col] = max(self.history[y, col], 1.0 - wave_i * 0.2)
        
        # Add beat pulses
        if onset > 0.5:
            for col in range(0, self.w, 4):
                for row in range(self.h):
                    self.history[row, col] = max(self.history[row, col], onset * 0.5)
        
        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                val = min(1.0, self.history[row, col])
                
                if val < 0.1:
                    char = ' '
                    color = (0, 0, 0)
                else:
                    char_idx = int(val * (len(self.chars) - 1))
                    char = self.chars[min(char_idx, len(self.chars) - 1)]
                    # Vertical gradient + palette
                    t_color = row / self.h
                    color = self._palette_color(t_color, val * (0.7 + loud * 0.5))
                
                line.append((char, *color))
            grid.append(line)
        return grid


class RainRenderer(BaseRenderer):
    """Matrix-style falling characters."""
    
    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "other", seed_prompt)
        self.chars = RAIN_CHARS
        # Initialize rain drops: each column has a drop position and speed
        self.drops = np.zeros(w)
        self.speeds = self.rng.uniform(0.5, 2.0, size=w)
        self.trail_length = np.full(w, 8, dtype=int)
        self.char_grid = self.rng.integers(0, len(self.chars), size=(h, w))
        
    def reset(self, section_idx: int):
        super().reset(section_idx)
        # Randomize speeds on section change
        self.speeds = self.rng.uniform(0.5 + section_idx * 0.2, 2.0 + section_idx * 0.3, size=self.w)
        
    def render_frame(self, feat):
        loud = feat["loudness"]
        onset = feat["onset"]
        flux = feat["flux"]
        is_beat = feat.get("is_beat", False)
        
        # Speed up with loudness
        speed_mult = 1.0 + loud * 2.0
        
        # Update drop positions
        self.drops += self.speeds * speed_mult
        self.drops[self.drops > self.h + 10] = -self.rng.integers(0, 10, size=np.sum(self.drops > self.h + 10))
        
        # Longer trails on beats
        if is_beat:
            self.trail_length = np.clip(self.trail_length + 3, 5, 20)
        else:
            self.trail_length = np.clip(self.trail_length - 0.5, 5, 20).astype(int)
        
        # Shuffle some characters occasionally
        if onset > 0.6:
            shuffle_cols = self.rng.choice(self.w, size=int(self.w * 0.1))
            for c in shuffle_cols:
                self.char_grid[:, c] = self.rng.integers(0, len(self.chars), size=self.h)
        
        grid = []
        for row in range(self.h):
            line = []
            for col in range(self.w):
                drop_y = self.drops[col]
                dist = drop_y - row
                trail = self.trail_length[col]
                
                if 0 <= dist < trail:
                    # In the trail
                    intensity = 1.0 - (dist / trail)
                    char_idx = self.char_grid[row, col]
                    char = self.chars[char_idx]
                    # Head is brightest
                    is_head = dist < 1
                    if is_head:
                        color = (255, 255, 255)  # White head
                    else:
                        color = self._palette_color(intensity, intensity * (0.6 + loud * 0.4))
                else:
                    char = ' '
                    color = (0, 0, 0)
                
                line.append((char, *color))
            grid.append(line)
        return grid


class StarfieldRenderer(BaseRenderer):
    """3D parallax starfield flying through space."""
    
    def __init__(self, w, h, seed_prompt=None):
        super().__init__(w, h, "drums", seed_prompt)
        self.chars = STAR_CHARS
        self.n_stars = 150
        self._init_stars()
        
    def _init_stars(self):
        # Stars: (x, y, z) where z is depth (0-1, smaller = closer)
        self.stars_x = self.rng.uniform(-2, 2, self.n_stars)
        self.stars_y = self.rng.uniform(-2, 2, self.n_stars)
        self.stars_z = self.rng.uniform(0.01, 1.0, self.n_stars)
        
    def reset(self, section_idx: int):
        super().reset(section_idx)
        # Scatter stars on section change
        self.stars_z = self.rng.uniform(0.01, 1.0, self.n_stars)
        
    def render_frame(self, feat):
        loud = feat["loudness"]
        onset = feat["onset"]
        bright = feat["brightness"]
        is_beat = feat.get("is_beat", False)
        
        # Speed based on loudness
        speed = 0.02 + loud * 0.08
        if is_beat:
            speed *= 2
        
        # Move stars toward camera (decrease z)
        self.stars_z -= speed
        
        # Respawn stars that passed the camera
        respawn = self.stars_z <= 0
        self.stars_z[respawn] = 1.0
        self.stars_x[respawn] = self.rng.uniform(-2, 2, np.sum(respawn))
        self.stars_y[respawn] = self.rng.uniform(-2, 2, np.sum(respawn))
        
        # Build grid
        grid = [[(' ', 0, 0, 0) for _ in range(self.w)] for _ in range(self.h)]
        
        cx, cy = self.w // 2, self.h // 2
        
        for i in range(self.n_stars):
            # Project 3D to 2D with perspective
            z = max(0.01, self.stars_z[i])
            px = int(cx + self.stars_x[i] / z * self.w * 0.3)
            py = int(cy + self.stars_y[i] / z * self.h * 0.3)
            
            if 0 <= px < self.w and 0 <= py < self.h:
                # Closer stars are brighter and larger
                intensity = 1.0 - z
                char_idx = int(intensity * (len(self.chars) - 1))
                char = self.chars[min(char_idx, len(self.chars) - 1)]
                
                # Trail effect based on speed
                trail_len = int(intensity * 3 * (1 + loud))
                
                color = self._palette_color(intensity, 0.5 + intensity * 0.5 + loud * 0.3)
                
                # Draw star and short trail
                for t in range(trail_len + 1):
                    trail_y = py + t
                    if 0 <= trail_y < self.h:
                        trail_intensity = 1.0 - (t / (trail_len + 1))
                        trail_color = tuple(int(c * trail_intensity) for c in color)
                        grid[trail_y][px] = (char if t == 0 else '·', *trail_color)
        
        return grid


# ── RENDERER POOL ─────────────────────────────────────────────────────────────
# All available renderers for dynamic selection
RENDERER_POOL = [
    JuliaRenderer,
    PlasmaRenderer,
    LifeRenderer,
    ReactionRenderer,
    SpiralRenderer,
    WaveformRenderer,
    RainRenderer,
    StarfieldRenderer,
]

def select_renderers_for_section(prompt: str, section_idx: int, audio_energy: float = 0.5) -> list:
    """
    Select 4 renderers for a section based on prompt hash + section index + audio characteristics.
    Returns list of 4 renderer classes.
    """
    # Hash for deterministic but varied selection
    seed = int(hashlib.md5(f"{prompt}:{section_idx}".encode()).hexdigest(), 16)
    rng = np.random.default_rng(seed)
    
    # Shuffle pool based on seed
    pool = RENDERER_POOL.copy()
    rng.shuffle(pool)
    
    # High energy sections favor more dynamic renderers
    if audio_energy > 0.6:
        # Prioritize Julia, Spiral, Starfield (more motion)
        dynamic = [JuliaRenderer, SpiralRenderer, StarfieldRenderer, PlasmaRenderer]
        rng.shuffle(dynamic)
        selected = dynamic[:2] + pool[:2]
    else:
        selected = pool[:4]
    
    # Ensure 4 unique renderers
    seen = set()
    result = []
    for r in selected:
        if r not in seen:
            result.append(r)
            seen.add(r)
    while len(result) < 4:
        for r in pool:
            if r not in seen:
                result.append(r)
                seen.add(r)
                break
    
    return result[:4]

def select_palette_for_section(prompt: str, section_idx: int) -> str:
    """Select a palette for a section, rotating through palettes to ensure variety."""
    palette_names = list(PALETTES.keys())
    
    # Check prompt for keyword-based palette first
    prompt_lower = prompt.lower()
    for keyword, palette in THEME_KEYWORDS.items():
        if keyword in prompt_lower:
            # Rotate through related palettes
            base_idx = palette_names.index(palette) if palette in palette_names else 0
            return palette_names[(base_idx + section_idx) % len(palette_names)]
    
    # Hash-based rotation
    seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    return palette_names[(seed + section_idx) % len(palette_names)]


# ── Compositing ──────────────────────────────────────────────────────────────

def composite_layers(layers: dict, width: int, height: int) -> list:
    """
    Merge 4 track grids (each HxW of (char, r, g, b)) with opacity blending.
    Bottom layer = drums (full opacity), stacking up.
    For blending: we treat the character's brightness as alpha, weighted by layer opacity.
    The character chosen comes from the layer with highest weighted brightness at that cell.
    The color is blended additively.
    """
    merged = []
    for row in range(height):
        line = []
        for col in range(width):
            # Accumulate color via weighted additive blend
            r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
            best_char = ' '
            best_weight = 0.0

            for track in LAYER_ORDER:
                char, r, g, b = layers[track][row][col]
                opacity = LAYER_OPACITY[track]
                # Character brightness as local alpha
                brightness = (r + g + b) / (3 * 255.0) if (r + g + b) > 0 else 0
                weight = brightness * opacity

                r_acc += r * opacity
                g_acc += g * opacity
                b_acc += b * opacity

                if weight > best_weight and char != ' ':
                    best_weight = weight
                    best_char = char

            # Clamp color
            r_out = int(min(255, r_acc))
            g_out = int(min(255, g_acc))
            b_out = int(min(255, b_acc))

            line.append((best_char, r_out, g_out, b_out))
        merged.append(line)
    return merged


# ── Frame → Image Rendering ─────────────────────────────────────────────────

def load_font():
    """Try to load a monospace font with good Unicode block character support."""
    # Prioritize fonts with full Unicode block character and symbol support
    mono_paths = [
        # macOS - Fonts with better Unicode coverage FIRST
        "/System/Library/Fonts/SFNSMono.ttf",  # System SF Mono - good Unicode
        "/System/Library/Fonts/Menlo.ttc",  # Menlo - reasonable block support
        # Linux - DejaVu has excellent Unicode coverage
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        # macOS fallbacks (may have gaps in Unicode)
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Andale Mono.ttf",
        "/Library/Fonts/Courier New.ttf",
    ]
    
    # Test characters that the visualizer uses
    test_chars = "█▓▒░▁▂▃▄▅▆▇♪♫♬★✦✧∿⌇"
    
    for p in mono_paths:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, 20)
                # Quick verification: try to get bbox for test chars
                # If font doesn't have glyphs, some implementations return tiny boxes
                from PIL import Image, ImageDraw
                test_img = Image.new("L", (100, 30))
                test_draw = ImageDraw.Draw(test_img)
                try:
                    bbox = test_draw.textbbox((0, 0), test_chars, font=font)
                    # If bbox width is reasonable (not 0), font likely supports chars
                    if bbox[2] - bbox[0] > 50:  # At least 50px wide for ~20 chars
                        return font
                except:
                    pass
                # If verification fails, still return font (better than nothing)
                return font
            except Exception:
                continue
    return ImageFont.load_default()


def grid_to_image(grid: list, width: int, height: int, font) -> Image.Image:
    """Render a character grid to a PIL Image."""
    img_w = width * CELL_W
    img_h = height * CELL_H
    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for row_idx, row in enumerate(grid):
        for col_idx, (char, r, g, b) in enumerate(row):
            if char != ' ' and (r + g + b) > 0:
                x = col_idx * CELL_W
                y = row_idx * CELL_H
                draw.text((x, y), char, fill=(r, g, b), font=font)

    return img


# ── Stem Splitting ───────────────────────────────────────────────────────────

def split_stems(input_path: str, output_dir: str) -> dict:
    """
    Use Demucs to separate audio into 4 stems.
    Returns dict mapping track name → audio file path.
    """
    print("🎵 Splitting stems with Demucs (htdemucs)...")
    print("   This may take a while on first run (model download)...")

    cmd = [
        sys.executable, "-m", "demucs",

        "-n", "htdemucs",
        "--out", output_dir,
        input_path,
    ]

    # Demucs outputs to <out>/<model>/<track_name>/
    subprocess.run(cmd, check=True)

    # Find the output directory
    stem_name = Path(input_path).stem
    stem_dir = Path(output_dir) / "htdemucs" / stem_name

    if not stem_dir.exists():
        # Try without model subdirectory
        candidates = list(Path(output_dir).rglob(f"{stem_name}"))
        if candidates:
            stem_dir = candidates[0]

    stems = {}
    # Demucs htdemucs outputs: drums.wav, bass.wav, other.wav, vocals.wav
    # We'll map vocals → keys (since "keys" is closest to vocals channel for many tracks)
    # User asked for: drums, bass, keys, other
    mapping = {
        "drums": "drums.wav",
        "bass": "bass.wav",
        "keys": "vocals.wav",   # vocals stem ≈ melodic/keys layer
        "other": "other.wav",
    }

    for track, filename in mapping.items():
        path = stem_dir / filename
        if path.exists():
            stems[track] = str(path)
        else:
            print(f"   ⚠ Stem '{filename}' not found, looking for alternatives...")
            wavs = list(stem_dir.glob("*.wav"))
            print(f"   Found stems: {[w.name for w in wavs]}")

    if len(stems) < 4:
        print(f"   Found stems: {list(stems.keys())}")
        print(f"   In directory: {stem_dir}")
        available = list(stem_dir.glob("*.wav")) if stem_dir.exists() else []
        print(f"   Available files: {[f.name for f in available]}")

    return stems

def analyze_audio(audio_path, fsync=1.0):
    """
    Analyzes audio for BPM and Duration to calculate Auto-FPS.
    Returns (bpm, duration, beat_frames, suggested_fps)
    """
    try:
        # Load a snippet or full audio? Full is safer for BPM.
        y, sr = librosa.load(audio_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Tempo is an array or float. Normalize.
        bpm = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)
        
        # Calculate Frames Per Beat (FPB) targeting ~4-12 FPS range
        frames_per_beat = 4 
        bps = bpm / 60.0
        fps = (bps * frames_per_beat) * fsync
        
        print(f"   🎵 Audio Analysis: {bpm:.1f} BPM | {duration:.1f}s")
        print(f"   🎵 Derived FPS: {fps:.2f} (based on {frames_per_beat} frames/beat * fsync {fsync})")
        
        return bpm, duration, frames_per_beat, fps
        
    except Exception as e:
        print(f"   ❌ Audio Analysis Failed: {e}")
        return 120.0, 60.0, 4, 8.0 * fsync # Fallback


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="4-Track ANSI Audio Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python ansi_visualizer.py --mu song.mp3 --fps 24
    python ansi_visualizer.py --mu song.wav --fps 30 --width 160 --height 50
        """
    )
    parser.add_argument("--mu", required=True, help="Path to input audio (wav/aif/mp3)")
    parser.add_argument("prompt", nargs="?", default="A Beautiful Glitch", help="Text prompt to seed the design")
    parser.add_argument("--fps", type=float, default=0.0, help="Frames per second. Default: Auto-calculated if 0")
    parser.add_argument("--fsync", type=float, default=1.0, help="FPS Sync Multiplier (0.1 - 6.0). Scales auto-calculated FPS.")
    parser.add_argument("--width", type=int, default=120, help="Grid width in characters (Default: 120)")
    parser.add_argument("--height", type=int, default=68, help="Grid height in characters (Default: 68)")
    parser.add_argument("--output", help="Path to output video (mp4)")
    parser.add_argument("--preview", action="store_true", help="Preview a single frame to terminal (ANSI)")
    parser.add_argument("--keep-stems", action="store_true", help="Don't delete separated stems after processing")
    args = parser.parse_args()

    input_path = args.mu
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # Output Directory Logic
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_out_dir = os.path.join(base_dir, "z_test-outputs")
    os.makedirs(default_out_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(default_out_dir, Path(input_path).stem + "_visualized.mp4")

    if args.fps > 0:
        fps = float(args.fps)
    else:
        print("   🎵 Auto-calculating FPS from audio...")
        bpm, duration, fpb, suggested_fps = analyze_audio(input_path, fsync=args.fsync)
        fps = suggested_fps

    W, H = args.width, args.height

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   ANSI Audio Visualizer — 4-Track Renderer   ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Input:  {Path(input_path).name:<36} ║")
    print(f"║  FPS:    {fps:<36} ║")
    print(f"║  Canvas: {W}×{H} chars{'':<22} ║")
    print(f"║  Output: {Path(output_path).name:<36} ║")
    print(f"╚══════════════════════════════════════════════╝")

    # ── Step 1: Split stems ──────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="ansi_viz_")
    try:
        stems = split_stems(input_path, tmp_dir)
        if len(stems) < 4:
            print("Error: Could not extract all 4 stems. Aborting.")
            sys.exit(1)

        # ── Step 2: Analyze each stem ────────────────────────────────────
        print("\n📊 Analyzing audio features per stem...")
        analyzers = {}
        for track_name, stem_path in stems.items():
            print(f"   Analyzing {track_name}...")
            analyzers[track_name] = TrackAnalyzer(stem_path, fps)

        n_frames = min(a.n_frames for a in analyzers.values())
        duration = n_frames / fps
        print(f"   Duration: {duration:.1f}s, {n_frames} frames")

        # ── Step 3: Section-Aware Renderer Setup ─────────────────────────
        print("\n🎨 Setting up section-aware rendering...")
        
        # Get section boundaries from the drums analyzer (main reference)
        drums_analyzer = analyzers["drums"]
        section_boundaries = list(drums_analyzer.section_boundaries)
        n_sections = len(section_boundaries)
        print(f"   {n_sections} sections detected")
        
        # Pre-select renderers and palettes for each section
        section_configs = []
        for sec_idx in range(n_sections):
            # Calculate section energy (avg loudness in that section)
            start_frame = section_boundaries[sec_idx]
            end_frame = section_boundaries[sec_idx + 1] if sec_idx + 1 < n_sections else n_frames
            sec_loudness = np.mean(drums_analyzer.loudness[start_frame:end_frame])
            
            # Select renderers for this section
            renderer_classes = select_renderers_for_section(args.prompt, sec_idx, sec_loudness)
            palette = select_palette_for_section(args.prompt, sec_idx)
            
            section_configs.append({
                "start": start_frame,
                "end": end_frame,
                "renderers": renderer_classes,
                "palette": palette,
                "energy": sec_loudness,
            })
            print(f"   Section {sec_idx}: frames {start_frame}-{end_frame}, "
                  f"palette={palette}, renderers={[r.__name__[:8] for r in renderer_classes]}")
        
        # Initialize first section's renderers
        current_section = 0
        active_palette = section_configs[0]["palette"]
        renderers = {}
        for i, track_name in enumerate(LAYER_ORDER):
            renderer_class = section_configs[0]["renderers"][i % len(section_configs[0]["renderers"])]
            renderers[track_name] = renderer_class(W, H, args.prompt)
            renderers[track_name].set_palette(active_palette)
        
        # ── Optional: Terminal preview ───────────────────────────────────
        if args.preview:
            mid = n_frames // 2
            print(f"\n🖥  Preview frame {mid} (middle of track):\n")
            layers = {}
            for track_name in LAYER_ORDER:
                feat = analyzers[track_name].get_frame(mid)
                layers[track_name] = renderers[track_name].render_frame(feat)
            merged = composite_layers(layers, W, H)
            # Print with ANSI colors
            for row in merged:
                line = ""
                for char, r, g, b in row:
                    line += f"\033[38;2;{r};{g};{b}m{char}"
                line += "\033[0m"
                print(line)
            print()
            if not args.output:
                return

        # ── Step 4: Render all frames with section awareness ─────────────
        print("\n🎬 Rendering frames with section-aware effects...")
        font = load_font()
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Beat flash decay
        beat_flash_intensity = 0.0

        for frame_idx in range(n_frames):
            if frame_idx % fps == 0:
                elapsed = frame_idx / fps
                pct = (frame_idx / n_frames) * 100
                print(f"   Frame {frame_idx}/{n_frames} ({pct:.0f}%) — {elapsed:.0f}s / {duration:.0f}s")

            # Check for section change
            for sec_idx, config in enumerate(section_configs):
                if config["start"] <= frame_idx < config["end"] and sec_idx != current_section:
                    # SECTION TRANSITION!
                    current_section = sec_idx
                    active_palette = config["palette"]
                    print(f"   ↳ Section {sec_idx}: switching to palette={active_palette}")
                    
                    # Reinitialize renderers with new classes for this section
                    for i, track_name in enumerate(LAYER_ORDER):
                        renderer_class = config["renderers"][i % len(config["renderers"])]
                        renderers[track_name] = renderer_class(W, H, args.prompt)
                        renderers[track_name].set_palette(active_palette)
                        renderers[track_name].reset(sec_idx)
                    break
            
            # Get features for all tracks
            feats = {track: analyzers[track].get_frame(frame_idx) for track in LAYER_ORDER}
            
            # Check for beat (use drums track for beat detection)
            is_beat = feats["drums"].get("is_beat", False)
            is_section_boundary = feats["drums"].get("is_section_boundary", False)
            
            # Update beat flash
            if is_beat:
                beat_flash_intensity = 0.7  # Flash on beat
            else:
                beat_flash_intensity *= 0.7  # Decay

            # Render each track
            layers = {}
            for track_name in LAYER_ORDER:
                layers[track_name] = renderers[track_name].render_frame(feats[track_name])

            # Composite
            merged = composite_layers(layers, W, H)
            
            # Apply beat flash overlay (additive white flash)
            if beat_flash_intensity > 0.05:
                for row_idx in range(H):
                    for col_idx in range(W):
                        char, r, g, b = merged[row_idx][col_idx]
                        flash = int(beat_flash_intensity * 80)
                        r = min(255, r + flash)
                        g = min(255, g + flash)
                        b = min(255, b + flash)
                        merged[row_idx][col_idx] = (char, r, g, b)
            
            # Section transition: STARBURST FLASH (bright additive color from palette)
            if is_section_boundary:
                # Get a bright color from the current palette
                palette = PALETTES.get(active_palette, PALETTES["psychedelic"])
                burst_color = palette[frame_idx % len(palette)]  # Cycle through palette
                
                for row_idx in range(H):
                    for col_idx in range(W):
                        char, r, g, b = merged[row_idx][col_idx]
                        # Bright additive flash with palette color
                        flash_intensity = 0.9
                        r = min(255, int(r * 0.3 + burst_color[0] * flash_intensity))
                        g = min(255, int(g * 0.3 + burst_color[1] * flash_intensity))
                        b = min(255, int(b * 0.3 + burst_color[2] * flash_intensity))
                        merged[row_idx][col_idx] = (char, r, g, b)

            # Render to image
            img = grid_to_image(merged, W, H, font)
            img.save(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"))

        print(f"   ✓ Rendered {n_frames} frames")

        # ── Step 5: Encode video with ffmpeg ────────────────────────────
        print("\n🎥 Encoding video with ffmpeg...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-i", input_path,
            # Scaling: Downscale to max 1080p height if larger, keeping aspect ratio
            "-vf", "scale=-2:'min(1080,ih)'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"\n✅ Done! Output: {output_path}")
        print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    finally:
        if not args.keep_stems:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"   Stems kept in: {tmp_dir}")


if __name__ == "__main__":
    main()
