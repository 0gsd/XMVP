#!/usr/bin/env python3
"""
UNICODE Audio Visualizer — 4-Track Demucs Stem Splitter + Procedural Unicode Animation
======================================================================================
Extended version of ansi_visualizer.py that leverages Unicode's 140K+ character
repertoire for richer, more varied visualizations including emoji, braille,
geometric shapes, and themed character pools.

Dependencies (pip):
    librosa, demucs, torch, numpy, Pillow

System deps:
    ffmpeg  (for final video mux)

Usage:
    python unicode_visualizer.py --mu song.mp3 --fps 24 --theme matrix
    python unicode_visualizer.py --mu song.wav --fps 30 --theme emoji
    python unicode_visualizer.py --mu song.mp3 --theme random  # randomize per section
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

# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER POOL SYSTEM - The heart of Unicode Visualizer
# ══════════════════════════════════════════════════════════════════════════════

class CharacterPool:
    """
    Manages themed character sets with randomization for varied visualizations.
    Each theme provides different character sets for density gradients.
    """
    
    # Density pools (sparse → dense) - used for intensity mapping
    DENSITY_LEVELS = {
        "minimal":    " ·.,:;",
        "dots":       " ·•●◉◎○◯⊙⊚⊛",
        "blocks":     " ░▒▓█▀▄▌▐",
        "geometric":  " ◇◆○●△▲□■◊⬟⬢",
        "braille":    " ⠁⠃⠇⠏⠟⠿⡿⣿",  # 8-dot braille patterns
        "bars":       " ▁▂▃▄▅▆▇█",
    }
    
    # Thematic character sets
    THEMES = {
        "classic": {
            "name": "Classic ANSI",
            "density": " .·:;=+*#%@█",
            "drums":   " ·.oO0@█▓▒░╳╬",
            "bass":    " ._~≈≋∼∽║▐█▓▒",
            "keys":    " .·°•◦○◎●♪♫♬★",
            "other":   " .:░▒▓╱╲╳◇◆▲△",
            "special": "╔═╗║╚╝╠╣╬╪╫",
        },
        "matrix": {
            "name": "Matrix Rain",
            "density": " ·01ｦｱｳｴｵ日月火",
            "drums":   " 01ｶｷｸｹｺ水木金",
            "bass":    " ｻｼｽｾｿ土天人中",
            "keys":    " ﾀﾁﾂﾃﾄ大小上下",
            "other":   " ﾅﾆﾇﾈﾉ左右前後",
            "special": "零一二三四五六七八九",
        },
        "stars": {
            "name": "Cosmic Stars",
            "density": " ·∗✦✧★☆✴✵✶✷",
            "drums":   " ·✦✧★✴✵✶✷⋆∗",
            "bass":    " ·○◯◎●◉⊙⊚⦿⊛",
            "keys":    " ·✧✦★☆✴✵✶✷⭐",
            "other":   " ·∗⁂※✱✲✳✺✻✼",
            "special": "⭐🌟💫✨⚡🌙☀️🌈",
        },
        "fire": {
            "name": "Inferno",
            "density": " ·.▴△▲◊◆▰▱█",
            "drums":   " ·*+△▲▴◊◆◈❖",
            "bass":    " ·~∿≈≋∼∽〰⌇▐",
            "keys":    " ·°•◦○◎●◉⊙⊚",
            "other":   " ·▪▫▬▭▮▯▰▱▲",
            "special": "🔥💥⚡✨💫🌟",
        },
        "nature": {
            "name": "Garden",
            "density": " ·.°•◦○◎●◉★",
            "drums":   " ·°•◦✿❀❁❂❃❄",
            "bass":    " ·~∿≈≋∼∽〰⌇▐",
            "keys":    " ·✿❀❁❂❃❄✾✺❈",
            "other":   " ·°•✿❀❁❂❃✾✺",
            "special": "🌿🌸🌺🌻🍃🌲🌳🌴",
        },
        "music": {
            "name": "Symphony",
            "density": " ·°♩♪♫♬🎵🎶🎼",
            "drums":   " ·oO●◉⬤▮▯█░",
            "bass":    " ·_~≈≋∼∽〰♩♪",
            "keys":    " ·°♩♪♫♬🎵🎶🎼",
            "other":   " ·:;♩♪♫♬🎵🎶",
            "special": "🎹🥁🎸🎻🎺🎷🪘",
        },
        "geometric": {
            "name": "Shapes",
            "density": " ·○●◯◎◉⬟⬢⬡⬠",
            "drums":   " ·△▲▴▵▶▷▸▹◊◆",
            "bass":    " ·□■▢▣▤▥▦▧▨▩",
            "keys":    " ·○●◯◎◉⊙⊚⊛⊜",
            "other":   " ·◇◆◈◊⬟⬠⬡⬢⎔",
            "special": "⬛⬜🔲🔳🔴🔵🟢🟡🟠🟣",
        },
        "braille": {
            "name": "Braille Patterns",
            "density": " ⠁⠃⠇⠏⠟⠿⡿⣿",
            "drums":   " ⠄⠆⠇⠧⠷⠿⣿⣿",
            "bass":    " ⠂⠆⠖⠶⠾⣶⣿⣿",
            "keys":    " ⠁⠃⠋⠛⠻⢻⣻⣿",
            "other":   " ⠈⠘⠸⢸⣸⣿⣿⣿",
            "special": "⣾⣷⣯⣟⡿⢿⣻⣽⣿",
        },
        "arrows": {
            "name": "Directional",
            "density": " ·→←↑↓↗↘↙↖⟿",
            "drums":   " ·▶▷►▻▸▹⏵➤➔",
            "bass":    " ·↔↕⇔⇕⟷⟺⬌⬍",
            "keys":    " ·↱↲↳↴↵↶↷↺↻",
            "other":   " ·⇢⇠⇡⇣⇤⇥⮕⬅",
            "special": "➡️⬅️⬆️⬇️↗️↘️↙️↖️🔄",
        },
        "emoji": {
            "name": "Full Emoji",
            "density": " ·⚪⚫🔴🟠🟡🟢🔵🟣",
            "drums":   " 💥⚡✨🔥💫⭐🌟🎆🎇",
            "bass":    " 🌊💧💦🌀🌌🌠🌜🌛🌝",
            "keys":    " 🎵🎶🎼🎹🎸🎺🎷🥁🎻",
            "other":   " 🌸🌺🌻🌹🌷💐🌿🍀🌲",
            "special": "🎉🎊🪩💎💍👑🏆🥇",
        },
        "tech": {
            "name": "Cyberpunk",
            "density": " ·01⌘⌥⎋⏎⌫⌦⏏",
            "drums":   " ·╳╬╪╫┼├┤┬┴┼",
            "bass":    " ·═║╔╗╚╝╠╣╦╩",
            "keys":    " ·⌘⌥⎋⏎⌫⌦⏏☰☱",
            "other":   " ·01░▒▓█⌐¬¦│",
            "special": "⚙️🔧🔩💻🖥️📱⌨️🖱️",
        },
        "weather": {
            "name": "Atmospheric",
            "density": " ·☆★◦●◉⬤▓█",
            "drums":   " ·⚡⛈️🌩️⛰️🌋🔥💥",
            "bass":    " ·💧🌊💦☔🌧️🌨️❄️",
            "keys":    " ·☀️🌤️⛅🌥️☁️🌦️🌈",
            "other":   " ·🌬️💨🌪️🌫️🌁❄️☃️",
            "special": "☀️☁️☂️☃️⚡🌈🌪️🌊",
        },
    }
    
    # Heavy emoji themes require special font handling
    EMOJI_HEAVY_THEMES = {"emoji", "fire", "nature", "music", "weather"}
    
    def __init__(self, theme: str = "classic", seed: int = None):
        """
        Initialize character pool with a theme.
        
        Args:
            theme: Theme name from THEMES, or "random" for per-section randomization
            seed: Optional seed for reproducible randomization
        """
        self.base_theme = theme
        self.current_theme = theme if theme != "random" else "classic"
        self.rng = np.random.default_rng(seed)
        self._apply_theme(self.current_theme)
    
    def _apply_theme(self, theme_name: str):
        """Apply a theme's character sets."""
        theme = self.THEMES.get(theme_name, self.THEMES["classic"])
        self.density_chars = theme["density"]
        self.drums_chars = theme["drums"]
        self.bass_chars = theme["bass"]
        self.keys_chars = theme["keys"]
        self.other_chars = theme["other"]
        self.special_chars = theme["special"]
        self.current_theme = theme_name
    
    def randomize_for_section(self, section_idx: int, prompt: str = ""):
        """
        Randomize character selection for a new section.
        If base_theme is "random", picks a new theme.
        Otherwise, shuffles within the current theme's special characters.
        """
        if self.base_theme == "random":
            # Pick a new theme based on section index
            theme_names = list(self.THEMES.keys())
            seed = int(hashlib.md5(f"{prompt}:{section_idx}".encode()).hexdigest(), 16)
            theme_idx = seed % len(theme_names)
            self._apply_theme(theme_names[theme_idx])
        else:
            # Stay with base theme but maybe shuffle special chars
            self._apply_theme(self.base_theme)
    
    def get_chars_for_track(self, track_name: str) -> str:
        """Get character set for a specific track type."""
        mapping = {
            "drums": self.drums_chars,
            "bass": self.bass_chars,
            "keys": self.keys_chars,
            "other": self.other_chars,
        }
        return mapping.get(track_name, self.density_chars)
    
    def get_density_chars(self) -> str:
        """Get the current density gradient characters."""
        return self.density_chars
    
    def get_random_special(self) -> str:
        """Get a random special character from the current theme."""
        if self.special_chars:
            idx = self.rng.integers(0, len(self.special_chars))
            return self.special_chars[idx]
        return "★"
    
    def is_emoji_heavy(self) -> bool:
        """Check if current theme uses lots of emoji (needs special font)."""
        return self.current_theme in self.EMOJI_HEAVY_THEMES
    
    @classmethod
    def list_themes(cls) -> list:
        """List all available theme names."""
        return list(cls.THEMES.keys())
    
    @classmethod
    def get_theme_info(cls, theme_name: str) -> str:
        """Get description of a theme."""
        theme = cls.THEMES.get(theme_name, {})
        return theme.get("name", "Unknown theme")


# Global pool instance (set during main)
CHAR_POOL = None


# ── Dynamic Color Palettes ───────────────────────────────────────────────────
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
    "matrix": [(0, 50, 0), (0, 128, 0), (0, 200, 0), (0, 255, 0), (144, 238, 144)],
}

# Keywords in prompt that trigger specific palettes
THEME_KEYWORDS = {
    "space": "cyber", "cosmic": "cyber", "stars": "cyber", "galaxy": "aurora",
    "fire": "fire", "flame": "fire", "burn": "fire", "hot": "fire",
    "water": "ocean", "ocean": "ocean", "sea": "ocean", "wave": "ocean",
    "neon": "neon", "electric": "neon", "synth": "neon", "retro": "neon",
    "sunset": "sunset", "dawn": "sunset", "dusk": "sunset",
    "psychedelic": "psychedelic", "trippy": "psychedelic", "acid": "psychedelic",
    "aurora": "aurora", "northern": "aurora", "lights": "aurora",
    "blood": "blood", "dark": "blood", "evil": "blood", "death": "blood",
    "toxic": "toxic", "poison": "toxic", "matrix": "matrix", "hack": "matrix",
    "ice": "ice", "cold": "ice", "frozen": "ice", "winter": "ice",
    "vapor": "vaporwave", "aesthetic": "vaporwave",
}

# Base hue per track (0-360) - fallback
TRACK_HUES = {
    "drums": 15,    # orange/red
    "bass":  260,   # purple/blue
    "keys":  130,   # green/cyan
    "other": 45,    # gold/yellow
}

# Layer compositing opacity (bottom to top)
LAYER_OPACITY = {
    "drums": 1.0,
    "bass":  0.65,
    "keys":  0.50,
    "other": 0.40,
}

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

        self._compute_features()
        
        if detect_sections:
            self._detect_sections()
        else:
            self.section_boundaries = []
            self.section_frames = []
            
        self._detect_beats()

    def _compute_features(self):
        """Compute loudness, spectral centroid, onset strength, spectral bandwidth."""
        frame_length = max(self.hop * 2, 2048)
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=self.hop)[0]
        self.loudness = self._normalize(rms[:self.n_frames])

        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, hop_length=self.hop)[0]
        self.brightness = self._normalize(cent[:self.n_frames])

        onset = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop)
        self.onset = self._normalize(onset[:self.n_frames])

        bw = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr, hop_length=self.hop)[0]
        self.bandwidth = self._normalize(bw[:self.n_frames])

        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=self.hop)
        self.chroma_peak = np.argmax(self.chroma[:, :self.n_frames], axis=0)

        S = np.abs(librosa.stft(self.y, hop_length=self.hop))
        flux = np.sqrt(np.mean(np.diff(S, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])
        self.flux = self._normalize(flux[:self.n_frames])

    def _detect_sections(self):
        """Detect song sections using spectral clustering on chroma features."""
        try:
            R = librosa.segment.recurrence_matrix(self.chroma, mode='affinity', 
                                                   sym=True, sparse=True)
            n_sections = max(4, min(12, int(self.duration / 15)))
            bounds = librosa.segment.agglomerative(self.chroma, n_sections)
            
            chroma_times = librosa.frames_to_time(bounds, sr=self.sr, hop_length=self.hop)
            self.section_boundaries = (chroma_times * self.fps).astype(int)
            self.section_boundaries = self.section_boundaries[self.section_boundaries < self.n_frames]
            self.section_boundaries = np.unique(np.concatenate([[0], self.section_boundaries]))
            
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
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.hop)
            self.beat_frames = set((beat_times * self.fps).astype(int))
            self.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]) if len(tempo) > 0 else 120.0
            print(f"   Detected {len(self.beat_frames)} beats, tempo ~{self.tempo:.0f} BPM")
        except Exception as e:
            print(f"   ⚠ Beat detection failed: {e}")
            self.beat_frames = set()
            self.tempo = 120.0

    @staticmethod
    def _normalize(arr):
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def get_frame(self, frame_idx: int) -> dict:
        i = min(frame_idx, self.n_frames - 1)
        is_section_boundary = any(abs(i - b) <= 2 for b in self.section_boundaries)
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
            "section": int(self.section_frames[i]) if len(self.section_frames) > i else 0,
            "is_section_boundary": is_section_boundary,
            "is_beat": is_beat,
            "tempo": self.tempo,
        }


# ── Animation Renderers ─────────────────────────────────────────────────────

class BaseVisualizer:
    """Base class for track-specific Unicode animation with palette support."""

    def __init__(self, width: int, height: int, track_name: str, seed_prompt: str = None, palette_name: str = None):
        global CHAR_POOL
        self.w = width
        self.h = height
        self.track = track_name
        self.seed_prompt = seed_prompt or "Entropy"
        self.base_hue = TRACK_HUES.get(track_name, 180)
        
        # Get characters from global pool
        self.chars = CHAR_POOL.get_chars_for_track(track_name) if CHAR_POOL else " .·:;=+*#%@█"
        
        self.rng = np.random.default_rng(
            int(hashlib.md5((track_name + self.seed_prompt).encode()).hexdigest()[:8], 16)
        )
        self.grid_x, self.grid_y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2*height/width, 2*height/width, height))
        
        self.palette_name = palette_name or self._detect_palette_from_prompt()
        self.palette = PALETTES.get(self.palette_name, PALETTES["psychedelic"])
        self.current_section = -1
        
    def _detect_palette_from_prompt(self) -> str:
        prompt_lower = self.seed_prompt.lower()
        for keyword, palette in THEME_KEYWORDS.items():
            if keyword in prompt_lower:
                return palette
        phash = int(hashlib.md5(self.seed_prompt.encode()).hexdigest(), 16)
        palette_names = list(PALETTES.keys())
        return palette_names[phash % len(palette_names)]
    
    def set_palette(self, palette_name: str):
        self.palette_name = palette_name
        self.palette = PALETTES.get(palette_name, PALETTES["psychedelic"])
    
    def get_density_chars(self) -> str:
        """Proxy for CHAR_POOL.get_density_chars()"""
        return CHAR_POOL.get_density_chars() if CHAR_POOL else " .:-=+*#%@"

    def get_random_special(self) -> str:
        """Proxy for CHAR_POOL.get_random_special()"""
        return CHAR_POOL.get_random_special() if CHAR_POOL else "*"

    def refresh_chars(self):
        """Refresh character set from global pool (called at section changes)."""
        global CHAR_POOL
        if CHAR_POOL:
            self.chars = CHAR_POOL.get_chars_for_track(self.track)
    
    def reset_for_section(self, section_idx: int, prompt: str = ""):
        """Reset internal state for a new section - creates visual variety."""
        self.current_section = section_idx
        self.refresh_chars()
        # Reinitialize random state with section-specific seed
        seed = int(hashlib.md5(f"{self.track}:{prompt}:{section_idx}".encode()).hexdigest()[:8], 16)
        self.rng = np.random.default_rng(seed)
        # Subclasses can override to reset their specific state
        self._reset_internal_state(section_idx, prompt)
    
    def _reset_internal_state(self, section_idx: int, prompt: str):
        """Override in subclasses to reset renderer-specific state."""
        pass
        
    def reset(self, section_idx: int):
        self.current_section = section_idx
        self.refresh_chars()

    def render_frame(self, feat: dict) -> list:
        raise NotImplementedError

    def _hue_for_frame(self, feat: dict) -> float:
        chroma_shift = (feat["chroma"] / 12.0) * 30
        hue = (self.base_hue + chroma_shift) % 360
        return hue / 360.0

    def _color(self, hue: float, saturation: float, value: float) -> tuple:
        # Ensure minimum brightness to prevent invisible characters
        value = max(0.25, value)  # Floor at 25% brightness
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _palette_color(self, t: float, brightness: float = 1.0) -> tuple:
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
        
        # Ensure minimum brightness to prevent invisible characters
        brightness = max(0.35, brightness)  # Floor at 35%
        return tuple(int(min(255, max(50, v * brightness))) for v in c)  # Min 50 per channel


# ── ANIMATION RENDERERS (REMOVED LEGACY CLASSES) ──────────────────────────────
# Julia, Plasma, Spiral, Starfield replaced by ProceduralRenderer

class FoundationRenderer(BaseVisualizer):
    """
    Renders the foundation layer: a unique-per-run procedural pixel-art landscape
    that degrades into chaos.
    """

    def __init__(self, w, h, seed_prompt, total_frames):
        super().__init__(w, h, "foundation", seed_prompt)
        self.total_frames = total_frames
        
        # ── LANDSCAPE GENERATION PARAMETERS ──────────────────────────────────
        # Expanded Biome System
        self.biome_type = self.rng.choice(['forest', 'desert', 'ice', 'cyber', 'ocean'])
        
        # Sub-biomes for variety
        if self.biome_type == 'forest':
            self.subtype = self.rng.choice(['deciduous', 'autumn', 'dead', 'magic'])
        elif self.biome_type == 'desert':
            self.subtype = self.rng.choice(['sandy', 'red_rock', 'white_sands'])
        elif self.biome_type == 'ice':
            self.subtype = self.rng.choice(['glacial', 'snowy', 'tundra'])
        elif self.biome_type == 'cyber':
            self.subtype = self.rng.choice(['neon', 'matrix', 'gold'])
        else: # ocean
            self.subtype = self.rng.choice(['tropical', 'stormy', 'alien', 'frozen'])

        # Terrain generation parameters (1D Fractal Noise)
        self.terrain_seed = self.rng.integers(0, 10000)
        self.terrain_roughness = self.rng.uniform(0.3, 0.7)
        self.terrain_height_offset = self.rng.uniform(-0.2, 0.2) # Horizon shift
        
        # Celestial Bodies (Sun/Moon)
        self.sun_x = self.rng.uniform(0.1, 0.9)
        self.sun_y = self.rng.uniform(0.1, 0.4) # High up
        self.sun_size = self.rng.uniform(0.05, 0.15)
        self.has_moon = self.rng.random() > 0.5
        
        # Cloud/Sky parameters
        self.cloud_seed = self.rng.integers(0, 10000)
        self.sky_gradient_type = self.rng.choice(['day', 'sunset', 'night', 'alien'])
        
        print(f"   Foundation: {self.biome_type.upper()} Landscape ({self.subtype})")

    def _get_terrain_height(self, x: float) -> float:
        """1D Fractal Noise for terrain height at normalized x position (0-1)."""
        height = 0.0
        amp = 1.0
        freq = 3.0
        
        for _ in range(4): # 4 octaves
            # Simple sine-based pseudo-noise for reproducibility without external lib
            val = math.sin(x * freq + self.terrain_seed) * 0.5 + 0.5
            val += math.sin(x * freq * 2.3 + self.terrain_seed * 1.5) * 0.25
            height += val * amp
            amp *= self.terrain_roughness
            freq *= 2.0
            
        # Normalize roughly to 0-1 range
        return height * 0.3 + 0.4 + self.terrain_height_offset

    def _get_sky_color(self, y: float, time: float, chaos: float) -> tuple:
        """Get sky color at normalized height y (0=top, 1=horizon)."""
        # Time-of-day shift
        t_shift = math.sin(time * 0.1) * 0.2
        
        if self.sky_gradient_type == 'day':
            top = (0, 100, 255)
            bot = (135, 206, 250)
        elif self.sky_gradient_type == 'sunset':
            top = (75, 0, 130)
            bot = (255, 140, 0)
        elif self.sky_gradient_type == 'night':
            top = (0, 0, 20)
            bot = (25, 25, 112)
        else: # alien
            top = (0, 50, 20)
            bot = (255, 0, 128)
            
        # Randomize sky colors slightly per run
        if self.rng.random() < 0.5:
             top = (min(255, top[0]+20), min(255, top[1]+20), min(255, top[2]+20))

        # Interpolate
        r = int(top[0] + (bot[0] - top[0]) * y)
        g = int(top[1] + (bot[1] - top[1]) * y)
        b = int(top[2] + (bot[2] - top[2]) * y)
        
        # Chaos Shift
        if chaos > 0.5:
            r = (r + int(chaos * 100)) % 255
            g = (g - int(chaos * 50)) % 255
        
        return (r, g, b)

    def render_frame(self, feat: dict, frame_idx: int = 0) -> list:
        progress = frame_idx / max(1, self.total_frames)
        t = feat["time"]
        loud = feat["loudness"]
        
        # Descent into Chaos curve
        chaos = max(0, (progress - 0.4) / 0.6)
        
        grid = []
        
        # ── BIOME COLOR/CHAR DEFINITIONS ─────────────────────────────────────
        # Default fallbacks
        ground_char = '░'
        ground_color = (100, 100, 100)
        feature_char = '?'
        
        if self.biome_type == 'forest':
            if self.subtype == 'autumn':
                 ground_char, ground_color, feature_char = '🍂', (139, 69, 19), '🍁'
            elif self.subtype == 'dead':
                 ground_char, ground_color, feature_char = '🕸', (105, 105, 105), '💀'
            elif self.subtype == 'magic':
                 ground_char, ground_color, feature_char = '🍄', (75, 0, 130), '✨'
            else: # deciduous
                 ground_char, ground_color, feature_char = '🌲', (34, 139, 34), '🌳'

        elif self.biome_type == 'desert':
            if self.subtype == 'red_rock':
                ground_char, ground_color, feature_char = '▒', (165, 42, 42), '🏜️'
            elif self.subtype == 'white_sands':
                ground_char, ground_color, feature_char = '░', (245, 245, 245), '🌵'
            else: # sandy
                ground_char, ground_color, feature_char = '░', (210, 180, 140), '🐪'

        elif self.biome_type == 'ice':
             ground_char, ground_color, feature_char = '❄️', (240, 248, 255), '🏔️'
             if self.subtype == 'tundra': ground_color = (143, 188, 143)

        elif self.biome_type == 'cyber':
            if self.subtype == 'matrix':
                 ground_char, ground_color, feature_char = '0', (0, 20, 0), '1'
            elif self.subtype == 'gold':
                 ground_char, ground_color, feature_char = '▚', (50, 40, 0), '👑'
            else: # neon
                 ground_char, ground_color, feature_char = '▦', (20, 0, 40), '🏢'

        else: # ocean
            if self.subtype == 'stormy':
                ground_char, ground_color, feature_char = '≈', (47, 79, 79), '⚡'
            elif self.subtype == 'alien':
                ground_char, ground_color, feature_char = '≈', (75, 0, 130), '👾'
            elif self.subtype == 'frozen':
                ground_char, ground_color, feature_char = '≈', (176, 224, 230), '🧊'
            else: # tropical
                ground_char, ground_color, feature_char = '≈', (0, 105, 148), '🌴'
                
        # Chaos overrides
        if chaos > 0.5:
             ground_char = '☠' if self.rng.random() < 0.1 else ground_char

        for row in range(self.h):
            grid_row = []
            ny = row / self.h # 0 at top, 1 at bottom
            
            for col in range(self.w):
                nx = col / self.w
                
                # Apply Chaos Warping to coordinates
                if chaos > 0:
                    nx += math.sin(ny * 10 + t) * (chaos * 0.1)
                    ny += math.cos(nx * 10 - t) * (chaos * 0.1)
                
                # Aspect
                aspect = self.w / self.h * 0.5 
                
                # ── SKY RENDERING ────────────────────────────────────────────
                bg_r, bg_g, bg_b = self._get_sky_color(ny, t, chaos)
                
                # Default: Empty char with Sky Background
                char = ' '
                r, g, b = 0, 0, 0
                br, bg, bb = bg_r, bg_g, bg_b
                
                # Sun/Moon SDF
                sun_dx = nx - self.sun_x
                sun_dy = ny - self.sun_y
                sun_dist = math.sqrt(sun_dx*sun_dx + sun_dy*sun_dy)
                
                if sun_dist < self.sun_size:
                    # Sun body
                    char = '●'
                    r, g, b = 255, 255, 0 # Yellow sun default
                    if self.sky_gradient_type == 'night':
                        r, g, b = 200, 200, 200 # White moon
                    elif self.sky_gradient_type == 'alien':
                        r, g, b = 100, 255, 100 # Green sun
                
                # ── TERRAIN RENDERING ────────────────────────────────────────
                # Get terrain height at this x
                terrain_y = self._get_terrain_height(nx)
                
                if ny > terrain_y:
                    # Ground
                    depth = (ny - terrain_y) / (1.0 - terrain_y) 
                    
                    # Distance fading
                    cur_ground_color = list(ground_color)
                    # Fade to sky color at horizon
                    for i in range(3):
                        cur_ground_color[i] = int(cur_ground_color[i] * depth + bg_r * (1-depth))
                    
                    r, g, b = tuple(cur_ground_color)
                    char = ground_char
                    
                    # Ground Background: Darker version of ground color for solid feel
                    br = max(0, r - 40)
                    bg = max(0, g - 40)
                    bb = max(0, b - 40)
                    
                    # Random features on ground
                    feature_noise = math.sin(nx * 50 + row * 10) 
                    if feature_noise > 0.95 - (depth * 0.2): 
                        char = feature_char
                        r = min(255, r + 50)
                        g = min(255, g + 50)
                        b = min(255, b + 50)
                        
                # ── CHAOS OVERLAY ────────────────────────────────────────────
                # ── CHAOS OVERLAY ────────────────────────────────────────────
                if self.rng.random() < (chaos * chaos * 0.2):
                    char = self.get_random_special()
                    # Glitch colors (cast to int bc rng returns np.int64 which PIL hates)
                    r = int(self.rng.integers(0, 255))
                    g = int(self.rng.integers(0, 255))
                    b = int(self.rng.integers(0, 255))
                    # Random glitch background
                    br = int(self.rng.integers(0, 100))
                    bg = int(self.rng.integers(0, 100))
                    bb = int(self.rng.integers(0, 100))

                grid_row.append((char, r, g, b, br, bg, bb))
            grid.append(grid_row)
        return grid


def select_palette_for_section(prompt: str, section_idx: int) -> str:
    palette_names = list(PALETTES.keys())
    
    prompt_lower = prompt.lower()
    for keyword, palette in THEME_KEYWORDS.items():
        if keyword in prompt_lower:
            base_idx = palette_names.index(palette) if palette in palette_names else 0
            return palette_names[(base_idx + section_idx) % len(palette_names)]
    
    seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    return palette_names[(seed + section_idx) % len(palette_names)]


# ── Compositing ──────────────────────────────────────────────────────────────

def composite_layers(layers: dict, width: int, height: int) -> list:
    merged = []
    for row in range(height):
        line = []
        for col in range(width):
            best_char = ' '
            final_r, final_g, final_b = 0, 0, 0
            total_weight = 0
            
            for track in LAYER_ORDER:
                if track not in layers:
                    continue
                cell = layers[track][row][col]
                char, r, g, b = cell
                opacity = LAYER_OPACITY.get(track, 0.5)
                
                brightness = (r + g + b) / (3 * 255.0)
                weight = brightness * opacity
                
                final_r += r * opacity
                final_g += g * opacity
                final_b += b * opacity
                total_weight += opacity
                
                if weight > 0.1 and (char != ' '):
                    best_char = char
            
            if total_weight > 0:
                final_r = min(255, int(final_r / total_weight))
                final_g = min(255, int(final_g / total_weight))
                final_b = min(255, int(final_b / total_weight))
            
            line.append((best_char, final_r, final_g, final_b))
        merged.append(line)
    return merged


def calculate_dynamic_layer_opacity(progress: float, total_sections: int, current_section: int) -> float:
    """
    Calculate the dynamic layer opacity based on song progress.
    
    Curve:
    - Start: 0% opacity
    - First major section change (~12.5%): 33% opacity
    - 75% of sections complete: 66% opacity (peak)
    - End of song: 0% opacity (fade out)
    
    Returns opacity as 0.0 to 1.0
    """
    # Key points in the curve
    first_major = 0.125   # 12.5% of song
    peak_point = 0.75     # 75% of song
    
    if progress < first_major:
        # Fade in: 0% → 33%
        return (progress / first_major) * 0.33
    elif progress < peak_point:
        # Gradual increase: 33% → 66%
        blend = (progress - first_major) / (peak_point - first_major)
        return 0.33 + blend * 0.33
    else:
        # Fade out: 66% → 0%
        blend = (progress - peak_point) / (1.0 - peak_point)
        return 0.66 * (1.0 - blend)


def composite_two_layers(foundation_grid: list, dynamic_grid: list, 
                         dynamic_opacity: float, width: int, height: int) -> list:
    """
    Composite foundation and dynamic layers with alpha blending.
    
    Args:
        foundation_grid: The base layer grid (always visible)
        dynamic_grid: The top layer grid (with opacity)
        dynamic_opacity: Opacity of dynamic layer (0.0 to 1.0)
        width, height: Grid dimensions
    
    Returns:
        Composited grid
    """
    result = []
    
    for row in range(height):
        line = []
        for col in range(width):
            # Get cells from both layers
            f_char, f_r, f_g, f_b = foundation_grid[row][col]
            d_char, d_r, d_g, d_b = dynamic_grid[row][col]
            
            # If dynamic layer is completely transparent or has no character
            if dynamic_opacity < 0.01 or d_char == ' ':
                line.append((f_char, f_r, f_g, f_b))
                continue
            
            # Alpha blend colors
            alpha = dynamic_opacity
            final_r = int(f_r * (1 - alpha) + d_r * alpha)
            final_g = int(f_g * (1 - alpha) + d_g * alpha)
            final_b = int(f_b * (1 - alpha) + d_b * alpha)
            
            # Choose character: if dynamic has significant opacity, use its char
            # Otherwise blend based on which is brighter
            if alpha > 0.4:
                final_char = d_char if d_char != ' ' else f_char
            else:
                # Use brightness to decide
                f_brightness = (f_r + f_g + f_b) / 765
                d_brightness = (d_r + d_g + d_b) / 765 * alpha
                final_char = d_char if d_brightness > f_brightness and d_char != ' ' else f_char
            
            line.append((final_char, final_r, final_g, final_b))
        result.append(line)
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN LIBRARY - 30+ Procedural Micro-Patterns
# ══════════════════════════════════════════════════════════════════════════════

class PatternLibrary:
    """
    Collection of procedural pattern generators.
    Each function signature: f(x, y, t, energy, entropy, params) -> intensity (0.0-1.0)
    x, y: Normalized coordinates (-1.0 to 1.0)
    t: Time in seconds
    energy: Audio energy (loudness/onset) 0.0-1.0
    entropy: Section progress/chaos factor 0.0-1.0
    params: Random parameters dict specific to the pattern instance
    """
    
    @staticmethod
    def _rotate(x, y, angle):
        c, s = math.cos(angle), math.sin(angle)
        return x * c - y * s, x * s + y * c

    @staticmethod
    def _noise(x, y, seed=0):
        # Pseudo-random noise
        n = math.sin(x * 12.9898 + y * 78.233 + seed) * 43758.5453
        return n - math.floor(n)

    # ── GEOMETRIC PATTERNS ──────────────────────────────────────────────────────

    @staticmethod
    def concentric_squares(x, y, t, energy, entropy, p):
        # Zooming squares
        zoom = (t * 0.5) % 1.0
        dist = max(abs(x), abs(y))
        
        # Warp with entropy
        if entropy > 0.3:
            dist += 0.1 * entropy * math.sin(math.atan2(y, x) * 4 + t)
            
        val = math.sin((dist - zoom) * 20.0)
        return 0.5 + 0.5 * val

    @staticmethod
    def hex_grid(x, y, t, energy, entropy, p):
        # Hexagonal grid
        # Skew coordinates for hex grid
        u = x * math.cos(math.pi/6) + y * math.sin(math.pi/6)
        v = y
        
        scale = 5.0 + energy * 2.0
        u *= scale
        v *= scale
        
        u = abs(u - math.floor(u + 0.5))
        v = abs(v - math.floor(v + 0.5))
        dist = max(u, v, abs(u - v))
        
        thickness = 0.1 + entropy * 0.2
        return 1.0 if dist > (0.5 - thickness) else 0.0

    @staticmethod
    def sine_waves(x, y, t, energy, entropy, p):
        # Intersecting sine waves
        val = math.sin(x * 10 + t) + math.sin(y * 10 - t * 0.5)
        val += math.sin((x + y) * 5 + t) * entropy  # Add diagonal chaos
        return (val + 2) / 4.0

    @staticmethod
    def radial_spokes(x, y, t, energy, entropy, p):
        angle = math.atan2(y, x)
        dist = math.sqrt(x*x + y*y)
        
        spokes = 12 + int(entropy * 10)
        val = math.sin(angle * spokes + t * 2 + dist * 5 * entropy)
        return 0.5 + 0.5 * val

    @staticmethod
    def checkerboard_warp(x, y, t, energy, entropy, p):
        # Warped checkerboard
        warp = math.sin(t) * 0.5 + energy
        x += math.sin(y * 5 + t) * 0.2 * warp
        y += math.cos(x * 5 - t) * 0.2 * warp
        
        check = math.floor(x * 5) + math.floor(y * 5)
        return 1.0 if check % 2 == 0 else 0.0

    @staticmethod
    def moire_patterns(x, y, t, energy, entropy, p):
        dist1 = math.sqrt((x-0.2)**2 + (y-0.2)**2)
        dist2 = math.sqrt((x+0.2)**2 + (y+0.2)**2)
        
        freq = 30.0 * (1.0 - entropy * 0.5)
        v1 = math.sin(dist1 * freq + t)
        v2 = math.sin(dist2 * freq - t)
        
        return (v1 * v2 + 1) / 2

    @staticmethod
    def islamic_star_grid(x, y, t, energy, entropy, p):
        # Simplified geometric star pattern
        scale = 3.0
        x, y = x * scale, y * scale
        
        # Symmetry folding
        x, y = abs(x), abs(y)
        if x < y: x, y = y, x
        
        val = math.sin(x * 4 + t) * math.cos(y * 4)
        if entropy > 0.5:
            val += math.sin(x * 10) * (entropy - 0.5)
            
        return (val + 1) / 2

    @staticmethod
    def triangle_tessellation(x, y, t, energy, entropy, p):
        scale = 5.0
        ys = y * scale * 0.866
        xs = x * scale + y * scale * 0.5
        
        part_x = xs - math.floor(xs)
        part_y = ys - math.floor(ys)
        
        if part_x + part_y > 1.0:
            part_x = 1.0 - part_x
            part_y = 1.0 - part_y
            
        dist = min(part_x, part_y, 1.0 - part_x - part_y)
        edge = 0.05 + energy * 0.1
        return 1.0 if dist < edge else 0.0

    @staticmethod
    def circuit_paths(x, y, t, energy, entropy, p):
        # Manhattan distance structure
        scale = 8.0
        ix, iy = math.floor(x * scale), math.floor(y * scale)
        coin = (ix * 1341 + iy * 2351) % 2
        
        fx, fy = x * scale - ix, y * scale - iy
        dist = abs(fx - fy) if coin == 0 else abs(fx - (1 - fy))
        
        width = 0.1 + energy * 0.2
        return 1.0 if dist < width else 0.0

    @staticmethod
    def spiral_galaxy(x, y, t, energy, entropy, p):
        r = math.sqrt(x*x + y*y)
        a = math.atan2(y, x)
        
        spiral = a + r * 5.0 + t - entropy * r * 5.0
        val = math.sin(spiral * 3.0)
        
        return (val + 1) / 2 * (1.0 - r * 0.8)

    # ── ORGANIC PATTERNS ────────────────────────────────────────────────────────

    @staticmethod
    def cell_noise(x, y, t, energy, entropy, p):
        # Cellular / Worley noise-ish
        scale = 4.0
        min_dist = 1.0
        
        # Check neighbor cells
        ix, iy = math.floor(x * scale), math.floor(y * scale)
        fx, fy = x * scale - ix, y * scale - iy
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                # Random point in cell
                seed_x = ix + dx
                seed_y = iy + dy
                # Pseudo random offset
                rx = (math.sin(seed_x * 12.989 + seed_y * 78.233 + t * 0.5) * 0.5 + 0.5)
                ry = (math.cos(seed_x * 43.123 + seed_y * 91.111 + t * 0.3) * 0.5 + 0.5)
                
                d = math.sqrt((dx + rx - fx)**2 + (dy + ry - fy)**2)
                min_dist = min(min_dist, d)
        
        edge = 1.0 - entropy
        return 1.0 - min_dist if min_dist < edge else 0.0

    @staticmethod
    def liquid_waves(x, y, t, energy, entropy, p):
        for i in range(1, 4):
            x += math.sin(y * i + t) * 0.2
            y += math.cos(x * i + t) * 0.2
            
        val = math.sin(x * 5 + t) * math.cos(y * 5)
        # Entropy adds turbulence
        if entropy > 0:
            val += PatternLibrary._noise(x, y) * entropy
        return (val + 1) / 2

    @staticmethod
    def veins(x, y, t, energy, entropy, p):
        val = 0.0
        scale = 3.0
        amp = 1.0
        for i in range(4):
            val += math.sin(x * scale + t) * math.cos(y * scale) * amp
            scale *= 2.0
            amp *= 0.5
            
        # Ridge noise
        return 1.0 - abs(val)

    @staticmethod
    def smoke(x, y, t, energy, entropy, p):
        # FBM noise
        val = 0.0
        scale = 2.0
        amp = 1.0
        x += t * 0.2  # Drift
        
        for i in range(5):
            val += (math.sin(x * scale) + math.cos(y * scale)) * amp
            scale *= 2.0
            amp *= 0.5
            
        val = val * 0.5 + 0.5
        # Energy brightens core
        return max(0, min(1, val * (1 + energy)))

    @staticmethod
    def reaction_diffusion(x, y, t, energy, entropy, p):
        # Simulated pattern
        d = math.sqrt(x*x + y*y)
        val = math.sin(d * 10 - t * 2) + math.cos(math.atan2(y, x) * 5)
        
        # Entropy breaks symmetry
        if entropy > 0.4:
            val += math.sin(x * 20) * entropy
            
        return (val + 2) / 4.0

    @staticmethod
    def bacterial_growth(x, y, t, energy, entropy, p):
        # Noisy clumps
        d = math.sqrt(x*x + y*y)
        angle = math.atan2(y, x)
        
        radius = 0.5 + PatternLibrary._noise(angle, t * 0.1) * 0.4
        
        if d < radius:
            # Interior texture
            return PatternLibrary._noise(x * 10, y * 10)
        return 0.0

    @staticmethod
    def slime_mold(x, y, t, energy, entropy, p):
        # Network-like
        val = abs(math.sin(x * 5 + math.sin(y * 5 + t)))
        val = 1.0 - val
        val = pow(val, 4.0)  # Thin lines
        return val

    # ── COSMIC PATTERNS ─────────────────────────────────────────────────────────

    @staticmethod
    def starfield_zoom(x, y, t, energy, entropy, p):
        # Radial streaks
        a = math.atan2(y, x)
        d = math.sqrt(x*x + y*y)
        
        # Zoom effect
        z = (d - t * 0.5) % 1.0
        z = 1.0 - z  # Invert so stars come towards camera
        
        val = PatternLibrary._noise(a * 10, z * 20)
        if val > 0.95:
            return (val - 0.95) * 20
        return 0.0

    @staticmethod
    def wormhole(x, y, t, energy, entropy, p):
        a = math.atan2(y, x)
        d = math.sqrt(x*x + y*y)
        
        # Twisting tunnel
        twist = a + 1.0 / (d + 0.1) * (1 + energy) + t
        val = math.sin(twist * 5)
        
        return (val + 1) / 2 * d

    @staticmethod
    def matrix_rain(x, y, t, energy, entropy, p):
        # Vertical dripping columns
        col = math.floor(x * 20)
        
        # Speed varies by column
        speed = 1.0 + PatternLibrary._noise(col, 0)
        y_offset = (y + t * speed) % 2.0
        
        brightness = max(0, 1.0 - abs(y_offset - 1.0) * 2.0)
        
        if entropy > 0.5:
            # Glitch columns horizontally
            brightness *= (1.0 + math.sin(y * 50) * entropy)
            
        return brightness if brightness > 0.1 else 0.0

    @staticmethod
    def solar_flares(x, y, t, energy, entropy, p):
        a = math.atan2(y, x)
        d = math.sqrt(x*x + y*y)
        
        # Explosive radial noise
        noise = PatternLibrary._noise(a * 5, t)
        radius = 0.3 + energy * 0.2 + noise * 0.2
        
        if d < radius:
            return 1.0
        # Corona
        return max(0, 1.0 - (d - radius) * 5)

    @staticmethod
    def hyperspace(x, y, t, energy, entropy, p):
        # Streaks from center
        x /= max(0.01, abs(x)) * 0.1  # Perspective division
        y /= max(0.01, abs(y)) * 0.1
        
        val = PatternLibrary._noise(x + t, y)
        if val > 0.8:
            return 1.0
        return 0.0

    # ── CHAOS / GLITCH PATTERNS ────────────────────────────────────────────────

    @staticmethod
    def tv_static_bands(x, y, t, energy, entropy, p):
        # Horizontal bands of noise
        row = math.floor(y * 10 + t * 5)
        if row % 2 == 0:
            return PatternLibrary._noise(x * 50, t * 10) * entropy
        return PatternLibrary._noise(x * 10, y * 10) * 0.2

    @staticmethod
    def pixel_sort(x, y, t, energy, entropy, p):
        # Stretching pixels downwards
        if PatternLibrary._noise(x * 10, 0) > 0.5:
            y = (y + t * 0.5) % 1.0
        
        return PatternLibrary._noise(x * 20, y * 20)

    @staticmethod
    def data_mosh(x, y, t, energy, entropy, p):
        # Blocky compression artifacts
        block_size = 5.0 - entropy * 4.0 # Smaller blocks with entropy
        bx = math.floor(x * block_size) / block_size
        by = math.floor(y * block_size) / block_size
        
        # Color drag
        dv = math.sin(bx * 10 + t)
        return (dv + 1) / 2

    @staticmethod
    def broken_voronoi(x, y, t, energy, entropy, p):
        # Voronoi but vertices jiggle disjointedly
        min_d = 1.0
        for i in range(5):
            sx = math.sin(i * 123.4 + t * (1+entropy)) * 0.8
            sy = math.cos(i * 321.4 + t * (1+entropy)) * 0.8
            d = math.sqrt((x-sx)**2 + (y-sy)**2)
            min_d = min(min_d, d)
            
        return 1.0 - min_d

    @staticmethod
    def quantized_ripples(x, y, t, energy, entropy, p):
        d = math.sqrt(x*x + y*y)
        val = math.sin(d * 20 - t * 5)
        
        # Hard quantization
        steps = 4 - entropy * 3 # Fewer steps = blockier
        if steps < 1: steps = 1
        val = math.floor(val * steps) / steps
        
        return (val + 1) / 2

    @staticmethod
    def feedback_loop(x, y, t, energy, entropy, p):
        # recursive rotation
        for i in range(3):
            x, y = PatternLibrary._rotate(x, y, t * 0.1 + entropy)
            x = abs(x) - 0.5
            
        return max(0, 1.0 - math.sqrt(x*x + y*y) * 2)

    @staticmethod
    def get_all_patterns():
        """Retrieve all pattern functions."""
        return [
            # Geometric
            PatternLibrary.concentric_squares, PatternLibrary.hex_grid, 
            PatternLibrary.sine_waves, PatternLibrary.radial_spokes, 
            PatternLibrary.checkerboard_warp, PatternLibrary.moire_patterns,
            PatternLibrary.islamic_star_grid, PatternLibrary.triangle_tessellation,
            PatternLibrary.circuit_paths, PatternLibrary.spiral_galaxy,
            
            # Organic
            PatternLibrary.cell_noise, PatternLibrary.liquid_waves,
            PatternLibrary.veins, PatternLibrary.smoke,
            PatternLibrary.reaction_diffusion, PatternLibrary.bacterial_growth,
            PatternLibrary.slime_mold,
            
            # Cosmic
            PatternLibrary.starfield_zoom, PatternLibrary.wormhole,
            PatternLibrary.matrix_rain, PatternLibrary.solar_flares,
            PatternLibrary.hyperspace,
            
            # Chaos
            PatternLibrary.tv_static_bands, PatternLibrary.pixel_sort,
            PatternLibrary.data_mosh, PatternLibrary.broken_voronoi,
            PatternLibrary.quantized_ripples, PatternLibrary.feedback_loop
        ]


# ── PROCEDURAL RENDERER ──────────────────────────────────────────────────────

class ProceduralRenderer(BaseVisualizer):
    """
    Universal renderer that uses PatternLibrary functions.
    Replaces the specific class implementations with a flexible engine.
    """
    
    def __init__(self, w, h, track_name, seed_prompt=None):
        super().__init__(w, h, track_name, seed_prompt=seed_prompt)
        self.all_patterns = PatternLibrary.get_all_patterns()
        self.current_patterns = []
        self.pattern_params = {}
        self.params = {}
        
        # Initialize with random patterns
        self.reset_for_section(0, seed_prompt)
        
    def _reset_internal_state(self, section_idx: int, prompt: str):
        """Pick new patterns and parameters for this section."""
        # Pick 1-2 patterns to blend
        n_patterns = 1 if self.rng.random() > 0.3 else 2
        
        # Seed rng for pattern selection based on section to keep it deterministic per run
        # but varied across sections
        self.current_patterns = []
        
        # Shuffle patterns using section seed
        indices = list(range(len(self.all_patterns)))
        self.rng.shuffle(indices)
        
        for i in range(n_patterns):
            self.current_patterns.append(self.all_patterns[indices[i]])
            
        # Random parameters for patterns
        self.params = {
            "scale": self.rng.random() * 2.0 + 0.5,
            "speed": self.rng.random() * 1.5 + 0.5,
            "complexity": self.rng.random(),
            "rotation": self.rng.random() * math.pi * 2,
            "offset_x": self.rng.random() * 2 - 1,
            "offset_y": self.rng.random() * 2 - 1
        }
        
    def render_frame(self, feat: dict) -> list:
        t = feat["time"] * self.params["speed"]
        loud = feat["loudness"]
        entropy = feat.get("section_progress", 0.0) # 0 to 1 over section
        hue = self._hue_for_frame(feat)
        
        # Entropy adds chaos to time and params
        eff_t = t + entropy * math.sin(t) * 2.0
        
        grid = []
        
        # Pre-calc constants
        hw = self.w / 2
        hh = self.h / 2
        
        for row in range(self.h):
            line = []
            
            # Row coordinate (-1 to 1)
            ny = (row - hh) / hh
            
            for col in range(self.w):
                # Col coordinate (-1 to 1)
                nx = (col - hw) / hw
                
                # Aspect correction
                nx *= (self.w / self.h) * 0.5 # 0.5 factor to handle char aspect ratio approx
                
                # Apply rotation/offset
                rx, ry = PatternLibrary._rotate(nx, ny, self.params["rotation"] + eff_t * 0.1 * entropy)
                rx += self.params["offset_x"]
                ry += self.params["offset_y"]
                
                # Evaluate patterns
                val = 0.0
                for pat in self.current_patterns:
                    val += pat(rx, ry, eff_t, loud, entropy, self.params)
                
                if len(self.current_patterns) > 1:
                    val /= len(self.current_patterns)
                    
                # Audio reactivity
                val *= (0.6 + loud * 0.8) # Loudness boosts value significantly
                
                # Hard floor
                if val < 0.1:
                    line.append((' ', 0, 0, 0))
                    continue
                
                # Normalize 0-1
                val = min(1.0, val)
                
                # Map to char
                char_idx = int(val * (len(self.chars) - 1))
                char = self.chars[min(char_idx, len(self.chars) - 1)]
                
                # Color logic
                sat = 0.7 + loud * 0.3
                
                # BRIGHTNESS LOGIC:
                # Minimum brightness 40% (0.4) always
                # Peaks at 1.0
                bright = 0.4 + 0.6 * val
                
                # Entropy shifts hue
                h_shift = hue + val * 0.2 + entropy * 0.1
                
                # Get color from palette or HSV fallback
                if self.palette:
                    r, g, b = self._palette_color(val, bright)
                else:
                    r, g, b = self._color(h_shift % 1.0, sat, bright)
                
                line.append((char, r, g, b))
            grid.append(line)
            
        return grid


# ── BACKGROUND RENDERER ──────────────────────────────────────────────────────

class BackgroundRenderer:
    """
    Renders a dynamic gradient, non-ASCII background.
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.phase = 0.0
        
    def render(self, feat: dict) -> list:
        # Returns grid of (r, g, b) tuples
        t = feat["time"]
        loud = feat["loudness"]
        section_idx = feat.get("section", 0)
        
        # Color cycle based on section and time
        hue1 = (section_idx * 0.3 + t * 0.05) % 1.0
        hue2 = (hue1 + 0.5) % 1.0
        
        # Base brightness pulses with audio (min 20%, max 60%)
        # User requested bright backgrounds
        base_bright = 0.10 + loud * 0.40
        
        # Create gradient
        grid = []
        for row in range(self.h):
            row_line = []
            y_fac = row / self.h
            
            # Gradient mixing
            h_mix = hue1 * (1-y_fac) + hue2 * y_fac
            
            # Simple RGB conversion for speed
            r, g, b = colorsys.hsv_to_rgb(h_mix, 0.8, base_bright)
            color = (int(r*255), int(g*255), int(b*255))
            
            # Just fill the row efficiently
            row_line = [color] * self.w
            grid.append(row_line)
            
        return grid


# ── Compositing (Updated for SCREEN BLEND) ───────────────────────────────────

def composite_layers(layers: dict, width: int, height: int, bg_grid: list = None) -> list:
    """
    Composite layers using Additive/Screen blending for maximum brightness.
    """
    # Pre-calculate layer weights? No, screen blend handles it.
    
    merged = []
    
    # Cache layer access for speed
    active_tracks = [t for t in LAYER_ORDER if t in layers]
    
    for row in range(height):
        line = []
        
        # Get background row if available
        bg_row = bg_grid[row] if bg_grid else [(0,0,0)] * width
        
        for col in range(width):
            # Start with background
            final_r, final_g, final_b = bg_row[col] if bg_grid else (0,0,0)
            best_char = ' '
            max_val = 0
            
            for track in active_tracks:
                char, r, g, b = layers[track][row][col]
                
                if char == ' ':
                    continue

                # Screen Blend: 1 - (1-A)*(1-B)
                # Conceptually: We want to ADD light.
                
                # Normalize to 0-1
                in_r, in_g, in_b = r/255.0, g/255.0, b/255.0
                curr_r, curr_g, curr_b = final_r/255.0, final_g/255.0, final_b/255.0
                
                # Apply opacity weight from track
                opacity = LAYER_OPACITY.get(track, 0.8)
                in_r *= opacity
                in_g *= opacity
                in_b *= opacity
                
                # Screen blend formula
                out_r = 1.0 - (1.0 - curr_r) * (1.0 - in_r)
                out_g = 1.0 - (1.0 - curr_g) * (1.0 - in_g)
                out_b = 1.0 - (1.0 - curr_b) * (1.0 - in_b)
                
                final_r = int(out_r * 255.0)
                final_g = int(out_g * 255.0)
                final_b = int(out_b * 255.0)
                
                # Character contest: Keep the one with highest visual weight
                # (Simple heuristic: brightness)
                val = r + g + b
                if val > max_val:
                    max_val = val
                    best_char = char
            
            # Special case: Full block to blackout background?
            # If best_char is a block and color is black... 
            # But we want additive mixing mostly.
            
            line.append((best_char, final_r, final_g, final_b))
        merged.append(line)
    return merged


def composite_two_layers(foundation_grid: list, dynamic_grid: list, 
                         dynamic_opacity: float, width: int, height: int) -> list:
    """
    Screen blend foundation and dynamic layers.
    """
    result = []
    
    for row in range(height):
        line = []
        for col in range(width):
            # Foundation
            f_char, f_r, f_g, f_b = foundation_grid[row][col]
            
            # Dynamic
            d_char, d_r, d_g, d_b = dynamic_grid[row][col]
            
            # Normalize
            fr, fg, fb = f_r/255.0, f_g/255.0, f_b/255.0
            dr, dg, db = d_r/255.0, d_g/255.0, d_b/255.0
            
            # Weight dynamic
            dr *= dynamic_opacity
            dg *= dynamic_opacity
            db *= dynamic_opacity
            
            # Screen Blend
            rr = 1.0 - (1.0 - fr) * (1.0 - dr)
            rg = 1.0 - (1.0 - fg) * (1.0 - dg)
            rb = 1.0 - (1.0 - fb) * (1.0 - db)
            
            final_r = int(rr * 255)
            final_g = int(rg * 255)
            final_b = int(rb * 255)
            
            # Char choice
            # If dynamic is bright enough, show it
            d_val = (d_r + d_g + d_b) * dynamic_opacity
            f_val = (f_r + f_g + f_b)
            
            char = d_char if (d_val > f_val and d_char != ' ') else f_char
            
            line.append((char, final_r, final_g, final_b))
        result.append(line)
    return result

# ── RENDERER POOL ─────────────────────────────────────────────────────────────
# Replaced with Generic Procedural Instances


def load_font(emoji_support: bool = False):
    """Load a font with good Unicode/emoji support."""
    font_size = 18  # Slightly larger for better readability
    
    if emoji_support:
        # Try emoji-capable fonts first
        emoji_paths = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/Library/Fonts/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        ]
        for p in emoji_paths:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, font_size)
                    print(f"   ✓ Loaded emoji font: {p}")
                    return font
                except Exception as e:
                    print(f"   ✗ Failed to load {p}: {e}")
                    continue
    
    # PRIORITIZE fonts with FULL Unicode support (braille, symbols, math, etc.)
    # Apple Symbols has the BEST coverage for symbols including braille!
    unicode_fonts = [
        # Apple Symbols - BEST for braille and math symbols!
        "/System/Library/Fonts/Apple Symbols.ttf",
        # Arial Unicode - good general coverage
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        # Last Resort font - designed to show glyphs for everything
        "/System/Library/Fonts/LastResort.otf",
        # DejaVu family
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        # Noto Sans
        "/Library/Fonts/NotoSans-Regular.ttf",
        # Symbola
        "/Library/Fonts/Symbola.ttf",
        # Standard fallbacks
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    
    for p in unicode_fonts:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, font_size)
                print(f"   ✓ Loaded Unicode font: {os.path.basename(p)}")
                return font
            except Exception as e:
                print(f"   ✗ Failed to load {os.path.basename(p)}: {e}")
                continue
    
    # Last resort - default font (limited Unicode)
    print("   ⚠️ Warning: Using PIL default font - Unicode support VERY limited!")
    return ImageFont.load_default()


def load_multi_font():
    """
    Load multiple fonts for complete Unicode coverage.
    Returns a FontSet with symbol font and emoji font.
    """
    font_size = 18
    fonts = {"symbol": None, "emoji": None, "fallback": None}
    
    # Load symbol font (for braille, math, music, arrows, etc.)
    symbol_paths = [
        "/System/Library/Fonts/Apple Symbols.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
    ]
    for p in symbol_paths:
        if os.path.exists(p):
            try:
                fonts["symbol"] = ImageFont.truetype(p, font_size)
                print(f"   ✓ Symbol font: {os.path.basename(p)}")
                break
            except:
                continue
    
    # Load emoji font (for color emoji)
    emoji_paths = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/Library/Fonts/NotoColorEmoji.ttf",
    ]
    candidate_sizes = [20, 32, 40, 48, 64, 96, 160, 18] # Try known sizes first, then fallback
    for p in emoji_paths:
        if os.path.exists(p):
            # Try sizes
            for size in candidate_sizes:
                try:
                    f = ImageFont.truetype(p, size)
                    fonts["emoji"] = f
                    print(f"   ✓ Emoji font: {os.path.basename(p)} @ {size}px")
                    break
                except:
                    continue
            if fonts["emoji"]:
                break
    
    # Fallback font
    fallback_paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    for p in fallback_paths:
        if os.path.exists(p):
            try:
                fonts["fallback"] = ImageFont.truetype(p, font_size)
                break
            except:
                continue
    
    if not fonts["fallback"]:
        fonts["fallback"] = ImageFont.load_default()
    
    return fonts


def is_emoji(char: str) -> bool:
    """
    Check if a character needs the emoji font.
    Includes full emoji plus dingbat flowers and symbols that Apple Symbols lacks.
    """
    if not char:
        return False
    cp = ord(char[0])
    
    # Character ranges that need emoji font (Apple Symbols doesn't render well)
    emoji_ranges = [
        # Core emoji blocks
        (0x1F300, 0x1F9FF),  # Miscellaneous Symbols and Pictographs, Emoticons
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F1E0, 0x1F1FF),  # Flags
        
        # Miscellaneous symbols that often render better with emoji font
        (0x2600, 0x26FF),    # Misc Symbols (☀️, ⚡, ☔, etc.)
        
        # Dingbats - flowers and decorative symbols
        (0x2700, 0x27BF),    # Dingbats (✿, ❀, ❁, ✂, ✈, etc.)
        (0x2702, 0x27B0),    # More dingbats
        
        # Supplemental symbols
        (0x1FA00, 0x1FA6F),  # Chess, card symbols
        (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
        
        # Musical symbols that render better with emoji
        (0x1F3B5, 0x1F3BC),  # Musical notes emoji
        
        # Specific overrides
        (0x2B50, 0x2B50),    # ⭐ Medium White Star
    ]
    
    for start, end in emoji_ranges:
        if start <= cp <= end:
            return True
    return False


def grid_to_image(grid: list, width: int, height: int, fonts) -> Image.Image:
    """
    Render a character grid to a PIL Image using multiple fonts.
    
    Args:
        grid: Character grid
        width, height: Grid dimensions
        fonts: Either a single font or a dict of fonts from load_multi_font()
    """
    img_w = width * CELL_W
    img_h = height * CELL_H
    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Handle both single font and multi-font modes
    if isinstance(fonts, dict):
        symbol_font = fonts.get("symbol") or fonts.get("fallback")
        emoji_font = fonts.get("emoji") or symbol_font
        fallback_font = fonts.get("fallback") or symbol_font
    else:
        # Single font mode (backwards compatible)
        symbol_font = emoji_font = fallback_font = fonts

    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            # Unpack cell logic
            if len(cell) == 4:
                char, r, g, b = cell
                br, bg, bb = 0, 0, 0
            elif len(cell) == 7:
                char, r, g, b, br, bg, bb = cell
            else:
                continue # Invalid cell format
                
            x = col_idx * CELL_W
            y = row_idx * CELL_H

            # Draw background if present
            if br > 0 or bg > 0 or bb > 0:
                draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], fill=(br, bg, bb))
            elif char == '\u2588':
                # Full block optimization
                draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], fill=(r, g, b))
                continue

            if char == ' ' or (r + g + b) == 0:
                continue

            # Select appropriate font for this character
            if is_emoji(char):
                font = emoji_font
            else:
                font = symbol_font

            try:
                draw.text((x, y), char, fill=(r, g, b), font=font)
            except Exception:
                # Fallback if character can't be rendered
                try:
                    draw.text((x, y), char, fill=(r, g, b), font=fallback_font)
                except:
                    pass  # Skip unrenderable characters

    return img


# ── Stem Splitting ───────────────────────────────────────────────────────────

def split_stems(input_path: str, output_dir: str) -> dict:
    """Use Demucs to separate audio into 4 stems. Fallback to original if fails."""
    print("🎵 Splitting stems with Demucs (htdemucs)...")
    
    stems = {}
    mapping = {
        "drums": "drums.wav",
        "bass": "bass.wav",
        "keys": "vocals.wav",
        "other": "other.wav",
    }
    
    try:
        # Check if already processed
        stem_name = Path(input_path).stem
        stem_dir = Path(output_dir) / "htdemucs" / stem_name
        
        # If directory doesn't exist, try running demucs
        if not stem_dir.exists():
            print("   Running Demucs separation...")
            cmd = [
                sys.executable, "-m", "demucs",
                "-n", "htdemucs",
                "--out", output_dir,
                input_path,
            ]
            subprocess.run(cmd, check=True)
            
        # Re-check dir
        if not stem_dir.exists():
            candidates = list(Path(output_dir).rglob(f"{stem_name}"))
            if candidates:
                stem_dir = candidates[0]
        
        # Collect stems
        for track, filename in mapping.items():
            path = stem_dir / filename
            if path.exists():
                stems[track] = str(path)
            else:
                print(f"   ⚠ Stem '{filename}' not found")
                
    except (subprocess.CalledProcessError, FileNotFoundError, ImportError) as e:
        print(f"   ⚠️ Demucs failed or not installed: {e}")
        print("   ⚠️ Fallback: Using original audio for all tracks (visuals will still react)")
        for track in mapping:
            stems[track] = str(input_path)
    except Exception as e:
        print(f"   ⚠️ Error during stem splitting: {e}")
        print("   ⚠️ Fallback: Using original audio")
        for track in mapping:
            stems[track] = str(input_path)

    # Ensure we have something
    if not stems:
        print("   ⚠️ No stems found even after fallback? Using original.")
        for track in mapping:
            stems[track] = str(input_path)
            
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global CHAR_POOL
    
    parser = argparse.ArgumentParser(description="Unicode Audio Visualizer with Themed Character Pools")
    parser.add_argument("--mu", help="Input audio file (MP3, WAV, etc.)")
    parser.add_argument("--fps", type=float, default=0.0, help="Output video FPS. Default: Auto-calc if 0")
    parser.add_argument("--fsync", type=float, default=1.0, help="FPS Sync Multiplier (0.1 - 6.0).")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Canvas width in chars (default: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Canvas height in chars (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--out", type=str, help="Output filename (default: auto-generated)")
    parser.add_argument("--prompt", type=str, default="", help="Creative prompt for style hints")
    parser.add_argument("--theme", type=str, default="classic", 
                        choices=CharacterPool.list_themes() + ["random"],
                        help="Character theme (default: classic). Use 'random' to switch per section.")
    parser.add_argument("--list-themes", action="store_true", help="List available themes and exit")
    args = parser.parse_args()
    
    # List themes mode
    if args.list_themes:
        print("\n🎨 Available Unicode Themes:\n")
        for theme_name in CharacterPool.list_themes():
            info = CharacterPool.get_theme_info(theme_name)
            sample = CharacterPool.THEMES[theme_name]["density"]
            print(f"  {theme_name:12} - {info}")
            print(f"               Sample: {sample}\n")
        return
    
    # Require --mu for actual processing
    if not args.mu:
        parser.error("--mu is required (unless using --list-themes)")
    
    # Handle --theme random: Pick ONE random theme for consistency
    import random
    if args.theme == "random":
        available_themes = [t for t in CharacterPool.list_themes() if t != "random"]
        args.theme = random.choice(available_themes)
        print(f"\n🎲 Random theme selected: {args.theme}")
    
    # Initialize character pool with the selected theme
    global CHAR_POOL
    seed_val = int(hashlib.md5(args.prompt.encode()).hexdigest(), 16) if args.prompt else None
    CHAR_POOL = CharacterPool(theme=args.theme, seed=seed_val)
    print(f"🎨 Theme: {CHAR_POOL.get_theme_info(args.theme)}")
    
    # Setup output paths
    script_dir = Path(__file__).parent
    output_root = script_dir / "z_test-outputs"
    output_root.mkdir(exist_ok=True)
    
    input_path = Path(args.mu)
    timestamp = hashlib.md5(str(input_path).encode()).hexdigest()[:8]
    project_name = f"unicode_{input_path.stem}_{args.theme}_{timestamp}"
    project_dir = output_root / project_name
    project_dir.mkdir(exist_ok=True)

    if args.fps > 0:
        fps = float(args.fps)
    else:
        print("   🎵 Auto-calculating FPS from audio...")
        bpm, duration, fpb, suggested_fps = analyze_audio(str(input_path), fsync=args.fsync)
        fps = suggested_fps
    
    print(f"\n{'='*60}")
    print(f"  UNICODE AUDIO VISUALIZER 2.0 (High Brightness)")
    print(f"{'='*60}")
    print(f"║  Input: {input_path.name}")
    print(f"║  Theme: {args.theme}")
    print(f"║  Canvas: {args.width}×{args.height} chars")
    print(f"║  FPS: {fps}")
    print(f"{'='*60}\n")
    
    W, H = args.width, args.height
    
    # Split stems
    stems_dir = project_dir / "stems"
    stems_dir.mkdir(exist_ok=True)
    stems = split_stems(str(input_path), str(stems_dir))
    
    if len(stems) < 4:
        print(f"⚠ Only {len(stems)} stems found. Using available stems.")
    
    # Analyze tracks
    print("\n📊 Analyzing audio features...")
    analyzers = {}
    for track, path in stems.items():
        print(f"   Analyzing {track}...")
        analyzers[track] = TrackAnalyzer(path, fps)
    
    n_frames = max(a.n_frames for a in analyzers.values())
    
    # Initialize renderers
    print("\n🖌️ Initializing renderers...")
    
    # Dynamic layer renderers (per-section changing)
    dynamic_renderers = {
        "drums": ProceduralRenderer(W, H, "drums", args.prompt),
        "bass": ProceduralRenderer(W, H, "bass", args.prompt),
        "keys": ProceduralRenderer(W, H, "keys", args.prompt),
        "other": ProceduralRenderer(W, H, "other", args.prompt),
    }
    
    # Foundation layer renderer (consistent evolution throughout song)
    # Salt the prompt with timestamp to ensure unique landscape every run
    import time
    seed_salt = f"{args.prompt}_{int(time.time())}"
    foundation_renderer = FoundationRenderer(W, H, seed_salt, n_frames)
    
    # Background renderer
    bg_renderer = BackgroundRenderer(W, H)
    
    # Load fonts - multi-font system for complete Unicode coverage
    print("   Loading fonts...")
    fonts = load_multi_font()
    print(f"   Multi-font system ready")
    
    # Get total sections for pacing
    total_sections = len(analyzers["drums"].section_boundaries) if "drums" in analyzers else 8
    print(f"   Total sections: {total_sections}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: RENDER FOUNDATION LAYER
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n🎬 PHASE 1: Rendering Foundation Layer ({n_frames} frames)...")
    print("   Foundation layer: consistent design, sparse→busy evolution")
    
    foundation_frames_dir = project_dir / "foundation_frames"
    foundation_frames_dir.mkdir(exist_ok=True)
    
    # Use drums track for beat/loudness features (or first available)
    ref_track = "drums" if "drums" in analyzers else list(analyzers.keys())[0]
    
    for frame_idx in range(n_frames):
        feat = analyzers[ref_track].get_frame(frame_idx)
        
        # Render foundation (with frame_idx for progress)
        foundation_grid = foundation_renderer.render_frame(feat, frame_idx)
        
        # Convert to image and save
        img = grid_to_image(foundation_grid, W, H, fonts)
        frame_path = foundation_frames_dir / f"frame_{frame_idx:05d}.png"
        img.save(frame_path)
        
        # Progress
        if frame_idx % 200 == 0 or frame_idx == n_frames - 1:
            pct = (frame_idx + 1) / n_frames * 100
            print(f"   Foundation: {frame_idx + 1}/{n_frames} ({pct:.1f}%)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: RENDER DYNAMIC LAYER
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n🎬 PHASE 2: Rendering Dynamic Layer ({n_frames} frames)...")
    print("   Dynamic layer: section changes, procedural patterns, entropy build-up")
    print("                  + Dynamic Background Gradients")
    
    dynamic_frames_dir = project_dir / "dynamic_frames"
    dynamic_frames_dir.mkdir(exist_ok=True)
    
    current_section = -1
    layer_reveal_order = ["drums", "bass", "keys", "other"]
    active_layers = []
    layers_per_reveal = max(1, total_sections // 4)
    
    # Determine section lengths for entropy calculation
    section_boundaries = analyzers[ref_track].section_boundaries
    
    for frame_idx in range(n_frames):
        # Get section info
        ref_feat = analyzers[ref_track].get_frame(frame_idx)
        section = ref_feat["section"]
        
        # Calculate section progress (entropy)
        # Find start and end frames of current section
        section_start_frame = section_boundaries[section] if section < len(section_boundaries) else 0
        section_end_frame = section_boundaries[section+1] if section+1 < len(section_boundaries) else n_frames
        
        # Safe calc
        if section_end_frame > section_start_frame:
            section_progress = (frame_idx - section_start_frame) / (section_end_frame - section_start_frame)
        else:
            section_progress = 0.0
            
        section_progress = max(0.0, min(1.0, section_progress))
        
        # Update feature dict with progress
        ref_feat["section_progress"] = section_progress
        
        # Handle section changes
        if ref_feat["is_section_boundary"] and section != current_section:
            current_section = section
            
            # Progressive layer reveal
            layers_to_have = min(len(layer_reveal_order), 1 + section // max(1, layers_per_reveal))
            while len(active_layers) < layers_to_have:
                new_layer = layer_reveal_order[len(active_layers)]
                active_layers.append(new_layer)
                print(f"   ✨ Section {section}: Revealed {new_layer.upper()} layer!")
            
            # Update theme/palette
            CHAR_POOL.randomize_for_section(section, args.prompt)
            new_palette = select_palette_for_section(args.prompt, section)
            
            # Reset renderers
            for name, r in dynamic_renderers.items():
                r.reset_for_section(section, args.prompt)
                r.set_palette(new_palette)
            
            print(f"   Section {section}: palette={new_palette}, layers={len(active_layers)}")
        
        # Ensure at least drums layer
        if not active_layers:
            active_layers = ["drums"]
        
        # Render background
        bg_grid = bg_renderer.render(ref_feat)
        
        # Render active layers
        layers = {}
        for track in active_layers:
            if track in dynamic_renderers and track in analyzers:
                feat = analyzers[track].get_frame(frame_idx)
                feat["section_progress"] = section_progress # Pass entropy
                layers[track] = dynamic_renderers[track].render_frame(feat)
        
        # Composite dynamic layers WITH BACKGROUND via Screen Blend
        dynamic_grid = composite_layers(layers, W, H, bg_grid=bg_grid)
        
        # Convert to image and save
        img = grid_to_image(dynamic_grid, W, H, fonts)
        frame_path = dynamic_frames_dir / f"frame_{frame_idx:05d}.png"
        img.save(frame_path)
        
        # Progress
        if frame_idx % 200 == 0 or frame_idx == n_frames - 1:
            pct = (frame_idx + 1) / n_frames * 100
            print(f"   Dynamic: {frame_idx + 1}/{n_frames} ({pct:.1f}%) [Ent: {section_progress:.2f}]")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2.5: RENDER HIGHLIGHT LAYER (drums + keys at high contrast)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n🎬 PHASE 2.5: Rendering Highlight Layer ({n_frames} frames)...")
    print("   Highlight layer: drums + keys combined for screen blend")
    
    highlight_frames_dir = project_dir / "highlight_frames"
    highlight_frames_dir.mkdir(exist_ok=True)
    
    for frame_idx in range(n_frames):
        # Need section progress again for correct sync
        ref_feat = analyzers[ref_track].get_frame(frame_idx)
        section = ref_feat["section"]
        section_start_frame = section_boundaries[section] if section < len(section_boundaries) else 0
        section_end_frame = section_boundaries[section+1] if section+1 < len(section_boundaries) else n_frames
        if section_end_frame > section_start_frame:
            progress = (frame_idx - section_start_frame) / (section_end_frame - section_start_frame)
        else:
            progress = 0.0
        
        # Render drums and keys only for highlight
        highlight_layers = {}
        for track in ["drums", "keys"]:
            if track in dynamic_renderers and track in analyzers:
                feat = analyzers[track].get_frame(frame_idx)
                feat["section_progress"] = progress
                # Helper method to get the grid without re-rendering? 
                # Ideally we'd cache frames but memory is tight. Re-render is safer.
                # ProceduralRenderer is deterministic based on time/params so it matches.
                grid = dynamic_renderers[track].render_frame(feat)
                
                # Boost brightness for highlight effect
                boosted_grid = []
                for row in grid:
                    boosted_row = []
                    for char, r, g, b in row:
                        # Boost brightness by 40%
                        r = min(255, int(r * 1.4))
                        g = min(255, int(g * 1.4))
                        b = min(255, int(b * 1.4))
                        boosted_row.append((char, r, g, b))
                    boosted_grid.append(boosted_row)
                highlight_layers[track] = boosted_grid
        
        # Composite highlight layers (same as dynamic but just drums+keys, NO BG)
        if highlight_layers:
            highlight_grid = composite_layers(highlight_layers, W, H, bg_grid=None)
        else:
            highlight_grid = [[(' ', 0, 0, 0) for _ in range(W)] for _ in range(H)]
        
        # Convert to image and save
        img = grid_to_image(highlight_grid, W, H, fonts)
        frame_path = highlight_frames_dir / f"frame_{frame_idx:05d}.png"
        img.save(frame_path)
        
        # Progress
        if frame_idx % 200 == 0 or frame_idx == n_frames - 1:
            pct = (frame_idx + 1) / n_frames * 100
            print(f"   Highlight: {frame_idx + 1}/{n_frames} ({pct:.1f}%)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: COMPOSITE LAYERS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n🎬 PHASE 3: Compositing Layers ({n_frames} frames)...")
    print("   Blending: Foundation + Dynamic (opacity curve) + Highlight (25% screen)")
    
    composite_frames_dir = project_dir / "frames"
    composite_frames_dir.mkdir(exist_ok=True)
    
    # Highlight layer opacity - constant 25%
    HIGHLIGHT_OPACITY = 0.25
    
    for frame_idx in range(n_frames):
        progress = frame_idx / n_frames
        
        # Get section for opacity calculation
        ref_feat = analyzers[ref_track].get_frame(frame_idx)
        section = ref_feat["section"]
        
        # Calculate dynamic layer opacity
        dynamic_opacity = calculate_dynamic_layer_opacity(progress, total_sections, section)
        
        # Load all layer frames
        foundation_img = Image.open(foundation_frames_dir / f"frame_{frame_idx:05d}.png")
        dynamic_img = Image.open(dynamic_frames_dir / f"frame_{frame_idx:05d}.png")
        highlight_img = Image.open(highlight_frames_dir / f"frame_{frame_idx:05d}.png")
        
        # Step 1: Alpha blend foundation + dynamic
        if dynamic_opacity > 0.01:
            blended = Image.blend(foundation_img, dynamic_img, dynamic_opacity)
        else:
            blended = foundation_img.copy()
        
        # Step 2: Screen blend the highlight layer on top
        # Screen formula: result = 1 - (1-a) * (1-b)
        # This always brightens or maintains brightness
        base_arr = np.array(blended, dtype=np.float32) / 255.0
        highlight_arr = np.array(highlight_img, dtype=np.float32) / 255.0
        
        # Apply screen blend with opacity
        # screen = 1 - (1 - base) * (1 - highlight * opacity)
        screen_result = 1.0 - (1.0 - base_arr) * (1.0 - highlight_arr * HIGHLIGHT_OPACITY)
        
        # Convert back to image
        final_arr = np.clip(screen_result * 255, 0, 255).astype(np.uint8)
        blended = Image.fromarray(final_arr)
        
        # Save composite
        frame_path = composite_frames_dir / f"frame_{frame_idx:05d}.png"
        blended.save(frame_path)
        
        # Progress
        if frame_idx % 200 == 0 or frame_idx == n_frames - 1:
            pct = (frame_idx + 1) / n_frames * 100
            opacity_pct = dynamic_opacity * 100
            print(f"   Composite: {frame_idx + 1}/{n_frames} ({pct:.1f}%) - Dynamic: {opacity_pct:.0f}%")
    
    # Assemble video
    print("\n🎬 Assembling video...")
    raw_video = project_dir / "raw.mp4"
    final_video = project_dir / (args.out if args.out else f"{project_name}.mp4")
    
    # FFmpeg encode
    cmd_encode = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(composite_frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(raw_video)
    ]
    subprocess.run(cmd_encode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Mux audio
    print("🔊 Muxing audio...")
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", str(raw_video),
        "-i", str(input_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_video)
    ]
    subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETE: {final_video}")
    print(f"{'='*60}\n")




if __name__ == "__main__":
    main()
