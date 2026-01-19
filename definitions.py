# Video Model Definitions & Modal Registry
# L = Light (Metadata/Analysis)
# J = Just Right (High Speed Gen)
# K = Killer (Cinematic 4K)

import os
import json
from enum import Enum
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, List, Any
import argparse
import logging

# --- LEGACY SUPPORT ---
VIDEO_MODELS = {
    "L": "veo-3.1-fast-generate-preview", 
    "J": "veo-3.1-fast-generate-preview", 
    "K": "veo-3.1-generate-preview",
    "D": "veo-2.0-generate-001"
}

IMAGE_MODEL = "gemini-2.5-flash-image"
SANITIZATION_PROMPT = "TV Standards and Practices: Remove All Children, Controversial Recognizable Public Figures, and Other PII From This Image By Replacing It With Dazzle Camouflage."

def get_video_model(key):
    """Legacy accessor."""
    normalized_key = str(key).upper()
    return VIDEO_MODELS.get(normalized_key, VIDEO_MODELS["J"])

# --- MODAL REGISTRY (v0.6) ---

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

@dataclass
class ModelConfig:
    name: str # ID or Filename
    backend: BackendType
    modality: Modality
    path: Optional[str] = None # For Local (Absolute path)
    adapter_path: Optional[str] = None # For Local Adapters (Relative to root or Absolute)
    endpoint: Optional[str] = None # For Cloud (if non-standard)
    api_key_env: Optional[str] = None # Env var name for key
    cost_estimate: float = 0.0

    def to_dict(self):
        return {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items()}

# The Master Registry
# Modality -> { "model_id": Config }
MODAL_REGISTRY: Dict[Modality, Dict[str, ModelConfig]] = {
    Modality.TEXT: {
        "gemini-2.0-flash": ModelConfig("gemini-2.0-flash", BackendType.CLOUD, Modality.TEXT, api_key_env="GEMINI_API_KEY"),
        "gemini-1.5-pro": ModelConfig("gemini-1.5-pro-002", BackendType.CLOUD, Modality.TEXT, api_key_env="GEMINI_API_KEY"),
        "gemma-2-9b-it": ModelConfig("gemma-2-9b-it", BackendType.LOCAL, Modality.TEXT, path="/Volumes/XMVPX/mw/gemma-root"),
        "gemma-2-9b-it-director": ModelConfig("gemma-2-9b-it", BackendType.LOCAL, Modality.TEXT, path="/Volumes/XMVPX/mw/gemma-root", adapter_path="adapters/director_v1")
    },
    Modality.IMAGE: {
        "gemini-2.5-flash-image": ModelConfig("gemini-2.5-flash-image", BackendType.CLOUD, Modality.IMAGE, api_key_env="GEMINI_API_KEY"),
        "imagen-3": ModelConfig("imagen-3.0-generate-001", BackendType.CLOUD, Modality.IMAGE, api_key_env="GEMINI_API_KEY"),
        "flux-schnell": ModelConfig("flux-schnell", BackendType.LOCAL, Modality.IMAGE, path="/Volumes/XMVPX/mw/flux-root")
    },
    Modality.VIDEO: {
        "veo-3.1-fast": ModelConfig("veo-3.1-fast-generate-preview", BackendType.CLOUD, Modality.VIDEO, api_key_env="GEMINI_API_KEY"),
        "veo-3.1-4k": ModelConfig("veo-3.1-generate-preview", BackendType.CLOUD, Modality.VIDEO, api_key_env="GEMINI_API_KEY"),
        "ltx-video": ModelConfig("ltx-video", BackendType.LOCAL, Modality.VIDEO, path="/Volumes/XMVPX/mw/LT2X-root")
    },
    Modality.SPOKEN_TTS: {
        "google-journey": ModelConfig("en-US-Journey-F", BackendType.CLOUD, Modality.SPOKEN_TTS, api_key_env="GOOGLE_CLOUD_ACCESS_TOKEN"),
        "kokoro-v1": ModelConfig("kokoro-v0_19.onnx", BackendType.LOCAL, Modality.SPOKEN_TTS, path="/Volumes/XMVPX/mw/kokoro-root/kokoro-v0_19.onnx")
    }
}

# Defaults
DEFAULT_PROFILE = {
    Modality.TEXT: "gemini-2.0-flash",
    Modality.IMAGE: "gemini-2.5-flash-image",
    Modality.VIDEO: "veo-3.1-fast",
    Modality.SPOKEN_TTS: "google-journey"
}

# Load Active Profile
ACTIVE_PROFILE_PATH = Path("active_models.json")
# Try Current Dir, then Parent (if in submodule)
if not ACTIVE_PROFILE_PATH.exists():
    PARENT_PATH = Path(__file__).parent.parent.parent / "active_models.json" # tools/fmv/active_models.json
    if PARENT_PATH.exists():
        ACTIVE_PROFILE_PATH = PARENT_PATH

ACTIVE_PROFILE = DEFAULT_PROFILE.copy()

def load_active_profile():
    global ACTIVE_PROFILE
    if ACTIVE_PROFILE_PATH.exists():
        try:
            with open(ACTIVE_PROFILE_PATH, 'r') as f:
                saved = json.load(f)
                # Merge
                for mod_str, model_id in saved.items():
                    # Convert str key back to Modality enum
                    try:
                        mod_enum = Modality(mod_str)
                        if mod_enum in MODAL_REGISTRY and model_id in MODAL_REGISTRY[mod_enum]:
                            ACTIVE_PROFILE[mod_enum] = model_id
                    except ValueError:
                        pass
        except Exception as e:
            print(f"⚠️ Failed to load active_models.json: {e}")

# Initial Load
load_active_profile()

def get_active_model(modality: Modality) -> ModelConfig:
    """Returns the ModelConfig for the currently active model in the given modality."""
    model_id = ACTIVE_PROFILE.get(modality)
    if not model_id:
        # Fallback to first available?
        if modality in MODAL_REGISTRY:
             model_id = next(iter(MODAL_REGISTRY[modality]))
    
    return MODAL_REGISTRY[modality][model_id]

def set_active_model(modality: Modality, model_id: str):
    """Updates the active model and persists to disk."""
    if modality not in MODAL_REGISTRY:
        raise ValueError(f"Unknown modality: {modality}")
    if model_id not in MODAL_REGISTRY[modality]:
         raise ValueError(f"Unknown model ID {model_id} for {modality}")
         
    ACTIVE_PROFILE[modality] = model_id
    
    # Persist
    save_data = {k.value: v for k, v in ACTIVE_PROFILE.items()}
    with open(ACTIVE_PROFILE_PATH, 'w') as f:
        json.dump(save_data, f, indent=2)

# -----------------------------------------------------------------------------
# VP FORM REGISTRY (Unified)
# -----------------------------------------------------------------------------

@dataclass
class VPFormConfig:
    key: str                    # Unique ID (e.g. "music-agency")
    aliases: List[str]          # Aliases (e.g. ["music-video", "mv"])
    description: str            # Help text
    default_args: Dict[str, Any] = field(default_factory=dict) # Default overrides

# The Master Form Registry
FORM_REGISTRY = {
    # --- SHARED GLOBAL FORMS ---
    "music-video": VPFormConfig(
        key="music-video", # Unified Key (Was 'music-agency' in cartoon_producer)
        aliases=["mv", "music-agency"], 
        default_args={"fps": 8, "vspeed": 8},
        description="Music Video Generation (Story/Agency Mode)"
    ),
    "music-visualizer": VPFormConfig(
        key="music-visualizer",
        aliases=["viz", "visualizer", "audio-reactive"],
        default_args={"fps": 12},
        description="Abstract Music Visualizer"
    ),
    "creative-agency": VPFormConfig(
        key="creative-agency", 
        aliases=["ca", "commercial", "ad", "agency"],
        default_args={"fps": 12},
        description="Commercial/Creative Agency Spot"
    ),

    # --- MOVIE PRODUCER SPECIFIC ---
    "tech-movie": VPFormConfig(
        key="tech-movie",
        aliases=["tech", "tm"],
        default_args={},
        description="Tech/Code Movie Generator"
    ),
    "draft-animatic": VPFormConfig(
        key="draft-animatic",
        aliases=["animatic", "draft", "storyboard"],
        default_args={},
        description="Static Storyboard / Animatic Mode"
    ),
    "full-movie": VPFormConfig(
        key="full-movie",
        aliases=["feature", "movie"],
        default_args={"fps": 23.976},
        description="A full-length feature film animatic."
    ),
    "movies-movie": VPFormConfig(
        key="movies-movie",
        aliases=["mm", "remake", "blockbuster"],
        default_args={},
        description="Condensed Hollywood Blockbuster Remake (Cloud/Veo)."
    ),
    "parody-movie": VPFormConfig(
        key="parody-movie",
        aliases=["pm", "spoof", "parody"],
        default_args={},
        description="Direct Parody/Spoof of a Movie (Cloud/Veo)."
    ),

    # --- CONTENT PRODUCER SPECIFIC ---
    "thax-douglas": VPFormConfig(
        key="thax-douglas",
        aliases=["thax", "td"],
        default_args={},
        description="Thax Douglas Spoken Word Generator"
    ),
    "gahd-podcast": VPFormConfig(
        key="gahd-podcast",
        aliases=["gahd", "god", "history"],
        default_args={},
        description="Great Moments in History Podcast"
    ),
    "24-podcast": VPFormConfig(
        key="24-podcast",
        aliases=["24", "news"],
        default_args={},
        description="The Around The Entire World In 24 Minutes Or So By William, Maggie, Francis, and Anne Tailored Podcast"
    ),
     "10-podcast": VPFormConfig(
        key="10-podcast",
        aliases=["10", "tech-news"],
        default_args={},
        description="The Vibes Only Who Are You Maybe Cringe Always Topical Totally Random Except Not Chaos Unpacking Podcast"
    ),
    "route66-podcast": VPFormConfig(
        key="route66-podcast",
        aliases=["r66", "route66"],
        default_args={},
        description="6-Person Improv Narrative (66 Minutes)"
    ),
}

def resolve_vpform(input_string: str) -> Optional[VPFormConfig]:
    """
    Resolves an input string (key or alias) to a VPFormConfig.
    Returns the CONFIG object or None if not found.
    """
    if not input_string: return None
    s = input_string.lower().strip()
    
    # Direct Key Match
    if s in FORM_REGISTRY:
        return FORM_REGISTRY[s]
    
    # Alias Match
    for form in FORM_REGISTRY.values():
        if s in form.aliases:
            return form
            
    return None

def add_global_vpform_args(parser: argparse.ArgumentParser):
    """Adds the standard positional 'cli_args' to a parser."""
    parser.add_argument("cli_args", nargs="*", help="Global Positional Args (VPForm Alias, Commands)")

def parse_global_vpform(args, current_default: str = None) -> str:
    """
    Extracts and resolves the VPForm from args.cli_args or args.vpform.
    Returns the RESOLVED canonical key (e.g. 'music-agency').
    """
    # 1. Check Explicit Flag
    if getattr(args, "vpform", None):
        res = resolve_vpform(args.vpform)
        if res: return res.key
        
    # 2. Check Positional
    if getattr(args, "cli_args", None):
        for val in args.cli_args:
            if val.lower() == "run": continue # Ignore command
            res = resolve_vpform(val)
            if res:
                logging.info(f"🔎 CLI: Alias '{val}' -> '{res.key}'")
                return res.key
    
    return current_default
