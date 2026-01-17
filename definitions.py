# Video Model Definitions & Modal Registry
# L = Light (Metadata/Analysis)
# J = Just Right (High Speed Gen)
# K = Killer (Cinematic 4K)

import os
import json
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

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
        "gemma-2-9b-it": ModelConfig("gemma-2-9b-it", BackendType.LOCAL, Modality.TEXT, path="/Volumes/XMVPX/mw/gemma-root")
    },
    Modality.IMAGE: {
        "gemini-2.5-flash-image": ModelConfig("gemini-2.5-flash-image", BackendType.CLOUD, Modality.IMAGE, api_key_env="GEMINI_API_KEY"),
        "imagen-3": ModelConfig("imagen-3.0-generate-001", BackendType.CLOUD, Modality.IMAGE, api_key_env="GEMINI_API_KEY"),
        "flux-schnell": ModelConfig("flux-schnell", BackendType.LOCAL, Modality.IMAGE, path="/Volumes/XMVPX/mw/flux-root")
    },
    Modality.VIDEO: {
        "veo-3.1-fast": ModelConfig("veo-3.1-fast-generate-preview", BackendType.CLOUD, Modality.VIDEO, api_key_env="GEMINI_API_KEY"),
        "veo-3.1-4k": ModelConfig("veo-3.1-generate-preview", BackendType.CLOUD, Modality.VIDEO, api_key_env="GEMINI_API_KEY"),
        "ltx-video": ModelConfig("ltx-video", BackendType.LOCAL, Modality.VIDEO, path="/Volumes/XMVPX/mw/LT2X-root/ltxv-13b-0.9.8-distilled.safetensors")
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
