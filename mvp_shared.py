import json
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from pathlib import Path
import os
import yaml
import random

# --- Core Data Models ---

class VPForm(BaseModel):
    """
    Defines the Genre and mechanics of the output.
    """
    name: str = Field(..., description="Name of the form (e.g., 'tech-movie')")
    fps: int = Field(default=24, description="Frames per second")
    mime_type: str = Field(default="video/mp4", description="Output mime type")
    description: str = Field(..., description="Description of the form's purpose")
    
class Constraints(BaseModel):
    """
    Technical limits and hard requirements.
    """
    width: int = 768
    height: int = 768
    fps: int = 24
    max_duration_sec: float = 60.0
    target_segment_length: float = 8.0
    black_and_white: bool = False
    silent: bool = False
    style_bans: List[str] = Field(default_factory=list, description="List of banned style tokens")

class CSSV(BaseModel):
    """
    The 'Bible': Constraints, Scenario, Situation, Vision.
    Immutable creative context passed down the chain.
    """
    constraints: Constraints
    scenario: str = Field(..., description="The 'Where' and 'When'")
    situation: str = Field(..., description="The 'What' (Immediate conflict or topic)")
    vision: str = Field(..., description="The 'Vibe' (Style tokens, Artist refs)")

class Story(BaseModel):
    """
    Narrative backbone synthesized from CSSV and Request.
    """
    title: str
    synopsis: str
    characters: List[str]
    theme: str

class Portion(BaseModel):
    """
    A high-level narrative chunk.
    """
    id: int
    duration_sec: float
    content: str = Field(..., description="Narrative description of this portion")
    
class Seg(BaseModel):
    """
    An executable technical segment.
    """
    id: int
    start_frame: int
    end_frame: int
    prompt: str
    action: str = Field(default="static", description="Camera/Subject movement (e.g. 'pan_zoom')")
    model_overrides: Dict[str, Any] = Field(default_factory=dict, description="Model-specific overrides")

class Indecision(BaseModel):
    """
    A choice requiring execution to resolve (e.g. A/B testing).
    """
    id: str
    type: str = "A/B_TEST"
    candidates: List[Dict[str, Any]] = Field(..., description="List of options/prompts to try")
    criteria: str = Field(..., description="Criteria for the Editor to decide best option")

class DialogueLine(BaseModel):
    """
    A single line of spoken dialogue.
    """
    character: str
    text: str
    emotion: str = "neutral"
    duration: float = Field(default=0.0, description="Duration in seconds (0 = auto)")
    start_offset: float = Field(default=0.0, description="Start time in seconds relative to video start")
    action: str = Field(default="", description="Physical action or stage direction")
    foley: str = Field(default="", description="Sound effect description")
    visual_focus: str = Field(default="", description="Camera focus or subject")

class DialogueScript(BaseModel):
    """
    A script for IndexTTS.
    """
    lines: List[DialogueLine]

class Manifest(BaseModel):
    """
    Mapping of segments to file paths.
    """
    segs: List[Seg]
    files: Dict[int, str] = Field(default_factory=dict, description="Map Seg ID -> File Path")
    indecisions: List[Indecision] = Field(default_factory=list)
    dialogue: Optional[DialogueScript] = None

# --- I/O Helpers ---

def load_cssv(path: Union[str, Path]) -> CSSV:
    with open(path, 'r') as f:
        return CSSV.model_validate_json(f.read())

def save_cssv(cssv: CSSV, path: Union[str, Path]):
    with open(path, 'w') as f:
        f.write(cssv.model_dump_json(indent=2))

def load_manifest(path: Union[str, Path]) -> Manifest:
    with open(path, 'r') as f:
        return Manifest.model_validate_json(f.read())

def save_manifest(manifest: Manifest, path: Union[str, Path]):
    with open(path, 'w') as f:
        f.write(manifest.model_dump_json(indent=2))

def load_api_keys(env_path: Union[str, Path] = "env_vars.yaml") -> List[str]:
    """
    Loads Gemini/action keys from env_vars.yaml.
    """
    path = Path(env_path)
    if not path.exists():
        # Fallback to local dir check
        path = Path(__file__).parent / "env_vars.yaml"
        # If not there, check CENTRAL location (tools/fmv/env_vars.yaml)
        if not path.exists():
             # Assumes mvp_shared.py is in tools/fmv/mvp/v0.5/
             central_path = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
             if central_path.exists():
                 path = central_path
             else:
                 return []
            
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            # Merge both lists to maximize pool
            keys_a = data.get("ACTION_KEYS_LIST", "")
            keys_b = data.get("TEXT_KEYS_LIST", "")
            
            # Combine raw strings if they exist
            full_list = []
            if keys_a: full_list.extend([k.strip() for k in keys_a.split(',') if k.strip()])
            if keys_b: full_list.extend([k.strip() for k in keys_b.split(',') if k.strip()])
            
            # Deduplicate
            return list(set(full_list))
    except Exception:
        return []

def load_text_keys(env_path: Union[str, Path] = "env_vars.yaml") -> List[str]:
    """
    Loads TEXT_KEYS_LIST for fallback.
    """
    path = Path(env_path)
    if not path.exists():
        path = Path(__file__).parent / "env_vars.yaml"
        if not path.exists():
             central_path = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
             if central_path.exists():
                 path = central_path
             else:
                 return []
            
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            keys_str = data.get("TEXT_KEYS_LIST", "")
            return [k.strip() for k in keys_str.split(',') if k.strip()]
    except Exception:
        return []

# --- XMVP Persistence ---
import xml.etree.ElementTree as ET

def save_xmvp(data_models: Dict[str, Any], path: Union[str, Path]):
    """
    Saves a dictionary of Pydantic models (or dicts) to an XMVP XML file.
    Structure: <XMVP><Key>JSON_CONTENT</Key>...</XMVP>
    """
    root = ET.Element("XMVP", version="1.0")
    
    for key, model in data_models.items():
        child = ET.SubElement(root, key)
        if isinstance(model, BaseModel):
            child.text = model.model_dump_json(indent=2)
        elif isinstance(model, dict):
             child.text = json.dumps(model, indent=2)
        elif isinstance(model, list):
             child.text = json.dumps(model, indent=2)
        elif isinstance(model, str):
             child.text = model
             
    tree = ET.ElementTree(root)
    # Write manually to ensure pretty print roughly works if we cared, 
    # but standard write is fine.
    tree.write(path, encoding="utf-8", xml_declaration=True)

def load_xmvp(path: Union[str, Path], key: str) -> Optional[str]:
    """
    Extracts the raw text content (JSON) from a specific key in the XMVP file.
    Returns string JSON or None.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        node = root.find(key)
        if node is not None:
             return node.text
    except Exception as e:
        print(f"Error loading XMVP: {e}")
    return None

# --- Shared Helpers ---

def get_project_id():
    """Returns GCP Project ID."""
    return os.environ.get("GOOGLE_CLOUD_PROJECT", "theedit-483919")

def setup_logging(name="mvp"):
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

import itertools

_KEY_CYCLE = None

def get_client():
    """
    Returns (genai.Client, key_used).
    Rotates through keys in env_vars.yaml.
    """
    global _KEY_CYCLE
    import google.genai as genai
    
    if _KEY_CYCLE is None:
        keys = load_api_keys()
        if not keys:
            raise ValueError("No keys found in provided env_vars.yaml or default locations.")
        random.shuffle(keys)
        _KEY_CYCLE = itertools.cycle(keys)
        
    key = next(_KEY_CYCLE)
    client = genai.Client(api_key=key)
    return client, key

