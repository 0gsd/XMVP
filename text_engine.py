#!/usr/bin/env python3
import os
import sys
import yaml
import json
import logging
import random
from pathlib import Path

import itertools

# Try to import Gemini SDK (Standard)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Lazy import for MLX (Apple Silicon)
# We don't import at top level to avoid crashing on non-Macs or if missing
MLX_AVAILABLE = False
try:
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    pass

# --- THE FILTER ---
# The immutable persona that forces any model to act like a component
SYSTEM_FILTER = (
    "You are the central engine of the XMVP Movie Studio. "
    "You are a rigid, deterministic JSON generator. "
    "You do NOT chat. You do NOT offer explanations. "
    "You accept a data contract (CSSV, Story, Portion) and you return the transformed output "
    "strictly adhering to the schema. "
    "You are specialized in creative screenwriting, formatting, and structural logic."
)

from mvp_shared import load_api_keys, load_text_keys

class TextEngine:
    def __init__(self, config_path=None):
        self.backend = "gemini_api" # Default
        self.local_model_path = "google/gemma-2-9b-it"
        self.api_keys = []
        self.mlx_model = None
        self.mlx_tokenizer = None
        
        # Load API Keys via Shared Logic (Robust)
        self.api_keys = load_api_keys()
        self.fallback_keys = load_text_keys()
        self.using_fallback = False
        
        # Load Config for Engine Settings (Optional)
        if not config_path:
            # Try to find env_vars.yaml in same dir or parent
            base = Path(__file__).parent
            candidates = [base / "env_vars.yaml", base.parent / "env_vars.yaml"]
            for c in candidates:
                if c.exists():
                    config_path = c
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # 1. Determine Backend
                if config and config.get("TEXT_ENGINE") == "local_gemma":
                    self.backend = "local_gemma"
                    custom_path = config.get("LOCAL_MODEL_PATH")
                    if custom_path:
                        # Check if it's a full path, or just a name in the volume
                        volume_path = Path("/Volumes/ORICO/1_o_gemmas") / custom_path
                        if volume_path.exists():
                            self.local_model_path = str(volume_path)
                        else:
                            self.local_model_path = custom_path
                
                # Auto-detect default if not explicitly set but mode IS local
                elif self.backend == "local_gemma":
                     pass
                
                # 2. Key Loading (Fallback if shared failed?)
                # If shared loaded nothing, try config? 
                # mvp_shared is usually better because it checks central locations.
                if not self.api_keys:
                     keys_str = config.get("ACTION_KEYS_LIST", "") or config.get("GEMINI_API_KEYS", "") or config.get("GEMINI_API_KEY", "")
                     if keys_str:
                        self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
                     
            except Exception as e:
                logging.error(f"[-] TextEngine Config Load Error: {e}")
        
        # Initialize Rotation
        if self.api_keys:
            random.shuffle(self.api_keys) # Shuffle once to avoid sync patterns
        # Initialize Rotation
        if self.api_keys:
            random.shuffle(self.api_keys) # Shuffle once to avoid sync patterns
            self.key_cycle = itertools.cycle(self.api_keys)
        else:
            self.key_cycle = None
            
        # Fallback Rotation
        if self.fallback_keys:
            random.shuffle(self.fallback_keys)
            self.fallback_cycle = itertools.cycle(self.fallback_keys)
        else:
            self.fallback_cycle = None

        logging.info(f"🧠 TextEngine initialized. Backend: {self.backend.upper()}")
        if self.backend == "local_gemma":
            self._init_local_model()

    def _init_local_model(self):
        """Loads MLX model into memory (if available)."""
        global MLX_AVAILABLE
        if not MLX_AVAILABLE:
            try:
                import mlx_lm
                MLX_AVAILABLE = True
            except ImportError:
                logging.error("[-] TextEngine: 'mlx_lm' not installed. Falling back to Gemini API.")
                self.backend = "gemini_api"
                return

        logging.info(f"   ⏳ Loading Local Model: {self.local_model_path}...")
        try:
            from mlx_lm import load
            self.mlx_model, self.mlx_tokenizer = load(self.local_model_path)
            logging.info("   ✅ Local Model Loaded.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load local model: {e}. Falling back to Gemini.")
            self.backend = "gemini_api"

    def get_gemini_client(self):
        if not self.api_keys or not self.key_cycle:
            logging.error("[-] TextEngine: No Gemini API keys found.")
            return None
        key = next(self.key_cycle)
    def get_gemini_client(self):
        # Choose cycle based on state
        cycle = self.fallback_cycle if (self.using_fallback and self.fallback_cycle) else self.key_cycle
        
        if not cycle:
             # Try failover if primary empty but fallback exists (edge case)
             if not self.using_fallback and self.fallback_cycle:
                 self.using_fallback = True
                 cycle = self.fallback_cycle
             else:
                 logging.error("[-] TextEngine: No Gemini API keys found.")
                 return None

        key = next(cycle)
        return genai.Client(api_key=key)

    def generate(self, prompt, temperature=0.7, json_schema=None):
        """
        Universal generation method.
        Returns: String (Text or JSON string)
        """
        full_prompt = f"{SYSTEM_FILTER}\n\nUSER REQUEST:\n{prompt}"
        
        if self.backend == "local_gemma" and self.mlx_model:
            return self._generate_local(full_prompt, temperature)
        else:
            return self._generate_gemini(full_prompt, temperature, json_schema)

    def _generate_local(self, prompt, temperature):
        from mlx_lm import generate
        # Simple generation
        # Gemma 2 instruct formatting usually handled by tokenizer?
        # We can use tokenizer.apply_chat_template if available, or just raw prompt if model expects it.
        # For robustness, we'll try to use chat template if tokenizer supports it.
        
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            input_text = self.mlx_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for raw completion models
            input_text = prompt
            
        logging.info("   🧠 Local Inference (Gemma)...")
        response = generate(
            self.mlx_model, 
            self.mlx_tokenizer, 
            prompt=input_text, 
            temp=temperature, 
            max_tokens=8192,  # Liberal limit for scripts
            verbose=False
        )
        return response

    def _generate_gemini(self, prompt, temperature, json_schema):
        client = self.get_gemini_client()
        if not client: return ""
        
        logging.info("   ☁️  Cloud Inference (Gemini)...")
        
        config_args = {
            "temperature": temperature,
        }
        
        # If strict JSON schema provided
        if json_schema:
            config_args["response_mime_type"] = "application/json"
            # OpenAI/Gemini strict schema support varies, usually prompt engineering is enough with 'response_mime_type'
            
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", # Fast, smart
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            if response.text:
                return response.text
            return ""
        except Exception as e:
            logging.error(f"[-] Gemini Gen Error: {e}")
        # Retry Loop for 429
        max_retries = 3
        for attempt in range(max_retries):
            client = self.get_gemini_client()
            if not client: return ""
            
            try:
                # Logging key source? No, secure.
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash", # Fast, smart
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_args)
                )
                if response.text:
                    return response.text
                return ""
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    logging.warning(f"   ⚠️ Quota Hit (Attempt {attempt+1}).")
                    if not self.using_fallback and self.fallback_cycle:
                        logging.warning("   🔄 Switching to TEXT_KEYS_LIST fallback pool.")
                        self.using_fallback = True
                        # Retry immediately with new pool
                        continue
                    else:
                        # Backoff
                        import time
                        time.sleep(2 * (attempt + 1))
                else:
                    logging.error(f"[-] Gemini Gen Error: {e}")
                    return ""
        return ""

# Singleton instance helper
_ENGINE = None
def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TextEngine()
    return _ENGINE
