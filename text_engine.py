#!/usr/bin/env python3
import os
import sys
import yaml
import json
import logging
import random
from pathlib import Path

import itertools

# Strict Import (No more legacy fallback)
from google import genai
from google.genai import types
GEMINI_AVAILABLE = True

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
        self.local_model_path = "mlx-community/gemma-2-9b-it-4bit"
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
        
        # 0. Check Environment Variable Override (Highest Priority)
        # This allows movie_producer --local to force local mode without editing yaml
        if os.environ.get("TEXT_ENGINE") == "local_gemma":
             logging.info("   🔧 Overriding Text Engine Backend via ENV: local_gemma")
             self.backend = "local_gemma"
             # Optionally set model path if also in env, otherwise default
             self.local_model_path = os.environ.get("LOCAL_MODEL_PATH", "mlx-community/gemma-2-9b-it-4bit")

        elif config_path and os.path.exists(config_path):
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
        all_keys = self.api_keys[:]
        
        # User requested aggressive pooling of ALL keys for high-throughput modes?
        # If we want to support "round robin through all available ACTION_KEYS_LIST -and- TEXT_KEYS_LIST"
        # We can implement a flag or just do it by default if fallback keys exist?
        # User said: "we might as well round robin through all available... with intelligent backoff"
        # Let's pool them if we can.
        if self.fallback_keys:
             # Dedup
             extras = [k for k in self.fallback_keys if k not in all_keys]
             all_keys.extend(extras)
             logging.info(f"   🔑 TextEngine: Pooled {len(all_keys)} Total Keys (Action + Text) for Rotation.")
        
        if all_keys:
            random.shuffle(all_keys) 
            self.key_cycle = itertools.cycle(all_keys)
            self.fallback_cycle = itertools.cycle(all_keys) # Same pool for fallback
        else:
            self.key_cycle = None
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
            logging.error(f"   ❌ Failed to load local model: {e}.")
            # STRICT LOCAL (Requested by User): Die if local fails.
            # logging.error("Falling back to Gemini.") # NOPE.
            # self.backend = "gemini_api"
            raise RuntimeError("CRITICAL: Local Model Failed in Strict Local Mode.")
            return

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

    def get_model_instance(self):
        """
        Returns a configured genai.GenerativeModel object (V1 SDK).
        Useful for passing to legacy scripts like frame_canvas.py
        that expect a model object, not just a client.
        """
        # Retrieve key logic duplicated from get_gemini_client
        cycle = self.fallback_cycle if (self.using_fallback and self.fallback_cycle) else self.key_cycle
        
        if not cycle:
             # Try failover
             if not self.using_fallback and self.fallback_cycle:
                 self.using_fallback = True
                 cycle = self.fallback_cycle
             else:
                 logging.error("[-] TextEngine: No Gemini API keys found.")
                 return None

        key = next(cycle)

        # Quick Fix (MVP): Re-import V1 here and return V1 model object using the key.
        try:
             import google.generativeai as genai_v1
             genai_v1.configure(api_key=key)
             # Safety: Set API version if needed, but default is usually fine
             return genai_v1.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
             logging.error(f"[-] Failed to bridge V1 model: {e}")
             return None

    def generate(self, prompt, temperature=0.7, json_schema=None):
        """
        Universal generation method.
        Returns: String (Text or JSON string)
        """
        full_prompt = f"{SYSTEM_FILTER}\n\nUSER REQUEST:\n{prompt}"
        
        raw_text = ""
        if self.backend == "local_gemma" and self.mlx_model:
            raw_text = self._generate_local(full_prompt, temperature)
        else:
            raw_text = self._generate_gemini(full_prompt, temperature, json_schema)
            
        if not raw_text: return ""
        
        # --- CLEANING ---
        # 1. Strip Markdown Code Blocks (common in Gemma/Gemini)
        # Handle ```json ... ``` or just ``` ... ```
        clean_text = raw_text
        if "```" in clean_text:
            try:
                # Split by ``` and take the first block that looks like content
                # Usually it's: text ```json CONTENT ``` text
                parts = clean_text.split("```")
                if len(parts) >= 3:
                     # part 0: pre, part 1: content, part 2: post
                     clean_text = parts[1]
                     # If it starts with 'json' or 'python', strip that
                     if clean_text.startswith("json"): clean_text = clean_text[4:]
                     elif clean_text.startswith("python"): clean_text = clean_text[6:]
                else:
                    # Maybe just one block at end?
                    clean_text = parts[1]
            except:
                pass # Fallback to raw if logic fails
        
        return clean_text.strip()

    def _generate_local(self, prompt, temperature):
        from mlx_lm import stream_generate
        import mlx.core as mx
        
        # 1. Prepare Inputs
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            input_text = self.mlx_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = prompt
            
        logging.info("   🧠 Local Inference (Gemma)...")
        
        # 2. Define Sampler (Fix for TypeError: 'temperature')
        # mlx_lm requires a callable sampler if we want to control temp
        def temp_sampler(logits):
            if temperature == 0:
                return mx.argmax(logits, axis=-1)
            # Apply temp
            logits = logits / temperature
            return mx.random.categorical(logits)
            
        # 3. Stream Generation with Progress
        print("   Generating", end="", flush=True)
        response_text = ""
        token_count = 0
        
        # Use simple args 
        # Note: generate_step() in mlx_lm usually takes 'temp' if using high-level generate(), 
        # but here we go direct.
        try:
            for response in stream_generate(
                self.mlx_model, 
                self.mlx_tokenizer, 
                prompt=input_text, 
                max_tokens=8192,
                sampler=temp_sampler
            ):
                text_chunk = response.text
                response_text += text_chunk
                token_count += 1
                
                # Progress Indicator
                if token_count % 10 == 0:
                    print(".", end="", flush=True)
                if token_count % 100 == 0:
                    print(f"[{token_count}]", end="", flush=True)

            print(f" ✅ ({token_count} tokens)")
            return response_text
            
        except TypeError as e:
            # Fallback if sampler arg is also rejected (version chaos)
            logging.error(f"Generation Error (Args): {e}")
            logging.warning("Retrying with default greedy options...")
            return self._generate_local_fallback(input_text)
            
    def _generate_local_fallback(self, prompt):
        from mlx_lm import generate
        return generate(self.mlx_model, self.mlx_tokenizer, prompt=prompt, verbose=False)

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
