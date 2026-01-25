#!/usr/bin/env python3
import os
import sys
import yaml
import json
import logging
import random
from pathlib import Path

import itertools

# Strict Import with Legacy Fallback
GEMINI_AVAILABLE = False
try:
    # Try New SDK (v1.0+)
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    IS_LEGACY_SDK = False
except ImportError:
    # Try Legacy SDK (google-generativeai)
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        IS_LEGACY_SDK = True
    except ImportError:
        logging.warning("⚠️ Google GenAI SDK (v1.0+ or Legacy) not found. Cloud features disabled.")
        pass

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
        self.local_adapter_path = None
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
        env_backend = os.environ.get("TEXT_ENGINE")
        
        if env_backend == "local_gemma":
             logging.info("   🔧 Overriding Text Engine Backend via ENV: local_gemma")
             self.backend = "local_gemma"
             # Optionally set model path if also in env, otherwise default
             self.local_model_path = os.environ.get("LOCAL_MODEL_PATH", "mlx-community/gemma-2-9b-it-4bit")
             self.local_adapter_path = os.environ.get("LOCAL_ADAPTER_PATH")

        elif env_backend == "gemini_cloud":
             logging.info("   ☁️  Overriding Text Engine Backend via ENV: gemini_cloud (Forcing Cloud)")
             self.backend = "gemini_api"
             # Skip profile check below
        
        # 0.5 Check active_models.json (Profile Priority)
        # If the user has explicitly set their profile to a local text model, respect it!
        elif True: # Use elif to chain correctly after the explicit overrides checking config logic... 
            # Wait, the structure was: 
            # if env == local: ...
            # else: check profile
            # So I should keep that structure but handle gemini_cloud
            pass 
            
            try:
                # Look for active_models.json in standard locations
                am_candidates = [
                    Path("active_models.json"),
                    Path(__file__).parent / "active_models.json",
                    Path(__file__).parent.parent.parent / "active_models.json"
                ]
                for am_path in am_candidates:
                    if am_path.exists():
                        with open(am_path, 'r') as f:
                            profile = json.load(f)
                            text_model = profile.get("text", "")
                            # If model is gemma-2-9b-it (Local), switch backend
                            if "gemma" in text_model or "local" in text_model:
                                logging.info(f"   🔧 Active Profile ({text_model}) requests Local Backend.")
                                self.backend = "local_gemma"
                                # We leave paths to defaults or config (loaded below)
                                # But we MUST ensure we don't accidentally revert to gemini later.
                        break
            except Exception as e:
                pass # Fail silently, fall back to config

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # 1. Determine Backend (Yaml Config overrides Profile)
                if config and config.get("TEXT_ENGINE") == "local_gemma":
                    self.backend = "local_gemma"
                    custom_path = config.get("LOCAL_MODEL_PATH")
                    if custom_path:
                        # Check if it's a full path, or just a name in the volume
                        volume_path = Path("/Volumes/XMVPX/mw/gemma-root") / custom_path
                        if volume_path.exists():
                            self.local_model_path = str(volume_path)
                        else:
                            self.local_model_path = custom_path
                    self.local_adapter_path = config.get("LOCAL_ADAPTER_PATH")
                
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
                logging.error("[-] TextEngine: 'mlx_lm' not installed.")
                if self.backend == "local_gemma":
                     logging.error("❌ CRITICAL: Local Mode requested but MLX not found. Aborting to prevent accidental Cloud costs.")
                     raise RuntimeError("MISSING_DEPENDENCY: mlx_lm is required for 'local_gemma' backend.")
                
                logging.error("Falling back to Gemini API.")
                self.backend = "gemini_api"
                return

        logging.info(f"   ⏳ Loading Local Model: {self.local_model_path}...")
        if self.local_adapter_path:
             logging.info(f"   🧩 Loading Adapter: {self.local_adapter_path}")

        try:
            from mlx_lm import load
            self.mlx_model, self.mlx_tokenizer = load(
                self.local_model_path, 
                adapter_path=self.local_adapter_path
            )
            logging.info("   ✅ Local Model Loaded.")
        except Exception as e:
            logging.error(f"   ❌ Failed to load local model: {e}.")
            # STRICT LOCAL (Requested by User): Die if local fails.
            # logging.error("Falling back to Gemini.") # NOPE.
            # self.backend = "gemini_api"
            raise RuntimeError("CRITICAL: Local Model Failed in Strict Local Mode.")
            return

    def unload(self):
        """
        Manually unloads the local model to free memory.
        Crucial when running concurrent heavy models (e.g. Wan Video).
        """
        if self.mlx_model is not None:
             logging.info("   🗑️  Unloading Text Engine from RAM...")
             del self.mlx_model
             del self.mlx_tokenizer
             self.mlx_model = None
             self.mlx_tokenizer = None
             
             import gc
             gc.collect()
             
             try:
                 import mlx.core as mx
                 if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                 else:
                    mx.metal.clear_cache()
             except:
                 pass
                 
             logging.info("   ✅ Text Engine Unloaded.")
             
    def _ensure_loaded(self):
        """Lazy load check."""
        if self.backend == "local_gemma" and self.mlx_model is None:
             logging.info("   ♻️  Reloading Text Engine (Lazy Load)...")
             self._init_local_model()

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
        
        # LEGACY SDK SUPPORT
        if IS_LEGACY_SDK:
            try:
                genai.configure(api_key=key)
                return "LEGACY_CLIENT" # Signal for _generate_gemini to use legacy path
            except Exception as e:
                logging.error(f"Legacy Configure Error: {e}")
                return None
                
        return genai.Client(api_key=key)

    def _generate_gemini_legacy(self, prompt, temperature, json_schema):
         """Fallback for older google-generativeai SDK."""
         try:
             # Map 2.0 -> 1.5 if on legacy? No, try 2.0 first.
             model_name = "gemini-2.0-flash" 
             model = genai.GenerativeModel(model_name)
             
             config = genai.types.GenerationConfig(
                 temperature=temperature
             )
             if json_schema:
                 config.response_mime_type = "application/json"
                 
             response = model.generate_content(prompt, generation_config=config)
             if response.text:
                 return response.text
             return ""
         except Exception as e:
             # Retry logic handled by caller? No, we need internal retry or bubble up.
             # For simplicity, fallback to bubble up -> _generate_gemini loop?
             # Actually _generate_gemini calls this.
             if "404" in str(e) or "not found" in str(e).lower():
                 # Maybe 2.0 not avail on legacy? Fallback to 1.5
                 try:
                     logging.info("   ⚠️ Legacy SDK: 2.0 Flash not found. Falling back to 1.5 Pro.")
                     model = genai.GenerativeModel("gemini-1.5-pro")
                     response = model.generate_content(prompt, generation_config=config)
                     return response.text
                 except:
                     pass
             raise e

    def _generate_gemini(self, prompt, temperature, json_schema):
        client = self.get_gemini_client()
        if not client: return ""
        
        logging.info("   ☁️  Cloud Inference (Gemini)...")
        
        # LEGACY BRANCH
        if IS_LEGACY_SDK or client == "LEGACY_CLIENT":
            try:
                return self._generate_gemini_legacy(prompt, temperature, json_schema)
            except Exception as e:
                logging.error(f"[-] Gemini Legacy Error: {e}")
                # Simple retry logic reuse?
                # For now just fail soft.
                return ""
        
        # V1 BRANCH (Standard)
        config_args = {
            "temperature": temperature,
        }
        
        # If strict JSON schema provided
        if json_schema:
            config_args["response_mime_type"] = "application/json"
            
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
            client = self.get_gemini_client() # Rotates key
            if not client: return ""
            
            # LEGACY RETRY
            if IS_LEGACY_SDK or client == "LEGACY_CLIENT":
                try:
                    return self._generate_gemini_legacy(prompt, temperature, json_schema)
                except:
                    import time
                    time.sleep(2)
                    continue

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
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    logging.warning(f"   ⚠️ Quota Hit (Attempt {attempt+1}).")
                    if not self.using_fallback and self.fallback_cycle:
                        logging.warning("   🔄 Switching to TEXT_KEYS_LIST fallback pool.")
                        self.using_fallback = True
                        continue
                    else:
                        import time
                        time.sleep(2 * (attempt + 1))
                else:
                    logging.error(f"[-] Gemini Gen Error: {e}")
                    return ""
        return ""

    def _clean_json_output(self, text):
        """
        Robustly extracts JSON from text.
        """
        clean_text = text
        if "```" in clean_text:
            try:
                parts = clean_text.split("```")
                if len(parts) >= 3:
                     clean_text = parts[1]
                     if clean_text.lower().startswith("json"): clean_text = clean_text[4:]
                     elif clean_text.lower().startswith("python"): clean_text = clean_text[6:]
                elif len(parts) == 2:
                     clean_text = parts[1]
            except:
                pass 
        clean_text = clean_text.strip()
        import re
        # Strip special tokens (e.g. <pad>, <end_of_turn>)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = re.sub(r'[\x00-\x1f\x7f]', lambda m: "" if m.group(0) in "\n\r\t" else "", clean_text)
        try:
            json.loads(clean_text)
            return clean_text
        except:
            pass
        idx = -1
        first_brace = clean_text.find('{')
        first_bracket = clean_text.find('[')
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket): idx = first_brace
        elif first_bracket != -1: idx = first_bracket
        if idx != -1:
            try:
                obj, end = json.JSONDecoder().raw_decode(clean_text[idx:])
                return clean_text[idx:idx+end]
            except:
                pass
        return clean_text

    def generate(self, prompt, temperature=0.7, json_schema=None):
        """
        Universal generation method.
        """
        full_prompt = f"{SYSTEM_FILTER}\n\nUSER REQUEST:\n{prompt}"
        raw_text = ""
        if self.backend == "local_gemma" and self.mlx_model:
            raw_text = self._generate_local(full_prompt, temperature)
        else:
            raw_text = self._generate_gemini(full_prompt, temperature, json_schema)
        if not raw_text: return ""
        if json_schema:
            return self._clean_json_output(raw_text)
        clean_text = raw_text
        if "```" in clean_text:
            try:
                parts = clean_text.split("```")
                if len(parts) >= 3:
                     clean_text = parts[1]
                     if clean_text.startswith("json"): clean_text = clean_text[4:]
                     elif clean_text.startswith("python"): clean_text = clean_text[6:]
                else:
                    clean_text = parts[1]
            except:
                pass 
        return clean_text.strip()

    def _generate_local(self, prompt, temperature):
        self._ensure_loaded()
        from mlx_lm import stream_generate
        import mlx.core as mx
        
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            try:
                input_text = self.mlx_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logging.warning(f"   ⚠️ Chat Template Error: {e}. using Manual Gemma Formatting.")
                input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            input_text = prompt
            
        logging.info("   🧠 Local Inference (Gemma)...")
        
        def temp_sampler(logits):
            if temperature == 0: return mx.argmax(logits, axis=-1)
            logits = logits / temperature
            return mx.random.categorical(logits)
            
        print("   Generating", end="", flush=True)
        response_text = ""
        token_count = 0
        
        try:
            for response in stream_generate(self.mlx_model, self.mlx_tokenizer, prompt=input_text, max_tokens=8192, sampler=temp_sampler):
                text_chunk = response.text
                response_text += text_chunk
                token_count += 1
                if token_count % 10 == 0: print(".", end="", flush=True)
                if token_count % 100 == 0: print(f"[{token_count}]", end="", flush=True)
                
            print(f" ✅ ({token_count} tokens)")
            return response_text
            
        except TypeError as e:
            logging.error(f"Generation Error (Args): {e}")
            logging.warning("Retrying with default greedy options...")
            return self._generate_local_fallback(input_text)
            
        finally:
            # MEMORY OPTIMIZATION: Clear Metal Cache
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                else:
                    mx.metal.clear_cache()
            except:
                pass

    def _generate_local_fallback(self, prompt):
        from mlx_lm import generate
        import mlx.core as mx
        res = generate(self.mlx_model, self.mlx_tokenizer, prompt=prompt, verbose=False)
        mx.metal.clear_cache()
        return res

    def clear_cache(self):
        """Manually clears backend cache."""
        if self.backend == "local_gemma" and MLX_AVAILABLE:
            try:
                import mlx.core as mx
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                else:
                    mx.metal.clear_cache()
                # logging.debug("   🧹 Metal Cache Cleared.")
            except:
                pass

# Singleton instance helper
_ENGINE = None
def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TextEngine()
    return _ENGINE
