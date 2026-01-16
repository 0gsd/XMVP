import os
import logging
from google import genai
from google.genai import types
import definitions
from text_engine import TextEngine, get_engine
from mvp_shared import load_api_keys, load_text_keys

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TruthSafety:
    def __init__(self, api_key=None):
        # 1. Vision Client (Standard Key for Images)
        # We still need a standard key for image generation/vision tasks
        if not api_key:
            keys = load_api_keys()
            api_key = keys[0] if keys else None
        
        self.client = None
        if api_key:
             self.client = genai.Client(api_key=api_key)
             
        self.l_model = definitions.VIDEO_MODELS["L"] 
        self.image_model = definitions.IMAGE_MODEL # Use central definition
        self.safety_prompt = definitions.SANITIZATION_PROMPT
        
        # 2. Text Engine (Privileged Text Keys)
        # TruthSafety uses the high-quota text keys for its logic to preserve Action keys.
        text_keys = load_text_keys()
        # Initialize TextEngine with specific keys if available, otherwise it falls back
        # We create a specialized engine instance for TruthSafety
        # Initialize TextEngine 
        # TextEngine loads keys internally (api_keys check env_vars.yaml)
        # It handles fallback to TEXT_KEYS_LIST automatically.
        self.engine = get_engine()

    def describe_image(self, image_path):
        """Get a dense description of the image for reconstruction."""
        # Vision requires API
        if not self.client:
            logging.error("[-] No API key for Vision tasks.")
            return "A cinematic scene."

        logging.info(f"   👁️ Description Scan: {os.path.basename(image_path)}...")
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Gemini Vision Query
            prompt = "Describe this image in extreme detail for the purpose of DALL-E/Midjourney recreation. Focus on composition, colors, and subject matter. Do not mention names of people."
            
            response = self.client.models.generate_content(
                model=self.l_model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=prompt),
                            types.Part(inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=image_bytes
                            ))
                        ]
                    )
                ]
            )
            return response.text
        except Exception as e:
            logging.error(f"   ❌ Description Failed: {e}")
            return "A cinematic scene."

    def wash_image(self, image_path):
        """
        1. Describes image.
        2. Generates new sanitized image (Dazzle Camouflage).
        3. Returns path to new image.
        """
        if not os.path.exists(image_path):
            return None

        logging.info("   🧼 Sanitizing Context Image (Truth & Safety)...")
        
        # 1. Describe
        description = self.describe_image(image_path)
        
        # 2. Re-Generate (Sanitize)
        # Combine Description + Safety Prompt
        full_prompt = f"{description}\n\nSTYLE INSTRUCTION: {self.safety_prompt}"
        
        logging.info(f"   🎨 Repainting with Image Model: {self.image_model}...")
        
        try:
            # POLYGLOT LOGIC (Same as cartoon_producer.py)
            if "imagen" in self.image_model.lower():
                # Method A: Imagen (generate_images)
                response = self.client.models.generate_images(
                    model=self.image_model,
                    prompt=full_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9" 
                    )
                )
                if response.generated_images:
                    new_image_bytes = response.generated_images[0].image.image_bytes
                    return self._save_image(new_image_bytes, image_path)
            
            else:
                # Method B: Gemini Flash (generate_content)
                ar_prompt = f"Generate an image of {full_prompt} --aspect_ratio 16:9"
                
                response = self.client.models.generate_content(
                    model=self.image_model,
                    contents=ar_prompt
                )
                
                # Check for inline data
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            return self._save_image(part.inline_data.data, image_path)
                            
            logging.warning("   ⚠️ Repaint failed: No image data returned.")

        except Exception as e:
            logging.error(f"   ❌ Repaint Failed: {e}")

        return image_path # Final Fallback to original

    def _save_image(self, image_bytes, original_path):
        """Helper to save sanitized image."""
        try:
            dir_name = os.path.dirname(original_path)
            base_name = os.path.basename(original_path)
            sanitized_path = os.path.join(dir_name, f"sanitized_{base_name}")
            
            with open(sanitized_path, "wb") as f:
                f.write(image_bytes)
            
            logging.info(f"   ✨ Sanitized Image Saved: {sanitized_path}")
            return sanitized_path
        except Exception as e:
             logging.error(f"   ❌ Failed to save image: {e}")
             return original_path

    def refine_prompt(self, prompt, context_dict=None, pg_mode=False):
        """
        The Core Truth & Safety Logic.
        1. TRUTH: Checks for physical coherence, logic, and style alignment.
        2. SAFETY: Enforces safety guidelines (Standard or PG).
        Returns the refined prompt.
        """
        logging.info(f"   ⚖️  Truth & Safety Audit (PG: {pg_mode})...")
        
        # Build Context String
        ctx_str = ""
        if context_dict:
            for k, v in context_dict.items():
                ctx_str += f"- {k.upper()}: {v}\n"
        
        # Truth Instructions
        truth_instruction = (
            "You are the Truth & Safety Engine for a video production pipeline.\n"
            "Your Goal: Refine the User Prompt to be Safe, Coherent, and High Quality.\n\n"
            "PHASE 1: TRUTH (Coherence & Physics)\n"
            "- Ensure the scene obeys standard physics (unless context implies fantasy).\n"
            "- If the prompt mentions text (e.g., 'a sign saying Hello'), REMOVE the specific text request and describe the visual instead (e.g., 'a shop sign with legible lettering'). Video models fail at text.\n"
            "- Ensure lighting and composition terms are consistent.\n\n"
            "PHASE 2: CONTEXT ALIGNMENT\n"
            f"Context Provided:\n{ctx_str}\n"
            "- If context (Style, Character) is provided, subtly weave it into the visual description if missing.\n\n"
        )
        
        # Safety Instructions (From soften_prompt)
        if pg_mode:
            # RELAXED / PG MODE
            safety_instruction = (
                "PHASE 3: PG SAFETY GUIDELINES\n"
                "1. Remove any mention of children. Replace 'boy', 'girl', 'child' with 'adult man' or 'adult woman'. "
                "2. Remove violence, gore, or nudity. "
                "3. CELEBRITY HANDLING: If a specific celebrity is named, replace their name with their Profession + Initials. "
                "   Example: 'Nicolas Cage' -> 'actor N.C.', 'Taylor Swift' -> 'singer T.S.'. "
            )
        else:
            # STANDARD / SAFE MODE
            safety_instruction = (
                "PHASE 3: STRICT SAFETY GUIDELINES\n"
                "1. Remove any mention of children, public figures, violence, or gore. "
                "2. CELEBRITY HANDLING: If a specific celebrity or public figure is named, RETAIN the name but explicitly phrase it as 'an impersonator performing in character as [Name]'. "
                "   Example: 'Tom Cruise jumping' -> 'an impersonator performing in character as Tom Cruise jumping'. "
                "3. For non-celebrity people (e.g. 'a boy', 'a girl'), change them to 'an adult man' or 'an adult woman'. "
            )
            
        final_instruction = (
            "OUTPUT INSTRUCTION:\n"
            "Return ONLY the refined prompt. Do not add 'Here is the prompt:' or explanations."
        )
        
        full_wrapper = f"{truth_instruction}\n{safety_instruction}\n{final_instruction}\n\nUSER PROMPT: {prompt}"
        
        try:
            # Use the Text Engine (with Text Keys)
            new_prompt = self.engine.generate(full_wrapper, temperature=0.7)
            
            if new_prompt:
                clean_prompt = new_prompt.strip()
                # logging.info(f"   ✨ Refined: {clean_prompt[:60]}...")
                return clean_prompt
            else:
                logging.warning("   ⚠️ TruthSafety returned empty. Using original.")
                return prompt
                
        except Exception as e:
            logging.error(f"   ❌ TruthSafety Failed: {e}")
            return prompt

    def soften_prompt(self, prompt, pg_mode=False):
        """Legacy wrapper for backward compatibility, now routes to refine_prompt."""
        return self.refine_prompt(prompt, pg_mode=pg_mode)
