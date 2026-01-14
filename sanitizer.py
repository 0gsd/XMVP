import os
import logging
from google import genai
from google.genai import types
import action
import definitions
from text_engine import get_engine

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Sanitizer:
    def __init__(self, api_key=None):
        # Keep client for Vision tasks (Describe/Wash)
        if not api_key:
            keys = action.load_action_keys()
            api_key = keys[0] if keys else None
        
        self.client = None
        if api_key:
             self.client = genai.Client(api_key=api_key)
             
        self.l_model = definitions.VIDEO_MODELS["L"] 
        self.image_model = "gemini-2.5-flash-image"
        self.safety_prompt = definitions.SANITIZATION_PROMPT
        
        # Text Engine for Text Logic (Softening)
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

        logging.info("   🧼 Sanitizing Context Image...")
        
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

    def soften_prompt(self, prompt, pg_mode=False):
        """Rewrites a prompt to be safe for work while maintaining visual intent."""
        # Uses TextEngine (Local or Cloud)
        logging.info(f"   🛡️ Softening Prompt via {self.l_model} (PG Mode: {pg_mode})...")
        
        if pg_mode:
            # RELAXED / PG MODE
            safety_instruction = (
                "Rewrite the following video prompt to be compliant with PG safety guidelines. "
                "1. Remove any mention of children. Replace 'boy', 'girl', 'child' with 'adult man' or 'adult woman'. "
                "2. Remove violence, gore, or nudity. "
                "3. CELEBRITY HANDLING: If a specific celebrity is named, replace their name with their Profession + Initials. "
                "   Example: 'Nicolas Cage' -> 'actor N.C.', 'Taylor Swift' -> 'singer T.S.'. "
                "   Do NOT use the word 'impersonator'. "
                "Keep the visual style and composition exactly the same. "
                "Output ONLY the new prompt."
            )
        else:
            # STANDARD / SAFE MODE
            safety_instruction = (
                "Rewrite the following video prompt to be compliant with strict safety guidelines. "
                "Remove any mention of children, public figures, violence, or gore. "
                "CRITICAL: If a specific celebrity or public figure is named, RETAIN the name but explicitly phrase it as 'an impersonator performing in character as [Name]'. "
                "Example: 'Tom Cruise jumping' -> 'an impersonator performing in character as Tom Cruise jumping'. "
                "For non-celebrity people (e.g. 'a boy', 'a girl'), change them to 'an adult man' or 'an adult woman' to avoid child safety triggers. "
                "Keep the visual style and composition exactly the same. "
                "Output ONLY the new prompt."
            )
        
        try:
            full_prompt = f"{safety_instruction}\n\nPROMPT: {prompt}"
            # Use Text Engine
            new_prompt = self.engine.generate(full_prompt, temperature=0.7)
            
            if new_prompt:
                clean_prompt = new_prompt.strip()
                logging.info(f"   ✨ Softened: {clean_prompt[:50]}...")
                return clean_prompt
                
        except Exception as e:
            logging.error(f"   ❌ Soften Failed: {e}")
            
        return prompt # Fallback: Return original
