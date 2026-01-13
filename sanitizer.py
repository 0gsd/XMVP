import os
import logging
from google import genai
from google.genai import types
import action
import definitions
import time

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Sanitizer:
    def __init__(self, api_key=None):
        if not api_key:
            keys = action.load_action_keys()
            api_key = keys[0] if keys else None
        
        if not api_key:
            raise ValueError("No API key available for Sanitizer.")

        self.client = genai.Client(api_key=api_key)
        self.l_model = definitions.VIDEO_MODELS["L"] # Gemini Flash Lite/Preview
        self.image_model = definitions.IMAGE_MODEL # Imagen 4 Fast
        self.safety_prompt = definitions.SANITIZATION_PROMPT

    def describe_image(self, image_path):
        """Get a dense description of the image for reconstruction."""
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
        # Try Imagen First
        try:
            logging.info(f"   🎨 Repainting with Image Model: {self.image_model}...")
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
                
        except Exception as e:
            logging.warning(f"   ⚠️ Imagen Repaint Failed: {e}. Falling back to Gemini...")
            
            # Fallback: Gemini 2.5 Flash Image
            try:
                fallback_model = "models/gemini-2.5-flash-image"
                logging.info(f"   🎨 Repainting with Fallback: {fallback_model}...")
                
                # Gemini generate_content for images
                start_t = time.time()
                response = self.client.models.generate_content(
                    model=fallback_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(response_mime_type="image/png") 
                )
                
                # Check for inline data
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            return self._save_image(part.inline_data.data, image_path)
                
                logging.error("   ❌ Fallback failed: No inline image data returned.")

            except Exception as e2:
                logging.error(f"   ❌ Fallback Repaint Failed: {e2}")

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

    def soften_prompt(self, prompt):
        """Rewrites a prompt to be safe for work while maintaining visual intent."""
        logging.info(f"   🛡️ Softening Prompt via {self.l_model}...")
        try:
            safety_instruction = (
                "Rewrite the following video prompt to be compliant with strict safety guidelines. "
                "Remove any mention of children, public figures, violence, or gore. "
                "CRITICAL: If a specific celebrity or public figure is named, RETAIN the name but explicitly phrase it as 'an impersonator performing in character as [Name]'. "
                "Example: 'Tom Cruise jumping' -> 'an impersonator performing in character as Tom Cruise jumping'. "
                "For non-celebrity people (e.g. 'a boy', 'a girl'), change them to 'an adult man' or 'an adult woman' to avoid child safety triggers. "
                "Keep the visual style and composition exactly the same. "
                "Output ONLY the new prompt."
            )
            
            response = self.client.models.generate_content(
                model=self.l_model,
                contents=f"{safety_instruction}\n\nPROMPT: {prompt}"
            )
            
            if response.text:
                new_prompt = response.text.strip()
                logging.info(f"   ✨ Softened: {new_prompt[:50]}...")
                return new_prompt
                
        except Exception as e:
            logging.error(f"   ❌ Soften Failed: {e}")
            
        return prompt # Fallback: Return original

