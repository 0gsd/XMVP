import os
import logging
import yaml
from pathlib import Path
from google import genai
from google.genai import types

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_keys():
    # From tools/fmv/mvp/v0.5/test_gen_capabilities.py -> tools/fmv/env_vars.yaml
    env_file = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"
    if env_file.exists():
        with open(env_file, "r") as f:
            data = yaml.safe_load(f)
            keys = data.get("ACTION_KEYS_LIST", "").split(",")
            return [k.strip() for k in keys if k.strip()]
    return [os.environ.get("GEMINI_API_KEY")]

def test_imagen_model(model_name):
    keys = load_keys()
    if not keys or not keys[0]:
        print("No keys found.")
        return

    client = genai.Client(api_key=keys[0])
    prompt = "A cute cartoon cat."
    
    print(f"\n🧪 Testing Imagen Model: {model_name}")
    try:
        response = client.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1", 
            )
        )
        if response.generated_images:
            img = response.generated_images[0]
            if img.image:
                print("   ✅ SUCCESS! Received image.")
            else:
                 print("   ⚠️ Received response but no image bytes?")
        else:
             print("   ❌ No images returned.")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")

def test_gemini_model(model_name):
    keys = load_keys()
    if not keys or not keys[0]:
        print("No keys found.")
        return

    client = genai.Client(api_key=keys[0])
    prompt = "Generate a cute cartoon cat."
    
    print(f"\n🧪 Testing Gemini Model: {model_name}")
    try:
        # Method: generate_content with prompt asking for image
        ar_prompt = f"Generate an image of {prompt}" # Simulating Polyglot logic
        
        response = client.models.generate_content(
            model=model_name,
            contents=ar_prompt
        )
        
        if response.candidates:
            part = response.candidates[0].content.parts[0]
            if part.inline_data:
                print(f"   ✅ SUCCESS! Received inline_data (Mime: {part.inline_data.mime_type})")
                
                # Check mime type
                if "image" in part.inline_data.mime_type:
                    print("   🎉 It is an image!")
                else:
                    print(f"   ⚠️ Warning: Mime type is {part.inline_data.mime_type}")
                    
            elif part.text:
                print(f"   ❌ RECEIVED TEXT INSTEAD: {part.text[:100]}...")
            else:
                print("   ❓ Received empty/unknown part.")
        else:
            print("   ❌ No candidates returned.")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")

if __name__ == "__main__":
    # Test L-Tier Candidate (Gemini Flash Image)
    test_gemini_model("gemini-2.5-flash-image")
    
    # Test J-Tier Candidate
    # test_imagen_model("imagen-4.0-fast-generate-001")
