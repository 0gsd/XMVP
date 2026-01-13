import os
import json
import time
import logging
import yaml
import argparse
import random
import re
import requests
from pathlib import Path
import google.generativeai as genai
import definitions
import sanitizer

import base64

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
# Point to CENTRAL env_vars.yaml in tools/fmv/
ENV_FILE = Path(__file__).resolve().parent.parent.parent / "env_vars.yaml"

# Output Directories
DIR_SCRIPTS = "actionscript"
DIR_REPORTS = "producerjasonwoliner"
DIR_PARTS = "componentparts"
DIR_FINAL = "finalcuts"

def ensure_directories():
    for d in [DIR_SCRIPTS, DIR_REPORTS, DIR_PARTS, DIR_FINAL]:
        if not os.path.exists(d):
            os.makedirs(d)

def load_action_keys():
    """Loads a list of keys for rotation from env_vars.yaml."""
    keys = []
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, 'r') as f:
                secrets = yaml.safe_load(f)
                
                # 1. Try ACTION_KEYS_LIST (List of strings OR Comma-Separated String)
                action_list = secrets.get("ACTION_KEYS_LIST")
                
                if action_list:
                    if isinstance(action_list, list):
                        logging.info(f"Loaded {len(action_list)} keys from ACTION_KEYS_LIST (List) for rotation.")
                        return action_list
                    elif isinstance(action_list, str):
                        # Split by comma for Cloud Deploy compatibility
                        parsed_list = [k.strip() for k in action_list.split(',') if k.strip()]
                        logging.info(f"Loaded {len(parsed_list)} keys from ACTION_KEYS_LIST (String) for rotation.")
                        return parsed_list
                    
        except Exception as e:
            logging.error(f"Failed to parse env_vars.yaml: {e}")
    
    # Strict Failure: No Fallback
    logging.error("Strict Mode: No ACTION_KEYS_LIST found in env_vars.yaml.")
    return []

def get_optimal_model_name():
    """Same dynamic model finder as daily_grind."""
    logging.info("Connecting to Gemini...")
    try:
        all_models = list(genai.list_models())
        supported_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        flash_models = [m for m in supported_models if 'flash' in m.lower()]
        flash_models.sort(reverse=True)

        # Explicitly prefer gemini-2.0-flash (Cheapest)
        target_model = "models/gemini-2.0-flash"
        if target_model in supported_models:
            logging.info(f"Selected Target Model: {target_model}")
            return target_model

        if flash_models:
            return flash_models[0]
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

def generate_tech_script(api_key):
    """Simulates a Writers Room creating a 64s TV Show Segment (Tech/Sci-Fi)."""
    genai.configure(api_key=api_key)
    model_name = get_optimal_model_name()
    model = genai.GenerativeModel(model_name)
    
    logging.info(f"Showrunner (Tech) active: {model_name}")

    megaprompt = """
    THE TV WRITERS ROOM (YEAR 2125)
    
    CONTEXT:
    TV in 2125 is bizarre, hyper-specific, and visually dense.
    We need to create a Concept, properly script a 64-second scene, and then break it into 8 coherent Video Generation Prompts (8 seconds each).

    THE CAST:
    1. THE SHOWRUNNER (Eccentric, Visionary): Pitches the high-concept weirdness.
    2. THE CONTINUITY COP: Obsessed with visual consistency. Ensures characters wear the EXACT same clothes and faces in every shot.
    3. THE CINEMATOGRAPHER: Focuses on camera angles and 8-second loops.
    4. THE EDITOR: Compiles the final Output.

    TASK:
    1. BRAINSTORM: Pitch a specific, weird TV show concept (e.g., "Real Housewives of the Mars Terraform Colony" or "CSI: Dreamscape").
    2. STORY: Write a 1-minute scene script (Dialogue/Action) based on the pitch.
    3. VISUAL LOCK: Define the Global Assets (Character A appearance, Character B appearance, Setting details). These MUST act as constants.
    4. CHUNKING: Break the story into 8 distinct Prompts.
    
    CONSTRAINT:
    - Each Prompt must start with the VISUAL LOCK description to ensure Veo generates the same characters.
    - Each Prompt must represent exactly 8 seconds of action.
    - Total 8 Prompts.

    OUTPUT FORMAT:
    ## The Pitch & Argument
    [Transcript of PERSONAS arguing about the concept]

    ## Visual Lock (The Bible)
    [Character/Set descriptions to paste into every prompt]

    ## Shooting Schedule (8 Parts)
    1. **Prompt 1 (00-08s)**: [Full Prompt]
    2. **Prompt 2 (08-16s)**: [Full Prompt]
    ...
    8. **Prompt 8 (56-64s)**: [Full Prompt]
    """
    
    try:
        response = model.generate_content(megaprompt)
        if response and response.text:
            return response.text
    except Exception as e:
        logging.error(f"Writers Room collapsed: {e}")
        return None
    return None

def generate_movie_script(api_key, num_segments=12):
    """
    Simulates a Panic Room of Hollywood Writers remaking a classic (1979-2001) in X segments.
    """
    genai.configure(api_key=api_key)
    model_name = get_optimal_model_name()
    model = genai.GenerativeModel(model_name)
    
    # 1. Pick a Movie (The Assignment)
    # We do this first to ensure the writers have a clear target.
    year = random.randint(1979, 2001)
    selector_prompt = f"Pick ONE major Hollywood blockbuster movie released in {year}. Output ONLY the title."
    try:
        resp = model.generate_content(selector_prompt)
        movie_title = resp.text.strip().replace('"', '')
        logging.info(f"🎬 The Assignment: Remake '{movie_title}' ({year}) in {num_segments} clips.")
    except Exception as e:
        movie_title = "The Matrix (1999)" # Fallback
        logging.warning(f"Using fallback movie: {e}")

    # 2. The Panic Room Prompt
    megaprompt = f"""
    THE HOLLYWOOD PANIC ROOM
    
    CONTEXT:
    You are a team of expert screenwriters given an IMPOSSIBLE deadline.
    You must remake the entire movie "{movie_title}" ({year}) in exactly {num_segments} video clips (8 seconds each).
    
    THE CREW:
    - THE HACK: Wants to just skip to the explosions.
    - THE AUTEUR: Wants to capture the "vibe" and "theme".
    - THE MATH NERD: Needs to divide the runtime perfectly.
    
    TASK:
    1.  **Analyze**: Condensed plot summary of {movie_title}.
    2.  **Visual Lock**: Define the specific look of the Protagonist and Antagonist for this remake.
    3.  **The Cut**: Generate exactly {num_segments} Video Generation Prompts.
        -   Each prompt must represent a key beat of the movie.
        -   The sequence must tell the WHOLE story from start to finish.
        -   Each prompt is 8 seconds long.
    
    CONSTRAINT:
    -   Prompt 1 must establish the world.
    -   The final Prompt must be the climax/resolution.
    -   Include the "Visual Lock" descriptions in EVERY PROMPT for consistency.
    
    OUTPUT FORMAT:
    ## The Panic (Transcript)
    [Brief dialogue of them arguing about how to cut it]

    ## Visual Lock
    [Character Consistencies]

    ## Shooting Schedule ({num_segments} Parts)
    1. **Prompt 1 (00-08s)**: [Full Prompt]
    ...
    {num_segments}. **Prompt {num_segments}**: [Full Prompt]
    """
    
    try:
        logging.info(f"   Writing Script for '{movie_title}'...")
        response = model.generate_content(megaprompt)
        if response and response.text:
            return response.text
    except Exception as e:
        logging.error(f"Writers Room collapsed: {e}")
        return None
    return None

def generate_studio_script(api_key, num_segments=12):
    """
    Simulates a Mockumentary Crew filming the chaotic 'Making Of' a classic movie.
    """
    genai.configure(api_key=api_key)
    model_name = get_optimal_model_name()
    model = genai.GenerativeModel(model_name)
    
    # 1. Pick a Movie (The Subject)
    year = random.randint(1979, 2001)
    selector_prompt = f"Pick ONE major Hollywood blockbuster movie released in {year}. Output ONLY the title."
    try:
        resp = model.generate_content(selector_prompt)
        movie_title = resp.text.strip().replace('"', '')
        logging.info(f"🎬 The Studio Subject: The chaotic production of '{movie_title}' ({year}).")
    except Exception as e:
        movie_title = "Titanic (1997)" 
        logging.warning(f"Using fallback movie: {e}")

    # 2. The Mockumentary Prompt
    megaprompt = f"""
    THE STUDIO MOCKUMENTARY
    
    CONTEXT:
    We are filming a "Making Of" documentary about the chaotic production of "{movie_title}" ({year}).
    However, it's a parody. The crew is incompetent, the stars are divas, and everything is going wrong.
    We need exactly {num_segments} video clips (8 seconds each) that capture the behind-the-scenes madness.
    
    THE ARCHETYPES:
    - THE DIVA STAR: Always complaining about their trailer/costume.
    - THE STRESSED DIRECTOR: On the verge of a breakdown.
    - THE TECHNICAL DISASTER: Props failing, green screens falling down.
    
    TASK:
    1.  **Analyze**: Specific production details of {movie_title} (e.g. if it's Titanic, the water tank is leaking; if it's Matrix, the wires are tangled).
    2.  **Visual Lock**: Define the "Behind the Scenes" look (film equipment visible, coffee cups, half-built sets).
    3.  **The Cut**: Generate exactly {num_segments} Video Generation Prompts.
        -   Shows the actors *out of character* or the crew struggling.
        -   Funny, chaotic, gossip-filled.
        -   Each prompt is 8 seconds long.
    
    CONSTRAINT:
    -   Prompt 1 establishes the "Set".
    -   Include the "Visual Lock" descriptions in EVERY PROMPT.
    -   Make it feel like a "Found Footage" or "Fly on the Wall" documentary.
    
    OUTPUT FORMAT:
    ## The Gossip (Transcript)
    [Crew whispering about the disaster]

    ## Visual Lock
    [Set descriptions, visible boom mics, craft services tables]

    ## Shooting Schedule ({num_segments} Parts)
    1. **Prompt 1 (00-08s)**: [Full Prompt]
    ...
    {num_segments}. **Prompt {num_segments}**: [Full Prompt]
    """
    
    try:
        logging.info(f"   Filming Making-Of for '{movie_title}'...")
        response = model.generate_content(megaprompt)
        if response and response.text:
            return response.text
    except Exception as e:
        logging.error(f"Writers Room collapsed: {e}")
        return None
    return None

# --- VEO ACTION ---

class VeoDirector:
    def __init__(self, api_key, model_version=3, model_name=None):
        self.api_key = api_key
        # Priority: model_name > model_version
        if model_name:
            self.model_endpoint = model_name
        else:
            # Fallback legacy logic
            self.model_endpoint = "veo-2.0-generate-001" if model_version == 2 else "veo-3.0-generate-001"
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate_segment(self, prompt, context_uri=None, context_type="video", retry_safe=True):
        """
        Generates a video segment.
        context_uri: Optional URI (gs:// or https://) for the previous clip/image.
        context_type: "video" or "image".
        retry_safe: If True, attempts to soften prompt on safety trigger.
        """
        # --- GEMINI PATH (L-Tier) ---
        if "gemini" in self.model_endpoint.lower():
            logging.info(f"   🎥 Rolling Gemini (L-Tier): {self.model_endpoint}...")
            genai.configure(api_key=self.api_key)
            try:
                model = genai.GenerativeModel(self.model_endpoint)
                prompt_text = f"Generate a short video clip: {prompt}"
                
                # Gemini Video Generation (if supported via generate_content)
                # Currently Gemini Flash 2.5/Pro supports image generation, but video generation is experimental.
                # Assuming standard text-to-video prompt structure for this model.
                response = model.generate_content(prompt_text)
                
                # Extract Video URI from response (Experimental)
                # Usually returns a File API URI or inline data
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        # Safe check for video_metadata
                        if hasattr(part, 'video_metadata') and part.video_metadata:
                             logging.info(f"   🎥 Found Video Metadata: {part.video_metadata}")
                             # If there's a URI in here, extract it. 
                             # For now, we assume this path isn't fully live for public API.
                             pass
                        
                        if hasattr(part, 'text') and part.text:
                             logging.info(f"   📄 Text Response: {part.text[:200]}...")

                # Fallback: Just return text for debug if IT IS NOT A VIDEO MODEL
                # BUT the user claims L is "gemini-3-flash-preview" for video analysis/fast gen?
                # Actually, Gemini 3 Flash Preview might just be TEXT/IMAGE model for now?
                # "L: Light: Video analysis & fast metadata" <-- User description.
                # User's request implies IT GENERATES VIDEO.
                # If it doesn't, this will fail.
                # Let's assume it returns a URI in the text or metadata.
                
                # HACK: If this is purely for Metadata (Analysis), we return a dummy URI?
                # No, user asked to "call low quality/cost video models".
                
                # Let's try to assume it works like Imagen but for Video?
                # There is no public Gemini Video Gen API via generateContent yet.
                # However, if the user insists, we wrap it.
                
                logging.warning(f"   ⚠️ Gemini Video Gen is experimental. Response: {response.text[:100]}")
                return None 

            except Exception as e:
                logging.error(f"   Gemini Error: {e}")
                return None

        # --- VEO PATH (J/K Tier) ---
        # Robustness: Strip "models/" prefix if present, as base_url already includes it.
        clean_endpoint = self.model_endpoint.replace("models/", "")
        url = f"{self.base_url}/{clean_endpoint}:predictLongRunning?key={self.api_key}"
        headers = { "Content-Type": "application/json" }
        
        # 1. Try with Context (Base64 Image)
        if context_uri and os.path.exists(context_uri) and context_type == "image":
             try:
                 logging.info(f"   🎥 Rolling with Context (Base64): {context_uri}...")
                 with open(context_uri, "rb") as image_file:
                     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                 
                 payload = {
                   "instances": [{ 
                       "prompt": prompt, 
                       "image": { "bytesBase64Encoded": encoded_string, "mimeType": "image/jpeg" } 
                   }]
                 }
                 
                 res = requests.post(url, json=payload, headers=headers)
                 if res.status_code == 200:
                     return res.json().get('name')
                 else:
                     logging.warning(f"   ⚠️ Context Request Rejected ({res.status_code}): {res.text[:200]}... Retrying standalone...")
             except Exception as e:
                 logging.error(f"   ⚠️ Context Error: {e}")
                
        # 2. Standalone (No Context)
        payload = {
            "instances": [{ "prompt": prompt }]
        }
        logging.info("   🎥 Rolling Standalone...")
        
        try:
            res = requests.post(url, json=payload, headers=headers)
            
            # --- SAFETY CHECK & RETRY ---
            if res.status_code == 400 and "policy" in res.text.lower():
                logging.warning(f"   ⚠️ Safety Trigger: {res.text[:100]}")
                if retry_safe:
                    logging.info("   🛡️ Initiating Safety Protocol: Softening Prompt...")
                    try:
                        cleaner = sanitizer.Sanitizer(api_key=self.api_key)
                        safe_prompt = cleaner.soften_prompt(prompt)
                        
                        if safe_prompt != prompt:
                            logging.info("   🔄 Retrying with Softened Prompt...")
                            return self.generate_segment(safe_prompt, context_uri, context_type, retry_safe=False)
                    
                    except Exception as e_safe:
                        logging.error(f"   ❌ Safety Cleanup Failed: {e_safe}")

            if res.status_code != 200:
                logging.error(f"Veo Request Failed ({res.status_code}): {res.text}")
                return None
            
            data = res.json()
            op_name = data.get('name')
            if not op_name:
                logging.error(f"No operation name returned: {data}")
                return None
                
            return op_name

        except Exception as e:
            logging.error(f"Director Error: {e}")
            return None

    def wait_for_lro(self, op_name):
        """Polls for completion."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={self.api_key}"
        logging.info(f"   > Action! Polling {op_name}...")
        
        start_t = time.time()
        while time.time() - start_t < 600: # 10m timeout
            try:
                res = requests.get(url)
                data = res.json()
                
                if 'done' in data and data['done']:
                    if 'error' in data:
                        logging.error(f"   x Cut! Error: {data['error']}")
                        return None
                    
                    # Success! Extract URI
                    # Structure: response -> result -> videos -> [0] -> uri
                    logging.info("   > Cut! (Success)")
                    try:
                        # Inspect the 'response' field structure
                        # It is generic 'Any', usually a dict
                        result = data.get('response')
                        if not result: 
                            # Sometimes it's metadata?
                            return "UNKNOWN_URI"
                        
                        # Find the video URI
                        # This is a guess based on standard Google LRO patterns
                        # We return the whole result blob to parse later
                        return result
                    except:
                        return data
                    
                time.sleep(10)
                print(".", end="", flush=True)
                
            except Exception as e:
                logging.warning(f"Polling glitch: {e}")
                time.sleep(5)
                
        logging.error("   x Cut! Timeout.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Action! The Veo Director")
    parser.add_argument("--cut", action="store_true", help="Actually generate video using the Writers Room output.")
    parser.add_argument("--genre", type=str, default="movies", choices=["movies", "tech", "studio"], help="Writers Room Genre (default: movies)")
    parser.add_argument("--seg", type=int, default=12, help="Number of segments for Movies mode (default: 12)")
    parser.add_argument("--vm", type=str, default="J", help="Video Model Tier (L=Light, J=Just, K=Killer). Default: J")
    # Legacy args kept for compatibility but --vm takes precedence if set (logic below)
    parser.add_argument("--v", type=int, default=3, help="Legacy: Veo Model Version (2 or 3)")
    parser.add_argument("--veo", type=str, default=None, help="Legacy: Specific Model Name Override")
    args = parser.parse_args()

    # Resolve Model
    if args.veo: 
        # Manual override
        target_model = args.veo
    else:
        # Use LJK Definition
        target_model = definitions.get_video_model(args.vm)
        print(f"🎥 Video Model Selected: {args.vm} -> {target_model}")

    keys = load_action_keys()
    if not keys: 
        print("❌ No API Keys found in env_vars.yaml")
        return
    
    ensure_directories()

    # 1. Run Writers Room
    print(f"📺 Writers Room: Convening ({args.genre})...")
    
    script_text = None
    if args.genre == "movies":
        script_text = generate_movie_script(keys[0], args.seg)
    elif args.genre == "studio":
        script_text = generate_studio_script(keys[0], args.seg)
    else:
        script_text = generate_tech_script(keys[0])

    if not script_text:
        print("❌ Writers Room failed to produce a script.")
        return
        
    print("\n--- SCRIPT LOCKED ---\n")
    print(script_text)
    print("\n---------------------\n")
    
    # Save Log
    ts = int(time.time())
    script_filename = os.path.join(DIR_SCRIPTS, f"action_script_{ts}.md")
    with open(script_filename, "w") as f:
        f.write(script_text)

    # 2. Extract Prompts
    prompts = []
    # Regex: Look for "Prompt <digits>" followed by stuff and a colon.
    # We use re.search to ignore leading "1. **" or "**"
    # Matches:
    #   1. **Prompt 1 (00-08s)**: Content
    #   **Prompt 1**: Content
    pattern = r"(?i)prompt\s+\d+.*?:+(.*)"
    
    for line in script_text.split('\n'):
        line = line.strip()
        if not line: continue
        
        m = re.search(pattern, line)
        if m:
            raw_content = m.group(1).strip()
            # Clean leading/trailing markdown bold stars
            clean_content = raw_content.lstrip('*').strip()
            prompts.append(clean_content)
            
    if not prompts:
        print("⚠️ No prompts extracted. Dumping input text for debug:")
        print(script_text[:500] + "...")
        print("Check regex against above format.")
        return

    print(f"🎬 Parsed {len(prompts)} distinct shots.")

    # 3. Action! (If --cut)
    if args.cut:
        print(f"\n🎥 ACTION! Sending to Veo {args.v}...")
        
        last_clip_uri = None
        last_context_type = "video"
        results_log = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n🎬 SHOT {i+1}/{len(prompts)}")
            print(f"   Prompt: {prompt[:50]}...")
            
            # RETRY LOGIC: Try keys until one works or we run out
            op_name = None
            success_key_index = -1
            
            # We start trying from a random key to balance load
            # Cycle through all keys starting from there
            start_key_idx = random.randint(0, len(keys) - 1)
            
            for k_offset in range(len(keys)):
                k_idx = (start_key_idx + k_offset) % len(keys)
                current_key = keys[k_idx]
                
                # Check for "burned" keys? (Optional, but let's just try)
                
                director = VeoDirector(current_key, model_version=args.v, model_name=target_model)
                print(f"   [Key #{k_idx+1}] Rolling...")
                
                op_name = director.generate_segment(prompt, context_uri=last_clip_uri, context_type=last_context_type)
                
                if op_name:
                    success_key_index = k_idx
                    break # Success!
                else:
                    print(f"   ⚠️ Key #{k_idx+1} Failed. Switching...")
            
            if not op_name:
                print("   ❌ All keys failed for this shot. Aborting chain.")
                break
                
            # Wait (Using the successful director/key)
            result = director.wait_for_lro(op_name)
            
            if result:
                # Log success
                results_log.append({
                    "shot": i+1,
                    "prompt": prompt,
                    "result": result,
                    "key_used": success_key_index + 1
                })
                
                # Extract URI logic ...
                # Actual Structure observed:
                # {
                #   'generateVideoResponse': {
                #       'generatedSamples': [
                #           {'video': {'uri': 'https://...'}}
                #       ]
                #   }
                # }
                # --- SAFETY RETRY (RAI FILTERS) ---
                # Check for Child Safety / RAI triggers in valid response
                rai_filtered = False
                if 'generateVideoResponse' in result:
                    gvr = result['generateVideoResponse']
                    if 'raiMediaFilteredReasons' in gvr and gvr['raiMediaFilteredReasons']:
                        reasons = gvr['raiMediaFilteredReasons']
                        logging.warning(f"   🛡️ RAI FILTER TRIGGERED: {reasons}")
                        rai_filtered = True
                
                if rai_filtered and last_clip_uri:
                    logging.warning("   🔄 Retrying Shot as TEXT-ONLY (Dropping Context) to bypass filter...")
                    # Retry logic using the SAME key (success_key_index)
                    retry_director = VeoDirector(keys[success_key_index], model_version=args.v, model_name=target_model)
                    
                    # Call with NO context
                    retry_op = retry_director.generate_segment(prompt, context_uri=None, context_type="video")
                    if retry_op:
                        logging.info(f"   🔄 Retry Rolling (Text-Only)...")
                        retry_result = retry_director.wait_for_lro(retry_op)
                        if retry_result:
                            logging.info("   ✅ Retry Successful!")
                            result = retry_result # Overwrite result
                            # Clear failure flag if successful
                            rai_filtered = False
                        else:
                            logging.error("   ❌ Retry Failed.")
                    else:
                         logging.error("   ❌ Retry Failed to Start.")

                try:
                    # Generic traversal to find 'uri'
                    video_uri = None
                    if 'generateVideoResponse' in result:
                         samples = result['generateVideoResponse'].get('generatedSamples')
                         if samples:
                             video_uri = samples[0]['video']['uri']
                         else:
                             logging.warning(f"   ⚠️ No samples in generateVideoResponse. Full: {json.dumps(result, indent=2)}")
                    elif 'videos' in result:
                        video_uri = result['videos'][0]['uri']
                    elif 'video' in result:
                         video_uri = result['video']['uri']
                    
                    if not video_uri:
                        raise ValueError("Could not locate 'uri' in response.")

                    last_clip_uri = video_uri
                    print(f"   ✅ Shot {i+1} Wrapped. URI: {video_uri}")
                    
                    # DOWNLOAD
                    # DOWNLOAD
                    local_filename = os.path.join(DIR_PARTS, f"shot_{i+1:02d}_{ts}.mp4")
                    download_video(video_uri, local_filename, keys[success_key_index])
                    results_log[-1]['local_file'] = local_filename

                    # UPLOAD CONTEXT (Last Frame)
                    if i < len(prompts) - 1:
                        # Extract last frame
                        last_frame = extract_last_frame(local_filename)
                        if last_frame:
                            # --- SANITIZATION STEP ---
                            try:
                                api_key = keys[success_key_index]
                                cleaner = sanitizer.Sanitizer(api_key=api_key)
                                last_frame = cleaner.wash_image(last_frame)
                            except Exception as e:
                                logging.warning(f"   ⚠️ Sanitization Skipped: {e}")
                            # -------------------------

                            # Skip upload, use local path for Base64
                            last_clip_uri = last_frame
                            last_context_type = "image"

                    
                except Exception as e:
                    print(f"   ⚠️ Could not extract URI: {e}")
                    print(f"   Debug Data: {json.dumps(result, indent=2)}")

            else:
                print("   ❌ Shot Failed (LRO Error).")
                break
                
            # Cooldown
            if i < len(prompts) - 1:
                delay = 32 if len(keys) == 1 else 5 
                print(f"   ⏳ Cooling down for {delay}s...")
                time.sleep(delay)
                
            # Cooldown (Rate Limit: 2 RPM per Key)
            # If we rotated keys, we might not need to sleep as long?
            # But to be safe, we sleep anyway unless we have >8 keys.
            if i < len(prompts) - 1:
                delay = 32 if len(keys) == 1 else 15
                print(f"   ⏳ Cooling down for {delay}s...")
                time.sleep(delay) 
            
        # Log final report
        # Log final report
        report_filename = os.path.join(DIR_REPORTS, f"production_report_{ts}.json")
        with open(report_filename, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"\n✅ Production Wrap! Report saved to {report_filename}")
        
        # STITCH
        print("\n🧵 Stitching Dailies...")
        stitch_videos(results_log, os.path.join(DIR_FINAL, f"FINAL_CUT_{ts}.mp4"))

def download_video(uri, local_path, api_key):
    """Downloads video from URI using requests with API Key authentication."""
    print(f"   ⬇️ Downloading to {local_path}...")
    
    try:
        if uri.startswith("gs://"):
            cmd = f"gcloud storage cp {uri} {local_path}"
            os.system(cmd)
            return

        # Prepare HTTP request
        # If it's a Gemini File API URL, it usually needs the key.
        # Check if key is already in query param?
        
        headers = {}
        params = {}
        
        if "generativelanguage.googleapis.com" in uri:
            params['key'] = api_key
            
        r = requests.get(uri, params=params, stream=True)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
            print("      ✓ Saved.")
        else:
            print(f"      ❌ Download Failed ({r.status_code}): {r.text[:100]}")
            
    except Exception as e:
        print(f"      ❌ Download Error: {e}")

def stitch_videos(log, output_filename):
    """Stitches mp4s using FFmpeg concat."""
    # Create unique list file in componentparts to avoid collisions
    ts = int(time.time())
    list_file = os.path.join(DIR_PARTS, f"stitch_list_{ts}.txt")
    valid_files = []
    
    with open(list_file, 'w') as f:
        for entry in log:
            if 'local_file' in entry and os.path.exists(entry['local_file']):
                abs_path = os.path.abspath(entry['local_file'])
                f.write(f"file '{abs_path}'\n")
                valid_files.append(abs_path)
    
    if not valid_files:
        print("   ❌ No videos to stitch.")
        return

    print(f"   🧵 Combining {len(valid_files)} clips...")
    # ffmpeg concat
    # -safe 0 to allow relative paths
    cmd = f"ffmpeg -f concat -safe 0 -i {list_file} -c copy {output_filename} -y"
    
    ret = os.system(cmd)
    if ret == 0:
        print(f"   ✅ stitched: {output_filename}")
        # Cleanup?
        # os.remove(list_file)
    else:
        print("   ❌ FFmpeg failed.")

def upload_to_gcs(local_path, destination_blob_name=None):
    """Uploads a file to the context bucket and returns the gs:// URI."""
    bucket_name = "0i0-brain"
    prefix = "fmv_context"
    
    if not destination_blob_name:
        destination_blob_name = os.path.basename(local_path)
        
    gcs_uri = f"gs://{bucket_name}/{prefix}/{destination_blob_name}"
    
    print(f"   ☁️ Uploading context to {gcs_uri}...")
    
    # Use gcloud storage cp
    cmd = f"gcloud storage cp {local_path} {gcs_uri}"
    ret = os.system(cmd)
    
    if ret == 0:
        return gcs_uri
    else:
        print("   ⚠️ Context Upload Failed.")
        return None

def extract_last_frame(video_path):
    """Extracts the last frame of a video as a JPG."""
    if not os.path.exists(video_path):
        return None
        
    output_img = video_path.replace(".mp4", "_last.jpg")
    print(f"   🖼️ Extracting last frame to {output_img}...")
    
    # ffmpeg: seek to last second, grab last frame
    cmd = f"ffmpeg -sseof -1 -i {video_path} -update 1 -q:v 2 {output_img} -y >/dev/null 2>&1"
    
    ret = os.system(cmd)
    if ret == 0 and os.path.exists(output_img):
        return output_img
    else:
        print("   ⚠️ Frame Extraction Failed.")
        return None

def upload_to_gemini(path, api_key):
    """Uploads a file to Gemini File API."""
    print(f"   ☁️ Uploading to Gemini: {path}...")
    try:
        genai.configure(api_key=api_key)
        f = genai.upload_file(path)
        print(f"      ✓ Uploaded: {f.name} ({f.uri})")
        return f.uri
    except Exception as e:
        print(f"      ❌ Upload Failed: {e}")
        return None

if __name__ == "__main__":
    main()
