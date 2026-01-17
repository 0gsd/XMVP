
# --- GAHD PODCAST PRODUCER (Season 2) ---

class GAHDProducer:
    def __init__(self, output_dir=None, text_engine=None):
        self.output_dir = output_dir if output_dir else os.path.join(OUTPUT_DIR, "gahd-scripts-vids")
        os.makedirs(self.output_dir, exist_ok=True)
        self.text_engine = text_engine if text_engine else TextEngine()
        self.memory = self.load_memory()
        
    def load_memory(self):
        """Scans output dir for past episodes to build exclusion lists."""
        print("[Memory] Scanning Archive for Past Bits...")
        memory = {
            "used_jokes": [],
            "past_pitches": [],
            "actor_history": [],
            "chaos_seeds_used": []
        }
        
        # Scan JSONs
        files = glob.glob(os.path.join(self.output_dir, "*.json"))
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    
                # Extract meta
                if "meta" in data:
                    meta = data["meta"]
                    if "seeds" in meta: memory["chaos_seeds_used"].extend(meta["seeds"])
                    
                # Extract jokes/pitches (Naive extract from text if structured, otherwise just store summary)
                # Ideally we ask Gemini to summarize them, but for now we just log file existence
                # Real implementation: Read "meta" fields if we save them there.
                
            except Exception as e:
                print(f"[-] Memory Load Error ({fpath}): {e}")
                
        print(f"   [+] Loaded {len(files)} past episodes.")
        return memory

    def cast_actors(self):
        """Generates 4-person cast (2M, 2F) using Gemini."""
        print("[Casting] Summoning Ensemble...")
        
        prompt = (
            "Generate a cast of 4 distinct 'Character Actors' (2 Male, 2 Female) active 1975-2005.\n"
            "CRITERIA: Recognizable but not A-List. Distinct voices/types.\n"
            "OUTPUT JSON: "
            "[{ 'name': 'Name', 'gender': 'Male/Female', 'persona': 'Vocal/Personality description', 'known_for': 'Role' }, ...]"
        )
        
        try:
            raw = self.text_engine.generate(prompt, json_schema=True)
            # Json cleanup
            if "```json" in raw: raw = raw.split("```json")[1].split("```")[0]
            cast_list = json.loads(raw)
            
            # Assign voices
            final_cast = {}
            # Voice Slots
            slots = [
                {'gender': 'Male', 'voice': 'en-US-Journey-D', 'pitch': 1},
                {'gender': 'Female', 'voice': 'en-US-Journey-F', 'pitch': -2},
                {'gender': 'Male', 'voice': 'en-US-Journey-D', 'pitch': -2},
                {'gender': 'Female', 'voice': 'en-US-Journey-F', 'pitch': 1}
            ]
            
            # Naive match
            males = [c for c in cast_list if c['gender'] == 'Male'][:2]
            females = [c for c in cast_list if c['gender'] == 'Female'][:2]
            
            # combine
            ensemble = males + females
            random.shuffle(ensemble)
            
            for i, actor in enumerate(ensemble):
                # Find matching slot
                slot = next((s for s in slots if s['gender'] == actor['gender']), None)
                if not slot: slot = slots[0] # Fallback
                slots.remove(slot)
                
                final_cast[actor['name']] = {
                    "name": actor['name'],
                    "persona": actor['persona'],
                    "voice": slot['voice'],
                    "pitch": slot['pitch'],
                    "gender": actor['gender']
                }
                print(f"   + {actor['name']} ({actor['gender']}) as {slot['voice']} (p{slot['pitch']})")
                
            return final_cast
            
        except Exception as e:
            print(f"[-] Casting Failed: {e}")
            return None

    def produce_show(self, duration_override=None):
        """Runs the 4-Phase Show."""
        
        # 1. Cast
        cast = self.cast_actors()
        if not cast: return
        
        cast_desc = "\n".join([f"{n}: {d['persona']}" for n, d in cast.items()])
        
        # 2. Seeds
        from vision_producer import get_chaos_seed
        seeds = [get_chaos_seed(), get_chaos_seed()]
        print(f"[Seeds] {seeds}")
        
        # 3. Setup Session
        timestamp = int(time.time())
        episode_dir = os.path.join(self.output_dir, f"ep_{timestamp}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 4. State Machine Loop
        # Phase 1: Anecdotes (0-16m)
        # Phase 2: Seeds Reveal (16-24m)
        # Phase 3: Pitch (24-54m)
        # Phase 4: Summary (54-End)
        
        print("[Action] Starting GAHD Recording...")
        
        # Initialize History
        history = [] # List of {speaker, text}
        segments = [] # List of Segment objects
        
        # System Prompt
        sys_prompt = (
            f"You are the Showrunner of 'Golden Age of Hollywood Dreams' (GAHD).\n"
            f"CAST:\n{cast_desc}\n\n"
            f"STYLE: Witty, insider-y, cynical but loving. No cliches.\n"
            f"Output JSON: {{ 'speaker': 'Name', 'text': 'Dialogue', 'visual_prompt': 'Image description' }}"
        )
        
        # SIMULATION LOOP (Simplified for MVP Integration)
        # We need to generate LINES -> AUDIO -> IMAGE immediately to manage context?
        # Or batch? The user asked for "line by line... stitching at end".
        
        target_lines = 50 if duration_override else 300 # Approx 50 mins? 
        # Actually, let's just loop until logical conclusion or max lines.
        
        current_phase = "Anecdotes"
        
        for i in range(target_lines):
            # Determine Phase
            # For MVP, just do a short sequence
            if i < 4: current_phase = "Anecdotes"
            elif i == 4: current_phase = "Seeds Reveal"
            else: current_phase = "The Pitch"
            
            prompt = (
                f"Phase: {current_phase}.\n"
                f"Seeds: {seeds}\n"
                f"History: {history[-10:]}\n"
                f"Generate next turn."
            )
            
            # Generate Text
            try:
                raw = self.text_engine.generate(f"{sys_prompt}\n\n{prompt}", json_schema=True)
                if "```json" in raw: raw = raw.split("```json")[1].split("```")[0]
                turn = json.loads(raw)
                
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                vis = turn.get('visual_prompt', '')
                
                # Validation
                if speaker not in cast: speaker = random.choice(list(cast.keys()))
                
                print(f"   [{i}] {speaker}: {text[:50]}...")
                
                # Create Segment
                seg = Segment(speaker, text)
                
                # Audio
                c_data = cast[speaker]
                wav_path = os.path.join(episode_dir, f"line_{i:04d}.wav")
                
                gen_path = generate_audio_asset(
                    text, wav_path, 
                    voice_name=c_data['voice'], 
                    pitch=c_data['pitch'],
                    mode="cloud",
                    project_id=get_project_id()
                )
                
                if gen_path:
                    seg.audio_path = gen_path
                    seg.duration = get_audio_duration(gen_path)
                else:
                    # Fallback silence
                    create_silence(2.0, wav_path)
                    seg.audio_path = wav_path
                    seg.duration = 2.0
                    
                # Image
                img_path = os.path.join(episode_dir, f"line_{i:04d}.png")
                # Use class image generator? or local helper
                # Reuse generate_image function from main scope
                # We need to construct a prompt
                img_prompt = f"{vis} --aspect_ratio 1:1"
                generate_image(img_prompt, img_path) # Assumes generate_image is available (it is in py)
                seg.image_path = img_path
                
                segments.append(seg)
                history.append({'speaker': speaker, 'text': text})
                
            except Exception as e:
                print(f"   [-] Turn Failed: {e}")
                time.sleep(1)
        
        # Stitch
        print("[Post] Stitching Episode...")
        # ... Reuse stitching logic ...
        # (For implementation, I will call a helper function similar to process_triplet stitching)
        self.stitch_segments(segments, episode_dir, "gahd_episode.mp4")

    def stitch_segments(self, segments, temp_dir, output_filename):
        # Implementation of stitching logic
        audio_list = os.path.join(temp_dir, "audio_list.txt")
        video_list = os.path.join(temp_dir, "video_list.txt")
        output_mp4 = os.path.join(self.output_dir, output_filename)
        
        with open(audio_list, 'w') as fa, open(video_list, 'w') as fv:
            for seg in segments:
                if seg.audio_path and seg.image_path:
                    fa.write(f"file '{seg.audio_path}'\n")
                    fv.write(f"file '{seg.image_path}'\n")
                    fv.write(f"duration {seg.duration:.4f}\n")

        full_audio = os.path.join(temp_dir, "full_audio.wav")
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', audio_list, '-c', 'copy', full_audio, '-y', '-loglevel', 'error'], check=True)
        
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', video_list,
            '-i', full_audio,
            '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k', '-shortest', output_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[+] Produced: {output_mp4}")
