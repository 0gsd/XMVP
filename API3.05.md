# XMVP API Reference v3.05
## Complete Command-Line Interface Documentation

---

# Table of Contents

1. [Creative Engines (Entry Points)](#creative-engines)
   - [cartoon_producer.py](#cartoon_producerpy)
   - [content_producer.py](#content_producerpy)
   - [post_production.py](#post_productionpy)
   - [xmvp_converter.py](#xmvp_converterpy)
2. [Standalone Visualizers](#standalone-visualizers)
   - [ansi_visualizer.py](#ansi_visualizerpy)
   - [unicode_visualizer.py](#unicode_visualizerpy)
3. [Pipeline Modules (Internal)](#pipeline-modules)
   - [vision_producer.py](#vision_producerpy)
   - [stub_reification.py](#stub_reificationpy)
   - [writers_room.py](#writers_roompy)
   - [portion_control.py](#portion_controlpy)
   - [dispatch_director.py](#dispatch_directorpy)
   - [dispatch_clip_video.py](#dispatch_clip_videopy)
4. [Audio & Speech Modules](#audio--speech-modules)
   - [foley_talk.py](#foley_talkpy)
   - [thax_audio.py](#thax_audiopy)
   - [sfx_bridge.py](#sfx_bridgepy)
5. [Bridge Modules (Local Inference)](#bridge-modules)
   - [flux_bridge.py](#flux_bridgepy)
   - [kokoro_bridge.py](#kokoro_bridgepy)
   - [hunyuan_foley_bridge.py](#hunyuan_foley_bridgepy)
6. [Core Libraries](#core-libraries)
   - [text_engine.py](#text_enginepy)
   - [truth_safety.py](#truth_safetypy)
   - [definitions.py](#definitionspy)
   - [mvp_shared.py](#mvp_sharedpy)
7. [Utility & Management Modules](#utility--management-modules)
   - [model_scout.py](#model_scoutpy)
   - [populate_models_xmvp.py](#populate_models_xmvppy)
   - [sassprilla_carbonator.py](#sassprilla_carbonatorpy)
   - [dialogue_critic.py](#dialogue_criticpy)
   - [nicotime_index.py](#nicotime_indexpy)
   - [train_mll.py](#train_mllpy)
   - [prep_movie_assets.py](#prep_movie_assetspy)
   - [convert_voices.py](#convert_voicespy)
   - [count_lines.py](#count_linespy)
   - [test_gen_capabilities.py](#test_gen_capabilitiespy)
8. [Data Models & Schemas](#data-models--schemas)
9. [Configuration Files](#configuration-files)
10. [Training Data & Adapters](#training-data--adapters)

---

# Creative Engines

## cartoon_producer.py

**The Animator** — Primary creative engine for frame-by-frame animation, music video syncing, video restyling, and LLM-directed visual production. 3,221 lines.

Supports multiple production modes dispatched via `--vpform`: creative-agency (prompt → story → frames), music-video (beat-synced narrative to audio), music-visualizer (procedural stem-reactive animation), cartoon-video (frame-by-frame restyling of existing video), clip-video (beat-matched montage), and full-movie (long-form feature animatic).

### Usage
```bash
python3 cartoon_producer.py [OPTIONS]
# Or with positional alias:
python3 cartoon_producer.py music-video --mu track.mp3 --prompt "Neon dreams"
```

### Options

#### Core Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vpform` | str | `creative-agency` | VP Form (see VP Forms Reference) |
| `--prompt` | str | `None` | Creative prompt. Short title-case prompts are auto-carbonated by SASSPRILLA. |
| `--style` | str | `"high resolution 4K UHD video"` | Visual style definition passed to the image generator |
| `--slength` | float | `60.0` | Target length in seconds (when no music provided) |
| `--fps` | int | `4` | Output FPS or expansion factor |
| `--cs` | int | `0` | Chaos Seeds level (0–3). Injects random Wikipedia concepts. |
| `--bpm` | float | `None` | Manual BPM override (bypasses detection) |
| `--pg` | flag | `False` | Enable PG Mode |

#### Source Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mu` | str | `None` | Path to music/audio file for sync. In cartoon-video mode, the input video. |
| `--xb` | str | `None` | Path to XMVP XML manifest for re-rendering |
| `--tf` | Path | (default path) | Transcript folder (legacy FBF source) |
| `--vf` | Path | `/Volumes/XMVPX/fmv_corpus` | Video folder (corpus for clip-video) |
| `--f` | str | `None` | Source folder for clip-video mode |
| `--project` | str | `None` | Specific project name to process |

#### Processing Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--local` | flag | `False` | Local mode (Gemma + Flux, no API costs, no content filters) |
| `--cloud` | flag | `False` | Force cloud mode (Gemini 2.x) |
| `--delay` | float | `5.0` | Delay between API requests in seconds |
| `--limit` | int | `0` | Limit number of frames (0 = unlimited) |
| `--smin` | float | `0.0` | Minimum duration filter in seconds |
| `--smax` | float | `None` | Maximum duration filter in seconds |
| `--shuffle` | flag | `False` | Shuffle projects before processing |
| `--vspeed` | float | `8.0` | Visualizer speed (FPS) for music-agency mode |
| `--fc` | flag | `False` | Enable Frame & Canvas (Code Painter) mode |
| `--retcon` | flag | `False` | Retcon mode: rewrite script/beats of input XML |
| `--wan` | flag | `False` | Use Wan 2.1 keyframe animation (local only) |
| `--kid` | int | `512` | Keyframe init dimension (higher = better composition) |
| `--w` | int | `None` | Override width |
| `--h` | int | `None` | Override height |
| `--strength` | int | `50` | Img2Img noise strength 1–99. Lower = more frame coherence. 10–30: stable, 30–50: balanced, 50–80: creative. |
| `--fsync` | float | `1.0` | FPS sync multiplier (0.1–6.0) |

### Key Functions
| Function | Description |
|----------|-------------|
| `plan_shots(total_frames, fps, bpm)` | Generates beat-synced shot list with 1–6s durations |
| `run_wan_keyframe_anim(args, prompts, fps, out, dur, bpm)` | Wan 2.1 keyframe animation pipeline |
| `analyze_audio(audio_path, fsync)` | BPM detection, beat tracking, section analysis |
| `analyze_audio_profile(audio_path, duration)` | Loudness/spectral sonic map per second |
| `generate_frame_universal(...)` | Unified frame generation (cloud or local, txt2img or img2img) |
| `process_project(project_dir, vf_dir, key_cycle, args, ...)` | Main project processing loop (mode dispatch) |
| `run_ascii_forge(input_video, output_video)` | ASCII art post-processing overlay |
| `blend_videos(base, overlay, output, opacity)` | FFmpeg alpha-blended video compositing |
| `load_full_xmvp(path)` | Load all sections from XMVP XML |
| `scan_projects(tf_dir)` | Scan transcript directory for project folders |

### Examples
```bash
# Creative agency (default)
python3 cartoon_producer.py --prompt "A melancholy astronaut" --style "Pixel Art"

# Music video with beat sync
python3 cartoon_producer.py mv --mu song.mp3 --prompt "Cyberpunk chase"

# Procedural stem-reactive visualizer
python3 cartoon_producer.py viz --mu ambient.wav

# Frame-by-frame video restyling
python3 cartoon_producer.py cartoon-video --mu input.mp4 --style "Oil painting" --strength 30

# Full-movie animatic (local, 10 minutes)
python3 cartoon_producer.py movie --prompt "The Odyssey" --local --slength 600

# Wan keyframe animation
python3 cartoon_producer.py --prompt "Dancing in rain" --wan --local --kid 768

# Beat-matched clip montage
python3 cartoon_producer.py --vpform clip-video --mu track.mp3 --f /path/to/clips/
```

---

## content_producer.py

**The Podcast Factory** — Unified generator for podcast, improv, spoken word, and audio-visual content. 2,791 lines.

Supports generative improv (multi-character scripts from scratch), Thax Douglas spoken word, Element 47 audio plays (from Fountain scripts), full-movie slideshow rendering, audio-only play generation, and audio-to-video workflows via SkyReels.

### Usage
```bash
python3 content_producer.py [OPTIONS]
# Or with positional alias:
python3 content_producer.py e47 --xb script.fountain --local
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vpform` | str | `24-podcast` | VP Form (see VP Forms Reference) |
| `--local` | flag | `False` | Use local engines (Gemma + Flux + Kokoro) |
| `--geminiapi` | flag | `False` | Force cloud Gemini API for text (overrides local default) |
| `--slength` | float | `0.0` | Override duration in seconds |
| `--ep` | int | `None` | Episode number (format: SSE for Season S, Episode E) |
| `--location` | str | `None` | Override visual location |
| `--rvc` | flag | `False` | Enable RVC voice conversion |
| `--foley` | str | `"off"` | Enable generative foley (`on`/`off`) |
| `--fc` | flag | `False` | Code Painter mode |
| `--mll` | str | `"off"` | Enable movie-level LoRA training step (`on`/`off`) |
| `--w` | int | `512` | Image width |
| `--h` | int | `288` | Image height |
| `--band` | str | `None` | Band name (thax-douglas mode) |
| `--poem` | str | `None` | Poem text (thax-douglas mode) |
| `--xml` / `--xb` | str | `None` | Input XMVP XML or Fountain script path |
| `--mu` | str | `None` | Master audio input (audio-movie mode) |
| `--out` | str | `None` | Output directory override |
| `--seeds` | str | `None` | Explicit chaos seeds list |
| `--project` | str | `None` | Project name override |

### Content VP Forms

| Form | Aliases | Default Duration | Description |
|------|---------|------------------|-------------|
| `24-podcast` | `24`, `news` | 24 min (1440s) | 4-person improv comedy. Auto-generates cast, topics, and script. |
| `10-podcast` | `10`, `tech-news` | 10 min (600s) | Topical tech podcast |
| `route66-podcast` | `r66`, `route66` | 66 min (3960s) | 6-person road trip narrative |
| `gahd-podcast` | `gahd`, `god`, `history` | Variable | Great Moments in History radio drama |
| `thax-douglas` | `thax`, `td` | Variable | Spoken word poetry (RVC voice model included) |
| `element-47` | `e47`, `element47` | Variable | Audio play from Fountain script. 4-character cast (Burn, Drip, Cruise, Anchor). |
| `fullmovie-still` | `fms`, `slideshow` | Variable | XMVP XML → frame+audio slideshow |
| `black-box` | `bb`, `theater`, `stage`, `min` | Variable | Minimalist theater/black box mode |
| `audio-play` | `ap`, `audioplay`, `play` | Variable | Audio-only play (MP3 output) |
| `audio-movie` | `am`, `a2v`, `audiomovie` | Variable | Audio-to-video via SkyReels A2V |

### Key Functions
| Function | Description |
|----------|-------------|
| `run_improv_session(vpform, output_dir, text_engine, args)` | Main generative improv pipeline |
| `run_thax_douglas_session(band, poem, output_dir)` | Thax Douglas spoken word generation |
| `run_element47_production(manifest_path, output_dir, args)` | Element 47 audio play production |
| `run_fullmovie_still_mode(xml, output_dir, te, args)` | Slideshow rendering from XMVP XML |
| `run_audio_play_mode(xml, output_dir, args)` | Audio-only play generation |
| `run_audio_movie_mode(xml, audio, output_dir, args)` | SkyReels audio-to-video pipeline |
| `generate_image(prompt, output_path, ts, init_image, strength)` | Image generation (cloud or local) |
| `generate_dynamic_cast(text_engine, seeds)` | LLM-generated character cast |
| `generate_ensemble_cast(text_engine)` | Full ensemble cast generation |
| `generate_location_context(text_engine, args, seeds)` | Location/setting generation |
| `run_rvc_conversion(wav_path, character_name)` | RVC voice cloning post-process |
| `export_xmvp_manifest(output_dir, ...)` | Export production to XMVP XML |
| `stitch_assets(assets, temp_dir, output_mp4)` | FFmpeg asset stitching |

### Examples
```bash
# 24-minute improv (local text, cloud images)
python3 content_producer.py --vpform 24-podcast

# Historical radio drama
python3 content_producer.py gahd --ep 207 --local --location "The Colosseum at Dawn"

# Route 66 with voice cloning
python3 content_producer.py r66 --rvc --local --slength 3960 --ep 301

# Element 47 from Fountain script
python3 content_producer.py e47 --xb episode.fountain --local

# Full-movie slideshow
python3 content_producer.py fms --xb manifest.xml

# Audio-to-video
python3 content_producer.py am --xb manifest.xml --mu master_audio.wav --local
```

---

## post_production.py

**The Editor** — Upscaling, frame interpolation, retiming, audio stitching, and VDJ blend mode. 1,179 lines.

### Usage
```bash
python3 post_production.py [INPUT] [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input` | positional | — | Input video file or directory of frames |
| `--input` | str | `None` | Input (flag alternative to positional) |
| `--output` | str | `None` | Output directory |
| `-x` | int | `2` | Frame expansion factor (AI tweening) |
| `--scale` | float | `2.0` | Upscale factor |
| `--restyle` | str | `None` | Restyle mode (e.g. `ascii`) |
| `--local` | flag | `True` | Run locally (Flux img2img). Default. |
| `--cloud` | flag | `False` | Use Gemini for upscale/tween |
| `--more` | flag | `False` | Secondary pass (4x total interpolation/upscale) |
| `--mu` | str | `None` | Audio file for sync or VDJ mode |
| `--stitch-audio` | flag | `False` | Force-stitch frames to match audio duration |
| `--framerate` | float | `None` | Retime video to specific FPS |
| `--vvaudio` | flag | `False` | VDJ blend mode |
| `--bottomvideo` | str | `None` | VDJ: bottom layer (100% opacity) |
| `--topvideo` | str | `None` | VDJ: top layer (40% opacity) |

### Key Classes & Functions
| Name | Description |
|------|-------------|
| `Obsessionator` | Image detail enhancer (sharpening, local contrast) |
| `Obsessionator.upscale(image, output, scale, local, more)` | Flux img2img upscale pipeline |
| `FrameTweener` | AI-powered frame interpolation |
| `FrameTweener.generate_tween(img_a, img_b, output, ...)` | Generate intermediate frame between two keyframes |
| `process(args)` | Main processing pipeline |
| `stitch_videos(log, output_filename)` | Concatenate video segments via FFmpeg |
| `run_ascii_forge(input, output)` | ASCII art post-processing overlay |
| `run_vdj_blend(args, output_root)` | VDJ dual-layer video blending |
| `change_framerate(input, target_fps)` | Retime video to target FPS |

### Examples
```bash
# Upscale 2x
python3 post_production.py video.mp4 --scale 2.0

# Interpolate + upscale (4x smoother, 2x larger)
python3 post_production.py video.mp4 -x 2 --more

# Stitch frames to audio
python3 post_production.py /path/to/frames/ --mu audio.mp3 --stitch-audio

# VDJ blend
python3 post_production.py --vvaudio --bottomvideo base.mp4 --topvideo overlay.mp4 --mu mix.mp3
```

---

## xmvp_converter.py

**The Converter** — Converts text files, screenplays, and Fountain scripts into XMVP XML manifests. Also handles smart chunking, GemmaW director integration, and Element 47 Fountain parsing. 1,555 lines.

### Usage
```bash
python3 xmvp_converter.py INPUT_PATH [OPTIONS]
```

### Key Functions
| Function | Description |
|----------|-------------|
| `process_file(input_path, args)` | Main conversion pipeline |
| `smart_chunk_script(text, num_chunks)` | Scene-aware text chunking |
| `smart_chunk_ingest(content)` | Intelligent content segmentation |
| `parse_fountain(content)` | Standard Fountain screenplay parser |
| `parse_element47_fountain(content)` | Element 47-specific Fountain parser |
| `setup_gemma_director()` | Initialize GemmaW director model + adapters |
| `get_best_inspiration(engine, snippet)` | NICOTIME entity lookup for creative enrichment |
| `get_celebrity_anchor(gender)` | Generate celebrity reference anchor |
| `load_hj24()` | Load HJ24 beat template data |

---

# Standalone Visualizers

## ansi_visualizer.py

**ANSI Audio Visualizer** — 4-track Demucs stem splitter + procedural ASCII animation. 1,432 lines.

Splits audio into drums/bass/keys/other via Demucs, generates per-track colorized ASCII animations driven by loudness and spectral character, composites four layers with opacity blending, and muxes synced audio into a final MP4. No AI inference required — pure signal processing.

### Usage
```bash
python3 ansi_visualizer.py --mu AUDIO_PATH [OPTIONS]
```

### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mu` | str | required | Path to audio file |
| `--fps` | int | `24` | Output framerate |
| `--width` | int | `120` | Canvas width in characters |
| `--height` | int | `40` | Canvas height in characters |

### Character Palettes
Each stem uses a specialized character set ordered by visual density:
- **Drums**: ` ·.oO0@█▓▒░╳╬`
- **Bass**: ` ._~≈≋∼∽║▐█▓▒`
- **Keys**: ` .·°•◦○◎●♪♫♬★`
- **Other**: ` .:░▒▓╱╲╳◇◆▲△`

Additional palettes: Spiral, Wave, Rain, Star — selected dynamically based on audio characteristics.

---

## unicode_visualizer.py

**Unicode Audio Visualizer** — Extended version using 140K+ Unicode characters with themed character pools. 2,319 lines.

Same Demucs stem-splitting pipeline as `ansi_visualizer.py` with richer character repertoire including emoji, braille patterns, geometric shapes, CJK characters, and themed pools.

### Usage
```bash
python3 unicode_visualizer.py --mu AUDIO_PATH [OPTIONS]
```

### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mu` | str | required | Path to audio file |
| `--fps` | int | `24` | Output framerate |
| `--width` | int | `120` | Canvas width in characters |
| `--height` | int | `40` | Canvas height in characters |
| `--theme` | str | `classic` | Theme name (see below) or `random` for per-section randomization |

### Themes
| Theme | Description |
|-------|-------------|
| `classic` | Traditional ANSI characters |
| `matrix` | Matrix rain (katakana + kanji + digits) |
| `emoji` | Emoji-based visualization |
| `braille` | 8-dot braille patterns |
| `geometric` | Geometric shapes and symbols |
| `random` | Randomizes theme per audio section |

### Character Pool System
The `CharacterPool` class manages themed character sets with density gradients for intensity mapping. Each theme provides per-stem character sets (drums, bass, keys, other) plus special characters for transitions.

---

# Pipeline Modules

## vision_producer.py

**The Vision Producer** — Creates the "Bible" (CSSV: Constraints, Scenario, Situation, Vision) from a prompt and VP Form. 425 lines.

### Usage
```bash
python3 vision_producer.py --vpform FORM --prompt "concept" --out bible.json
```

### Key Functions
| Function | Description |
|----------|-------------|
| `get_default_vision(form_name, seg_count)` | Generate style/aesthetic/pacing vision for a VP Form |
| `get_chaos_seed()` | Fetch random Wikipedia summary for creative injection |
| `get_specific_seed(query)` | Fetch Wikipedia summary for a specific topic |
| `analyze_audio(audio_path)` | BPM + beat analysis for music-driven productions |
| `run_producer(vpform_name, prompt, slength, ...)` | Full Bible generation pipeline |

### Output: CSSV (Bible)
```json
{
  "constraints": {"width": 768, "height": 768, "fps": 24, "max_duration_sec": 60.0, ...},
  "scenario": "A 60.0-second creative-agency",
  "situation": "CONCEPT: ...",
  "vision": "STYLE: ... AESTHETIC: ... PACING: ..."
}
```

---

## stub_reification.py

**Story Synthesizer** — Expands a CSSV Bible into a full Story (title, synopsis, characters, theme). 182 lines.

### Usage
```bash
python3 stub_reification.py --bible bible.json --out story.json
```

### Key Functions
| Function | Description |
|----------|-------------|
| `synthesize_story(cssv)` | LLM-driven story expansion from Bible |
| `run_stub(bible_path, out_path, request)` | Full stub-to-story pipeline |

---

## writers_room.py

**The Writers Room** — Breaks a Story into timed Portions (scenes with dialogue, action, duration). 273 lines.

### Usage
```bash
python3 writers_room.py --bible bible.json --story story.json --out portions.json
```

### Key Functions
| Function | Description |
|----------|-------------|
| `break_story(story, cssv)` | Decompose story into scene portions |
| `run_writers(bible_path, story_path, out_path)` | Full writers room pipeline |

---

## portion_control.py

**Portion Control** — Converts Portions into a frame-accurate Manifest (segments with frame ranges and prompts). 163 lines.

### Usage
```bash
python3 portion_control.py --bible bible.json --portions portions.json --out manifest.json
```

### Key Functions
| Function | Description |
|----------|-------------|
| `run_portion(bible_path, portions_path, out_path, max_seg_dur)` | Calculate frame ranges and generate manifest |

---

## dispatch_director.py

**The Director** — Generates video/image assets per manifest segment. Supports Veo 3.1 (cloud), LTX-Video (local), Wan 2.1 (local), and SkyReels (local). 878 lines.

### Key Classes
| Class | Description |
|-------|-------------|
| `VeoDispatcher` | Cloud video generation via Veo 3.1 (fast or 4K) |
| `LTXDispatcher` | Local video generation via LTX-Video |
| `UnifiedDispatcher` | Unified dispatcher routing to Veo or LTX based on active profile |
| `WanDirector` | Wan 2.1 video generation with image conditioning |
| `SkyReelsDirector` | SkyReels audio-to-video generation |

### VeoDispatcher Methods
| Method | Description |
|--------|-------------|
| `generate_segment(prompt, context_uri, context_type, retry_safe)` | Generate video segment with automatic retry and safety refinement |
| `wait_for_lro(op_name)` | Poll long-running operation until complete |

### UnifiedDispatcher Methods
| Method | Description |
|--------|-------------|
| `generate(prompt, output_path, context_uri)` | Route to Veo or LTX based on backend config |

---

## dispatch_clip_video.py

**Clip Video Pipeline** — Beat-matched montage engine. Scans a folder of source videos, analyzes audio for intensity profile, and selects/trims clips to match beat timing. 355 lines.

### Key Functions
| Function | Description |
|----------|-------------|
| `run_clip_video_pipeline(args)` | Full clip montage pipeline |
| `scan_for_videos(folder_path)` | Discover video files in a directory |
| `select_clip_smart(source_video, duration, usage_tracker)` | Smart clip selection avoiding repeat segments |
| `analyze_audio_profile(audio_path, duration)` | Per-second intensity map from audio |

---

# Audio & Speech Modules

## foley_talk.py

**Audio Engine** — Cloud and local speech synthesis, foley generation, and audio mixing. 700 lines.

### Key Functions
| Function | Description |
|----------|-------------|
| `generate_audio_asset(text, output, voice, pitch, mode, ...)` | Unified TTS (cloud Journey or local Kokoro) |
| `synthesize_text_cloud(text, voice, output, project_id)` | Google Cloud TTS synthesis |
| `pitch_shift_file(input_file, semitones)` | FFmpeg pitch shifting |
| `compose_track(assets, duration, output)` | Compose timed audio assets into a single track |
| `mix_audio(video, foley, dialogue_files, output)` | Multi-layer audio mixing |
| `generate_cloud_dialogue(script, output_dir, project_id)` | Generate full dialogue script via Cloud TTS |
| `generate_hunyuan_batch(manifest, output_dir)` | Batch foley generation via Hunyuan |
| `assign_kokoro_voice_deterministic(character_name)` | Deterministic voice assignment for Kokoro |

---

## thax_audio.py

**Thax Douglas Engine** — RVC-based voice generation for the Thax Douglas spoken word mode. Loads the included voice model and generates speech with Thax's voice characteristics. 188 lines.

### Key Functions
| Function | Description |
|----------|-------------|
| `get_thax_engine()` | Initialize and return the Thax audio engine |
| `ThaxEngine.generate(text, output_path)` | Generate Thax-voiced audio |

---

## sfx_bridge.py

**SFX Generator** — Bridge for sound effects and music generation. Tries AudioCraft (MusicGen/AudioGen) first, falls back to Stable Audio Open via Diffusers. 212 lines.

### SFXGenerator Class
| Method | Description |
|--------|-------------|
| `generate_sfx(prompt, duration, output_path)` | Generate sound effect from text description |
| `generate_music(prompt, duration, output_path)` | Generate music from text description |

---

# Bridge Modules

## flux_bridge.py

**Flux Image Generator** — Local inference bridge for Flux.1-schnell and Flux 2 Klein 9B on Apple Silicon (MPS). Supports txt2img, img2img, LoRA loading, and HuggingFace Inference Endpoint fallback. 924 lines.

### FluxBridge Class
| Method | Description |
|--------|-------------|
| `__init__(model_path, device)` | Initialize with model path (default MPS) |
| `load_pipeline(model_path)` | Load Flux pipeline with MPS memory optimizations |
| `generate(prompt, width, height, steps, seed, guidance, image, strength)` | Text-to-image generation |
| `load_img2img()` | Load img2img pipeline variant |
| `generate_img2img(prompt, image, strength, ...)` | Image-to-image generation |
| `load_lora(lora_path, adapter_name, scale)` | Load LoRA adapter weights |
| `unload()` | Free GPU memory |

### Standalone Function
| Function | Description |
|----------|-------------|
| `get_flux_bridge(path)` | Singleton factory for FluxBridge |
| `generate_via_hf_endpoint(prompt, ..., endpoint_url)` | Cloud fallback via HF Inference API |

---

## kokoro_bridge.py

**Kokoro TTS** — Local text-to-speech via Kokoro ONNX model. 156 lines.

### KokoroBridge Class
| Method | Description |
|--------|-------------|
| `__init__(model_path, voices_path)` | Initialize with model and voices paths |
| `load()` | Load ONNX model |
| `generate(text, output_path, voice_name, speed)` | Generate speech audio |
| `get_voice_list()` | List available voice names |

### Factory
| Function | Description |
|----------|-------------|
| `get_kokoro_bridge(model_path)` | Singleton factory |

---

## hunyuan_foley_bridge.py

**Hunyuan Foley** — Local foley sound effect generation via HunyuanVideo-Foley model. 90 lines.

### Key Functions
| Function | Description |
|----------|-------------|
| `generate_foley_asset(prompt, output_path, duration)` | Generate foley SFX from text description |

---

# Core Libraries

## text_engine.py

**Text Engine** — Unified text generation interface supporting cloud (Gemini 2.0 Flash, Gemini 1.5 Pro) and local (Gemma via MLX) backends. Singleton pattern. 533 lines.

### TextEngine Class
| Method | Description |
|--------|-------------|
| `__init__(config_path)` | Initialize from env_vars.yaml. Auto-detects backend. |
| `generate(prompt, temperature, json_schema)` | Generate text (routes to cloud or local) |
| `get_gemini_client()` | Get rotated Gemini API client |
| `unload()` | Free local model memory |
| `clear_cache()` | Clear MLX/MPS caches |

### Backend Selection
The backend is determined by the `TEXT_ENGINE` environment variable:
- `"gemini_api"` → Cloud Gemini
- `"local_gemma"` → Local Gemma via MLX

### Factory
| Function | Description |
|----------|-------------|
| `get_engine()` | Singleton factory for TextEngine |

---

## truth_safety.py

**Truth & Safety** — Content moderation, prompt refinement, and image description. 284 lines.

### TruthSafety Class
| Method | Description |
|--------|-------------|
| `describe_image(image_path)` | Generate text description of an image |
| `wash_image(image_path)` | Sanitize image (remove PII, apply dazzle camo) |
| `refine_prompt(prompt, context, pg_mode, local_mode, parody_safe)` | Hyper-detailed prompt expansion for image generation |
| `soften_prompt(prompt, pg_mode)` | PG-safe prompt softening |
| `critique_dialogue(draft_line, character, context)` | Dialogue quality critique |

---

## definitions.py

**Model Registry & VP Forms** — Central registry for all models, VP Form configurations, and active profile management. 386 lines.

### Enums
- `BackendType`: `CLOUD`, `LOCAL`
- `Modality`: `TEXT`, `IMAGE`, `VIDEO`, `FOLEY`, `SPOKEN_TTS`, `CLONED_TTS`

### Model Registry Functions
| Function | Description |
|----------|-------------|
| `get_video_model(key)` | Legacy video model accessor (L/J/K/D tiers) |
| `get_active_model(modality, local)` | Get currently active ModelConfig for a modality |
| `set_active_model(modality, model_id, local)` | Set and persist active model |
| `load_active_profile()` | Load active_models.json (supports old flat + new nested format) |

### VP Form Functions
| Function | Description |
|----------|-------------|
| `resolve_vpform(input_string)` | Resolve form key or alias to VPFormConfig |
| `add_global_vpform_args(parser)` | Add positional cli_args to an argparse parser |
| `parse_global_vpform(args, current_default)` | Extract and resolve VPForm from args |

### Registered Models

#### Text
| ID | Backend | Notes |
|----|---------|-------|
| `gemini-2.0-flash` | cloud | Default cloud text |
| `gemini-1.5-pro` | cloud | Higher quality |
| `gemma-2-9b-it` | local | Default local text |
| `gemma-2-9b-it-director` | local | With director_v1 adapter |

#### Image
| ID | Backend | Notes |
|----|---------|-------|
| `gemini-2.5-flash-image` | cloud | Gemini image generation |
| `imagen-3` | cloud | Imagen 3 |
| `flux-schnell` | local | Flux.1-schnell |
| `flux-klein` | local | Flux 2 Klein 9B (default local) |
| `flux-gguf` | local | GGUF quantized (low memory) |
| `flux-2-klein-hf` | cloud | HF Inference Endpoint |

#### Video
| ID | Backend | Notes |
|----|---------|-------|
| `veo-3.1-fast` | cloud | Veo 3.1 fast |
| `veo-3.1-4k` | cloud | Veo 3.1 cinematic 4K |
| `ltx-video` | local | LTX-Video 13B |
| `skyreels` | local | SkyReels A2V |

#### TTS
| ID | Backend | Notes |
|----|---------|-------|
| `google-journey` | cloud | Google Journey voices |
| `kokoro-v1` | local | Kokoro ONNX |

### Active Profile Defaults

**Cloud**: gemini-2.0-flash (text), flux-klein (image), veo-3.1-fast (video), google-journey (TTS)

**Local**: gemma-2-9b-it (text), flux-klein (image), ltx-video (video), kokoro-v1 (TTS)

---

## mvp_shared.py

**Shared Data Models & Utilities** — Pydantic models, I/O functions, and XMVP XML serialization. 386 lines.

### Data Models (Pydantic)
| Model | Description |
|-------|-------------|
| `VPForm` | Genre and output mechanics |
| `Constraints` | Technical limits (resolution, FPS, duration) |
| `CSSV` | The "Bible" — Constraints, Scenario, Situation, Vision |
| `Story` | Narrative backbone (title, synopsis, characters, theme) |
| `Portion` | High-level narrative chunk (scene) |
| `Seg` | Executable technical segment (frame range + prompt) |
| `Manifest` | Segment-to-file mapping |
| `Indecision` | A/B test choice |
| `DialogueLine` | Single line of dialogue |
| `DialogueScript` | Full dialogue script |

### I/O Functions
| Function | Description |
|----------|-------------|
| `load_cssv(path)` | Load CSSV from JSON |
| `save_cssv(cssv, path)` | Save CSSV to JSON |
| `load_manifest(path)` | Load Manifest from JSON |
| `save_manifest(manifest, path)` | Save Manifest to JSON |
| `load_api_keys(env_path)` | Load ACTION_KEYS_LIST from YAML |
| `load_text_keys(env_path)` | Load TEXT_KEYS_LIST from YAML |
| `save_xmvp(data_models, path)` | Save to XMVP XML format |
| `safe_save_xmvp(out_path, bible, story, manifest, extra_meta)` | Safe XMVP save with error handling |
| `load_xmvp(path, key)` | Load specific key from XMVP XML |
| `load_nicotime_context(prompt_text, nicotime_root)` | Load NICOTIME entity context for a prompt |
| `get_client()` | Get rotated genai.Client |
| `get_project_id()` | Get GCP project ID |
| `setup_logging(name)` | Configure logging |

---

# Utility & Management Modules

## model_scout.py

**Model Scout** — Model registry status, switching, scanning, and downloading. 200 lines.

### Usage
```bash
python3 model_scout.py --status          # Check all model paths
python3 model_scout.py --list            # List registered models
python3 model_scout.py --list text       # List models for a modality
python3 model_scout.py --switch text gemma-2-9b-it  # Switch active model
python3 model_scout.py --scan            # Scan /Volumes/XMVPX/mw/ for models
python3 model_scout.py --probe MODEL     # Test a specific model endpoint
python3 model_scout.py --pull MODEL      # Download model from HuggingFace
```

---

## populate_models_xmvp.py

**Model Downloader** — Interactive script to download all model weights to `/Volumes/XMVPX/mw/`. Prompts for HuggingFace token. 311 lines.

---

## sassprilla_carbonator.py

**SASSPRILLA Carbonator** — Auto-expands short title-style prompts into dense, genre-appropriate visual concepts. 101 lines.

### Usage
```bash
python3 sassprilla_carbonator.py "Midnight Train To Georgia"
```

### Key Function
| Function | Description |
|----------|-------------|
| `carbonate_prompt(title, artist, extra_context)` | Expand title into rich visual concept (200–400 words) |

Automatically triggered in cartoon_producer and content_producer when prompt is short, title-case, and contains no periods.

---

## dialogue_critic.py

**Dialogue Critic** — Few-shot dialogue refinement using a corpus of parsed screenplays. 163 lines.

### DialogueCritic Class
| Method | Description |
|--------|-------------|
| `__init__(text_engine, corpus_root)` | Initialize with engine and screenplay corpus |
| `get_examples(k)` | Sample k dialogue examples from corpus |
| `refine(draft_line, character, context)` | Refine a draft dialogue line using few-shot examples |

---

## nicotime_index.py

**NICOTIME Indexer** — Noospheric entity research system. Distills prompts into interconnected concept entities and saves them as structured XML documents for creative enrichment. 244 lines.

### NicotimeIndexer Class
| Method | Description |
|--------|-------------|
| `distill_entities(prompt)` | Extract noospheric entities from a prompt |
| `research_entity(entity)` | Deep research on a single entity |
| `index_prompt(prompt)` | Full pipeline: distill → research → save XML |

---

## train_mll.py

**Movie-Level LoRA Trainer** — Trains Flux LoRA adapters on MPS/CPU using Rectified Flow Matching. Creates per-production style LoRAs from generated keyframes. 351 lines.

### Usage
```bash
python3 train_mll.py --images /path/to/frames/ --output lora_output/ --steps 500
```

---

## prep_movie_assets.py

**Asset Prep** — Generates character reference images, location reference images, and style reference images for production consistency. Uses Flux or Gemini. Includes outlier culling. 538 lines.

### Key Functions
| Function | Description |
|----------|-------------|
| `generate_char_prompts(char_name, style, count, anchor, wardrobe)` | Generate character reference prompts |
| `generate_loc_prompts(loc_name, style, count)` | Generate location reference prompts |
| `generate_style_prompts(style, count)` | Generate style reference prompts |
| `extract_locations(manifest)` | Extract unique locations from manifest |
| `cull_outliers(image_paths, keep_count)` | Remove inconsistent reference images |

---

## convert_voices.py

**Voice Converter** — Converts individual Kokoro `.pt` voice files into a single `.npz` archive. 38 lines.

---

## count_lines.py

**Line Counter** — Counts total dialogue lines in an XMVP XML manifest's Portions section. 27 lines.

---

## test_gen_capabilities.py

**Generation Tester** — Tests Imagen and Gemini image generation endpoints with API keys. 151 lines.

---

# Data Models & Schemas

## CSSV (Bible) Structure
```json
{
  "constraints": {
    "width": 768, "height": 768, "fps": 24,
    "max_duration_sec": 60.0, "target_segment_length": 4.0,
    "black_and_white": false, "silent": false, "style_bans": []
  },
  "scenario": "A 60.0-second creative-agency",
  "situation": "CONCEPT: Your concept here",
  "vision": "STYLE: ... AESTHETIC: ... PACING: ..."
}
```

## Manifest Structure
```json
{
  "segs": [
    {
      "id": 1, "start_frame": 0, "end_frame": 96,
      "prompt": "Scene description...", "action": "static",
      "model_overrides": {}
    }
  ],
  "files": {},
  "indecisions": [],
  "dialogue": null
}
```

## XMVP XML Format
```xml
<?xml version='1.0' encoding='utf-8'?>
<XMVP version="3.00">
  <Bible>{JSON}</Bible>
  <Story>{JSON}</Story>
  <Manifest>{JSON}</Manifest>
</XMVP>
```

---

# Configuration Files

## env_vars.yaml
```yaml
TEXT_ENGINE: "gemini_api"       # or "local_gemma"
LOCAL_MODEL_PATH: ""            # HuggingFace ID or path

GEMINI_API_KEY: "YOUR_KEY"
ACTION_KEYS_LIST: "key1,key2,key3,..."   # 16 keys for video/image (rotated)
TEXT_KEYS_LIST: "key4,key5,..."          # 8 keys for text operations
```

## active_models.json
Auto-generated profile tracking current active models. Supports both flat (legacy) and nested (cloud/local) formats:

```json
{
  "cloud": {
    "text": "gemini-2.0-flash",
    "image": "flux-klein",
    "video": "veo-3.1-fast",
    "spoken_tts": "google-journey"
  },
  "local": {
    "text": "gemma-2-9b-it",
    "image": "flux-klein",
    "video": "ltx-video",
    "spoken_tts": "kokoro-v1"
  }
}
```

---

# Training Data & Adapters

## Directory Structure
```
adapters/
├── director_v1/              # GemmaW director adapter (50-step checkpoints)
│   ├── 0000050_adapters.safetensors
│   ├── ...
│   └── 0000500_adapters.safetensors
└── movies/                   # Movie-Level LoRA templates
    ├── 24_Template.safetensors
    └── Route66_Template.safetensors

z_training_data/
├── thax_voice/               # Thax Douglas RVC voice model (shared with permission)
├── e47_voices/               # Element 47 cast voice references
│   ├── Anchor/
│   ├── Burn/
│   ├── Cruise/
│   └── Drip/
├── nicotime/                 # NICOTIME entity research documents (XML)
├── example_parodies/         # Reference parody scripts
├── hj24.csv                  # HJ24 beat template data
├── atui_235.md               # ATUI reference document
└── tlp.md                    # TLP reference document
```

## Using the Thax Douglas Voice
1. Ensure files are in `z_training_data/thax_voice/model/`
2. Set up RVC environment: `conda create -n rvc_env python=3.10 && pip install rvc-python`
3. Run: `python3 content_producer.py --vpform thax-douglas`