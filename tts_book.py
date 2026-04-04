#!/usr/bin/env python3
"""
tts_book.py — Standalone Book-to-Audiobook Converter

Converts prose books (novels, non-fiction) into audiobooks using local Kokoro TTS,
with LLM-driven speaker attribution and a 3-layer pronunciation correction system.

Usage:
    python3 tts_book.py book.txt                      # Multi-voice, full book
    python3 tts_book.py book.docx --n                 # Narrator mode
    python3 tts_book.py book.md --n --voice af_bella  # Female narrator
    python3 tts_book.py book.txt --chapter 3          # Test single chapter
    python3 tts_book.py book.txt --no-verify          # Skip Whisper (faster)
    python3 tts_book.py book.txt --local              # Use local Gemma, no API
    python3 tts_book.py book.txt --dict my_words.json # Custom pronunciation dict
    python3 tts_book.py book.txt --speed 0.9          # Slower narration
    python3 tts_book.py book.txt --out /path/to/dir   # Custom output dir
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Conditional imports ---

DOCX_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    pass

WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    pass

# --- Local imports ---
from kokoro_bridge import get_kokoro_bridge
from f5_bridge import get_f5_bridge
from cloud_tts_bridge import get_cloud_tts_bridge
from rvc_bridge import get_rvc_bridge
from foley_talk import assign_kokoro_voice_deterministic, pitch_shift_file
from text_engine import TextEngine

# --- Configuration ---
KOKORO_MODEL = "/Users/m3u/METMcloud/METMroot/tools/fmv/weights/kokoro-root/kokoro-v0_19.onnx"
F5TTS_MODEL_DIR = "/Users/m3u/METMcloud/METMroot/tools/fmv/weights/f5tts-root"
MAX_TTS_CHARS = 600  # Max characters per TTS call; longer segments get split

# --- Initial Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# Phase 1: Parse
# =============================================================================

def load_book_text(file_path):
    """Load text from .txt, .md, or .docx files."""
    ext = Path(file_path).suffix.lower()

    if ext in ('.txt', '.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.docx':
        if not DOCX_AVAILABLE:
            logging.error("python-docx not installed. Run: pip install python-docx")
            sys.exit(1)
        doc = docx.Document(file_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    else:
        logging.error(f"Unsupported file format: {ext}")
        sys.exit(1)


def split_into_chapters(text):
    """Split text into chapters by detecting common patterns."""
    lines = text.split('\n')

    # Patterns that signal a chapter boundary
    chapter_re = re.compile(
        r'^(?:Chapter|CHAPTER)\s+\w+'   # Chapter 1, CHAPTER ONE
        r'|^#{1,3}\s+\S'                # Markdown headers
        r'|^-{3,}\s*$'                  # --- separators
        r'|^\*{3,}\s*$',                # *** separators
        re.IGNORECASE
    )

    boundaries = []
    for i, line in enumerate(lines):
        if chapter_re.match(line.strip()):
            boundaries.append(i)

    # Detect large blank-line clusters (5+ consecutive blanks) as boundaries
    blank_run = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            blank_run += 1
        else:
            if blank_run >= 5 and i not in boundaries:
                boundaries.append(i)
            blank_run = 0

    boundaries = sorted(set(boundaries))

    if not boundaries:
        return [{"title": "Full Text", "text": text}]

    chapters = []

    # Handle text before first boundary (preamble)
    if boundaries[0] > 0:
        preamble = '\n'.join(lines[:boundaries[0]]).strip()
        if preamble and len(preamble) > 100:
            chapters.append({"title": "Preamble", "text": preamble})

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        chapter_lines = lines[start:end]

        # First non-empty line as title
        title = f"Chapter {len(chapters) + 1}"
        for line in chapter_lines:
            stripped = line.strip()
            if stripped:
                clean = re.sub(r'^[#\-*]+\s*', '', stripped)
                if clean:
                    title = clean[:80]
                break

        chapter_text = '\n'.join(chapter_lines).strip()
        if chapter_text:
            chapters.append({"title": title, "text": chapter_text})

    if not chapters:
        return [{"title": "Full Text", "text": text}]

    return chapters


# =============================================================================
# Phase 2: Attribute Speakers
# =============================================================================

SECTION_BREAK_RE = re.compile(r'^[\-\*\~\=]{3,}\s*$')


def extract_segments_regex(text):
    """Fallback regex-only dialogue detection."""
    segments = []
    paragraphs = re.split(r'\n\s*\n', text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect section breaks
        if SECTION_BREAK_RE.match(para):
            segments.append({
                "speaker": "NARRATOR",
                "text": "---",
                "is_dialogue": False,
                "is_section_break": True
            })
            continue

        # Split on quoted speech (straight and smart quotes)
        quote_pattern = r'[\u201c""](.*?)[\u201d""]'
        parts = re.split(quote_pattern, para)

        if len(parts) > 1:
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                if i % 2 == 1:  # Inside quotes
                    segments.append({
                        "speaker": "UNKNOWN_SPEAKER",
                        "text": part,
                        "is_dialogue": True
                    })
                else:
                    segments.append({
                        "speaker": "NARRATOR",
                        "text": part,
                        "is_dialogue": False
                    })
        else:
            segments.append({
                "speaker": "NARRATOR",
                "text": para,
                "is_dialogue": False
            })

    return segments


class SpeakerAttributor:
    """LLM-driven speaker attribution with caching."""

    def __init__(self, text_engine, cache_dir):
        self.engine = text_engine
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def attribute_chapter(self, chapter_num, chapter_text):
        """Attribute speakers for a chapter, using cache if available."""
        cache_file = self.cache_dir / f"attribution_ch{chapter_num}.json"

        if cache_file.exists():
            logging.info(f"   Using cached attribution for chapter {chapter_num}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Chunk into ~2000-word blocks
        words = chapter_text.split()
        chunk_size = 2000
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(' '.join(words[i:i + chunk_size]))

        all_segments = []

        for chunk_idx, chunk in enumerate(chunks):
            logging.info(f"   Attributing speakers (chapter {chapter_num}, "
                         f"chunk {chunk_idx + 1}/{len(chunks)})...")

            prompt = (
                "Analyze this text passage and identify narration vs dialogue.\n"
                "For each piece of text, determine:\n"
                "1. Whether it is narration or dialogue\n"
                "2. If dialogue, who is speaking (character name)\n"
                "3. The exact text content\n\n"
                "Return a JSON array where each element has:\n"
                '- "speaker": character name or "NARRATOR"\n'
                '- "text": the exact text content\n'
                '- "is_dialogue": true for dialogue, false for narration\n\n'
                "Rules:\n"
                "- Preserve ALL text — every word must appear in exactly one segment\n"
                "- Dialogue = text inside quotation marks\n"
                "- Use context clues (said X, X replied) to attribute speakers\n"
                '- Unknown speakers → "UNKNOWN_SPEAKER"\n'
                "- Narration = everything outside quotes\n"
                "- Keep segments in original order\n"
                "- Section breaks (--- or ***) → NARRATOR with text \"---\"\n\n"
                f"TEXT:\n{chunk}\n\n"
                "Return ONLY the JSON array."
            )

            try:
                result = self.engine.generate(prompt, temperature=0.1, json_schema=True)
                segments = json.loads(result)
                if isinstance(segments, list):
                    all_segments.extend(segments)
                else:
                    logging.warning(f"   LLM returned non-list for chunk {chunk_idx}, "
                                    "falling back to regex")
                    all_segments.extend(extract_segments_regex(chunk))
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"   LLM attribution failed ({e}), "
                                f"falling back to regex for chunk {chunk_idx}")
                all_segments.extend(extract_segments_regex(chunk))

        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(all_segments, f, indent=2)

        return all_segments


# =============================================================================
# Phase 3: Assign Voices
# =============================================================================

class VoiceAssigner:
    """Manages consistent voice/pitch assignments for speakers."""

    def __init__(self, available_voices, narrator_mode=False, narrator_voice="am_michael", use_f5=False):
        self.available_voices = available_voices
        self.narrator_mode = narrator_mode
        self.narrator_voice = narrator_voice
        self.use_f5 = use_f5
        self.assignments = {}

    def assign(self, speaker):
        """Get or create voice assignment for a speaker."""
        if speaker in self.assignments:
            return self.assignments[speaker]

        if self.narrator_mode:
            # Narrator mode: single base voice, pitch variants for characters
            if speaker == "NARRATOR":
                assignment = (self.narrator_voice, 0)
            else:
                pitch_options = [-4, -3, -2, -1, 1, 2, 3, 4]
                seed = sum(ord(c) for c in speaker)
                pitch = pitch_options[seed % len(pitch_options)]
                assignment = (self.narrator_voice, pitch)
        else:
            # Multi-voice mode: unique voice per character
            if speaker == "NARRATOR":
                base = self.available_voices[0] if self.available_voices else "am_michael"
                assignment = (base, 0)
            else:
                assignment = assign_kokoro_voice_deterministic(speaker, self.available_voices)

        self.assignments[speaker] = assignment
        if not self.use_f5:
            logging.info(f"   Voice: {speaker} -> {assignment[0]} @ {assignment[1]:+d} st")
        return assignment


# =============================================================================
# Phase 4: Pronunciation Correction (3 Layers)
# =============================================================================

class PronunciationCorrector:
    """3-layer pronunciation correction system.

    Layer 1: Static dictionary (pronunciation_dict.json + book_overrides.json)
    Layer 2: LLM heteronym resolution (one call per chapter, cached)
    Layer 3: Whisper verification loop (optional, post-render)
    """

    def __init__(self, base_dict_path, book_overrides_path, text_engine=None, cache_dir=None):
        self.base_dict = {"simple": {}, "heteronyms": []}
        self.book_overrides = {}
        self.text_engine = text_engine
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.book_overrides_path = book_overrides_path

        # Load base dictionary
        if base_dict_path and os.path.exists(base_dict_path):
            with open(base_dict_path, 'r') as f:
                self.base_dict = json.load(f)
            n_simple = len(self.base_dict.get('simple', {}))
            n_het = len(self.base_dict.get('heteronyms', []))
            logging.info(f"   Loaded {n_simple} simple + {n_het} heteronym pronunciation rules")

        # Load book overrides (accumulated from Whisper corrections)
        if book_overrides_path and os.path.exists(book_overrides_path):
            with open(book_overrides_path, 'r') as f:
                self.book_overrides = json.load(f)
            logging.info(f"   Loaded {len(self.book_overrides)} book-specific overrides")
        else:
            self.book_overrides = {}

    def correct_layer1(self, text):
        """Layer 1: Static dictionary replacements."""
        result = text

        # Book overrides take precedence
        override_keys = set()
        for word, replacement in self.book_overrides.items():
            if replacement is None:
                continue
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            override_keys.add(word.lower())

        # Simple replacements (skip if overridden)
        for word, replacement in self.base_dict.get('simple', {}).items():
            if word.lower() not in override_keys:
                if replacement is None:
                    continue
                pattern = r'\b' + re.escape(word) + r'\b'
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Context-aware heteronyms (regex patterns)
        for entry in self.base_dict.get('heteronyms', []):
            pattern = entry.get('pattern', '')
            replacement = entry.get('replacement')
            if pattern and replacement:
                try:
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                except re.error:
                    pass

        return result

    def correct_layer2(self, text, chapter_num):
        """Layer 2: LLM heteronym resolution (one call per chapter, cached)."""
        if not self.text_engine:
            return text

        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"heteronyms_ch{chapter_num}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                for word, replacement in cached.items():
                    if replacement is None:
                        continue
                    pattern = r'\b' + re.escape(word) + r'\b'
                    text = re.sub(pattern, replacement, text)
                return text

        prompt = (
            "Identify heteronyms in this text — words spelled the same but "
            "pronounced differently depending on meaning.\n\n"
            "Common heteronyms: read (reed/red), lead (leed/led), "
            "live (liv/lyve), close (kloce/kloze), wound (woond/wownd), "
            "tear (teer/tare), bow (boh/bow), bass (base/bass), "
            "minute (MIN-it/my-NOOT), etc.\n\n"
            "For EACH heteronym found, determine the correct pronunciation "
            "from context and provide a phonetic respelling a TTS engine "
            "would pronounce correctly.\n\n"
            "Return a JSON object mapping each heteronym to its respelling.\n"
            "If no heteronyms found, return {}.\n\n"
            f"TEXT:\n{text[:3000]}\n\n"
            "Return ONLY the JSON object."
        )

        try:
            result = self.text_engine.generate(prompt, temperature=0.1, json_schema=True)
            corrections = json.loads(result)
            if isinstance(corrections, dict) and corrections:
                if cache_file:
                    with open(cache_file, 'w') as f:
                        json.dump(corrections, f, indent=2)
                for word, replacement in corrections.items():
                    if replacement is None:
                        continue
                    pattern = r'\b' + re.escape(word) + r'\b'
                    text = re.sub(pattern, replacement, text)
        except Exception as e:
            logging.warning(f"   Layer 2 correction failed: {e}")

        return text

    def correct(self, text, chapter_num=0):
        """Apply Layer 1 static dictionary corrections on a copy of the text.
        (Layer 2 LLM pass is disabled by default to prevent over-eager respellings)."""
        result = self.correct_layer1(text)
        return result

    def verify_and_correct(self, audio_path, original_text, renderer_cb):
        """Layer 3: Whisper verification loop.

        After rendering a line, transcribe with faster-whisper tiny,
        diff against source. On mismatch: phonetic respelling, re-render.
        Max 2 retries per segment. Returns final audio path.

        renderer_cb(corrected_text) -> new_audio_path  (re-render callback)
        """
        if not WHISPER_AVAILABLE:
            return audio_path

        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception:
            return audio_path

        current_audio = audio_path

        for attempt in range(2):
            try:
                segs, _ = model.transcribe(current_audio)
                transcribed = ' '.join(s.text for s in segs).strip()
            except Exception:
                break

            # Strip punctuation for comparison
            def normalize_word(w):
                return re.sub(r'[^\w]', '', w.lower())

            orig_words = original_text.split()
            trans_words = transcribed.split()

            import difflib

            # Expanded STOP_WORDS to cover more common verbs and prepositions
            STOP_WORDS = {
                "the", "and", "a", "to", "of", "in", "is", "it", "that", "was", "for", "on", "are", 
                "with", "as", "i", "his", "they", "be", "at", "one", "have", "this", "from", "or", 
                "had", "by", "hot", "but", "some", "what", "there", "out", "so", "were", "when", 
                "your", "can", "said", "all", "we", "how", "has", "do", "up", "out", "if", "about", 
                "who", "get", "which", "go", "me", "very", "my", "then", "into", "like", "just", 
                "him", "know", "take", "people", "could", "make", "than", "now", "look", "only", 
                "come", "its", "over", "think", "also", "back", "after", "use", "two", "our", 
                "work", "first", "well", "way", "even"
            }

            # Pre-compute overridden keys to avoid re-correcting manual fixes
            overridden_keys = {normalize_word(k) for k in self.book_overrides}
            overridden_keys.update({normalize_word(k) for k in self.base_dict.get('simple', {})})

            mismatches = []
            for i, o in enumerate(orig_words):
                norm_o = normalize_word(o)
                
                # Skip words under 4 characters, stop words, and words we already provided a fix for
                if len(norm_o) <= 3 or norm_o in STOP_WORDS or norm_o in overridden_keys:
                    continue
                
                # Sliding window of transcribed words (+/- 3 words)
                window = trans_words[max(0, i-3):min(len(trans_words), i+4)]
                
                match_found = False
                for tw in window:
                    norm_tw = normalize_word(tw)
                    # Use fuzzy matching to allow for slight Whisper transcription variances 
                    # (e.g., "calloused" -> "callus", "sorting" -> "sortin")
                    similarity = difflib.SequenceMatcher(None, norm_o, norm_tw).ratio()
                    if similarity > 0.75: # 75% similarity threshold
                        match_found = True
                        break
                
                if not match_found:
                    # Give it one more chance against the combined window string
                    # (in case Whisper missed a space entirely)
                    combined_window = "".join(normalize_word(tw) for tw in window)
                    if norm_o in combined_window:
                        continue

                    # Truly missing or heavily mispronounced
                    mismatches.append((o, "???")) 

            if not mismatches:
                break

            logging.info(f"   Whisper: {len(mismatches)} mismatch(es), attempt {attempt + 1}")

            corrected_text = original_text
            new_corrections = False
            for orig_word, trans_word in mismatches[:3]:
                phonetic = self._phonetic_respelling(orig_word, trans_word)
                if phonetic and phonetic != orig_word:
                    corrected_text = corrected_text.replace(orig_word, phonetic)
                    self.book_overrides[orig_word] = phonetic
                    new_corrections = True

            if new_corrections:
                # Save updated overrides
                if self.book_overrides_path:
                    with open(self.book_overrides_path, 'w') as f:
                        json.dump(self.book_overrides, f, indent=2)
                # Re-render
                new_audio = renderer_cb(corrected_text)
                if new_audio:
                    current_audio = new_audio

        return current_audio

    def _phonetic_respelling(self, original, transcribed):
        """Generate a phonetic respelling for a problem word via LLM.
        Forces lowercase to prevent Cloud TTS from treating ALL CAPS as acronyms."""
        if not self.text_engine:
            return original

        prompt = (
            f'The TTS engine pronounced "{original}" as "{transcribed}".\n'
            "Generate a phonetic respelling that would make a TTS engine "
            "pronounce it correctly.\n"
            "CRITICAL RULE: DO NOT use ANY capital letters. All caps are read as acronyms.\n"
            "Return ONLY the lowercase respelling, nothing else. Examples:\n"
            '- "colonel" -> "kernel"\n'
            '- "debris" -> "deh-bree"\n'
            '- "quinoa" -> "keen-wah"\n\n'
            f"Word: {original}\nRespelling:"
        )

        try:
            result = self.text_engine.generate(prompt, temperature=0.1)
            # Strip quoting and force lowercase to guarantee no acronym behavior
            return result.strip().strip('"').strip("'").lower()
        except Exception:
            return original


# =============================================================================
# Phase 5: Render
# =============================================================================

def split_long_text(text, max_chars=MAX_TTS_CHARS):
    """Split long text on sentence boundaries for TTS."""
    if len(text) <= max_chars:
        return [text]

    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()

    if current:
        chunks.append(current.strip())

    # If any chunk is still too long, force-split on commas/spaces
    final = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            # Split on commas
            parts = chunk.split(', ')
            sub = ""
            for part in parts:
                if len(sub) + len(part) + 2 > max_chars and sub:
                    final.append(sub.strip())
                    sub = part
                else:
                    sub = (sub + ", " + part).strip(', ')
            if sub:
                final.append(sub.strip())

    return final if final else [text]


class BookRenderer:
    """Renders attributed segments to audio files using Kokoro or F5-TTS."""

    def __init__(self, voice_assigner, corrector, output_dir, speed=1.0, verify=True, use_f5=False, use_kokoro=False, ref_audio=None, ref_text="", rvc_dir=None):
        self.voice_assigner = voice_assigner
        self.corrector = corrector
        self.audio_dir = Path(output_dir) / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.speed = speed
        self.verify = verify
        self.use_f5 = use_f5
        self.use_kokoro = use_kokoro
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.rvc_dir = rvc_dir
        self._bridge = None

    @property
    def bridge(self):
        if self._bridge is None:
            if self.use_f5:
                self._bridge = get_f5_bridge(F5TTS_MODEL_DIR)
                self._bridge.load()
            elif self.use_kokoro:
                self._bridge = get_kokoro_bridge(KOKORO_MODEL)
                self._bridge.load()
            else:
                self._bridge = get_cloud_tts_bridge() # Cloud TTS (Google) is the default
        return self._bridge

    def clean_text(self, text):
        """Clean text for TTS consumption (on a copy, never modifies source)."""
        t = re.sub(r'[*#]', '', text)
        t = re.sub(r'\(.*?\)', '', t)
        t = re.sub(r'\[.*?\]', '', t)
        t = re.sub(r'\u2014+', '...', t)     # em-dashes
        t = re.sub(r'--+', '...', t)          # double dashes
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def _render_tts(self, text, output_path, voice_name, pitch, speed):
        """Render text to audio via Kokoro bridge + optional pitch shift."""
        # --- Add pacing for Cloud TTS ---
        if not self.use_kokoro and not self.use_f5:
            # Journey voices don't support SSML, so we force a breath with ellipses
            text = re.sub(r'([.!?])(?=\s|$)', r'\1 ... ', text)

        if pitch == 0:
            ok = self.bridge.generate(text, output_path, voice_name=voice_name, speed=speed)
            return output_path if ok else None

        temp_raw = output_path.replace(".wav", "_raw.wav")
        ok = self.bridge.generate(text, temp_raw, voice_name=voice_name, speed=speed)
        if not ok:
            return None

        shifted = pitch_shift_file(temp_raw, pitch)
        if shifted != output_path:
            shutil.move(shifted, output_path)
        if os.path.exists(temp_raw) and temp_raw != output_path:
            os.remove(temp_raw)
        return output_path

    def render_segment(self, segment, seg_idx, chapter_num):
        """Render a single segment to audio. Returns list of audio paths."""
        speaker = segment.get('speaker', 'NARRATOR')
        raw_text = segment.get('text', '')

        # Section break -> silence
        if segment.get('is_section_break') or SECTION_BREAK_RE.match(raw_text.strip()):
            path = self._generate_silence(
                str(self.audio_dir / f"ch{chapter_num:03d}_break{seg_idx:04d}.wav"), 1.5)
            return [path] if path else []

        cleaned = self.clean_text(raw_text)
        if not cleaned or len(cleaned) < 2:
            return []

        # Get voice assignment
        voice_name, pitch = self.voice_assigner.assign(speaker)
        if not self.use_f5:
            logging.info(f"   Voice: {speaker} -> {voice_name} @ {pitch:+} st")

        # Split long text
        text_chunks = split_long_text(cleaned)
        audio_paths = []

        for ci, chunk in enumerate(text_chunks):
            # Apply pronunciation correction (on a copy)
            corrected = self.corrector.correct(chunk, chapter_num)

            suffix = f"_{ci}" if len(text_chunks) > 1 else ""
            out_path = str(self.audio_dir /
                           f"ch{chapter_num:03d}_seg{seg_idx:04d}{suffix}_{speaker}.wav")

            try:
                if os.path.exists(out_path):
                    result = True
                elif self.use_f5:
                    result = self.bridge.generate(corrected, out_path, ref_audio=self.ref_audio, ref_text=self.ref_text, speed=self.speed)
                else:
                    result = self._render_tts(corrected, out_path, voice_name, pitch, self.speed)

                if result:
                    if self.verify and WHISPER_AVAILABLE:
                        def rerender(new_text, _p=out_path, _v=voice_name,
                                     _pi=pitch, _s=self.speed):
                            rr = _p.replace(".wav", "_rr.wav")
                            if self.use_f5:
                                r = self.bridge.generate(new_text, rr, ref_audio=self.ref_audio, ref_text=self.ref_text, speed=_s)
                            else:
                                r = self._render_tts(new_text, rr, _v, _pi, _s)
                            if r:
                                shutil.move(rr, _p)
                                return _p
                            return None

                        self.corrector.verify_and_correct(out_path, chunk, rerender)

                    # --- Layer 4: RVC Post-Processing ---
                    if self.rvc_dir:
                        rvc = get_rvc_bridge()
                        rvc_ok = rvc.generate(out_path, self.rvc_dir)
                        if not rvc_ok:
                            logging.error(f"   ❌ RVC conversion failed for {out_path}")

                    audio_paths.append(out_path)
                else:
                    logging.warning(f"   Failed segment {seg_idx}, inserting silence")
                    sil = self._generate_silence(out_path, 0.5)
                    if sil:
                        audio_paths.append(sil)
            except Exception as e:
                logging.error(f"   Render error segment {seg_idx}: {e}")
                sil = self._generate_silence(out_path, 0.5)
                if sil:
                    audio_paths.append(sil)

        return audio_paths

    def _generate_silence(self, output_path, duration):
        """Generate a silence WAV file."""
        try:
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i',
                f'anullsrc=r=24000:cl=mono',
                '-t', str(duration), output_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path
        except Exception:
            return None

    def generate_pause(self, duration, pause_idx, chapter_num):
        """Generate a pause (silence) of given duration."""
        out_path = str(self.audio_dir / f"ch{chapter_num:03d}_pause{pause_idx:04d}.wav")
        return self._generate_silence(out_path, duration)

    def render_chapter(self, chapter_num, segments):
        """Render all segments in a chapter with appropriate pauses."""
        audio_files = []
        pause_idx = 0
        total = len(segments)

        for seg_idx, segment in enumerate(segments):
            if seg_idx % 10 == 0:
                logging.info(f"   Rendering segment {seg_idx + 1}/{total}...")

            paths = self.render_segment(segment, seg_idx, chapter_num)
            audio_files.extend(paths)

            # Pause between segments
            if seg_idx < total - 1:
                next_seg = segments[seg_idx + 1]
                cur_dial = segment.get('is_dialogue', False)
                nxt_dial = next_seg.get('is_dialogue', False)

                # Section breaks already have built-in silence
                if segment.get('is_section_break') or SECTION_BREAK_RE.match(
                        segment.get('text', '').strip()):
                    continue
                if next_seg.get('is_section_break') or SECTION_BREAK_RE.match(
                        next_seg.get('text', '').strip()):
                    continue

                if cur_dial and nxt_dial:
                    pause_dur = 0.4   # Quick dialogue exchange
                else:
                    pause_dur = 0.8   # Paragraph break

                pause_path = self.generate_pause(pause_dur, pause_idx, chapter_num)
                if pause_path:
                    audio_files.append(pause_path)
                pause_idx += 1

        return audio_files


# =============================================================================
# Phase 6: Normalize
# =============================================================================

def normalize_audio(input_path, output_path):
    """Loudness-normalize a single audio file to -16 LUFS."""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
            '-ar', '44100', '-ac', '2',
            '-c:a', 'pcm_s16le',
            output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except Exception as e:
        logging.warning(f"   Normalization failed for {Path(input_path).name}: {e}")
        shutil.copy(input_path, output_path)
        return output_path


def normalize_batch(audio_files, norm_dir):
    """Normalize all audio files, returning list of normalized paths."""
    normalized = []
    total = len(audio_files)
    for i, audio_path in enumerate(audio_files):
        if i % 20 == 0:
            logging.info(f"   Normalizing {i + 1}/{total}...")
        norm_path = os.path.join(norm_dir, os.path.basename(audio_path))
        normalize_audio(audio_path, norm_path)
        normalized.append(norm_path)
    return normalized


# =============================================================================
# Phase 7: Stitch & Export
# =============================================================================

def stitch_and_export(audio_files, output_dir, book_title):
    """Concatenate normalized audio files and export as MP3 at 192kbps."""
    concat_file = os.path.join(output_dir, "concat.txt")
    final_wav = os.path.join(output_dir, "final_mix.wav")
    final_mp3 = os.path.join(output_dir, f"{book_title}.mp3")

    # Write concat file
    with open(concat_file, 'w') as f:
        for audio_path in audio_files:
            f.write(f"file '{os.path.abspath(audio_path)}'\n")

    # Concat to WAV
    logging.info("   Concatenating audio files...")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c:a', 'pcm_s16le',
            final_wav
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        logging.error(f"Concat failed: {e}")
        return None

    # Export to MP3
    logging.info("   Exporting MP3 at 192kbps...")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', final_wav,
            '-codec:a', 'libmp3lame', '-b:a', '192k',
            final_mp3
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        logging.error(f"MP3 export failed: {e}")
        return final_wav

    # Clean up intermediate WAV
    try:
        os.remove(final_wav)
    except OSError:
        pass

    return final_mp3


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="tts_book -- Convert books to audiobooks using Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 tts_book.py book.txt                      # Multi-voice\n"
            "  python3 tts_book.py book.docx --n                 # Narrator mode\n"
            "  python3 tts_book.py book.md --n --voice af_bella  # Female narrator\n"
            "  python3 tts_book.py book.txt --chapter 3          # Single chapter\n"
            "  python3 tts_book.py book.txt --no-verify          # Skip Whisper\n"
            "  python3 tts_book.py book.txt --local              # Local Gemma\n"
            "  python3 tts_book.py book.txt --f5                 # Use F5-TTS\n"
            "  python3 tts_book.py book.txt --speed 0.9          # Slower\n"
        )
    )

    parser.add_argument("book", help="Input book file (.txt, .md, or .docx)")
    parser.add_argument("--n", action="store_true", dest="narrative",
                        help="Narrator mode: single voice, pitch variants for dialogue")
    parser.add_argument("--voice", default="am_michael",
                        help="Base voice for narrator mode (default: am_michael)")
    parser.add_argument("--chapter", type=int,
                        help="Render only this chapter number (1-indexed)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip Whisper verification (faster)")
    parser.add_argument("--local", action="store_true",
                        help="Use local Gemma instead of Gemini API")
    parser.add_argument("--dict", default=None,
                        help="Custom pronunciation dictionary path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="TTS speed multiplier (default: 1.0)")
    parser.add_argument("--out", default=None,
                        help="Custom output directory")
    parser.add_argument("--google", action="store_true",
                        help="Force Google Cloud TTS (Default natively if no --kokoro or --f5)")
    parser.add_argument("--kokoro", action="store_true",
                        help="Use local Kokoro TTS instead of Cloud TTS")
    parser.add_argument("--f5", action="store_true",
                        help="Use F5-TTS instead of Cloud/Kokoro")
    parser.add_argument("--vf", type=str, default=None,
                        help="Path to an RVC model folder containing .pth and .index for post-processing")
    parser.add_argument("--ref-audio", default=None,
                        help="Reference audio for F5-TTS voice cloning")
    parser.add_argument("--ref-text", default="",
                        help="Transcript of reference audio for F5-TTS")
    parser.add_argument("--mp3s", action="store_true",
                        help="Export each chapter as a separate MP3 file (e.g., BookTitle-01.mp3)")

    args = parser.parse_args()

    # --- Validate input ---
    if not os.path.exists(args.book):
        logging.error(f"File not found: {args.book}")
        sys.exit(1)

    # Check espeak (required by Kokoro)
    if not shutil.which("espeak-ng") and not shutil.which("espeak"):
        logging.warning("espeak/espeak-ng not found. Kokoro TTS may fail.")

    # --- Output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    book_stem = Path(args.book).stem

    if args.out:
        output_dir = args.out
    else:
        output_dir = os.path.join("z_test-outputs", f"ttsbook_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    audio_dir = os.path.join(output_dir, "audio")
    norm_dir = os.path.join(output_dir, "normalized")
    cache_dir = os.path.join(output_dir, "cache")
    for d in [audio_dir, norm_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    book_overrides_path = os.path.join(output_dir, "book_overrides.json")

    # --- Set TextEngine mode ---
    if args.local:
        os.environ["TEXT_ENGINE"] = "local_gemma"

    logging.info(f"Loading book: {args.book}")

    # =========================================================================
    # Phase 1: Parse
    # =========================================================================
    text = load_book_text(args.book)
    chapters = split_into_chapters(text)
    logging.info(f"   Found {len(chapters)} chapter(s)")

    if args.chapter:
        if args.chapter < 1 or args.chapter > len(chapters):
            logging.error(f"Chapter {args.chapter} out of range (1-{len(chapters)})")
            sys.exit(1)
        chapters = [chapters[args.chapter - 1]]
        logging.info(f"   Rendering only chapter {args.chapter}: {chapters[0]['title']}")

    # =========================================================================
    # Initialize components
    # =========================================================================
    text_engine = TextEngine()

    attributor = SpeakerAttributor(text_engine, cache_dir)

    available_voices = []
    if args.google:
        available_voices = ["en-US-Journey-D", "en-US-Journey-F", "en-US-Journey-O"]
        logging.info(f"   Google Cloud TTS Voices available: {len(available_voices)}")
    elif not args.f5:
        try:
            # Init Kokoro and get voice list
            bridge = get_kokoro_bridge(KOKORO_MODEL)
            bridge.load()
            available_voices = bridge.get_voice_list()
        except Exception:
            available_voices = []
        if not available_voices:
            available_voices = ["af_bella", "af_sarah", "am_michael", "am_adam"]
        logging.info(f"   Kokoro Voices available: {len(available_voices)} "
                     f"(expanded to {len(available_voices) * 3} slots)")
    else:
        logging.info("   F5-TTS Mode Enabled (Zero-shot cloning)")

    voice_assigner = VoiceAssigner(
        available_voices,
        narrator_mode=args.narrative,
        narrator_voice=args.voice,
        use_f5=args.f5
    )

    # Pronunciation dict
    dict_path = args.dict
    if not dict_path:
        dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "pronunciation_dict.json")
    
    # Do not use baseline pronunciation dict for Google TTS or F5-TTS
    # as these engines are natively better at handling standard English
    # and the base dict is heavily overtuned for Kokoro.
    if args.google or args.f5:
        dict_path = None

    corrector = PronunciationCorrector(
        base_dict_path=dict_path,
        book_overrides_path=book_overrides_path,
        text_engine=text_engine,
        cache_dir=cache_dir
    )

    renderer = BookRenderer(
        voice_assigner, corrector,
        output_dir,
        speed=args.speed,
        verify=not args.no_verify,
        use_f5=args.f5,
        use_kokoro=args.kokoro,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        rvc_dir=args.vf
    )

    # =========================================================================
    # Process chapters
    # =========================================================================
    all_audio_files = []

    for ch_idx, chapter in enumerate(chapters):
        ch_num = args.chapter if args.chapter else ch_idx + 1

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Chapter {ch_num}: {chapter['title']}")
        logging.info(f"{'=' * 60}")

        # Phase 2: Attribute speakers
        segments = attributor.attribute_chapter(ch_num, chapter['text'])
        logging.info(f"   {len(segments)} segments attributed")

        # Phase 5: Render
        chapter_audio = renderer.render_chapter(ch_num, segments)
        logging.info(f"   {len(chapter_audio)} audio files generated for chapter {ch_num}")

        # Chapter break pause (3 seconds) between chapters
        if ch_idx < len(chapters) - 1:
            pause_path = renderer.generate_pause(3.0, 9999, ch_num)
            if pause_path:
                chapter_audio.append(pause_path)

        all_audio_files.extend(chapter_audio)

    if not all_audio_files:
        logging.error("No audio generated!")
        sys.exit(1)

    # =========================================================================
    # Phase 6: Normalize
    # =========================================================================
    logging.info(f"\nNormalizing {len(all_audio_files)} audio files...")
    normalized_files = normalize_batch(all_audio_files, norm_dir)

    # =========================================================================
    # Phase 7: Stitch & Export
    # =========================================================================
    logging.info("Stitching and exporting...")
    book_title = re.sub(r'[^\w\s-]', '', book_stem).strip()
    if not book_title:
        book_title = "Audiobook"

    if args.mp3s:
        import collections
        ch_files = collections.defaultdict(list)
        for f in normalized_files:
            m = re.search(r'ch(\d+)_', os.path.basename(f))
            if m:
                ch_files[int(m.group(1))].append(f)
            else:
                ch_files[0].append(f)
                
        final_path = None
        for ch_num in sorted(ch_files.keys()):
            ch_title = f"{book_title}-{ch_num:02d}"
            logging.info(f"\n   Stitching Chapter {ch_num}...")
            ch_path = stitch_and_export(ch_files[ch_num], output_dir, ch_title)
            if ch_path:
                logging.info(f"   Chapter {ch_num} saved to: {ch_path}")
                final_path = ch_path
                
    else:
        final_path = stitch_and_export(normalized_files, output_dir, book_title)

    if final_path:
        logging.info(f"\nDone! Audiobook saved to: {output_dir}")
    else:
        logging.error("Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
