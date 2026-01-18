#!/usr/bin/env python3
"""
dialogue_critic.py
------------------
"Gemma Wittgenstein" - A Dialogue Refiner.

This module validates and polishes generated dialogue by comparing it against
a local corpus of professional screenplays ("Hollywood Babylon").

Features:
- Loads parsed screenplay JSONs from `z_training_data/parsed_scripts`.
- Uses Few-Shot prompting to align generated text with cinematic standards.
- GRACEFUL FALLBACK: If no corpus is found (e.g., public repo pull), 
  it returns the original text unmodified or applies a basic "Script Doctor" pass.

Usage:
    from dialogue_critic import DialogueCritic
    critic = DialogueCritic(text_engine)
    better_line = critic.refine("Hello there friend, I am sad.", character="Bob")
"""

import os
import json
import random
import logging
from pathlib import Path

# Config
# We assume this path relative to the tool execution or MV root
CORPUS_REL_PATH = "z_training_data/parsed_scripts"

class DialogueCritic:
    def __init__(self, text_engine=None, corpus_root=None):
        self.engine = text_engine
        self.corpus_lines = []
        self.loaded = False
        
        # Determine Path
        if corpus_root:
            self.root = Path(corpus_root)
        else:
            # Try to find it relative to this file
            self.root = Path(__file__).parent / CORPUS_REL_PATH
            
        self._load_corpus()
        
    def _load_corpus(self):
        """Loads all parsed JSON scripts into memory."""
        if not self.root.exists():
            logging.warning(f"🎭 GemmaW: Corpus path not found ({self.root}). Running in FALLBACK mode.")
            return

        json_files = list(self.root.glob("*.json"))
        if not json_files:
            logging.warning(f"🎭 GemmaW: No scripts found in {self.root}.")
            return

        count = 0
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    # Extract dialogue lines
                    # Structure: {"script": [ {"type": "dialogue", "text": "...", "character": "..."}, ... ]}
                    script_blocks = data.get("script", [])
                    for block in script_blocks:
                        if block.get("type") == "dialogue":
                            char = block.get("character")
                            text = block.get("text")
                            if char and text and len(text.split()) > 3: # Ignore short grunts
                                self.corpus_lines.append(f"{char}: {text}")
                count += 1
            except Exception as e:
                logging.warning(f"   [-] Failed to load {jf.name}: {e}")
        
        self.loaded = True
        logging.info(f"🎭 GemmaW: Loaded {len(self.corpus_lines)} lines from {count} scripts.")

    def get_examples(self, k=3):
        """Returns k random dialogue examples from the corpus."""
        if not self.corpus_lines:
            return []
        
        # TODO: In V2, use vector search to find SEMANTICALLY similar lines?
        # For now, random aesthetic sampling.
        return random.sample(self.corpus_lines, min(k, len(self.corpus_lines)))

    def refine(self, draft_line, character="Unknown", context=None):
        """
        Critiques and refines a draft line of dialogue.
        """
        # 1. Fallback Check
        if not self.loaded or not self.engine:
            # If no corpus, maybe just return original?
            # Or do a lightweight "Script Doctor" pass without examples?
            # Let's return original to be safe and fast for public users.
            return draft_line

        # 2. Construct Prompt
        examples = self.get_examples(k=5)
        ex_str = "\n".join([f"   {ex}" for ex in examples])
        
        prompt = (
            f"You are a legendary Screenwriter (e.g. Tarantino, Sorkin). "
            f"Refine this draft line of dialogue to feel like a real movie line.\n\n"
            f"REFERENCE (REAL SCREENPLAY LINES):\n{ex_str}\n\n"
            f"CONTEXT:\n"
            f"   Speaker: {character}\n"
            f"   Draft: \"{draft_line}\"\n"
            f"   Scene Context: {context if context else 'A dramatic scene.'}\n\n"
            f"INSTRUCTION:\n"
            f"- Make it natural, subtext-heavy, or punchy.\n"
            f"- Avoid 'on-the-nose' exposition.\n"
            f"- Match the loose, realistic cadence of the examples.\n"
            f"- output ONLY the refined line text. No quotes, no speaker name.\n"
        )
        
        try:
            # Call Engine
            refined = self.engine.generate(prompt, temperature=0.7)
            if refined:
                refined = refined.strip().strip('"')
                # logging.info(f"   ✨ GW Refined: '{draft_line}' -> '{refined}'")
                return refined
            else:
                return draft_line
        except Exception as e:
            logging.error(f"🎭 GemmaW Error: {e}")
            return draft_line

def main():
    # Simple Test
    print("Initializing Critic...")
    
    # Mock Engine
    class MockEngine:
        def generate(self, p, temperature=0.7, json_schema=None):
            return "Yeah, I heard you the first time."
            
    critic = DialogueCritic(text_engine=MockEngine())
    
    line = "I am very angry at you for betraying me."
    print(f"\nDraft: {line}")
    print(f"Refined: {critic.refine(line, character='Hero')}")

if __name__ == "__main__":
    main()
