#!/usr/bin/env python3
import argparse
import logging
import sys
import os

# Import MVP Text Engine
from text_engine import get_engine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carbonate_prompt(title, artist=None, extra_context=None):
    """
    Expands a Song Title into a SASSPRILLA-maxed Situation Prompt.
    """
    engine = get_engine()
    
    # Construct the Carbonation Prompt
    base_prompt = f"Song Title: '{title}'"
    if artist: base_prompt += f"\nArtist: '{artist}'"
    if extra_context: base_prompt += f"\nContext: '{extra_context}'"
    
    system_instruction = (
        "You are the SASSPRILLA CARBONATOR, a specialized creative engine designed to expand "
        "minimalist song titles into dense, vivid, and culturally resonant music video concepts.\n\n"
        "YOUR TASK:\n"
        "1. ANALYZE the Title: Extract its metaphorical weight, potential themes (Cyberpunk, Noir, Abstract, Emotional), and hidden irony.\n"
        "2. EXTRAPOLATE a Purpose: What is the 'Lesson' or 'thesis' of this video?\n"
        "3. GENERATE a 'Situation' Prompt: Write a single, highly detailed paragraph describing the music video's core concept. "
        "Use 'Sassprilla' density: specific imagery, lighting codes, camera movements, and emotional textures.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY the final prompt text. Do not include 'Here is the prompt:' or markdown blocks. Just the raw, carbonated text ready for production."
    )
    
    logging.info(f"🫧 Carbonating '{title}'...")
    
    try:
        # Use a high temperature for creativity
        response = engine.generate(
            f"{system_instruction}\n\nINPUT:\n{base_prompt}\n\nOUTPUT PROMPT:",
            temperature=0.8
        )
        return response.strip()
    except Exception as e:
        logging.error(f"Carbonation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="SASSPRILLA CARBONATOR: Prompt Expander")
    parser.add_argument("title", help="Song Title")
    parser.add_argument("--artist", help="Artist Name")
    parser.add_argument("--context", help="Additional context (e.g. 'Cyberpunk', 'Slow', 'Sad')")
    parser.add_argument("--run", action="store_true", help="Execute movie_producer automatically (Experimental)")
    
    args = parser.parse_args()
    
    # Force Local Text Engine if defined in env? 
    # movie_producer usually sets it. We assume env is set or defaults are fine.
    
    prompt = carbonate_prompt(args.title, args.artist, args.context)
    
    if prompt:
        print("\n" + "="*40)
        print("✨ CARBONATED PROMPT ✨")
        print("="*40)
        print(prompt)
        print("="*40 + "\n")
        
        # Output purely the prompt to stdout if piped?
        # But we printed headers.
        # If user wants pipe, we should maybe add --quiet flag.
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
