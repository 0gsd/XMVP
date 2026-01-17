#!/usr/bin/env python3
import sys
import subprocess

def install_and_check():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi

    api = HfApi()
    print("Searching for 'gemma-3'...")
    models = api.list_models(search="gemma-3", limit=5)
    found = False
    for m in models:
        print(f"- {m.id}")
        if "google/gemma-3" in m.id:
            found = True
    
    if not found:
        print("No official 'google/gemma-3' found. Searching 'gemma-2'...")
        models = api.list_models(search="gemma-2", limit=5)
        for m in models:
            print(f"- {m.id}")

if __name__ == "__main__":
    install_and_check()
