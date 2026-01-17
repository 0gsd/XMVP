from google import genai
import definitions
from mvp_shared import load_api_keys
from definitions import Modality, BackendType, MODAL_REGISTRY, get_active_model, set_active_model
import argparse
import os
import sys

def get_keys():
    keys = load_api_keys()
    if not keys:
        print("❌ No keys found in env_vars.yaml")
        sys.exit(1)
    return keys

def print_status():
    print("\n📊 MODAL REGISTRY STATUS")
    print(f"{'MODALITY':<15} {'ACTIVE MODEL':<25} {'BACKEND':<10} {'DETAILS':<30}")
    print("-" * 80)
    
    for mod in Modality:
        try:
            config = get_active_model(mod)
            details = config.endpoint if config.backend == BackendType.CLOUD else config.path
            # Truncate details if too long
            if details and len(str(details)) > 28:
                details = "..." + str(details)[-25:]
            elif not details:
                details = "-"
                
            print(f"{mod.value:<15} {config.name:<25} {config.backend.value:<10} {details:<30}")
        except Exception as e:
            print(f"{mod.value:<15} {'ERROR':<25} {'-':<10} {str(e):<30}")
    print("-" * 80)
    print(f"Active Profile: {definitions.ACTIVE_PROFILE_PATH.resolve()}")

def list_models(modality_str=None):
    mods_to_list = [Modality(modality_str)] if modality_str else Modality
    
    for mod in mods_to_list:
        print(f"\n📂 {mod.value.upper()} Models:")
        if mod not in MODAL_REGISTRY:
            print("   (No models registered)")
            continue
            
        for mid, config in MODAL_REGISTRY[mod].items():
            active_marker = "*" if mid == definitions.ACTIVE_PROFILE.get(mod) else " "
            print(f" {active_marker} [{mid}] ({config.backend.value})")
            if config.path: print(f"      Path: {config.path}")
            if config.endpoint: print(f"      Endpoint: {config.endpoint}")

def switch_model(modality_str, model_id):
    try:
        mod = Modality(modality_str)
        set_active_model(mod, model_id)
        print(f"✅ Switched {mod.value} to '{model_id}'")
        print_status()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Use --list to see valid options.")

def run_scan():
    keys = get_keys()
    client = genai.Client(api_key=keys[0])
    
    print("\n🔍 CLOUD SCAN: Listing available Gemini models...\n")
    try:
        models = list(client.models.list())
        for m in models:
            print(f" - {m.name}")
    except Exception as e:
        print(f"❌ API Error: {e}")

# --- Legacy Probe/Pull ---
def probe_model(keys, model_name):
    """Stress tests a model."""
    print(f"\n🧪 PROBING MODEL: {model_name}")
    import requests
    import time
    
    api_key = keys[0]
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    clean_name = model_name.replace("models/", "")
    is_veo = "veo" in clean_name.lower()
    
    success_count = 0
    fail_count = 0
    
    for i in range(1, 6):
        print(f"   👉 Attempt {i}/5...", end="", flush=True)
        try:
            if is_veo:
                url = f"{base_url}/{clean_name}:predictLongRunning?key={api_key}"
                payload = {"instances": [{"prompt": f"Probe test {i}"}]}
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            else:
                url = f"{base_url}/{clean_name}:generateContent?key={api_key}"
                payload = {"contents": [{"parts": [{"text": "Test"}]}]}
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            if res.status_code == 200:
                print(" ✅ 200 OK")
                success_count += 1
            elif res.status_code == 429:
                print(" ⛔ 429 RATE LIMIT")
                fail_count += 1
            else:
                print(f" ❌ {res.status_code}")
                fail_count += 1
        except Exception as e:
            print(f" 💥 {e}")
            fail_count += 1
        time.sleep(1)

def pull_model(model_name, dest_dir="/Volumes/ORICO/1_o_gemmas"):
    """Downloads/Converts HF model via mlx-lm."""
    print(f"\n📦 MODEL PULL: {model_name} -> {dest_dir}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    cmd = [
        "python3", "-m", "mlx_lm.convert",
        "--hf-path", model_name,
        "--q-bits", "4",
        "--output", os.path.join(dest_dir, model_name.split("/")[-1])
    ]
    print(f"   > {' '.join(cmd)}")
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMVP Model Scout & Registry Manager")
    
    # Registry Commands
    parser.add_argument("--status", action="store_true", help="Show current active models")
    parser.add_argument("--list", type=str, nargs="?", const="ALL", help="List models for modality (or ALL)")
    parser.add_argument("--switch", nargs=2, metavar=('MODALITY', 'MODEL_ID'), help="Switch active model")
    
    # Cloud/Ops Commands
    parser.add_argument("--scan", action="store_true", help="Scan cloud for available models")
    parser.add_argument("--probe", type=str, help="Probe a specific model for rate limits")
    parser.add_argument("--pull", type=str, help="Download HF model via MLX")
    
    args = parser.parse_args()
    
    if args.status:
        print_status()
    elif args.list:
        mod = None if args.list == "ALL" else args.list
        list_models(mod)
    elif args.switch:
        switch_model(args.switch[0], args.switch[1])
    elif args.scan:
        run_scan()
    elif args.probe:
        probe_model(get_keys(), args.probe)
    elif args.pull:
        pull_model(args.pull)
    else:
        # Default to status
        print_status()
