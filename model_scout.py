from google import genai
import action
import definitions

def get_keys():
    keys = action.load_action_keys()
    if not keys:
        print("❌ No keys found in env_vars.yaml")
        exit(1)
    return keys

def main():
    keys = get_keys()
    client = genai.Client(api_key=keys[0])
    
    print("\n🔍 MODEL SCOUT: Scanning for L/J/K Candidates...\n")
    
    # Fetch models
    try:
        models = list(client.models.list())
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    # Categories
    gemini_flash_candidates = []
    veo_candidates = []
    others = []

    for m in models:
        # m is a Model object, name e.g. "models/gemini-..."
        name = m.name
        if "gemini" in name and "flash" in name:
            gemini_flash_candidates.append(name)
        elif "veo" in name:
            veo_candidates.append(name)
        else:
            others.append(name)
            
    # Sort for readability
    gemini_flash_candidates.sort(reverse=True) # Newest first (roughly)
    veo_candidates.sort(reverse=True)
    
    print(f"⚡ CANDIDATES FOR 'L' (Lite / Metadata / Fast Gen):")
    for name in gemini_flash_candidates[:5]: # Top 5
        print(f"   - {name}")
        
    print(f"\n🎥 CANDIDATES FOR 'J/K' (Veo Video Gen):")
    for name in veo_candidates:
        print(f"   - {name}")
        
    print("\n---------------------------------------------------")
    print("📋 CURRENT DEFINITIONS (definitions.py):")
    print(f"   L (Light) : {definitions.VIDEO_MODELS.get('L')}")
    print(f"   J (Just)  : {definitions.VIDEO_MODELS.get('J')}")
    print(f"   K (Killer): {definitions.VIDEO_MODELS.get('K')}")
    print("---------------------------------------------------\n")

    # Recommendation Logic
    rec_l = gemini_flash_candidates[0] if gemini_flash_candidates else "None"
    
    # Find best Veo 3.1
    veo_fast = next((m for m in veo_candidates if "veo-3.1" in m and "fast" in m), None)
    veo_full = next((m for m in veo_candidates if "veo-3.1" in m and "fast" not in m), None)
    
    # Fallback to Veo 3.0
    if not veo_fast: veo_fast = next((m for m in veo_candidates if "veo-3" in m and "fast" in m), "None")
    if not veo_full: veo_full = next((m for m in veo_candidates if "veo-3" in m and "fast" not in m), "None")

    print(f"   Suggested K: {veo_full}")
    
    # Check alignment
    current_l = definitions.VIDEO_MODELS.get('L', '')
    current_j = definitions.VIDEO_MODELS.get('J', '')
    current_k = definitions.VIDEO_MODELS.get('K', '')
    
    if current_l == rec_l and current_j == veo_fast and current_k == veo_full:
        print("\n✅ STATUS: GREEN. Definitions are optimized.")
    else:
        print("\n⚠️ STATUS: YELLOW. Updates available.")
        if current_l != rec_l: print(f"   - Update L: {current_l} -> {rec_l}")
        if current_j != veo_fast: print(f"   - Update J: {current_j} -> {veo_fast}")
        if current_k != veo_full: print(f"   - Update K: {current_k} -> {veo_full}")

def probe_model(keys, model_name):
    """
    Stress tests a model to find empirical rate limits.
    """
    print(f"\n🧪 PROBING MODEL: {model_name}")
    print("   Starting stress test (5 attempts, 1s delay)...")
    
    import requests
    import json
    import time
    
    success_count = 0
    fail_count = 0
    timings = []
    
    # We use the first key for probing to test SINGLE KEY limits
    api_key = keys[0]
    
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    # Clean model name
    clean_name = model_name.replace("models/", "")
    
    # Determine endpoint type
    is_veo = "veo" in clean_name.lower()
    
    for i in range(1, 6):
        print(f"   👉 Attempt {i}/5...", end="", flush=True)
        start_t = time.time()
        
        try:
            if is_veo:
                # Veo Probe
                url = f"{base_url}/{clean_name}:predictLongRunning?key={api_key}"
                payload = {"instances": [{"prompt": f"A probing test video of a spinning wireframe cube. Test ID {i}."}]}
                headers = {"Content-Type": "application/json"}
                
                res = requests.post(url, json=payload, headers=headers)
                
                if res.status_code == 200:
                    print(" ✅ 200 OK (Started)")
                    success_count += 1
                elif res.status_code == 429:
                    print(" ⛔ 429 RATE LIMIT")
                    fail_count += 1
                else:
                    print(f" ❌ {res.status_code} {res.reason}")
                    fail_count += 1
                    
            else:
                # Gemini Probe
                url = f"{base_url}/{clean_name}:generateContent?key={api_key}"
                payload = {"contents": [{"parts": [{"text": f"Just say 'Test {i}'"}]}]}
                headers = {"Content-Type": "application/json"}
                
                res = requests.post(url, json=payload, headers=headers)
                
                if res.status_code == 200:
                    print(" ✅ 200 OK")
                    success_count += 1
                elif res.status_code == 429:
                    print(" ⛔ 429 RATE LIMIT")
                    fail_count += 1
                else:
                    print(f" ❌ {res.status_code} Error")
                    fail_count += 1

        except Exception as e:
            print(f" 💥 Exception: {e}")
            fail_count += 1
            
        elapsed = time.time() - start_t
        timings.append(elapsed)
        time.sleep(1) # Mild aggression
        
    print("\n📊 PROBE RESULTS:")
    print(f"   Success: {success_count}/5")
    print(f"   Failures: {fail_count}/5")
    print(f"   Avg Response Time: {sum(timings)/len(timings):.2f}s")
    
    if fail_count > 0:
        print("   ⚠️ RATE LIMIT HIT. This model has strict quotas.")
    else:
        print("   ✅ STABLE. No rate limits active in this short burst.")
        
    print("   NOTE: Cost is usually determined by output duration/pixels. Check Google AI Pricing page.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=str, help="Model name to probe (e.g., veo-2.0-generate-001)")
    args = parser.parse_args()
    
    if args.probe:
        keys = get_keys()
        probe_model(keys, args.probe)
    else:
        main()
