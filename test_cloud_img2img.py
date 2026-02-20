#!/usr/bin/env python3
"""Quick test: Does HF Cloud Flux Img2Img actually work?"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from flux_bridge import generate_via_hf_endpoint
from PIL import Image

# Create a simple test image (red square)
test_img = Image.new("RGB", (480, 320), color="red")

print("🧪 Test 1: Cloud Text-to-Image...")
t2i = generate_via_hf_endpoint("A serene mountain lake at sunset", width=480, height=320, steps=12)
if t2i:
    t2i.save("test_cloud_t2i.png")
    print(f"   ✅ T2I worked! Saved test_cloud_t2i.png ({t2i.size})")
else:
    print("   ❌ T2I failed")
    sys.exit(1)

print("\n🧪 Test 2: Cloud Img2Img (using T2I result as input)...")
i2i = generate_via_hf_endpoint("Same lake but now it's nighttime with stars", width=480, height=320, steps=12, image=t2i, strength=0.65)
if i2i:
    i2i.save("test_cloud_i2i.png")
    print(f"   ✅ Img2Img worked! Saved test_cloud_i2i.png ({i2i.size})")
    print("   Compare test_cloud_t2i.png and test_cloud_i2i.png — they should look related!")
else:
    print("   ❌ Img2Img failed (fell back to T2I or errored)")
