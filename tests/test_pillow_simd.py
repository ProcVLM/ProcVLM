"""
python grounding/tests/test_pillow_simd.py
"""

import time
import numpy as np
from PIL import Image

def benchmark():
    # generate a large random image
    print("Generating random image (4000x4000)...")
    arr = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    # warm up
    img.resize((512, 512), resample=Image.Resampling.LANCZOS)
    
    # start benchmark
    # Using BICUBIC interpolation because it is computationally intensive and best demonstrates SIMD acceleration
    # VLM training usually resizes to 224, 336, or 448 sizes
    N = 20
    print(f"Running {N} resize operations (LANCZOS)...")
    
    start_time = time.time()
    for _ in range(N):
        # force resize
        _ = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / N
    print(f"\nResult:")
    print(f"Total time: {end_time - start_time:.4f}s")
    print(f"Average time per image: {avg_time:.4f}s")

    # Simple judgment criteria
    print("-" * 30)
    if avg_time < 0.15:
        print("🚀 Speed: VERY FAST (Likely Pillow-SIMD)")
    elif avg_time < 0.4:
        print("⚠️ Speed: MODERATE (Could be standard Pillow)")
    else:
        print("🐢 Speed: SLOW (Definitely standard Pillow or slow CPU)")

if __name__ == "__main__":
    try:
        import PIL
        print(f"Testing PIL version: {PIL.__version__}")
        benchmark()
    except ImportError:
        print("Please install pillow or pillow-simd first.")