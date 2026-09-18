"""
BDD100k Image Quality Statistics & Visual Properties Extractor
Calculates:
  - Brightness: Mean luminance (mu_L = 1/N * sum(L_i))
  - Contrast: Standard deviation of luminance (sigma_L)
  - Blur: Variance of the Laplacian (Var(Laplacian(L)))
  - Exposure:
      * Underexposed pixel percentage (L <= 10)
      * Overexposed pixel percentage (L >= 245)
Integrates metadata:
  - Weather (clear, rainy, overcast, snowy, partly cloudy, foggy, undefined)
  - Time of day (daytime, night, dawn/dusk, undefined)
  - Split (train, val, test)
"""

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import pandas as pd

# Define paths
WORKSPACE_DIR = Path("g:/project")
BASE_IMAGE_DIR = WORKSPACE_DIR / "archive" / "bdd100k" / "bdd100k" / "images" / "10k"
LABELS_DIR = WORKSPACE_DIR / "archive" / "bdd100k_labels_release" / "bdd100k" / "labels"
VAL_JSON_PATH = LABELS_DIR / "bdd100k_labels_images_val.json"
TRAIN_JSON_PATH = LABELS_DIR / "bdd100k_labels_images_train.json"
OUTPUT_DIR = WORKSPACE_DIR / "results"
OUTPUT_CSV = OUTPUT_DIR / "bdd100k_10k_image_quality.csv"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata():
    """
    Loads attributes from val.json and train.json.
    Supports exact matching and video prefix fallback.
    """
    print("Loading label metadata...")
    exact_map = {}
    prefix_map = {}

    # 1. Load val.json
    if VAL_JSON_PATH.exists():
        print(f"Reading {VAL_JSON_PATH.name}...")
        with open(VAL_JSON_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        for item in val_data:
            name = item.get("name")
            attrs = item.get("attributes", {})
            exact_map[name] = attrs
            prefix = name.split("-")[0]
            if prefix not in prefix_map:
                prefix_map[prefix] = attrs

    # 2. Fast scan train.json
    if TRAIN_JSON_PATH.exists():
        print(f"Scanning {TRAIN_JSON_PATH.name}...")
        current_name = None
        current_attr = {}
        in_attributes = False

        with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if '"name":' in line and '"box2d"' not in line:
                    parts = line.split('"name":')
                    if len(parts) > 1:
                        sub = parts[1].strip()
                        if sub.startswith('"'):
                            current_name = sub.split('"')[1]
                            current_attr = {}
                elif current_name and '"attributes":' in line:
                    in_attributes = True
                elif in_attributes:
                    if '"weather":' in line:
                        current_attr["weather"] = line.split('"weather":')[1].strip().strip('", \t\r\n')
                    elif '"timeofday":' in line:
                        current_attr["timeofday"] = line.split('"timeofday":')[1].strip().strip('", \t\r\n')
                    elif '"scene":' in line:
                        current_attr["scene"] = line.split('"scene":')[1].strip().strip('", \t\r\n')
                    if "}" in line:
                        in_attributes = False
                        exact_map[current_name] = current_attr
                        prefix = current_name.split("-")[0]
                        if prefix not in prefix_map:
                            prefix_map[prefix] = current_attr
                        current_name = None

    print(f"Loaded {len(exact_map)} exact image annotations and {len(prefix_map)} video prefix annotations.")
    return exact_map, prefix_map


def compute_image_metrics(img_info):
    """
    Computes quality metrics for a single image.
    """
    file_path, file_name, split, exact_map, prefix_map = img_info

    # Read image
    img = cv2.imread(str(file_path))
    if img is None:
        return None

    h, w, c = img.shape
    total_pixels = float(h * w)

    # Convert to Grayscale Luminance (ITU-R BT.601 standard)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)

    # 1. Brightness: Mean luminance mu_L
    mean_luminance = float(np.mean(gray_f))

    # 2. Contrast: Standard deviation of luminance sigma_L
    contrast = float(np.std(gray_f))

    # 3. Blur: Variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_laplacian_var = float(laplacian.var())

    # 4. Exposure:
    # Underexposed pixels: L <= 10
    underexposed_count = int(np.count_nonzero(gray <= 10))
    underexposed_pct = (underexposed_count / total_pixels) * 100.0

    # Overexposed pixels: L >= 245
    overexposed_count = int(np.count_nonzero(gray >= 245))
    overexposed_pct = (overexposed_count / total_pixels) * 100.0

    # Additional standard exposure indicators for robustness:
    # Severely clipped blacks (L <= 5) and blown highlights (L >= 250)
    underexposed_severe_pct = (int(np.count_nonzero(gray <= 5)) / total_pixels) * 100.0
    overexposed_severe_pct = (int(np.count_nonzero(gray >= 250)) / total_pixels) * 100.0

    # Percentiles of luminance
    p05, p25, p50, p75, p95 = np.percentile(gray, [5, 25, 50, 75, 95])

    # Metadata resolution
    attrs = exact_map.get(file_name)
    match_type = "exact"
    if attrs is None:
        prefix = file_name.split("-")[0]
        attrs = prefix_map.get(prefix)
        match_type = "prefix" if attrs else "none"

    weather = attrs.get("weather", "unlabeled") if attrs else "unlabeled"
    timeofday = attrs.get("timeofday", "unlabeled") if attrs else "unlabeled"
    scene = attrs.get("scene", "unlabeled") if attrs else "unlabeled"

    return {
        "image_name": file_name,
        "split": split,
        "width": w,
        "height": h,
        "mean_luminance": round(mean_luminance, 4),
        "contrast_std": round(contrast, 4),
        "blur_laplacian_var": round(blur_laplacian_var, 4),
        "underexposed_pct": round(underexposed_pct, 4),
        "overexposed_pct": round(overexposed_pct, 4),
        "underexposed_severe_pct": round(underexposed_severe_pct, 4),
        "overexposed_severe_pct": round(overexposed_severe_pct, 4),
        "luminance_p05": round(float(p05), 2),
        "luminance_p25": round(float(p25), 2),
        "luminance_median": round(float(p50), 2),
        "luminance_p75": round(float(p75), 2),
        "luminance_p95": round(float(p95), 2),
        "weather": weather,
        "timeofday": timeofday,
        "scene": scene,
        "match_type": match_type,
        "file_path": str(file_path.relative_to(WORKSPACE_DIR)).replace("\\", "/")
    }


def main():
    start_time = time.time()
    print("=" * 70)
    print("Starting BDD100k Image Quality Statistics Extraction")
    print("=" * 70)

    # 1. Load Metadata
    exact_map, prefix_map = load_metadata()

    # 2. Gather image files across splits
    all_images = []
    for split in ["train", "val", "test"]:
        split_dir = BASE_IMAGE_DIR / split
        if split_dir.exists():
            files = [f for f in split_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            print(f"Found {len(files)} images in split '{split}'")
            for f in files:
                all_images.append((f, f.name, split, exact_map, prefix_map))

    total_images = len(all_images)
    print(f"\nTotal images to process: {total_images}")

    # 3. Process images in parallel
    results = []
    workers = min(12, os.cpu_count() or 4)
    print(f"Processing with {workers} worker threads...")

    processed_count = 0
    t_prev = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(compute_image_metrics, img_info): img_info[1] for img_info in all_images}

        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_images:
                elapsed = time.time() - t_prev
                rate = 1000.0 / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed_count}/{total_images} images ({processed_count/total_images*100:.1f}%) | Speed: {rate:.1f} imgs/s")
                t_prev = time.time()

    # 4. Save results to DataFrame and CSV
    df = pd.DataFrame(results)
    df.sort_values(by=["split", "image_name"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)

    total_time = time.time() - start_time
    print(f"\nSuccessfully extracted quality statistics for {len(df)} images.")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total time elapsed: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")
    print("=" * 70)


if __name__ == "__main__":
    main()
