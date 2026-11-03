"""
Extract and cache all 3,529 tooth ROI crops directly from raw radiographs
to outputs/v6_gold_standard/crops/ for lightning-fast training and zero I/O delays.
"""

import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
OUT_CROPS = ROOT / "outputs" / "v6_gold_standard" / "crops"
OUT_CROPS.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = ROOT / "data" / "processed_v2" / "tooth_roi_manifest_v2.csv"
RAW_DIR = ROOT / "data" / "raw" / "dentex" / "training_data" / "quadrant-enumeration-disease" / "xrays"


def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} tooth ROIs")
    
    # Pre-cache raw images in RAM dictionary to avoid repeated disk reads
    unique_xrays = df["original_name"].unique()
    print(f"Loading {len(unique_xrays)} unique raw radiographs into RAM cache...")
    raw_cache = {}
    for x_name in unique_xrays:
        xp = RAW_DIR / x_name
        if xp.exists():
            img = cv2.imread(str(xp))
            if img is not None:
                raw_cache[x_name] = img
    print(f"Successfully cached {len(raw_cache)} raw radiographs in memory!")
    
    saved_count = 0
    t0 = time.time()
    for idx, row in df.iterrows():
        roi_uid = row["roi_uid"]
        out_p = OUT_CROPS / f"{roi_uid}.png"
        if out_p.exists():
            saved_count += 1
            continue
            
        x_name = row["original_name"]
        raw_img = raw_cache.get(x_name)
        if raw_img is None:
            continue
            
        y1, y2 = int(row["crop_y1"]), int(row["crop_y2"])
        x1, x2 = int(row["crop_x1"]), int(row["crop_x2"])
        crop = raw_img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue
            
        crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_p), crop_256)
        saved_count += 1
        
        if saved_count % 500 == 0 or saved_count == len(df):
            print(f"Extracted {saved_count}/{len(df)} crops ({time.time()-t0:.1f}s)...")
            
    print(f"Pre-extraction complete: {saved_count} crops stored in {OUT_CROPS}")


if __name__ == "__main__":
    main()
