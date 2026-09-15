"""
Dental AI v2.0 — Phase 1: Data Preparation & CLAHE Preprocessing
=================================================================
Prepares all 3 datasets (DentalAI, DENTEX, TUFTS) for proper Q1 training:
  1. Parses annotations → binary masks / disease labels
  2. Applies CLAHE contrast enhancement
  3. Resizes to 512×512 with aspect-ratio-preserving letterbox
  4. Creates train/val/test splits with reproducible manifest
  5. Caches preprocessed tensors for fast DataLoader

Usage:
    python scripts/prepare_dataset_v2.py
"""

import os
import sys
import json
import csv
import hashlib
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(r"j:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
DATA_RAW = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed_v2"

DENTALAI_ROOT = DATA_RAW / "dentalai"
DENTEX_ROOT = DATA_RAW / "dentex"
TUFTS_ROOT = DATA_RAW / "tufts"

IMG_SIZE = 512
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
SEED = 42

# DENTEX disease mapping (from COCO categories_3)
DISEASE_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "DeepCaries"}


def apply_clahe(image_bgr):
    """Apply CLAHE to a BGR image on the L channel of LAB color space."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def letterbox_resize(image, target_size, fill_value=0):
    """Resize image preserving aspect ratio with letterbox padding."""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas
    if len(image.shape) == 3:
        canvas = np.full((target_size, target_size, image.shape[2]), fill_value, dtype=image.dtype)
    else:
        canvas = np.full((target_size, target_size), fill_value, dtype=image.dtype)

    # Center the image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def letterbox_resize_mask(mask, target_size):
    """Resize binary mask with nearest-neighbor interpolation."""
    h, w = mask.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_size, target_size), dtype=mask.dtype)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    if len(resized.shape) == 2:
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    else:
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized[:, :, 0]
    return canvas


# ── 1. DentalAI Processing ────────────────────────────────────────────

def parse_supervisely_annotation(json_path, img_shape):
    """Parse Supervisely JSON annotation to binary tooth mask."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(json_path, "r") as f:
        ann = json.load(f)

    for obj in ann.get("objects", []):
        if obj.get("geometryType") == "polygon" and obj.get("classTitle") == "Tooth":
            exterior = obj["points"]["exterior"]
            pts = np.array(exterior, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


def process_dentalai():
    """Process all DentalAI splits into CLAHE-enhanced images + binary masks."""
    print("\n" + "=" * 60)
    print("Processing DentalAI (Tooth Segmentation)")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "dentalai"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for split in ["train", "valid", "test"]:
        img_dir = DENTALAI_ROOT / split / "img"
        ann_dir = DENTALAI_ROOT / split / "ann"

        if not img_dir.exists():
            print(f"  [SKIP] {split}: {img_dir} not found")
            continue

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {split}: {len(img_files)} images found")

        for i, img_name in enumerate(img_files):
            img_path = img_dir / img_name
            ann_path = ann_dir / (img_name + ".json")

            # Read image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"    [WARN] Cannot read: {img_path}")
                continue

            # Parse annotation to mask
            if ann_path.exists():
                mask = parse_supervisely_annotation(str(ann_path), img_bgr.shape)
            else:
                mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

            # Apply CLAHE
            img_enhanced = apply_clahe(img_bgr)

            # Letterbox resize
            img_resized = letterbox_resize(img_enhanced, IMG_SIZE, fill_value=0)
            mask_resized = letterbox_resize_mask(mask, IMG_SIZE)

            # Binarize mask
            mask_binary = (mask_resized > 127).astype(np.uint8) * 255

            # Save
            uid = hashlib.md5(img_name.encode()).hexdigest()[:12]
            img_out = out_dir / f"{uid}_img.png"
            mask_out = out_dir / f"{uid}_mask.png"
            cv2.imwrite(str(img_out), img_resized)
            cv2.imwrite(str(mask_out), mask_binary)

            tooth_pixels = int(np.sum(mask_binary > 0))
            all_records.append({
                "uid": uid,
                "source": "dentalai",
                "original_split": split,
                "original_name": img_name,
                "img_path": str(img_out),
                "mask_path": str(mask_out),
                "tooth_pixels": tooth_pixels,
                "has_teeth": 1 if tooth_pixels > 100 else 0,
            })

            if (i + 1) % 200 == 0:
                print(f"    Processed {i + 1}/{len(img_files)}")

    print(f"  Total DentalAI records: {len(all_records)}")
    return all_records


# ── 2. DENTEX Processing ──────────────────────────────────────────────

def process_dentex():
    """Process DENTEX QED images with disease annotations."""
    print("\n" + "=" * 60)
    print("Processing DENTEX (Disease Classification)")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "dentex"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    qed_json = DENTEX_ROOT / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
    xray_dir = DENTEX_ROOT / "training_data" / "quadrant-enumeration-disease" / "xrays"

    with open(qed_json, "r") as f:
        coco = json.load(f)

    # Build image id → filename map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}

    # Build image id → annotations map
    img_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        img_annotations[ann["image_id"]].append(ann)

    all_records = []

    for img_id, filename in sorted(id_to_file.items()):
        img_path = xray_dir / filename
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # Apply CLAHE
        img_enhanced = apply_clahe(img_bgr)

        # Letterbox resize
        img_resized = letterbox_resize(img_enhanced, IMG_SIZE, fill_value=0)

        # Disease labels (multi-label binary vector)
        # [Impacted, Caries, Periapical, DeepCaries]
        disease_vector = [0, 0, 0, 0]
        tooth_bboxes = []

        for ann in img_annotations.get(img_id, []):
            cat3 = ann["category_id_3"]
            if 0 <= cat3 <= 3:
                disease_vector[cat3] = 1

            # Store bbox for tooth ROI extraction
            bbox = ann["bbox"]  # [x, y, w, h]
            tooth_bboxes.append({
                "bbox": bbox,
                "disease_id": cat3,
                "disease_name": DISEASE_NAMES.get(cat3, "Unknown"),
                "segmentation": ann.get("segmentation", []),
            })

        # Create tooth segmentation mask from polygon annotations
        h, w = img_bgr.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in img_annotations.get(img_id, []):
            for seg_poly in ann.get("segmentation", []):
                pts = np.array(seg_poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(seg_mask, [pts], 255)

        mask_resized = letterbox_resize_mask(seg_mask, IMG_SIZE)
        mask_binary = (mask_resized > 127).astype(np.uint8) * 255

        # Save
        uid = f"dentex_{img_id:04d}"
        img_out = out_dir / f"{uid}_img.png"
        mask_out = out_dir / f"{uid}_mask.png"
        cv2.imwrite(str(img_out), img_resized)
        cv2.imwrite(str(mask_out), mask_binary)

        all_records.append({
            "uid": uid,
            "source": "dentex",
            "original_name": filename,
            "img_path": str(img_out),
            "mask_path": str(mask_out),
            "disease_impacted": disease_vector[0],
            "disease_caries": disease_vector[1],
            "disease_periapical": disease_vector[2],
            "disease_deep_caries": disease_vector[3],
            "num_annotations": len(img_annotations.get(img_id, [])),
            "has_disease": 1 if sum(disease_vector) > 0 else 0,
        })

    print(f"  Total DENTEX records: {len(all_records)}")

    # Disease stats
    for i, name in DISEASE_NAMES.items():
        count = sum(1 for r in all_records if r.get(f"disease_{name.lower()}", 0) == 1)
        key = ["disease_impacted", "disease_caries", "disease_periapical", "disease_deep_caries"][i]
        count = sum(1 for r in all_records if r.get(key, 0) == 1)
        print(f"    {name}: {count} images")

    return all_records


# ── 3. TUFTS Processing ──────────────────────────────────────────────

def process_tufts():
    """Process TUFTS external holdout (radiographs + tooth masks)."""
    print("\n" + "=" * 60)
    print("Processing TUFTS (External Holdout)")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "tufts"
    out_dir.mkdir(parents=True, exist_ok=True)

    rad_dir = TUFTS_ROOT / "Radiographs" / "Radiographs"
    mask_dir = TUFTS_ROOT / "Segmentation" / "Segmentation" / "teeth_mask"

    if not rad_dir.exists():
        print(f"  [ERROR] Radiographs not found: {rad_dir}")
        return []

    rad_files = sorted([f for f in os.listdir(rad_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Radiographs found: {len(rad_files)}")

    all_records = []

    for i, rad_name in enumerate(rad_files):
        rad_path = rad_dir / rad_name

        img_bgr = cv2.imread(str(rad_path))
        if img_bgr is None:
            continue

        # Apply CLAHE
        img_enhanced = apply_clahe(img_bgr)
        img_resized = letterbox_resize(img_enhanced, IMG_SIZE, fill_value=0)

        # Try to find matching mask
        # TUFTS mask filenames may match by number
        stem = Path(rad_name).stem  # e.g., "1" or "100"
        mask_path = None
        for ext in [".jpg", ".png", ".JPG", ".PNG"]:
            candidate = mask_dir / (stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path and mask_path.exists():
            mask_bgr = cv2.imread(str(mask_path))
            if mask_bgr is not None:
                mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                mask_resized = letterbox_resize_mask(mask_gray, IMG_SIZE)
                mask_binary = (mask_resized > 127).astype(np.uint8) * 255
            else:
                mask_binary = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            mask_binary = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Save
        uid = f"tufts_{int(stem):04d}" if stem.isdigit() else f"tufts_{stem}"
        img_out = out_dir / f"{uid}_img.png"
        mask_out = out_dir / f"{uid}_mask.png"
        cv2.imwrite(str(img_out), img_resized)
        cv2.imwrite(str(mask_out), mask_binary)

        tooth_pixels = int(np.sum(mask_binary > 0))
        all_records.append({
            "uid": uid,
            "source": "tufts",
            "original_name": rad_name,
            "img_path": str(img_out),
            "mask_path": str(mask_out),
            "tooth_pixels": tooth_pixels,
            "has_teeth": 1 if tooth_pixels > 100 else 0,
        })

        if (i + 1) % 200 == 0:
            print(f"    Processed {i + 1}/{len(rad_files)}")

    non_empty = sum(1 for r in all_records if r["has_teeth"])
    print(f"  Total TUFTS records: {len(all_records)} ({non_empty} with teeth masks)")
    return all_records


# ── 4. Split & Manifest ──────────────────────────────────────────────

def create_splits(dentalai_records, dentex_records, tufts_records):
    """Create reproducible train/val/test splits and save manifest CSV."""
    print("\n" + "=" * 60)
    print("Creating Train/Val/Test Splits")
    print("=" * 60)

    np.random.seed(SEED)

    # ── DentalAI: 80/10/10 split ──
    dai = dentalai_records.copy()
    np.random.shuffle(dai)
    n = len(dai)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    for r in dai[:n_train]:
        r["split"] = "train"
    for r in dai[n_train:n_train + n_val]:
        r["split"] = "val"
    for r in dai[n_train + n_val:]:
        r["split"] = "test"

    dai_train = sum(1 for r in dai if r["split"] == "train")
    dai_val = sum(1 for r in dai if r["split"] == "val")
    dai_test = sum(1 for r in dai if r["split"] == "test")
    print(f"  DentalAI: train={dai_train}, val={dai_val}, test={dai_test}")

    # ── DENTEX: 78/7/15 split (keep ~107 for test to match paper) ──
    dtx = dentex_records.copy()
    np.random.shuffle(dtx)
    n = len(dtx)
    n_test = 107  # Match the original paper's test count
    n_val = 51    # Match DENTEX official validation count
    n_train = n - n_test - n_val

    for r in dtx[:n_train]:
        r["split"] = "train"
    for r in dtx[n_train:n_train + n_val]:
        r["split"] = "val"
    for r in dtx[n_train + n_val:]:
        r["split"] = "test"

    dtx_train = sum(1 for r in dtx if r["split"] == "train")
    dtx_val = sum(1 for r in dtx if r["split"] == "val")
    dtx_test = sum(1 for r in dtx if r["split"] == "test")
    print(f"  DENTEX: train={dtx_train}, val={dtx_val}, test={dtx_test}")

    # Disease distribution in test set
    for key in ["disease_impacted", "disease_caries", "disease_periapical", "disease_deep_caries"]:
        count = sum(1 for r in dtx if r["split"] == "test" and r.get(key, 0) == 1)
        print(f"    Test {key}: {count}")

    # ── TUFTS: all external holdout ──
    for r in tufts_records:
        r["split"] = "external_holdout"
    print(f"  TUFTS: external_holdout={len(tufts_records)}")

    # ── Save manifest CSV ──
    all_records = dai + dtx + tufts_records
    manifest_path = OUTPUT_DIR / "dataset_manifest_v2.csv"

    # Collect all possible keys
    all_keys = set()
    for r in all_records:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n  Manifest saved: {manifest_path}")
    print(f"  Total records: {len(all_records)}")

    # Summary
    print("\n  === FINAL SPLIT SUMMARY ===")
    for split in ["train", "val", "test", "external_holdout"]:
        count = sum(1 for r in all_records if r["split"] == split)
        sources = defaultdict(int)
        for r in all_records:
            if r["split"] == split:
                sources[r["source"]] += 1
        src_str = ", ".join(f"{k}={v}" for k, v in sorted(sources.items()))
        print(f"    {split}: {count} ({src_str})")

    return manifest_path


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Dental AI v2.0 — Data Preparation Pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"CLAHE: clipLimit={CLAHE_CLIP}, tile={CLAHE_TILE}x{CLAHE_TILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    dentalai_records = process_dentalai()
    dentex_records = process_dentex()
    tufts_records = process_tufts()

    # Create splits and manifest
    manifest_path = create_splits(dentalai_records, dentex_records, tufts_records)

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE!")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)
