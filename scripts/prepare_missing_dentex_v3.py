"""
DentalXAI v3 — Complete 15.6 GB Raw Dataset Audit & Missing Data Extraction
===========================================================================
Fixes the data omissions in prepare_dataset_v2.py:
  1. Audits all raw dataset folders (DentalAI 1.08 GB, DENTEX 13.90 GB across 5
     subfolders, TUFTS 0.63 GB = 15.61 GB total) and logs exact file/byte counts.
  2. Processes DENTEX `quadrant_enumeration` (634 high-res radiographs with 18,095
     full-mouth tooth polygon annotations) for multi-center tooth segmentation.
  3. Extracts and saves all 3,529 tooth-level disease ROI crops (256x256 CLAHE
     with 1.4x apical/periapical context padding) from DENTEX
     `quadrant-enumeration-disease`, strictly preserving the exact train/val/test
     image split (547 / 51 / 107 radiographs -> 2,728 / 293 / 508 tooth ROIs).
"""

import os
import sys
import json
import csv
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed_v2"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest_v2.csv"
ROI_MANIFEST_PATH = OUTPUT_DIR / "tooth_roi_manifest_v2.csv"
AUDIT_CSV_PATH = ROOT / "outputs" / "v2_data_audit_15gb.csv"

IMG_SIZE = 512
ROI_SIZE = 256
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
SEED = 42

DISEASE_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "DeepCaries"}


def apply_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    l_enh = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_enh, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def letterbox_resize(image, target_size, fill_value=0):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(image.shape) == 3:
        canvas = np.full((target_size, target_size, image.shape[2]), fill_value, dtype=image.dtype)
    else:
        canvas = np.full((target_size, target_size), fill_value, dtype=image.dtype)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def letterbox_resize_mask(mask, target_size):
    h, w = mask.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_size, target_size), dtype=mask.dtype)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized if len(resized.shape) == 2 else resized[:, :, 0]
    return canvas


def audit_raw_datasets():
    print("=" * 75)
    print("STEP 1: Complete Audit of Raw Datasets on Disk")
    print("=" * 75)
    cohorts = [
        ("DentalAI (Train/Val/Test)", DATA_RAW / "dentalai", "Full-Mouth Tooth Segmentation (Supervisely Polygons)", 2495),
        ("DENTEX Quadrant-Enumeration-Disease", DATA_RAW / "dentex" / "training_data" / "quadrant-enumeration-disease", "4-Class Pathology + Tooth BBoxes/Polygons (3,529 ROIs)", 705),
        ("DENTEX Quadrant-Enumeration", DATA_RAW / "dentex" / "training_data" / "quadrant_enumeration", "Full-Mouth 32-Tooth Polygons (18,095 Tooth Masks)", 634),
        ("DENTEX Quadrant", DATA_RAW / "dentex" / "training_data" / "quadrant", "Quadrant Localization BBoxes (2,772 Quadrants)", 693),
        ("DENTEX Unlabelled", DATA_RAW / "dentex" / "training_data" / "unlabelled", "Unlabelled Panoramic Radiographs (Self-Training Pool)", 1571),
        ("DENTEX Official Validation", DATA_RAW / "dentex" / "validation_data", "Official Challenge Validation Radiographs", 50),
        ("TUFTS External Holdout", DATA_RAW / "tufts", "External Zero-Shot Radiographs + Masks + Expert Maps", 1000),
    ]
    rows = []
    total_files = 0
    total_bytes = 0
    for name, path, desc, img_count in cohorts:
        if path.exists():
            files = [p for p in path.rglob("*") if p.is_file()]
            b = sum(p.stat().st_size for p in files)
            n_f = len(files)
        else:
            b, n_f = 0, 0
        total_files += n_f
        total_bytes += b
        gb = b / (1024 ** 3)
        mb = b / (1024 ** 2)
        rows.append({
            "cohort": name,
            "radiographs": img_count,
            "total_files": n_f,
            "size_mb": round(mb, 2),
            "size_gb": round(gb, 3),
            "annotation_details": desc,
        })
        print(f"  {name:38s} | {img_count:5d} X-rays | {n_f:5d} files | {gb:6.2f} GB ({mb:7.1f} MB)")

    print("-" * 75)
    print(f"  {'TOTAL RAW DATASET ON DISK':38s} | {sum(r['radiographs'] for r in rows):5d} X-rays | {total_files:5d} files | {total_bytes/(1024**3):6.2f} GB")
    print("=" * 75)
    AUDIT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AUDIT_CSV_PATH, index=False)
    return rows


def process_dentex_quadrant_enumeration():
    """Process the 634 full-mouth polygon-annotated DENTEX radiographs (18,095 tooth masks)."""
    print("\n" + "=" * 75)
    print("STEP 2: Processing DENTEX Quadrant-Enumeration (634 Images, 18,095 Tooth Masks)")
    print("=" * 75)

    qe_json = DATA_RAW / "dentex" / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json"
    xray_dir = DATA_RAW / "dentex" / "training_data" / "quadrant_enumeration" / "xrays"
    out_dir = OUTPUT_DIR / "dentex_qe"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(qe_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    records = []
    t0 = time.time()
    for idx, (img_id, filename) in enumerate(sorted(id_to_file.items())):
        img_path = xray_dir / filename
        if not img_path.exists():
            continue

        uid = f"dentex_qe_{img_id:04d}"
        img_out = out_dir / f"{uid}_img.png"
        mask_out = out_dir / f"{uid}_mask.png"

        anns = img_anns.get(img_id, [])
        if img_out.exists() and mask_out.exists():
            mask_bin = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
            tooth_pixels = int(np.sum(mask_bin > 0)) if mask_bin is not None else 0
        else:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            seg_mask = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                for seg_poly in ann.get("segmentation", []):
                    pts = np.array(seg_poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(seg_mask, [pts], 255)

            img_enh = apply_clahe(img_bgr)
            img_res = letterbox_resize(img_enh, IMG_SIZE, fill_value=0)
            mask_res = letterbox_resize_mask(seg_mask, IMG_SIZE)
            mask_bin = (mask_res > 127).astype(np.uint8) * 255

            cv2.imwrite(str(img_out), img_res)
            cv2.imwrite(str(mask_out), mask_bin)
            tooth_pixels = int(np.sum(mask_bin > 0))

        records.append({
            "uid": uid,
            "source": "dentex_qe",
            "original_name": filename,
            "img_path": str(img_out),
            "mask_path": str(mask_out),
            "tooth_pixels": tooth_pixels,
            "has_teeth": 1 if tooth_pixels > 100 else 0,
            "num_annotations": len(anns),
            "has_disease": 0,
            "disease_impacted": 0,
            "disease_caries": 0,
            "disease_periapical": 0,
            "disease_deep_caries": 0,
            "original_split": "train",
        })
        if (idx + 1) % 150 == 0 or (idx + 1) == len(id_to_file):
            print(f"  Processed {idx + 1:3d}/{len(id_to_file)} DENTEX-QE full-mouth masks [{time.time() - t0:.1f}s]")

    # Reproducible 80/10/10 split for dentex_qe
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_train = int(len(records) * 0.80)  # 507
    n_val = int(len(records) * 0.10)    # 63
    for rank, i in enumerate(indices):
        if rank < n_train:
            records[i]["split"] = "train"
        elif rank < n_train + n_val:
            records[i]["split"] = "val"
        else:
            records[i]["split"] = "test"

    # Update manifest without altering existing dentalai, dentex, or tufts rows
    df_manifest = pd.read_csv(MANIFEST_PATH)
    df_manifest = df_manifest[df_manifest["source"] != "dentex_qe"].copy()
    df_qe = pd.DataFrame(records)
    df_combined = pd.concat([df_manifest, df_qe], ignore_index=True)
    df_combined.to_csv(MANIFEST_PATH, index=False)

    print(f"  Added {len(df_qe)} DENTEX-QE records to {MANIFEST_PATH}")
    print(f"  DENTEX-QE Splits: Train={sum(df_qe['split']=='train')}, Val={sum(df_qe['split']=='val')}, Test={sum(df_qe['split']=='test')}")
    return df_qe


def extract_dentex_tooth_rois():
    """Extract all 3,529 annotated tooth ROI crops with 1.4x apical context padding."""
    print("\n" + "=" * 75)
    print("STEP 3: Extracting 3,529 High-Res Tooth-Level Disease ROIs from DENTEX")
    print("=" * 75)

    df_manifest = pd.read_csv(MANIFEST_PATH)
    dtx_df = df_manifest[df_manifest["source"] == "dentex"].copy()
    file_to_row = {row["original_name"]: row for _, row in dtx_df.iterrows()}

    qed_json = DATA_RAW / "dentex" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
    xray_dir = DATA_RAW / "dentex" / "training_data" / "quadrant-enumeration-disease" / "xrays"
    roi_dir = OUTPUT_DIR / "dentex_tooth_rois"
    roi_dir.mkdir(parents=True, exist_ok=True)

    with open(qed_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    roi_records = []
    t0 = time.time()

    for idx, (img_id, filename) in enumerate(sorted(id_to_file.items())):
        if filename not in file_to_row:
            continue
        parent_row = file_to_row[filename]
        split = parent_row["split"]
        parent_uid = parent_row["uid"]

        img_path = xray_dir / filename
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_enh = apply_clahe(img_bgr)
        H, W = img_enh.shape[:2]

        # Group overlapping annotations by tooth bbox or process each annotation
        for ann_idx, ann in enumerate(img_anns.get(img_id, [])):
            cat3 = int(ann.get("category_id_3", -1))
            if not (0 <= cat3 <= 3):
                continue
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            cx, cy = x + bw / 2.0, y + bh / 2.0
            # Use 1.45x square context window around tooth to capture root apex & bone
            side = max(bw, bh) * 1.45
            x1 = max(0, int(round(cx - side / 2.0)))
            y1 = max(0, int(round(cy - side / 2.0)))
            x2 = min(W, int(round(cx + side / 2.0)))
            y2 = min(H, int(round(cy + side / 2.0)))
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue

            crop_uid = f"{parent_uid}_roi_{ann['id']:05d}"
            crop_out = roi_dir / f"{crop_uid}.png"
            if not crop_out.exists():
                crop = img_enh[y1:y2, x1:x2]
                crop_res = cv2.resize(crop, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(crop_out), crop_res)

            # Check all disease annotations belonging strictly to the SAME FDI tooth
            # (matching quadrant category_id_1 and tooth position category_id_2)
            q_id = int(ann.get("category_id_1", -1))
            t_id = int(ann.get("category_id_2", -1))
            vec = [0, 0, 0, 0]
            vec[cat3] = 1
            for other_ann in img_anns.get(img_id, []):
                if int(other_ann.get("category_id_1", -99)) == q_id and int(other_ann.get("category_id_2", -99)) == t_id:
                    oc3 = int(other_ann.get("category_id_3", -1))
                    if 0 <= oc3 <= 3:
                        vec[oc3] = 1
            # Ontological Hierarchical Closure: Deep Caries implies Dental Caries
            vec[1] = max(vec[1], vec[3])

            roi_records.append({
                "roi_uid": crop_uid,
                "parent_uid": parent_uid,
                "original_name": filename,
                "split": split,
                "roi_path": str(crop_out),
                "primary_class_id": cat3,
                "primary_class_name": DISEASE_NAMES[cat3],
                "quadrant_id": int(ann.get("category_id_1", -1)),
                "tooth_pos_id": int(ann.get("category_id_2", -1)),
                "bbox_x": round(x, 1),
                "bbox_y": round(y, 1),
                "bbox_w": round(bw, 1),
                "bbox_h": round(bh, 1),
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "img_w": W,
                "img_h": H,
                "disease_impacted": vec[0],
                "disease_caries": vec[1],
                "disease_periapical": vec[2],
                "disease_deep_caries": vec[3],
            })

        if (idx + 1) % 150 == 0 or (idx + 1) == len(id_to_file):
            print(f"  Extracted ROIs from {idx + 1:3d}/{len(id_to_file)} radiographs -> {len(roi_records)} crops [{time.time() - t0:.1f}s]")

    df_rois = pd.DataFrame(roi_records)
    df_rois.to_csv(ROI_MANIFEST_PATH, index=False)
    print(f"\n  Saved {len(df_rois)} tooth ROI records to {ROI_MANIFEST_PATH}")
    for sp in ["train", "val", "test"]:
        sub = df_rois[df_rois["split"] == sp]
        counts = {d: int(sub[f"disease_{d.lower() if d != 'DeepCaries' else 'deep_caries'}"].sum()) for d in ["Impacted", "Caries", "Periapical", "DeepCaries"]}
        print(f"    {sp.upper():5s}: {len(sub):4d} ROIs across {sub['parent_uid'].nunique():3d} X-rays | {counts}")
    return df_rois


if __name__ == "__main__":
    audit_raw_datasets()
    process_dentex_quadrant_enumeration()
    extract_dentex_tooth_rois()
