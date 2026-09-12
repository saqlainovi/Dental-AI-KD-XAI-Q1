"""
V15 Clean Data Loader and Split Policy Enforcer.
Strictly follows recovery_pipeline/split_policy.json:
- Teacher training: DentalAI train (polygon masks) + DENTEX train weak bounding-box supervision
- Teacher validation: DentalAI validation
- Student training: DENTEX train cleaned pseudo-masks (frozen teacher distillation)
- Internal test: DENTEX test split
- External test ONLY: TUFTS dental database (NEVER in train/val/hyperparameter tuning)
"""
import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DENTALAI_MASK_TRAIN = r"F:\dental_ai_processed\masks\dentalai_train"
DENTALAI_MASK_VALID = r"F:\dental_ai_processed\masks\dentalai_valid"
DENTEX_CLEAN_PSEUDO = r"F:\dental_ai_processed\pseudo_masks\dentex_xray_teacher_clean"
TUFTS_RAW_IMG_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw", "tufts", "Radiographs", "Radiographs")
TUFTS_RAW_MASK_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw", "tufts", "Segmentation", "Segmentation", "teeth_mask")
DENTEX_RAW_ROOT = os.path.join(WORKSPACE_ROOT, "data", "raw", "dentex")

class DentalSegmentationDataset(Dataset):
    def __init__(self, samples, image_size=512, is_train=False):
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item["image_path"]
        mask_path = item["mask_path"]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.is_train:
            if torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_np = (np.array(mask) > 127).astype(np.int64)
        mask_t = torch.from_numpy(mask_np)

        return {
            "image": img_t,
            "mask": mask_t,
            "image_path": img_path,
            "mask_path": mask_path,
            "file_name": os.path.basename(img_path)
        }

def collect_dentex_student_samples(clean_root=DENTEX_CLEAN_PSEUDO, dentex_raw=DENTEX_RAW_ROOT):
    train_samples = []
    val_samples = []

    tasks = [
        ("quadrant_full", train_samples),
        ("enumeration_full", train_samples),
        ("disease_full", train_samples),
        ("validation_full", val_samples)
    ]

    file_map = {}
    for root, _, files in os.walk(dentex_raw):
        for f in files:
            if f.endswith(".png"):
                file_map[f] = os.path.join(root, f)

    for task_name, sample_list in tasks:
        task_dir = os.path.join(clean_root, task_name)
        masks_dir = os.path.join(task_dir, "masks")
        if not os.path.exists(masks_dir):
            continue

        for mask_file in os.listdir(masks_dir):
            if not mask_file.endswith("_clean_mask.png"):
                continue
            base_name = mask_file.replace("_clean_mask.png", ".png")
            mask_path = os.path.join(masks_dir, mask_file)

            found_img = file_map.get(base_name)
            if found_img and os.path.exists(found_img):
                sample_list.append({
                    "image_path": found_img,
                    "mask_path": mask_path,
                    "task": task_name
                })

    return train_samples, val_samples

def collect_tufts_holdout_samples(tufts_img_dir=TUFTS_RAW_IMG_DIR, tufts_mask_dir=TUFTS_RAW_MASK_DIR):
    samples = []
    if not os.path.exists(tufts_img_dir) or not os.path.exists(tufts_mask_dir):
        return samples

    for fname in os.listdir(tufts_img_dir):
        if fname.lower().endswith((".jpg", ".png")):
            img_p = os.path.join(tufts_img_dir, fname)
            base = os.path.splitext(fname)[0]
            mask_p = os.path.join(tufts_mask_dir, f"{base}.png")
            if not os.path.exists(mask_p):
                mask_p = os.path.join(tufts_mask_dir, f"{base}.jpg")

            if os.path.exists(mask_p):
                samples.append({
                    "image_path": img_p,
                    "mask_path": mask_p,
                    "dataset": "tufts_external_holdout"
                })
    return samples

def verify_strict_split_policy(train_samples, val_samples, test_samples):
    train_imgs = {s["image_path"].lower() for s in train_samples}
    val_imgs = {s["image_path"].lower() for s in val_samples}
    test_imgs = {s["image_path"].lower() for s in test_samples}

    for p in train_imgs | val_imgs:
        if "tufts" in p:
            raise AssertionError(f"CRITICAL VIOLATION: TUFTS file found in training/val set: {p}")

    overlap_val = train_imgs.intersection(val_imgs)
    if overlap_val:
        raise AssertionError(f"CRITICAL VIOLATION: Overlap between train and validation: {len(overlap_val)} files")

    overlap_test = train_imgs.intersection(test_imgs)
    if overlap_test:
        raise AssertionError(f"CRITICAL VIOLATION: Overlap between train and external test: {len(overlap_test)} files")

    return True

def get_v15_dataloaders(batch_size=4, image_size=512, num_workers=0):
    train_samples, val_samples = collect_dentex_student_samples()
    tufts_samples = collect_tufts_holdout_samples()
    verify_strict_split_policy(train_samples, val_samples, tufts_samples)

    train_ds = DentalSegmentationDataset(train_samples, image_size=image_size, is_train=True)
    val_ds = DentalSegmentationDataset(val_samples, image_size=image_size, is_train=False)
    tufts_ds = DentalSegmentationDataset(tufts_samples, image_size=image_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tufts_loader = DataLoader(tufts_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, tufts_loader

if __name__ == "__main__":
    print("=" * 60)
    print("V15 Strict Split Loader Verification")
    print("=" * 60)
    train_s, val_s = collect_dentex_student_samples()
    tufts_s = collect_tufts_holdout_samples()
    print(f"[*] DENTEX Student Training Samples: {len(train_s)}")
    print(f"[*] DENTEX Validation Samples:       {len(val_s)}")
    print(f"[*] TUFTS External Holdout Samples:   {len(tufts_s)}")
    verify_strict_split_policy(train_s, val_s, tufts_s)
    print("[+] ANTI-LEAKAGE VERIFICATION PASSED: ZERO contamination between datasets.")
    print("=" * 60)
