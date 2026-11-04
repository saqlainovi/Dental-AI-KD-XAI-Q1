"""V15 Stage 1: Joint Teacher model trained on DentalAI masks + DENTEX disease labels.

The teacher is a Swin-T UPerNet with:
  - segmentation head: supervised by DentalAI polygon masks (CE + Dice)
  - classification head: supervised by DENTEX disease labels (focal BCE)

Both datasets share a backbone, so the model learns rich dental features
before generating pseudo-masks for the larger DENTEX set (Stage 2).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.augmentations import prepare_segmentation_pair
from dental_ai.training.swin_upernet import build_segmentation_model, segmentation_logits
from train_dentex_student_segmentation import GPUEpochSampler

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DentalAISegDataset(Dataset):
    """DentalAI images + real polygon masks for segmentation supervision."""

    def __init__(self, rows: list[dict], image_size: int, num_classes: int, augment: bool) -> None:
        self.rows = rows
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"])
        image, mask = prepare_segmentation_pair(image, mask, self.image_size, augment=self.augment, use_clahe=False)
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))
        mask_arr = np.asarray(mask).astype(np.int64)
        mask_arr = np.clip(mask_arr, 0, self.num_classes - 1)
        return torch.from_numpy(image_arr), torch.from_numpy(mask_arr)


class DentexDiseaseDataset(Dataset):
    """DENTEX images + multi-label disease vectors for classification supervision."""

    def __init__(self, records: list[dict], image_size: int, augment: bool) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        dummy_mask = Image.new("L", image.size, 0)
        image, _ = prepare_segmentation_pair(image, dummy_mask, self.image_size, augment=self.augment, use_clahe=True)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
        labels = torch.tensor(rec["labels"], dtype=torch.float32)
        return tensor, labels


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_coco(coco_root: Path, split: str) -> dict:
    return json.loads((coco_root / f"{split}.json").read_text(encoding="utf-8"))


def build_seg_index(
    raw_root: Path, coco_root: Path, mask_root: Path, split: str, max_images: int | None,
) -> tuple[list[dict], list[str]]:
    data = load_coco(coco_root, split)
    rows = []
    for img in data["images"][:max_images]:
        image_path = raw_root / img["file_name"]
        mask_path = mask_root / f"dentalai_{split}" / f"{Path(img['file_name']).stem}_mask.png"
        if image_path.exists() and mask_path.exists():
            rows.append({"split": split, "image_path": str(image_path), "mask_path": str(mask_path)})
    class_names = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]
    return rows, class_names


def load_disease_records(coco_json: Path, image_root: Path) -> tuple[list[dict], list[str]]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = sorted(coco.get("categories", []), key=lambda c: int(c.get("id", 0)))
    class_names = [str(c.get("name", f"disease_{i}")) for i, c in enumerate(categories)]
    dentex_names = ["caries", "deep_caries", "periapical_lesion", "impacted"]
    if len(class_names) == 4 and all(n.startswith("disease_class_") for n in class_names):
        class_names = dentex_names
    images = {int(img["id"]): img for img in coco.get("images", [])}
    labels = {iid: np.zeros(len(class_names), dtype=np.float32) for iid in images}
    for ann in coco.get("annotations", []):
        iid = int(ann.get("image_id", -1))
        cid = int(ann.get("category_id", -1))
        if iid in labels and 0 <= cid < len(class_names):
            labels[iid][cid] = 1.0
    records = []
    for iid, img in images.items():
        image_path = image_root / img["file_name"]
        if not image_path.exists():
            for sub in ["training_data/quadrant-enumeration-disease", "training_data/quadrant", "training_data/quadrant_enumeration"]:
                candidate = image_root / sub / Path(img["file_name"]).name
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path.exists():
            records.append({"image_id": iid, "file_name": img["file_name"], "image_path": str(image_path), "labels": labels[iid].tolist()})
    return records, class_names


def split_records(records: list[dict], train_frac: float, val_frac: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    present = one_hot.sum(dims) > 0
    present[0] = False
    dice = (2 * intersection + eps) / (union + eps)
    if present.any():
        return 1 - dice[present].mean()
    return torch.tensor(0.0, device=logits.device)


def focal_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    if gamma <= 0:
        return bce.mean()
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def compute_pixel_class_weights(rows: list[dict], num_classes: int, max_rows: int | None = 400) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    selected = rows if max_rows is None else rows[:max_rows]
    for row in selected:
        mask = np.asarray(Image.open(row["mask_path"]))
        vals, pix = np.unique(np.clip(mask, 0, num_classes - 1), return_counts=True)
        counts[vals] += pix
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(counts.sum() / (counts * num_classes))
    weights = np.clip(weights, 0.25, 4.0)
    weights[0] = min(weights[0], 0.50)
    return torch.tensor(weights, dtype=torch.float32)


def seg_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    scores = []
    for cls in range(1, num_classes):
        pred_c = pred == cls
        target_c = target == cls
        if target_c.sum().item() == 0:
            continue
        inter = (pred_c & target_c).sum().float()
        denom = pred_c.sum().float() + target_c.sum().float()
        union = (pred_c | target_c).sum().float()
        scores.append((float((2 * inter + eps) / (denom + eps)), float((inter + eps) / (union + eps))))
    return {
        "dice": sum(s[0] for s in scores) / max(1, len(scores)),
        "iou": sum(s[1] for s in scores) / max(1, len(scores)),
        "pixel_accuracy": float((pred == target).float().mean()),
    }


def cls_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8)
    tp = (pred * target_np).sum(axis=0).astype(np.float32)
    fp = (pred * (1 - target_np)).sum(axis=0).astype(np.float32)
    fn = ((1 - pred) * target_np).sum(axis=0).astype(np.float32)
    f1 = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    return {"macro_f1": float(np.mean(f1)), "exact_match": float((pred == target_np).all(axis=1).mean())}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 1: Train joint teacher on DentalAI + DENTEX.")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-family", default="swin_t_upernet")
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Resolve paths ---
    mask_root = args.mask_root or ROOT / "data" / "processed" / "masks"
    if not mask_root.exists():
        for candidate in [Path("E:/dental_ai_processed/masks"), Path("F:/dental_ai_processed/masks")]:
            if candidate.exists():
                mask_root = candidate
                break

    if args.mode == "fast":
        epochs = args.epochs or 3
        image_size = args.image_size or 224
        batch_size = args.batch_size or 2
        max_seg = 80
        max_cls = 50
    else:
        epochs = args.epochs or 40
        image_size = args.image_size or 384
        batch_size = args.batch_size or 4
        max_seg = None
        max_cls = None

    # --- Load DentalAI segmentation data ---
    raw_root = ROOT / "data" / "raw" / "dentalai"
    coco_root = ROOT / "data" / "processed" / "coco" / "dentalai"
    seg_train, class_names = build_seg_index(raw_root, coco_root, mask_root, "train", max_seg)
    seg_valid, _ = build_seg_index(raw_root, coco_root, mask_root, "valid", max_seg)
    seg_test, _ = build_seg_index(raw_root, coco_root, mask_root, "test", max_seg)
    num_seg_classes = len(class_names) + 1  # +1 for background

    # --- Load DENTEX disease data ---
    disease_coco = ROOT / "data" / "processed" / "coco" / "dentex" / "disease.json"
    dentex_root = ROOT / "data" / "raw" / "dentex"
    disease_records, disease_names = load_disease_records(disease_coco, dentex_root)
    if max_cls:
        disease_records = disease_records[:max_cls]
    dis_train, dis_val, dis_test = split_records(disease_records, 0.70, 0.15, args.seed)
    num_disease_classes = len(disease_names)

    print(f"=== V15 Joint Teacher ===")
    print(f"mode={args.mode} epochs={epochs} image_size={image_size} batch_size={batch_size}")
    print(f"DentalAI seg: train={len(seg_train)} valid={len(seg_valid)} test={len(seg_test)} classes={['background'] + class_names}")
    print(f"DENTEX disease: train={len(dis_train)} valid={len(dis_val)} test={len(dis_test)} classes={disease_names}")
    print(f"mask_root={mask_root}")

    if not seg_train:
        print(f"\nERROR: No DentalAI training rows found.")
        print(f"Masks expected at: {mask_root}/dentalai_train/")
        print(f"Run: python scripts/build_segmentation_targets.py --coco {coco_root}/train.json --output {mask_root}/dentalai_train")
        sys.exit(1)

    # --- Build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = build_segmentation_model(
        model_family=args.model_family,
        num_classes=num_seg_classes,
        decoder_channels=args.decoder_channels,
        num_classification_labels=num_disease_classes,
        gradient_checkpointing=args.gradient_checkpointing,
        pretrained_imagenet=args.pretrained_imagenet,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    seg_class_weights = compute_pixel_class_weights(seg_train, num_seg_classes).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Disease pos weights
    label_arr = np.array([r["labels"] for r in dis_train], dtype=np.float32)
    pos_counts = label_arr.sum(axis=0)
    neg_counts = label_arr.shape[0] - pos_counts
    cls_pos_weight = torch.tensor(np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 10.0), dtype=torch.float32, device=device)

    # --- DataLoaders ---
    seg_train_loader = DataLoader(DentalAISegDataset(seg_train, image_size, num_seg_classes, augment=True), batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    seg_valid_loader = DataLoader(DentalAISegDataset(seg_valid, image_size, num_seg_classes, augment=False), batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dis_train_loader = DataLoader(DentexDiseaseDataset(dis_train, image_size, augment=True), batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dis_val_loader = DataLoader(DentexDiseaseDataset(dis_val, image_size, augment=False), batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dis_test_loader = DataLoader(DentexDiseaseDataset(dis_test, image_size, augment=False), batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Checkpoint setup ---
    output_root = Path(args.output_root)
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "teacher_joint_latest.pt"
    best_ckpt_path = ckpt_dir / "teacher_joint_best.pt"
    history_path = output_root / "teacher_joint_epoch_history.csv"

    best_composite = -math.inf
    best_epoch = 0
    no_improve = 0
    start_epoch = 1

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_composite = ckpt.get("best_composite", -math.inf)
        best_epoch = ckpt.get("best_epoch", 0)
        no_improve = ckpt.get("no_improve", 0)
        print(f"Resumed from epoch {start_epoch}, best_composite={best_composite:.4f}")

    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)
    history_rows: list[dict] = []

    print(f"\nStarting training from epoch {start_epoch} to {epochs}...", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        sampler.start()
        model.train()

        # ---- Segmentation training pass ----
        seg_losses = []
        seg_dices = []
        seg_ious = []
        for images, masks in seg_train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(images)
                logits = segmentation_logits(output)
                loss = args.seg_loss_weight * (F.cross_entropy(logits, masks, weight=seg_class_weights) + dice_loss(logits, masks, num_seg_classes))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            seg_losses.append(float(loss.detach().cpu()))
            m = seg_metrics(logits.detach(), masks, num_seg_classes)
            seg_dices.append(m["dice"])
            seg_ious.append(m["iou"])

        # ---- Classification training pass ----
        cls_losses = []
        cls_f1s = []
        for images, labels in dis_train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(images)
                if isinstance(output, dict) and "classification" in output:
                    cls_logits = output["classification"]
                    loss = args.cls_loss_weight * focal_bce(cls_logits, labels, cls_pos_weight, args.focal_gamma)
                else:
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            cls_losses.append(float(loss.detach().cpu()))
            cm = cls_metrics(cls_logits, labels)
            cls_f1s.append(cm["macro_f1"])

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_dices = []
        val_ious = []
        with torch.no_grad():
            for images, masks in seg_valid_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = segmentation_logits(model(images))
                m = seg_metrics(logits, masks, num_seg_classes)
                val_dices.append(m["dice"])
                val_ious.append(m["iou"])

        val_cls_f1s = []
        with torch.no_grad():
            for images, labels in dis_val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(images)
                    if isinstance(output, dict) and "classification" in output:
                        cm = cls_metrics(output["classification"], labels)
                        val_cls_f1s.append(cm["macro_f1"])

        elapsed = time.perf_counter() - epoch_start
        gpu_stats = sampler.stop()

        train_seg_loss = float(np.mean(seg_losses)) if seg_losses else math.nan
        train_seg_dice = float(np.mean(seg_dices)) if seg_dices else math.nan
        train_seg_iou = float(np.mean(seg_ious)) if seg_ious else math.nan
        train_cls_loss = float(np.mean(cls_losses)) if cls_losses else math.nan
        train_cls_f1 = float(np.mean(cls_f1s)) if cls_f1s else math.nan
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        val_iou = float(np.mean(val_ious)) if val_ious else 0.0
        val_cls_f1 = float(np.mean(val_cls_f1s)) if val_cls_f1s else 0.0

        # Composite score: 0.5 * val_dice + 0.5 * val_cls_f1
        composite = 0.5 * val_dice + 0.5 * val_cls_f1
        is_best = composite > best_composite + 1e-4
        if is_best:
            best_composite = composite
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch, "train_seg_loss": train_seg_loss, "train_seg_dice": train_seg_dice,
            "train_seg_iou": train_seg_iou, "train_cls_loss": train_cls_loss, "train_cls_f1": train_cls_f1,
            "val_dice": val_dice, "val_iou": val_iou, "val_cls_f1": val_cls_f1,
            "composite": composite, "best_composite": best_composite, "best_epoch": best_epoch,
            "is_best": int(is_best), "seconds": elapsed, "lr": scheduler.get_last_lr()[0],
            "power_w_mean": gpu_stats.get("power_draw_w_mean", math.nan),
            "gpu_util_mean": gpu_stats.get("gpu_util_percent_mean", math.nan),
        }
        history_rows.append(row)

        # Write CSV history
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

        print(
            f"epoch {epoch:02d}/{epochs}: "
            f"seg_loss={train_seg_loss:.4f} seg_dice={train_seg_dice:.4f}/{val_dice:.4f} "
            f"seg_iou={train_seg_iou:.4f}/{val_iou:.4f} "
            f"cls_loss={train_cls_loss:.4f} cls_f1={train_cls_f1:.4f}/{val_cls_f1:.4f} "
            f"composite={composite:.4f} best={best_composite:.4f}@ep{best_epoch} "
            f"no_improve={no_improve} lr={scheduler.get_last_lr()[0]:.6f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

        # Save checkpoints
        payload = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_composite": best_composite, "best_epoch": best_epoch, "no_improve": no_improve,
            "model_family": args.model_family, "num_seg_classes": num_seg_classes,
            "num_disease_classes": num_disease_classes, "seg_class_names": ["background"] + class_names,
            "disease_class_names": disease_names, "class_names": ["background"] + class_names,
            "classification_classes": disease_names,
            "config": {
                "mode": args.mode, "model_family": args.model_family, "image_size": image_size,
                "batch_size": batch_size, "num_segmentation_classes": num_seg_classes,
                "decoder_channels": args.decoder_channels, "gradient_checkpointing": args.gradient_checkpointing,
                "pretrained_imagenet": args.pretrained_imagenet, "learning_rate": args.learning_rate,
                "seg_loss_weight": args.seg_loss_weight, "cls_loss_weight": args.cls_loss_weight,
                "focal_gamma": args.focal_gamma, "pipeline": "v15_teacher_student_kd",
            },
        }
        torch.save(payload, ckpt_path)
        if is_best:
            torch.save(payload, best_ckpt_path)

        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            print(f"EARLY_STOPPING epoch={epoch} best_epoch={best_epoch} composite={best_composite:.4f}", flush=True)
            break

    # ---- Final test evaluation ----
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])

    model.eval()
    test_dices = []
    test_ious = []
    with torch.no_grad():
        for images, masks in DataLoader(DentalAISegDataset(seg_test, image_size, num_seg_classes, augment=False), batch_size=batch_size, num_workers=0):
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = segmentation_logits(model(images))
            m = seg_metrics(logits, masks, num_seg_classes)
            test_dices.append(m["dice"])
            test_ious.append(m["iou"])

    test_cls_probs = []
    test_cls_targets = []
    with torch.no_grad():
        for images, labels in dis_test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(images)
                if isinstance(output, dict) and "classification" in output:
                    test_cls_probs.append(torch.sigmoid(output["classification"]).cpu().numpy())
                    test_cls_targets.append(labels.cpu().numpy())

    test_dice = float(np.mean(test_dices)) if test_dices else 0.0
    test_iou = float(np.mean(test_ious)) if test_ious else 0.0
    if test_cls_probs:
        test_prob = np.concatenate(test_cls_probs)
        test_tgt = np.concatenate(test_cls_targets)
        test_pred = (test_prob >= 0.5).astype(np.uint8)
        tp = (test_pred * test_tgt).sum(axis=0)
        fp = (test_pred * (1 - test_tgt)).sum(axis=0)
        fn = ((1 - test_pred) * test_tgt).sum(axis=0)
        test_f1 = float(np.mean((2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)))
    else:
        test_f1 = 0.0

    summary = {"best_epoch": best_epoch, "best_composite": best_composite, "test_dice": test_dice, "test_iou": test_iou, "test_macro_f1": test_f1}
    (output_root / "teacher_joint_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTEACHER_JOINT_DONE best_epoch={best_epoch} composite={best_composite:.4f} test_dice={test_dice:.4f} test_iou={test_iou:.4f} test_macro_f1={test_f1:.4f}")
    print(f"best_checkpoint={best_ckpt_path}")


if __name__ == "__main__":
    main()
