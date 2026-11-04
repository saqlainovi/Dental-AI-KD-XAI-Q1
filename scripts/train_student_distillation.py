"""V15 Stage 3: Student model trained via Knowledge Distillation from the joint teacher.

Training data = DentalAI (real masks) + DENTEX (teacher pseudo-masks).
Loss = alpha * L_hard(student, GT) + (1-alpha) * T^2 * KL(student/T, teacher/T)

The student is a lightweight TinyUNet (or configurable family) that can be
deployed without the heavy Swin-T teacher.
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.augmentations import prepare_segmentation_pair
from dental_ai.training.swin_upernet import (
    build_model_from_checkpoint,
    build_segmentation_model,
    segmentation_logits,
)
from train_dentex_student_segmentation import GPUEpochSampler

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """Generic segmentation dataset: image + mask PNG."""

    def __init__(self, rows: list[dict], image_size: int, augment: bool, use_clahe: bool = True) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        image, mask = prepare_segmentation_pair(image, mask, self.image_size, augment=self.augment, use_clahe=self.use_clahe)
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        mask_arr = np.asarray(mask).astype(np.int64)
        return torch.from_numpy(np.transpose(image_arr, (2, 0, 1))), torch.from_numpy(mask_arr)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def hard_seg_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor, num_classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights)
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    present = one_hot.sum(dims) > 0
    present[0] = False
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    dice_l = 1 - dice[present].mean() if present.any() else torch.tensor(0.0, device=logits.device)
    return ce + dice_l


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL-divergence distillation loss between student and teacher soft outputs."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    # KL(teacher || student) summed over classes, averaged over batch & pixels
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return temperature * temperature * kl


def seg_metrics(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    pred_b = pred.bool()
    target_b = target.bool()
    inter = (pred_b & target_b).sum().float()
    union = (pred_b | target_b).sum().float()
    denom = pred_b.sum().float() + target_b.sum().float()
    dice = float(((2 * inter + eps) / (denom + eps)).cpu())
    iou = float(((inter + eps) / (union + eps)).cpu())
    acc = float((pred == target).float().mean().cpu())
    return {"dice": dice, "iou": iou, "accuracy": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 3: Student KD training.")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True, help="Best teacher .pt file")
    parser.add_argument("--pseudo-mask-root", type=Path, required=True, help="Dir with DENTEX pseudo masks")
    parser.add_argument("--mask-root", type=Path, default=None, help="DentalAI mask root")
    parser.add_argument("--student-family", default="tinyunet", choices=["tinyunet", "swin_t_upernet"])
    parser.add_argument("--student-width", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5, help="Hard loss weight (1-alpha = KD weight)")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve mask root
    mask_root = args.mask_root or ROOT / "data" / "processed" / "masks"
    if not mask_root.exists():
        for c in [Path("E:/dental_ai_processed/masks"), Path("F:/dental_ai_processed/masks")]:
            if c.exists():
                mask_root = c
                break

    if args.mode == "fast":
        epochs = args.epochs or 3
        image_size = args.image_size or 224
        batch_size = args.batch_size or 4
        max_dentalai = 60
        max_dentex = 40
    else:
        epochs = args.epochs or 50
        image_size = args.image_size or 224
        batch_size = args.batch_size or 8
        max_dentalai = None
        max_dentex = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Load teacher ---
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher_class_names = teacher_ckpt.get("class_names") or teacher_ckpt.get("seg_class_names") or ["background", "Caries", "Cavity", "Crack", "Tooth"]
    num_classes = len(teacher_class_names)
    teacher = build_model_from_checkpoint(teacher_ckpt, num_classes=num_classes).to(device)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded: {args.teacher_checkpoint} classes={teacher_class_names}")

    # --- Load DentalAI data ---
    raw_root = ROOT / "data" / "raw" / "dentalai"
    coco_root = ROOT / "data" / "processed" / "coco" / "dentalai"
    dentalai_rows: list[dict] = []
    for split in ["train", "valid"]:
        coco_data = json.loads((coco_root / f"{split}.json").read_text(encoding="utf-8"))
        for img in coco_data["images"][:max_dentalai]:
            ip = raw_root / img["file_name"]
            mp = mask_root / f"dentalai_{split}" / f"{Path(img['file_name']).stem}_mask.png"
            if ip.exists() and mp.exists():
                dentalai_rows.append({"image_path": str(ip), "mask_path": str(mp), "source": "dentalai"})

    # --- Load DENTEX pseudo-mask data ---
    pseudo_mask_dir = args.pseudo_mask_root / "masks"
    dentex_rows: list[dict] = []
    dentex_root = ROOT / "data" / "raw" / "dentex"
    if pseudo_mask_dir.exists():
        for mask_file in sorted(pseudo_mask_dir.glob("*_pseudo_mask.png")):
            stem = mask_file.stem.replace("_pseudo_mask", "")
            # Try to find the source image
            image_path = None
            for sub in ["training_data/quadrant-enumeration-disease", "training_data/quadrant", "training_data/quadrant_enumeration"]:
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = dentex_root / sub / (stem + ext)
                    if candidate.exists():
                        image_path = candidate
                        break
                if image_path:
                    break
            if image_path:
                dentex_rows.append({"image_path": str(image_path), "mask_path": str(mask_file), "source": "dentex_pseudo"})
            if max_dentex and len(dentex_rows) >= max_dentex:
                break

    # --- Split ---
    all_rows = dentalai_rows + dentex_rows
    random.Random(args.seed).shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train:n_train + n_val]
    test_rows = all_rows[n_train + n_val:]

    print(f"=== V15 Student Distillation ===")
    print(f"mode={args.mode} epochs={epochs} image_size={image_size} batch_size={batch_size}")
    print(f"DentalAI rows={len(dentalai_rows)} DENTEX pseudo rows={len(dentex_rows)}")
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"alpha={args.alpha} temperature={args.temperature} student_family={args.student_family}")

    if not train_rows:
        print("ERROR: No training data found!", flush=True)
        sys.exit(1)

    # --- Build student ---
    student = build_segmentation_model(
        model_family=args.student_family,
        num_classes=num_classes,
        width=args.student_width,
        gradient_checkpointing=False,
        pretrained_imagenet=(args.student_family != "tinyunet"),
    ).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Class weights from training masks
    class_weights = torch.ones(num_classes, device=device)
    class_weights[0] = 0.35
    for i in range(1, num_classes):
        class_weights[i] = 1.0

    # DataLoaders
    train_ds = SegDataset(train_rows, image_size, augment=True, use_clahe=True)
    val_ds = SegDataset(val_rows, image_size, augment=False, use_clahe=True)
    test_ds = SegDataset(test_rows, image_size, augment=False, use_clahe=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Checkpoint setup
    output_root = Path(args.output_root)
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "student_latest.pt"
    best_ckpt_path = ckpt_dir / "student_best.pt"
    history_path = output_root / "student_epoch_history.csv"

    best_val_dice = -math.inf
    best_epoch = 0
    no_improve = 0
    start_epoch = 1

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", -math.inf)
        best_epoch = ckpt.get("best_epoch", 0)
        no_improve = ckpt.get("no_improve", 0)
        print(f"Resumed from epoch {start_epoch}")

    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)
    history_rows: list[dict] = []

    print(f"\nStarting training from epoch {start_epoch} to {epochs}...", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        sampler.start()
        student.train()

        train_losses = []
        train_hard_losses = []
        train_kd_losses = []
        train_dices = []

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                # Student forward
                student_output = student(images)
                student_logits = segmentation_logits(student_output)

                # Hard loss (CE + Dice against GT masks)
                h_loss = hard_seg_loss(student_logits, masks, class_weights, num_classes)

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_output = teacher(images)
                    teacher_logits = segmentation_logits(teacher_output)

                # KD loss (KL-divergence with temperature)
                kd_l = kd_loss(student_logits, teacher_logits, args.temperature)

                # Combined loss
                loss = args.alpha * h_loss + (1 - args.alpha) * kd_l

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))
            train_hard_losses.append(float(h_loss.detach().cpu()))
            train_kd_losses.append(float(kd_l.detach().cpu()))
            m = seg_metrics(student_logits.detach(), masks)
            train_dices.append(m["dice"])

        scheduler.step()

        # Validation
        student.eval()
        val_dices = []
        val_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = segmentation_logits(student(images))
                m = seg_metrics(logits, masks)
                val_dices.append(m["dice"])
                val_ious.append(m["iou"])

        elapsed = time.perf_counter() - epoch_start
        gpu_stats = sampler.stop()

        train_loss = float(np.mean(train_losses))
        train_hard = float(np.mean(train_hard_losses))
        train_kd = float(np.mean(train_kd_losses))
        train_dice = float(np.mean(train_dices))
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        val_iou = float(np.mean(val_ious)) if val_ious else 0.0

        is_best = val_dice > best_val_dice + 1e-4
        if is_best:
            best_val_dice = val_dice
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch, "train_loss": train_loss, "train_hard_loss": train_hard,
            "train_kd_loss": train_kd, "train_dice": train_dice,
            "val_dice": val_dice, "val_iou": val_iou,
            "best_val_dice": best_val_dice, "best_epoch": best_epoch,
            "is_best": int(is_best), "lr": scheduler.get_last_lr()[0],
            "seconds": elapsed,
        }
        history_rows.append(row)

        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

        print(
            f"epoch {epoch:02d}/{epochs}: "
            f"loss={train_loss:.4f} hard={train_hard:.4f} kd={train_kd:.4f} "
            f"train_dice={train_dice:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
            f"best={best_val_dice:.4f}@ep{best_epoch} no_improve={no_improve} "
            f"lr={scheduler.get_last_lr()[0]:.6f} seconds={elapsed:.1f}",
            flush=True,
        )

        payload = {
            "epoch": epoch, "model": student.state_dict(), "optimizer": optimizer.state_dict(),
            "best_val_dice": best_val_dice, "best_epoch": best_epoch, "no_improve": no_improve,
            "model_family": args.student_family, "class_names": teacher_class_names,
            "config": {
                "model_family": args.student_family, "width": args.student_width,
                "image_size": image_size, "num_segmentation_classes": num_classes,
                "alpha": args.alpha, "temperature": args.temperature,
                "pipeline": "v15_student_kd",
            },
        }
        torch.save(payload, ckpt_path)
        if is_best:
            torch.save(payload, best_ckpt_path)

        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            print(f"EARLY_STOPPING epoch={epoch}", flush=True)
            break

    # --- Final test ---
    if best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(best["model"])
    student.eval()
    test_dices = []
    test_ious = []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = segmentation_logits(student(images))
            m = seg_metrics(logits, masks)
            test_dices.append(m["dice"])
            test_ious.append(m["iou"])

    test_dice = float(np.mean(test_dices)) if test_dices else 0.0
    test_iou = float(np.mean(test_ious)) if test_ious else 0.0

    summary = {"best_epoch": best_epoch, "best_val_dice": best_val_dice, "test_dice": test_dice, "test_iou": test_iou}
    (output_root / "student_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSTUDENT_KD_DONE best_epoch={best_epoch} val_dice={best_val_dice:.4f} test_dice={test_dice:.4f} test_iou={test_iou:.4f}")
    print(f"best_checkpoint={best_ckpt_path}")


if __name__ == "__main__":
    main()
