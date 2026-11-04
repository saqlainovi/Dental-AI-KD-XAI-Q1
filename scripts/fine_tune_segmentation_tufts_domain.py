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
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFile
from torch.utils.data import ConcatDataset, DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits
from generate_final_evaluation_xai import load_internal_test_rows_from_checkpoint, resolve_path


def build_tufts_pairs(raw_root: Path) -> list[dict]:
    image_dir = raw_root / "Radiographs" / "Radiographs"
    mask_dir = raw_root / "Segmentation" / "Segmentation" / "teeth_mask"
    mask_by_stem = {path.stem.lower(): path for path in mask_dir.glob("*") if path.is_file()}
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_by_stem.get(image_path.stem.lower())
        if mask_path:
            rows.append({"image": image_path.name, "image_path": str(image_path), "mask_path": str(mask_path)})
    return rows


def split_tufts_rows(rows: list[dict], calibration_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_cal = max(1, int(round(len(shuffled) * calibration_fraction)))
    return shuffled[:n_cal], shuffled[n_cal:]


def write_tufts_manifest(path: Path, calibration_rows: list[dict], holdout_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "image_path", "mask_path"])
        writer.writeheader()
        for split, rows in [("calibration", calibration_rows), ("holdout", holdout_rows)]:
            for row in rows:
                writer.writerow({"split": split, **row})


def augment_pair(image: Image.Image, mask: Image.Image, image_size: int, augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
    image = image.convert("RGB")
    mask = mask.convert("L")
    if augment:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.25))
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.15))
    image = image.resize((image_size, image_size), Image.BILINEAR)
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    image_arr = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_arr, (2, 0, 1)))
    mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype(np.int64))
    return image_tensor, mask_tensor


class SegRowsDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, augment: bool, mask_key: str = "mask_path") -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.mask_key = mask_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = resolve_path(row.get("image_path", ""))
        mask_path = resolve_path(row.get(self.mask_key, row.get("clean_mask", row.get("mask_path", ""))))
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        return augment_pair(image, mask, self.image_size, self.augment)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1]
    target_f = target.float()
    intersection = (prob * target_f).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
    return (1.0 - (2 * intersection + eps) / (denom + eps)).mean()


def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = logits.argmax(dim=1).bool()
    target_b = target.bool()
    intersection = torch.logical_and(pred, target_b).sum().float()
    pred_sum = pred.sum().float()
    target_sum = target_b.sum().float()
    union = torch.logical_or(pred, target_b).sum().float()
    dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice.detach().cpu()), float(iou.detach().cpu())


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    class_weights: torch.Tensor,
    dice_loss_weight: float,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    dices = []
    ious = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = segmentation_logits(model(images))
                loss = F.cross_entropy(logits, masks, weight=class_weights)
                loss = loss + dice_loss_weight * dice_loss(logits, masks)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        d, i = dice_iou_from_logits(logits.detach(), masks)
        dices.append(d)
        ious.append(i)
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "dice": float(np.mean(dices)) if dices else math.nan,
        "iou": float(np.mean(ious)) if ious else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DENTEX segmentation checkpoint on a TUFTS calibration split.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tufts-root", type=Path, default=ROOT / "data/raw/tufts")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, default=None, help="Optional DENTEX clean root used only as metadata fallback.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--tufts-calibration-fraction", type=float, default=0.30)
    parser.add_argument("--dentex-replay-weight", type=float, default=1.0)
    parser.add_argument("--max-dentex-replay", type=int, default=200)
    parser.add_argument("--foreground-class-weight", type=float, default=1.0)
    parser.add_argument("--background-class-weight", type=float, default=0.35)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = args.output_root / "tables"
    checkpoints = args.output_root / "checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    base_checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    config = base_checkpoint.get("config", {})
    image_size = int(config.get("image_size", 512))
    model = build_model_from_checkpoint(base_checkpoint, num_classes=2).to(device)
    model.load_state_dict(base_checkpoint["model"])

    all_tufts_rows = build_tufts_pairs(args.tufts_root)
    calibration_rows, holdout_rows = split_tufts_rows(all_tufts_rows, args.tufts_calibration_fraction, args.seed)
    manifest_path = tables / "tufts_calibration_holdout_manifest.csv"
    write_tufts_manifest(manifest_path, calibration_rows, holdout_rows)

    dentex_rows = load_internal_test_rows_from_checkpoint(base_checkpoint)
    if args.max_dentex_replay:
        dentex_rows = dentex_rows[: args.max_dentex_replay]
    train_datasets: list[Dataset] = [SegRowsDataset(calibration_rows, image_size, augment=True, mask_key="mask_path")]
    if dentex_rows and args.dentex_replay_weight > 0:
        train_datasets.append(SegRowsDataset(dentex_rows, image_size, augment=True, mask_key="clean_mask"))
    train_dataset = ConcatDataset(train_datasets)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "tufts_calibration": DataLoader(SegRowsDataset(calibration_rows, image_size, augment=False, mask_key="mask_path"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "tufts_holdout": DataLoader(SegRowsDataset(holdout_rows, image_size, augment=False, mask_key="mask_path"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }
    if dentex_rows:
        loaders["dentex_replay"] = DataLoader(SegRowsDataset(dentex_rows, image_size, augment=False, mask_key="clean_mask"), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    class_weights = torch.tensor([args.background_class_weight, args.foreground_class_weight], dtype=torch.float32, device=device)
    ckpt_path = checkpoints / "tufts_domain_finetuned.pt"
    best_path = checkpoints / "tufts_domain_finetuned_best_holdout_dice.pt"
    start_epoch = 1
    best_holdout_dice = -1.0
    history: list[dict] = []
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_holdout_dice = float(checkpoint.get("best_holdout_dice", best_holdout_dice))
        history = list(checkpoint.get("history", []))
        print(f"resuming_tufts_domain_finetune_from_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        train_stats = run_epoch(model, loaders["train"], device, optimizer, scaler, class_weights, args.dice_loss_weight)
        calibration_stats = run_epoch(model, loaders["tufts_calibration"], device, None, scaler, class_weights, args.dice_loss_weight)
        holdout_stats = run_epoch(model, loaders["tufts_holdout"], device, None, scaler, class_weights, args.dice_loss_weight)
        dentex_stats = run_epoch(model, loaders["dentex_replay"], device, None, scaler, class_weights, args.dice_loss_weight) if "dentex_replay" in loaders else {"dice": math.nan, "iou": math.nan, "loss": math.nan}
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_dice": train_stats["dice"],
            "tufts_calibration_dice": calibration_stats["dice"],
            "tufts_calibration_iou": calibration_stats["iou"],
            "tufts_holdout_dice": holdout_stats["dice"],
            "tufts_holdout_iou": holdout_stats["iou"],
            "dentex_replay_dice": dentex_stats["dice"],
            "dentex_replay_iou": dentex_stats["iou"],
            "seconds": time.perf_counter() - start,
        }
        is_best = holdout_stats["dice"] > best_holdout_dice
        row["is_best_holdout_dice"] = int(is_best)
        if is_best:
            best_holdout_dice = float(holdout_stats["dice"])
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "config": {
                **config,
                "image_size": image_size,
                "model_family": config.get("model_family", base_checkpoint.get("model_family", "swin_t_upernet")),
                "tufts_domain_finetune": True,
                "tufts_manifest": str(manifest_path),
                "tufts_calibration_fraction": args.tufts_calibration_fraction,
                "base_checkpoint": str(args.base_checkpoint),
                "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
            },
            "model_family": base_checkpoint.get("model_family", config.get("model_family", "swin_t_upernet")),
            "class_names": base_checkpoint.get("class_names", ["background", "tooth"]),
            "split_manifest": base_checkpoint.get("split_manifest"),
            "tufts_manifest": str(manifest_path),
            "best_val_dice": float(base_checkpoint.get("best_val_dice", 0.0)),
            "best_holdout_dice": best_holdout_dice,
        }
        torch.save(checkpoint, ckpt_path)
        if is_best:
            torch.save(checkpoint, best_path)
        pd.DataFrame(history).to_csv(tables / "tufts_domain_finetune_epoch_history.csv", index=False)
        print(
            f"epoch {epoch:02d}/{args.epochs}: train_loss={train_stats['loss']:.4f} "
            f"cal_dice={calibration_stats['dice']:.4f} holdout_dice={holdout_stats['dice']:.4f} "
            f"holdout_iou={holdout_stats['iou']:.4f} dentex_dice={dentex_stats['dice']:.4f} best_holdout={best_holdout_dice:.4f}",
            flush=True,
        )

    summary = {
        "checkpoint": str(best_path if best_path.exists() else ckpt_path),
        "base_checkpoint": str(args.base_checkpoint),
        "tufts_manifest": str(manifest_path),
        "tufts_total_rows": len(all_tufts_rows),
        "tufts_calibration_rows": len(calibration_rows),
        "tufts_holdout_rows": len(holdout_rows),
        "dentex_replay_rows": len(dentex_rows),
        "best_holdout_dice": best_holdout_dice,
        "epochs": args.epochs,
        "image_size": image_size,
        "seed": args.seed,
    }
    pd.DataFrame([summary]).to_csv(tables / "tufts_domain_finetune_summary.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"TUFTS_DOMAIN_FINETUNE_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
