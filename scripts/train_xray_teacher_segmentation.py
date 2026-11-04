from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.augmentations import prepare_segmentation_pair
from dental_ai.training.swin_upernet import build_segmentation_model, segmentation_logits
from train_dentex_student_segmentation import GPUEpochSampler


class TuftsToothDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, Path]], image_size: int, augment: bool) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = prepare_segmentation_pair(image, mask, self.image_size, augment=self.augment, use_clahe=True)
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))
        mask_arr = (np.asarray(mask) > 0).astype(np.int64)
        return torch.from_numpy(image_arr), torch.from_numpy(mask_arr)


def build_tufts_rows(raw_root: Path) -> list[tuple[Path, Path]]:
    image_dir = raw_root / "Radiographs" / "Radiographs"
    mask_dir = raw_root / "Segmentation" / "Segmentation" / "teeth_mask"
    mask_by_stem = {path.stem.lower(): path for path in mask_dir.glob("*") if path.is_file()}
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_by_stem.get(image_path.stem.lower())
        if mask_path:
            rows.append((image_path, mask_path))
    return rows


def split_rows(rows: list[tuple[Path, Path]], seed: int = 42) -> dict[str, list[tuple[Path, Path]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.80)
    n_valid = int(n * 0.10)
    return {
        "train": rows[:n_train],
        "valid": rows[n_train : n_train + n_valid],
        "test": rows[n_train + n_valid :],
    }


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_f = target.float()
    intersection = (probs * target_f).sum()
    denom = probs.sum() + target_f.sum()
    return 1 - (2 * intersection + eps) / (denom + eps)


def scores(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = logits.argmax(dim=1).bool()
    target_b = target.bool()
    intersection = (pred & target_b).sum().float()
    union = (pred | target_b).sum().float()
    denom = pred.sum().float() + target_b.sum().float()
    dice = (2 * intersection + eps) / (denom + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice.cpu()), float(iou.cpu())


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    dices = []
    ious = []
    accuracies = []
    class_weights = torch.tensor([0.35, 1.0], dtype=torch.float32, device=device)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = segmentation_logits(model(images))
                loss = F.cross_entropy(logits, masks, weight=class_weights) + dice_loss(logits, masks)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        d, i = scores(logits.detach(), masks)
        dices.append(d)
        ious.append(i)
        accuracies.append(float((logits.detach().argmax(dim=1) == masks).float().mean().detach().cpu()))
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an X-ray tooth segmentation teacher on TUFTS.")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("E:/dental_ai_checkpoints"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    parser.add_argument("--model-family", choices=["tinyunet", "swin_t_upernet"], default="swin_t_upernet")
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-imagenet", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.mode == "fast":
        epochs = args.epochs or 3
        image_size = args.image_size or 224
        batch_size = args.batch_size or 1
    else:
        epochs = args.epochs or 40
        image_size = args.image_size or 224
        batch_size = args.batch_size or 1

    raw_root = ROOT / "data" / "raw" / "tufts"
    splits = split_rows(build_tufts_rows(raw_root))
    if args.mode == "fast":
        splits["train"] = splits["train"][:160]
        splits["valid"] = splits["valid"][:60]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    loaders = {
        split: DataLoader(
            TuftsToothDataset(rows, image_size, augment=(split == "train")),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, rows in splits.items()
    }

    model = build_segmentation_model(
        model_family=args.model_family,
        num_classes=2,
        width=32,
        decoder_channels=args.decoder_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        pretrained_imagenet=args.pretrained_imagenet,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_root / f"teacher_{args.model_family}_tufts_xray_{args.mode}.pt"
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_accuracy": [],
        "test_loss": [],
        "test_dice": [],
        "test_iou": [],
        "test_accuracy": [],
        "seconds": [],
        "peak_allocated_mb": [],
        "peak_reserved_mb": [],
        "power_draw_w_mean": [],
        "power_draw_w_max": [],
        "power_draw_w_rms": [],
        "gpu_util_percent_mean": [],
        "gpu_util_percent_max": [],
        "memory_used_mb_mean": [],
        "memory_used_mb_max": [],
        "temperature_c_mean": [],
        "temperature_c_max": [],
        "gpu_samples": [],
    }
    start_epoch = 1
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(loaded_history.get(key, history[key]))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}: {checkpoint_path}")

    print(
        f"mode={args.mode} model_family={args.model_family} amp={device.type == 'cuda'} "
        f"gradient_checkpointing={args.gradient_checkpointing} device={device} image_size={image_size} batch={batch_size} "
        f"train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}",
        flush=True,
    )

    epoch_history_path = args.checkpoint_root / f"teacher_{args.model_family}_tufts_xray_{args.mode}_epoch_history.csv"
    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)

    def write_epoch_history() -> None:
        max_len = max((len(values) for values in history.values()), default=0)
        rows = []
        for idx in range(max_len):
            row = {"epoch": idx + 1}
            for key, values in history.items():
                row[key] = values[idx] if idx < len(values) else math.nan
            rows.append(row)
        if rows:
            with epoch_history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    if start_epoch > 1:
        write_epoch_history()

    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        sampler.start()
        train_stats = run_epoch(model, loaders["train"], optimizer, device, scaler, True)
        val_stats = run_epoch(model, loaders["valid"], None, device, scaler, True)
        test_stats = run_epoch(model, loaders["test"], None, device, scaler, True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - epoch_start
        gpu_stats = sampler.stop()
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else math.nan
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else math.nan
        history["train_loss"].append(train_stats["loss"])
        history["train_dice"].append(train_stats["dice"])
        history["train_iou"].append(train_stats["iou"])
        history["train_accuracy"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])
        history["val_iou"].append(val_stats["iou"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["test_loss"].append(test_stats["loss"])
        history["test_dice"].append(test_stats["dice"])
        history["test_iou"].append(test_stats["iou"])
        history["test_accuracy"].append(test_stats["accuracy"])
        history["seconds"].append(elapsed)
        history["peak_allocated_mb"].append(peak_allocated_mb)
        history["peak_reserved_mb"].append(peak_reserved_mb)
        for key in [
            "power_draw_w_mean",
            "power_draw_w_max",
            "power_draw_w_rms",
            "gpu_util_percent_mean",
            "gpu_util_percent_max",
            "memory_used_mb_mean",
            "memory_used_mb_max",
            "temperature_c_mean",
            "temperature_c_max",
            "gpu_samples",
        ]:
            history[key].append(gpu_stats.get(key, math.nan))
        print(
            f"epoch {epoch:02d}/{epochs}: train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} train_dice={train_stats['dice']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice']:.4f} "
            f"val_iou={val_stats['iou']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"test_dice={test_stats['dice']:.4f} test_iou={test_stats['iou']:.4f} "
            f"power_mean_w={gpu_stats.get('power_draw_w_mean', math.nan):.2f} "
            f"power_rms_w={gpu_stats.get('power_draw_w_rms', math.nan):.2f} "
            f"gpu_util_mean={gpu_stats.get('gpu_util_percent_mean', math.nan):.1f}% "
            f"mem_max_mb={gpu_stats.get('memory_used_mb_max', math.nan):.0f} seconds={elapsed:.1f}",
            flush=True,
        )
        write_epoch_history()
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "class_names": ["background", "tooth"],
                "model_family": args.model_family,
                "config": {
                    "mode": args.mode,
                    "model_family": args.model_family,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "num_workers": args.num_workers,
                    "decoder_channels": args.decoder_channels,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "pretrained_imagenet": args.pretrained_imagenet,
                    "num_segmentation_classes": 2,
                    "split_policy": "TUFTS source-domain split: 80% train, 10% validation, 10% held-out test",
                    "train_rows": len(splits["train"]),
                    "valid_rows": len(splits["valid"]),
                    "test_rows": len(splits["test"]),
                    "augmentation": "CLAHE, horizontal flip, rotation +/-15, affine +/-10 and translate 5%, brightness/contrast jitter",
                    "source": "TUFTS X-ray teeth masks",
                },
            },
            checkpoint_path,
        )
    print(f"XRAY_TEACHER_TRAINING_DONE checkpoint={checkpoint_path} epoch_history={epoch_history_path}", flush=True)


if __name__ == "__main__":
    main()
