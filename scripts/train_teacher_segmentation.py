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


class DentalSegDataset(Dataset):
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


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.d1 = DoubleConv(3, width)
        self.d2 = DoubleConv(width, width * 2)
        self.d3 = DoubleConv(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.c2 = DoubleConv(width * 4, width * 2)
        self.u1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.c1 = DoubleConv(width * 2, width)
        self.out = nn.Conv2d(width, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        y = self.u2(x3)
        y = self.c2(torch.cat([y, x2], dim=1))
        y = self.u1(y)
        y = self.c1(torch.cat([y, x1], dim=1))
        return self.out(y)


def load_coco(coco_root: Path, split: str) -> dict:
    return json.loads((coco_root / f"{split}.json").read_text(encoding="utf-8"))


def build_index(
    raw_root: Path,
    coco_root: Path,
    mask_root: Path,
    split: str,
    max_images: int | None,
) -> tuple[list[dict], list[str]]:
    data = load_coco(coco_root, split)
    rows = []
    for image in data["images"][:max_images]:
        image_path = raw_root / image["file_name"]
        mask_path = mask_root / f"dentalai_{split}" / f"{Path(image['file_name']).stem}_mask.png"
        if image_path.exists() and mask_path.exists():
            rows.append({"split": split, "image_path": image_path, "mask_path": mask_path})
    class_names = [item["name"] for item in sorted(data["categories"], key=lambda item: item["id"])]
    return rows, class_names


def compute_pixel_class_weights(rows: list[dict], num_classes: int, max_rows: int | None) -> tuple[torch.Tensor, np.ndarray]:
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
    return torch.tensor(weights, dtype=torch.float32), counts


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


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    pixel_correct = (pred == target).sum().item()
    pixel_total = target.numel()
    scores = []
    class_accs = []
    for cls in range(1, num_classes):
        pred_c = pred == cls
        target_c = target == cls
        target_count = target_c.sum().item()
        if target_count == 0:
            continue
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        denom = pred_c.sum().float() + target_c.sum().float()
        iou = (inter + eps) / (union + eps)
        dice = (2 * inter + eps) / (denom + eps)
        scores.append((float(dice.cpu()), float(iou.cpu())))
        class_accs.append(float((inter / max(1, target_count)).cpu()))
    return {
        "dice": sum(s[0] for s in scores) / len(scores) if scores else 0.0,
        "iou": sum(s[1] for s in scores) / len(scores) if scores else 0.0,
        "pixel_correct": float(pixel_correct),
        "pixel_total": float(pixel_total),
        "pixel_accuracy": pixel_correct / max(1, pixel_total),
        "mean_class_accuracy": float(np.mean(class_accs)) if class_accs else 0.0,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor,
    num_classes: int,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    max_grad_norm: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    dices: list[float] = []
    ious: list[float] = []
    mean_class_accs: list[float] = []
    pixel_correct = 0.0
    pixel_total = 0.0
    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = segmentation_logits(model(images))
                loss = F.cross_entropy(logits, masks, weight=class_weights) + dice_loss(logits, masks, num_classes)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at batch {batch_idx}: loss={float(loss.detach().cpu())}")
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        metrics = segmentation_metrics(logits.detach(), masks, num_classes)
        dices.append(metrics["dice"])
        ious.append(metrics["iou"])
        mean_class_accs.append(metrics["mean_class_accuracy"])
        pixel_correct += metrics["pixel_correct"]
        pixel_total += metrics["pixel_total"]
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": pixel_correct / max(1.0, pixel_total),
        "mean_class_accuracy": float(np.mean(mean_class_accs)) if mean_class_accs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DentalAI teacher segmentation baseline.")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("E:/dental_ai_checkpoints"))
    parser.add_argument("--mask-root", type=Path, default=Path("E:/dental_ai_processed/masks"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-family", choices=["tinyunet", "swin_t_upernet"], default="swin_t_upernet")
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--best-metric", choices=["val_dice", "val_iou", "val_loss"], default="val_dice")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.mode == "fast":
        epochs = args.epochs or 3
        image_size = args.image_size or 224
        batch_size = args.batch_size or 1
        max_train = 160
        max_valid = 60
    else:
        epochs = args.epochs or 30
        image_size = args.image_size or 224
        batch_size = args.batch_size or 1
        max_train = None
        max_valid = None
    eval_batch_size = args.eval_batch_size or batch_size

    raw_root = ROOT / "data" / "raw" / "dentalai"
    coco_root = ROOT / "data" / "processed" / "coco" / "dentalai"
    train_rows, class_names = build_index(raw_root, coco_root, args.mask_root, "train", max_train)
    valid_rows, _ = build_index(raw_root, coco_root, args.mask_root, "valid", max_valid)
    test_rows, _ = build_index(raw_root, coco_root, args.mask_root, "test", max_valid)

    num_classes = len(class_names) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader = DataLoader(
        DentalSegDataset(train_rows, image_size, num_classes, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        DentalSegDataset(valid_rows, image_size, num_classes, augment=False),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        DentalSegDataset(test_rows, image_size, num_classes, augment=False),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_segmentation_model(
        model_family=args.model_family,
        num_classes=num_classes,
        width=32,
        decoder_channels=args.decoder_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        pretrained_imagenet=args.pretrained_imagenet,
    ).to(device)
    learning_rate = args.learning_rate if args.learning_rate is not None else (1e-4 if args.model_family == "swin_t_upernet" else 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    class_weights, pixel_counts = compute_pixel_class_weights(train_rows, num_classes, max_rows=400)
    class_weights = class_weights.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_root / f"teacher_{args.model_family}_dentalai_{args.mode}.pt"
    best_checkpoint_path = args.checkpoint_root / f"teacher_{args.model_family}_dentalai_{args.mode}_best.pt"
    epoch_history_path = args.checkpoint_root / f"teacher_{args.model_family}_dentalai_{args.mode}_epoch_history.csv"
    history = {
        "train_loss": [],
        "train_pixel_accuracy": [],
        "train_mean_class_accuracy": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_pixel_accuracy": [],
        "val_mean_class_accuracy": [],
        "test_loss": [],
        "test_dice": [],
        "test_iou": [],
        "test_pixel_accuracy": [],
        "test_mean_class_accuracy": [],
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
        "best_metric": [],
        "best_metric_value": [],
        "is_best": [],
    }
    start_epoch = 1
    best_metric_value = math.inf if args.best_metric == "val_loss" else -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        loaded_history = checkpoint.get("history", {})
        loaded_train_loss = np.asarray(loaded_history.get("train_loss", []), dtype=np.float64)
        checkpoint_is_finite = loaded_train_loss.size == 0 or np.isfinite(loaded_train_loss).all()
        if checkpoint_is_finite:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            loaded_epoch_count = len(loaded_history.get("train_loss", []))
            for key in history:
                values = list(loaded_history.get(key, history[key]))
                if len(values) < loaded_epoch_count:
                    values.extend([math.nan] * (loaded_epoch_count - len(values)))
                history[key] = values
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metric_value = float(checkpoint.get("best_metric_value", best_metric_value))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
            if best_epoch == 0 and history.get(args.best_metric):
                metric_values = np.asarray(history[args.best_metric], dtype=np.float64)
                finite = np.isfinite(metric_values)
                if finite.any():
                    finite_idx = np.where(finite)[0]
                    selected_idx = finite_idx[np.argmin(metric_values[finite])] if args.best_metric == "val_loss" else finite_idx[np.argmax(metric_values[finite])]
                    best_epoch = int(selected_idx + 1)
                    best_metric_value = float(metric_values[selected_idx])
            print(f"Resuming from epoch {start_epoch}: {checkpoint_path}")
        else:
            print(f"Found non-finite loss history in {checkpoint_path}; ignoring checkpoint and restarting from epoch 1.", flush=True)

    print(
        f"mode={args.mode} model_family={args.model_family} amp={device.type == 'cuda'} "
        f"gradient_checkpointing={args.gradient_checkpointing} device={device} image_size={image_size} "
        f"train_batch={batch_size} eval_batch={eval_batch_size}"
    )
    total_split = max(1, len(train_rows) + len(valid_rows) + len(test_rows))
    print(
        f"train_images={len(train_rows)} ({len(train_rows) / total_split:.1%}) "
        f"valid_images={len(valid_rows)} ({len(valid_rows) / total_split:.1%}) "
        f"test_images={len(test_rows)} ({len(test_rows) / total_split:.1%}) "
        f"classes={['background', *class_names]}"
    )
    print(f"pixel_counts={np.asarray(pixel_counts, dtype=np.int64).tolist()}")
    print(f"class_weights={[round(float(x), 3) for x in class_weights.detach().cpu()]}")
    print(
        f"learning_rate={learning_rate} max_grad_norm={args.max_grad_norm} "
        f"best_metric={args.best_metric} early_stopping_patience={args.early_stopping_patience}"
    )

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

    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)

    def metric_improved(current: float) -> bool:
        if args.best_metric == "val_loss":
            return current < (best_metric_value - args.min_delta)
        return current > (best_metric_value + args.min_delta)

    def checkpoint_payload(
        epoch: int,
        is_best: bool,
        current_best_metric_value: float,
        current_best_epoch: int,
        current_epochs_without_improvement: int,
    ) -> dict:
        return {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "class_names": ["background", *class_names],
            "model_family": args.model_family,
            "best_metric": args.best_metric,
            "best_metric_value": current_best_metric_value,
            "best_epoch": current_best_epoch,
            "is_best": is_best,
            "epochs_without_improvement": current_epochs_without_improvement,
            "config": {
                "mode": args.mode,
                "model_family": args.model_family,
                "image_size": image_size,
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "num_workers": args.num_workers,
                "num_classes": num_classes,
                "num_segmentation_classes": num_classes,
                "decoder_channels": args.decoder_channels,
                "gradient_checkpointing": args.gradient_checkpointing,
                "pretrained_imagenet": args.pretrained_imagenet,
                "gpu_sample_interval": args.gpu_sample_interval,
                "learning_rate": learning_rate,
                "max_grad_norm": args.max_grad_norm,
                "best_metric": args.best_metric,
                "early_stopping_patience": args.early_stopping_patience,
                "min_delta": args.min_delta,
                "split_policy": "DatasetNinja official DentalAI split",
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "test_rows": len(test_rows),
                "augmentation": "horizontal flip, rotation +/-15, affine +/-10 and translate 5%, brightness/contrast jitter",
            },
        }

    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        sampler.start()
        train_stats = run_epoch(model, train_loader, optimizer, class_weights, num_classes, device, scaler, True, args.max_grad_norm)
        val_stats = run_epoch(model, valid_loader, None, class_weights, num_classes, device, scaler, True, args.max_grad_norm)
        test_stats = run_epoch(model, test_loader, None, class_weights, num_classes, device, scaler, True, args.max_grad_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - epoch_start
        gpu_stats = sampler.stop()
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else math.nan
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else math.nan
        history["train_loss"].append(train_stats["loss"])
        history["train_pixel_accuracy"].append(train_stats["pixel_accuracy"])
        history["train_mean_class_accuracy"].append(train_stats["mean_class_accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])
        history["val_iou"].append(val_stats["iou"])
        history["val_pixel_accuracy"].append(val_stats["pixel_accuracy"])
        history["val_mean_class_accuracy"].append(val_stats["mean_class_accuracy"])
        history["test_loss"].append(test_stats["loss"])
        history["test_dice"].append(test_stats["dice"])
        history["test_iou"].append(test_stats["iou"])
        history["test_pixel_accuracy"].append(test_stats["pixel_accuracy"])
        history["test_mean_class_accuracy"].append(test_stats["mean_class_accuracy"])
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
        current_metric = val_stats[args.best_metric.removeprefix("val_")] if args.best_metric in {"val_dice", "val_iou"} else val_stats["loss"]
        is_best = metric_improved(float(current_metric))
        if is_best:
            best_metric_value = float(current_metric)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history["best_metric"].append(args.best_metric)
        history["best_metric_value"].append(best_metric_value)
        history["is_best"].append(int(is_best))
        write_epoch_history()
        print(
            f"epoch {epoch:02d}/{epochs}: "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"train_acc={train_stats['pixel_accuracy']:.4f} val_acc={val_stats['pixel_accuracy']:.4f} "
            f"test_acc={test_stats['pixel_accuracy']:.4f} "
            f"val_dice={val_stats['dice']:.4f} val_iou={val_stats['iou']:.4f} "
            f"test_dice={test_stats['dice']:.4f} test_iou={test_stats['iou']:.4f} "
            f"val_mca={val_stats['mean_class_accuracy']:.4f} test_mca={test_stats['mean_class_accuracy']:.4f} "
            f"power_mean_w={gpu_stats.get('power_draw_w_mean', math.nan):.2f} "
            f"power_rms_w={gpu_stats.get('power_draw_w_rms', math.nan):.2f} "
            f"gpu_util_mean={gpu_stats.get('gpu_util_percent_mean', math.nan):.1f}% "
            f"mem_max_mb={gpu_stats.get('memory_used_mb_max', math.nan):.0f} seconds={elapsed:.1f} "
            f"best_{args.best_metric}=epoch{best_epoch}:{best_metric_value:.4f} "
            f"no_improve={epochs_without_improvement}",
            flush=True,
        )
        payload = checkpoint_payload(epoch, is_best, best_metric_value, best_epoch, epochs_without_improvement)
        torch.save(payload, checkpoint_path)
        if is_best:
            torch.save(payload, best_checkpoint_path)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"EARLY_STOPPING epoch={epoch} best_epoch={best_epoch} "
                f"best_{args.best_metric}={best_metric_value:.4f} best_checkpoint={best_checkpoint_path}",
                flush=True,
            )
            break
    print(f"TEACHER_TRAINING_DONE checkpoint={checkpoint_path} best_checkpoint={best_checkpoint_path} best_epoch={best_epoch} best_{args.best_metric}={best_metric_value:.4f}")


if __name__ == "__main__":
    main()
