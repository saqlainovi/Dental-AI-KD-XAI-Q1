from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]


def resolve_image_path(file_name: str, image_root: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() and path.exists():
        return path
    for candidate in [image_root / file_name, ROOT / file_name, ROOT / "data" / "raw" / "dentex" / file_name]:
        if candidate.exists():
            return candidate
    return image_root / file_name


def load_records(coco_json: Path, image_root: Path) -> tuple[list[dict], list[str]]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = sorted(coco.get("categories", []), key=lambda item: int(item.get("id", 0)))
    class_names = [str(item.get("name", f"disease_class_{idx}")) for idx, item in enumerate(categories)]
    dentex_disease_names = ["caries", "deep_caries", "periapical_lesion", "impacted"]
    if len(class_names) == 4 and all(name.startswith("disease_class_") for name in class_names):
        class_names = dentex_disease_names
    images = {int(image["id"]): image for image in coco.get("images", [])}
    labels = {image_id: np.zeros(len(class_names), dtype=np.float32) for image_id in images}
    boxes = {image_id: [] for image_id in images}
    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        category_id = int(ann.get("category_id", -1))
        if image_id in labels and 0 <= category_id < len(class_names):
            labels[image_id][category_id] = 1.0
            if ann.get("bbox"):
                x, y, w, h = [float(v) for v in ann["bbox"]]
                boxes[image_id].append({"category_id": category_id, "bbox": [x, y, w, h]})
    records = []
    for image_id, image in images.items():
        image_path = resolve_image_path(str(image["file_name"]), image_root)
        if image_path.exists():
            records.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "image_path": str(image_path),
                    "labels": labels[image_id].tolist(),
                    "boxes": boxes[image_id],
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                }
            )
    return records, class_names


def split_records(records: list[dict], train_fraction: float, val_fraction: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n_total = len(rows)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    return rows[:n_train], rows[n_train : n_train + n_val], rows[n_train + n_val :]


def square_box(box: tuple[float, float, float, float], width: int, height: int, expand: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * expand
    x1 = max(0, int(round(cx - side / 2)))
    y1 = max(0, int(round(cy - side / 2)))
    x2 = min(width, int(round(cx + side / 2)))
    y2 = min(height, int(round(cy + side / 2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def grid_boxes(width: int, height: int, grid: int) -> list[tuple[int, int, int, int]]:
    boxes = [(0, 0, width, height)]
    if grid <= 1:
        return boxes
    step_x = width / grid
    step_y = height / grid
    for gy in range(grid):
        for gx in range(grid):
            x1 = int(round(gx * step_x))
            y1 = int(round(gy * step_y))
            x2 = int(round((gx + 1) * step_x))
            y2 = int(round((gy + 1) * step_y))
            boxes.append((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
    return boxes


def box_center_inside(box_xywh: list[float], crop: tuple[int, int, int, int]) -> bool:
    x, y, w, h = box_xywh
    cx = x + w / 2
    cy = y + h / 2
    x1, y1, x2, y2 = crop
    return x1 <= cx <= x2 and y1 <= cy <= y2


def make_crop_samples(records: list[dict], num_classes: int, bbox_expand: float, grid: int, include_grid: bool) -> list[dict]:
    samples = []
    for rec in records:
        width = int(rec.get("width") or 0)
        height = int(rec.get("height") or 0)
        if width <= 0 or height <= 0:
            with Image.open(rec["image_path"]) as image:
                width, height = image.size
        full_label = np.asarray(rec["labels"], dtype=np.float32)
        samples.append({"record": rec, "crop": (0, 0, width, height), "labels": full_label.tolist(), "kind": "full"})
        for ann in rec["boxes"]:
            label = np.zeros(num_classes, dtype=np.float32)
            label[int(ann["category_id"])] = 1.0
            samples.append(
                {
                    "record": rec,
                    "crop": square_box(tuple(ann["bbox"]), width, height, bbox_expand),
                    "labels": label.tolist(),
                    "kind": "bbox",
                }
            )
        if include_grid:
            for crop in grid_boxes(width, height, grid)[1:]:
                label = np.zeros(num_classes, dtype=np.float32)
                for ann in rec["boxes"]:
                    if box_center_inside(ann["bbox"], crop):
                        label[int(ann["category_id"])] = 1.0
                samples.append({"record": rec, "crop": crop, "labels": label.tolist(), "kind": "grid"})
    return samples


class RoiCropDataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int, augment: bool) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample["record"]["image_path"]).convert("RGB")
        crop = image.crop(sample["crop"])
        if self.augment:
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.4:
                crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.85, 1.25))
            if random.random() < 0.4:
                crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.9, 1.15))
        crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(crop).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
        labels = torch.tensor(sample["labels"], dtype=torch.float32)
        return tensor, labels


def build_model(num_classes: int, name: str, pretrained: bool) -> nn.Module:
    if name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def threshold_to_array(threshold: float | list[float] | np.ndarray, num_classes: int) -> np.ndarray:
    if isinstance(threshold, (list, tuple, np.ndarray)):
        arr = np.asarray(threshold, dtype=np.float32)
        if arr.size == num_classes:
            return arr.reshape(1, num_classes)
    return np.full((1, num_classes), float(threshold), dtype=np.float32)


def threshold_to_text(threshold: float | list[float] | np.ndarray) -> str:
    if isinstance(threshold, (list, tuple, np.ndarray)):
        return "|".join(f"{float(value):.3f}" for value in np.asarray(threshold).reshape(-1))
    return f"{float(threshold):.3f}"


def macro_f1(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tp = np.logical_and(pred_b, target_b).sum(axis=0)
    fp = np.logical_and(pred_b, ~target_b).sum(axis=0)
    fn = np.logical_and(~pred_b, target_b).sum(axis=0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def focal_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    if gamma <= 0:
        return bce.mean()
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    pos_weight: torch.Tensor,
    threshold: float | list[float] | np.ndarray,
    focal_gamma: float,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    probs = []
    targets = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = focal_bce(logits, labels, pos_weight, focal_gamma)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, pos_weight.numel()))
    target = np.concatenate(targets, axis=0) if targets else np.zeros_like(prob)
    pred = (prob >= threshold_to_array(threshold, pos_weight.numel())).astype(np.uint8)
    return {"loss": total_loss / max(1, len(loader.dataset)), "prob": prob, "target": target, "macro_f1": macro_f1(pred, target) if len(prob) else math.nan}


def aggregate_image_probabilities(
    model: nn.Module,
    records: list[dict],
    image_size: int,
    grid: int,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for rec in records:
            with Image.open(rec["image_path"]) as image_raw:
                image = image_raw.convert("RGB")
                width, height = image.size
                crops = []
                for crop_box in grid_boxes(width, height, grid):
                    crop = image.crop(crop_box).resize((image_size, image_size), Image.BILINEAR)
                    arr = np.asarray(crop).astype(np.float32) / 255.0
                    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                    arr = (arr - mean) / std
                    crops.append(np.transpose(arr, (2, 0, 1)))
                crop_probs = []
                for start in range(0, len(crops), batch_size):
                    tensor = torch.tensor(np.stack(crops[start : start + batch_size]), dtype=torch.float32, device=device)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logits = model(tensor)
                    crop_probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
                probs.append(np.concatenate(crop_probs, axis=0).max(axis=0))
                targets.append(np.asarray(rec["labels"], dtype=np.float32))
    return np.stack(probs).reshape(len(records), num_classes), np.stack(targets).reshape(len(records), num_classes)


def tune_threshold(prob: np.ndarray, target: np.ndarray, class_names: list[str]) -> tuple[list[float], pd.DataFrame]:
    rows = []
    thresholds = []
    grid = np.arange(0.05, 0.96, 0.05)
    for class_idx, class_name in enumerate(class_names):
        best = (-1.0, 0.5)
        for threshold in grid:
            pred = prob[:, class_idx] >= threshold
            target_c = target[:, class_idx].astype(bool)
            tp = np.logical_and(pred, target_c).sum()
            fp = np.logical_and(pred, ~target_c).sum()
            fn = np.logical_and(~pred, target_c).sum()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            rows.append({"class": class_name, "threshold": round(float(threshold), 3), "class_f1": f1})
            if f1 > best[0]:
                best = (float(f1), float(threshold))
        thresholds.append(best[1])
    pred = (prob >= threshold_to_array(thresholds, len(class_names))).astype(np.uint8)
    rows.append({"class": "__overall__", "threshold": threshold_to_text(thresholds), "macro_f1": macro_f1(pred, target)})
    return thresholds, pd.DataFrame(rows)


def per_class_metrics(prob: np.ndarray, target: np.ndarray, class_names: list[str], threshold: list[float]) -> pd.DataFrame:
    pred = prob >= threshold_to_array(threshold, len(class_names))
    rows = []
    for idx, name in enumerate(class_names):
        p = pred[:, idx].astype(bool)
        t = target[:, idx].astype(bool)
        tp = int(np.logical_and(p, t).sum())
        fp = int(np.logical_and(p, ~t).sum())
        fn = int(np.logical_and(~p, t).sum())
        tn = int(np.logical_and(~p, ~t).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        rows.append({"class": name, "support": int(t.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def save_probability_table(records: list[dict], class_names: list[str], prob: np.ndarray, target: np.ndarray, path: Path) -> None:
    rows = []
    for idx, rec in enumerate(records):
        row = {"image_id": rec["image_id"], "file_name": rec["file_name"], "image_path": rec["image_path"]}
        for class_idx, class_name in enumerate(class_names):
            row[f"target_{class_name}"] = int(target[idx, class_idx])
            row[f"prob_{class_name}"] = float(prob[idx, class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_plots(history: list[dict], output_root: Path) -> None:
    if not history:
        return
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val crop")
    axes[0].legend()
    axes[0].set_title("ROI crop loss")
    axes[1].plot(df["epoch"], df["val_image_macro_f1"], label="val image macro-F1")
    axes[1].plot(df["epoch"], df["test_image_macro_f1"], label="test image macro-F1")
    axes[1].legend()
    axes[1].set_title("Image-level max-crop macro-F1")
    plt.tight_layout()
    fig.savefig(figures / "roi_disease_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ROI/grid-crop DENTEX disease classifier.")
    parser.add_argument("--coco-json", type=Path, default=ROOT / "data/processed/coco/dentex/disease.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--bbox-expand", type=float, default=2.2)
    parser.add_argument("--train-grid", type=int, default=2)
    parser.add_argument("--inference-grid", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = args.output_root / "tables"
    checkpoints = args.output_root / "checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    records, class_names = load_records(args.coco_json, args.image_root)
    train_records, val_records, test_records = split_records(records, args.train_fraction, args.val_fraction, split_seed)
    train_samples = make_crop_samples(train_records, len(class_names), args.bbox_expand, args.train_grid, include_grid=True)
    val_samples = make_crop_samples(val_records, len(class_names), args.bbox_expand, args.train_grid, include_grid=False)
    labels = np.asarray([sample["labels"] for sample in train_samples], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight_np = np.clip(negatives / np.maximum(positives, 1), 1.0, args.pos_weight_cap).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = build_model(len(class_names), args.model, args.pretrained_imagenet).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    train_sampler = None
    train_shuffle = True
    if args.use_weighted_sampler:
        sample_weights = 1.0 + (labels * pos_weight_np.reshape(1, -1)).max(axis=1)
        train_sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        train_shuffle = False
    loaders = {
        "train": DataLoader(RoiCropDataset(train_samples, args.image_size, True), batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val_crop": DataLoader(RoiCropDataset(val_samples, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }

    ckpt_path = checkpoints / "dentex_roi_disease.pt"
    best_path = checkpoints / "dentex_roi_disease_best.pt"
    start_epoch = 1
    best_f1 = -1.0
    best_threshold: list[float] = [0.5] * len(class_names)
    history: list[dict] = []
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_val_macro_f1", best_f1))
        best_threshold = list(checkpoint.get("best_threshold", best_threshold))
        history = list(checkpoint.get("history", []))
        print(f"resuming_roi_classifier_from_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        train_stats = run_epoch(model, loaders["train"], device, optimizer, scaler, pos_weight, best_threshold, args.focal_gamma)
        val_crop_stats = run_epoch(model, loaders["val_crop"], device, None, scaler, pos_weight, 0.5, args.focal_gamma)
        val_prob, val_target = aggregate_image_probabilities(model, val_records, args.image_size, args.inference_grid, device, args.batch_size, len(class_names))
        test_prob, test_target = aggregate_image_probabilities(model, test_records, args.image_size, args.inference_grid, device, args.batch_size, len(class_names))
        threshold, threshold_df = tune_threshold(val_prob, val_target, class_names)
        val_pred = (val_prob >= threshold_to_array(threshold, len(class_names))).astype(np.uint8)
        test_pred = (test_prob >= threshold_to_array(threshold, len(class_names))).astype(np.uint8)
        val_f1 = macro_f1(val_pred, val_target)
        test_f1 = macro_f1(test_pred, test_target)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_crop_stats["loss"],
            "train_crop_macro_f1": train_stats["macro_f1"],
            "val_crop_macro_f1": val_crop_stats["macro_f1"],
            "val_image_macro_f1": val_f1,
            "test_image_macro_f1": test_f1,
            "threshold": threshold_to_text(threshold),
            "seconds": time.perf_counter() - start,
        }
        is_best = val_f1 > best_f1
        row["is_best"] = int(is_best)
        if is_best:
            best_f1 = float(val_f1)
            best_threshold = list(threshold)
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "class_names": class_names,
            "config": vars(args),
            "split_seed": split_seed,
            "best_val_macro_f1": best_f1,
            "best_threshold": best_threshold,
            "pos_weight": pos_weight_np.tolist(),
        }
        torch.save(checkpoint, ckpt_path)
        if is_best:
            torch.save(checkpoint, best_path)
            threshold_df.to_csv(tables / "roi_threshold_tuning_best_validation.csv", index=False)
            per_class_metrics(val_prob, val_target, class_names, threshold).to_csv(tables / "roi_val_per_class_best.csv", index=False)
            per_class_metrics(test_prob, test_target, class_names, threshold).to_csv(tables / "roi_test_per_class_best.csv", index=False)
        pd.DataFrame(history).to_csv(tables / "roi_disease_epoch_history.csv", index=False)
        print(
            f"epoch {epoch:02d}/{args.epochs}: train_loss={train_stats['loss']:.4f} "
            f"val_image_f1={val_f1:.4f} test_image_f1={test_f1:.4f} threshold={threshold_to_text(threshold)} "
            f"best_val_f1={best_f1:.4f}",
            flush=True,
        )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False) if best_path.exists() else torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    threshold = list(best_checkpoint.get("best_threshold", best_threshold))
    val_prob, val_target = aggregate_image_probabilities(model, val_records, args.image_size, args.inference_grid, device, args.batch_size, len(class_names))
    test_prob, test_target = aggregate_image_probabilities(model, test_records, args.image_size, args.inference_grid, device, args.batch_size, len(class_names))
    val_pred = (val_prob >= threshold_to_array(threshold, len(class_names))).astype(np.uint8)
    test_pred = (test_prob >= threshold_to_array(threshold, len(class_names))).astype(np.uint8)
    summary = {
        "checkpoint": str(best_path if best_path.exists() else ckpt_path),
        "best_epoch": int(best_checkpoint.get("epoch", args.epochs)),
        "threshold": threshold_to_text(threshold),
        "val_macro_f1": macro_f1(val_pred, val_target),
        "test_macro_f1": macro_f1(test_pred, test_target),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "train_crops": len(train_samples),
        "class_names": "|".join(class_names),
        "pos_weight": "|".join(f"{v:.4f}" for v in pos_weight_np),
        "model": args.model,
        "seed": args.seed,
        "split_seed": split_seed,
        "inference_grid": args.inference_grid,
    }
    pd.DataFrame([summary]).to_csv(tables / "roi_disease_summary.csv", index=False)
    per_class_metrics(test_prob, test_target, class_names, threshold).to_csv(tables / "roi_test_per_class.csv", index=False)
    save_probability_table(val_records, class_names, val_prob, val_target, tables / "roi_val_probabilities.csv")
    save_probability_table(test_records, class_names, test_prob, test_target, tables / "roi_test_probabilities.csv")
    save_plots(history, args.output_root)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DENTEX_ROI_DISEASE_CLASSIFIER_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
