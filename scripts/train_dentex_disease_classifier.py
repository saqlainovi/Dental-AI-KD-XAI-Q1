from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dentex_student_segmentation import GPUEpochSampler, prepare_segmentation_pair


def resolve_image_path(file_name: str, image_root: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() and path.exists():
        return path
    for candidate in [image_root / file_name, ROOT / file_name, ROOT / "data" / "raw" / "dentex" / file_name]:
        if candidate.exists():
            return candidate
    return image_root / file_name


def load_disease_records(coco_json: Path, image_root: Path) -> tuple[list[dict], list[str]]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = sorted(coco.get("categories", []), key=lambda item: int(item.get("id", 0)))
    class_names = [str(item.get("name", f"disease_{idx}")) for idx, item in enumerate(categories)]
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
                boxes[image_id].append([float(v) for v in ann["bbox"]])

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
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


class DiseaseDataset(Dataset):
    def __init__(self, records: list[dict], image_size: int, augment: bool) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[index]
        image = Image.open(rec["image_path"]).convert("RGB")
        mask = Image.new("L", image.size, 0)
        image, _mask = prepare_segmentation_pair(image, mask, self.image_size, augment=self.augment, use_clahe=True)
        arr = np.asarray(image).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
        labels = torch.tensor(rec["labels"], dtype=torch.float32)
        return tensor, labels


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
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


def per_class_metrics(prob: np.ndarray, target: np.ndarray, class_names: list[str], threshold: float | list[float] | np.ndarray) -> pd.DataFrame:
    threshold_arr = threshold_to_array(threshold, len(class_names))
    pred = prob >= threshold_arr
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


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, gamma: float) -> torch.Tensor:
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
                loss = focal_bce_with_logits(logits, labels, pos_weight=pos_weight, gamma=focal_gamma)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
    prob_arr = np.concatenate(probs, axis=0) if probs else np.zeros((0, pos_weight.numel()))
    target_arr = np.concatenate(targets, axis=0) if targets else np.zeros_like(prob_arr)
    pred_arr = (prob_arr >= threshold_to_array(threshold, pos_weight.numel())).astype(np.uint8)
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "macro_f1": macro_f1(pred_arr, target_arr) if len(prob_arr) else math.nan,
        "exact_match": float((pred_arr == target_arr).all(axis=1).mean()) if len(prob_arr) else math.nan,
        "prob": prob_arr,
        "target": target_arr,
    }


def tune_threshold(prob: np.ndarray, target: np.ndarray, mode: str, class_names: list[str]) -> tuple[float | list[float], pd.DataFrame]:
    rows = []
    grid = np.arange(0.05, 0.96, 0.05)
    if mode == "per_class":
        thresholds = []
        for class_idx, class_name in enumerate(class_names):
            best_threshold = 0.5
            best_f1 = -1.0
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
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_threshold = float(threshold)
            thresholds.append(best_threshold)
        pred = (prob >= threshold_to_array(thresholds, len(class_names))).astype(np.uint8)
        rows.append({"class": "__overall__", "threshold": threshold_to_text(thresholds), "class_f1": math.nan, "macro_f1": macro_f1(pred, target), "exact_match": float((pred == target).all(axis=1).mean())})
        return thresholds, pd.DataFrame(rows)
    for threshold in grid:
        pred = (prob >= threshold).astype(np.uint8)
        rows.append({"class": "__overall__", "threshold": round(float(threshold), 3), "macro_f1": macro_f1(pred, target), "exact_match": float((pred == target).all(axis=1).mean())})
    df = pd.DataFrame(rows).sort_values(["macro_f1", "exact_match"], ascending=False)
    return float(df.iloc[0]["threshold"]), df


def save_plots(history: list[dict], output_root: Path) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="validation")
    axes[0].set_title("Disease Classification Loss")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="val macro-F1")
    axes[1].plot(df["epoch"], df["test_macro_f1"], label="test macro-F1")
    axes[1].set_title("Disease Classification Macro-F1")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(figures / "disease_classifier_training_curves.png", dpi=180)
    plt.close(fig)


def save_probability_table(
    records: list[dict],
    class_names: list[str],
    prob: np.ndarray,
    target: np.ndarray,
    path: Path,
) -> None:
    rows = []
    for idx, rec in enumerate(records):
        row = {
            "image_id": rec["image_id"],
            "file_name": rec["file_name"],
            "image_path": rec["image_path"],
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"target_{class_name}"] = int(target[idx, class_idx])
            row[f"prob_{class_name}"] = float(prob[idx, class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dedicated DENTEX multilabel disease classifier.")
    parser.add_argument("--coco-json", type=Path, default=ROOT / "data/processed/coco/dentex/disease.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/disease_classifier_v1"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="Optional fixed split seed for multi-seed ensembling.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = args.output_root / "tables"
    checkpoints = args.output_root / "checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    records, class_names = load_disease_records(args.coco_json, args.image_root)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_records, val_records, test_records = split_records(records, args.train_fraction, args.val_fraction, split_seed)
    split_rows = []
    for split, rows in [("train", train_records), ("validation", val_records), ("test", test_records)]:
        for row in rows:
            split_rows.append({"split": split, "image_id": row["image_id"], "file_name": row["file_name"], "image_path": row["image_path"], **{class_names[i]: row["labels"][i] for i in range(len(class_names))}})
    pd.DataFrame(split_rows).to_csv(tables / "disease_classifier_split_manifest.csv", index=False)

    labels = np.asarray([row["labels"] for row in train_records], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight_np = np.clip(negatives / np.maximum(positives, 1), 1.0, args.pos_weight_cap).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = build_model(len(class_names), args.pretrained_imagenet).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    train_sampler = None
    train_shuffle = True
    if args.use_weighted_sampler:
        train_labels = np.asarray([row["labels"] for row in train_records], dtype=np.float32)
        sample_weights = 1.0 + (train_labels * pos_weight_np.reshape(1, -1)).max(axis=1)
        train_sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        train_shuffle = False

    loaders = {
        "train": DataLoader(DiseaseDataset(train_records, args.image_size, True), batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(DiseaseDataset(val_records, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "test": DataLoader(DiseaseDataset(test_records, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }

    ckpt_path = checkpoints / "dentex_disease_swin_t.pt"
    best_path = checkpoints / "dentex_disease_swin_t_best.pt"
    start_epoch = 1
    best_f1 = -1.0
    best_threshold = 0.5
    history: list[dict] = []
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_val_macro_f1", best_f1))
        best_threshold = checkpoint.get("best_threshold", best_threshold)
        history = list(checkpoint.get("history", []))
        print(f"resuming_disease_classifier_from_epoch={start_epoch}", flush=True)

    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        sampler.start()
        train_stats = run_epoch(model, loaders["train"], device, optimizer, scaler, pos_weight, best_threshold, args.focal_gamma)
        val_raw = run_epoch(model, loaders["val"], device, None, scaler, pos_weight, 0.5, args.focal_gamma)
        threshold, threshold_df = tune_threshold(val_raw["prob"], val_raw["target"], args.threshold_mode, class_names)
        val_stats = run_epoch(model, loaders["val"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
        test_stats = run_epoch(model, loaders["test"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu = sampler.stop()
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "test_loss": test_stats["loss"],
            "threshold": threshold_to_text(threshold),
            "train_macro_f1": train_stats["macro_f1"],
            "val_macro_f1": val_stats["macro_f1"],
            "test_macro_f1": test_stats["macro_f1"],
            "val_exact_match": val_stats["exact_match"],
            "test_exact_match": test_stats["exact_match"],
            "seconds": elapsed,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else math.nan,
            **gpu,
        }
        is_best = val_stats["macro_f1"] > best_f1
        row["is_best"] = int(is_best)
        if is_best:
            best_f1 = float(val_stats["macro_f1"])
            best_threshold = threshold
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
            threshold_df.to_csv(tables / "disease_threshold_tuning_best_validation.csv", index=False)
            per_class_metrics(val_stats["prob"], val_stats["target"], class_names, threshold).to_csv(tables / "disease_val_per_class_best.csv", index=False)
            per_class_metrics(test_stats["prob"], test_stats["target"], class_names, threshold).to_csv(tables / "disease_test_per_class_best.csv", index=False)
        pd.DataFrame(history).to_csv(tables / "disease_classifier_epoch_history.csv", index=False)
        print(
            f"epoch {epoch:02d}/{args.epochs}: train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_f1={val_stats['macro_f1']:.4f} test_f1={test_stats['macro_f1']:.4f} threshold={threshold_to_text(threshold)} "
            f"power_mean_w={gpu.get('power_draw_w_mean', math.nan):.2f} gpu_util_mean={gpu.get('gpu_util_percent_mean', math.nan):.1f}% "
            f"mem_max_mb={gpu.get('memory_used_mb_max', math.nan):.0f} seconds={elapsed:.1f} best_val_f1={best_f1:.4f}",
            flush=True,
        )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False) if best_path.exists() else torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    threshold = best_checkpoint.get("best_threshold", best_threshold)
    val_stats = run_epoch(model, loaders["val"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
    test_stats = run_epoch(model, loaders["test"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
    summary = {
        "checkpoint": str(best_path if best_path.exists() else ckpt_path),
        "best_epoch": int(best_checkpoint.get("epoch", args.epochs)),
        "threshold": threshold_to_text(threshold),
        "val_macro_f1": val_stats["macro_f1"],
        "test_macro_f1": test_stats["macro_f1"],
        "val_exact_match": val_stats["exact_match"],
        "test_exact_match": test_stats["exact_match"],
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "class_names": "|".join(class_names),
        "pos_weight": "|".join(f"{v:.4f}" for v in pos_weight_np),
        "threshold_mode": args.threshold_mode,
        "focal_gamma": args.focal_gamma,
        "use_weighted_sampler": args.use_weighted_sampler,
        "seed": args.seed,
        "split_seed": split_seed,
    }
    pd.DataFrame([summary]).to_csv(tables / "disease_classifier_summary.csv", index=False)
    per_class_metrics(test_stats["prob"], test_stats["target"], class_names, threshold).to_csv(tables / "disease_test_per_class.csv", index=False)
    save_probability_table(val_records, class_names, val_stats["prob"], val_stats["target"], tables / "disease_val_probabilities.csv")
    save_probability_table(test_records, class_names, test_stats["prob"], test_stats["target"], tables / "disease_test_probabilities.csv")
    save_plots(history, args.output_root)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DENTEX_DISEASE_CLASSIFIER_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
