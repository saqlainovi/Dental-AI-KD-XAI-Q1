from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dentex_roi_disease_classifier import (
    ROOT,
    RoiCropDataset,
    box_center_inside,
    build_model,
    focal_bce,
    grid_boxes,
    load_records,
    square_box,
    split_records,
    threshold_to_array,
    threshold_to_text,
)


def resolve_target_index(target_class: str, class_names: list[str]) -> int:
    if target_class.isdigit():
        idx = int(target_class)
        if 0 <= idx < len(class_names):
            return idx
    if target_class in class_names:
        return class_names.index(target_class)
    raise ValueError(f"Unknown target class {target_class!r}. Available classes: {class_names}")


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def image_size(rec: dict) -> tuple[int, int]:
    width = int(rec.get("width") or 0)
    height = int(rec.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    with Image.open(rec["image_path"]) as image:
        return image.size


def target_inside_crop(rec: dict, crop: tuple[int, int, int, int], target_idx: int) -> bool:
    return any(int(ann["category_id"]) == target_idx and box_center_inside(ann["bbox"], crop) for ann in rec["boxes"])


def dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen = set()
    out = []
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        out.append(box)
    return out


def make_training_samples(
    records: list[dict],
    target_idx: int,
    bbox_expands: list[float],
    train_grids: list[int],
    include_full: bool,
    max_negative_grids_per_image: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    for rec in records:
        width, height = image_size(rec)
        image_label = float(rec["labels"][target_idx] > 0.5)
        if include_full:
            samples.append({"record": rec, "crop": (0, 0, width, height), "labels": [image_label], "kind": "full"})

        target_anns = [ann for ann in rec["boxes"] if int(ann["category_id"]) == target_idx]
        other_anns = [ann for ann in rec["boxes"] if int(ann["category_id"]) != target_idx]
        for ann in target_anns:
            for expand in bbox_expands:
                crop = square_box(tuple(ann["bbox"]), width, height, expand)
                samples.append({"record": rec, "crop": crop, "labels": [1.0], "kind": "target_bbox"})

        for ann in other_anns:
            for expand in bbox_expands[:2]:
                crop = square_box(tuple(ann["bbox"]), width, height, expand)
                if not target_inside_crop(rec, crop, target_idx):
                    samples.append({"record": rec, "crop": crop, "labels": [0.0], "kind": "other_bbox"})

        grid_candidates = []
        for grid in train_grids:
            for crop in grid_boxes(width, height, grid)[1:]:
                label = 1.0 if target_inside_crop(rec, crop, target_idx) else 0.0
                grid_candidates.append((crop, label))
        positives = [(crop, label) for crop, label in grid_candidates if label > 0.5]
        negatives = [(crop, label) for crop, label in grid_candidates if label <= 0.5]
        if max_negative_grids_per_image > 0 and len(negatives) > max_negative_grids_per_image:
            negatives = rng.sample(negatives, max_negative_grids_per_image)
        for crop, label in positives + negatives:
            samples.append({"record": rec, "crop": crop, "labels": [label], "kind": "grid"})
    return samples


def make_inference_boxes(width: int, height: int, grids: list[int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for grid in grids:
        boxes.extend(grid_boxes(width, height, grid))
    return dedupe_boxes(boxes)


def binary_f1(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> tuple[float, float, float, int, int, int, int]:
    p = pred.astype(bool).reshape(-1)
    t = target.astype(bool).reshape(-1)
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(eps, precision + recall)
    return f1, precision, recall, tp, fp, fn, tn


def tune_threshold(prob: np.ndarray, target: np.ndarray, grid_step: float) -> tuple[float, pd.DataFrame]:
    thresholds = np.arange(grid_step, 1.0, grid_step)
    rows = []
    best_key = (-1.0, -1.0, -1.0)
    best_threshold = 0.5
    for threshold in thresholds:
        pred = (prob.reshape(-1) >= threshold).astype(np.uint8)
        f1, precision, recall, tp, fp, fn, tn = binary_f1(pred, target)
        row = {
            "threshold": round(float(threshold), 4),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        rows.append(row)
        key = (float(f1), float(precision), float(recall))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, pd.DataFrame(rows)


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pos_weight: torch.Tensor,
    threshold: float,
    focal_gamma: float,
) -> dict:
    model.train(True)
    total_loss = 0.0
    probs = []
    targets = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = focal_bce(logits, labels, pos_weight, focal_gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
    prob = np.concatenate(probs, axis=0).reshape(-1) if probs else np.zeros((0,), dtype=np.float32)
    target = np.concatenate(targets, axis=0).reshape(-1) if targets else np.zeros_like(prob)
    pred = (prob >= threshold).astype(np.uint8)
    f1, precision, recall, *_ = binary_f1(pred, target)
    return {"loss": total_loss / max(1, len(loader.dataset)), "f1": f1, "precision": precision, "recall": recall}


def preprocess_crop(crop: Image.Image, image_size: int) -> np.ndarray:
    crop = crop.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(crop).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    return np.transpose((arr - mean) / std, (2, 0, 1))


def aggregate_image_probabilities(
    model: torch.nn.Module,
    records: list[dict],
    target_idx: int,
    image_size_value: int,
    inference_grids: list[int],
    device: torch.device,
    batch_size: int,
    aggregate: str,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for rec in records:
            with Image.open(rec["image_path"]) as image_raw:
                image = image_raw.convert("RGB")
                width, height = image.size
                crops = [preprocess_crop(image.crop(crop_box), image_size_value) for crop_box in make_inference_boxes(width, height, inference_grids)]
            crop_probs = []
            for start in range(0, len(crops), batch_size):
                tensor = torch.tensor(np.stack(crops[start : start + batch_size]), dtype=torch.float32, device=device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(tensor)
                crop_probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy().reshape(-1))
            flat = np.concatenate(crop_probs) if crop_probs else np.asarray([0.0], dtype=np.float32)
            if aggregate == "topk_mean":
                k = min(max(1, topk), len(flat))
                prob = float(np.sort(flat)[-k:].mean())
            elif aggregate == "mean_max":
                k = min(max(1, topk), len(flat))
                prob = float(0.5 * flat.max() + 0.5 * np.sort(flat)[-k:].mean())
            else:
                prob = float(flat.max())
            probs.append(prob)
            targets.append(float(rec["labels"][target_idx] > 0.5))
    return np.asarray(probs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def save_probability_table(records: list[dict], target_name: str, prob: np.ndarray, target: np.ndarray, path: Path) -> None:
    rows = []
    for idx, rec in enumerate(records):
        rows.append(
            {
                "image_id": rec["image_id"],
                "file_name": rec["file_name"],
                "image_path": rec["image_path"],
                f"target_{target_name}": int(target[idx] > 0.5),
                f"prob_{target_name}": float(prob[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_metrics(path: Path, target_name: str, prob: np.ndarray, target: np.ndarray, threshold: float) -> None:
    pred = (prob >= threshold).astype(np.uint8)
    f1, precision, recall, tp, fp, fn, tn = binary_f1(pred, target)
    pd.DataFrame(
        [
            {
                "class": target_name,
                "support": int(target.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ]
    ).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary annotation-crop DENTEX disease specialist.")
    parser.add_argument("--coco-json", type=Path, default=ROOT / "data/processed/coco/dentex/disease.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-cap", type=float, default=24.0)
    parser.add_argument("--pos-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--positive-sample-weight", type=float, default=8.0)
    parser.add_argument("--bbox-expands", default="1.4,2.0,3.0")
    parser.add_argument("--train-grids", default="2,3")
    parser.add_argument("--inference-grids", default="1,2,3,4")
    parser.add_argument("--max-negative-grids-per-image", type=int, default=8)
    parser.add_argument("--aggregate", choices=["max", "topk_mean", "mean_max"], default="mean_max")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--include-full", action="store_true")
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
    target_idx = resolve_target_index(args.target_class, class_names)
    target_name = class_names[target_idx]
    train_records, val_records, test_records = split_records(records, args.train_fraction, args.val_fraction, split_seed)
    train_samples = make_training_samples(
        train_records,
        target_idx,
        parse_float_list(args.bbox_expands),
        parse_int_list(args.train_grids),
        args.include_full,
        args.max_negative_grids_per_image,
        args.seed,
    )
    labels = np.asarray([sample["labels"][0] for sample in train_samples], dtype=np.float32)
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    pos_weight_value = min(negatives / max(1.0, positives), args.pos_weight_cap) * args.pos_weight_multiplier
    pos_weight_np = np.asarray([pos_weight_value], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = build_model(1, args.model, args.pretrained_imagenet).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        weights = np.where(labels > 0.5, args.positive_sample_weight, 1.0)
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        RoiCropDataset(train_samples, args.image_size, True),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    ckpt_path = checkpoints / f"dentex_crop_{target_name}.pt"
    best_path = checkpoints / f"dentex_crop_{target_name}_best.pt"
    start_epoch = 1
    best_f1 = -1.0
    best_threshold = 0.5
    history: list[dict] = []
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_val_f1", best_f1))
        best_threshold = float(checkpoint.get("best_threshold", best_threshold))
        history = list(checkpoint.get("history", []))
        print(f"resuming_crop_specialist_from_epoch={start_epoch}", flush=True)

    inference_grids = parse_int_list(args.inference_grids)
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        train_stats = run_train_epoch(model, train_loader, device, optimizer, scaler, pos_weight, best_threshold, args.focal_gamma)
        val_prob, val_target = aggregate_image_probabilities(model, val_records, target_idx, args.image_size, inference_grids, device, args.batch_size, args.aggregate, args.topk)
        test_prob, test_target = aggregate_image_probabilities(model, test_records, target_idx, args.image_size, inference_grids, device, args.batch_size, args.aggregate, args.topk)
        threshold, threshold_df = tune_threshold(val_prob, val_target, args.threshold_step)
        val_pred = (val_prob >= threshold).astype(np.uint8)
        test_pred = (test_prob >= threshold).astype(np.uint8)
        val_f1, val_precision, val_recall, *_ = binary_f1(val_pred, val_target)
        test_f1, test_precision, test_recall, *_ = binary_f1(test_pred, test_target)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_f1": train_stats["f1"],
            "threshold": threshold_to_text(threshold),
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "seconds": time.perf_counter() - start,
        }
        is_best = val_f1 > best_f1
        row["is_best"] = int(is_best)
        if is_best:
            best_f1 = float(val_f1)
            best_threshold = float(threshold)
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "class_names": [target_name],
            "target_class": target_name,
            "target_class_index": target_idx,
            "config": vars(args),
            "split_seed": split_seed,
            "best_val_f1": best_f1,
            "best_threshold": best_threshold,
            "pos_weight": pos_weight_np.tolist(),
        }
        torch.save(checkpoint, ckpt_path)
        if is_best:
            torch.save(checkpoint, best_path)
            threshold_df.to_csv(tables / "crop_threshold_tuning_best_validation.csv", index=False)
            save_metrics(tables / "crop_val_per_class_best.csv", target_name, val_prob, val_target, threshold)
            save_metrics(tables / "crop_test_per_class_best.csv", target_name, test_prob, test_target, threshold)
        pd.DataFrame(history).to_csv(tables / "crop_disease_epoch_history.csv", index=False)
        print(
            f"epoch {epoch:02d}/{args.epochs}: target={target_name} train_loss={train_stats['loss']:.4f} "
            f"val_f1={val_f1:.4f} test_f1={test_f1:.4f} threshold={threshold:.3f} best_val_f1={best_f1:.4f}",
            flush=True,
        )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False) if best_path.exists() else torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    threshold = float(best_checkpoint.get("best_threshold", best_threshold))
    val_prob, val_target = aggregate_image_probabilities(model, val_records, target_idx, args.image_size, inference_grids, device, args.batch_size, args.aggregate, args.topk)
    test_prob, test_target = aggregate_image_probabilities(model, test_records, target_idx, args.image_size, inference_grids, device, args.batch_size, args.aggregate, args.topk)
    val_pred = (val_prob >= threshold).astype(np.uint8)
    test_pred = (test_prob >= threshold).astype(np.uint8)
    val_f1, val_precision, val_recall, *_ = binary_f1(val_pred, val_target)
    test_f1, test_precision, test_recall, *_ = binary_f1(test_pred, test_target)
    summary = {
        "checkpoint": str(best_path if best_path.exists() else ckpt_path),
        "target_class": target_name,
        "target_class_index": target_idx,
        "best_epoch": int(best_checkpoint.get("epoch", args.epochs)),
        "threshold": threshold_to_text(threshold),
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "train_samples": len(train_samples),
        "train_sample_positives": int(positives),
        "train_sample_negatives": int(negatives),
        "pos_weight": f"{float(pos_weight_np[0]):.4f}",
        "model": args.model,
        "aggregate": args.aggregate,
        "topk": args.topk,
        "image_size": args.image_size,
        "bbox_expands": args.bbox_expands,
        "train_grids": args.train_grids,
        "inference_grids": args.inference_grids,
        "seed": args.seed,
        "split_seed": split_seed,
    }
    pd.DataFrame([summary]).to_csv(tables / "crop_disease_summary.csv", index=False)
    save_metrics(tables / "crop_val_per_class.csv", target_name, val_prob, val_target, threshold)
    save_metrics(tables / "crop_test_per_class.csv", target_name, test_prob, test_target, threshold)
    save_probability_table(val_records, target_name, val_prob, val_target, tables / "crop_val_probabilities.csv")
    save_probability_table(test_records, target_name, test_prob, test_target, tables / "crop_test_probabilities.csv")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DENTEX_BINARY_CROP_SPECIALIST_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
