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
from torch.utils.data import DataLoader, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dentex_disease_classifier import (
    ROOT,
    DiseaseDataset,
    build_model,
    load_disease_records,
    macro_f1,
    per_class_metrics,
    run_epoch,
    save_plots,
    save_probability_table,
    split_records,
    threshold_to_text,
    tune_threshold,
)


def resolve_target_index(target_class: str, class_names: list[str]) -> int:
    if target_class.isdigit():
        idx = int(target_class)
        if 0 <= idx < len(class_names):
            return idx
    if target_class in class_names:
        return class_names.index(target_class)
    raise ValueError(f"Unknown target class {target_class!r}. Available: {class_names}")


def binary_records(records: list[dict], class_idx: int) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        row = dict(rec)
        row["labels"] = [float(rec["labels"][class_idx])]
        rows.append(row)
    return rows


def write_split_manifest(path: Path, splits: dict[str, list[dict]], target_name: str) -> None:
    rows = []
    for split, records in splits.items():
        for rec in records:
            rows.append(
                {
                    "split": split,
                    "image_id": rec["image_id"],
                    "file_name": rec["file_name"],
                    "image_path": rec["image_path"],
                    target_name: int(float(rec["labels"][0]) > 0.5),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary DENTEX disease specialist for one weak class.")
    parser.add_argument("--coco-json", type=Path, default=ROOT / "data/processed/coco/dentex/disease.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-class", required=True, help="Class name or class index, e.g. periapical_lesion or 2.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--pos-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--positive-sample-weight", type=float, default=8.0)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-imagenet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = args.output_root / "tables"
    checkpoints = args.output_root / "checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    records, all_class_names = load_disease_records(args.coco_json, args.image_root)
    class_idx = resolve_target_index(args.target_class, all_class_names)
    target_name = all_class_names[class_idx]
    records = binary_records(records, class_idx)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_records, val_records, test_records = split_records(records, args.train_fraction, args.val_fraction, split_seed)
    class_names = [target_name]
    write_split_manifest(
        tables / "binary_disease_split_manifest.csv",
        {"train": train_records, "val": val_records, "test": test_records},
        target_name,
    )

    train_labels = np.asarray([rec["labels"][0] for rec in train_records], dtype=np.float32)
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight_value = min(negatives / max(positives, 1.0), args.pos_weight_cap) * args.pos_weight_multiplier
    pos_weight_np = np.asarray([pos_weight_value], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = build_model(1, args.pretrained_imagenet).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    train_sampler = None
    train_shuffle = True
    if args.use_weighted_sampler:
        sample_weights = np.where(train_labels > 0.5, args.positive_sample_weight, 1.0)
        train_sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    loaders = {
        "train": DataLoader(DiseaseDataset(train_records, args.image_size, True), batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(DiseaseDataset(val_records, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "test": DataLoader(DiseaseDataset(test_records, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }

    ckpt_path = checkpoints / f"dentex_binary_{target_name}.pt"
    best_path = checkpoints / f"dentex_binary_{target_name}_best.pt"
    start_epoch = 1
    best_f1 = -1.0
    best_threshold: float | list[float] = 0.5
    history: list[dict] = []
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_val_f1", best_f1))
        best_threshold = checkpoint.get("best_threshold", best_threshold)
        history = list(checkpoint.get("history", []))
        print(f"resuming_binary_specialist_from_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        train_stats = run_epoch(model, loaders["train"], device, optimizer, scaler, pos_weight, best_threshold, args.focal_gamma)
        val_raw = run_epoch(model, loaders["val"], device, None, scaler, pos_weight, 0.5, args.focal_gamma)
        threshold, threshold_df = tune_threshold(val_raw["prob"], val_raw["target"], "per_class", class_names)
        val_stats = run_epoch(model, loaders["val"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
        test_stats = run_epoch(model, loaders["test"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "test_loss": test_stats["loss"],
            "threshold": threshold_to_text(threshold),
            "train_f1": train_stats["macro_f1"],
            "val_f1": val_stats["macro_f1"],
            "test_f1": test_stats["macro_f1"],
            "val_macro_f1": val_stats["macro_f1"],
            "test_macro_f1": test_stats["macro_f1"],
            "val_exact_match": val_stats["exact_match"],
            "test_exact_match": test_stats["exact_match"],
            "seconds": time.perf_counter() - start,
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
            "all_class_names": all_class_names,
            "target_class": target_name,
            "target_class_index": class_idx,
            "config": vars(args),
            "split_seed": split_seed,
            "best_val_f1": best_f1,
            "best_threshold": best_threshold,
            "pos_weight": pos_weight_np.tolist(),
        }
        torch.save(checkpoint, ckpt_path)
        if is_best:
            torch.save(checkpoint, best_path)
            threshold_df.to_csv(tables / "binary_threshold_tuning_best_validation.csv", index=False)
            per_class_metrics(val_stats["prob"], val_stats["target"], class_names, threshold).to_csv(tables / "binary_val_per_class_best.csv", index=False)
            per_class_metrics(test_stats["prob"], test_stats["target"], class_names, threshold).to_csv(tables / "binary_test_per_class_best.csv", index=False)
        pd.DataFrame(history).to_csv(tables / "binary_disease_epoch_history.csv", index=False)
        print(
            f"epoch {epoch:02d}/{args.epochs}: target={target_name} train_loss={train_stats['loss']:.4f} "
            f"val_f1={val_stats['macro_f1']:.4f} test_f1={test_stats['macro_f1']:.4f} "
            f"threshold={threshold_to_text(threshold)} best_val_f1={best_f1:.4f}",
            flush=True,
        )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False) if best_path.exists() else torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    threshold = best_checkpoint.get("best_threshold", best_threshold)
    val_stats = run_epoch(model, loaders["val"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
    test_stats = run_epoch(model, loaders["test"], device, None, scaler, pos_weight, threshold, args.focal_gamma)
    summary = {
        "checkpoint": str(best_path if best_path.exists() else ckpt_path),
        "target_class": target_name,
        "target_class_index": class_idx,
        "best_epoch": int(best_checkpoint.get("epoch", args.epochs)),
        "threshold": threshold_to_text(threshold),
        "val_f1": val_stats["macro_f1"],
        "test_f1": test_stats["macro_f1"],
        "val_exact_match": val_stats["exact_match"],
        "test_exact_match": test_stats["exact_match"],
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "train_positives": int(positives),
        "train_negatives": int(negatives),
        "pos_weight": f"{float(pos_weight_np[0]):.4f}",
        "focal_gamma": args.focal_gamma,
        "positive_sample_weight": args.positive_sample_weight,
        "image_size": args.image_size,
        "seed": args.seed,
        "split_seed": split_seed,
    }
    pd.DataFrame([summary]).to_csv(tables / "binary_disease_summary.csv", index=False)
    per_class_metrics(test_stats["prob"], test_stats["target"], class_names, threshold).to_csv(tables / "binary_test_per_class.csv", index=False)
    save_probability_table(val_records, class_names, val_stats["prob"], val_stats["target"], tables / "binary_val_probabilities.csv")
    save_probability_table(test_records, class_names, test_stats["prob"], test_stats["target"], tables / "binary_test_probabilities.csv")
    save_plots(history, args.output_root)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DENTEX_BINARY_DISEASE_SPECIALIST_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
