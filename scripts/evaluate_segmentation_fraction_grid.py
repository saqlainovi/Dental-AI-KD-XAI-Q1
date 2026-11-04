from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint
from generate_final_evaluation_xai import (
    build_tufts_pairs,
    dice_iou,
    load_clean_rows,
    load_internal_test_rows_from_checkpoint,
    parse_threshold_grid,
    predict_probability,
    probability_to_prediction,
    resolve_path,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_models(checkpoint_paths: list[Path], device: torch.device) -> tuple[list[torch.nn.Module], int, dict]:
    models = []
    image_size = None
    first_checkpoint = {}
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        current_image_size = int(config.get("image_size", 512))
        if image_size is None:
            image_size = current_image_size
            first_checkpoint = checkpoint
        elif current_image_size != image_size:
            raise RuntimeError(f"Checkpoint image_size mismatch: {path} has {current_image_size}, expected {image_size}")
        model = build_model_from_checkpoint(checkpoint, num_classes=2).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    if image_size is None:
        raise RuntimeError("No checkpoints were provided.")
    return models, image_size, first_checkpoint


def ensemble_probability(
    models: list[torch.nn.Module],
    image: Image.Image,
    image_size: int,
    device: torch.device,
    tta_horizontal_flip: bool,
) -> np.ndarray:
    probs = [
        predict_probability(model, image, image_size, device, tta_horizontal_flip=tta_horizontal_flip)
        for model in models
    ]
    return np.mean(np.stack(probs, axis=0), axis=0)


def candidate_rows(thresholds: list[float], min_fractions: list[float], max_fractions: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        for min_fraction in min_fractions:
            for max_fraction in max_fractions:
                if max_fraction < min_fraction:
                    continue
                rows.append(
                    {
                        "threshold": float(threshold),
                        "min_pred_fraction": float(min_fraction),
                        "max_pred_fraction": float(max_fraction),
                    }
                )
    return rows


def evaluate_source(
    rows: list[dict],
    source: str,
    models: list[torch.nn.Module],
    image_size: int,
    device: torch.device,
    candidates: list[dict],
    output_dir: Path,
    tta_horizontal_flip: bool,
    max_rows: int,
) -> pd.DataFrame:
    selected = rows[:max_rows] if max_rows else rows
    accum = {
        idx: {
            "dice": [],
            "iou": [],
            "pred_fraction": [],
            "target_fraction": [],
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
        for idx in range(len(candidates))
    }

    for row_idx, row in enumerate(selected, start=1):
        image_path = resolve_path(row.get("image_path", row.get("image", "")))
        mask_path = resolve_path(row.get("clean_mask", row.get("mask_path", "")))
        image = Image.open(image_path).convert("RGB")
        target = np.asarray(Image.open(mask_path).convert("L")) > 0
        prob = ensemble_probability(models, image, image_size, device, tta_horizontal_flip)
        target_fraction = float(target.mean())
        for idx, candidate in enumerate(candidates):
            pred = probability_to_prediction(
                prob,
                candidate["threshold"],
                candidate["min_pred_fraction"],
                candidate["max_pred_fraction"],
            )
            dice, iou = dice_iou(pred, target)
            pred_b = pred.astype(bool)
            target_b = target.astype(bool)
            values = accum[idx]
            values["dice"].append(dice)
            values["iou"].append(iou)
            values["pred_fraction"].append(float(pred_b.mean()))
            values["target_fraction"].append(target_fraction)
            values["tn"] += int(np.logical_and(~pred_b, ~target_b).sum())
            values["fp"] += int(np.logical_and(pred_b, ~target_b).sum())
            values["fn"] += int(np.logical_and(~pred_b, target_b).sum())
            values["tp"] += int(np.logical_and(pred_b, target_b).sum())
        if row_idx == 1 or row_idx % 25 == 0 or row_idx == len(selected):
            print(f"{source}: fraction_grid={row_idx}/{len(selected)}", flush=True)

    out_rows = []
    for idx, candidate in enumerate(candidates):
        values = accum[idx]
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        pixel_dice = (2 * tp + 1e-6) / max(1e-6, 2 * tp + fp + fn + 1e-6)
        pixel_iou = (tp + 1e-6) / max(1e-6, tp + fp + fn + 1e-6)
        out_rows.append(
            {
                "source": source,
                "rows": len(selected),
                **candidate,
                "mean_image_dice": float(np.mean(values["dice"])) if values["dice"] else math.nan,
                "mean_image_iou": float(np.mean(values["iou"])) if values["iou"] else math.nan,
                "pixel_dice": float(pixel_dice),
                "pixel_iou": float(pixel_iou),
                "mean_target_fraction": float(np.mean(values["target_fraction"])) if values["target_fraction"] else math.nan,
                "mean_pred_fraction": float(np.mean(values["pred_fraction"])) if values["pred_fraction"] else math.nan,
                "tn": values["tn"],
                "fp": values["fp"],
                "fn": values["fn"],
                "tp": values["tp"],
            }
        )
    df = pd.DataFrame(out_rows).sort_values(["mean_image_dice", "pixel_dice"], ascending=False)
    df.to_csv(output_dir / f"{source}_fraction_grid.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast segmentation threshold/min/max foreground fraction sweep.")
    parser.add_argument("--student-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--tufts-root", type=Path, default=ROOT / "data/raw/tufts")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold-grid", default="0.15:0.75:0.10")
    parser.add_argument("--min-pred-fractions", default="0.0,0.01,0.02")
    parser.add_argument("--max-pred-fractions", default="0.18,0.20,0.22,0.24,0.26,0.30,0.42")
    parser.add_argument("--max-dentex", type=int, default=0)
    parser.add_argument("--max-tufts", type=int, default=240)
    parser.add_argument("--tta-horizontal-flip", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, image_size, checkpoint = load_models(args.student_checkpoints, device)
    candidates = candidate_rows(
        parse_threshold_grid(args.threshold_grid),
        parse_float_list(args.min_pred_fractions),
        parse_float_list(args.max_pred_fractions),
    )

    dentex_rows = load_clean_rows(args.clean_root, "validation_full")
    dentex_source = "dentex_validation"
    if not dentex_rows:
        dentex_rows = load_internal_test_rows_from_checkpoint(checkpoint)
        dentex_source = "dentex_internal_test"
    tufts_rows = build_tufts_pairs(args.tufts_root)

    print(
        json.dumps(
            {
                "device": str(device),
                "checkpoints": [str(path) for path in args.student_checkpoints],
                "image_size": image_size,
                "candidates": len(candidates),
                "dentex_rows": len(dentex_rows),
                "tufts_rows": len(tufts_rows),
            },
            indent=2,
        ),
        flush=True,
    )

    dentex_df = evaluate_source(
        dentex_rows,
        dentex_source,
        models,
        image_size,
        device,
        candidates,
        args.output_root,
        args.tta_horizontal_flip,
        args.max_dentex,
    )
    tufts_df = evaluate_source(
        tufts_rows,
        "tufts_external",
        models,
        image_size,
        device,
        candidates,
        args.output_root,
        args.tta_horizontal_flip,
        args.max_tufts,
    )
    merged = dentex_df.merge(
        tufts_df,
        on=["threshold", "min_pred_fraction", "max_pred_fraction"],
        suffixes=("_dentex", "_tufts"),
    )
    merged["q1_screen_score"] = merged["mean_image_dice_dentex"] + merged["mean_image_dice_tufts"]
    merged = merged.sort_values(["q1_screen_score", "mean_image_dice_tufts"], ascending=False)
    merged.to_csv(args.output_root / "combined_fraction_grid.csv", index=False)
    print("FRACTION_GRID_DONE", args.output_root, flush=True)


if __name__ == "__main__":
    main()
