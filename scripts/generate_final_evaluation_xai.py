from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFile
from sklearn.metrics import auc, roc_curve


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    normalized = str(value).replace("\\", "/")
    replacements = [
        ("E:/dental_ai_results", "F:/dental_ai_results"),
        ("E:/dental_ai_checkpoints", "F:/dental_ai_checkpoints"),
        ("E:/dental_ai_processed", "F:/dental_ai_processed"),
        ("E:/torch_cache", "F:/torch_cache"),
        ("H:/OneDrive/WORK/RECHARCH TEAM/DENTAL", str(ROOT).replace("\\", "/")),
        ("h:/python/python.exe", "C:/Python31011/python.exe"),
    ]
    for old, new in replacements:
        if normalized.lower().startswith(old.lower()):
            candidate = Path(new + normalized[len(old) :])
            if candidate.exists():
                return candidate
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    return path


def load_clean_rows(clean_root: Path, task: str) -> list[dict]:
    summary_path = clean_root / task / "summary.json"
    if not summary_path.exists():
        return []
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("status") == "kept"]


def load_split_rows_from_checkpoint(checkpoint: dict, split_name: str) -> list[dict]:
    manifest_path = checkpoint.get("split_manifest")
    if not manifest_path:
        return []
    path = resolve_path(manifest_path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "split" not in df.columns:
        return []
    df = df[df["split"] == split_name].copy()
    rows = []
    for row in df.to_dict("records"):
        image_path = resolve_path(row.get("image_path", ""))
        mask_path = resolve_path(row.get("mask_path", ""))
        if image_path.exists() and mask_path.exists():
            rows.append(
                {
                    "file_name": row.get("file_name", image_path.name),
                    "image_path": str(image_path),
                    "clean_mask": str(mask_path),
                    "clean_tooth_fraction": float((np.asarray(Image.open(mask_path).convert("L")) > 0).mean()),
                    "status": "kept",
                }
            )
    return rows


def load_internal_test_rows_from_checkpoint(checkpoint: dict) -> list[dict]:
    return load_split_rows_from_checkpoint(checkpoint, "internal_test")


def build_tufts_pairs(raw_root: Path) -> list[dict]:
    image_dir = raw_root / "Radiographs" / "Radiographs"
    mask_dir = raw_root / "Segmentation" / "Segmentation" / "teeth_mask"
    mask_by_stem = {path.stem.lower(): path for path in mask_dir.glob("*") if path.is_file()}
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_by_stem.get(image_path.stem.lower())
        if mask_path:
            rows.append({"name": image_path.name, "image_path": str(image_path), "mask_path": str(mask_path)})
    return rows


def load_tufts_rows_from_manifest(path: Path, split_name: str) -> list[dict]:
    path = resolve_path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if split_name and "split" in df.columns:
        df = df[df["split"] == split_name].copy()
    rows = []
    for row in df.to_dict("records"):
        image_path = resolve_path(row.get("image_path", ""))
        mask_path = resolve_path(row.get("mask_path", ""))
        if image_path.exists() and mask_path.exists():
            rows.append({"name": row.get("image", image_path.name), "image_path": str(image_path), "mask_path": str(mask_path)})
    return rows


def preprocess(image: Image.Image, image_size: int, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0).to(device)
    tensor.requires_grad_(requires_grad)
    return tensor


def resize_float(array: np.ndarray, size: tuple[int, int], resample: int = Image.BILINEAR) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F").resize(size, resample)
    return np.asarray(image).astype(np.float32)


def predict_probability(
    model: torch.nn.Module | list[torch.nn.Module],
    image: Image.Image,
    image_size: int,
    device: torch.device,
    tta_horizontal_flip: bool = False,
) -> np.ndarray:
    if isinstance(model, list):
        probs = [
            predict_probability(item, image, image_size, device, tta_horizontal_flip=tta_horizontal_flip)
            for item in model
        ]
        return np.mean(np.stack(probs, axis=0), axis=0)
    tensor = preprocess(image, image_size, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = segmentation_logits(model(tensor))
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            if tta_horizontal_flip:
                flipped_tensor = torch.flip(tensor, dims=[3])
                flipped_logits = segmentation_logits(model(flipped_tensor))
                flipped_probs = torch.softmax(flipped_logits.float(), dim=1)[:, 1]
                probs = 0.5 * (probs + torch.flip(flipped_probs, dims=[2]))
    prob_small = probs.squeeze(0).detach().cpu().numpy()
    return resize_float(prob_small, image.size)


def dice_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    dice = (2 * intersection + eps) / (pred.sum() + target.sum() + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def probability_to_prediction(
    prob: np.ndarray,
    threshold: float,
    min_pred_fraction: float = 0.0,
    max_pred_fraction: float = 1.0,
) -> np.ndarray:
    pred = prob >= threshold
    total = pred.size
    if total == 0:
        return pred
    min_pred_fraction = max(0.0, min(1.0, float(min_pred_fraction)))
    max_pred_fraction = max(min_pred_fraction, min(1.0, float(max_pred_fraction)))
    pred_count = int(pred.sum())
    min_count = int(round(total * min_pred_fraction))
    max_count = int(round(total * max_pred_fraction))
    flat = prob.reshape(-1)
    if min_count > 0 and pred_count < min_count:
        cutoff = np.partition(flat, -min_count)[-min_count]
        pred = prob >= cutoff
    if max_count > 0 and int(pred.sum()) > max_count:
        cutoff = np.partition(flat, -max_count)[-max_count]
        pred = prob >= cutoff
    return pred


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    return {
        "tn": int(np.logical_and(~pred_b, ~target_b).sum()),
        "fp": int(np.logical_and(pred_b, ~target_b).sum()),
        "fn": int(np.logical_and(~pred_b, target_b).sum()),
        "tp": int(np.logical_and(pred_b, target_b).sum()),
    }


def evaluate_rows(
    rows: list[dict],
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    source: str,
    output_csv: Path,
    pixel_sample_per_image: int,
    threshold: float,
    tta_horizontal_flip: bool = False,
    min_pred_fraction: float = 0.0,
    max_pred_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    records = []
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    sampled_probs = []
    sampled_targets = []

    for idx, row in enumerate(rows, start=1):
        image_path = resolve_path(row.get("image_path", row.get("image", "")))
        mask_path = resolve_path(row.get("clean_mask", row.get("mask_path", "")))
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        target = np.asarray(mask) > 0
        prob = predict_probability(model, image, image_size, device, tta_horizontal_flip=tta_horizontal_flip)
        pred = probability_to_prediction(prob, threshold, min_pred_fraction, max_pred_fraction)
        dice, iou = dice_iou(pred, target)
        item_counts = confusion_counts(pred, target)
        for key, value in item_counts.items():
            counts[key] += value

        flat_prob = prob.reshape(-1)
        flat_target = target.reshape(-1).astype(np.uint8)
        n_pixels = len(flat_prob)
        n_sample = min(pixel_sample_per_image, n_pixels)
        sample_idx = rng.choice(n_pixels, size=n_sample, replace=False)
        sampled_probs.append(flat_prob[sample_idx])
        sampled_targets.append(flat_target[sample_idx])

        target_fraction = float(target.mean())
        pred_fraction = float(pred.mean())
        records.append(
            {
                "source": source,
                "image": image_path.name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dice": dice,
                "iou": iou,
                "target_fraction": target_fraction,
                "pred_fraction": pred_fraction,
                "target_empty": target_fraction == 0.0,
                "threshold": threshold,
                **item_counts,
            }
        )
        if idx == 1 or idx % 100 == 0 or idx == len(rows):
            print(f"{source}: evaluated={idx}/{len(rows)}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df, counts, np.concatenate(sampled_probs), np.concatenate(sampled_targets)


def parse_threshold_grid(value: str) -> list[float]:
    if "," in value:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        start, stop, step = [float(item) for item in value.split(":")]
        thresholds = [round(float(item), 4) for item in np.arange(start, stop + step / 2, step)]
    return [min(0.99, max(0.01, threshold)) for threshold in thresholds]


def tune_threshold(
    rows: list[dict],
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    thresholds: list[float],
    output_csv: Path,
    max_rows: int,
    sample_per_image: int,
    tta_horizontal_flip: bool = False,
    seed: int = 2026,
) -> tuple[float, pd.DataFrame]:
    selected = rows[:max_rows] if max_rows else rows
    if not selected:
        df = pd.DataFrame(
            columns=[
                "threshold",
                "sampled_pixel_dice",
                "sampled_pixel_iou",
                "mean_image_dice",
                "mean_image_iou",
                "rows",
            ]
        )
        df.to_csv(output_csv, index=False)
        return 0.5, df
    rng = np.random.default_rng(seed)
    all_probs = []
    all_targets = []
    image_dice_rows = []
    for idx, row in enumerate(selected, start=1):
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        mask = Image.open(resolve_path(row.get("clean_mask", row.get("mask_path")))).convert("L")
        target = np.asarray(mask) > 0
        prob = predict_probability(model, image, image_size, device, tta_horizontal_flip=tta_horizontal_flip)
        flat_prob = prob.reshape(-1)
        flat_target = target.reshape(-1)
        n_sample = min(sample_per_image, len(flat_prob))
        sample_idx = rng.choice(len(flat_prob), size=n_sample, replace=False)
        all_probs.append(flat_prob[sample_idx])
        all_targets.append(flat_target[sample_idx])
        for threshold in thresholds:
            dice, iou = dice_iou(prob >= threshold, target)
            image_dice_rows.append({"threshold": threshold, "image_dice": dice, "image_iou": iou})
        if idx == 1 or idx % 50 == 0 or idx == len(selected):
            print(f"threshold_tune={idx}/{len(selected)}", flush=True)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(bool)
    rows_out = []
    image_df = pd.DataFrame(image_dice_rows)
    for threshold in thresholds:
        pred = probs >= threshold
        intersection = np.logical_and(pred, targets).sum()
        dice = float((2 * intersection + 1e-6) / (pred.sum() + targets.sum() + 1e-6))
        iou = float((intersection + 1e-6) / (np.logical_or(pred, targets).sum() + 1e-6))
        subset = image_df[image_df["threshold"] == threshold]
        rows_out.append(
            {
                "threshold": threshold,
                "sampled_pixel_dice": dice,
                "sampled_pixel_iou": iou,
                "mean_image_dice": float(subset["image_dice"].mean()),
                "mean_image_iou": float(subset["image_iou"].mean()),
                "rows": len(selected),
            }
        )
    df = pd.DataFrame(rows_out).sort_values(["mean_image_dice", "sampled_pixel_dice"], ascending=False)
    df.to_csv(output_csv, index=False)
    return float(df.iloc[0]["threshold"]), df


def confusion_matrix_array(counts: dict[str, int]) -> np.ndarray:
    return np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=np.float64)


def _history_frame(history: dict | list) -> pd.DataFrame:
    if isinstance(history, list):
        return pd.DataFrame(history)
    return pd.DataFrame(history)


def _plot_if_present(ax, df: pd.DataFrame, columns: list[str], labels: list[str]) -> bool:
    plotted = False
    for column, label in zip(columns, labels):
        if column in df.columns:
            ax.plot(df[column], marker="o", label=label)
            plotted = True
    return plotted


def plot_training_curves(history: dict | list, output_path: Path) -> None:
    history_df = pd.DataFrame(history)
    if history_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not _plot_if_present(axes[0], history_df, ["train_loss", "val_loss", "tufts_calibration_dice", "tufts_holdout_dice"], ["train", "validation", "tufts calibration", "tufts holdout"]):
        axes[0].text(0.5, 0.5, "No loss history", ha="center", va="center")
    axes[0].set_title("Student Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss / Score")
    axes[0].legend()
    if not _plot_if_present(axes[1], history_df, ["val_dice", "val_iou", "tufts_holdout_dice", "tufts_holdout_iou", "dentex_replay_dice"], ["Dice", "IoU", "TUFTS holdout Dice", "TUFTS holdout IoU", "DENTEX replay Dice"]):
        axes[1].text(0.5, 0.5, "No metric history", ha="center", va="center")
    axes[1].set_title("DENTEX Pseudo-Mask Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(
    dentex_counts: dict[str, int],
    tufts_counts: dict[str, int],
    output_path: Path,
    dentex_title: str = "DENTEX Evaluation",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, counts in [
        (axes[0], dentex_title, dentex_counts),
        (axes[1], "TUFTS External", tufts_counts),
    ]:
        matrix = confusion_matrix_array(counts)
        row_norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([0, 1], labels=["Pred bg", "Pred tooth"])
        ax.set_yticks([0, 1], labels=["True bg", "True tooth"])
        for y in range(2):
            for x in range(2):
                ax.text(x, y, f"{row_norm[y, x]:.3f}\n{int(matrix[y, x]):,}", ha="center", va="center")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    dentex_probs: np.ndarray,
    dentex_targets: np.ndarray,
    tufts_probs: np.ndarray,
    tufts_targets: np.ndarray,
    output_path: Path,
    dentex_label: str = "DENTEX evaluation",
) -> dict[str, float]:
    result = {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, probs, targets in [
        (dentex_label, dentex_probs, dentex_targets),
        ("TUFTS external", tufts_probs, tufts_targets),
    ]:
        if len(np.unique(targets)) < 2:
            continue
        fpr, tpr, _ = roc_curve(targets, probs)
        score = float(auc(fpr, tpr))
        result[label] = score
        ax.plot(fpr, tpr, label=f"{label} AUC={score:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("Pixel-Level ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result


def plot_metric_bars(summary: dict, output_path: Path) -> None:
    labels = [
        "DENTEX val Dice",
        "DENTEX val IoU",
        "TUFTS Dice",
        "TUFTS IoU",
        "TUFTS Dice\n(non-empty)",
        "TUFTS IoU\n(non-empty)",
    ]
    values = [
        summary["dentex_val_mean_dice"],
        summary["dentex_val_mean_iou"],
        summary["tufts_all_mean_dice"],
        summary["tufts_all_mean_iou"],
        summary["tufts_nonempty_mean_dice"],
        summary["tufts_nonempty_mean_iou"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f58518", "#eeca3b", "#54a24b", "#b279a2"])
    ax.set_ylim(0, 1)
    ax.set_title("Student Segmentation Performance")
    ax.set_ylabel("Score")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_radar_chart(summary: dict, teacher_summary: dict, output_path: Path) -> None:
    metrics = ["Teacher Dice", "Teacher IoU", "Student Dice", "Student IoU", "External Dice", "External IoU"]
    values = [
        teacher_summary["xray_teacher_best_val_dice"],
        teacher_summary["xray_teacher_best_val_iou"],
        summary["dentex_val_mean_dice"],
        summary["dentex_val_mean_iou"],
        summary["tufts_nonempty_mean_dice"],
        summary["tufts_nonempty_mean_iou"],
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_cycle = values + values[:1]
    angles_cycle = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_cycle, values_cycle, color="#4c78a8", linewidth=2)
    ax.fill(angles_cycle, values_cycle, color="#4c78a8", alpha=0.22)
    ax.set_xticks(angles, metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Teacher-Student External Validation Radar", y=1.08)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def augmentation_panel(row: dict, output_path: Path) -> None:
    image = Image.open(resolve_path(row["image_path"])).convert("RGB")
    mask = Image.open(resolve_path(row["clean_mask"])).convert("L")
    augments = [
        ("Original", image),
        ("Horizontal flip", image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        ("Contrast +20%", ImageEnhance.Contrast(image).enhance(1.20)),
        ("Brightness -10%", ImageEnhance.Brightness(image).enhance(0.90)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for idx, (title, aug) in enumerate(augments):
        axes[0, idx].imshow(aug)
        axes[0, idx].set_title(title)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(mask if idx != 1 else mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT), cmap="gray")
        axes[1, idx].set_title("Mask target")
        axes[1, idx].axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def xai_maps(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = preprocess(image, image_size, device, requires_grad=True)
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = segmentation_logits(model(tensor))
        probs = torch.softmax(logits.float(), dim=1)
        tooth_prob = probs[:, 1]
        flat = tooth_prob.reshape(1, -1)
        top_k = max(1, int(flat.shape[1] * 0.05))
        score = flat.topk(top_k, dim=1).values.mean()
    score.backward()
    prob_small = tooth_prob.squeeze(0).detach().cpu().numpy()
    prob = resize_float(prob_small, image.size)
    entropy_small = (-probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1).squeeze(0).detach().cpu().numpy() / math.log(2)
    entropy = resize_float(entropy_small, image.size)
    feature = getattr(model, "last_decoder_feature", None)
    if not isinstance(feature, torch.Tensor):
        feature = getattr(model, "last_feature_map", None)
    if isinstance(feature, torch.Tensor) and feature.grad is not None:
        gradients = feature.grad.detach().float()
        activations = feature.detach().float()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)
        cam_arr = cam.cpu().numpy()
        saliency_small = (cam_arr - cam_arr.min()) / (cam_arr.max() - cam_arr.min() + 1e-6)
    else:
        grad = tensor.grad.detach().abs().max(dim=1)[0].squeeze(0).cpu().numpy()
        saliency_small = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    saliency = resize_float(saliency_small, image.size)
    return prob, saliency, entropy


def error_rgb(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tp = np.logical_and(pred, target)
    fp = np.logical_and(pred, ~target)
    fn = np.logical_and(~pred, target)
    rgb[tp] = (80, 200, 120)
    rgb[fp] = (240, 80, 80)
    rgb[fn] = (80, 140, 255)
    return rgb


def plot_xai_panels(
    rows: list[dict],
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    output_path: Path,
    title_prefix: str,
    threshold: float,
    min_pred_fraction: float = 0.0,
    max_pred_fraction: float = 1.0,
) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(20, 2.25 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, row in enumerate(rows):
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        mask = Image.open(resolve_path(row.get("clean_mask", row.get("mask_path")))).convert("L")
        target = np.asarray(mask) > 0
        prob, saliency, entropy = xai_maps(model, image, image_size, device)
        pred = probability_to_prediction(prob, threshold, min_pred_fraction, max_pred_fraction)
        overlay = image.copy()
        color = Image.new("RGB", image.size, (80, 220, 255))
        alpha = Image.fromarray((pred.astype(np.uint8) * 115))
        overlay.paste(color, (0, 0), alpha)
        panels = [
            (image, "Image", None),
            (mask, "Target", "gray"),
            (overlay, "Prediction overlay", None),
            (error_rgb(pred, target), "Error map", None),
            (saliency, "Grad-CAM / saliency", "magma"),
            (entropy, "Softmax uncertainty", "inferno"),
        ]
        for col, (panel, title, cmap) in enumerate(panels):
            axes[row_idx, col].imshow(panel, cmap=cmap)
            axes[row_idx, col].set_title(title if row_idx == 0 else "")
            axes[row_idx, col].axis("off")
        name = row.get("image", Path(row["image_path"]).name)
        dice, iou = dice_iou(pred, target)
        axes[row_idx, 0].set_ylabel(f"{name}\nDice={dice:.3f} IoU={iou:.3f}", fontsize=9)
    fig.subplots_adjust(wspace=0.06, hspace=0.18)
    fig.suptitle(title_prefix, y=1.02)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def pointing_game_metrics(
    rows: list[dict],
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    output_csv: Path,
    source: str,
    max_rows: int,
) -> pd.DataFrame:
    selected = rows[:max_rows] if max_rows else rows
    records = []
    for idx, row in enumerate(selected, start=1):
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        mask = Image.open(resolve_path(row.get("clean_mask", row.get("mask_path")))).convert("L")
        target = np.asarray(mask) > 0
        _, cam, _ = xai_maps(model, image, image_size, device)
        cam = np.asarray(cam, dtype=np.float32)
        y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        top_n = max(1, int(cam.size * 0.01))
        threshold = np.partition(cam.reshape(-1), -top_n)[-top_n]
        top_mask = cam >= threshold
        records.append(
            {
                "source": source,
                "image": row.get("image", Path(row["image_path"]).name),
                "pointing_hit": int(bool(target[y, x])),
                "top1_percent_target_overlap": float(np.logical_and(top_mask, target).sum() / max(1, top_mask.sum())),
                "target_fraction": float(target.mean()),
                "cam_peak_x": int(x),
                "cam_peak_y": int(y),
            }
        )
        if idx == 1 or idx % 20 == 0 or idx == len(selected):
            print(f"{source} pointing_game={idx}/{len(selected)}", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df


def plot_failure_panels(rows: list[dict], output_path: Path, title: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(15, 2.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, row in enumerate(rows):
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        target = np.asarray(Image.open(resolve_path(row["mask_path"])).convert("L")) > 0
        pred_path = Path(row["pred_mask_path"])
        pred = np.asarray(Image.open(pred_path).convert("L")) > 0
        overlay = image.copy()
        color = Image.new("RGB", image.size, (80, 220, 255))
        alpha = Image.fromarray((pred.astype(np.uint8) * 115))
        overlay.paste(color, (0, 0), alpha)
        panels = [(image, row["image"]), (target, "Target"), (overlay, "Prediction"), (error_rgb(pred, target), "Error map")]
        for col, (panel, panel_title) in enumerate(panels):
            axes[row_idx, col].imshow(panel, cmap="gray" if col == 1 else None)
            axes[row_idx, col].set_title(panel_title if row_idx == 0 else "")
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel(f"Dice={row['dice']:.3f}\nIoU={row['iou']:.3f}", fontsize=9)
    fig.subplots_adjust(wspace=0.08, hspace=0.18)
    fig.suptitle(title, y=1.02)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_prediction_masks_for_failures(
    df: pd.DataFrame,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    output_dir: Path,
    n: int = 4,
    threshold: float = 0.5,
    min_pred_fraction: float = 0.0,
    max_pred_fraction: float = 1.0,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = df[df["target_empty"] == False].sort_values("dice").head(n).to_dict("records")
    rows = []
    for row in selected:
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        prob = predict_probability(model, image, image_size, device)
        pred = probability_to_prediction(prob, threshold, min_pred_fraction, max_pred_fraction)
        pred_path = output_dir / f"{Path(row['image']).stem}_pred_mask.png"
        Image.fromarray((pred.astype(np.uint8) * 255)).save(pred_path)
        row["pred_mask_path"] = str(pred_path)
        rows.append(row)
    return rows


def best_dice_iou_from_checkpoint(path: Path) -> dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", {})
    if "val_dice" not in history:
        return {}
    idx = int(np.argmax(history["val_dice"]))
    return {
        "xray_teacher_best_val_dice": float(history["val_dice"][idx]),
        "xray_teacher_best_val_iou": float(history.get("val_iou", [0.0])[idx]),
        "xray_teacher_best_epoch": idx + 1,
    }


def history_value(history: dict | list, key: str, epoch: int) -> float:
    idx = max(0, epoch - 1)
    try:
        if isinstance(history, list):
            if idx >= len(history):
                return math.nan
            row = history[idx]
            if isinstance(row, dict):
                return float(row.get(key, math.nan))
            return math.nan
        values = history.get(key, [])
        if idx >= len(values):
            return math.nan
        return float(values[idx])
    except Exception:
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final Dental AI evaluation, XAI, and uncertainty artifacts.")
    parser.add_argument("--student-checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/student_tinyunet_dentex_pseudo_full_best.pt"))
    parser.add_argument("--student-checkpoints", nargs="+", type=Path, default=None, help="Optional checkpoint ensemble. Probabilities are averaged; XAI uses the first checkpoint.")
    parser.add_argument("--xray-teacher-checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/teacher_tinyunet_tufts_xray_full.pt"))
    parser.add_argument("--clean-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_xray_teacher_clean"))
    parser.add_argument("--tufts-root", type=Path, default=ROOT / "data/raw/tufts")
    parser.add_argument("--tufts-split-manifest", type=Path, default=None, help="Optional CSV with image_path, mask_path, and split columns.")
    parser.add_argument("--tufts-split-name", default="holdout", help="Split name used when --tufts-split-manifest is supplied.")
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/final_eval_xai_v1"))
    parser.add_argument("--pixel-sample-per-image", type=int, default=5000)
    parser.add_argument("--max-tufts", type=int, default=0, help="Use 0 for all TUFTS images.")
    parser.add_argument("--pointing-game-max", type=int, default=80, help="Rows per source for Grad-CAM pointing-game scoring. Use 0 for all.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed probability threshold when validation tuning is disabled.")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune the segmentation threshold on held-out validation rows.")
    parser.add_argument("--threshold-grid", default="0.15:0.85:0.05", help="Comma list or start:stop:step grid for threshold tuning.")
    parser.add_argument("--threshold-tune-max-rows", type=int, default=160, help="Validation rows used for threshold tuning. Use 0 for all.")
    parser.add_argument("--threshold-sample-per-image", type=int, default=12000, help="Pixels sampled per image for sampled threshold metrics.")
    parser.add_argument("--tta-horizontal-flip", action="store_true", help="Average normal and horizontal-flipped inference probabilities.")
    parser.add_argument("--min-pred-fraction", type=float, default=0.0, help="Unsupervised lower bound on predicted foreground fraction per image.")
    parser.add_argument("--max-pred-fraction", type=float, default=1.0, help="Unsupervised upper bound on predicted foreground fraction per image.")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    tables_dir = args.output_root / "tables"
    prediction_dir = args.output_root / "prediction_masks"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_checkpoint_paths = args.student_checkpoints if args.student_checkpoints else [args.student_checkpoint]
    checkpoints = [torch.load(path, map_location=device, weights_only=False) for path in student_checkpoint_paths]
    checkpoint = checkpoints[0]
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 512))
    models = []
    for path, item in zip(student_checkpoint_paths, checkpoints):
        item_config = item.get("config", {})
        item_image_size = int(item_config.get("image_size", 512))
        if item_image_size != image_size:
            raise RuntimeError(f"Checkpoint image_size mismatch: {path} has {item_image_size}, expected {image_size}")
        item_model = build_model_from_checkpoint(item, num_classes=2).to(device)
        item_model.load_state_dict(item["model"])
        item_model.eval()
        models.append(item_model)
    prediction_model: torch.nn.Module | list[torch.nn.Module] = models[0] if len(models) == 1 else models
    xai_model = models[0]

    print(
        f"device={device} student_checkpoints={[str(path) for path in student_checkpoint_paths]} image_size={image_size} "
        f"model_family={checkpoint.get('model_family', config.get('model_family', 'unknown'))}",
        flush=True,
    )

    validation_rows_for_threshold = load_clean_rows(args.clean_root, "validation_full")
    if not validation_rows_for_threshold:
        validation_rows_for_threshold = load_split_rows_from_checkpoint(checkpoint, "validation")

    dentex_rows_raw = load_clean_rows(args.clean_root, "validation_full")
    dentex_source_name = "dentex_validation"
    if not dentex_rows_raw:
        dentex_rows_raw = load_internal_test_rows_from_checkpoint(checkpoint)
        dentex_source_name = "dentex_internal_test"
        print(
            f"validation_full has no kept masks; using student split_manifest internal_test rows={len(dentex_rows_raw)}",
            flush=True,
        )
    if not dentex_rows_raw:
        raise RuntimeError(
            f"No DENTEX evaluation rows found. Expected validation_full under {args.clean_root} "
            "or internal_test rows in the student checkpoint split manifest."
        )
    dentex_rows = [
        {
            "image": Path(row["image_path"]).name,
            "image_path": row["image_path"],
            "clean_mask": row["clean_mask"],
        }
        for row in dentex_rows_raw
    ]
    if args.tufts_split_manifest:
        tufts_rows = load_tufts_rows_from_manifest(args.tufts_split_manifest, args.tufts_split_name)
        if not tufts_rows:
            raise RuntimeError(f"No TUFTS rows found in {args.tufts_split_manifest} split={args.tufts_split_name!r}")
    else:
        tufts_rows = build_tufts_pairs(args.tufts_root)
    if args.max_tufts:
        tufts_rows = tufts_rows[: args.max_tufts]

    threshold = min(0.99, max(0.01, float(args.threshold)))
    threshold_source = "fixed"
    threshold_tuning_df = pd.DataFrame()
    if args.tune_threshold:
        threshold, threshold_tuning_df = tune_threshold(
            validation_rows_for_threshold,
            prediction_model,
            image_size,
            device,
            parse_threshold_grid(args.threshold_grid),
            tables_dir / "threshold_tuning_validation.csv",
            args.threshold_tune_max_rows,
            args.threshold_sample_per_image,
            args.tta_horizontal_flip,
        )
        threshold_source = "validation_split" if validation_rows_for_threshold else "fixed_no_validation_rows"
        print(f"selected_threshold={threshold:.3f} source={threshold_source}", flush=True)

    dentex_eval_csv = tables_dir / f"{dentex_source_name}_student_eval.csv"
    dentex_df, dentex_counts, dentex_probs, dentex_targets = evaluate_rows(
        dentex_rows,
        prediction_model,
        image_size,
        device,
        dentex_source_name,
        dentex_eval_csv,
        args.pixel_sample_per_image,
        threshold,
        args.tta_horizontal_flip,
        args.min_pred_fraction,
        args.max_pred_fraction,
        seed=42,
    )
    if dentex_eval_csv.name != "dentex_validation_student_eval.csv":
        dentex_df.to_csv(tables_dir / "dentex_validation_student_eval.csv", index=False)
    tufts_df, tufts_counts_all, tufts_probs, tufts_targets = evaluate_rows(
        tufts_rows,
        prediction_model,
        image_size,
        device,
        "tufts_external",
        tables_dir / "tufts_external_student_eval.csv",
        args.pixel_sample_per_image,
        threshold,
        args.tta_horizontal_flip,
        args.min_pred_fraction,
        args.max_pred_fraction,
        seed=123,
    )

    tufts_nonempty_df = tufts_df[tufts_df["target_empty"] == False].copy()
    tufts_nonempty_counts = {
        key: int(tufts_nonempty_df[key].sum()) for key in ["tn", "fp", "fn", "tp"]
    }
    tufts_df.to_csv(tables_dir / "tufts_external_student_eval_with_target_frac.csv", index=False)
    tufts_nonempty_df.to_csv(tables_dir / "tufts_external_student_eval_nonempty.csv", index=False)

    teacher_summary = best_dice_iou_from_checkpoint(args.xray_teacher_checkpoint)
    student_epoch = int(checkpoint.get("epoch", 0))
    student_history = checkpoint.get("history", {})
    summary = {
        "student_checkpoint": str(student_checkpoint_paths[0]),
        "student_checkpoints": "|".join(str(path) for path in student_checkpoint_paths),
        "student_checkpoint_ensemble_size": len(student_checkpoint_paths),
        "student_epoch": student_epoch,
        "student_best_val_dice_from_training": float(checkpoint.get("best_val_dice", 0.0)),
        "student_best_score_from_training": float(checkpoint.get("best_score", checkpoint.get("best_val_dice", 0.0))),
        "student_val_classification_macro_f1": history_value(student_history, "val_classification_macro_f1", student_epoch),
        "student_test_classification_macro_f1": history_value(student_history, "test_classification_macro_f1", student_epoch),
        "student_val_classification_exact_match": history_value(student_history, "val_classification_exact_match", student_epoch),
        "student_test_classification_exact_match": history_value(student_history, "test_classification_exact_match", student_epoch),
        "student_val_classification_valid_rows": history_value(student_history, "val_classification_valid_rows", student_epoch),
        "student_test_classification_valid_rows": history_value(student_history, "test_classification_valid_rows", student_epoch),
        "dentex_eval_source": dentex_source_name,
        "segmentation_threshold": float(threshold),
        "threshold_source": threshold_source,
        "tta_horizontal_flip": bool(args.tta_horizontal_flip),
        "min_pred_fraction": float(args.min_pred_fraction),
        "max_pred_fraction": float(args.max_pred_fraction),
        "threshold_tuning_rows": int(len(validation_rows_for_threshold)),
        "threshold_tuning_best_mean_image_dice": (
            float(threshold_tuning_df.iloc[0]["mean_image_dice"]) if not threshold_tuning_df.empty else math.nan
        ),
        "threshold_tuning_best_sampled_pixel_dice": (
            float(threshold_tuning_df.iloc[0]["sampled_pixel_dice"]) if not threshold_tuning_df.empty else math.nan
        ),
        "dentex_eval_rows": int(len(dentex_df)),
        "dentex_eval_mean_dice": float(dentex_df["dice"].mean()),
        "dentex_eval_mean_iou": float(dentex_df["iou"].mean()),
        "dentex_val_rows": int(len(dentex_df)),
        "dentex_val_mean_dice": float(dentex_df["dice"].mean()),
        "dentex_val_mean_iou": float(dentex_df["iou"].mean()),
        "tufts_all_rows": int(len(tufts_df)),
        "tufts_empty_target_rows": int(tufts_df["target_empty"].sum()),
        "tufts_all_mean_dice": float(tufts_df["dice"].mean()),
        "tufts_all_mean_iou": float(tufts_df["iou"].mean()),
        "tufts_nonempty_rows": int(len(tufts_nonempty_df)),
        "tufts_nonempty_mean_dice": float(tufts_nonempty_df["dice"].mean()),
        "tufts_nonempty_mean_iou": float(tufts_nonempty_df["iou"].mean()),
        **teacher_summary,
    }

    plot_training_curves(checkpoint.get("history", {}), figures_dir / "student_training_curves_final.png")
    dentex_plot_label = dentex_source_name.replace("_", " ").title()
    plot_confusion_matrices(
        dentex_counts,
        tufts_nonempty_counts,
        figures_dir / "segmentation_confusion_matrices.png",
        dentex_plot_label,
    )
    roc_summary = plot_roc_curves(
        dentex_probs,
        dentex_targets,
        tufts_probs,
        tufts_targets,
        figures_dir / "pixel_level_roc_curves.png",
        dentex_plot_label,
    )
    summary["dentex_pixel_roc_auc"] = roc_summary.get(dentex_plot_label)
    summary["tufts_pixel_roc_auc"] = roc_summary.get("TUFTS external")
    plot_metric_bars(summary, figures_dir / "student_metric_bars.png")
    plot_radar_chart(summary, teacher_summary, figures_dir / "teacher_student_radar.png")
    augmentation_panel(dentex_rows_raw[len(dentex_rows_raw) // 2], figures_dir / "augmentation_examples.png")

    dentex_xai_rows = []
    kept_dentex = sorted(dentex_rows_raw, key=lambda row: float(row["clean_tooth_fraction"]))
    for row in [kept_dentex[0], kept_dentex[len(kept_dentex) // 2], kept_dentex[-1]]:
        dentex_xai_rows.append({"image_path": row["image_path"], "clean_mask": row["clean_mask"], "image": Path(row["image_path"]).name})
    plot_xai_panels(
        dentex_xai_rows,
        xai_model,
        image_size,
        device,
        figures_dir / "dentex_xai_uncertainty_panels.png",
        "DENTEX Student XAI and Uncertainty",
        threshold,
        args.min_pred_fraction,
        args.max_pred_fraction,
    )

    tufts_xai_rows = []
    for row in tufts_nonempty_df.sort_values("dice").head(2).to_dict("records"):
        tufts_xai_rows.append({"image_path": row["image_path"], "mask_path": row["mask_path"], "image": row["image"]})
    for row in tufts_nonempty_df.sort_values("dice", ascending=False).head(2).to_dict("records"):
        tufts_xai_rows.append({"image_path": row["image_path"], "mask_path": row["mask_path"], "image": row["image"]})
    plot_xai_panels(
        tufts_xai_rows,
        xai_model,
        image_size,
        device,
        figures_dir / "tufts_xai_uncertainty_panels.png",
        "TUFTS External XAI and Uncertainty",
        threshold,
        args.min_pred_fraction,
        args.max_pred_fraction,
    )

    dentex_pointing_df = pointing_game_metrics(
        dentex_rows,
        xai_model,
        image_size,
        device,
        tables_dir / "dentex_gradcam_pointing_game.csv",
        dentex_source_name,
        args.pointing_game_max,
    )
    tufts_pointing_df = pointing_game_metrics(
        tufts_nonempty_df.to_dict("records"),
        xai_model,
        image_size,
        device,
        tables_dir / "tufts_gradcam_pointing_game.csv",
        "tufts_external_nonempty",
        args.pointing_game_max,
    )
    summary["dentex_gradcam_pointing_hit_rate"] = float(dentex_pointing_df["pointing_hit"].mean())
    summary["dentex_gradcam_top1_percent_overlap"] = float(dentex_pointing_df["top1_percent_target_overlap"].mean())
    summary["tufts_gradcam_pointing_hit_rate"] = float(tufts_pointing_df["pointing_hit"].mean())
    summary["tufts_gradcam_top1_percent_overlap"] = float(tufts_pointing_df["top1_percent_target_overlap"].mean())

    failure_rows = save_prediction_masks_for_failures(
        tufts_df,
        prediction_model,
        image_size,
        device,
        prediction_dir,
        n=4,
        threshold=threshold,
        min_pred_fraction=args.min_pred_fraction,
        max_pred_fraction=args.max_pred_fraction,
    )
    plot_failure_panels(failure_rows, figures_dir / "tufts_failure_case_panels.png", "TUFTS Lowest-Dice Non-Empty Failure Cases")

    summary_path = tables_dir / "final_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(tables_dir / "final_metrics_summary.csv", index=False)

    figure_manifest = sorted(str(path) for path in figures_dir.glob("*.png"))
    (args.output_root / "artifact_manifest.json").write_text(
        json.dumps({"figures": figure_manifest, "tables": sorted(str(path) for path in tables_dir.glob("*"))}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"FINAL_EVALUATION_XAI_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
