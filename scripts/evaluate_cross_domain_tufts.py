"""V15 Stage 4: Cross-domain validation on TUFTS dental radiographs.

Evaluates both teacher and student models on TUFTS (out-of-domain):
  - Segmentation: Dice, IoU, pixel ROC AUC
  - Visual overlays: prediction vs GT side-by-side
  - Failure analysis: worst-N images by Dice
  - Teacher vs Student comparison table
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.augmentations import prepare_segmentation_pair
from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TuftsEvalDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, Path]], image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path = self.rows[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = prepare_segmentation_pair(image, mask, self.image_size, augment=False, use_clahe=True)
        arr = np.asarray(image).astype(np.float32) / 255.0
        mask_arr = (np.asarray(mask) > 0).astype(np.int64)
        return torch.from_numpy(np.transpose(arr, (2, 0, 1))), torch.from_numpy(mask_arr), img_path.name


def build_tufts_rows(tufts_root: Path, max_rows: int = 0) -> list[tuple[Path, Path]]:
    img_dir = tufts_root / "Radiographs" / "Radiographs"
    mask_dir = tufts_root / "Segmentation" / "Segmentation" / "teeth_mask"
    if not img_dir.exists() or not mask_dir.exists():
        return []
    mask_by_stem = {p.stem.lower(): p for p in mask_dir.glob("*") if p.is_file()}
    rows = []
    for img in sorted(img_dir.glob("*")):
        mp = mask_by_stem.get(img.stem.lower())
        if mp:
            rows.append((img, mp))
    if max_rows:
        rows = rows[:max_rows]
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> list[dict]:
    model.eval()
    per_image: list[dict] = []
    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device)
            masks = masks.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = segmentation_logits(model(images))
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i in range(images.size(0)):
                pred = preds[i]
                target = masks[i]
                prob = probs[i, 1] if num_classes == 2 else probs[i].max(dim=0).values

                pred_b = pred.bool()
                target_b = target.bool()
                is_empty = target_b.sum().item() == 0

                eps = 1e-6
                inter = (pred_b & target_b).sum().float()
                union = (pred_b | target_b).sum().float()
                denom = pred_b.sum().float() + target_b.sum().float()
                dice = float(((2 * inter + eps) / (denom + eps)).cpu())
                iou = float(((inter + eps) / (union + eps)).cpu())
                acc = float((pred == target).float().mean().cpu())

                # Pixel-level ROC AUC (manual computation)
                prob_np = prob.cpu().numpy().ravel()
                target_np = target.cpu().numpy().ravel().astype(np.float32)
                roc_auc = compute_roc_auc(prob_np, target_np)

                per_image.append({
                    "file": names[i],
                    "dice": dice,
                    "iou": iou,
                    "accuracy": acc,
                    "pixel_roc_auc": roc_auc,
                    "is_empty": is_empty,
                    "pred_foreground_ratio": float(pred_b.float().mean().cpu()),
                    "gt_foreground_ratio": float(target_b.float().mean().cpu()),
                })
    return per_image


def compute_roc_auc(prob: np.ndarray, target: np.ndarray) -> float:
    """Simple ROC AUC computation without sklearn dependency."""
    if target.sum() == 0 or target.sum() == len(target):
        return float("nan")
    pos = prob[target > 0.5]
    neg = prob[target <= 0.5]
    # Approximate with sample if too large
    if len(pos) > 10000:
        rng = np.random.RandomState(42)
        pos = rng.choice(pos, 10000, replace=False)
    if len(neg) > 10000:
        rng = np.random.RandomState(42)
        neg = rng.choice(neg, 10000, replace=False)
    # Count concordant pairs
    count = 0
    total = 0
    for p in pos:
        count += (neg < p).sum() + 0.5 * (neg == p).sum()
        total += len(neg)
    return float(count / max(1, total))


def summarize_results(per_image: list[dict]) -> dict:
    nonempty = [r for r in per_image if not r["is_empty"]]
    all_aucs = [r["pixel_roc_auc"] for r in per_image if not math.isnan(r["pixel_roc_auc"])]
    return {
        "total_images": len(per_image),
        "nonempty_images": len(nonempty),
        "mean_dice": float(np.mean([r["dice"] for r in per_image])) if per_image else 0.0,
        "nonempty_mean_dice": float(np.mean([r["dice"] for r in nonempty])) if nonempty else 0.0,
        "mean_iou": float(np.mean([r["iou"] for r in per_image])) if per_image else 0.0,
        "nonempty_mean_iou": float(np.mean([r["iou"] for r in nonempty])) if nonempty else 0.0,
        "pixel_roc_auc": float(np.mean(all_aucs)) if all_aucs else 0.0,
        "mean_accuracy": float(np.mean([r["accuracy"] for r in per_image])) if per_image else 0.0,
    }


def save_overlay(image_path: Path, mask_path: Path, pred_mask: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    gt = np.asarray(Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST))
    pred = Image.fromarray(pred_mask).resize(image.size, Image.NEAREST)
    pred_np = np.asarray(pred)

    # Create side-by-side: original | GT overlay | Pred overlay
    w, h = image.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(image, (0, 0))

    # GT overlay (green)
    gt_color = np.zeros((h, w, 3), dtype=np.uint8)
    gt_color[gt > 0] = (0, 255, 0)
    gt_overlay = Image.fromarray(gt_color)
    gt_alpha = Image.fromarray(((gt > 0).astype(np.uint8) * 100))
    gt_img = image.copy()
    gt_img.paste(gt_overlay, (0, 0), gt_alpha)
    canvas.paste(gt_img, (w, 0))

    # Pred overlay (red)
    pred_color = np.zeros((h, w, 3), dtype=np.uint8)
    pred_color[pred_np > 0] = (255, 80, 80)
    pred_overlay = Image.fromarray(pred_color)
    pred_alpha = Image.fromarray(((pred_np > 0).astype(np.uint8) * 100))
    pred_img = image.copy()
    pred_img.paste(pred_overlay, (0, 0), pred_alpha)
    canvas.paste(pred_img, (w * 2, 0))

    canvas.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 4: TUFTS cross-domain evaluation.")
    parser.add_argument("--student-checkpoint", type=Path, default=None, help="Student best .pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None, help="Teacher best .pt")
    parser.add_argument("--tufts-root", type=Path, default=ROOT / "data" / "raw" / "tufts")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rows = build_tufts_rows(args.tufts_root, args.max_images)
    if not rows:
        print(f"ERROR: No TUFTS images found at {args.tufts_root}")
        sys.exit(1)
    print(f"=== V15 TUFTS Cross-Domain Evaluation ===")
    print(f"TUFTS images: {len(rows)}")

    results = {}

    for model_name, ckpt_path in [("teacher", args.teacher_checkpoint), ("student", args.student_checkpoint)]:
        if ckpt_path is None or not ckpt_path.exists():
            print(f"Skipping {model_name}: checkpoint not provided or missing")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        class_names = ckpt.get("class_names", ["background", "tooth"])
        config = ckpt.get("config", {})
        image_size = args.image_size or int(config.get("image_size", 224))

        # Build model — for TUFTS tooth segmentation use binary (2 classes)
        num_classes = min(len(class_names), 2)  # binary for TUFTS
        model = build_model_from_checkpoint(ckpt, num_classes=len(class_names)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        print(f"\nEvaluating {model_name} (image_size={image_size}, classes={class_names})...")
        ds = TuftsEvalDataset(rows, image_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        start = time.perf_counter()
        per_image = evaluate_model(model, loader, device, num_classes=len(class_names))
        elapsed = time.perf_counter() - start

        summary = summarize_results(per_image)
        summary["model"] = model_name
        summary["checkpoint"] = str(ckpt_path)
        summary["image_size"] = image_size
        summary["elapsed_seconds"] = elapsed
        results[model_name] = summary

        # Save per-image results
        with (tables_dir / f"tufts_{model_name}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            writer.writeheader()
            writer.writerows(per_image)

        print(f"  {model_name}: dice={summary['nonempty_mean_dice']:.4f} iou={summary['nonempty_mean_iou']:.4f} "
              f"roc_auc={summary['pixel_roc_auc']:.4f} images={summary['total_images']} elapsed={elapsed:.1f}s")

        # Save worst-N failure cases
        sorted_by_dice = sorted(per_image, key=lambda r: r["dice"])
        worst = sorted_by_dice[:10]
        with (tables_dir / f"tufts_{model_name}_worst10.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(worst[0].keys()))
            writer.writeheader()
            writer.writerows(worst)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save comparison summary
    (output_root / "tufts_evaluation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Comparison table
    if len(results) >= 2:
        comp_rows = []
        for metric in ["nonempty_mean_dice", "nonempty_mean_iou", "pixel_roc_auc", "mean_accuracy"]:
            row = {"metric": metric}
            for model_name in results:
                row[model_name] = results[model_name].get(metric, math.nan)
            if len(results) == 2:
                vals = list(row.values())[1:]
                row["delta"] = vals[1] - vals[0] if len(vals) == 2 else math.nan
            comp_rows.append(row)
        with (tables_dir / "teacher_vs_student_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comp_rows)
        print("\n=== Teacher vs Student Comparison ===")
        for r in comp_rows:
            print(f"  {r['metric']}: " + " | ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "metric"))

    # Build summary for Q1 gate format
    final_model = "student" if "student" in results else ("teacher" if "teacher" in results else None)
    if final_model:
        r = results[final_model]
        q1_summary = {
            "tufts_nonempty_mean_dice": r.get("nonempty_mean_dice", 0.0),
            "tufts_nonempty_mean_iou": r.get("nonempty_mean_iou", 0.0),
            "tufts_pixel_roc_auc": r.get("pixel_roc_auc", 0.0),
        }
        (output_root / "tufts_q1_metrics.json").write_text(json.dumps(q1_summary, indent=2), encoding="utf-8")

    print(f"\nTUFTS_EVALUATION_DONE output={output_root}")


if __name__ == "__main__":
    main()
