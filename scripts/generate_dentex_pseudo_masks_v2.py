"""V15 Stage 2: Generate DENTEX pseudo-masks with multi-scale TTA from the joint teacher.

Improvements over v1:
  - Multi-scale TTA: inference at [0.75x, 1.0x, 1.25x] and average probabilities
  - Confidence filtering: only keep pixels with max-prob > threshold
  - Morphological post-processing: open/close for cleaner masks
  - Quality audit: per-image stats + summary JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits

try:
    from skimage import morphology as morph
    HAS_MORPH = True
except ImportError:
    HAS_MORPH = False


def unique_images(coco_path: Path, dentex_root: Path, max_images: int | None) -> list[dict]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    rows: list[dict] = []
    for img in data.get("images", []):
        fname = img.get("file_name")
        if not fname or fname in seen:
            continue
        seen.add(fname)
        path = dentex_root / fname
        if not path.exists():
            for sub in ["training_data/quadrant-enumeration-disease", "training_data/quadrant", "training_data/quadrant_enumeration", "training_data/unlabelled"]:
                candidate = dentex_root / sub / Path(fname).name
                if candidate.exists():
                    path = candidate
                    break
        if path.exists():
            rows.append({"id": img.get("id"), "file_name": fname, "path": path})
        if max_images and len(rows) >= max_images:
            break
    return rows


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)


def multi_scale_inference(
    model: torch.nn.Module,
    image: Image.Image,
    base_size: int,
    scales: list[float],
    device: torch.device,
) -> np.ndarray:
    """Run inference at multiple scales and average the probability maps."""
    probs_sum = None
    for scale in scales:
        size = max(32, int(base_size * scale))
        tensor = preprocess(image, size).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = segmentation_logits(model(tensor))
        # Resize to base_size before averaging
        logits = F.interpolate(logits, size=(base_size, base_size), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs
    return probs_sum / len(scales)


def clean_mask(mask_arr: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Morphological open+close to clean up noisy predictions."""
    if not HAS_MORPH:
        return mask_arr
    cleaned = mask_arr.copy()
    for cls_val in np.unique(mask_arr):
        if cls_val == 0:
            continue
        binary = (mask_arr == cls_val).astype(np.uint8)
        # Open to remove small noise, close to fill gaps
        selem = morph.disk(3)
        binary = morph.binary_opening(binary, selem).astype(np.uint8)
        binary = morph.binary_closing(binary, selem).astype(np.uint8)
        # Remove small components
        labeled = morph.label(binary)
        for region_val in np.unique(labeled):
            if region_val == 0:
                continue
            if (labeled == region_val).sum() < min_area:
                binary[labeled == region_val] = 0
        # Update cleaned mask
        cleaned[mask_arr == cls_val] = 0
        cleaned[binary > 0] = cls_val
    return cleaned


def color_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask_arr = np.asarray(mask)
    color = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
    palette = {1: (255, 80, 80), 2: (255, 210, 70), 3: (80, 220, 255), 4: (80, 255, 120)}
    for val, rgb in palette.items():
        color[mask_arr == val] = rgb
    overlay = Image.fromarray(color).resize(image.size, Image.NEAREST)
    alpha = Image.fromarray(((mask_arr > 0).astype(np.uint8) * 110)).resize(image.size, Image.NEAREST)
    out = image.copy()
    out.paste(overlay, (0, 0), alpha)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 2: Generate DENTEX pseudo-masks from the joint teacher.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to teacher_joint_best.pt")
    parser.add_argument("--coco", type=Path, default=ROOT / "data/processed/coco/dentex/disease.json")
    parser.add_argument("--dentex-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--scales", type=str, default="0.75,1.0,1.25", help="Comma-separated TTA scales")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Min probability for mask pixels")
    parser.add_argument("--min-area", type=int, default=50, help="Min connected component area in pixels")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--also-quadrant", action="store_true", help="Also process quadrant/enumeration COCOs")
    args = parser.parse_args()

    scales = [float(s.strip()) for s in args.scales.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names") or ckpt.get("seg_class_names") or ["background", "Caries", "Cavity", "Crack", "Tooth"]
    config = ckpt.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 224))
    model = build_model_from_checkpoint(ckpt, num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mask_dir = args.output / "masks"
    preview_dir = args.output / "previews"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Collect all COCO JSONs to process
    coco_files = [args.coco]
    if args.also_quadrant:
        for name in ["quadrant.json", "enumeration.json"]:
            p = args.coco.parent / name
            if p.exists():
                coco_files.append(p)

    # Deduplicate images across COCOs
    all_rows: list[dict] = []
    seen_files: set[str] = set()
    for coco_path in coco_files:
        max_img = None if args.max_images == 0 else args.max_images
        for row in unique_images(coco_path, args.dentex_root, max_img):
            if row["file_name"] not in seen_files:
                seen_files.add(row["file_name"])
                all_rows.append(row)
    if args.max_images:
        all_rows = all_rows[:args.max_images]

    print(f"=== V15 Pseudo-Mask Generation ===")
    print(f"images={len(all_rows)} checkpoint={args.checkpoint} image_size={image_size}")
    print(f"scales={scales} confidence_threshold={args.confidence_threshold} min_area={args.min_area}")
    print(f"classes={class_names} morph_available={HAS_MORPH}")

    summary: list[dict] = []
    start = time.perf_counter()

    with torch.no_grad():
        for idx, row in enumerate(all_rows, start=1):
            image = Image.open(row["path"]).convert("RGB")

            # Multi-scale TTA inference
            probs = multi_scale_inference(model, image, image_size, scales, device)  # (C, H, W)
            confidence = probs.max(axis=0)  # (H, W)
            pred = probs.argmax(axis=0).astype(np.uint8)  # (H, W)

            # Confidence filtering
            pred[confidence < args.confidence_threshold] = 0

            # Morphological cleanup
            pred = clean_mask(pred, min_area=args.min_area)

            # Resize to original image size
            mask = Image.fromarray(pred).resize(image.size, Image.NEAREST)
            stem = Path(row["file_name"]).stem
            mask_path = mask_dir / f"{stem}_pseudo_mask.png"
            mask.save(mask_path)

            mask_np = np.asarray(mask)
            values, counts = np.unique(mask_np, return_counts=True)
            total_pixels = mask_np.size
            foreground_ratio = float((mask_np > 0).sum()) / max(1, total_pixels)

            summary.append({
                "file_name": row["file_name"],
                "mask": str(mask_path),
                "mean_confidence": float(confidence.mean()),
                "min_confidence": float(confidence.min()),
                "foreground_ratio": foreground_ratio,
                "pixel_counts": {int(v): int(c) for v, c in zip(values, counts)},
            })

            if args.preview:
                color_overlay(image, mask).save(preview_dir / f"{stem}_overlay.png")

            if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(all_rows)):
                elapsed = time.perf_counter() - start
                eta = (elapsed / idx) * (len(all_rows) - idx)
                print(f"processed={idx}/{len(all_rows)} file={row['file_name']} conf={confidence.mean():.3f} fg_ratio={foreground_ratio:.3f} eta={eta:.0f}s", flush=True)

    # Quality audit
    confs = [s["mean_confidence"] for s in summary]
    fg_ratios = [s["foreground_ratio"] for s in summary]
    suspicious = [s for s in summary if s["mean_confidence"] < 0.6 or s["foreground_ratio"] < 0.01]

    audit = {
        "total_images": len(summary),
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "min_confidence": float(np.min(confs)) if confs else 0.0,
        "mean_foreground_ratio": float(np.mean(fg_ratios)) if fg_ratios else 0.0,
        "suspicious_count": len(suspicious),
        "suspicious_files": [s["file_name"] for s in suspicious[:20]],
        "scales": scales,
        "confidence_threshold": args.confidence_threshold,
        "min_area": args.min_area,
        "class_names": class_names,
        "image_size": image_size,
    }

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "quality_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start
    print(f"\nPSEUDO_MASK_DONE images={len(summary)} mean_conf={audit['mean_confidence']:.3f} "
          f"suspicious={audit['suspicious_count']} elapsed={elapsed:.1f}s output={args.output}")


if __name__ == "__main__":
    main()
