from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from skimage import morphology


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]


def binary_to_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image = image.convert("RGB")
    overlay = Image.new("RGB", image.size, (255, 80, 80))
    alpha = Image.fromarray((mask.astype(np.uint8) * 110)).resize(image.size, Image.NEAREST)
    out = image.copy()
    out.paste(overlay, (0, 0), alpha)
    return out


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    height, width = mask.shape

    # Panoramic exports often contain labels/text or scan borders at the edges.
    x_margin = max(1, int(width * 0.025))
    y_margin_top = max(1, int(height * 0.02))
    y_margin_bottom = max(1, int(height * 0.01))
    mask[:, :x_margin] = False
    mask[:, width - x_margin :] = False
    mask[:y_margin_top, :] = False
    mask[height - y_margin_bottom :, :] = False

    min_object = max(64, int(height * width * 0.0006))
    min_hole = max(64, int(height * width * 0.0004))
    mask = morphology.remove_small_objects(mask, min_size=min_object)
    mask = morphology.remove_small_holes(mask, area_threshold=min_hole)

    if mask.any():
        row_sum = mask.sum(axis=1).astype(np.float32)
        kernel = max(9, int(height * 0.025))
        smooth = np.convolve(row_sum, np.ones(kernel, dtype=np.float32) / kernel, mode="same")
        center = int(np.argmax(smooth))
        top = max(0, center - int(height * 0.30))
        bottom = min(height, center + int(height * 0.34))
        band = np.zeros_like(mask)
        band[top:bottom, :] = True
        mask &= band
        mask = morphology.remove_small_objects(mask, min_size=min_object)
        mask = morphology.remove_small_holes(mask, area_threshold=min_hole)

    return mask.astype(np.uint8)


def tooth_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def load_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def process_task(
    task_dir: Path,
    output_dir: Path,
    dentex_root: Path,
    min_tooth_fraction: float,
    max_tooth_fraction: float,
    min_confidence: float,
    preview: bool,
    max_images: int | None,
) -> list[dict]:
    task_dir = task_dir.resolve()
    output_dir = output_dir.resolve()
    dentex_root = dentex_root.resolve()
    rows = load_summary(task_dir / "summary.json")
    if max_images:
        rows = rows[:max_images]

    mask_dir = output_dir / "masks"
    preview_dir = output_dir / "previews"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    cleaned_rows: list[dict] = []
    for index, row in enumerate(rows, start=1):
        raw_mask_path = Path(row["mask"])
        if not raw_mask_path.is_absolute():
            raw_mask_path = (ROOT / raw_mask_path).resolve()
        image_path = (dentex_root / row["file_name"]).resolve()
        raw_mask = np.asarray(Image.open(raw_mask_path))
        original_binary = raw_mask > 0
        original_fraction = tooth_fraction(original_binary)
        confidence = float(row.get("mean_confidence", 0.0))

        status = "kept"
        clean_mask = clean_binary_mask(original_binary)
        clean_fraction = tooth_fraction(clean_mask)
        if confidence < min_confidence:
            status = "filtered_low_confidence"
        elif original_fraction < min_tooth_fraction or original_fraction > max_tooth_fraction:
            status = "filtered_original_fraction"
        elif clean_fraction < min_tooth_fraction or clean_fraction > max_tooth_fraction:
            status = "filtered_clean_fraction"

        stem = Path(row["file_name"]).stem
        clean_mask_path = (mask_dir / f"{stem}_clean_mask.png").resolve()
        Image.fromarray((clean_mask * 255).astype(np.uint8)).save(clean_mask_path)

        if preview and image_path.exists():
            image = Image.open(image_path).convert("RGB")
            binary_to_overlay(image, clean_mask).save(preview_dir / f"{stem}_clean_overlay.png")

        cleaned_rows.append(
            {
                "file_name": row["file_name"],
                "image_path": str(image_path),
                "raw_mask": str(raw_mask_path),
                "clean_mask": str(clean_mask_path),
                "mean_confidence": confidence,
                "original_tooth_fraction": original_fraction,
                "clean_tooth_fraction": clean_fraction,
                "status": status,
            }
        )
        if index == 1 or index % 100 == 0 or index == len(rows):
            kept = sum(item["status"] == "kept" for item in cleaned_rows)
            print(f"{task_dir.name}: processed={index}/{len(rows)} kept={kept}", flush=True)

    (output_dir / "summary.json").write_text(json.dumps(cleaned_rows, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cleaned_rows[0].keys()) if cleaned_rows else [])
        if cleaned_rows:
            writer.writeheader()
            writer.writerows(cleaned_rows)
    return cleaned_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and quality-gate DENTEX pseudo masks.")
    parser.add_argument("--input-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_xray_teacher"))
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_xray_teacher_clean"))
    parser.add_argument("--dentex-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["quadrant_full", "enumeration_full", "disease_full", "validation_full"],
    )
    parser.add_argument("--min-tooth-fraction", type=float, default=0.06)
    parser.add_argument("--max-tooth-fraction", type=float, default=0.32)
    parser.add_argument("--min-confidence", type=float, default=0.96)
    parser.add_argument("--max-images", type=int, default=0, help="Use 0 for all images.")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    max_images = None if args.max_images == 0 else args.max_images
    global_report = []
    for task in args.tasks:
        task_dir = args.input_root / task
        if not (task_dir / "summary.json").exists():
            print(f"skip_missing={task_dir}", flush=True)
            continue
        rows = process_task(
            task_dir=task_dir,
            output_dir=args.output_root / task,
            dentex_root=args.dentex_root,
            min_tooth_fraction=args.min_tooth_fraction,
            max_tooth_fraction=args.max_tooth_fraction,
            min_confidence=args.min_confidence,
            preview=args.preview,
            max_images=max_images,
        )
        kept = sum(row["status"] == "kept" for row in rows)
        global_report.append({"task": task, "rows": len(rows), "kept": kept, "filtered": len(rows) - kept})

    (args.output_root / "quality_report.json").write_text(json.dumps(global_report, indent=2), encoding="utf-8")
    print(json.dumps(global_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
