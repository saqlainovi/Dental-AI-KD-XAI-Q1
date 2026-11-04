from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]


def resolve_image_size(image_row: dict, image_path: Path) -> tuple[int, int]:
    width = image_row.get("width")
    height = image_row.get("height")
    if width and height:
        return int(width), int(height)
    with Image.open(image_path) as image:
        return image.size


def draw_segmentation(mask: Image.Image, segmentation: object) -> None:
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not polygon or len(polygon) < 6:
                continue
            points = [(float(polygon[idx]), float(polygon[idx + 1])) for idx in range(0, len(polygon) - 1, 2)]
            draw.polygon(points, fill=255)


def draw_bbox(mask: Image.Image, bbox: Iterable[float], mode: str) -> None:
    x, y, w, h = [float(value) for value in bbox]
    box = [x, y, x + w, y + h]
    draw = ImageDraw.Draw(mask)
    if mode == "ellipse":
        draw.ellipse(box, fill=255)
    else:
        draw.rectangle(box, fill=255)


def build_mask(image_size: tuple[int, int], annotations: list[dict], mask_source: str) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    for ann in annotations:
        if mask_source == "segmentation" and ann.get("segmentation"):
            draw_segmentation(mask, ann.get("segmentation"))
        elif ann.get("bbox"):
            draw_bbox(mask, ann["bbox"], "ellipse" if mask_source == "ellipse" else "bbox")
    return mask


def binary_to_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask_arr = np.asarray(mask) > 0
    overlay = Image.new("RGB", image.size, (255, 80, 80))
    alpha = Image.fromarray((mask_arr.astype(np.uint8) * 110), mode="L")
    out = image.copy()
    out.paste(overlay, (0, 0), alpha)
    return out


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_task(
    coco_path: Path,
    dentex_root: Path,
    output_dir: Path,
    task: str,
    mask_source: str,
    preview: bool,
    preview_limit: int,
    max_images: int | None,
) -> dict:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    if max_images:
        images = images[:max_images]

    annotations_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        annotations_by_image.setdefault(image_id, []).append(ann)

    mask_dir = output_dir / task / "masks"
    preview_dir = output_dir / task / "previews"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    kept = 0
    for index, image_row in enumerate(images, start=1):
        file_name = image_row.get("file_name", "")
        image_path = (dentex_root / file_name).resolve()
        anns = annotations_by_image.get(int(image_row.get("id", -1)), [])
        status = "kept" if anns else "filtered_no_annotation"
        if not image_path.exists():
            status = "filtered_missing_image"
            image_size = (int(image_row.get("width", 1) or 1), int(image_row.get("height", 1) or 1))
        else:
            image_size = resolve_image_size(image_row, image_path)

        mask = build_mask(image_size, anns, mask_source) if anns else Image.new("L", image_size, 0)
        fraction = float((np.asarray(mask) > 0).mean())
        if anns and fraction <= 0:
            status = "filtered_empty_mask"

        stem = Path(file_name).stem or f"image_{index:05d}"
        mask_path = (mask_dir / f"{stem}_clean_mask.png").resolve()
        mask.save(mask_path)
        if status == "kept":
            kept += 1

        if preview and image_path.exists() and (preview_limit <= 0 or index <= preview_limit):
            with Image.open(image_path) as image:
                binary_to_overlay(image, mask).save(preview_dir / f"{stem}_clean_overlay.png")

        rows.append(
            {
                "file_name": file_name,
                "image_path": str(image_path),
                "raw_mask": str(mask_path),
                "clean_mask": str(mask_path),
                "mean_confidence": 1.0 if status == "kept" else 0.0,
                "original_tooth_fraction": fraction,
                "clean_tooth_fraction": fraction,
                "annotation_count": len(anns),
                "mask_source": mask_source,
                "status": status,
            }
        )
        if index == 1 or index % 100 == 0 or index == len(images):
            print(f"{task}: processed={index}/{len(images)} kept={kept}", flush=True)

    (output_dir / task / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_rows_csv(output_dir / task / "summary.csv", rows)
    return {"task": task, "rows": len(rows), "kept": kept, "filtered": len(rows) - kept, "mask_source": mask_source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DENTEX training masks directly from COCO polygons/bboxes.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dentex-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--task", action="append", nargs=2, metavar=("TASK_NAME", "COCO_JSON"), required=True)
    parser.add_argument("--mask-source", choices=["segmentation", "bbox", "ellipse"], default="segmentation")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=12, help="Maximum overlays per task. Use 0 for all.")
    parser.add_argument("--max-images", type=int, default=0, help="Use 0 for all images.")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    max_images = None if args.max_images == 0 else args.max_images
    report = []
    for task, coco_json in args.task:
        report.append(
            process_task(
                coco_path=Path(coco_json),
                dentex_root=args.dentex_root,
                output_dir=args.output_root,
                task=task,
                mask_source=args.mask_source,
                preview=args.preview,
                preview_limit=args.preview_limit,
                max_images=max_images,
            )
        )
    (args.output_root / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
