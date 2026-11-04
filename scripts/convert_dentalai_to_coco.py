from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "dentalai"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "coco" / "dentalai"


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def bbox_from_points(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def load_classes(dataset_dir: Path) -> dict[str, int]:
    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    return {item["title"]: idx + 1 for idx, item in enumerate(meta.get("classes", []))}


def convert_split(dataset_dir: Path, output_dir: Path, split: str) -> None:
    class_to_id = load_classes(dataset_dir)
    images = []
    annotations = []
    ann_id = 1

    image_dir = dataset_dir / split / "img"
    ann_dir = dataset_dir / split / "ann"

    for image_id, image_path in enumerate(sorted(image_dir.glob("*")), start=1):
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.relative_to(dataset_dir)).replace("\\", "/"),
                "width": width,
                "height": height,
            }
        )

        ann_path = ann_dir / f"{image_path.name}.json"
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        for obj in data.get("objects", []):
            class_title = obj.get("classTitle")
            points = obj.get("points", {}).get("exterior", [])
            if class_title not in class_to_id or len(points) < 3:
                continue
            segmentation = [coord for point in points for coord in point]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_to_id[class_title],
                    "segmentation": [segmentation],
                    "bbox": bbox_from_points(points),
                    "area": polygon_area(points),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": cid, "name": name} for name, cid in class_to_id.items()]
    output = {"images": images, "annotations": annotations, "categories": categories}
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{split}.json"
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[write] {out_file} images={len(images)} annotations={len(annotations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DentalAI Supervisely annotations to COCO.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        convert_split(args.input, args.output, split)


if __name__ == "__main__":
    main()
