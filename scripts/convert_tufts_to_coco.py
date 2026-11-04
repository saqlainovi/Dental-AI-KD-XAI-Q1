from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "tufts"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "coco" / "tufts_teeth.json"


def coco_bbox(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def polygon_area(flat: list[float]) -> float:
    if len(flat) < 6:
        return 0.0
    points = list(zip(flat[0::2], flat[1::2]))
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def build_image_index(radiograph_dir: Path) -> dict[str, tuple[int, int, int, str]]:
    index = {}
    for image_id, path in enumerate(sorted(radiograph_dir.glob("*")), start=1):
        if not path.is_file():
            continue
        with Image.open(path) as image:
            width, height = image.size
        index[path.name.lower()] = (image_id, width, height, path.name)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TUFTS tooth annotations to COCO.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    seg_dir = args.input / "Segmentation" / "Segmentation"
    radiograph_dir = args.input / "Radiographs" / "Radiographs"
    bbox_path = seg_dir / "teeth_bbox.json"
    polygon_path = seg_dir / "teeth_polygon.json"

    image_index = build_image_index(radiograph_dir)
    bbox_records = json.loads(bbox_path.read_text(encoding="utf-8"))
    polygon_records = json.loads(polygon_path.read_text(encoding="utf-8")) if polygon_path.exists() else []
    polygon_by_id = {item.get("External ID", "").lower(): item for item in polygon_records}

    images = []
    annotations = []
    ann_id = 1
    seen_images = set()

    for record in bbox_records:
        external_id = record.get("External ID", "")
        key = external_id.lower()
        if key not in image_index:
            continue
        image_id, width, height, file_name = image_index[key]
        if image_id not in seen_images:
            images.append(
                {
                    "id": image_id,
                    "file_name": f"Radiographs/Radiographs/{file_name}",
                    "width": width,
                    "height": height,
                }
            )
            seen_images.add(image_id)

        poly_record = polygon_by_id.get(key, {})
        polygons_by_title = {
            str(obj.get("title")): obj.get("polygons", [])
            for obj in poly_record.get("Label", {}).get("objects", [])
        }

        for obj in record.get("Label", {}).get("objects", []):
            title = str(obj.get("title"))
            box = obj.get("bounding box")
            if not box or len(box) != 4:
                continue
            segmentations = []
            area = 0.0
            for polygon in polygons_by_title.get(title, []):
                if len(polygon) < 3:
                    continue
                flat = [coord for point in polygon for coord in point]
                segmentations.append(flat)
                area += polygon_area(flat)
            bbox = coco_bbox(box)
            if area == 0.0:
                area = bbox[2] * bbox[3]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentations,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "tooth_label": title,
                }
            )
            ann_id += 1

    output = {
        "images": sorted(images, key=lambda item: item["id"]),
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Tooth"}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[write] {args.output} images={len(images)} annotations={len(annotations)}")


if __name__ == "__main__":
    main()
