from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "dentex"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "coco" / "dentex"


TASKS = {
    "quadrant": {
        "json": "training_data/quadrant/train_quadrant.json",
        "image_dir": "training_data/quadrant/xrays",
        "category_key": "category_id",
        "categories": [
            {"id": 0, "name": "quadrant_2"},
            {"id": 1, "name": "quadrant_1"},
            {"id": 2, "name": "quadrant_3"},
            {"id": 3, "name": "quadrant_4"},
        ],
    },
    "enumeration": {
        "json": "training_data/quadrant_enumeration/train_quadrant_enumeration.json",
        "image_dir": "training_data/quadrant_enumeration/xrays",
        "category_key": "category_id_2",
        "categories": [{"id": idx, "name": f"tooth_position_{idx}"} for idx in range(8)],
    },
    "disease": {
        "json": "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        "image_dir": "training_data/quadrant-enumeration-disease/xrays",
        "category_key": "category_id_3",
        "categories": [{"id": idx, "name": f"disease_class_{idx}"} for idx in range(4)],
    },
}


def normalize_task(dataset_dir: Path, task_name: str, config: dict) -> dict:
    source = dataset_dir / config["json"]
    data = json.loads(source.read_text(encoding="utf-8"))
    image_dir = Path(config["image_dir"])

    images = []
    for image in data.get("images", []):
        item = dict(image)
        item["file_name"] = str(image_dir / image["file_name"]).replace("\\", "/")
        images.append(item)

    annotations = []
    category_key = config["category_key"]
    for ann in data.get("annotations", []):
        if category_key not in ann:
            continue
        item = {
            "id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": ann[category_key],
            "bbox": ann.get("bbox", []),
            "segmentation": ann.get("segmentation", []),
            "area": ann.get("area", 0),
            "iscrowd": ann.get("iscrowd", 0),
            "attributes": {},
        }
        for key in ("category_id_1", "category_id_2", "category_id_3"):
            if key in ann:
                item["attributes"][key] = ann[key]
        if not item["attributes"]:
            item.pop("attributes")
        annotations.append(item)

    return {
        "info": {
            "description": f"DENTEX {task_name} normalized annotations",
            "source_json": config["json"],
            "primary_category_key": category_key,
        },
        "images": images,
        "annotations": annotations,
        "categories": config["categories"],
    }


def build_validation_images(dataset_dir: Path) -> dict:
    image_dir = dataset_dir / "validation_data" / "quadrant_enumeration_disease" / "xrays"
    images = []
    for image_id, path in enumerate(sorted(image_dir.glob("*.png")), start=1):
        images.append(
            {
                "id": image_id,
                "file_name": str(path.relative_to(dataset_dir)).replace("\\", "/"),
            }
        )
    return {
        "info": {"description": "DENTEX validation images without released annotations"},
        "images": images,
        "annotations": [],
        "categories": TASKS["disease"]["categories"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize DENTEX task annotations for notebook experiments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for task_name, config in TASKS.items():
        output = normalize_task(args.input, task_name, config)
        out_file = args.output / f"{task_name}.json"
        out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(
            f"[write] {out_file} images={len(output['images'])} "
            f"annotations={len(output['annotations'])}"
        )

    validation = build_validation_images(args.input)
    validation_file = args.output / "validation_images.json"
    validation_file.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"[write] {validation_file} images={len(validation['images'])}")


if __name__ == "__main__":
    main()
