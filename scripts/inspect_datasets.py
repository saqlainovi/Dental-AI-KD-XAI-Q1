from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def count_files(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for item in path.rglob("*"):
        if item.is_file():
            counts[item.suffix.lower() or "<no_ext>"] += 1
    return counts


def print_counts(title: str, path: Path) -> None:
    print(f"\n== {title} ==")
    print(path)
    if not path.exists():
        print("missing")
        return
    for ext, count in count_files(path).most_common():
        print(f"{ext:10s} {count}")


def summarize_dentalai(path: Path) -> None:
    meta = path / "meta.json"
    if not meta.exists():
        return
    data = json.loads(meta.read_text(encoding="utf-8"))
    classes = [item["title"] for item in data.get("classes", [])]
    print(f"DentalAI classes: {', '.join(classes)}")
    for split in ["train", "valid", "test"]:
        img_count = len(list((path / split / "img").glob("*"))) if (path / split / "img").exists() else 0
        ann_count = len(list((path / split / "ann").glob("*.json"))) if (path / split / "ann").exists() else 0
        print(f"DentalAI {split}: images={img_count}, annotations={ann_count}")


def summarize_tufts(path: Path) -> None:
    root = path / "Radiographs" / "Radiographs"
    teeth_mask = path / "Segmentation" / "Segmentation" / "teeth_mask"
    jaw_mask = path / "Segmentation" / "Segmentation" / "maxillomandibular"
    print(f"TUFTS radiographs: {len(list(root.glob('*')) ) if root.exists() else 0}")
    print(f"TUFTS teeth masks: {len(list(teeth_mask.glob('*')) ) if teeth_mask.exists() else 0}")
    print(f"TUFTS jaw masks: {len(list(jaw_mask.glob('*')) ) if jaw_mask.exists() else 0}")
    for name in ["teeth_bbox.json", "teeth_polygon.json"]:
        file_path = path / "Segmentation" / "Segmentation" / name
        print(f"TUFTS {name}: {'yes' if file_path.exists() else 'no'}")


def summarize_dentex(path: Path) -> None:
    if not path.exists():
        return
    interesting = [
        path / "training_data",
        path / "validation_data",
        path / "training_data" / "quadrant_enumeration_disease",
        path / "validation_data" / "quadrant_enumeration_disease",
    ]
    for item in interesting:
        if item.exists():
            print(f"DENTEX found: {item.relative_to(ROOT)}")


def main() -> None:
    dentalai = RAW_DIR / "dentalai"
    tufts = RAW_DIR / "tufts"
    dentex = RAW_DIR / "dentex"

    print_counts("DentalAI", dentalai)
    summarize_dentalai(dentalai)

    print_counts("TUFTS", tufts)
    summarize_tufts(tufts)

    print_counts("DENTEX", dentex)
    summarize_dentex(dentex)


if __name__ == "__main__":
    main()
