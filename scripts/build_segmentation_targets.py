from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.segmentation.masks import coco_to_instance_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PNG segmentation masks from COCO polygon annotations.")
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate masks even if output files already exist.")
    args = parser.parse_args()

    stats = coco_to_instance_masks(args.coco, args.output, args.max_images, skip_existing=not args.overwrite)
    print(
        f"wrote={stats['written']} skipped_empty={stats['skipped_empty']} "
        f"skipped_existing={stats['skipped_existing']} output={args.output}"
    )


if __name__ == "__main__":
    main()
