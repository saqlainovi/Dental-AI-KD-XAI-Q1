from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits


def unique_images(coco_path: Path, dentex_root: Path, max_images: int | None) -> list[dict]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    seen = set()
    rows = []
    for image in data.get("images", []):
        file_name = image.get("file_name")
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        path = dentex_root / file_name
        if path.exists():
            rows.append({"id": image.get("id"), "file_name": file_name, "path": path})
        if max_images and len(rows) >= max_images:
            break
    return rows


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def color_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask_arr = np.asarray(mask)
    color = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
    palette = {
        1: (255, 80, 80),
        2: (255, 210, 70),
        3: (80, 220, 255),
        4: (80, 255, 120),
    }
    for value, rgb in palette.items():
        color[mask_arr == value] = rgb
    overlay = Image.fromarray(color).resize(image.size, Image.NEAREST)
    alpha = Image.fromarray(((mask_arr > 0).astype(np.uint8) * 110)).resize(image.size, Image.NEAREST)
    out = image.copy()
    out.paste(overlay, (0, 0), alpha)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DENTEX pseudo masks from the TUFTS X-ray teacher model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/teacher_tinyunet_tufts_xray_full.pt"))
    parser.add_argument("--coco", type=Path, default=ROOT / "data/processed/coco/dentex/disease.json")
    parser.add_argument("--dentex-root", type=Path, default=ROOT / "data/raw/dentex")
    parser.add_argument("--output", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_teacher_smoke"))
    parser.add_argument("--max-images", type=int, default=12, help="Number of images to process. Use 0 for all images.")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", ["background", "Caries", "Cavity", "Crack", "Tooth"])
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 224))
    model = build_model_from_checkpoint(checkpoint, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mask_dir = args.output / "masks"
    preview_dir = args.output / "previews"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    max_images = None if args.max_images == 0 else args.max_images
    rows = unique_images(args.coco, args.dentex_root, max_images)
    print(f"processing_images={len(rows)} output={args.output}", flush=True)
    summary = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            image = Image.open(row["path"]).convert("RGB")
            tensor = preprocess(image, image_size).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = segmentation_logits(model(tensor))
                probs = torch.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)
            mask_small = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
            conf_mean = float(confidence.mean().detach().cpu())
            mask = Image.fromarray(mask_small).resize(image.size, Image.NEAREST)
            stem = Path(row["file_name"]).stem
            mask_path = mask_dir / f"{stem}_pseudo_mask.png"
            mask.save(mask_path)
            values, counts = np.unique(np.asarray(mask), return_counts=True)
            summary.append(
                {
                    "file_name": row["file_name"],
                    "mask": str(mask_path),
                    "mean_confidence": conf_mean,
                    "pixel_counts": {int(v): int(c) for v, c in zip(values, counts)},
                }
            )
            if args.preview:
                color_overlay(image, mask).save(preview_dir / f"{stem}_overlay.png")
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(rows)):
                print(f"processed={index}/{len(rows)} file={row['file_name']}", flush=True)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote_masks={len(summary)} output={args.output} classes={class_names}")


if __name__ == "__main__":
    main()
