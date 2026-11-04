from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFile
from skimage import measure, morphology


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits


def resolve_image_path(file_name: str, image_root: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() and path.exists():
        return path
    for candidate in (image_root / file_name, ROOT / file_name):
        if candidate.exists():
            return candidate
    return image_root / file_name


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(x, width - 1.0))
    y1 = max(0.0, min(y, height - 1.0))
    x2 = max(0.0, min(x + w, float(width)))
    y2 = max(0.0, min(y + h, float(height)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return [x1, y1, x2, y2]


def load_coco_records(coco_json: Path, image_root: Path, max_images: int = 0) -> list[dict]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco.get("images", [])}
    anns_by_image = {image_id: [] for image_id in images}
    for ann in coco.get("annotations", []):
        if ann.get("bbox") and ann["image_id"] in anns_by_image:
            anns_by_image[ann["image_id"]].append(ann)
    records = []
    for image_id, image in images.items():
        image_path = resolve_image_path(image["file_name"], image_root)
        if not image_path.exists():
            continue
        width = int(image.get("width") or Image.open(image_path).width)
        height = int(image.get("height") or Image.open(image_path).height)
        boxes = []
        for ann in anns_by_image.get(image_id, []):
            box = clamp_box(ann["bbox"], width, height)
            if box is not None:
                boxes.append(box)
        if boxes:
            records.append(
                {
                    "image_id": int(image_id),
                    "file_name": image["file_name"],
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )
    if max_images:
        return records[:max_images]
    return records


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xa = np.maximum(x11[:, None], x21[None, :])
    ya = np.maximum(y11[:, None], y21[None, :])
    xb = np.minimum(x12[:, None], x22[None, :])
    yb = np.minimum(y12[:, None], y22[None, :])
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area1 = np.maximum(0, x12 - x11) * np.maximum(0, y12 - y11)
    area2 = np.maximum(0, x22 - x21) * np.maximum(0, y22 - y21)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def compute_ap_at_iou(predictions: list[dict], gt_by_image: dict[int, np.ndarray], iou_threshold: float) -> float:
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_by_image.items()}
    flat = []
    for pred in predictions:
        for box, score in zip(pred["boxes"], pred["scores"]):
            flat.append((pred["image_id"], float(score), box))
    flat.sort(key=lambda row: row[1], reverse=True)
    tps, fps = [], []
    for image_id, _score, box in flat:
        gt_boxes = gt_by_image.get(image_id, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            tps.append(0)
            fps.append(1)
            continue
        ious = box_iou(np.asarray([box], dtype=np.float32), gt_boxes)[0]
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_threshold and not matched[image_id][best_idx]:
            matched[image_id][best_idx] = True
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    if not tps:
        return 0.0
    tp = np.cumsum(tps)
    fp = np.cumsum(fps)
    recalls = tp / max(1, total_gt)
    precisions = tp / np.maximum(tp + fp, 1)
    return average_precision(recalls, precisions)


def preprocess(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)


@torch.no_grad()
def predict_prob(model: torch.nn.Module, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    tensor = preprocess(image, image_size, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = segmentation_logits(model(tensor))
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
    small = prob.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.asarray(Image.fromarray(small, mode="F").resize(image.size, Image.BILINEAR))


def mask_to_component_boxes(prob: np.ndarray, threshold: float, min_area: int) -> tuple[list[list[float]], list[float]]:
    mask = prob >= threshold
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    labels = measure.label(mask)
    boxes = []
    scores = []
    for region in measure.regionprops(labels, intensity_image=prob):
        if region.area < min_area:
            continue
        y1, x1, y2, x2 = region.bbox
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        scores.append(float(region.mean_intensity))
    return boxes, scores


def union_box(boxes: list[list[float]] | np.ndarray) -> list[float] | None:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return None
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())]


def evaluate_source(
    source: str,
    records: list[dict],
    model: torch.nn.Module,
    image_size: int,
    threshold: float,
    min_area_fraction: float,
    device: torch.device,
    figures_dir: Path,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    gt_by_image = {rec["image_id"]: np.asarray(rec["boxes"], dtype=np.float32) for rec in records}
    predictions = []
    region_rows = []
    start = time.perf_counter()
    for rec in records:
        image = Image.open(rec["image_path"]).convert("RGB")
        prob = predict_prob(model, image, image_size, device)
        min_area = max(16, int(prob.shape[0] * prob.shape[1] * min_area_fraction))
        boxes, scores = mask_to_component_boxes(prob, threshold, min_area)
        predictions.append({"image_id": rec["image_id"], "boxes": boxes, "scores": scores})
        gt_union = union_box(rec["boxes"])
        pred_union = union_box(boxes)
        if gt_union is not None and pred_union is not None:
            iou = float(box_iou(np.asarray([pred_union]), np.asarray([gt_union]))[0, 0])
        else:
            iou = 0.0
        region_rows.append(
            {
                "source": source,
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "gt_boxes": len(rec["boxes"]),
                "pred_boxes": len(boxes),
                "region_iou": iou,
                "region_hit_50": int(iou >= 0.50),
                "region_hit_75": int(iou >= 0.75),
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - start)
    thresholds = [round(float(v), 2) for v in np.arange(0.50, 1.00, 0.05)]
    aps = {f"component_ap_{thr:.2f}": compute_ap_at_iou(predictions, gt_by_image, thr) for thr in thresholds}
    region_df = pd.DataFrame(region_rows)
    metrics = {
        "source": source,
        "images": len(records),
        "gt_boxes": int(sum(len(rec["boxes"]) for rec in records)),
        "pred_boxes": int(sum(len(pred["boxes"]) for pred in predictions)),
        "component_map_50_95": float(np.mean(list(aps.values()))),
        "component_map_50": aps["component_ap_0.50"],
        "component_map_75": aps["component_ap_0.75"],
        "region_mean_iou": float(region_df["region_iou"].mean()),
        "region_hit_50": float(region_df["region_hit_50"].mean()),
        "region_hit_75": float(region_df["region_hit_75"].mean()),
        "eval_images_per_second": len(records) / elapsed,
        "note": "Segmentation-derived localization. Component AP is instance-like and conservative; region IoU measures whole tooth-region localization.",
        **aps,
    }
    plot_examples(records, predictions, figures_dir / f"{source.lower().replace(' ', '_')}_segmentation_localization_examples.png")
    return metrics, region_df, predictions


def plot_examples(records: list[dict], predictions: list[dict], output_path: Path, count: int = 4) -> None:
    pred_by_id = {pred["image_id"]: pred for pred in predictions}
    examples = records[:count]
    thumbs = []
    for rec in examples:
        image = Image.open(rec["image_path"]).convert("RGB")
        scale = 460 / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        for box in rec["boxes"]:
            draw.rectangle([v * scale for v in box], outline=(20, 160, 90), width=2)
        for box in pred_by_id.get(rec["image_id"], {}).get("boxes", []):
            draw.rectangle([v * scale for v in box], outline=(220, 50, 50), width=2)
        thumbs.append(image)
    if not thumbs:
        return
    width = max(image.width for image in thumbs)
    height = sum(image.height for image in thumbs)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in thumbs:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation-derived localization/detection metrics.")
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/fresh_rerun_20260518_112354/student_tinyunet_dentex_pseudo_full_best.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/fresh_rerun_20260518_112354/complete_research_single_notebook_v1/segmentation_localization_v1"))
    parser.add_argument("--dentex-coco", type=Path, default=ROOT / "data" / "processed" / "coco" / "dentex" / "enumeration.json")
    parser.add_argument("--dentex-root", type=Path, default=ROOT / "data" / "raw" / "dentex")
    parser.add_argument("--tufts-coco", type=Path, default=ROOT / "data" / "processed" / "coco" / "tufts_teeth.json")
    parser.add_argument("--tufts-root", type=Path, default=ROOT / "data" / "raw" / "tufts")
    parser.add_argument("--max-dentex", type=int, default=0)
    parser.add_argument("--max-tufts", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-fraction", type=float, default=0.00008)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_root / "tables"
    figures_dir = args.output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 224))
    model = build_model_from_checkpoint(checkpoint, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sources = [
        ("DENTEX enumeration boxes", load_coco_records(args.dentex_coco, args.dentex_root, args.max_dentex)),
        ("TUFTS external boxes", load_coco_records(args.tufts_coco, args.tufts_root, args.max_tufts)),
    ]
    metric_rows = []
    region_frames = []
    for source, records in sources:
        metrics, region_df, _predictions = evaluate_source(
            source,
            records,
            model,
            image_size,
            args.threshold,
            args.min_area_fraction,
            device,
            figures_dir,
        )
        metric_rows.append(metrics)
        region_frames.append(region_df)
        print(
            f"{source}: component_mAP50={metrics['component_map_50']:.4f} "
            f"component_mAP50:95={metrics['component_map_50_95']:.4f} "
            f"region_iou={metrics['region_mean_iou']:.4f}",
            flush=True,
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables_dir / "segmentation_localization_metrics.csv", index=False)
    pd.concat(region_frames, ignore_index=True).to_csv(tables_dir / "segmentation_localization_per_image.csv", index=False)
    (args.output_root / "segmentation_localization_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "image_size": image_size,
                "threshold": args.threshold,
                "min_area_fraction": args.min_area_fraction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SEGMENTATION_LOCALIZATION_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
