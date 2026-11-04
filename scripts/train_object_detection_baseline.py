from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]

from train_dentex_student_segmentation import GPUEpochSampler


def resolve_image_path(file_name: str, image_root: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() and path.exists():
        return path
    candidate = image_root / file_name
    if candidate.exists():
        return candidate
    candidate = ROOT / file_name
    if candidate.exists():
        return candidate
    return image_root / file_name


def load_category_names(coco_json: Path, class_agnostic: bool) -> list[str]:
    if class_agnostic:
        return ["tooth"]
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = sorted(coco.get("categories", []), key=lambda category: int(category.get("id", 0)))
    return [str(category.get("name", f"class_{idx}")) for idx, category in enumerate(categories)]


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(x, width - 1.0))
    y1 = max(0.0, min(y, height - 1.0))
    x2 = max(0.0, min(x + w, float(width)))
    y2 = max(0.0, min(y + h, float(height)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return [x1, y1, x2, y2]


def load_coco_records(
    coco_json: Path,
    image_root: Path,
    class_agnostic: bool,
    max_images: int,
    seed: int,
) -> list[dict]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for ann in coco.get("annotations", []):
        if ann.get("bbox") and ann["image_id"] in anns_by_image:
            anns_by_image[ann["image_id"]].append(ann)

    records = []
    for image_id, image in images.items():
        image_path = resolve_image_path(image["file_name"], image_root)
        if not image_path.exists():
            continue
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        boxes = []
        labels = []
        for ann in anns_by_image.get(image_id, []):
            box = clamp_box(ann["bbox"], width, height)
            if box is None:
                continue
            boxes.append(box)
            labels.append(1 if class_agnostic else int(ann["category_id"]) + 1)
        if boxes:
            records.append(
                {
                    "image_id": int(image_id),
                    "file_name": image["file_name"],
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(records)
    if max_images:
        records = records[:max_images]
    return records


class CocoDetectionRecords(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        rec = self.records[index]
        image = Image.open(rec["image_path"]).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))
        target = {
            "boxes": torch.tensor(rec["boxes"], dtype=torch.float32),
            "labels": torch.tensor(rec["labels"], dtype=torch.int64),
            "image_id": torch.tensor([rec["image_id"]], dtype=torch.int64),
            "area": torch.tensor(
                [(box[2] - box[0]) * (box[3] - box[1]) for box in rec["boxes"]],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((len(rec["boxes"]),), dtype=torch.int64),
        }
        return tensor, target


def collate_detection(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


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


def compute_ap_at_iou(predictions: list[dict], gt_by_image: dict[int, np.ndarray], iou_threshold: float) -> tuple[float, pd.DataFrame]:
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return 0.0, pd.DataFrame()
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_by_image.items()}
    flat_preds = []
    for pred in predictions:
        image_id = int(pred["image_id"])
        for box, score in zip(pred["boxes"], pred["scores"]):
            flat_preds.append((image_id, float(score), box))
    flat_preds.sort(key=lambda item: item[1], reverse=True)

    tps = []
    fps = []
    for image_id, _score, box in flat_preds:
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
        return 0.0, pd.DataFrame({"recall": [0.0], "precision": [0.0]})
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    return average_precision(recalls, precisions), pd.DataFrame({"recall": recalls, "precision": precisions})


def compute_detection_confusion(
    predictions: list[dict],
    gt_boxes_by_image: dict[int, np.ndarray],
    gt_labels_by_image: dict[int, np.ndarray],
    class_names: list[str],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(class_names)
    matrix = pd.DataFrame(
        0,
        index=labels + ["background_fp"],
        columns=labels + ["missed_fn"],
        dtype=np.int64,
    )
    per_class = {
        idx + 1: {"support": 0, "matched": 0, "class_correct": 0, "false_positive": 0, "missed": 0}
        for idx in range(len(labels))
    }

    pred_by_image = {int(pred["image_id"]): pred for pred in predictions}
    all_image_ids = sorted(set(gt_boxes_by_image) | set(pred_by_image))
    for image_id in all_image_ids:
        gt_boxes = gt_boxes_by_image.get(image_id, np.zeros((0, 4), dtype=np.float32))
        gt_labels = gt_labels_by_image.get(image_id, np.zeros((0,), dtype=np.int64))
        for label in gt_labels:
            if int(label) in per_class:
                per_class[int(label)]["support"] += 1
        pred = pred_by_image.get(
            image_id,
            {"boxes": np.zeros((0, 4), dtype=np.float32), "scores": np.zeros((0,), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)},
        )
        keep = np.asarray(pred["scores"]) >= score_threshold
        pred_boxes = np.asarray(pred["boxes"])[keep]
        pred_scores = np.asarray(pred["scores"])[keep]
        pred_labels = np.asarray(pred["labels"])[keep].astype(np.int64)
        order = np.argsort(-pred_scores)
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        for pred_idx in order:
            pred_box = pred_boxes[pred_idx]
            pred_label = int(pred_labels[pred_idx])
            pred_name = labels[pred_label - 1] if 1 <= pred_label <= len(labels) else f"class_{pred_label}"
            if len(gt_boxes) == 0:
                if 1 <= pred_label <= len(labels):
                    per_class[pred_label]["false_positive"] += 1
                if pred_name not in matrix.columns:
                    matrix[pred_name] = 0
                matrix.loc["background_fp", pred_name] += 1
                continue

            ious = box_iou(np.asarray([pred_box], dtype=np.float32), gt_boxes)[0]
            sorted_gt = np.argsort(-ious)
            best_gt = None
            for candidate in sorted_gt:
                if not matched_gt[candidate] and ious[candidate] >= iou_threshold:
                    best_gt = int(candidate)
                    break
            if best_gt is None:
                if 1 <= pred_label <= len(labels):
                    per_class[pred_label]["false_positive"] += 1
                if pred_name not in matrix.columns:
                    matrix[pred_name] = 0
                matrix.loc["background_fp", pred_name] += 1
                continue

            matched_gt[best_gt] = True
            true_label = int(gt_labels[best_gt])
            true_name = labels[true_label - 1] if 1 <= true_label <= len(labels) else f"class_{true_label}"
            if true_name not in matrix.index:
                matrix.loc[true_name] = 0
            if pred_name not in matrix.columns:
                matrix[pred_name] = 0
            matrix.loc[true_name, pred_name] += 1
            if true_label in per_class:
                per_class[true_label]["matched"] += 1
                if pred_label == true_label:
                    per_class[true_label]["class_correct"] += 1
            if pred_label != true_label and 1 <= pred_label <= len(labels):
                per_class[pred_label]["false_positive"] += 1

        for gt_idx, was_matched in enumerate(matched_gt):
            if was_matched:
                continue
            true_label = int(gt_labels[gt_idx])
            true_name = labels[true_label - 1] if 1 <= true_label <= len(labels) else f"class_{true_label}"
            if true_name not in matrix.index:
                matrix.loc[true_name] = 0
            matrix.loc[true_name, "missed_fn"] += 1
            if true_label in per_class:
                per_class[true_label]["missed"] += 1

    rows = []
    for label_id, stats in per_class.items():
        class_correct = stats["class_correct"]
        false_positive = stats["false_positive"]
        missed = stats["missed"]
        support = stats["support"]
        precision = class_correct / max(1, class_correct + false_positive)
        recall = class_correct / max(1, support)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        rows.append(
            {
                "class_id": label_id,
                "class_name": labels[label_id - 1],
                "support": support,
                "matched_iou50": stats["matched"],
                "class_correct_iou50": class_correct,
                "missed_fn": missed,
                "false_positive": false_positive,
                "precision_iou50": precision,
                "recall_iou50": recall,
                "f1_iou50": f1,
            }
        )
    return matrix, pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gt_by_image: dict[int, np.ndarray],
) -> tuple[dict[str, float], pd.DataFrame, list[dict]]:
    model.eval()
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            predictions.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "boxes": output["boxes"].detach().cpu().numpy(),
                    "scores": output["scores"].detach().cpu().numpy(),
                    "labels": output["labels"].detach().cpu().numpy(),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - start)

    thresholds = [round(float(v), 2) for v in np.arange(0.50, 1.00, 0.05)]
    aps = {}
    pr50 = pd.DataFrame()
    for threshold in thresholds:
        ap, pr = compute_ap_at_iou(predictions, gt_by_image, threshold)
        aps[f"ap_{threshold:.2f}"] = ap
        if abs(threshold - 0.50) < 1e-6:
            pr50 = pr
    metrics = {
        "map_50_95": float(np.mean(list(aps.values()))),
        "map_50": aps["ap_0.50"],
        "map_75": aps["ap_0.75"],
        "val_images": len(gt_by_image),
        "val_gt_boxes": int(sum(len(v) for v in gt_by_image.values())),
        "eval_images_per_second": len(gt_by_image) / elapsed,
        **aps,
    }
    return metrics, pr50, predictions


def plot_pr_curve(pr: pd.DataFrame, output_path: Path, ap50: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if not pr.empty:
        ax.plot(pr["recall"], pr["precision"], color="#2563eb")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Object Detection PR Curve at IoU 0.50 (AP={ap50:.3f})")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: list[dict], output_path: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o")
    axes[0].set_title("Detector Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(df["epoch"], df["map_50"], marker="o", label="mAP@50")
    axes[1].plot(df["epoch"], df["map_75"], marker="o", label="mAP@75")
    axes[1].plot(df["epoch"], df["map_50_95"], marker="o", label="mAP@50:95")
    axes[1].set_title("Detector Validation mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_examples(records: list[dict], predictions: list[dict], output_path: Path, max_examples: int, score_threshold: float) -> None:
    pred_by_id = {int(pred["image_id"]): pred for pred in predictions}
    examples = [record for record in records if record["image_id"] in pred_by_id][:max_examples]
    if not examples:
        return
    thumbs = []
    for rec in examples:
        image = Image.open(rec["image_path"]).convert("RGB")
        scale = 420 / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        for box in rec["boxes"]:
            scaled = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]
            draw.rectangle(scaled, outline=(30, 180, 90), width=2)
        pred = pred_by_id[rec["image_id"]]
        for box, score in zip(pred["boxes"], pred["scores"]):
            if float(score) < score_threshold:
                continue
            scaled = [float(box[0]) * scale, float(box[1]) * scale, float(box[2]) * scale, float(box[3]) * scale]
            draw.rectangle(scaled, outline=(220, 50, 50), width=2)
            draw.text((scaled[0], scaled[1]), f"{float(score):.2f}", fill=(220, 50, 50))
        thumbs.append(image)

    width = max(image.width for image in thumbs)
    height = sum(image.height for image in thumbs)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in thumbs:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a DENTEX object detection baseline.")
    parser.add_argument("--coco-json", type=Path, default=ROOT / "data" / "processed" / "coco" / "dentex" / "enumeration.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "dentex")
    parser.add_argument("--external-coco-json", type=Path, default=ROOT / "data" / "processed" / "coco" / "tufts_teeth.json")
    parser.add_argument("--external-image-root", type=Path, default=ROOT / "data" / "raw" / "tufts")
    parser.add_argument("--external-max-images", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/object_detection_baseline_v1"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-coco", action="store_true")
    parser.add_argument("--class-agnostic", action="store_true", default=False)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_root / "tables"
    figures_dir = args.output_root / "figures"
    checkpoint_dir = args.output_root / "checkpoints"
    for directory in (tables_dir, figures_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    class_names = load_category_names(args.coco_json, args.class_agnostic)
    records = load_coco_records(args.coco_json, args.image_root, args.class_agnostic, args.max_images, args.seed)
    if len(records) < 10:
        raise RuntimeError(f"Too few detection records found: {len(records)}")
    split = max(1, int(len(records) * args.val_fraction))
    val_records = records[:split]
    train_records = records[split:]

    num_classes = 2 if args.class_agnostic else 1 + len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.pretrained_coco:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            min_size=args.image_size,
            max_size=args.image_size,
            box_detections_per_img=120,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=args.image_size,
            max_size=args.image_size,
            box_detections_per_img=120,
        )
    model = model.to(device)

    train_loader = DataLoader(
        CocoDetectionRecords(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
    )
    val_loader = DataLoader(
        CocoDetectionRecords(val_records),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection,
    )
    gt_by_image = {rec["image_id"]: np.asarray(rec["boxes"], dtype=np.float32) for rec in val_records}
    gt_labels_by_image = {rec["image_id"]: np.asarray(rec["labels"], dtype=np.int64) for rec in val_records}

    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = checkpoint_dir / "fasterrcnn_mobilenetv3_dentex_tooth.pt"
    history: list[dict] = []
    start_epoch = 1
    best_map = -1.0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_map = float(checkpoint.get("best_map_50_95", best_map))
        print(f"resuming_detector_from_epoch={start_epoch}", flush=True)

    config = {
        "coco_json": str(args.coco_json),
        "image_root": str(args.image_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "num_classes": num_classes,
        "class_names": class_names,
        "class_agnostic": args.class_agnostic,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "pretrained_coco": args.pretrained_coco,
    }
    (args.output_root / "object_detection_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"OBJECT_DETECTION_CONFIG {json.dumps(config)}", flush=True)

    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        sampler.start()
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(value for value in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu()) * len(images)
        train_loss = epoch_loss / max(1, len(train_records))
        metrics, _pr50, _predictions = evaluate_model(model, val_loader, device, gt_by_image)
        elapsed = time.perf_counter() - start
        gpu_stats = sampler.stop()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        reserved_mb = torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": elapsed,
            "peak_allocated_mb": peak_mb,
            "peak_reserved_mb": reserved_mb,
            **gpu_stats,
            **metrics,
        }
        history.append(row)
        is_best = metrics["map_50_95"] > best_map
        best_map = max(best_map, metrics["map_50_95"])
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_map_50_95": best_map,
                "config": config,
            },
            checkpoint_path,
        )
        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "history": history,
                    "best_map_50_95": best_map,
                    "config": config,
                },
                checkpoint_dir / "fasterrcnn_mobilenetv3_dentex_tooth_best.pt",
            )
        pd.DataFrame(history).to_csv(tables_dir / "object_detection_epoch_history.csv", index=False)
        print(
            f"detector epoch {epoch:02d}/{args.epochs}: loss={train_loss:.4f} "
            f"mAP50={metrics['map_50']:.4f} mAP75={metrics['map_75']:.4f} "
            f"mAP50:95={metrics['map_50_95']:.4f} "
            f"power_mean_w={gpu_stats.get('power_draw_w_mean', math.nan):.2f} "
            f"power_rms_w={gpu_stats.get('power_draw_w_rms', math.nan):.2f} "
            f"gpu_util_mean={gpu_stats.get('gpu_util_percent_mean', math.nan):.1f}% "
            f"mem_max_mb={gpu_stats.get('memory_used_mb_max', math.nan):.0f} seconds={elapsed:.1f}",
            flush=True,
        )

    metrics, pr50, predictions = evaluate_model(model, val_loader, device, gt_by_image)
    metrics.update(
        {
            "train_images": len(train_records),
            "checkpoint": str(checkpoint_path),
            "best_checkpoint": str(checkpoint_dir / "fasterrcnn_mobilenetv3_dentex_tooth_best.pt"),
            "note": "Internal DENTEX enumeration split; released DENTEX validation images have no detection annotations.",
        }
    )
    (args.output_root / "object_detection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(tables_dir / "object_detection_metrics.csv", index=False)
    pr50.to_csv(tables_dir / "object_detection_pr_curve_iou50.csv", index=False)
    if not args.class_agnostic:
        confusion, per_class = compute_detection_confusion(
            predictions,
            gt_by_image,
            gt_labels_by_image,
            class_names,
            iou_threshold=0.50,
            score_threshold=args.score_threshold,
        )
        confusion.to_csv(tables_dir / "object_detection_multiclass_confusion_iou50.csv")
        per_class.to_csv(tables_dir / "object_detection_multiclass_per_class_iou50.csv", index=False)
    plot_pr_curve(pr50, figures_dir / "object_detection_pr_curve_iou50.png", metrics["map_50"])
    plot_history(history, figures_dir / "object_detection_training_curves.png")
    plot_examples(val_records, predictions, figures_dir / "object_detection_examples.png", args.preview_count, args.score_threshold)
    if args.external_coco_json.exists():
        external_records = load_coco_records(
            args.external_coco_json,
            args.external_image_root,
            True,
            args.external_max_images,
            args.seed,
        )
        if external_records:
            external_loader = DataLoader(
                CocoDetectionRecords(external_records),
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_detection,
            )
            external_gt = {rec["image_id"]: np.asarray(rec["boxes"], dtype=np.float32) for rec in external_records}
            external_metrics, external_pr50, _external_predictions = evaluate_model(model, external_loader, device, external_gt)
            external_metrics.update(
                {
                    "source": "TUFTS external",
                    "external_images": len(external_records),
                    "note": "Class-agnostic tooth detection evaluated on TUFTS COCO boxes derived from masks.",
                }
            )
            pd.DataFrame([external_metrics]).to_csv(tables_dir / "object_detection_tufts_external_metrics.csv", index=False)
            external_pr50.to_csv(tables_dir / "object_detection_tufts_external_pr_curve_iou50.csv", index=False)
            plot_pr_curve(
                external_pr50,
                figures_dir / "object_detection_tufts_external_pr_curve_iou50.png",
                external_metrics["map_50"],
            )
    print(f"OBJECT_DETECTION_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
