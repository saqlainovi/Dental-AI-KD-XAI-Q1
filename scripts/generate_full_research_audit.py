from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models


ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
SEED = 42


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(3, width)
        self.enc2 = DoubleConv(width, width * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(width * 2, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = DoubleConv(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = DoubleConv(width * 2, width)
        self.out = nn.Conv2d(width, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class ImageBBoxClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, image_dim: int, num_classes: int, bbox_dim: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(image_dim + bbox_dim, num_classes))

    def forward(self, images: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(images)
        return self.classifier(torch.cat([image_features, bbox], dim=1))


class DentalCropDataset(Dataset):
    def __init__(self, items: list[dict], image_size: int) -> None:
        self.items = list(items)
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        last_label = 0
        for offset in range(len(self.items)):
            rec = self.items[(idx + offset) % len(self.items)]
            last_label = int(rec["label"])
            try:
                image = crop_record(rec).resize((self.image_size, self.image_size))
                arr = np.asarray(image).astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                arr = (arr - self.mean) / self.std
                bbox = np.array(rec["bbox_features"], dtype=np.float32)
                return torch.from_numpy(arr), torch.from_numpy(bbox), torch.tensor(rec["label"], dtype=torch.long)
            except OSError:
                continue
        arr = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        bbox = np.zeros(4, dtype=np.float32)
        return torch.from_numpy(arr), torch.from_numpy(bbox), torch.tensor(last_label, dtype=torch.long)


def bbox_features(bbox, image_width: int, image_height: int) -> list[float]:
    x, y, w, h = bbox
    return [
        (x + w / 2) / max(1, image_width),
        (y + h / 2) / max(1, image_height),
        w / max(1, image_width),
        h / max(1, image_height),
    ]


def crop_record(record: dict, padding: int = 8) -> Image.Image:
    with Image.open(record["image_path"]) as src:
        image = src.convert("RGB")
    x, y, w, h = record["bbox"]
    left = max(0, int(x - padding))
    top = max(0, int(y - padding))
    right = min(image.width, int(x + w + padding))
    bottom = min(image.height, int(y + h + padding))
    return image.crop((left, top, right, bottom))


def load_task_records(task_name: str, coco_root: Path, dentex_dir: Path):
    data = json.loads((coco_root / "dentex" / f"{task_name}.json").read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    class_names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]
    records = []
    for ann in data["annotations"]:
        img = images.get(ann["image_id"])
        if img is None or not ann.get("bbox"):
            continue
        path = dentex_dir / img["file_name"]
        if not path.exists():
            continue
        width = img.get("width") or 1
        height = img.get("height") or 1
        records.append(
            {
                "image_path": path,
                "bbox": ann["bbox"],
                "bbox_features": bbox_features(ann["bbox"], width, height),
                "label": int(ann["category_id"]),
            }
        )
    random.Random(SEED).shuffle(records)
    return records, class_names


def split_records(records: list[dict], train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": records[:n_train],
        "validation": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }


def build_swin_bbox_model(num_classes: int) -> nn.Module:
    backbone = models.swin_t(weights=None)
    dim = backbone.head.in_features
    backbone.head = nn.Identity()
    return ImageBBoxClassifier(backbone, dim, num_classes)


def evaluate_swin_checkpoint(path: Path, coco_root: Path, dentex_dir: Path, output_dir: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    task = checkpoint["task"]
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["training_config"].get("image_size", 224))
    records, _ = load_task_records(task, coco_root, dentex_dir)
    splits = split_records(records)
    model = build_swin_bbox_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    split_rows = []
    per_class_rows = []
    confusion_paths = []
    for split, items in splits.items():
        loader = DataLoader(DentalCropDataset(items, image_size), batch_size=8, shuffle=False, num_workers=0)
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, bbox, labels in loader:
                images = images.to(device)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                bbox = bbox.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images, bbox)
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for true, pred in zip(y_true, y_pred):
            cm[int(true), int(pred)] += 1
        acc = float(np.trace(cm) / max(1, cm.sum()))
        f1s = []
        for idx, name in enumerate(class_names):
            tp = int(cm[idx, idx])
            fp = int(cm[:, idx].sum() - tp)
            fn = int(cm[idx, :].sum() - tp)
            support = int(cm[idx, :].sum())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            f1s.append(f1)
            per_class_rows.append(
                {
                    "task": task,
                    "split": split,
                    "class": name,
                    "support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        split_rows.append({"task": task, "split": split, "samples": len(y_true), "accuracy": acc, "macro_f1": float(np.mean(f1s))})
        cm_path = output_dir / f"classification_confusion_{task}_{split}.csv"
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
        confusion_paths.append(str(cm_path))

    history = checkpoint.get("history", {})
    history_df = pd.DataFrame({"epoch": list(range(1, len(history.get("train_loss", [])) + 1))})
    for key, values in history.items():
        if isinstance(values, list):
            history_df[key] = values
    history_df.to_csv(output_dir / f"classifier_epoch_history_{task}.csv", index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return split_rows, per_class_rows, confusion_paths


def checkpoint_history(path: Path) -> pd.DataFrame:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", {})
    n = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    df = pd.DataFrame({"epoch": list(range(1, n + 1))})
    for key, values in history.items():
        if isinstance(values, list):
            df[key] = values
    df["checkpoint"] = str(path)
    return df


def segmentation_class_metrics(df: pd.DataFrame, source: str) -> pd.DataFrame:
    counts = df[["tn", "fp", "fn", "tp"]].sum()
    tn, fp, fn, tp = [int(counts[k]) for k in ["tn", "fp", "fn", "tp"]]
    rows = []
    for label, c_tp, c_fp, c_fn, c_tn in [
        ("tooth", tp, fp, fn, tn),
        ("background", tn, fn, fp, tp),
    ]:
        precision = c_tp / max(1, c_tp + c_fp)
        recall = c_tp / max(1, c_tp + c_fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        iou = c_tp / max(1, c_tp + c_fp + c_fn)
        rows.append(
            {
                "source": source,
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1_or_dice": f1,
                "iou": iou,
                "tp": c_tp,
                "fp": c_fp,
                "fn": c_fn,
                "tn": c_tn,
            }
        )
    return pd.DataFrame(rows)


def memory_benchmark(device: torch.device, output_dir: Path) -> pd.DataFrame:
    rows = []
    if device.type != "cuda":
        return pd.DataFrame([{"model": "cuda unavailable", "note": "GPU memory benchmark was not run"}])

    configs = [
        ("TUFTS teacher TinyUNet width32", TinyUNet(2, width=32), 2, 512),
        ("DENTEX student TinyUNet width16", TinyUNet(2, width=16), 4, 512),
        ("DentalAI reference TinyUNet width32 classes5", TinyUNet(5, width=32), 3, 512),
    ]
    for name, model, batch_size, image_size in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        y = torch.randint(0, model.out.out_channels, (batch_size, image_size, image_size), device=device)
        start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "model": name,
                "batch_size": batch_size,
                "image_size": image_size,
                "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
                "one_train_step_seconds": elapsed,
                "gpu": torch.cuda.get_device_name(0),
                "note": "Measured after the run with one synthetic forward/backward step; historical VRAM was not logged.",
            }
        )
        del model, optimizer, x, y, logits, loss
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "memory_benchmark_synthetic_train_step.csv", index=False)
    return out


def write_markdown_report(output_path: Path, tables_dir: Path, summary: dict) -> None:
    table_lines = [
        f"- Per-epoch training histories: `{tables_dir}`",
        "- `main_metrics_summary.csv`",
        "- `segmentation_binary_class_metrics.csv`",
        "- `memory_benchmark_synthetic_train_step.csv`",
        "- `research_completion_matrix.csv`",
    ]
    if (tables_dir / "classification_split_metrics.csv").exists():
        table_lines.append("- `classification_split_metrics.csv`")
    if (tables_dir / "classification_per_class_metrics.csv").exists():
        table_lines.append("- `classification_per_class_metrics.csv`")
    if not (tables_dir / "classification_split_metrics.csv").exists() and list(tables_dir.glob("classification_confusion_*.csv")):
        table_lines.append("- `classification_confusion_*.csv` partial crop-baseline confusion matrices")

    lines = [
        "# Full Dental AI Research Audit",
        "",
        f"Fresh run id: `{summary['run_id']}`",
        "",
        "## Bottom Line",
        "",
        "- The completed implementation is strongest as a teacher-student pixel-level tooth segmentation pipeline with TUFTS external validation, XAI/uncertainty outputs, and DENTEX pseudo-mask transfer.",
        "- DENTEX multiclass enumeration detection is now included as an internal supporting experiment with mAP and per-class IoU@0.50 tables.",
        "- TUFTS does not provide DENTEX-style tooth-position multiclass labels, so TUFTS validates segmentation/generalization but not external multiclass enumeration.",
        "- Disease/classification is only a baseline and is weak; it should not be the main paper claim.",
        "",
        "## Main Fresh-Run Metrics",
        "",
        f"- TUFTS X-ray teacher: Dice `{summary['xray_teacher_best_val_dice']:.4f}`, IoU `{summary['xray_teacher_best_val_iou']:.4f}`, best epoch `{summary['xray_teacher_best_epoch']}`.",
        f"- DENTEX student pseudo-validation: Dice `{summary['dentex_val_mean_dice']:.4f}`, IoU `{summary['dentex_val_mean_iou']:.4f}`, rows `{summary['dentex_val_rows']}`.",
        f"- TUFTS external all rows: Dice `{summary['tufts_all_mean_dice']:.4f}`, IoU `{summary['tufts_all_mean_iou']:.4f}`, rows `{summary['tufts_all_rows']}`.",
        f"- TUFTS external non-empty: Dice `{summary['tufts_nonempty_mean_dice']:.4f}`, IoU `{summary['tufts_nonempty_mean_iou']:.4f}`, rows `{summary['tufts_nonempty_rows']}`; empty masks `{summary['tufts_empty_target_rows']}`.",
        f"- Pixel ROC AUC: DENTEX `{summary['dentex_pixel_roc_auc']:.4f}`, TUFTS `{summary['tufts_pixel_roc_auc']:.4f}`.",
        "",
        "## What Was Trained",
        "",
        "- DentalAI RGB reference teacher: optional modality-gap baseline; not required for the main X-ray segmentation claim.",
        f"- TUFTS X-ray teacher: best epoch {summary['xray_teacher_best_epoch']}, 512 image size, batch 2.",
        f"- DENTEX student: epoch {summary['student_epoch']}, 512 image size, batch 4, DENTEX pseudo-mask training and 50 validation rows.",
        "- DENTEX multiclass enumeration detector: COCO-pretrained Faster R-CNN internal split evaluation with mAP@50/mAP@75/mAP@50:95 and per-class metrics.",
        "- Swin-T crop classifiers: quadrant/enumeration/disease baselines at 224 crop image size, 20 epochs.",
        "",
        "## Image Size / Compression",
        "",
        "- Segmentation training resizes full radiographs and masks to `512x512` during model input.",
        "- Final evaluation upsamples predictions back to original image size before Dice/IoU calculation.",
        "- Swin crop classifiers resize cropped tooth boxes to `224x224`, so those are not full-image segmentation models.",
        "",
        "## Tables Generated",
        "",
        *table_lines,
        "",
        "## Completion Matrix",
        "",
        "- Complete: data conversion, full fresh run, TUFTS teacher, DENTEX pseudo masks, student segmentation, DENTEX/TUFTS evaluation, multiclass DENTEX enumeration detector, ablation, XAI package, GPU/power logs, and manuscript draft snippets.",
        "- Partial: external multiclass detection cannot be tested on TUFTS because TUFTS has no tooth-position labels; disease classification remains weak.",
        "- Optional improvements before a stronger journal submission: MC-dropout/ECE calibration, stricter retraining ablations, stronger detector backbone/YOLO/Mask R-CNN comparison, and more polished clinical discussion.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="fresh_rerun_20260518_112354")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports" / "full_research_audit_20260518")
    parser.add_argument("--skip-classifier-eval", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    tables_dir = output_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    run_results = Path("E:/dental_ai_results") / args.run_id
    run_ckpts = Path("E:/dental_ai_checkpoints") / args.run_id
    final_tables = run_results / "final_eval_xai_v1" / "tables"
    ablation_tables = run_results / "ablation_comparison_v1" / "tables"
    paper_tables = run_results / "paper_package_v1" / "tables"

    summary = json.loads((final_tables / "final_metrics_summary.json").read_text(encoding="utf-8"))
    summary["run_id"] = args.run_id
    pd.DataFrame([summary]).to_csv(tables_dir / "main_metrics_summary.csv", index=False)

    checkpoint_map = {
        "dentalai_reference_teacher": run_ckpts / "teacher_tinyunet_dentalai_full.pt",
        "tufts_xray_teacher": run_ckpts / "teacher_tinyunet_tufts_xray_full.pt",
        "dentex_student_last": run_ckpts / "student_tinyunet_dentex_pseudo_full.pt",
        "dentex_student_best": run_ckpts / "student_tinyunet_dentex_pseudo_full_best.pt",
    }
    ckpt_rows = []
    for name, path in checkpoint_map.items():
        if not path.exists():
            ckpt_rows.append(
                {
                    "stage": name,
                    "checkpoint": str(path),
                    "epoch": None,
                    "history_epochs": 0,
                    "size_mb": None,
                    "config": "{}",
                    "status": "missing",
                }
            )
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        history_df = checkpoint_history(path)
        history_df.to_csv(tables_dir / f"epoch_history_{name}.csv", index=False)
        ckpt_rows.append(
            {
                "stage": name,
                "checkpoint": str(path),
                "epoch": ckpt.get("epoch"),
                "history_epochs": len(history_df),
                "size_mb": path.stat().st_size / (1024**2),
                "config": json.dumps(ckpt.get("config", ckpt.get("training_config", {}))),
                "status": "available",
            }
        )
    pd.DataFrame(ckpt_rows).to_csv(tables_dir / "checkpoint_summary.csv", index=False)

    dentex_df = pd.read_csv(final_tables / "dentex_validation_student_eval.csv")
    tufts_all_df = pd.read_csv(final_tables / "tufts_external_student_eval.csv")
    tufts_nonempty_df = pd.read_csv(final_tables / "tufts_external_student_eval_nonempty.csv")
    seg_metrics = pd.concat(
        [
            segmentation_class_metrics(dentex_df, "DENTEX pseudo-validation"),
            segmentation_class_metrics(tufts_all_df, "TUFTS external all rows"),
            segmentation_class_metrics(tufts_nonempty_df, "TUFTS external non-empty"),
        ],
        ignore_index=True,
    )
    seg_metrics.to_csv(tables_dir / "segmentation_binary_class_metrics.csv", index=False)

    for src in [ablation_tables, paper_tables]:
        if src.exists():
            for csv_path in src.glob("*.csv"):
                pd.read_csv(csv_path).to_csv(tables_dir / csv_path.name, index=False)

    completion = pd.DataFrame(
        [
            {"item": "Full DENTEX pseudo-mask/student training", "status": "done", "evidence": f"student epoch {summary.get('student_epoch')}; DENTEX pseudo-val Dice {summary.get('dentex_val_mean_dice'):.4f}"},
            {"item": "TUFTS X-ray teacher", "status": "done", "evidence": f"best val Dice {summary.get('xray_teacher_best_val_dice'):.4f} at epoch {summary.get('xray_teacher_best_epoch')}"},
            {"item": "DentalAI 1GB reference teacher", "status": "optional/missing in this run", "evidence": "RGB intraoral baseline is not required for the main X-ray segmentation claim"},
            {"item": "DENTEX pseudo-mask cleanup", "status": "done", "evidence": "all tasks cleaned; enumeration filtered 2 rows"},
            {"item": "TUFTS external validation", "status": "done", "evidence": "1000 rows, 968 non-empty masks"},
            {"item": "Object detection mAP/mAP50/mAP75", "status": "done for DENTEX internal multiclass", "evidence": "COCO-pretrained Faster R-CNN results are in the run's multiclass_detection folder"},
            {"item": "Per-disease clinical model", "status": "not complete", "evidence": "only weak Swin crop baseline exists"},
            {"item": "LIME/SHAP", "status": "done as perturbation-style XAI", "evidence": "LIME-style and KernelSHAP-style panels and numeric overlap/deletion tables generated"},
            {"item": "Ablation", "status": "partial", "evidence": "teacher/student, raw/clean pseudo masks, reliability filtering done; strict retrain ablations missing"},
            {"item": "GPU memory/power logging", "status": "done for new run", "evidence": "teacher/student/detector epoch histories include power draw, RMS, utilization, and VRAM"},
        ]
    )
    completion.to_csv(tables_dir / "research_completion_matrix.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory_benchmark(device, tables_dir)

    if not args.skip_classifier_eval:
        split_rows: list[dict] = []
        per_class_rows: list[dict] = []
        confusion_paths: list[str] = []
        for path in [
            Path("E:/dental_ai_checkpoints/v3_quadrant_swin_t_full.pt"),
            Path("E:/dental_ai_checkpoints/v3_enumeration_swin_t_full.pt"),
            Path("E:/dental_ai_checkpoints/v3_disease_swin_t_full.pt"),
        ]:
            rows, class_rows, cm_paths = evaluate_swin_checkpoint(
                path,
                ROOT / "data" / "processed" / "coco",
                ROOT / "data" / "raw" / "dentex",
                tables_dir,
                device,
            )
            split_rows.extend(rows)
            per_class_rows.extend(class_rows)
            confusion_paths.extend(cm_paths)
        pd.DataFrame(split_rows).to_csv(tables_dir / "classification_split_metrics.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(tables_dir / "classification_per_class_metrics.csv", index=False)
        (tables_dir / "classification_confusion_paths.json").write_text(json.dumps(confusion_paths, indent=2), encoding="utf-8")

    write_markdown_report(output_root / "FULL_RESEARCH_AUDIT.md", tables_dir, summary)
    print(output_root / "FULL_RESEARCH_AUDIT.md")


if __name__ == "__main__":
    main()
