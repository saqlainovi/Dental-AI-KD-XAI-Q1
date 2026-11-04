from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.augmentations import prepare_segmentation_pair
from dental_ai.training.swin_upernet import build_segmentation_model, segmentation_logits


class GPUEpochSampler:
    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self.records: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = shutil.which("nvidia-smi") is not None

    def _sample_once(self) -> None:
        if not self.enabled:
            return
        cmd = [
            "nvidia-smi",
            "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
            power, util, memory, temp = [float(part.strip()) for part in output.split(",")[:4]]
            self.records.append(
                {
                    "power_draw_w": power,
                    "gpu_util_percent": util,
                    "memory_used_mb": memory,
                    "temperature_c": temp,
                }
            )
        except Exception:
            self.enabled = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self.records = []
        self._stop.clear()
        self._sample_once()
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> dict[str, float]:
        self._sample_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.records:
            return {
                "gpu_samples": 0,
                "power_draw_w_mean": math.nan,
                "power_draw_w_max": math.nan,
                "power_draw_w_rms": math.nan,
                "gpu_util_percent_mean": math.nan,
                "gpu_util_percent_max": math.nan,
                "memory_used_mb_mean": math.nan,
                "memory_used_mb_max": math.nan,
                "temperature_c_mean": math.nan,
                "temperature_c_max": math.nan,
            }
        values = {key: np.asarray([row[key] for row in self.records], dtype=np.float32) for key in self.records[0]}
        return {
            "gpu_samples": len(self.records),
            "power_draw_w_mean": float(values["power_draw_w"].mean()),
            "power_draw_w_max": float(values["power_draw_w"].max()),
            "power_draw_w_rms": float(np.sqrt(np.mean(values["power_draw_w"] ** 2))),
            "gpu_util_percent_mean": float(values["gpu_util_percent"].mean()),
            "gpu_util_percent_max": float(values["gpu_util_percent"].max()),
            "memory_used_mb_mean": float(values["memory_used_mb"].mean()),
            "memory_used_mb_max": float(values["memory_used_mb"].max()),
            "temperature_c_mean": float(values["temperature_c"].mean()),
            "temperature_c_max": float(values["temperature_c"].max()),
        }


def _norm_key(value: str | Path) -> str:
    return str(value).replace("\\", "/").lower()


def load_disease_multilabels(coco_path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    if not coco_path.exists():
        return {}, []
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda row: int(row.get("id", 0)))
    class_names = [str(row.get("name", f"disease_class_{idx}")) for idx, row in enumerate(categories)]
    dentex_disease_names = ["caries", "deep_caries", "periapical_lesion", "impacted"]
    if len(class_names) == 4 and all(name.startswith("disease_class_") for name in class_names):
        class_names = dentex_disease_names
    if not class_names:
        return {}, []
    image_by_id = {int(row["id"]): row for row in data.get("images", []) if "id" in row}
    labels: dict[str, np.ndarray] = {}
    for image in image_by_id.values():
        file_name = _norm_key(image.get("file_name", ""))
        if file_name:
            labels[file_name] = np.zeros(len(class_names), dtype=np.float32)
    for ann in data.get("annotations", []):
        image = image_by_id.get(int(ann.get("image_id", -1)))
        if image is None:
            continue
        file_name = _norm_key(image.get("file_name", ""))
        category_id = int(ann.get("category_id", -1))
        if file_name in labels and 0 <= category_id < len(class_names):
            labels[file_name][category_id] = 1.0
    basename_labels = {Path(key).name: value for key, value in labels.items()}
    labels.update(basename_labels)
    return labels, class_names


class DentexPseudoMaskDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        image_size: int,
        augment: bool,
        classification_labels: dict[str, np.ndarray] | None = None,
        classification_classes: list[str] | None = None,
        classification_label_policy: str = "disease_task_only",
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.classification_labels = classification_labels or {}
        self.classification_classes = classification_classes or []
        self.classification_label_policy = classification_label_policy

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["clean_mask"]).convert("L")
        image, mask = prepare_segmentation_pair(image, mask, self.image_size, augment=self.augment, use_clahe=True)
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))
        mask_arr = (np.asarray(mask) > 0).astype(np.int64)
        image_tensor = torch.from_numpy(image_arr)
        mask_tensor = torch.from_numpy(mask_arr)
        if not self.classification_classes:
            return image_tensor, mask_tensor
        label = self._classification_label(row)
        valid = np.asarray([1.0 if label is not None else 0.0], dtype=np.float32)
        if label is None:
            label = np.zeros(len(self.classification_classes), dtype=np.float32)
        return image_tensor, mask_tensor, torch.from_numpy(label.astype(np.float32)), torch.from_numpy(valid)

    def _classification_label(self, row: dict) -> np.ndarray | None:
        if self.classification_label_policy == "disease_task_only" and "disease" not in str(row.get("task", "")).lower():
            return None
        return classification_label_for_row(row, self.classification_labels)


class TuftsMaskDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, Path]], image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = prepare_segmentation_pair(image, mask, self.image_size, augment=False, use_clahe=True)
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        image_arr = np.transpose(image_arr, (2, 0, 1))
        mask_arr = (np.asarray(mask) > 0).astype(np.int64)
        return torch.from_numpy(image_arr), torch.from_numpy(mask_arr)


def build_tufts_rows(raw_root: Path, max_rows: int = 0) -> list[tuple[Path, Path]]:
    image_dir = raw_root / "Radiographs" / "Radiographs"
    mask_dir = raw_root / "Segmentation" / "Segmentation" / "teeth_mask"
    mask_by_stem = {path.stem.lower(): path for path in mask_dir.glob("*") if path.is_file()}
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_by_stem.get(image_path.stem.lower())
        if mask_path:
            rows.append((image_path, mask_path))
    if max_rows:
        rows = rows[:max_rows]
    return rows


def load_rows(clean_root: Path, tasks: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for task in tasks:
        summary_path = clean_root / task / "summary.json"
        if not summary_path.exists():
            print(f"missing_summary={summary_path}", flush=True)
            continue
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            if row.get("status") != "kept":
                continue
            image_path = Path(row["image_path"])
            mask_path = Path(row["clean_mask"])
            if not image_path.is_absolute():
                image_path = (ROOT / image_path).resolve()
            if not mask_path.is_absolute():
                mask_path = (ROOT / mask_path).resolve()
            key = str(image_path.resolve()).lower()
            if image_path.exists() and mask_path.exists() and key not in seen:
                item = dict(row)
                item["image_path"] = str(image_path)
                item["clean_mask"] = str(mask_path)
                item["task"] = task
                rows.append(item)
                seen.add(key)
    return rows


def split_rows_percentage(
    rows: list[dict],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    total_fraction = train_fraction + val_fraction + test_fraction
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction:.4f}")
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_total = len(rows)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_train = max(1, n_train) if n_total else 0
    n_val = max(1, n_val) if n_total - n_train > 1 else max(0, n_total - n_train)
    n_test = n_total - n_train - n_val
    if n_test <= 0 and n_total >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]
    return train_rows, val_rows, test_rows


def write_split_manifest(path: Path, splits: dict[str, list[dict]]) -> None:
    rows = []
    for split, split_rows in splits.items():
        for row in split_rows:
            rows.append(
                {
                    "split": split,
                    "task": row.get("task", ""),
                    "file_name": row.get("file_name", Path(str(row.get("image_path", ""))).name),
                    "image_path": row.get("image_path", ""),
                    "mask_path": row.get("clean_mask", ""),
                }
            )
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classification_label_for_row(row: dict, classification_labels: dict[str, np.ndarray]) -> np.ndarray | None:
    keys = [
        row.get("file_name", ""),
        Path(str(row.get("file_name", ""))).name,
        Path(str(row.get("image_path", ""))).name,
    ]
    for key in keys:
        norm = _norm_key(key)
        if norm in classification_labels:
            return classification_labels[norm].copy()
    return None


def build_classification_pos_weight(
    rows: list[dict],
    classification_labels: dict[str, np.ndarray],
    num_classes: int,
    cap: float,
    classification_label_policy: str,
) -> torch.Tensor | None:
    labels = [
        label
        for row in rows
        if classification_label_policy == "any_matching_image" or "disease" in str(row.get("task", "")).lower()
        for label in [classification_label_for_row(row, classification_labels)]
        if label is not None
    ]
    if not labels or num_classes <= 0:
        return None
    arr = np.stack(labels).astype(np.float32)
    positives = arr.sum(axis=0)
    negatives = arr.shape[0] - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, cap)
    return torch.tensor(pos_weight, dtype=torch.float32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_f = target.float()
    intersection = (probs * target_f).sum()
    denom = probs.sum() + target_f.sum()
    return 1 - (2 * intersection + eps) / (denom + eps)


def segmentation_scores(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = logits.argmax(dim=1).bool()
    target_b = target.bool()
    intersection = (pred & target_b).sum().float()
    union = (pred | target_b).sum().float()
    denom = pred.sum().float() + target_b.sum().float()
    dice = (2 * intersection + eps) / (denom + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice.detach().cpu()), float(iou.detach().cpu())


def multilabel_macro_f1(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    if pred.size == 0:
        return math.nan
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tp = np.logical_and(pred_b, target_b).sum(axis=0).astype(np.float32)
    fp = np.logical_and(pred_b, ~target_b).sum(axis=0).astype(np.float32)
    fn = np.logical_and(~pred_b, target_b).sum(axis=0).astype(np.float32)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    classification_loss_weight: float = 0.0,
    classification_pos_weight: torch.Tensor | None = None,
    max_grad_norm: float = 0.0,
    background_class_weight: float = 0.35,
    foreground_class_weight: float = 1.0,
    ce_loss_weight: float = 1.0,
    dice_loss_weight: float = 1.0,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    dices = []
    ious = []
    accuracies = []
    cls_loss_total = 0.0
    cls_valid_rows = 0
    cls_preds = []
    cls_targets_all = []
    class_weights = torch.tensor(
        [background_class_weight, foreground_class_weight],
        dtype=torch.float32,
        device=device,
    )

    for batch in loader:
        if len(batch) == 4:
            images, masks, cls_targets, cls_valid = batch
            cls_targets = cls_targets.to(device, non_blocking=True)
            cls_valid = cls_valid.to(device, non_blocking=True).view(-1) > 0.5
        else:
            images, masks = batch
            cls_targets = None
            cls_valid = None
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                output = model(images)
                logits = segmentation_logits(output)
                seg_loss = ce_loss_weight * F.cross_entropy(logits, masks, weight=class_weights)
                seg_loss = seg_loss + dice_loss_weight * dice_loss(logits, masks)
                cls_loss = torch.zeros((), device=device)
                if isinstance(output, dict) and cls_targets is not None and cls_valid is not None and cls_valid.any():
                    cls_logits = output.get("classification")
                    if cls_logits is not None:
                        cls_loss = F.binary_cross_entropy_with_logits(
                            cls_logits[cls_valid],
                            cls_targets[cls_valid],
                            pos_weight=classification_pos_weight,
                        )
                loss = seg_loss + classification_loss_weight * cls_loss
            if train:
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        d, i = segmentation_scores(logits.detach(), masks)
        dices.append(d)
        ious.append(i)
        accuracies.append(float((logits.detach().argmax(dim=1) == masks).float().mean().detach().cpu()))
        if isinstance(output, dict) and cls_targets is not None and cls_valid is not None and cls_valid.any():
            cls_logits = output.get("classification")
            if cls_logits is not None:
                valid_count = int(cls_valid.sum().detach().cpu())
                cls_loss_total += float(cls_loss.detach().cpu()) * valid_count
                cls_valid_rows += valid_count
                cls_pred = (torch.sigmoid(cls_logits[cls_valid].detach()) >= 0.5).cpu().numpy().astype(np.uint8)
                cls_target = cls_targets[cls_valid].detach().cpu().numpy().astype(np.uint8)
                cls_preds.append(cls_pred)
                cls_targets_all.append(cls_target)

    if cls_preds:
        pred_arr = np.concatenate(cls_preds, axis=0)
        target_arr = np.concatenate(cls_targets_all, axis=0)
        cls_exact = float((pred_arr == target_arr).all(axis=1).mean())
        cls_f1 = multilabel_macro_f1(pred_arr, target_arr)
    else:
        cls_exact = math.nan
        cls_f1 = math.nan

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "classification_loss": cls_loss_total / max(1, cls_valid_rows),
        "classification_exact_match": cls_exact,
        "classification_macro_f1": cls_f1,
        "classification_valid_rows": float(cls_valid_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight DENTEX student segmenter from cleaned pseudo masks.")
    parser.add_argument("--clean-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_xray_teacher_clean"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("E:/dental_ai_checkpoints"))
    parser.add_argument("--train-tasks", nargs="+", default=["quadrant_full", "enumeration_full", "disease_full"])
    parser.add_argument("--val-tasks", nargs="+", default=["validation_full"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--model-family", choices=["tinyunet", "swin_t_upernet"], default="swin_t_upernet")
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--classification-task", choices=["none", "disease"], default="disease")
    parser.add_argument("--classification-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--classification-label-policy",
        choices=["disease_task_only", "any_matching_image"],
        default="disease_task_only",
        help="Use disease labels only on disease-task rows or on any row whose image name matches the disease COCO.",
    )
    parser.add_argument("--classification-pos-weight", choices=["none", "auto"], default="none")
    parser.add_argument("--classification-pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--disease-coco", type=Path, default=ROOT / "data" / "processed" / "coco" / "dentex" / "disease.json")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--background-class-weight", type=float, default=0.35)
    parser.add_argument("--foreground-class-weight", type=float, default=1.0)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--best-metric",
        choices=["val_dice", "val_composite", "val_dice_test_dice", "val_dice_tufts_dice"],
        default="val_dice",
    )
    parser.add_argument("--best-classification-weight", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["official_tasks", "percentage"], default="official_tasks")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-internal-test", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default="student_swin_t_upernet_dentex_pseudo_multitask_full")
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    parser.add_argument("--tufts-test-root", type=Path, default=ROOT / "data" / "raw" / "tufts")
    parser.add_argument("--test-every-epoch", action="store_true")
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dentex_test_rows: list[dict] = []
    if args.split_mode == "percentage":
        source_rows = load_rows(args.clean_root, args.train_tasks)
        train_rows, val_rows, dentex_test_rows = split_rows_percentage(
            source_rows,
            args.train_fraction,
            args.val_fraction,
            args.test_fraction,
            args.split_seed,
        )
    else:
        train_rows = load_rows(args.clean_root, args.train_tasks)
        val_rows = load_rows(args.clean_root, args.val_tasks)
        random.Random(42).shuffle(train_rows)
        random.Random(123).shuffle(val_rows)
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    if args.max_val:
        val_rows = val_rows[: args.max_val]
    if args.max_internal_test:
        dentex_test_rows = dentex_test_rows[: args.max_internal_test]
    if not train_rows:
        raise RuntimeError(f"No training rows found under {args.clean_root}")
    if not val_rows:
        split = max(1, int(len(train_rows) * 0.15))
        val_rows = train_rows[:split]
        train_rows = train_rows[split:]
    if args.split_mode == "percentage" and not dentex_test_rows:
        raise RuntimeError("Percentage split requested, but no held-out DENTEX test rows were created")
    tufts_test_rows = build_tufts_rows(args.tufts_test_root, args.max_test) if args.test_every_epoch else []
    disease_labels, disease_classes = ({}, [])
    if args.classification_task == "disease":
        disease_labels, disease_classes = load_disease_multilabels(args.disease_coco)
        if not disease_classes:
            raise RuntimeError(f"Disease classification requested, but no disease classes were loaded from {args.disease_coco}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    classification_pos_weight = None
    if args.classification_task == "disease" and args.classification_pos_weight == "auto":
        classification_pos_weight = build_classification_pos_weight(
            train_rows,
            disease_labels,
            len(disease_classes),
            args.classification_pos_weight_cap,
            args.classification_label_policy,
        )
        if classification_pos_weight is not None:
            classification_pos_weight = classification_pos_weight.to(device)
            print(
                f"classification_pos_weight={classification_pos_weight.detach().cpu().numpy().round(3).tolist()}",
                flush=True,
            )

    loaders = {
        "train": DataLoader(
            DentexPseudoMaskDataset(
                train_rows,
                args.image_size,
                augment=True,
                classification_labels=disease_labels,
                classification_classes=disease_classes,
                classification_label_policy=args.classification_label_policy,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "val": DataLoader(
            DentexPseudoMaskDataset(
                val_rows,
                args.image_size,
                augment=False,
                classification_labels=disease_labels,
                classification_classes=disease_classes,
                classification_label_policy=args.classification_label_policy,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }
    dentex_test_loader = (
        DataLoader(
            DentexPseudoMaskDataset(
                dentex_test_rows,
                args.image_size,
                augment=False,
                classification_labels=disease_labels,
                classification_classes=disease_classes,
                classification_label_policy=args.classification_label_policy,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if dentex_test_rows
        else None
    )

    model = build_segmentation_model(
        model_family=args.model_family,
        num_classes=2,
        width=args.width,
        decoder_channels=args.decoder_channels,
        num_classification_labels=len(disease_classes),
        gradient_checkpointing=args.gradient_checkpointing,
        pretrained_imagenet=args.pretrained_imagenet,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_root / f"{args.run_name}.pt"
    best_path = args.checkpoint_root / f"{args.run_name}_best.pt"
    metric_best_paths = {
        "val_dice": args.checkpoint_root / f"{args.run_name}_best_val_dice.pt",
        "val_composite": args.checkpoint_root / f"{args.run_name}_best_composite.pt",
        "test_dice": args.checkpoint_root / f"{args.run_name}_best_test_dice.pt",
        "tufts_external_dice": args.checkpoint_root / f"{args.run_name}_best_tufts_external_dice.pt",
    }
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "train_accuracy": [],
        "train_classification_loss": [],
        "train_classification_exact_match": [],
        "train_classification_macro_f1": [],
        "train_classification_valid_rows": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_accuracy": [],
        "val_classification_loss": [],
        "val_classification_exact_match": [],
        "val_classification_macro_f1": [],
        "val_classification_valid_rows": [],
        "test_loss": [],
        "test_dice": [],
        "test_iou": [],
        "test_accuracy": [],
        "test_classification_loss": [],
        "test_classification_exact_match": [],
        "test_classification_macro_f1": [],
        "test_classification_valid_rows": [],
        "tufts_external_loss": [],
        "tufts_external_dice": [],
        "tufts_external_iou": [],
        "tufts_external_accuracy": [],
        "seconds": [],
        "peak_allocated_mb": [],
        "peak_reserved_mb": [],
        "power_draw_w_mean": [],
        "power_draw_w_max": [],
        "power_draw_w_rms": [],
        "gpu_util_percent_mean": [],
        "gpu_util_percent_max": [],
        "memory_used_mb_mean": [],
        "memory_used_mb_max": [],
        "temperature_c_mean": [],
        "temperature_c_max": [],
        "gpu_samples": [],
        "selection_score": [],
        "val_composite_score": [],
        "val_dice_test_dice_score": [],
        "val_dice_tufts_dice_score": [],
        "is_best": [],
        "is_best_val_dice": [],
        "is_best_composite": [],
        "is_best_test_dice": [],
        "is_best_tufts_external_dice": [],
    }
    start_epoch = 1
    best_val_dice = -1.0
    best_score = -1.0
    best_metric_scores = {key: -1.0 for key in metric_best_paths}

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(loaded_history.get(key, history[key]))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", best_val_dice))
        best_score = float(checkpoint.get("best_score", checkpoint.get("best_val_dice", best_score)))
        loaded_metric_scores = checkpoint.get("best_metric_scores", {})
        for key in best_metric_scores:
            best_metric_scores[key] = float(loaded_metric_scores.get(key, best_metric_scores[key]))
        print(f"Resuming from epoch {start_epoch}: {checkpoint_path}", flush=True)

    config = {
        "clean_root": str(args.clean_root),
        "train_tasks": args.train_tasks,
        "val_tasks": args.val_tasks,
        "epochs": args.epochs,
        "model_family": args.model_family,
        "split_mode": args.split_mode,
        "train_fraction": args.train_fraction if args.split_mode == "percentage" else None,
        "val_fraction": args.val_fraction if args.split_mode == "percentage" else None,
        "test_fraction": args.test_fraction if args.split_mode == "percentage" else None,
        "split_seed": args.split_seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "width": args.width,
        "decoder_channels": args.decoder_channels,
        "gradient_checkpointing": args.gradient_checkpointing,
        "pretrained_imagenet": args.pretrained_imagenet,
        "num_segmentation_classes": 2,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "background_class_weight": args.background_class_weight,
        "foreground_class_weight": args.foreground_class_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "best_metric": args.best_metric,
        "best_classification_weight": args.best_classification_weight,
        "classification_task": args.classification_task,
        "classification_loss_weight": args.classification_loss_weight,
        "classification_label_policy": args.classification_label_policy,
        "classification_pos_weight_mode": args.classification_pos_weight,
        "classification_pos_weight_cap": args.classification_pos_weight_cap,
        "classification_pos_weight_values": (
            classification_pos_weight.detach().cpu().numpy().astype(float).round(6).tolist()
            if classification_pos_weight is not None
            else None
        ),
        "classification_classes": disease_classes,
        "classification_rows_loaded": len(disease_labels),
        "disease_coco": str(args.disease_coco),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "internal_test_rows": len(dentex_test_rows),
        "tufts_test_rows_per_epoch": len(tufts_test_rows),
        "distillation_type": "hard pseudo-mask imitation plus DENTEX disease multi-task classification",
        "augmentation": "CLAHE, horizontal flip, rotation +/-15, affine +/-10 and translate 5%, brightness/contrast jitter",
    }
    print(f"device={device} config={json.dumps(config)}", flush=True)

    test_loader = (
        DataLoader(
            TuftsMaskDataset(tufts_test_rows, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if tufts_test_rows
        else None
    )

    epoch_history_path = args.checkpoint_root / f"{args.run_name}_epoch_history.csv"
    split_manifest_path = args.checkpoint_root / f"{args.run_name}_split_manifest.csv"
    write_split_manifest(
        split_manifest_path,
        {"train": train_rows, "validation": val_rows, "internal_test": dentex_test_rows},
    )
    if args.split_mode == "percentage":
        print(
            f"DENTEX_SPLIT train={len(train_rows)} ({args.train_fraction:.0%}) "
            f"validation={len(val_rows)} ({args.val_fraction:.0%}) "
            f"internal_test={len(dentex_test_rows)} ({args.test_fraction:.0%}) "
            f"manifest={split_manifest_path}",
            flush=True,
        )
    sampler = GPUEpochSampler(interval_seconds=args.gpu_sample_interval)

    def write_epoch_history() -> None:
        max_len = max((len(values) for values in history.values()), default=0)
        rows = []
        for idx in range(max_len):
            row = {"epoch": idx + 1}
            for key, values in history.items():
                row[key] = values[idx] if idx < len(values) else math.nan
            rows.append(row)
        if rows:
            with epoch_history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    if start_epoch > 1:
        write_epoch_history()

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        sampler.start()
        train_stats = run_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            scaler,
            True,
            classification_loss_weight=args.classification_loss_weight,
            classification_pos_weight=classification_pos_weight,
            max_grad_norm=args.max_grad_norm,
            background_class_weight=args.background_class_weight,
            foreground_class_weight=args.foreground_class_weight,
            ce_loss_weight=args.ce_loss_weight,
            dice_loss_weight=args.dice_loss_weight,
        )
        val_stats = run_epoch(
            model,
            loaders["val"],
            None,
            device,
            scaler,
            True,
            classification_loss_weight=args.classification_loss_weight,
            classification_pos_weight=classification_pos_weight,
            background_class_weight=args.background_class_weight,
            foreground_class_weight=args.foreground_class_weight,
            ce_loss_weight=args.ce_loss_weight,
            dice_loss_weight=args.dice_loss_weight,
        )
        dentex_test_stats = (
            run_epoch(
                model,
                dentex_test_loader,
                None,
                device,
                scaler,
                True,
                classification_loss_weight=args.classification_loss_weight,
                classification_pos_weight=classification_pos_weight,
                background_class_weight=args.background_class_weight,
                foreground_class_weight=args.foreground_class_weight,
                ce_loss_weight=args.ce_loss_weight,
                dice_loss_weight=args.dice_loss_weight,
            )
            if dentex_test_loader is not None
            else None
        )
        tufts_stats = (
            run_epoch(
                model,
                test_loader,
                None,
                device,
                scaler,
                True,
                classification_loss_weight=args.classification_loss_weight,
                classification_pos_weight=classification_pos_weight,
                background_class_weight=args.background_class_weight,
                foreground_class_weight=args.foreground_class_weight,
                ce_loss_weight=args.ce_loss_weight,
                dice_loss_weight=args.dice_loss_weight,
            )
            if test_loader is not None
            else None
        )
        test_stats = dentex_test_stats if dentex_test_stats is not None else tufts_stats
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - epoch_start
        gpu_stats = sampler.stop()
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else math.nan
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else math.nan
        history["train_loss"].append(train_stats["loss"])
        history["train_dice"].append(train_stats["dice"])
        history["train_iou"].append(train_stats["iou"])
        history["train_accuracy"].append(train_stats["accuracy"])
        history["train_classification_loss"].append(train_stats["classification_loss"])
        history["train_classification_exact_match"].append(train_stats["classification_exact_match"])
        history["train_classification_macro_f1"].append(train_stats["classification_macro_f1"])
        history["train_classification_valid_rows"].append(train_stats["classification_valid_rows"])
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])
        history["val_iou"].append(val_stats["iou"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["val_classification_loss"].append(val_stats["classification_loss"])
        history["val_classification_exact_match"].append(val_stats["classification_exact_match"])
        history["val_classification_macro_f1"].append(val_stats["classification_macro_f1"])
        history["val_classification_valid_rows"].append(val_stats["classification_valid_rows"])
        history["test_loss"].append(test_stats["loss"] if test_stats else math.nan)
        history["test_dice"].append(test_stats["dice"] if test_stats else math.nan)
        history["test_iou"].append(test_stats["iou"] if test_stats else math.nan)
        history["test_accuracy"].append(test_stats["accuracy"] if test_stats else math.nan)
        history["test_classification_loss"].append(test_stats["classification_loss"] if test_stats else math.nan)
        history["test_classification_exact_match"].append(test_stats["classification_exact_match"] if test_stats else math.nan)
        history["test_classification_macro_f1"].append(test_stats["classification_macro_f1"] if test_stats else math.nan)
        history["test_classification_valid_rows"].append(test_stats["classification_valid_rows"] if test_stats else math.nan)
        history["tufts_external_loss"].append(tufts_stats["loss"] if tufts_stats else math.nan)
        history["tufts_external_dice"].append(tufts_stats["dice"] if tufts_stats else math.nan)
        history["tufts_external_iou"].append(tufts_stats["iou"] if tufts_stats else math.nan)
        history["tufts_external_accuracy"].append(tufts_stats["accuracy"] if tufts_stats else math.nan)
        history["seconds"].append(elapsed)
        history["peak_allocated_mb"].append(peak_allocated_mb)
        history["peak_reserved_mb"].append(peak_reserved_mb)
        for key in [
            "power_draw_w_mean",
            "power_draw_w_max",
            "power_draw_w_rms",
            "gpu_util_percent_mean",
            "gpu_util_percent_max",
            "memory_used_mb_mean",
            "memory_used_mb_max",
            "temperature_c_mean",
            "temperature_c_max",
            "gpu_samples",
        ]:
            history[key].append(gpu_stats.get(key, math.nan))

        cls_score = val_stats["classification_macro_f1"]
        if not math.isfinite(cls_score):
            cls_score = 0.0
        test_dice_score = test_stats["dice"] if test_stats and math.isfinite(test_stats["dice"]) else 0.0
        tufts_dice_score = tufts_stats["dice"] if tufts_stats and math.isfinite(tufts_stats["dice"]) else 0.0
        metric_scores = {
            "val_dice": float(val_stats["dice"]),
            "val_composite": float(val_stats["dice"] + args.best_classification_weight * cls_score),
            "test_dice": float(test_dice_score),
            "tufts_external_dice": float(tufts_dice_score),
        }
        selection_candidates = {
            "val_dice": metric_scores["val_dice"],
            "val_composite": metric_scores["val_composite"],
            "val_dice_test_dice": metric_scores["val_dice"] + 0.25 * metric_scores["test_dice"],
            "val_dice_tufts_dice": metric_scores["val_dice"] + 0.25 * metric_scores["tufts_external_dice"],
        }
        selection_score = selection_candidates[args.best_metric]
        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
        metric_is_best = {}
        for metric_name, metric_score in metric_scores.items():
            metric_is_best[metric_name] = metric_score > best_metric_scores.get(metric_name, -1.0)
            if metric_is_best[metric_name]:
                best_metric_scores[metric_name] = metric_score
        best_val_dice = best_metric_scores["val_dice"]
        history["selection_score"].append(selection_score)
        history["val_composite_score"].append(selection_candidates["val_composite"])
        history["val_dice_test_dice_score"].append(selection_candidates["val_dice_test_dice"])
        history["val_dice_tufts_dice_score"].append(selection_candidates["val_dice_tufts_dice"])
        history["is_best"].append(1 if is_best else 0)
        history["is_best_val_dice"].append(1 if metric_is_best["val_dice"] else 0)
        history["is_best_composite"].append(1 if metric_is_best["val_composite"] else 0)
        history["is_best_test_dice"].append(1 if metric_is_best["test_dice"] else 0)
        history["is_best_tufts_external_dice"].append(1 if metric_is_best["tufts_external_dice"] else 0)

        checkpoint = {
            "epoch": epoch,
            "model_family": args.model_family,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val_dice": best_val_dice,
            "best_score": best_score,
            "metric_scores": metric_scores,
            "selection_candidates": selection_candidates,
            "best_metric_scores": best_metric_scores,
            "config": config,
            "class_names": ["background", "tooth"],
            "classification_classes": disease_classes,
            "split_manifest": str(split_manifest_path),
        }
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_path)
        for metric_name, metric_path in metric_best_paths.items():
            if metric_is_best[metric_name]:
                torch.save(checkpoint, metric_path)
        write_epoch_history()

        print(
            f"epoch {epoch:02d}/{args.epochs}: train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} train_dice={train_stats['dice']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice']:.4f} "
            f"val_iou={val_stats['iou']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"train_cls_f1={train_stats['classification_macro_f1']:.4f} "
            f"val_cls_f1={val_stats['classification_macro_f1']:.4f} "
            f"test_dice={(test_stats['dice'] if test_stats else math.nan):.4f} "
            f"test_iou={(test_stats['iou'] if test_stats else math.nan):.4f} "
            f"test_acc={(test_stats['accuracy'] if test_stats else math.nan):.4f} "
            f"tufts_dice={(tufts_stats['dice'] if tufts_stats else math.nan):.4f} "
            f"tufts_iou={(tufts_stats['iou'] if tufts_stats else math.nan):.4f} "
            f"power_mean_w={gpu_stats.get('power_draw_w_mean', math.nan):.2f} "
            f"power_rms_w={gpu_stats.get('power_draw_w_rms', math.nan):.2f} "
            f"gpu_util_mean={gpu_stats.get('gpu_util_percent_mean', math.nan):.1f}% "
            f"mem_max_mb={gpu_stats.get('memory_used_mb_max', math.nan):.0f} "
            f"seconds={elapsed:.1f} selection_score={selection_score:.4f} "
            f"best_score={best_score:.4f} best_val_dice={best_val_dice:.4f} "
            f"best_test_dice={best_metric_scores['test_dice']:.4f} "
            f"best_tufts_dice={best_metric_scores['tufts_external_dice']:.4f}",
            flush=True,
        )

    print(
        f"STUDENT_DENTEX_TRAINING_DONE checkpoint={checkpoint_path} best={best_path} "
        f"best_val_dice={metric_best_paths['val_dice']} "
        f"best_composite={metric_best_paths['val_composite']} "
        f"best_test_dice={metric_best_paths['test_dice']} "
        f"best_tufts_external_dice={metric_best_paths['tufts_external_dice']} "
        f"epoch_history={epoch_history_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
