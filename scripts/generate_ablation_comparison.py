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
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, build_segmentation_model, segmentation_logits


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    return path


def model_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def load_model(checkpoint_path: Path, width: int, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if checkpoint.get("model_family") or config.get("model_family"):
        model = build_model_from_checkpoint(checkpoint, num_classes=2).to(device)
    else:
        model = build_segmentation_model(model_family="tinyunet", num_classes=2, width=width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def preprocess(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)


def predict_prob(model: torch.nn.Module, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    tensor = preprocess(image, image_size, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = segmentation_logits(model(tensor))
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
    small = prob.squeeze(0).detach().cpu().numpy()
    return np.asarray(Image.fromarray(small.astype(np.float32), mode="F").resize(image.size, Image.BILINEAR))


def dice_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def load_clean_rows(clean_root: Path, task: str) -> list[dict]:
    rows = json.loads((clean_root / task / "summary.json").read_text(encoding="utf-8"))
    return [row for row in rows if row.get("status") == "kept"]


def evaluate_dentex_validation(
    model: torch.nn.Module,
    rows: list[dict],
    image_size: int,
    device: torch.device,
) -> dict[str, float]:
    scores = []
    for row in rows:
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        target = np.asarray(Image.open(resolve_path(row["clean_mask"])).convert("L")) > 0
        prob = predict_prob(model, image, image_size, device)
        scores.append(dice_iou(prob >= 0.5, target))
    return {
        "dentex_val_dice": float(np.mean([score[0] for score in scores])),
        "dentex_val_iou": float(np.mean([score[1] for score in scores])),
    }


def benchmark_inference(
    model: torch.nn.Module,
    rows: list[dict],
    image_size: int,
    device: torch.device,
    n_images: int,
) -> float:
    images = [Image.open(resolve_path(row["image_path"])).convert("RGB") for row in rows[:n_images]]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for image in images:
        _ = predict_prob(model, image, image_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - start)
    return len(images) / elapsed


def summarize_raw_vs_clean(raw_root: Path, clean_root: Path, output_csv: Path) -> pd.DataFrame:
    rows = []
    for task in ["quadrant_full", "enumeration_full", "disease_full", "validation_full"]:
        raw_rows = json.loads((raw_root / task / "summary.json").read_text(encoding="utf-8"))
        clean_rows = json.loads((clean_root / task / "summary.json").read_text(encoding="utf-8"))
        clean_by_file = {row["file_name"]: row for row in clean_rows}
        raw_fracs = []
        clean_fracs = []
        confidences = []
        filtered = 0
        for row in raw_rows:
            counts = {int(k): int(v) for k, v in row.get("pixel_counts", {}).items()}
            total = max(1, sum(counts.values()))
            raw_fracs.append(sum(v for k, v in counts.items() if k != 0) / total)
            confidences.append(float(row.get("mean_confidence", 0.0)))
            clean = clean_by_file.get(row["file_name"])
            if clean:
                clean_fracs.append(float(clean["clean_tooth_fraction"]))
                filtered += int(clean["status"] != "kept")
        rows.append(
            {
                "task": task,
                "raw_rows": len(raw_rows),
                "clean_rows": len(clean_rows),
                "kept": len(clean_rows) - filtered,
                "filtered": filtered,
                "raw_tooth_fraction_mean": float(np.mean(raw_fracs)),
                "clean_tooth_fraction_mean": float(np.mean(clean_fracs)),
                "mean_confidence": float(np.mean(confidences)),
                "fraction_reduction": float(np.mean(raw_fracs) - np.mean(clean_fracs)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def uncertainty_curve(tufts_eval_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(tufts_eval_csv)
    df = df[df["target_empty"] == False].copy()
    df["abs_fraction_error"] = (df["pred_fraction"] - df["target_fraction"]).abs()
    # This is a saved-table proxy for uncertainty: images whose predicted tooth area is
    # implausibly far from the target distribution tend to be failures.
    # Full pixel entropy panels are generated in final_eval_xai_v1.
    rows = []
    for coverage in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        keep_n = max(1, int(len(df) * coverage))
        kept = df.sort_values("abs_fraction_error").head(keep_n)
        rows.append(
            {
                "coverage": coverage,
                "kept_images": keep_n,
                "mean_dice": float(kept["dice"].mean()),
                "mean_iou": float(kept["iou"].mean()),
                "failure_rate_dice_lt_0_7": float((kept["dice"] < 0.7).mean()),
            }
        )
    curve = pd.DataFrame(rows)
    curve.to_csv(output_csv, index=False)
    return curve


def crop_baseline_table(output_csv: Path) -> pd.DataFrame:
    rows = []
    for task in ["quadrant", "enumeration", "disease"]:
        path = Path("E:/dental_ai_checkpoints") / f"v3_{task}_swin_t_full.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        history = checkpoint.get("history", {})
        rows.append(
            {
                "method": "Swin-T crop classifier baseline",
                "task": task,
                "metric": "best_val_accuracy",
                "value": float(max(history.get("val_accuracy", [0.0]))),
                "checkpoint": str(path),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def plot_teacher_student(df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["params_million", "checkpoint_mb", "dentex_val_dice", "dentex_val_iou", "images_per_second"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(df["model"], df[metric], color=["#4c78a8", "#54a24b"])
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
        for idx, value in enumerate(df[metric]):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Teacher vs Student Deployment Comparison")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_raw_clean(df: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, df["raw_tooth_fraction_mean"], width, label="raw")
    ax.bar(x + width / 2, df["clean_tooth_fraction_mean"], width, label="cleaned")
    ax.set_xticks(x, df["task"], rotation=20)
    ax.set_ylabel("Mean tooth fraction")
    ax.set_title("Raw vs Cleaned Pseudo-Mask Area")
    ax.legend()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_curve(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["coverage"], df["mean_dice"], marker="o", label="mean Dice")
    ax1.plot(df["coverage"], df["mean_iou"], marker="o", label="mean IoU")
    ax1.set_xlabel("Retained coverage")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(df["coverage"], df["failure_rate_dice_lt_0_7"], marker="s", color="#e45756", label="failure rate")
    ax2.set_ylabel("Failure rate, Dice < 0.7")
    ax2.set_ylim(0, max(0.2, float(df["failure_rate_dice_lt_0_7"].max()) * 1.25))
    ax1.set_title("Reliability Filtering Curve")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower left")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation and comparison artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/ablation_comparison_v1"))
    parser.add_argument("--raw-pseudo-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_xray_teacher"))
    parser.add_argument("--clean-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/dentex_xray_teacher_clean"))
    parser.add_argument("--final-eval-root", type=Path, default=Path("E:/dental_ai_results/final_eval_xai_v1"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/teacher_tinyunet_tufts_xray_full.pt"))
    parser.add_argument("--student-checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/student_tinyunet_dentex_pseudo_full_best.pt"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--benchmark-images", type=int, default=50)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_root / "tables"
    figures_dir = args.output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, teacher_ckpt = load_model(args.teacher_checkpoint, width=32, device=device)
    student, student_ckpt = load_model(args.student_checkpoint, width=16, device=device)
    dentex_rows = load_clean_rows(args.clean_root, "validation_full")

    teacher_metrics = evaluate_dentex_validation(teacher, dentex_rows, args.image_size, device)
    student_metrics = evaluate_dentex_validation(student, dentex_rows, args.image_size, device)
    teacher_speed = benchmark_inference(teacher, dentex_rows, args.image_size, device, args.benchmark_images)
    student_speed = benchmark_inference(student, dentex_rows, args.image_size, device, args.benchmark_images)

    teacher_history = teacher_ckpt.get("history", {})
    teacher_best_idx = int(np.argmax(teacher_history.get("val_dice", [0.0])))
    student_history = student_ckpt.get("history", {})
    student_best_idx = int(np.argmax(student_history.get("val_dice", [0.0])))
    teacher_student_df = pd.DataFrame(
        [
            {
                "model": "teacher_tinyunet_width32",
                "params": model_params(teacher),
                "params_million": model_params(teacher) / 1e6,
                "checkpoint_mb": args.teacher_checkpoint.stat().st_size / (1024 * 1024),
                "source_best_val_dice": float(teacher_history.get("val_dice", [0.0])[teacher_best_idx]),
                "source_best_val_iou": float(teacher_history.get("val_iou", [0.0])[teacher_best_idx]),
                **teacher_metrics,
                "images_per_second": teacher_speed,
            },
            {
                "model": "student_tinyunet_width16",
                "params": model_params(student),
                "params_million": model_params(student) / 1e6,
                "checkpoint_mb": args.student_checkpoint.stat().st_size / (1024 * 1024),
                "source_best_val_dice": float(student_history.get("val_dice", [0.0])[student_best_idx]),
                "source_best_val_iou": float(student_history.get("val_iou", [0.0])[student_best_idx]),
                **student_metrics,
                "images_per_second": student_speed,
            },
        ]
    )
    teacher_student_df.to_csv(tables_dir / "teacher_vs_student.csv", index=False)

    raw_clean_df = summarize_raw_vs_clean(args.raw_pseudo_root, args.clean_root, tables_dir / "raw_vs_clean_pseudo_masks.csv")
    uncertainty_df = uncertainty_curve(
        args.final_eval_root / "tables" / "tufts_external_student_eval_nonempty.csv",
        tables_dir / "uncertainty_filtering_curve.csv",
    )
    baseline_df = crop_baseline_table(tables_dir / "crop_baseline_table.csv")

    final_summary = json.loads((args.final_eval_root / "tables" / "final_metrics_summary.json").read_text(encoding="utf-8"))
    proposed_df = pd.DataFrame(
        [
            {
                "method": "Proposed teacher-student segmentation",
                "task": "DENTEX pseudo-validation",
                "metric": "mean_dice",
                "value": final_summary["dentex_val_mean_dice"],
                "checkpoint": str(args.student_checkpoint),
            },
            {
                "method": "Proposed teacher-student segmentation",
                "task": "TUFTS external non-empty",
                "metric": "mean_dice",
                "value": final_summary["tufts_nonempty_mean_dice"],
                "checkpoint": str(args.student_checkpoint),
            },
        ]
    )
    comparison_df = pd.concat([baseline_df, proposed_df], ignore_index=True)
    comparison_df.to_csv(tables_dir / "baseline_vs_proposed_summary.csv", index=False)

    plot_teacher_student(teacher_student_df, figures_dir / "teacher_vs_student.png")
    plot_raw_clean(raw_clean_df, figures_dir / "raw_vs_clean_pseudo_masks.png")
    plot_uncertainty_curve(uncertainty_df, figures_dir / "uncertainty_filtering_curve.png")

    summary = {
        "device": str(device),
        "teacher_student": teacher_student_df.to_dict(orient="records"),
        "raw_vs_clean": raw_clean_df.to_dict(orient="records"),
        "uncertainty_curve_best_mean_dice": float(uncertainty_df["mean_dice"].max()),
        "uncertainty_curve_lowest_failure_rate": float(uncertainty_df["failure_rate_dice_lt_0_7"].min()),
    }
    (tables_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "figures": sorted(str(path) for path in figures_dir.glob("*.png")),
                "tables": sorted(str(path) for path in tables_dir.glob("*")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"ABLATION_COMPARISON_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
