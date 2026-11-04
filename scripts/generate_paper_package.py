from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def copy_figure(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stack_images(paths: list[Path], labels: list[str], output_path: Path, direction: str = "horizontal") -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    font = ImageFont.load_default()
    label_h = 26
    if direction == "horizontal":
        target_h = max(image.height for image in images)
        resized = []
        for image in images:
            scale = target_h / image.height
            resized.append(image.resize((int(image.width * scale), target_h), Image.Resampling.LANCZOS))
        width = sum(image.width for image in resized)
        height = target_h + label_h
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, image in zip(labels, resized):
            canvas.paste(image, (x, label_h))
            draw.text((x + 8, 7), label, fill=(0, 0, 0), font=font)
            x += image.width
    else:
        target_w = max(image.width for image in images)
        resized = []
        for image in images:
            scale = target_w / image.width
            resized.append(image.resize((target_w, int(image.height * scale)), Image.Resampling.LANCZOS))
        width = target_w
        height = sum(image.height + label_h for image in resized)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        y = 0
        for label, image in zip(labels, resized):
            draw.text((8, y + 7), label, fill=(0, 0, 0), font=font)
            y += label_h
            canvas.paste(image, (0, y))
            y += image.height
    canvas.save(output_path)


def method_diagram(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)

    boxes = [
        ("DENTEX\npanoramic X-rays\n11GB", 0.5, 5.0, 2.2, 1.0, "#2a9d8f"),
        ("DentalAI\nRGB mask data\n1GB", 0.5, 3.2, 2.2, 1.0, "#8ab17d"),
        ("TUFTS\nX-ray true masks\nexternal", 0.5, 1.4, 2.2, 1.0, "#577590"),
        ("COCO conversion\n+ mask targets", 3.5, 4.1, 2.3, 1.0, "#6d597a"),
        ("TUFTS X-ray\nteacher segmenter", 3.5, 1.6, 2.3, 1.0, "#355070"),
        ("DENTEX pseudo\nmask generation", 6.6, 4.1, 2.4, 1.0, "#e76f51"),
        ("Mask cleanup\n+ quality gate", 6.6, 2.5, 2.4, 1.0, "#f4a261"),
        ("Lightweight\nstudent U-Net", 9.8, 3.3, 2.4, 1.0, "#2d6a4f"),
        ("Evaluation\nDENTEX + TUFTS", 12.5, 4.3, 2.1, 1.0, "#457b9d"),
        ("XAI + uncertainty\nfailure panels", 12.5, 2.2, 2.1, 1.0, "#7209b7"),
    ]

    for text, x, y, w, h, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=10, weight="bold")

    arrows = [
        ((2.7, 5.5), (3.5, 4.7)),
        ((2.7, 3.7), (3.5, 4.4)),
        ((2.7, 1.9), (3.5, 2.1)),
        ((5.8, 4.6), (6.6, 4.6)),
        ((5.8, 2.1), (6.6, 4.2)),
        ((7.8, 4.1), (7.8, 3.5)),
        ((9.0, 3.0), (9.8, 3.8)),
        ((12.2, 3.8), (12.5, 4.7)),
        ((12.2, 3.8), (12.5, 2.7)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"})

    ax.text(7.5, 6.5, "Dental AI Teacher-Student XAI Pipeline", ha="center", fontsize=16, weight="bold")
    ax.text(7.5, 0.6, "Core novelty: X-ray teacher -> DENTEX pseudo masks -> lightweight student -> XAI and uncertainty", ha="center", fontsize=11)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready Dental AI figures, tables, and manuscript snippets.")
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/paper_package_v1"))
    parser.add_argument("--final-eval-root", type=Path, default=Path("E:/dental_ai_results/final_eval_xai_v1"))
    parser.add_argument("--ablation-root", type=Path, default=Path("E:/dental_ai_results/ablation_comparison_v1"))
    parser.add_argument("--detector-root", type=Path, default=None)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    tables_dir = args.output_root / "tables"
    manuscript_dir = args.output_root / "manuscript"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    final_tables = args.final_eval_root / "tables"
    final_figures = args.final_eval_root / "figures"
    ablation_tables = args.ablation_root / "tables"
    ablation_figures = args.ablation_root / "figures"
    detector_tables = args.detector_root / "tables" if args.detector_root else None
    detector_figures = args.detector_root / "figures" if args.detector_root else None

    summary = json.loads((final_tables / "final_metrics_summary.json").read_text(encoding="utf-8"))
    teacher_student = pd.read_csv(ablation_tables / "teacher_vs_student.csv")
    raw_clean = pd.read_csv(ablation_tables / "raw_vs_clean_pseudo_masks.csv")
    uncertainty = pd.read_csv(ablation_tables / "uncertainty_filtering_curve.csv")
    baseline_vs_proposed = pd.read_csv(ablation_tables / "baseline_vs_proposed_summary.csv")

    method_diagram(figures_dir / "Fig1_method_pipeline.png")
    copy_figure(final_figures / "student_training_curves_final.png", figures_dir / "Fig2_training_curves.png")
    stack_images(
        [
            final_figures / "segmentation_confusion_matrices.png",
            final_figures / "pixel_level_roc_curves.png",
        ],
        ["A. Segmentation confusion matrices", "B. Pixel-level ROC curves"],
        figures_dir / "Fig3_confusion_and_roc.png",
    )
    copy_figure(final_figures / "dentex_xai_uncertainty_panels.png", figures_dir / "Fig4_dentex_xai_uncertainty.png")
    copy_figure(final_figures / "tufts_xai_uncertainty_panels.png", figures_dir / "Fig5_tufts_xai_uncertainty.png")
    copy_figure(final_figures / "tufts_failure_case_panels.png", figures_dir / "Fig6_tufts_failure_cases.png")
    stack_images(
        [
            ablation_figures / "teacher_vs_student.png",
            ablation_figures / "uncertainty_filtering_curve.png",
        ],
        ["A. Teacher-student deployment comparison", "B. Reliability filtering curve"],
        figures_dir / "Fig7_ablation_summary.png",
        direction="vertical",
    )
    copy_figure(ablation_figures / "raw_vs_clean_pseudo_masks.png", figures_dir / "Fig8_pseudo_mask_cleanup.png")
    copy_figure(final_figures / "augmentation_examples.png", figures_dir / "Fig9_augmentation_examples.png")
    detector_metrics = None
    detector_external = None
    detector_per_class = None
    if detector_tables and (detector_tables / "object_detection_metrics.csv").exists():
        detector_metrics = pd.read_csv(detector_tables / "object_detection_metrics.csv")
        detector_metrics.to_csv(tables_dir / "Table8_multiclass_detection_results.csv", index=False)
        if (detector_tables / "object_detection_tufts_external_metrics.csv").exists():
            detector_external = pd.read_csv(detector_tables / "object_detection_tufts_external_metrics.csv")
            detector_external.to_csv(tables_dir / "Table9_tufts_external_detection_results.csv", index=False)
        if (detector_tables / "object_detection_multiclass_per_class_iou50.csv").exists():
            detector_per_class = pd.read_csv(detector_tables / "object_detection_multiclass_per_class_iou50.csv")
            detector_per_class.to_csv(tables_dir / "Table10_multiclass_per_class_iou50.csv", index=False)
        if detector_figures and (detector_figures / "object_detection_training_curves.png").exists():
            copy_figure(detector_figures / "object_detection_training_curves.png", figures_dir / "Fig10_multiclass_detection_training.png")
        if detector_figures and (detector_figures / "object_detection_examples.png").exists():
            copy_figure(detector_figures / "object_detection_examples.png", figures_dir / "Fig11_multiclass_detection_examples.png")

    dataset_table = pd.DataFrame(
        [
            {"dataset": "DENTEX", "role": "in-domain pseudo-mask/student training", "modality": "panoramic X-ray", "rows_or_images": "quadrant 693; enumeration 634; disease 705; validation 50"},
            {"dataset": "DentalAI", "role": "real-mask teacher reference, not DENTEX X-ray source", "modality": "RGB intraoral", "rows_or_images": "train 1991; valid 254; test 250 masks"},
            {"dataset": "TUFTS", "role": "X-ray teacher and external validation", "modality": "panoramic X-ray", "rows_or_images": "1000 external images; 968 non-empty masks"},
        ]
    )
    dataset_table.to_csv(tables_dir / "Table1_dataset_summary.csv", index=False)

    main_results = pd.DataFrame(
        [
            {"method": "TUFTS X-ray teacher", "evaluation": "TUFTS teacher validation", "dice": summary["xray_teacher_best_val_dice"], "iou": summary["xray_teacher_best_val_iou"], "extra": f"best epoch {summary['xray_teacher_best_epoch']}"},
            {"method": "DENTEX student", "evaluation": "DENTEX pseudo-validation", "dice": summary["dentex_val_mean_dice"], "iou": summary["dentex_val_mean_iou"], "extra": f"n={summary['dentex_val_rows']}"},
            {"method": "DENTEX student", "evaluation": "TUFTS external all rows", "dice": summary["tufts_all_mean_dice"], "iou": summary["tufts_all_mean_iou"], "extra": f"n={summary['tufts_all_rows']}"},
            {"method": "DENTEX student", "evaluation": "TUFTS external non-empty masks", "dice": summary["tufts_nonempty_mean_dice"], "iou": summary["tufts_nonempty_mean_iou"], "extra": f"n={summary['tufts_nonempty_rows']}; empty={summary['tufts_empty_target_rows']}"},
        ]
    )
    main_results.to_csv(tables_dir / "Table2_main_results.csv", index=False)

    deployment_table = teacher_student[
        ["model", "params_million", "checkpoint_mb", "dentex_val_dice", "dentex_val_iou", "images_per_second"]
    ].copy()
    deployment_table.to_csv(tables_dir / "Table3_teacher_student_deployment.csv", index=False)

    raw_clean.to_csv(tables_dir / "Table4_pseudo_mask_cleanup.csv", index=False)
    uncertainty.to_csv(tables_dir / "Table5_reliability_filtering.csv", index=False)
    baseline_vs_proposed.to_csv(tables_dir / "Table6_baseline_vs_proposed.csv", index=False)

    student_row = teacher_student[teacher_student["model"].str.contains("student", case=False, na=False)].iloc[0]
    reliability_base = uncertainty.iloc[0]
    reliability_85 = uncertainty.iloc[(uncertainty["coverage"] - 0.85).abs().argsort()].iloc[0]
    novelty_claims = pd.DataFrame(
        [
            {"claim": "Cross-domain X-ray teacher supervision", "evidence": f"TUFTS X-ray teacher achieved Dice {summary['xray_teacher_best_val_dice']:.4f} and IoU {summary['xray_teacher_best_val_iou']:.4f}, then generated DENTEX pseudo masks."},
            {"claim": "Lightweight deployable student", "evidence": f"Student has {student_row['params_million']:.3f}M params and {student_row['checkpoint_mb']:.2f} MB checkpoint, with DENTEX pseudo-val Dice {summary['dentex_val_mean_dice']:.4f}."},
            {"claim": "External generalization", "evidence": f"Student achieved TUFTS non-empty Dice {summary['tufts_nonempty_mean_dice']:.4f} and IoU {summary['tufts_nonempty_mean_iou']:.4f} on {int(summary['tufts_nonempty_rows'])} external non-empty masks."},
            {"claim": "DENTEX multiclass enumeration", "evidence": "A COCO-pretrained Faster R-CNN enumeration detector is reported with mAP and per-class IoU@0.50 metrics on an internal DENTEX split."},
            {"claim": "XAI and uncertainty output", "evidence": "Final package includes DENTEX/TUFTS saliency, uncertainty heatmaps, error maps, and failure panels."},
            {"claim": "Reliability-aware filtering", "evidence": f"Failure rate Dice < 0.7 drops from {reliability_base['failure_rate_dice_lt_0_7']:.4f} to {reliability_85['failure_rate_dice_lt_0_7']:.4f} at {int(reliability_85['coverage'] * 100)} percent retained coverage."},
        ]
    )
    novelty_claims.to_csv(tables_dir / "Table7_novelty_claims.csv", index=False)

    figure_manifest = []
    captions = {
        "Fig1_method_pipeline.png": "Overview of the proposed Dental AI teacher-student XAI pipeline.",
        "Fig2_training_curves.png": "Student training loss and pseudo-mask validation Dice/IoU across epochs.",
        "Fig3_confusion_and_roc.png": "Segmentation confusion matrices and pixel-level ROC curves for DENTEX validation and TUFTS external testing.",
        "Fig4_dentex_xai_uncertainty.png": "DENTEX validation examples with target masks, predictions, error maps, gradient saliency, and uncertainty maps.",
        "Fig5_tufts_xai_uncertainty.png": "TUFTS external examples with prediction, saliency, uncertainty, and error visualization.",
        "Fig6_tufts_failure_cases.png": "Lowest-Dice non-empty TUFTS failure cases.",
        "Fig7_ablation_summary.png": "Teacher-student deployment comparison and reliability filtering curve.",
        "Fig8_pseudo_mask_cleanup.png": "Raw versus cleaned pseudo-mask tooth-area statistics.",
        "Fig9_augmentation_examples.png": "Image augmentation examples used during student training.",
        "Fig10_multiclass_detection_training.png": "Multiclass DENTEX enumeration detector training loss and validation mAP curves.",
        "Fig11_multiclass_detection_examples.png": "Multiclass DENTEX enumeration detector prediction examples.",
    }
    for path in sorted(figures_dir.glob("Fig*.png")):
        figure_manifest.append({"figure": path.name, "path": str(path), "caption": captions.get(path.name, "")})
    pd.DataFrame(figure_manifest).to_csv(tables_dir / "figure_manifest.csv", index=False)

    abstract = f"""
# Draft Abstract

This study presents a teacher-student framework for explainable dental X-ray segmentation and external validation. A TUFTS-trained X-ray teacher achieved Dice {summary['xray_teacher_best_val_dice']:.4f} and IoU {summary['xray_teacher_best_val_iou']:.4f}, then generated quality-gated pseudo masks for DENTEX panoramic radiographs. A lightweight student model trained on the cleaned DENTEX pseudo masks achieved Dice {summary['dentex_val_mean_dice']:.4f} and IoU {summary['dentex_val_mean_iou']:.4f} on DENTEX pseudo-validation, and generalized to TUFTS external non-empty masks with Dice {summary['tufts_nonempty_mean_dice']:.4f} and IoU {summary['tufts_nonempty_mean_iou']:.4f}. The final system includes saliency maps, uncertainty heatmaps, error maps, failure panels, and reliability filtering analysis, supporting a deployable and dentist-readable workflow.
"""
    write_markdown(manuscript_dir / "abstract_draft.md", abstract)

    methods = """
# Draft Methods Summary

The pipeline starts by converting DENTEX, DentalAI, and TUFTS sources into normalized COCO-style structures. TUFTS X-ray tooth masks are used to train a segmentation teacher. The teacher generates DENTEX pseudo masks, which are post-processed to remove border artifacts and implausible tooth-area predictions. A lightweight U-Net student is trained on the cleaned pseudo masks using CUDA AMP, stable Windows dataloading, and E-drive checkpointing to avoid OneDrive write failures. Evaluation is performed on DENTEX pseudo-validation and TUFTS external masks, with empty TUFTS target masks reported separately. Explainability outputs include gradient saliency, softmax entropy uncertainty, prediction overlays, and error maps.
"""
    write_markdown(manuscript_dir / "methods_summary_draft.md", methods)

    student_epoch = int(summary.get("student_epoch", 0) or 0)
    tufts_empty = int(summary.get("tufts_empty_target_rows", 0) or 0)
    tufts_nonempty = int(summary.get("tufts_nonempty_rows", 0) or 0)
    student_params = float(student_row.get("params_million", 0.0))
    student_mb = float(student_row.get("checkpoint_mb", 0.0))
    teacher_row = teacher_student[teacher_student["model"].str.contains("teacher", case=False, na=False)].iloc[0]
    teacher_params = float(teacher_row.get("params_million", 0.0))
    teacher_mb = float(teacher_row.get("checkpoint_mb", 0.0))

    results = f"""
# Draft Results Summary

The DENTEX student completed {student_epoch} epochs. On DENTEX pseudo-validation, it achieved Dice {summary['dentex_val_mean_dice']:.4f}, IoU {summary['dentex_val_mean_iou']:.4f}, and pixel-level ROC AUC {summary['dentex_pixel_roc_auc']:.4f}. On TUFTS external validation, the all-row mean Dice and IoU were {summary['tufts_all_mean_dice']:.4f} and {summary['tufts_all_mean_iou']:.4f}. Because {tufts_empty} TUFTS target masks were empty, non-empty-mask reporting is also provided; on {tufts_nonempty} non-empty masks, the model achieved Dice {summary['tufts_nonempty_mean_dice']:.4f}, IoU {summary['tufts_nonempty_mean_iou']:.4f}, and pixel-level ROC AUC {summary['tufts_pixel_roc_auc']:.4f}. The student model used {student_params:.3f}M parameters and a {student_mb:.2f} MB checkpoint, compared with the teacher's {teacher_params:.3f}M parameters and {teacher_mb:.2f} MB checkpoint, while maintaining strong pseudo-validation Dice.
"""
    if detector_metrics is not None and not detector_metrics.empty:
        det = detector_metrics.iloc[0]
        results += f"\nFor DENTEX multiclass enumeration, the COCO-pretrained Faster R-CNN baseline achieved internal mAP@50 {det['map_50']:.4f}, mAP@75 {det['map_75']:.4f}, and mAP@50:95 {det['map_50_95']:.4f}. This should be reported as an internal DENTEX multiclass result, while TUFTS remains an external segmentation validation set because it does not provide DENTEX-style tooth-position class labels.\n"
    write_markdown(manuscript_dir / "results_summary_draft.md", results)

    limitations = """
# Draft Limitations

The DENTEX validation masks are pseudo masks, not manually verified clinical contours. TUFTS contains a small number of empty target masks that must be handled separately during reporting. TUFTS does not provide DENTEX-style tooth-position multiclass labels, so TUFTS can validate segmentation/generalization but not external multiclass enumeration. Current XAI uses gradient saliency, LIME-style perturbation, KernelSHAP-style perturbation, and softmax entropy as first-version explanations; reviewer-facing work may benefit from additional occlusion sensitivity or MC-dropout calibration. The first ablation package is evidence-based and uses existing checkpoints; full retraining ablations without pseudo-mask cleanup or uncertainty-aware filtering can be added if required.
"""
    write_markdown(manuscript_dir / "limitations_draft.md", limitations)

    package_summary = {
        "output_root": str(args.output_root),
        "figures": figure_manifest,
        "tables": sorted(str(path) for path in tables_dir.glob("*.csv")),
        "manuscript_snippets": sorted(str(path) for path in manuscript_dir.glob("*.md")),
    }
    (args.output_root / "paper_package_manifest.json").write_text(json.dumps(package_summary, indent=2), encoding="utf-8")
    print(json.dumps(package_summary, indent=2), flush=True)
    print(f"PAPER_PACKAGE_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
