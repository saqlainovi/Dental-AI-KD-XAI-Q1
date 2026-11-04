from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Dental_AI_JOURNAL_READY_ALL_RESULTS_v1.ipynb"
FALLBACK_NOTEBOOK_PATH = Path("E:/dental_ai_results/fresh_rerun_20260518_112354/Dental_AI_JOURNAL_READY_ALL_RESULTS_v1.ipynb")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            """
            # Dental AI Journal-Ready All Results Notebook

            This notebook is the single journal-facing artifact for the current Dental AI project.

            It includes data loading evidence, augmentation examples, full training histories, validation/testing
            metrics, TUFTS external validation, ablation studies, object-detection baselines, XAI/KernelSHAP-style
            values, and paper-ready figures.

            Scope note: the strongest publishable claim is **teacher-student pseudo-mask tooth segmentation with
            external TUFTS validation and explainability**. Detection/disease results are included as baselines,
            but should not be the main journal claim.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import json
            import subprocess
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import Image as IPyImage
            from IPython.display import Markdown, display

            PROJECT_ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
            RUN_ID = "fresh_rerun_20260518_112354"
            RESULT_ROOT = Path("E:/dental_ai_results") / RUN_ID
            CHECKPOINT_ROOT = Path("E:/dental_ai_checkpoints") / RUN_ID
            COMPLETE_ROOT = RESULT_ROOT / "complete_research_single_notebook_v1"
            PAPER_ROOT = RESULT_ROOT / "paper_package_v1"
            AUDIT_TABLES = RESULT_ROOT / "research_audit" / "tables"
            FINAL_EVAL = RESULT_ROOT / "final_eval_xai_v1"
            ABLATION_ROOT = RESULT_ROOT / "ablation_comparison_v1"
            PYTHON = Path(sys.executable)

            # Keep False for paper review. Set True only when you intentionally want to rerun long training.
            RUN_FULL_RETRAIN = False

            def read_json(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))

            def table(path, n=50, title=None):
                path = Path(path)
                if title:
                    display(Markdown(f"### {title}"))
                if not path.exists():
                    display(Markdown(f"Missing: `{path}`"))
                    return pd.DataFrame()
                df = pd.read_csv(path)
                display(df.head(n))
                return df

            def image(path, width=1000, title=None):
                path = Path(path)
                if title:
                    display(Markdown(f"### {title}"))
                if path.exists():
                    display(IPyImage(filename=str(path), width=width))
                else:
                    display(Markdown(f"Missing image: `{path}`"))

            def run_command(cmd):
                cmd = [str(part) for part in cmd]
                print("RUN:", " ".join(cmd))
                result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {result.returncode}")
                return result

            print("Project root:", PROJECT_ROOT)
            print("Run id:", RUN_ID)
            print("Result root:", RESULT_ROOT)
            """
        ),
        md("## 1. Data Loading and Dataset Evidence"),
        code(
            r"""
            coco_root = PROJECT_ROOT / "data" / "processed" / "coco"
            rows = []
            for path in [
                coco_root / "dentex" / "quadrant.json",
                coco_root / "dentex" / "enumeration.json",
                coco_root / "dentex" / "disease.json",
                coco_root / "dentex" / "validation_images.json",
                coco_root / "dentalai" / "train.json",
                coco_root / "dentalai" / "valid.json",
                coco_root / "dentalai" / "test.json",
                coco_root / "tufts_teeth.json",
            ]:
                data = read_json(path)
                rows.append({
                    "dataset_file": str(path.relative_to(PROJECT_ROOT)),
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                    "categories": ", ".join(str(c.get("name")) for c in data.get("categories", [])[:8]),
                })
            dataset_df = pd.DataFrame(rows)
            display(dataset_df)

            display(Markdown("### Paper dataset table"))
            table(PAPER_ROOT / "tables" / "Table1_dataset_summary.csv")
            """
        ),
        md("## 2. Data Augmentation and Preprocessing"),
        code(
            r"""
            image(PAPER_ROOT / "figures" / "Fig9_augmentation_examples.png", width=1000, title="Augmentation examples")
            display(Markdown(
                "- Segmentation inputs are resized to 512x512 during model training.\\n"
                "- Final evaluation upsamples prediction maps back to original image size before Dice/IoU.\\n"
                "- Crop-classifier baselines use 224x224 tooth crops and are not the main segmentation model."
            ))
            """
        ),
        md("## 3. Optional Full Training Commands"),
        code(
            r"""
            display(Markdown(
                "The current journal numbers below come from the completed fresh run. "
                "Set `RUN_FULL_RETRAIN = True` only if you want to rerun long training from this notebook."
            ))

            if RUN_FULL_RETRAIN:
                run_command([
                    PYTHON,
                    PROJECT_ROOT / "scripts" / "train_xray_teacher_segmentation.py",
                    "--checkpoint-root", CHECKPOINT_ROOT,
                ])
                run_command([
                    PYTHON,
                    PROJECT_ROOT / "scripts" / "train_dentex_student_segmentation.py",
                    "--clean-root", Path("E:/dental_ai_processed/pseudo_masks") / RUN_ID / "dentex_xray_teacher_clean",
                    "--checkpoint-root", CHECKPOINT_ROOT,
                    "--run-name", "student_tinyunet_dentex_pseudo_full",
                    "--epochs", "25",
                    "--image-size", "512",
                    "--batch-size", "4",
                    "--resume",
                ])
            """
        ),
        md("## 4. Full Training History: Per-Epoch Train/Validation"),
        code(
            r"""
            student_history = table(AUDIT_TABLES / "epoch_history_dentex_student_last.csv", n=30, title="DENTEX student per-epoch history")
            teacher_history = table(AUDIT_TABLES / "epoch_history_tufts_xray_teacher.csv", n=45, title="TUFTS X-ray teacher per-epoch history")
            dentalai_history = table(AUDIT_TABLES / "epoch_history_dentalai_reference_teacher.csv", n=35, title="DentalAI reference teacher per-epoch history")

            if not student_history.empty:
                fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                axes[0].plot(student_history["epoch"], student_history["train_loss"], marker="o", label="train loss")
                axes[0].plot(student_history["epoch"], student_history["val_loss"], marker="o", label="val loss")
                axes[0].set_title("DENTEX Student Loss")
                axes[0].set_xlabel("Epoch")
                axes[0].legend()
                axes[1].plot(student_history["epoch"], student_history["val_dice"], marker="o", label="val Dice")
                axes[1].plot(student_history["epoch"], student_history["val_iou"], marker="o", label="val IoU")
                axes[1].set_title("DENTEX Student Validation Metrics")
                axes[1].set_xlabel("Epoch")
                axes[1].legend()
                plt.tight_layout()
                plt.show()

            image(PAPER_ROOT / "figures" / "Fig2_training_curves.png", width=1000, title="Paper training curve")
            """
        ),
        md("## 5. Main Train/Test/Validation Results"),
        code(
            r"""
            summary = read_json(FINAL_EVAL / "tables" / "final_metrics_summary.json")
            display(pd.DataFrame([summary]).T.rename(columns={0: "value"}))

            table(AUDIT_TABLES / "segmentation_binary_class_metrics.csv", title="Per-class segmentation metrics")
            table(FINAL_EVAL / "tables" / "dentex_validation_student_eval.csv", n=10, title="DENTEX pseudo-validation sample rows")
            table(FINAL_EVAL / "tables" / "tufts_external_student_eval_nonempty.csv", n=10, title="TUFTS external non-empty sample rows")

            image(PAPER_ROOT / "figures" / "Fig3_confusion_and_roc.png", width=1100, title="Confusion matrices and ROC")
            image(PAPER_ROOT / "figures" / "Fig6_tufts_failure_cases.png", width=1100, title="TUFTS failure cases")
            """
        ),
        md("## 6. TUFTS External Testing/Validation"),
        code(
            r"""
            tufts = pd.read_csv(FINAL_EVAL / "tables" / "tufts_external_student_eval.csv")
            tufts_nonempty = pd.read_csv(FINAL_EVAL / "tables" / "tufts_external_student_eval_nonempty.csv")
            display(pd.DataFrame([
                {"split": "TUFTS all", "rows": len(tufts), "mean_dice": tufts["dice"].mean(), "mean_iou": tufts["iou"].mean(), "empty_targets": int(tufts["target_empty"].sum())},
                {"split": "TUFTS non-empty", "rows": len(tufts_nonempty), "mean_dice": tufts_nonempty["dice"].mean(), "mean_iou": tufts_nonempty["iou"].mean(), "empty_targets": 0},
            ]))
            image(PAPER_ROOT / "figures" / "Fig5_tufts_xai_uncertainty.png", width=1100, title="TUFTS XAI/uncertainty examples")
            """
        ),
        md("## 7. Ablation Studies"),
        code(
            r"""
            table(ABLATION_ROOT / "tables" / "teacher_vs_student.csv", title="Teacher vs student")
            table(ABLATION_ROOT / "tables" / "raw_vs_clean_pseudo_masks.csv", title="Raw vs cleaned pseudo masks")
            table(ABLATION_ROOT / "tables" / "uncertainty_filtering_curve.csv", title="Reliability / uncertainty filtering")
            table(COMPLETE_ROOT / "strict_raw_ablation_eval" / "tables" / "final_metrics_summary.csv", title="Strict raw pseudo-mask ablation eval")
            image(PAPER_ROOT / "figures" / "Fig7_ablation_summary.png", width=1000, title="Ablation summary")
            image(PAPER_ROOT / "figures" / "Fig8_pseudo_mask_cleanup.png", width=900, title="Pseudo-mask cleanup")
            """
        ),
        md("## 8. Object Detection and Localization Baselines"),
        code(
            r"""
            table(COMPLETE_ROOT / "object_detection_baseline_v1" / "tables" / "object_detection_metrics.csv", title="Faster R-CNN from scratch: internal DENTEX")
            table(COMPLETE_ROOT / "object_detection_baseline_v1" / "tables" / "object_detection_tufts_external_metrics.csv", title="Faster R-CNN from scratch: TUFTS external")
            table(COMPLETE_ROOT / "object_detection_pretrained_coco_v2" / "tables" / "object_detection_metrics.csv", title="COCO-pretrained Faster R-CNN: internal DENTEX")
            table(COMPLETE_ROOT / "object_detection_pretrained_coco_v2" / "tables" / "object_detection_tufts_external_metrics.csv", title="COCO-pretrained Faster R-CNN: TUFTS external")
            table(COMPLETE_ROOT / "segmentation_localization_v1" / "tables" / "segmentation_localization_metrics.csv", title="Segmentation-derived localization")
            image(COMPLETE_ROOT / "object_detection_pretrained_coco_v2" / "figures" / "object_detection_training_curves.png", width=1000, title="COCO-pretrained detector curves")
            image(COMPLETE_ROOT / "object_detection_pretrained_coco_v2" / "figures" / "object_detection_pr_curve_iou50.png", width=700, title="Internal detection PR curve")
            image(COMPLETE_ROOT / "object_detection_pretrained_coco_v2" / "figures" / "object_detection_tufts_external_pr_curve_iou50.png", width=700, title="TUFTS external detection PR curve")
            """
        ),
        md("## 9. XAI, KernelSHAP-Style Values, and Uncertainty"),
        code(
            r"""
            image(PAPER_ROOT / "figures" / "Fig4_dentex_xai_uncertainty.png", width=1100, title="DENTEX saliency and uncertainty")
            image(PAPER_ROOT / "figures" / "Fig5_tufts_xai_uncertainty.png", width=1100, title="TUFTS saliency and uncertainty")
            xai_summary = table(COMPLETE_ROOT / "lime_shap_xai_v1" / "tables" / "lime_shap_xai_summary.csv", title="LIME/KernelSHAP-style numeric summary")
            if not xai_summary.empty:
                value_cols = [
                    "source", "mean_tooth_probability", "lime_target_overlap", "kernel_shap_target_overlap",
                    "lime_pred_overlap", "kernel_shap_pred_overlap",
                    "lime_top20_deletion_drop", "kernel_shap_top20_deletion_drop"
                ]
                display(xai_summary[value_cols])
            image(COMPLETE_ROOT / "lime_shap_xai_v1" / "figures" / "dentex_lime_shap_panels.png", width=1100, title="DENTEX LIME/KernelSHAP-style panels")
            image(COMPLETE_ROOT / "lime_shap_xai_v1" / "figures" / "tufts_lime_shap_panels.png", width=1100, title="TUFTS LIME/KernelSHAP-style panels")
            """
        ),
        md("## 10. Memory, Runtime, and Deployment Evidence"),
        code(
            r"""
            table(AUDIT_TABLES / "memory_benchmark_synthetic_train_step.csv", title="CUDA AMP memory benchmark")
            table(PAPER_ROOT / "tables" / "Table3_teacher_student_deployment.csv", title="Teacher/student deployment table")
            """
        ),
        md("## 11. Paper Figures, Tables, and Manuscript Material"),
        code(
            r"""
            table(PAPER_ROOT / "tables" / "Table2_main_results.csv", title="Paper Table 2 main results")
            table(PAPER_ROOT / "tables" / "Table6_baseline_vs_proposed.csv", title="Paper Table 6 baseline comparison")
            table(PAPER_ROOT / "tables" / "Table7_novelty_claims.csv", title="Paper Table 7 novelty claims")
            image(PAPER_ROOT / "figures" / "Fig1_method_pipeline.png", width=1100, title="Method pipeline")
            display(Markdown((PAPER_ROOT / "manuscript" / "abstract_draft.md").read_text(encoding="utf-8")))
            display(Markdown((PAPER_ROOT / "manuscript" / "limitations_draft.md").read_text(encoding="utf-8")))
            """
        ),
        md("## 12. Final Journal Readiness Dashboard"),
        code(
            r"""
            dashboard = pd.DataFrame([
                {"item": "12GB DENTEX available training tasks", "status": "complete", "evidence": "2,030 pseudo-mask student training rows"},
                {"item": "Per-epoch train/validation history", "status": "complete", "evidence": "student, teacher, reference teacher CSVs and curves shown above"},
                {"item": "TUFTS external testing/validation", "status": "complete", "evidence": "1,000 rows; 968 non-empty masks"},
                {"item": "Trustworthy segmentation metrics", "status": "complete", "evidence": "Dice, IoU, ROC AUC, confusion counts, per-class metrics"},
                {"item": "Ablation", "status": "complete", "evidence": "teacher/student, raw/clean, uncertainty filtering, raw retrain"},
                {"item": "XAI/KernelSHAP-style values", "status": "complete", "evidence": "overlap and deletion-drop table plus panels"},
                {"item": "Object detection", "status": "baseline only", "evidence": "internal mAP improved with COCO pretraining, TUFTS external detection weak"},
                {"item": "Disease diagnosis", "status": "not main claim", "evidence": "crop baseline weak"},
            ])
            display(dashboard)
            display(Markdown(
                "**Submission recommendation:** write this as a segmentation + pseudo-mask transfer + XAI paper. "
                "Do not frame it as a strong object-detection or disease-diagnosis paper."
            ))
            """
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        nbf.write(nb, NOTEBOOK_PATH)
        print(NOTEBOOK_PATH)
    except OSError:
        FALLBACK_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, FALLBACK_NOTEBOOK_PATH)
        print(FALLBACK_NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
