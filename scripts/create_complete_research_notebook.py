from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Dental_AI_COMPLETE_RESEARCH_SINGLE_NOTEBOOK_v1.ipynb"
FALLBACK_NOTEBOOK_PATH = Path("E:/dental_ai_results/complete_research_notebook_build/Dental_AI_COMPLETE_RESEARCH_SINGLE_NOTEBOOK_v1.ipynb")


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
            # Dental AI Complete Research Single Notebook v1

            This is the one notebook to review and run the complete research package in one place.

            It consolidates:

            - existing fresh-run segmentation results
            - per-epoch training histories
            - TUFTS external validation
            - object detection baseline with mAP@50, mAP@75, mAP@50:95
            - strict raw-pseudo-mask ablation hook
            - LIME-style and KernelSHAP-style XAI panels
            - memory/speed evidence
            - final completion dashboard

            Important: the long training cells can take hours on the RTX 3060. Keep `RUN_LONG_TRAINING = True`
            only when you want to actually run the missing long experiments.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import json
            import os
            import subprocess
            import sys

            import pandas as pd
            from IPython.display import Image as IPyImage
            from IPython.display import Markdown, display

            PROJECT_ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
            RUN_ID = "fresh_rerun_20260518_112354"
            RESULT_ROOT = Path("E:/dental_ai_results") / RUN_ID
            CHECKPOINT_ROOT = Path("E:/dental_ai_checkpoints") / RUN_ID
            PROCESSED_ROOT = Path("E:/dental_ai_processed/pseudo_masks") / RUN_ID
            COMPLETE_ROOT = RESULT_ROOT / "complete_research_single_notebook_v1"
            COMPLETE_ROOT.mkdir(parents=True, exist_ok=True)

            PYTHON = Path(sys.executable)

            # Full mode will run the remaining long experiments if their output does not exist.
            # For quick review only, set this to False and run all cells.
            RUN_LONG_TRAINING = True
            RUN_OBJECT_DETECTION = RUN_LONG_TRAINING
            RUN_STRICT_RAW_ABLATION = RUN_LONG_TRAINING
            RUN_LIME_SHAP_XAI = True

            OBJECT_DETECTION_EPOCHS = 8
            OBJECT_DETECTION_IMAGE_SIZE = 512
            OBJECT_DETECTION_BATCH_SIZE = 2

            RAW_ABLATION_EPOCHS = 12
            RAW_ABLATION_IMAGE_SIZE = 512
            RAW_ABLATION_BATCH_SIZE = 4

            LIME_SHAP_EXAMPLES = 3
            LIME_SHAP_SAMPLES = 120
            LIME_SHAP_SEGMENTS = 70
            LIME_SHAP_SIZE = 256

            print("Project root:", PROJECT_ROOT)
            print("Current run:", RUN_ID)
            print("Complete notebook output:", COMPLETE_ROOT)
            print("Python:", PYTHON)
            """
        ),
        code(
            r"""
            def run_command(cmd):
                cmd = [str(part) for part in cmd]
                print("RUN:", " ".join(cmd))
                result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed with return code {result.returncode}")
                return result


            def read_json(path):
                path = Path(path)
                return json.loads(path.read_text(encoding="utf-8"))


            def display_csv(path, n=20):
                path = Path(path)
                if not path.exists():
                    display(Markdown(f"Missing table: `{path}`"))
                    return None
                df = pd.read_csv(path)
                display(df.head(n))
                return df


            def show_image(path, width=1000):
                path = Path(path)
                if path.exists():
                    display(IPyImage(filename=str(path), width=width))
                else:
                    display(Markdown(f"Missing figure: `{path}`"))
            """
        ),
        md("## 1. Current Fresh-Run Metrics"),
        code(
            r"""
            summary_path = RESULT_ROOT / "final_eval_xai_v1" / "tables" / "final_metrics_summary.json"
            summary = read_json(summary_path)
            current_metrics = pd.DataFrame([summary])
            display(current_metrics.T.rename(columns={0: "value"}))

            display(Markdown("### Student per-epoch history"))
            display_csv(RESULT_ROOT / "research_audit" / "tables" / "epoch_history_dentex_student_last.csv", n=30)

            display(Markdown("### Paper-ready training curve"))
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig2_training_curves.png", width=1000)
            """
        ),
        md("## 2. Existing Segmentation Evaluation, ROC, Confusion, XAI"),
        code(
            r"""
            display(Markdown("### Main evaluation tables"))
            display_csv(RESULT_ROOT / "research_audit" / "tables" / "segmentation_binary_class_metrics.csv")
            display_csv(RESULT_ROOT / "research_audit" / "tables" / "memory_benchmark_synthetic_train_step.csv")

            display(Markdown("### Confusion + ROC"))
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig3_confusion_and_roc.png", width=1100)

            display(Markdown("### Existing saliency / uncertainty XAI"))
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig4_dentex_xai_uncertainty.png", width=1100)
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig5_tufts_xai_uncertainty.png", width=1100)
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig6_tufts_failure_cases.png", width=1100)
            """
        ),
        md("## 3. Remaining Task A: Object Detection Baseline"),
        code(
            r"""
            DETECTION_ROOT = COMPLETE_ROOT / "object_detection_baseline_v1"
            detection_metrics_path = DETECTION_ROOT / "tables" / "object_detection_metrics.csv"

            if RUN_OBJECT_DETECTION and not detection_metrics_path.exists():
                run_command([
                    PYTHON,
                    PROJECT_ROOT / "scripts" / "train_object_detection_baseline.py",
                    "--output-root", DETECTION_ROOT,
                    "--epochs", OBJECT_DETECTION_EPOCHS,
                    "--image-size", OBJECT_DETECTION_IMAGE_SIZE,
                    "--batch-size", OBJECT_DETECTION_BATCH_SIZE,
                    "--resume",
                ])
            elif detection_metrics_path.exists():
                print("Object detection output already exists:", detection_metrics_path)
            else:
                display(Markdown("Object detection long training skipped. Set `RUN_LONG_TRAINING = True` and rerun this cell."))

            display(Markdown("### Object detection metrics"))
            detection_df = display_csv(detection_metrics_path)
            display(Markdown("### TUFTS external detection metrics"))
            display_csv(DETECTION_ROOT / "tables" / "object_detection_tufts_external_metrics.csv")
            show_image(DETECTION_ROOT / "figures" / "object_detection_training_curves.png", width=1000)
            show_image(DETECTION_ROOT / "figures" / "object_detection_pr_curve_iou50.png", width=700)
            show_image(DETECTION_ROOT / "figures" / "object_detection_tufts_external_pr_curve_iou50.png", width=700)
            show_image(DETECTION_ROOT / "figures" / "object_detection_examples.png", width=900)
            """
        ),
        md("## 4. Remaining Task B: Strict Raw Pseudo-Mask Ablation"),
        code(
            r"""
            RAW_INDEX_ROOT = COMPLETE_ROOT / "raw_pseudo_index"
            RAW_PSEUDO_ROOT = PROCESSED_ROOT / "dentex_xray_teacher"
            CLEAN_ROOT = PROCESSED_ROOT / "dentex_xray_teacher_clean"
            RAW_ABLATION_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "strict_raw_ablation"
            RAW_ABLATION_EVAL_ROOT = COMPLETE_ROOT / "strict_raw_ablation_eval"
            RAW_ABLATION_BEST = RAW_ABLATION_CHECKPOINT_ROOT / "student_tinyunet_dentex_raw_pseudo_ablation_best.pt"


            def build_raw_pseudo_index():
                RAW_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
                for task in ["quadrant_full", "enumeration_full", "disease_full", "validation_full"]:
                    raw_rows = read_json(RAW_PSEUDO_ROOT / task / "summary.json")
                    clean_rows = read_json(CLEAN_ROOT / task / "summary.json")
                    clean_by_file = {row["file_name"]: row for row in clean_rows}
                    out_rows = []
                    for row in raw_rows:
                        ref = clean_by_file.get(row["file_name"])
                        if not ref:
                            continue
                        out_rows.append({
                            "file_name": row["file_name"],
                            "image_path": ref["image_path"],
                            "raw_mask": row["mask"],
                            "clean_mask": row["mask"],
                            "mean_confidence": row.get("mean_confidence", 0.0),
                            "status": "kept",
                        })
                    task_dir = RAW_INDEX_ROOT / task
                    task_dir.mkdir(parents=True, exist_ok=True)
                    (task_dir / "summary.json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
                return RAW_INDEX_ROOT


            if RUN_STRICT_RAW_ABLATION and not RAW_ABLATION_BEST.exists():
                build_raw_pseudo_index()
                run_command([
                    PYTHON,
                    PROJECT_ROOT / "scripts" / "train_dentex_student_segmentation.py",
                    "--clean-root", RAW_INDEX_ROOT,
                    "--checkpoint-root", RAW_ABLATION_CHECKPOINT_ROOT,
                    "--run-name", "student_tinyunet_dentex_raw_pseudo_ablation",
                    "--epochs", RAW_ABLATION_EPOCHS,
                    "--image-size", RAW_ABLATION_IMAGE_SIZE,
                    "--batch-size", RAW_ABLATION_BATCH_SIZE,
                    "--resume",
                ])
            elif RAW_ABLATION_BEST.exists():
                print("Raw pseudo-mask ablation checkpoint already exists:", RAW_ABLATION_BEST)
            else:
                display(Markdown("Raw pseudo-mask ablation skipped. Set `RUN_LONG_TRAINING = True` and rerun this cell."))

            if RAW_ABLATION_BEST.exists() and not (RAW_ABLATION_EVAL_ROOT / "tables" / "final_metrics_summary.csv").exists():
                run_command([
                    PYTHON,
                    PROJECT_ROOT / "scripts" / "generate_final_evaluation_xai.py",
                    "--student-checkpoint", RAW_ABLATION_BEST,
                    "--xray-teacher-checkpoint", CHECKPOINT_ROOT / "teacher_tinyunet_tufts_xray_full.pt",
                    "--clean-root", CLEAN_ROOT,
                    "--output-root", RAW_ABLATION_EVAL_ROOT,
                ])

            display(Markdown("### Strict raw-pseudo ablation metrics"))
            display_csv(RAW_ABLATION_EVAL_ROOT / "tables" / "final_metrics_summary.csv")
            show_image(RAW_ABLATION_EVAL_ROOT / "figures" / "student_training_curves_final.png", width=1000)
            """
        ),
        md("## 5. Remaining Task C: LIME-Style and KernelSHAP-Style XAI"),
        code(
            r"""
            LIME_SHAP_ROOT = COMPLETE_ROOT / "lime_shap_xai_v1"
            lime_summary_path = LIME_SHAP_ROOT / "tables" / "lime_shap_xai_summary.csv"

            if RUN_LIME_SHAP_XAI and not lime_summary_path.exists():
                run_command([
                    PYTHON,
                    PROJECT_ROOT / "scripts" / "generate_lime_shap_xai.py",
                    "--checkpoint", CHECKPOINT_ROOT / "student_tinyunet_dentex_pseudo_full_best.pt",
                    "--clean-root", CLEAN_ROOT,
                    "--output-root", LIME_SHAP_ROOT,
                    "--examples", LIME_SHAP_EXAMPLES,
                    "--samples", LIME_SHAP_SAMPLES,
                    "--segments", LIME_SHAP_SEGMENTS,
                    "--xai-size", LIME_SHAP_SIZE,
                ])
            elif lime_summary_path.exists():
                print("LIME/KernelSHAP-style output already exists:", lime_summary_path)
            else:
                display(Markdown("LIME/KernelSHAP-style XAI skipped. Set `RUN_LIME_SHAP_XAI = True` and rerun this cell."))

            display_csv(lime_summary_path)
            show_image(LIME_SHAP_ROOT / "figures" / "dentex_lime_shap_panels.png", width=1100)
            show_image(LIME_SHAP_ROOT / "figures" / "tufts_lime_shap_panels.png", width=1100)
            """
        ),
        md("## 6. Ablation and Baseline Summary"),
        code(
            r"""
            display(Markdown("### Existing ablation comparison"))
            display_csv(RESULT_ROOT / "ablation_comparison_v1" / "tables" / "teacher_vs_student.csv")
            display_csv(RESULT_ROOT / "ablation_comparison_v1" / "tables" / "raw_vs_clean_pseudo_masks.csv")
            display_csv(RESULT_ROOT / "ablation_comparison_v1" / "tables" / "uncertainty_filtering_curve.csv")
            display_csv(RESULT_ROOT / "ablation_comparison_v1" / "tables" / "baseline_vs_proposed_summary.csv")
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig7_ablation_summary.png", width=1000)
            show_image(RESULT_ROOT / "paper_package_v1" / "figures" / "Fig8_pseudo_mask_cleanup.png", width=900)
            """
        ),
        md("## 7. Final Research Completion Dashboard"),
        code(
            r"""
            rows = [
                {"process": "Full DENTEX student segmentation training", "status": "done", "evidence": f"25 epochs, best val Dice {summary['student_best_val_dice_from_training']:.4f}"},
                {"process": "TUFTS external validation", "status": "done", "evidence": f"non-empty Dice {summary['tufts_nonempty_mean_dice']:.4f}, IoU {summary['tufts_nonempty_mean_iou']:.4f}"},
                {"process": "Existing saliency/uncertainty XAI", "status": "done", "evidence": "Fig4-Fig6 in paper package"},
                {"process": "Object detection mAP baseline", "status": "done" if detection_metrics_path.exists() else "pending", "evidence": str(detection_metrics_path)},
                {"process": "Strict raw pseudo-mask ablation", "status": "done" if RAW_ABLATION_BEST.exists() else "pending", "evidence": str(RAW_ABLATION_BEST)},
                {"process": "LIME/KernelSHAP-style XAI", "status": "done" if lime_summary_path.exists() else "pending", "evidence": str(lime_summary_path)},
                {"process": "Training-time memory logging", "status": "partial", "evidence": "Synthetic AMP benchmark exists; original long-run VRAM was not historically logged"},
            ]
            dashboard = pd.DataFrame(rows)
            dashboard_path = COMPLETE_ROOT / "research_completion_dashboard.csv"
            dashboard.to_csv(dashboard_path, index=False)
            display(dashboard)
            print("Dashboard saved:", dashboard_path)

            display(Markdown("### Notebook output root"))
            display(Markdown(f"`{COMPLETE_ROOT}`"))
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
