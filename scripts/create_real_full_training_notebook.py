from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Dental_AI_REAL_FULL_TRAINING_MULTICLASS_POWER_XAI_v1.ipynb"
FALLBACK_NOTEBOOK_PATH = Path("E:/dental_ai_results/Dental_AI_REAL_FULL_TRAINING_MULTICLASS_POWER_XAI_v1.ipynb")


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            """
            # Dental AI Real Full Training, Multiclass Detection, Power Logging, and XAI

            This is the single notebook for the serious rerun.

            It is intentionally not a fast report loader. With the default settings it starts a fresh run from epoch 1,
            trains the TUFTS X-ray teacher, generates and cleans DENTEX pseudo masks, trains the DENTEX student on the
            full selected DENTEX tasks, evaluates TUFTS external test data, trains a multiclass DENTEX enumeration
            detector, logs GPU power/utilization/VRAM per epoch, and then generates XAI/paper figures.

            Honest scope: the previous saved result is useful evidence, but it is not enough to promise journal
            acceptance. This notebook is built so the new run has the missing evidence your sir asked for.
            """
        ),
        md("## 1. Run Controls"),
        code(
            r"""
            from pathlib import Path
            import json
            import os
            import shutil
            import subprocess
            import sys
            import time
            from datetime import datetime

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import Image as IPyImage
            from IPython.display import Markdown, display

            PROJECT_ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
            DATA_ROOT = PROJECT_ROOT / "data"
            COCO_ROOT = DATA_ROOT / "processed" / "coco"
            DENTEX_ROOT = DATA_ROOT / "raw" / "dentex"
            TUFTS_ROOT = DATA_ROOT / "raw" / "tufts"

            PYTHON_EXE = Path(sys.executable)
            if not PYTHON_EXE.exists():
                PYTHON_EXE = Path(r"H:\python\python.exe")

            # Default is a real fresh run, not a saved-result loader.
            FORCE_FRESH_FROM_EPOCH1 = True
            RUN_ID = "real_full_multiclass_power_" + datetime.now().strftime("%Y%m%d_%H%M%S")

            TEACHER_EPOCHS = 50
            STUDENT_EPOCHS = 50
            DETECTOR_EPOCHS = 20
            IMAGE_SIZE = 512
            TEACHER_BATCH_SIZE = 2
            STUDENT_BATCH_SIZE = 4
            DETECTOR_BATCH_SIZE = 2
            NUM_WORKERS = 0

            RUN_TUFTS_TEACHER = True
            RUN_DENTEX_PSEUDO_MASKS = True
            RUN_DENTEX_STUDENT = True
            RUN_MULTICLASS_DETECTOR = True
            RUN_FINAL_EVALUATION = True
            RUN_LIME_SHAP_XAI = True
            RUN_ABLATION_AND_PAPER_PACKAGE = True

            CHECKPOINT_ROOT = Path("E:/dental_ai_checkpoints") / RUN_ID
            RESULT_ROOT = Path("E:/dental_ai_results") / RUN_ID
            PSEUDO_ROOT = Path("E:/dental_ai_processed/pseudo_masks") / RUN_ID
            RAW_PSEUDO_ROOT = PSEUDO_ROOT / "dentex_xray_teacher"
            CLEAN_PSEUDO_ROOT = PSEUDO_ROOT / "dentex_xray_teacher_clean"

            FINAL_EVAL_ROOT = RESULT_ROOT / "final_eval_xai_v1"
            XAI_ROOT = RESULT_ROOT / "lime_shap_xai_v1"
            ABLATION_ROOT = RESULT_ROOT / "ablation_comparison_v1"
            PAPER_ROOT = RESULT_ROOT / "paper_package_v1"
            AUDIT_ROOT = RESULT_ROOT / "research_audit"
            DETECTOR_ROOT = RESULT_ROOT / "multiclass_detection_enumeration_pretrained_coco_v1"

            TEACHER_CKPT = CHECKPOINT_ROOT / "teacher_tinyunet_tufts_xray_full.pt"
            STUDENT_RUN_NAME = "student_tinyunet_dentex_pseudo_full"
            STUDENT_CKPT = CHECKPOINT_ROOT / f"{STUDENT_RUN_NAME}.pt"
            STUDENT_BEST_CKPT = CHECKPOINT_ROOT / f"{STUDENT_RUN_NAME}_best.pt"

            SCRIPTS = {
                "teacher": PROJECT_ROOT / "scripts" / "train_xray_teacher_segmentation.py",
                "pseudo": PROJECT_ROOT / "scripts" / "generate_dentex_pseudo_masks.py",
                "clean": PROJECT_ROOT / "scripts" / "clean_dentex_pseudo_masks.py",
                "student": PROJECT_ROOT / "scripts" / "train_dentex_student_segmentation.py",
                "detector": PROJECT_ROOT / "scripts" / "train_object_detection_baseline.py",
                "final_eval": PROJECT_ROOT / "scripts" / "generate_final_evaluation_xai.py",
                "lime_shap": PROJECT_ROOT / "scripts" / "generate_lime_shap_xai.py",
                "ablation": PROJECT_ROOT / "scripts" / "generate_ablation_comparison.py",
                "paper": PROJECT_ROOT / "scripts" / "generate_paper_package.py",
                "audit": PROJECT_ROOT / "scripts" / "generate_full_research_audit.py",
            }

            for folder in [CHECKPOINT_ROOT, RESULT_ROOT, RAW_PSEUDO_ROOT, CLEAN_PSEUDO_ROOT]:
                folder.mkdir(parents=True, exist_ok=True)

            print("Python:", PYTHON_EXE)
            print("Run id:", RUN_ID)
            print("Checkpoints:", CHECKPOINT_ROOT)
            print("Results:", RESULT_ROOT)
            print("Fresh from epoch 1:", FORCE_FRESH_FROM_EPOCH1)
            """
        ),
        md("## 2. Helpers: Streaming Commands, GPU Snapshot, Tables, Figures"),
        code(
            r"""
            def run_stream(cmd, cwd=PROJECT_ROOT):
                cmd = [str(part) for part in cmd]
                env = os.environ.copy()
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env.setdefault("TORCH_HOME", "E:/torch_cache")
                print("\nRUN:", " ".join(cmd), flush=True)
                start = time.perf_counter()
                process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                rc = process.wait()
                seconds = time.perf_counter() - start
                print(f"\nDONE rc={rc} seconds={seconds:.1f}", flush=True)
                if rc != 0:
                    raise RuntimeError(f"Command failed with return code {rc}")
                return seconds


            def gpu_snapshot():
                if shutil.which("nvidia-smi") is None:
                    display(Markdown("`nvidia-smi` not found. GPU power logging will be unavailable."))
                    return pd.DataFrame()
                query = "name,memory.total,memory.used,utilization.gpu,power.draw,power.limit,temperature.gpu"
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
                rows = []
                for line in out.splitlines():
                    name, mem_total, mem_used, util, power, power_limit, temp = [part.strip() for part in line.split(",")]
                    rows.append({
                        "gpu": name,
                        "memory_total_mb": float(mem_total),
                        "memory_used_mb": float(mem_used),
                        "util_percent": float(util),
                        "power_draw_w": float(power),
                        "power_limit_w": float(power_limit),
                        "temperature_c": float(temp),
                    })
                df = pd.DataFrame(rows)
                display(df)
                return df


            def read_json(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))


            def show_table(path, n=30, title=None):
                path = Path(path)
                if title:
                    display(Markdown(f"### {title}"))
                if not path.exists():
                    display(Markdown(f"Missing table: `{path}`"))
                    return pd.DataFrame()
                df = pd.read_csv(path)
                display(df.head(n))
                return df


            def show_image(path, width=1000, title=None):
                path = Path(path)
                if title:
                    display(Markdown(f"### {title}"))
                if path.exists():
                    display(IPyImage(filename=str(path), width=width))
                else:
                    display(Markdown(f"Missing image: `{path}`"))


            def plot_history(df, title):
                if df.empty:
                    return
                fig, axes = plt.subplots(2, 2, figsize=(14, 8))
                axes = axes.ravel()
                if {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
                    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="train")
                    axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="validation")
                    if "test_loss" in df:
                        axes[0].plot(df["epoch"], df["test_loss"], marker="o", label="test")
                    axes[0].set_title("Loss")
                    axes[0].legend()
                for metric in ["dice", "iou", "accuracy"]:
                    col_id = {"dice": 1, "iou": 2, "accuracy": 3}[metric]
                    for split in ["train", "val", "test"]:
                        col = f"{split}_{metric}"
                        if col in df:
                            axes[col_id].plot(df["epoch"], df[col], marker="o", label=split)
                    axes[col_id].set_title(metric.upper())
                    axes[col_id].set_ylim(0, 1)
                    axes[col_id].legend()
                fig.suptitle(title)
                plt.tight_layout()
                plt.show()


            gpu_snapshot()
            """
        ),
        md("## 3. Dataset Inventory and Multiclass Labels"),
        code(
            r"""
            archive_rows = []
            for path in [
                PROJECT_ROOT / "DATASET" / "7812323.zip",
                PROJECT_ROOT / "DATASET" / "dentalai-DatasetNinja.tar",
                PROJECT_ROOT / "DATASET" / "tufts-dental-database.zip",
            ]:
                archive_rows.append({
                    "archive": path.name,
                    "exists": path.exists(),
                    "size_gb": round(path.stat().st_size / (1024 ** 3), 3) if path.exists() else np.nan,
                })
            display(pd.DataFrame(archive_rows))

            dataset_rows = []
            coco_files = {
                "DENTEX quadrant": COCO_ROOT / "dentex" / "quadrant.json",
                "DENTEX enumeration/multiclass detection": COCO_ROOT / "dentex" / "enumeration.json",
                "DENTEX disease": COCO_ROOT / "dentex" / "disease.json",
                "DENTEX validation images": COCO_ROOT / "dentex" / "validation_images.json",
                "DentalAI train": COCO_ROOT / "dentalai" / "train.json",
                "DentalAI validation": COCO_ROOT / "dentalai" / "valid.json",
                "DentalAI test": COCO_ROOT / "dentalai" / "test.json",
                "TUFTS teeth external": COCO_ROOT / "tufts_teeth.json",
            }
            for name, path in coco_files.items():
                data = read_json(path)
                dataset_rows.append({
                    "dataset": name,
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                    "categories": ", ".join(str(c.get("name")) for c in data.get("categories", [])),
                })
            display(pd.DataFrame(dataset_rows))
            """
        ),
        md("## 4. Train TUFTS X-ray Teacher: Full Epochs + Power/VRAM Log"),
        code(
            r"""
            if RUN_TUFTS_TEACHER:
                cmd = [
                    PYTHON_EXE, SCRIPTS["teacher"],
                    "--mode", "full",
                    "--epochs", TEACHER_EPOCHS,
                    "--image-size", IMAGE_SIZE,
                    "--batch-size", TEACHER_BATCH_SIZE,
                    "--num-workers", NUM_WORKERS,
                    "--checkpoint-root", CHECKPOINT_ROOT,
                    "--gpu-sample-interval", "2",
                ]
                if not FORCE_FRESH_FROM_EPOCH1:
                    cmd.append("--resume")
                run_stream(cmd)
            teacher_hist = show_table(CHECKPOINT_ROOT / "teacher_tinyunet_tufts_xray_full_epoch_history.csv", n=TEACHER_EPOCHS, title="TUFTS teacher epoch history with power")
            plot_history(teacher_hist, "TUFTS X-ray Teacher")
            gpu_snapshot()
            """
        ),
        md("## 5. Generate Full DENTEX Pseudo Masks"),
        code(
            r"""
            DENTEX_TASKS = {
                "quadrant_full": COCO_ROOT / "dentex" / "quadrant.json",
                "enumeration_full": COCO_ROOT / "dentex" / "enumeration.json",
                "disease_full": COCO_ROOT / "dentex" / "disease.json",
                "validation_full": COCO_ROOT / "dentex" / "validation_images.json",
            }
            if RUN_DENTEX_PSEUDO_MASKS:
                for task, coco_path in DENTEX_TASKS.items():
                    run_stream([
                        PYTHON_EXE, SCRIPTS["pseudo"],
                        "--checkpoint", TEACHER_CKPT,
                        "--coco", coco_path,
                        "--dentex-root", DENTEX_ROOT,
                        "--output", RAW_PSEUDO_ROOT / task,
                        "--image-size", IMAGE_SIZE,
                        "--max-images", "0",
                        "--progress-every", "50",
                        "--preview",
                    ])

            raw_counts = []
            for task in DENTEX_TASKS:
                summary_path = RAW_PSEUDO_ROOT / task / "summary.json"
                raw_counts.append({"task": task, "summary_exists": summary_path.exists(), "rows": len(read_json(summary_path)) if summary_path.exists() else 0})
            display(pd.DataFrame(raw_counts))
            """
        ),
        md("## 6. Clean Pseudo Masks and Show Mask Quality"),
        code(
            r"""
            run_stream([
                PYTHON_EXE, SCRIPTS["clean"],
                "--input-root", RAW_PSEUDO_ROOT,
                "--output-root", CLEAN_PSEUDO_ROOT,
                "--dentex-root", DENTEX_ROOT,
                "--tasks", "quadrant_full", "enumeration_full", "disease_full", "validation_full",
                "--preview",
            ])
            clean_counts = []
            for task in DENTEX_TASKS:
                summary_path = CLEAN_PSEUDO_ROOT / task / "summary.json"
                rows = read_json(summary_path) if summary_path.exists() else []
                clean_counts.append({
                    "task": task,
                    "rows": len(rows),
                    "kept": sum(1 for row in rows if row.get("status") == "kept"),
                    "rejected": sum(1 for row in rows if row.get("status") != "kept"),
                })
            display(pd.DataFrame(clean_counts))
            for task in DENTEX_TASKS:
                preview_paths = sorted((CLEAN_PSEUDO_ROOT / task / "previews").glob("*_clean_overlay.png"))[:4]
                for preview_path in preview_paths:
                    show_image(preview_path, width=520, title=f"{task}: {preview_path.name}")
            """
        ),
        md("## 7. Train DENTEX Student on Full Selected DENTEX Tasks + TUFTS Test Every Epoch"),
        code(
            r"""
            if RUN_DENTEX_STUDENT:
                cmd = [
                    PYTHON_EXE, SCRIPTS["student"],
                    "--clean-root", CLEAN_PSEUDO_ROOT,
                    "--checkpoint-root", CHECKPOINT_ROOT,
                    "--train-tasks", "quadrant_full", "enumeration_full", "disease_full",
                    "--val-tasks", "validation_full",
                    "--epochs", STUDENT_EPOCHS,
                    "--image-size", IMAGE_SIZE,
                    "--batch-size", STUDENT_BATCH_SIZE,
                    "--num-workers", NUM_WORKERS,
                    "--width", "16",
                    "--learning-rate", "0.001",
                    "--run-name", STUDENT_RUN_NAME,
                    "--gpu-sample-interval", "2",
                    "--tufts-test-root", TUFTS_ROOT,
                    "--test-every-epoch",
                ]
                if not FORCE_FRESH_FROM_EPOCH1:
                    cmd.append("--resume")
                run_stream(cmd)
            student_hist = show_table(CHECKPOINT_ROOT / f"{STUDENT_RUN_NAME}_epoch_history.csv", n=STUDENT_EPOCHS, title="DENTEX student epoch history with train/validation/TUFTS test/power")
            plot_history(student_hist, "DENTEX Student: Train, Validation, TUFTS Test")
            gpu_snapshot()
            """
        ),
        md("## 8. Multiclass DENTEX Enumeration Object Detection"),
        code(
            r"""
            if RUN_MULTICLASS_DETECTOR:
                run_stream([
                    PYTHON_EXE, SCRIPTS["detector"],
                    "--coco-json", COCO_ROOT / "dentex" / "enumeration.json",
                    "--image-root", DENTEX_ROOT,
                    "--external-coco-json", COCO_ROOT / "tufts_teeth.json",
                    "--external-image-root", TUFTS_ROOT,
                    "--output-root", DETECTOR_ROOT,
                    "--epochs", DETECTOR_EPOCHS,
                    "--batch-size", DETECTOR_BATCH_SIZE,
                    "--image-size", IMAGE_SIZE,
                    "--num-workers", NUM_WORKERS,
                    "--learning-rate", "0.0002",
                    "--pretrained-coco",
                    "--score-threshold", "0.35",
                    "--gpu-sample-interval", "2",
                ] + ([] if FORCE_FRESH_FROM_EPOCH1 else ["--resume"]))

            det_hist = show_table(DETECTOR_ROOT / "tables" / "object_detection_epoch_history.csv", n=DETECTOR_EPOCHS, title="Multiclass detector epoch history with mAP and power")
            show_table(DETECTOR_ROOT / "tables" / "object_detection_metrics.csv", title="Internal DENTEX enumeration detection metrics")
            show_table(DETECTOR_ROOT / "tables" / "object_detection_tufts_external_metrics.csv", title="TUFTS external class-agnostic box metrics")
            show_table(DETECTOR_ROOT / "tables" / "object_detection_multiclass_per_class_iou50.csv", title="Multiclass per-class detection metrics at IoU 0.50")
            show_table(DETECTOR_ROOT / "tables" / "object_detection_multiclass_confusion_iou50.csv", n=20, title="Multiclass detection confusion matrix at IoU 0.50")
            show_image(DETECTOR_ROOT / "figures" / "object_detection_training_curves.png", width=1000, title="Detector training curves")
            show_image(DETECTOR_ROOT / "figures" / "object_detection_examples.png", width=800, title="Detector prediction examples")
            gpu_snapshot()
            """
        ),
        md("## 9. Final DENTEX Validation and TUFTS External Test Evaluation"),
        code(
            r"""
            if RUN_FINAL_EVALUATION:
                run_stream([
                    PYTHON_EXE, SCRIPTS["final_eval"],
                    "--student-checkpoint", STUDENT_BEST_CKPT,
                    "--xray-teacher-checkpoint", TEACHER_CKPT,
                    "--clean-root", CLEAN_PSEUDO_ROOT,
                    "--tufts-root", TUFTS_ROOT,
                    "--output-root", FINAL_EVAL_ROOT,
                    "--pixel-sample-per-image", "2000",
                    "--max-tufts", "0",
                ])
            summary_json = FINAL_EVAL_ROOT / "tables" / "final_metrics_summary.json"
            if summary_json.exists():
                summary = read_json(summary_json)
                display(pd.DataFrame([summary]).T.rename(columns={0: "value"}))
            show_table(FINAL_EVAL_ROOT / "tables" / "dentex_validation_student_eval.csv", n=10, title="DENTEX pseudo-validation rows")
            show_table(FINAL_EVAL_ROOT / "tables" / "tufts_external_student_eval_nonempty.csv", n=10, title="TUFTS external non-empty rows")
            show_image(FINAL_EVAL_ROOT / "figures" / "segmentation_confusion_matrices.png", width=950, title="Binary segmentation confusion matrices")
            show_image(FINAL_EVAL_ROOT / "figures" / "pixel_level_roc_curves.png", width=800, title="Pixel ROC curves")
            show_image(FINAL_EVAL_ROOT / "figures" / "student_training_curves_final.png", width=1000, title="Student final training curves")
            show_image(FINAL_EVAL_ROOT / "figures" / "tufts_failure_case_panels.png", width=1100, title="TUFTS failure cases")
            """
        ),
        md("## 10. XAI: Saliency, LIME-Style, KernelSHAP-Style, Uncertainty"),
        code(
            r"""
            show_image(FINAL_EVAL_ROOT / "figures" / "dentex_xai_uncertainty_panels.png", width=1100, title="DENTEX saliency and uncertainty")
            show_image(FINAL_EVAL_ROOT / "figures" / "tufts_xai_uncertainty_panels.png", width=1100, title="TUFTS saliency and uncertainty")
            if RUN_LIME_SHAP_XAI:
                run_stream([
                    PYTHON_EXE, SCRIPTS["lime_shap"],
                    "--checkpoint", STUDENT_BEST_CKPT,
                    "--clean-root", CLEAN_PSEUDO_ROOT,
                    "--tufts-root", TUFTS_ROOT,
                    "--output-root", XAI_ROOT,
                    "--examples", "4",
                    "--samples", "96",
                    "--segments", "64",
                    "--xai-size", "256",
                ])
            xai_df = show_table(XAI_ROOT / "tables" / "lime_shap_xai_summary.csv", title="LIME/KernelSHAP-style numeric faithfulness table")
            show_image(XAI_ROOT / "figures" / "dentex_lime_shap_panels.png", width=1100, title="DENTEX LIME/KernelSHAP-style panels")
            show_image(XAI_ROOT / "figures" / "tufts_lime_shap_panels.png", width=1100, title="TUFTS LIME/KernelSHAP-style panels")
            """
        ),
        md("## 11. Ablation, Paper Package, and Full Research Audit"),
        code(
            r"""
            if RUN_ABLATION_AND_PAPER_PACKAGE:
                run_stream([
                    PYTHON_EXE, SCRIPTS["ablation"],
                    "--output-root", ABLATION_ROOT,
                    "--raw-pseudo-root", RAW_PSEUDO_ROOT,
                    "--clean-root", CLEAN_PSEUDO_ROOT,
                    "--final-eval-root", FINAL_EVAL_ROOT,
                    "--teacher-checkpoint", TEACHER_CKPT,
                    "--student-checkpoint", STUDENT_BEST_CKPT,
                    "--image-size", IMAGE_SIZE,
                    "--benchmark-images", "24",
                ])
                run_stream([
                    PYTHON_EXE, SCRIPTS["paper"],
                    "--output-root", PAPER_ROOT,
                    "--final-eval-root", FINAL_EVAL_ROOT,
                    "--ablation-root", ABLATION_ROOT,
                    "--detector-root", DETECTOR_ROOT,
                ])
                run_stream([
                    PYTHON_EXE, SCRIPTS["audit"],
                    "--run-id", RUN_ID,
                    "--output-root", AUDIT_ROOT,
                ])
            show_table(ABLATION_ROOT / "tables" / "teacher_vs_student.csv", title="Teacher vs student")
            show_table(ABLATION_ROOT / "tables" / "raw_vs_clean_pseudo_masks.csv", title="Raw vs clean pseudo masks")
            show_table(ABLATION_ROOT / "tables" / "uncertainty_filtering_curve.csv", title="Uncertainty filtering")
            show_table(PAPER_ROOT / "tables" / "Table2_main_results.csv", title="Paper main results table")
            show_table(PAPER_ROOT / "tables" / "Table8_multiclass_detection_results.csv", title="Paper multiclass detection table")
            show_table(PAPER_ROOT / "tables" / "Table10_multiclass_per_class_iou50.csv", title="Paper multiclass per-class table")
            show_table(AUDIT_ROOT / "tables" / "segmentation_binary_class_metrics.csv", title="Per-class segmentation table")
            """
        ),
        md("## 12. Final Readiness Check: Do Not Overclaim"),
        code(
            r"""
            readiness_rows = []
            summary_path = FINAL_EVAL_ROOT / "tables" / "final_metrics_summary.json"
            if summary_path.exists():
                summary = read_json(summary_path)
                readiness_rows.extend([
                    {"claim": "DENTEX pseudo-validation segmentation", "metric": "Dice", "value": summary.get("dentex_val_mean_dice"), "target_for_strong_paper": ">= 0.90"},
                    {"claim": "TUFTS external segmentation", "metric": "non-empty Dice", "value": summary.get("tufts_nonempty_mean_dice"), "target_for_strong_paper": ">= 0.85 to 0.90"},
                    {"claim": "TUFTS external segmentation", "metric": "non-empty IoU", "value": summary.get("tufts_nonempty_mean_iou"), "target_for_strong_paper": ">= 0.75 to 0.80"},
                    {"claim": "Pixel ranking", "metric": "TUFTS ROC AUC", "value": summary.get("tufts_pixel_roc_auc"), "target_for_strong_paper": ">= 0.95"},
                ])
            det_path = DETECTOR_ROOT / "tables" / "object_detection_metrics.csv"
            if det_path.exists():
                det = pd.read_csv(det_path).iloc[0].to_dict()
                readiness_rows.extend([
                    {"claim": "Multiclass DENTEX detection", "metric": "mAP@50", "value": det.get("map_50"), "target_for_strong_paper": ">= 0.70 preferred"},
                    {"claim": "Multiclass DENTEX detection", "metric": "mAP@50:95", "value": det.get("map_50_95"), "target_for_strong_paper": ">= 0.40 preferred"},
                ])
            external_det_path = DETECTOR_ROOT / "tables" / "object_detection_tufts_external_metrics.csv"
            if external_det_path.exists():
                ext = pd.read_csv(external_det_path).iloc[0].to_dict()
                readiness_rows.append({"claim": "TUFTS external detection", "metric": "mAP@50", "value": ext.get("map_50"), "target_for_strong_paper": ">= 0.50 preferred"})
            readiness = pd.DataFrame(readiness_rows)
            display(readiness)

            display(Markdown(
                "**Decision rule:** if detection mAP and XAI faithfulness stay weak, write the paper as "
                "teacher-student pseudo-mask tooth segmentation with external TUFTS validation and uncertainty/XAI. "
                "Do not sell it as a strong multiclass detector or disease diagnosis paper unless the metrics above improve."
            ))
            print("All outputs for this run:", RESULT_ROOT)
            print("All checkpoints for this run:", CHECKPOINT_ROOT)
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
