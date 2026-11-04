from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
PRIMARY_NOTEBOOK_PATH = ROOT / "notebooks" / "Dental_AI_Full_Dataset_Train_Test_Validation_v1.ipynb"
FALLBACK_NOTEBOOK_PATH = Path(
    r"E:\dental_ai_results\full_dataset_train_test_validation_v1\Dental_AI_Full_Dataset_Train_Test_Validation_v1.ipynb"
)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# Dental AI Full Dataset Train, Test, and Validation v1

This is the clean evidence notebook for the full run:

- DENTEX full dataset: pseudo-mask generation and student training.
- DentalAI 1GB dataset: RGB mask reference / modality-gap baseline, not the final X-ray validation set.
- TUFTS dataset: X-ray teacher source and external test/validation.
- Final outputs: DENTEX pseudo-validation, TUFTS external validation, XAI, uncertainty, failure panels, and paper package.

Important: the default mode reuses existing checkpoints. That means a "Run All" can finish quickly because it is verifying completed training, not retraining 12GB from zero. To start a real fresh long run, set `TRAINING_MODE = "fresh_rerun"` in the first code cell.
"""
    ),
    md("## 1. Setup and Run Controls"),
    code(
        r"""
from pathlib import Path
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
COCO_ROOT = DATA_ROOT / "processed" / "coco"

CHECKPOINT_ROOT = Path(r"E:\dental_ai_checkpoints")
PROCESSED_ROOT = Path(r"E:\dental_ai_processed")
RESULT_ROOT = Path(r"E:\dental_ai_results")

PYTHON_EXE = sys.executable

TRAINING_MODE = "reuse_existing_checkpoints"
# Choices:
# - "reuse_existing_checkpoints": verify/show completed outputs, no long retraining.
# - "resume_missing": train only missing/incomplete long stages, using checkpoint resume.
# - "fresh_rerun": create a new checkpoint/result folder and train from epoch 1.

FRESH_RUN_SUFFIX = datetime.now().strftime("%Y%m%d_%H%M%S")
FRESH_RERUN = TRAINING_MODE == "fresh_rerun"
RUN_LONG_TRAINING_IF_MISSING = TRAINING_MODE in {"resume_missing", "fresh_rerun"}
RUN_DENTALAI_REFERENCE_IF_MISSING = False
RUN_PSEUDO_GENERATION_IF_MISSING = True
RUN_CLEANING_IF_MISSING = True
RUN_FINAL_EVAL_IF_MISSING = True
RUN_ABLATION_IF_MISSING = True
RUN_PAPER_PACKAGE_IF_MISSING = True
FORCE_RETRAIN_OR_REGENERATE = FRESH_RERUN

TEACHER_EPOCHS = 40
STUDENT_EPOCHS = 25
IMAGE_SIZE = 512
TEACHER_BATCH_SIZE = 2
STUDENT_BATCH_SIZE = 4
NUM_WORKERS = 0

ACTIVE_CHECKPOINT_ROOT = CHECKPOINT_ROOT / f"fresh_rerun_{FRESH_RUN_SUFFIX}" if FRESH_RERUN else CHECKPOINT_ROOT
ACTIVE_RESULT_ROOT = RESULT_ROOT / f"fresh_rerun_{FRESH_RUN_SUFFIX}" if FRESH_RERUN else RESULT_ROOT
ACTIVE_PSEUDO_BASE_ROOT = PROCESSED_ROOT / "pseudo_masks" / f"fresh_rerun_{FRESH_RUN_SUFFIX}" if FRESH_RERUN else PROCESSED_ROOT / "pseudo_masks"
RAW_PSEUDO_ROOT = ACTIVE_PSEUDO_BASE_ROOT / "dentex_xray_teacher"
CLEAN_PSEUDO_ROOT = ACTIVE_PSEUDO_BASE_ROOT / "dentex_xray_teacher_clean"
FULL_RESULT_ROOT = ACTIVE_RESULT_ROOT / "full_dataset_train_test_validation_v1"
FINAL_EVAL_ROOT = ACTIVE_RESULT_ROOT / "final_eval_xai_v1"
ABLATION_ROOT = ACTIVE_RESULT_ROOT / "ablation_comparison_v1"
PAPER_PACKAGE_ROOT = ACTIVE_RESULT_ROOT / "paper_package_v1"

DENTALAI_CKPT = ACTIVE_CHECKPOINT_ROOT / "teacher_tinyunet_dentalai_full.pt"
TUFTS_TEACHER_CKPT = ACTIVE_CHECKPOINT_ROOT / "teacher_tinyunet_tufts_xray_full.pt"
STUDENT_RUN_NAME = "student_tinyunet_dentex_pseudo_full"
STUDENT_CKPT = ACTIVE_CHECKPOINT_ROOT / f"{STUDENT_RUN_NAME}.pt"
STUDENT_BEST_CKPT = ACTIVE_CHECKPOINT_ROOT / f"{STUDENT_RUN_NAME}_best.pt"

COCO_TASKS = {
    "quadrant_full": COCO_ROOT / "dentex" / "quadrant.json",
    "enumeration_full": COCO_ROOT / "dentex" / "enumeration.json",
    "disease_full": COCO_ROOT / "dentex" / "disease.json",
    "validation_full": COCO_ROOT / "dentex" / "validation_images.json",
}

SCRIPTS = {
    "dentalai_teacher": PROJECT_ROOT / "scripts" / "train_teacher_segmentation.py",
    "tufts_teacher": PROJECT_ROOT / "scripts" / "train_xray_teacher_segmentation.py",
    "pseudo": PROJECT_ROOT / "scripts" / "generate_dentex_pseudo_masks.py",
    "clean": PROJECT_ROOT / "scripts" / "clean_dentex_pseudo_masks.py",
    "student": PROJECT_ROOT / "scripts" / "train_dentex_student_segmentation.py",
    "final_eval": PROJECT_ROOT / "scripts" / "generate_final_evaluation_xai.py",
    "ablation": PROJECT_ROOT / "scripts" / "generate_ablation_comparison.py",
    "paper": PROJECT_ROOT / "scripts" / "generate_paper_package.py",
}

FULL_RESULT_ROOT.mkdir(parents=True, exist_ok=True)

print("Project:", PROJECT_ROOT)
print("Python:", PYTHON_EXE)
print("Training mode:", TRAINING_MODE)
print("Active checkpoint root:", ACTIVE_CHECKPOINT_ROOT)
print("Active result root:", ACTIVE_RESULT_ROOT)
print("Active pseudo-mask root:", ACTIVE_PSEUDO_BASE_ROOT)
print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
"""
    ),
    md("## 2. Helpers"),
    code(
        r"""
def run_command(cmd, cwd=PROJECT_ROOT):
    cmd = [str(x) for x in cmd]
    print("RUN:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    return result


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def count_coco(path):
    data = read_json(path, {})
    return {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(data.get("categories", [])),
    }


def file_mb(path):
    path = Path(path)
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 2)


def checkpoint_summary(path):
    path = Path(path)
    row = {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": file_mb(path),
        "epoch": None,
        "best_val_dice": None,
        "history_len": None,
        "modified": None,
    }
    if not path.exists():
        return row
    row["modified"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        row["epoch"] = ckpt.get("epoch")
        row["best_val_dice"] = ckpt.get("best_val_dice")
        history = ckpt.get("history", {})
        lengths = [len(v) for v in history.values() if isinstance(v, list)]
        row["history_len"] = max(lengths) if lengths else None
    except Exception as exc:
        row["load_error"] = repr(exc)
    return row


def summary_count(path):
    rows = read_json(path, [])
    return len(rows) if isinstance(rows, list) else 0


def show_image(path, width=950):
    path = Path(path)
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print("Missing image:", path)
"""
    ),
    md("## 3. Dataset Inventory: DENTEX 12GB, DentalAI 1GB, TUFTS"),
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
        "size_gb": None if not path.exists() else round(path.stat().st_size / (1024 ** 3), 3),
    })
display(pd.DataFrame(archive_rows))

coco_rows = []
for label, path in {
    "DENTEX quadrant": COCO_ROOT / "dentex" / "quadrant.json",
    "DENTEX enumeration": COCO_ROOT / "dentex" / "enumeration.json",
    "DENTEX disease": COCO_ROOT / "dentex" / "disease.json",
    "DENTEX validation": COCO_ROOT / "dentex" / "validation_images.json",
    "DentalAI train": COCO_ROOT / "dentalai" / "train.json",
    "DentalAI valid": COCO_ROOT / "dentalai" / "valid.json",
    "DentalAI test": COCO_ROOT / "dentalai" / "test.json",
    "TUFTS teeth": COCO_ROOT / "tufts_teeth.json",
}.items():
    row = {"dataset": label, "json": str(path), "exists": path.exists()}
    row.update(count_coco(path) if path.exists() else {"images": 0, "annotations": 0, "categories": 0})
    coco_rows.append(row)
display(pd.DataFrame(coco_rows))
"""
    ),
    md(
        """
## 4. Checkpoint Status

If these checkpoints already exist, the default notebook mode verifies and displays completed results. This is why rerunning the notebook can take minutes. For a real fresh long run, change `TRAINING_MODE` to `"fresh_rerun"` before running cells.
"""
    ),
    code(
        r"""
ckpt_df = pd.DataFrame([
    {"stage": "DentalAI RGB reference teacher", **checkpoint_summary(DENTALAI_CKPT)},
    {"stage": "TUFTS X-ray teacher", **checkpoint_summary(TUFTS_TEACHER_CKPT)},
    {"stage": "DENTEX full pseudo-mask student", **checkpoint_summary(STUDENT_CKPT)},
    {"stage": "DENTEX best student", **checkpoint_summary(STUDENT_BEST_CKPT)},
])
display(ckpt_df)
"""
    ),
    md(
        """
## 5. Optional DentalAI 1GB RGB Reference Teacher

DentalAI has real masks, but it is RGB intraoral imagery. It is useful as a reference/modality-gap baseline, not as the final DENTEX panoramic X-ray validation set. The final teacher used for DENTEX pseudo masks is TUFTS X-ray because it matches the panoramic X-ray domain.
"""
    ),
    code(
        r"""
if DENTALAI_CKPT.exists() and not FORCE_RETRAIN_OR_REGENERATE:
    print("DentalAI reference teacher checkpoint exists; skipping:", DENTALAI_CKPT)
elif RUN_DENTALAI_REFERENCE_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["dentalai_teacher"],
        "--mode", "full",
        "--epochs", "30",
        "--image-size", str(IMAGE_SIZE),
        "--batch-size", "3",
        "--num-workers", str(NUM_WORKERS),
        "--checkpoint-root", ACTIVE_CHECKPOINT_ROOT,
    ] + ([] if FRESH_RERUN else ["--resume"]))
else:
    print("DentalAI reference training disabled. Set RUN_DENTALAI_REFERENCE_IF_MISSING=True to run it.")
display(pd.DataFrame([checkpoint_summary(DENTALAI_CKPT)]))
"""
    ),
    md("## 6. TUFTS X-ray Teacher Training"),
    code(
        r"""
if TUFTS_TEACHER_CKPT.exists() and not FORCE_RETRAIN_OR_REGENERATE:
    print("TUFTS X-ray teacher checkpoint exists; skipping:", TUFTS_TEACHER_CKPT)
elif RUN_LONG_TRAINING_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["tufts_teacher"],
        "--mode", "full",
        "--epochs", str(TEACHER_EPOCHS),
        "--image-size", str(IMAGE_SIZE),
        "--batch-size", str(TEACHER_BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--checkpoint-root", ACTIVE_CHECKPOINT_ROOT,
    ] + ([] if FRESH_RERUN else ["--resume"]))
else:
    print("TUFTS teacher checkpoint missing, but long training is disabled.")
    print("Set RUN_LONG_TRAINING_IF_MISSING=True to start/resume full teacher training.")
display(pd.DataFrame([checkpoint_summary(TUFTS_TEACHER_CKPT)]))
"""
    ),
    md("## 7. Full DENTEX Pseudo-mask Generation From the TUFTS Teacher"),
    code(
        r"""
expected_counts = {
    "quadrant_full": 693,
    "enumeration_full": 634,
    "disease_full": 705,
    "validation_full": 50,
}

for task, coco_path in COCO_TASKS.items():
    output_dir = RAW_PSEUDO_ROOT / task
    current = summary_count(output_dir / "summary.json")
    complete = current >= expected_counts[task]
    if complete and not FORCE_RETRAIN_OR_REGENERATE:
        print(f"{task}: raw pseudo masks complete ({current}/{expected_counts[task]})")
        continue
    if RUN_PSEUDO_GENERATION_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
        run_command([
            PYTHON_EXE, SCRIPTS["pseudo"],
            "--checkpoint", TUFTS_TEACHER_CKPT,
            "--coco", coco_path,
            "--dentex-root", RAW_ROOT / "dentex",
            "--output", output_dir,
            "--max-images", "0",
            "--image-size", str(IMAGE_SIZE),
            "--preview",
            "--progress-every", "100",
        ])
    else:
        print(f"{task}: pseudo masks missing/incomplete and generation disabled ({current}/{expected_counts[task]})")

raw_rows = []
for task in COCO_TASKS:
    raw_rows.append({
        "task": task,
        "expected": expected_counts[task],
        "summary_rows": summary_count(RAW_PSEUDO_ROOT / task / "summary.json"),
    })
display(pd.DataFrame(raw_rows))
"""
    ),
    md("## 8. Clean and Quality-gate DENTEX Pseudo Masks"),
    code(
        r"""
def cleaned_complete():
    if FORCE_RETRAIN_OR_REGENERATE:
        return False
    for task, expected in expected_counts.items():
        if summary_count(CLEAN_PSEUDO_ROOT / task / "summary.json") < expected:
            return False
    return (CLEAN_PSEUDO_ROOT / "quality_report.json").exists()

if cleaned_complete():
    print("Cleaned pseudo masks complete:", CLEAN_PSEUDO_ROOT)
elif RUN_CLEANING_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["clean"],
        "--input-root", RAW_PSEUDO_ROOT,
        "--output-root", CLEAN_PSEUDO_ROOT,
        "--dentex-root", RAW_ROOT / "dentex",
        "--max-images", "0",
        "--preview",
    ])
else:
    print("Cleaned pseudo masks missing and cleaning disabled.")

quality_report = read_json(CLEAN_PSEUDO_ROOT / "quality_report.json", [])
display(pd.DataFrame(quality_report))
"""
    ),
    md("## 9. Full DENTEX Student Training on Cleaned Pseudo Masks"),
    code(
        r"""
student_status = checkpoint_summary(STUDENT_CKPT)
current_epoch = int(student_status["epoch"] or 0)
if current_epoch >= STUDENT_EPOCHS and STUDENT_BEST_CKPT.exists() and not FORCE_RETRAIN_OR_REGENERATE:
    print(f"Student already trained to epoch {current_epoch}; skipping long training.")
elif RUN_LONG_TRAINING_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["student"],
        "--clean-root", CLEAN_PSEUDO_ROOT,
        "--checkpoint-root", ACTIVE_CHECKPOINT_ROOT,
        "--train-tasks", "quadrant_full", "enumeration_full", "disease_full",
        "--val-tasks", "validation_full",
        "--epochs", str(STUDENT_EPOCHS),
        "--image-size", str(IMAGE_SIZE),
        "--batch-size", str(STUDENT_BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--width", "16",
        "--learning-rate", "0.001",
        "--run-name", "student_tinyunet_dentex_pseudo_full",
    ] + ([] if FRESH_RERUN else ["--resume"]))
else:
    print("Student checkpoint missing/incomplete, but long training is disabled.")
    print("Set RUN_LONG_TRAINING_IF_MISSING=True to start/resume full student training.")

display(pd.DataFrame([
    checkpoint_summary(STUDENT_CKPT),
    checkpoint_summary(STUDENT_BEST_CKPT),
]))
"""
    ),
    md("## 10. Generate Final DENTEX and TUFTS Evaluation/XAI Artifacts"),
    code(
        r"""
required_final_files = [
    FINAL_EVAL_ROOT / "figures" / "student_training_curves_final.png",
    FINAL_EVAL_ROOT / "figures" / "segmentation_confusion_matrices.png",
    FINAL_EVAL_ROOT / "figures" / "pixel_level_roc_curves.png",
    FINAL_EVAL_ROOT / "figures" / "dentex_xai_uncertainty_panels.png",
    FINAL_EVAL_ROOT / "figures" / "tufts_xai_uncertainty_panels.png",
    FINAL_EVAL_ROOT / "figures" / "tufts_failure_case_panels.png",
    FINAL_EVAL_ROOT / "tables" / "final_metrics_summary.csv",
]
final_complete = all(path.exists() for path in required_final_files)

if final_complete and not FORCE_RETRAIN_OR_REGENERATE:
    print("Final evaluation/XAI artifacts already exist:", FINAL_EVAL_ROOT)
elif RUN_FINAL_EVAL_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["final_eval"],
        "--student-checkpoint", STUDENT_BEST_CKPT,
        "--xray-teacher-checkpoint", TUFTS_TEACHER_CKPT,
        "--clean-root", CLEAN_PSEUDO_ROOT,
        "--tufts-root", RAW_ROOT / "tufts",
        "--output-root", FINAL_EVAL_ROOT,
        "--max-tufts", "0",
    ])
else:
    print("Final evaluation missing and generation disabled.")

metrics_path = FINAL_EVAL_ROOT / "tables" / "final_metrics_summary.csv"
if metrics_path.exists():
    display(pd.read_csv(metrics_path))
"""
    ),
    md("## 11. Display Direct Notebook Figures"),
    code(
        r"""
figures = [
    FINAL_EVAL_ROOT / "figures" / "student_training_curves_final.png",
    FINAL_EVAL_ROOT / "figures" / "segmentation_confusion_matrices.png",
    FINAL_EVAL_ROOT / "figures" / "pixel_level_roc_curves.png",
    FINAL_EVAL_ROOT / "figures" / "dentex_xai_uncertainty_panels.png",
    FINAL_EVAL_ROOT / "figures" / "tufts_xai_uncertainty_panels.png",
    FINAL_EVAL_ROOT / "figures" / "tufts_failure_case_panels.png",
    FINAL_EVAL_ROOT / "figures" / "augmentation_examples.png",
]

for figure in figures:
    display(Markdown(f"### {figure.name}"))
    show_image(figure, width=1000)
"""
    ),
    md("## 12. Ablation and Paper Package"),
    code(
        r"""
ablation_required = [
    ABLATION_ROOT / "figures" / "teacher_vs_student.png",
    ABLATION_ROOT / "figures" / "uncertainty_filtering_curve.png",
    ABLATION_ROOT / "tables" / "baseline_vs_proposed_summary.csv",
]
if all(path.exists() for path in ablation_required) and not FORCE_RETRAIN_OR_REGENERATE:
    print("Ablation artifacts already exist:", ABLATION_ROOT)
elif RUN_ABLATION_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["ablation"],
        "--output-root", ABLATION_ROOT,
        "--raw-pseudo-root", RAW_PSEUDO_ROOT,
        "--clean-root", CLEAN_PSEUDO_ROOT,
        "--final-eval-root", FINAL_EVAL_ROOT,
        "--teacher-checkpoint", TUFTS_TEACHER_CKPT,
        "--student-checkpoint", STUDENT_BEST_CKPT,
    ])

paper_required = [
    PAPER_PACKAGE_ROOT / "figures" / "Fig1_method_pipeline.png",
    PAPER_PACKAGE_ROOT / "figures" / "Fig9_augmentation_examples.png",
    PAPER_PACKAGE_ROOT / "tables" / "Table2_main_results.csv",
]
if all(path.exists() for path in paper_required) and not FORCE_RETRAIN_OR_REGENERATE:
    print("Paper package already exists:", PAPER_PACKAGE_ROOT)
elif RUN_PAPER_PACKAGE_IF_MISSING or FORCE_RETRAIN_OR_REGENERATE:
    run_command([
        PYTHON_EXE, SCRIPTS["paper"],
        "--output-root", PAPER_PACKAGE_ROOT,
        "--final-eval-root", FINAL_EVAL_ROOT,
        "--ablation-root", ABLATION_ROOT,
    ])

display(Markdown("### Paper package manifest"))
manifest = PAPER_PACKAGE_ROOT / "paper_package_manifest.json"
if manifest.exists():
    display(read_json(manifest, {}))
"""
    ),
    md("## 13. Final Evidence Summary"),
    code(
        r"""
summary_rows = []
table2 = PAPER_PACKAGE_ROOT / "tables" / "Table2_main_results.csv"
if table2.exists():
    display(pd.read_csv(table2))

summary_rows.append({"item": "Full DENTEX student checkpoint", "path": str(STUDENT_BEST_CKPT), "exists": STUDENT_BEST_CKPT.exists()})
summary_rows.append({"item": "Final evaluation folder", "path": str(FINAL_EVAL_ROOT), "exists": FINAL_EVAL_ROOT.exists()})
summary_rows.append({"item": "Paper package folder", "path": str(PAPER_PACKAGE_ROOT), "exists": PAPER_PACKAGE_ROOT.exists()})
summary_rows.append({"item": "Manuscript DOCX", "path": r"E:\dental_ai_results\manuscript_docx_v1\Dental_AI_Manuscript_Draft_v1.docx", "exists": Path(r"E:\dental_ai_results\manuscript_docx_v1\Dental_AI_Manuscript_Draft_v1.docx").exists()})
display(pd.DataFrame(summary_rows))

print("NOTE:")
print("- This notebook is the full train/test/validation control notebook.")
print("- Existing checkpoints make reruns fast.")
print("- Default mode only verifies/reuses existing checkpoints.")
print('- To force a real long rerun from epoch 1, set TRAINING_MODE = "fresh_rerun" in cell 1 and run again.')
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    payload = json.dumps(notebook, indent=1)
    try:
        PRIMARY_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRIMARY_NOTEBOOK_PATH.write_text(payload, encoding="utf-8")
        print(PRIMARY_NOTEBOOK_PATH)
    except OSError as exc:
        FALLBACK_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_NOTEBOOK_PATH.write_text(payload, encoding="utf-8")
        print(f"Primary notebook folder was not writable: {exc}")
        print(FALLBACK_NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
