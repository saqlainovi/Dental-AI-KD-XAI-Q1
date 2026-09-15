"""
Dental AI v2.0 — Master Overnight Execution Pipeline
====================================================
Orchestrates the complete Q1-grade research rebuild from start to finish:
  Phase 1: Dataset Preparation (CLAHE, 512x512 letterbox, splits)
  Phase 2: SegFormer-B2 Tooth Segmentation Fine-Tuning
  Phase 3: Multi-Pathology Classification Training
  Phase 4: Real 20-Pass MC Dropout Epistemic Triage
  Phase 5: Figure & Table Generation

Usage:
    python run_q1_pipeline_overnight.py [--dry-run]
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r"j:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
SCRIPTS_DIR = ROOT / "scripts"
OUTPUT_DIR = ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "q1_pipeline_execution.log"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(step_name, cmd):
    log(f"\n{'='*70}\nSTARTING: {step_name}\nCOMMAND:  {' '.join(cmd)}\n{'='*70}")
    start_t = time.time()

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    for line in process.stdout:
        try:
            print(line, end="", flush=True)
        except Exception:
            try:
                print(line.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
            except Exception:
                pass
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(line)

    process.wait()
    duration = time.time() - start_t
    hrs = duration / 3600
    mins = (duration % 3600) / 60

    if process.returncode != 0:
        log(f"FAILED: {step_name} with exit code {process.returncode} after {hrs:.1f}h ({mins:.1f}m)")
        sys.exit(process.returncode)
    else:
        log(f"COMPLETED: {step_name} successfully in {duration:.1f}s ({hrs:.2f}h)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("DENTAL AI v2.0 — MASTER OVERNIGHT EXECUTION PIPELINE STARTED")
    log("=" * 70)

    # Phase 1: Data Preparation
    manifest = ROOT / "data" / "processed_v2" / "dataset_manifest_v2.csv"
    if manifest.exists():
        log("Phase 1 already complete! Found dataset manifest. Skipping data prep.")
    else:
        run_step("Phase 1: Data Preparation & CLAHE Preprocessing", [
            sys.executable, str(SCRIPTS_DIR / "prepare_dataset_v2.py")
        ])

    # Phase 2: SegFormer-B2 Segmentation Training
    seg_eval = ROOT / "outputs" / "v2_segmentation" / "segformer_tooth_seg_final_eval.csv"
    if seg_eval.exists():
        log("Phase 2 already complete! Found segmentation final evaluation. Skipping Phase 2.")
    else:
        run_step("Phase 2: SegFormer-B2 Tooth Segmentation Training", [
            sys.executable, str(SCRIPTS_DIR / "train_segformer_tooth_seg.py"),
            "--epochs", "60",
            "--batch-size", "4",
            "--grad-accum", "2",
            "--lr", "6e-5",
            "--backbone", "convnext_tiny"
        ])

    # Phase 3: Multi-Pathology Classification Training
    run_step("Phase 3: Multi-Pathology Classification Training", [
        sys.executable, str(SCRIPTS_DIR / "train_disease_classifier_kd.py"),
        "--epochs", "40",
        "--batch-size", "8",
        "--lr", "1e-4",
        "--backbone", "efficientnet_b2"
    ])

    # Phase 4: Real MC Dropout Uncertainty Estimation
    run_step("Phase 4: Real 20-Pass MC Dropout Epistemic Triage", [
        sys.executable, str(SCRIPTS_DIR / "run_mc_dropout_real.py"),
        "--passes", "20",
        "--threshold", "0.05"
    ])

    # Phase 5: Figures and Artifact Generation
    run_step("Phase 5: Figure Generation & Metrics Aggregation", [
        sys.executable, str(SCRIPTS_DIR / "evaluate_full_pipeline_v2.py")
    ])

    log("\n" + "=" * 70)
    log("ALL 5 PHASES COMPLETED SUCCESSFULLY! DENTAL AI v2.0 IS READY FOR Q1 PUBLICATION.")
    log("=" * 70)


if __name__ == "__main__":
    main()
