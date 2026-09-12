# Dental AI Project Completion Plan

## Current assessment

The project has evidence of a teacher-to-student pipeline, pseudo-mask generation, XAI evaluation, and TUFTS external validation. Its final research status cannot yet be confirmed because the corresponding OneDrive files are cloud-only and cannot currently be read on this machine.

## Completion criteria

### 1. Reproducible experiment record

- Record the exact dataset splits, preprocessing, random seed, hardware, package versions, and training commands.
- Identify one final run directory and one final checkpoint for both the teacher and student models.
- Preserve training logs and epoch-level metrics.

### 2. Final internal and external evaluation

- Verify DENTEX validation/test results for segmentation and multi-pathology detection.
- Verify TUFTS external-validation results without leakage or training-time fine-tuning unless it is explicitly reported as adaptation.
- Report sample counts, confidence intervals where feasible, and per-class results; do not report only a single aggregate score.

### 3. Baselines and ablations

- Re-run or verify the agreed baselines using the same split and preprocessing.
- Produce an ablation table covering the teacher model, pseudo-mask cleanup, knowledge distillation, XAI, and uncertainty filtering.
- Make sure every reported comparison is reproducible from a saved command, configuration, and result file.

### 4. XAI and uncertainty validation

- Produce representative Grad-CAM/SHAP examples for correct, incorrect, and high-uncertainty predictions.
- Quantify explanation/localization quality when labels permit it.
- Evaluate calibration and uncertainty filtering; explain the clinical decision threshold.

### 5. Paper package

- Freeze final numbers, figures, and tables only after the above checks pass.
- Draft the manuscript using the verified results, including limitations and ethics/data statements.
- Have the supervisor review the method, claims, and target-journal fit before submission.

## Immediate blocker

OneDrive needs to make the project files available locally. The highest-priority folders are:

1. `dental_ai_results/swin_upernet_multitask_q1_v15_teacher_student_kd`
2. `FINAL OUTCOMES`
3. `scripts`
4. `notebooks`

After the files are available, begin with `q1_readiness_verdict_v15.csv`, `teacher_vs_student_comparison.csv`, the final metrics summaries, and the current-run README. This will determine whether we need retraining, only evaluation cleanup, or manuscript preparation.
