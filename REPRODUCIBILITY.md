# V2 Reproducibility and Claim Boundary

## Frozen scientific track

The current manuscript and root README describe only the **V2 dual-stage
pipeline**:

1. CLAHE preprocessing at 512 x 512 pixels.
2. ConvNeXt-Tiny encoder with a custom All-MLP decoder for binary tooth
   segmentation, trained on the DentalAI split.
3. EfficientNet-B2 multi-label pathology classification, trained on the DENTEX
   split with class-balanced BCE.
4. DENTEX-validation threshold calibration and experimental MC-dropout
   uncertainty stratification.

The historical V15 teacher--student / knowledge-distillation notebooks,
scripts, checkpoints, reports, and results are archival material.  They must
not be combined with V2 numbers in a manuscript or used as a V2 baseline
without a new, split-matched evaluation.

## Dataset boundary

`data/processed_v2/dataset_manifest_v2.csv` is the V2 split record:

| Source | Train | Validation | Test / external |
|---|---:|---:|---:|
| DentalAI (segmentation) | 1,996 | 249 | 250 |
| DENTEX (pathology) | 547 | 51 | 107 |
| TUFTS (segmentation external holdout) | 0 | 0 | 1,000; 968 non-empty masks |

TUFTS must remain excluded from V2 model fitting, threshold selection, and
checkpoint selection.  A result generated after using TUFTS for any of those
purposes is not an external-validation result.

## Re-running V2

Use a dedicated environment and install all declared dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/prepare_dataset_v2.py
python scripts/train_segformer_tooth_seg.py --epochs 60 --batch-size 4 --grad-accum 2 --lr 6e-5 --backbone convnext_tiny --seed 42
python scripts/train_disease_classifier_kd.py --epochs 40 --batch-size 8 --lr 1e-4 --backbone efficientnet_b2 --seed 42
python scripts/run_mc_dropout_real.py --passes 20 --threshold 0.05
python scripts/evaluate_full_pipeline_v2.py
```

The historical filename `train_disease_classifier_kd.py` is retained to avoid
breaking existing commands; the V2 script performs no knowledge distillation.

## Required evidence before submission

- Store the exact Git commit, package versions, seed, manifest checksum and
  checkpoint checksum for the final run.
- Regenerate every reported V2 table and figure from the final CSV/checkpoint.
- Report uncertainty stratification as exploratory until its threshold is
  selected on validation data and shown to improve a clinically relevant,
  pre-specified outcome.
- Do not claim autonomous approval, clinical-grade deployment, or
  state-of-the-art performance without prospective clinical validation and a
  fair, matched benchmark comparison.
