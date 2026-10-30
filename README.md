# Dental AI Research Project

This workspace contains the implementation scaffold for the dental panoramic X-ray research plan described in `Dental_AI_Research_Documentation_v2.docx`.

## Current Dataset Roles

- DENTEX (`DATASET/7812323.zip`): primary benchmark for dental enumeration and disease detection.
- DentalAI (`DATASET/dentalai-DatasetNinja.tar`): pixel-level polygon masks for segmentation supervision.
- TUFTS (`DATASET/tufts-dental-database.zip`): external validation dataset with radiographs, teeth masks, polygons, bounding boxes, and expert/student annotations.

## Suggested Workflow

1. Extract raw datasets:

   ```powershell
   python scripts/extract_datasets.py
   ```

2. Inspect extracted dataset structure:

   ```powershell
   python scripts/inspect_datasets.py
   ```

3. Convert annotations into a common training format:

   ```powershell
   python scripts/convert_dentalai_to_coco.py
   python scripts/convert_tufts_to_coco.py
   ```

The converter scripts are intentionally conservative first-pass utilities. They produce COCO-style JSON files that can be adapted for Mask R-CNN, Mask2Former, Detectron2, MMDetection, or Ultralytics workflows.

## Current Main Direction

Use `notebooks/Dental_AI_Novelty_Pipeline_v1.ipynb` as the current project guide. The older Swin crop-classification notebooks are baseline evidence only; they are not the final novelty method.

Current baseline checkpoints live outside OneDrive at:

```text
E:/dental_ai_checkpoints
```

To summarize the baseline checkpoints:

```powershell
python scripts/summarize_checkpoints.py
```

To build DentalAI segmentation mask targets for the teacher model:

```powershell
python scripts/build_segmentation_targets.py --coco data/processed/coco/dentalai/train.json --output E:/dental_ai_processed/masks/dentalai_train
```

The Q1-oriented novelty pipeline is:

1. Keep crop classifiers as baselines.
2. Build real DentalAI mask targets.
3. Train a teacher segmentation/detection model.
4. Generate DENTEX pseudo-masks.
5. Distill a lightweight student model.
6. Validate on TUFTS and report image-only XAI/uncertainty.
