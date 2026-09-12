# Recovered Checkpoint Audit

## Source

Recovered from the local checkpoint archive at `F:\dental_ai_checkpoints` after the OneDrive project files became unavailable.

## Verified training evidence

| Run | Student epochs | Best student validation Dice | Best student test Dice | Best teacher validation Dice |
| --- | ---: | ---: | ---: | ---: |
| `real_full_multiclass_power_20260529_092001` | 50 | 0.9422 (epoch 50) | 0.8952 (epoch 34) | 0.9114 (epoch 36) |
| `real_full_multiclass_power_20260529_210128` | 50 | 0.9369 (epoch 40) | 0.8937 (epoch 50) | 0.9111 (epoch 45) |

Each recovered run contains a student checkpoint, best-student checkpoint, teacher checkpoint, and epoch histories. The archive also contains historical Swin UPerNet checkpoints in `F:\dental_ai_checkpoints\swin_upernet_multitask_q1_v6_512_segfirst_multicheckpoint`.

## Important validity constraint

The recovered teacher histories are named `teacher_tinyunet_tufts_xray_full_epoch_history.csv`. This indicates that TUFTS was used for teacher training in these runs. Consequently, the reported results cannot be used as an independent DENTEX-trained to TUFTS-tested external-validation result unless the original split/configuration proves otherwise.

## Remaining work, in priority order

1. Recover the training code, configs, and final evaluation artifacts from another local backup, a Git remote, a notebook export, or the original execution machine.
2. Reconstruct a fixed DENTEX train/validation/test split and train the final model without TUFTS images.
3. Freeze the DENTEX-only model, then evaluate it once on untouched TUFTS data for external validation.
4. Re-run multi-pathology detection metrics, per-class results, calibration/uncertainty, and XAI evaluation.
5. Reproduce baseline and ablation tables, then build the manuscript from only verified numbers.

## Not yet recovered

- Final multi-pathology detection metrics
- TUFTS external-test-only metrics
- Baseline and ablation result tables
- Final XAI and uncertainty output files
- Reproducible training commands and configurations
- Manuscript/paper package
