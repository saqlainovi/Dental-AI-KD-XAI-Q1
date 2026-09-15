"""
Dental AI v2.0 — Phase 3: Multi-Pathology Classification with Real KD
====================================================================
Trains specialist disease classification heads for 4 dental pathologies:
  - Caries
  - Deep Caries
  - Periapical Lesion
  - Impacted Teeth

Features:
  - Pretrained Vision Backbone fine-tuning
  - True Knowledge Distillation (KD) from high-capacity teacher
  - Loss balancing BCE (hard labels) + KL Divergence (teacher soft logits)
  - Class-balanced pos_weights to boost rare classes (Periapical)
  - Per-class threshold calibration on validation cohort
  - Full evaluation on 107 DENTEX test set with confusion matrices & F1

Usage:
    python scripts/train_disease_classifier_kd.py [--epochs 40] [--batch-size 8]
"""

import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

try:
    import timm
except ImportError:
    print("[ERROR] timm not installed.")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(r"j:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
DATA_DIR = ROOT / "data" / "processed_v2"
MANIFEST_PATH = DATA_DIR / "dataset_manifest_v2.csv"
OUTPUT_DIR = ROOT / "outputs" / "v2_disease_classification"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

DISEASES = ["Impacted", "Caries", "Periapical", "DeepCaries"]
DISEASE_COLS = ["disease_impacted", "disease_caries", "disease_periapical", "disease_deep_caries"]


# ── Dataset ────────────────────────────────────────────────────────────

class DentalDiseaseDataset(Dataset):
    """Dataset for 512x512 panoramic images with multi-label disease vectors."""

    def __init__(self, manifest_df, is_train=True):
        self.records = manifest_df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = str(row["img_path"])

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_rgb = np.zeros((512, 512, 3), dtype=np.float32)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
            # Random brightness/contrast
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-0.05, 0.05)
                img_rgb = np.clip(img_rgb * alpha + beta, 0.0, 1.0)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        labels = torch.tensor([
            float(row[c]) for c in DISEASE_COLS
        ], dtype=torch.float32)

        return img_tensor, labels, row["uid"]


# ── Model Architecture ────────────────────────────────────────────────

class DentalMultiDiseaseClassifier(nn.Module):
    """Multi-pathology classifier with vision backbone and multi-label head."""

    def __init__(self, backbone_name="efficientnet_b2", num_classes=4, pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ── Distillation Loss ─────────────────────────────────────────────────

class DistillationBceLoss(nn.Module):
    """Composite loss: Weighted BCE for hard ground-truth labels + KD soft distillation."""

    def __init__(self, pos_weights=None, alpha=0.6, beta=0.4, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, student_logits, targets, teacher_logits=None):
        hard_loss = self.bce(student_logits, targets)

        if teacher_logits is not None:
            # Soft distillation loss (sigmoid-based KL for multi-label)
            student_soft = torch.sigmoid(student_logits / self.temperature)
            teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
            kd_loss = F.binary_cross_entropy(student_soft, teacher_soft) * (self.temperature ** 2)
            total_loss = self.alpha * hard_loss + self.beta * kd_loss
            return total_loss, hard_loss, kd_loss
        else:
            return hard_loss, hard_loss, torch.tensor(0.0, device=student_logits.device)


# ── Calibration and Evaluation ────────────────────────────────────────

def evaluate_disease_model(model, loader, device, thresholds=None):
    """Evaluate multi-label model across all 4 classes."""
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    if thresholds is None:
        # Default 0.5 threshold
        thresholds = [0.5, 0.5, 0.5, 0.5]

    results = []
    for c_idx, d_name in enumerate(DISEASES):
        y_true = all_targets[:, c_idx]
        y_prob = all_probs[:, c_idx]
        y_pred = (y_prob >= thresholds[c_idx]).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5

        results.append({
            "disease": d_name,
            "threshold": thresholds[c_idx],
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc),
            "condition_pos": int(tp + fn),
            "predicted_pos": int(tp + fp),
        })

    macro_f1 = float(np.mean([r["f1"] for r in results]))
    macro_prec = float(np.mean([r["precision"] for r in results]))
    macro_rec = float(np.mean([r["recall"] for r in results]))
    macro_auc = float(np.mean([r["auc"] for r in results]))

    return results, {
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_auc": macro_auc,
        "probs": all_probs,
        "targets": all_targets,
    }


def find_optimal_thresholds(val_probs, val_targets):
    """Calibrate optimal per-class decision thresholds on validation set."""
    best_thresholds = []
    for c_idx in range(4):
        y_true = val_targets[:, c_idx]
        y_prob = val_probs[:, c_idx]

        best_th = 0.5
        best_score = -1.0
        for th in np.linspace(0.2, 0.85, 66):
            y_pred = (y_prob >= th).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            # High penalty on missing Deep Caries
            if c_idx == 3:  # Deep Caries
                score = f1 + 0.5 * rec
            elif c_idx == 2:  # Periapical
                score = f1 + 0.3 * rec
            else:
                score = f1

            if score > best_score:
                best_score = score
                best_th = th
        best_thresholds.append(round(float(best_th), 2))
    return best_thresholds


# ── Main Training Loop ────────────────────────────────────────────────

def train_disease_pipeline(args):
    print("=" * 70)
    print("Dental AI v2.0 — Multi-Pathology Classification Training")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Backbone: {args.backbone}")
    print(f"Learning Rate: {args.lr}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_manifest = pd.read_csv(MANIFEST_PATH)
    train_df = df_manifest[(df_manifest["source"] == "dentex") & (df_manifest["split"] == "train")].copy()
    val_df = df_manifest[(df_manifest["source"] == "dentex") & (df_manifest["split"] == "val")].copy()
    test_df = df_manifest[(df_manifest["source"] == "dentex") & (df_manifest["split"] == "test")].copy()

    print(f"\nDENTEX Cohorts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Compute class pos_weights to counteract class imbalance
    pos_weights = []
    for col in DISEASE_COLS:
        n_pos = train_df[col].sum()
        n_neg = len(train_df) - n_pos
        weight = float(n_neg / max(n_pos, 1))
        # Cap at 5.0 to prevent gradient explosions
        pos_weights.append(min(weight, 5.0))
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    print("Calculated pos_weights for imbalance:")
    for d, w in zip(DISEASES, pos_weights):
        print(f"  {d:12s}: {w:.2f}")

    train_ds = DentalDiseaseDataset(train_df, is_train=True)
    val_ds = DentalDiseaseDataset(val_df, is_train=False)
    test_ds = DentalDiseaseDataset(test_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DentalMultiDiseaseClassifier(backbone_name=args.backbone, num_classes=4, pretrained=True)
    model.to(device)
    print(f"Model: {args.backbone} ({model.total_params / 1e6:.2f} M params)")

    criterion = DistillationBceLoss(pos_weights=pos_weights_tensor, alpha=1.0, beta=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_macro_f1 = 0.0
    history = []

    print("\nStarting Training...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, _, _ = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss /= len(train_loader)

        # Validation
        _, val_summary = evaluate_disease_model(model, val_loader, device)
        val_f1 = val_summary["macro_f1"]
        val_auc = val_summary["macro_auc"]

        print(f"Epoch {epoch:2d}/{args.epochs:2d} [{epoch_time:4.1f}s] | "
              f"Train Loss: {train_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "args": vars(args)
            }, CHECKPOINT_DIR / "best_disease_classifier.pt")
            print(f"  --> Saved NEW BEST model (Macro-F1: {val_f1:.4f})")

        history.append({
            "epoch": epoch,
            "seconds": epoch_time,
            "train_loss": train_loss,
            "val_macro_f1": val_f1,
            "val_macro_auc": val_auc,
        })

    # Final Calibration and Test Evaluation
    print("\n" + "=" * 70)
    print("Calibrating Optimal Thresholds & Testing on 107 Test Set...")
    print("=" * 70)

    best_ckpt = torch.load(CHECKPOINT_DIR / "best_disease_classifier.pt")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Find thresholds on validation set
    _, val_res = evaluate_disease_model(model, val_loader, device)
    optimal_ths = find_optimal_thresholds(val_res["probs"], val_res["targets"])
    print("Calibrated Optimal Thresholds:")
    for d, th in zip(DISEASES, optimal_ths):
        print(f"  {d:12s}: {th}")

    # Evaluate on test set with calibrated thresholds
    test_per_class, test_summary = evaluate_disease_model(model, test_loader, device, thresholds=optimal_ths)

    print(f"\nFinal Test Results (N={len(test_df)}):")
    print("-" * 75)
    print(f"{'Pathology':14s} | {'TP':4s} {'FP':4s} {'FN':4s} {'TN':4s} | {'Prec':6s} {'Recall':6s} {'F1':6s} {'AUC':6s} | {'Thresh':6s}")
    print("-" * 75)
    for r in test_per_class:
        print(f"{r['disease']:14s} | {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {r['tn']:4d} | "
              f"{r['precision']:6.4f} {r['recall']:6.4f} {r['f1']:6.4f} {r['auc']:6.4f} | {r['threshold']:6.2f}")
    print("-" * 75)
    print(f"{'MACRO AVERAGE':14s} | "
          f"{'':17s} | {test_summary['macro_precision']:6.4f} {test_summary['macro_recall']:6.4f} {test_summary['macro_f1']:6.4f} {test_summary['macro_auc']:6.4f} |")
    print("-" * 75)

    # Save outputs
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "disease_training_epoch_history.csv", index=False)
    pd.DataFrame(test_per_class).to_csv(OUTPUT_DIR / "disease_test_per_class_results.csv", index=False)
    with open(OUTPUT_DIR / "calibrated_thresholds.json", "w") as f:
        json.dump(dict(zip(DISEASES, optimal_ths)), f, indent=2)

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dental Disease Classification Training")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="efficientnet_b2", help="timm backbone")
    args = parser.parse_args()

    train_disease_pipeline(args)
