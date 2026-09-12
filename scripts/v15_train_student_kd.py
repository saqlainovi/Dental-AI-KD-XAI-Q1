"""
V15 Student Knowledge Distillation Training Pipeline.
Trains ultra-lightweight TinyUNetStudent (0.118M params, width=16) using clean pseudo-masks
distilled from the frozen teacher according to split_policy.json.
- Strictly trains on DENTEX clean pseudo-masks
- Evaluates per-epoch on DENTEX validation split
- Logs epoch history to CSV
- Saves best checkpoint to F:\dental_ai_checkpoints\v15_student_tinyunet_best.pt
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure local imports work
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from full_pipeline.models.tiny_unet import TinyUNetStudent
from scripts.v15_clean_data_loader import get_v15_dataloaders

CHECKPOINT_DIR = r"F:\dental_ai_checkpoints"
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # logits: [B, 2, H, W], targets: [B, H, W]
        bce_loss = self.bce(logits, targets)
        
        probs = F.softmax(logits, dim=1)[:, 1]  # tooth channel
        targets_f = targets.float()
        
        intersection = (probs * targets_f).sum(dim=(1, 2))
        total = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def compute_dice_iou(logits, targets, smooth=1e-5):
    preds = torch.argmax(logits, dim=1).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum().item()
    total = preds.sum().item() + targets.sum().item()
    union = total - intersection
    
    dice = (2.0 * intersection + smooth) / (total + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        d, i = compute_dice_iou(logits.detach(), masks)
        running_loss += loss.item()
        total_dice += d
        total_iou += i
        batches += 1

    return running_loss / max(1, batches), total_dice / max(1, batches), total_iou / max(1, batches)

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            d, i = compute_dice_iou(logits, masks)
            running_loss += loss.item()
            total_dice += d
            total_iou += i
            batches += 1

    return running_loss / max(1, batches), total_dice / max(1, batches), total_iou / max(1, batches)

def main():
    parser = argparse.ArgumentParser(description="V15 Student Distillation Training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick-verify", action="store_true", help="Run 1 quick epoch to verify end-to-end")
    args = parser.parse_args()

    epochs = 1 if args.quick_verify else args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"V15 Student Distillation Training on {device}")
    print(f"Epochs: {epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 60)

    # 1. Data Loaders
    print("[*] Initializing verified data loaders...")
    train_loader, val_loader, _ = get_v15_dataloaders(batch_size=args.batch_size, num_workers=0)
    print(f"[+] Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # 2. Model, Optimizer, Loss
    model = TinyUNetStudent(width=16, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[+] Student TinyUNet initialized: {num_params:.3f}M parameters (0.118M expected).")

    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Check if pre-trained best weights exist to initialize or resume
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "v15_student_tinyunet_best.pt")
    fallback_ckpt = os.path.join(CHECKPOINT_DIR, "student_tinyunet_dentex_pseudo_full_best.pt")
    
    best_val_dice = 0.0
    history = []

    print("[*] Starting training loop...")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_dice, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {ep:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}")

        record = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(elapsed, 2)
        }
        history.append(record)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "config": {
                    "width": 16,
                    "num_classes": 2,
                    "epochs": epochs,
                    "lr": args.lr
                }
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val Dice: {best_val_dice:.4f}) to {best_ckpt_path}")

    # Save epoch history to CSV
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    csv_out = os.path.join(OUTPUTS_DIR, "epoch_history_student_v15.csv")
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_out, index=False)
    print(f"[+] Saved epoch history to {csv_out}")
    print("=" * 60)

if __name__ == "__main__":
    main()
