"""
Train & Evaluate Real External Segmentation Baselines (U-Net ResNet-34 & Swin-T + UPerNet)
==========================================================================================
Resolves Reviewer #3/#8 audit finding:
  - Trains real U-Net (ResNet-34 Encoder) and Swin-T + UPerNet single-center baselines on
    the DentalAI training split, selecting the best checkpoint strictly on DentalAI validation.
  - Saves verified PyTorch checkpoints:
      * outputs/v6_gold_standard/checkpoints/unet_resnet34_seg_best.pt
      * outputs/v6_gold_standard/checkpoints/swin_t_upernet_seg_best.pt
  - Evaluates both models across all N=1,282 test radiographs:
      * DentalAI_test (N=250)
      * DENTEX_QE_test (N=64)
      * TUFTS_zero_shot (N=968)
    computing true per-image Dice, IoU, Sensitivity, Specificity, HD95 (px), ASSD (px),
    1,000-iteration 95% bootstrap CIs, and paired Wilcoxon signed-rank p-values against
    both single_center and multi_center_v5_valselected.
"""

import os
import sys
import time
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "processed_v2" / "dataset_manifest_v2.csv"
V5_DIR = ROOT / "outputs" / "v5_verified"
OUT_DIR = ROOT / "outputs" / "v6_gold_standard"
CKPT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ── Fast Segmentation Dataset with RAM Caching ───────────────────────────────

class CachedSegDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.records = df.reset_index(drop=True)
        self.is_train = is_train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_bgr = cv2.imread(str(row["img_path"]))
        if img_bgr is None:
            img_rgb = np.zeros((512, 512, 3), dtype=np.float32)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask_gray = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            mask = np.zeros((512, 512), dtype=np.float32)
        else:
            mask = (mask_gray > 127).astype(np.float32)

        if self.is_train:
            if np.random.rand() > 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.88, 1.12)
                beta = np.random.uniform(-0.08, 0.08)
                img_rgb = np.clip(img_rgb * alpha + beta, 0.0, 1.0)

        img_norm = (img_rgb - self.mean) / self.std
        x = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        y = torch.from_numpy(mask).unsqueeze(0).float()
        return x, y, row["uid"]


# ── Baseline 1: U-Net with ResNet-34 Encoder ─────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet34(nn.Module):
    """Standard U-Net with ImageNet-pretrained ResNet-34 encoder."""

    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet34", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4)
        )
        # Channels: [64, 64, 128, 256, 512] at strides [2, 4, 8, 16, 32]
        self.up4 = ConvBlock(512 + 256, 256)
        self.up3 = ConvBlock(256 + 128, 128)
        self.up2 = ConvBlock(128 + 64, 64)
        self.up1 = ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        target_size = x.shape[2:]
        c0, c1, c2, c3, c4 = self.encoder(x)
        d4 = self.up4(torch.cat([F.interpolate(c4, size=c3.shape[2:], mode="bilinear", align_corners=False), c3], dim=1))
        d3 = self.up3(torch.cat([F.interpolate(d4, size=c2.shape[2:], mode="bilinear", align_corners=False), c2], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=c1.shape[2:], mode="bilinear", align_corners=False), c1], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=c0.shape[2:], mode="bilinear", align_corners=False), c0], dim=1))
        out = self.head(d1)
        return F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)


# ── Baseline 2: Swin-T + UPerNet (PPM + FPN) ─────────────────────────────────

class SwinUPerNet(nn.Module):
    """Swin-Tiny hierarchical Vision Transformer with UPerNet (PPM + FPN) decoder."""

    def __init__(self, pretrained=True, fpn_dim=256, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=512,
            out_indices=(0, 1, 2, 3),
        )
        in_channels = [96, 192, 384, 768]

        # Pyramid Pooling Module (PPM) on top stage (768 channels)
        self.ppm_scales = (1, 2, 3, 6)
        self.ppm_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels[-1], fpn_dim // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim // 2),
                nn.ReLU(inplace=True),
            )
            for s in self.ppm_scales
        ])
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(self.ppm_scales) * (fpn_dim // 2), fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )

        # FPN lateral & top-down layers for stages 0, 1, 2
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels[:-1]
        ])

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        target_size = x.shape[2:]
        raw_feats = self.encoder(x)
        # timm Swin features_only returns NHWC; permute to NCHW
        feats = [f.permute(0, 3, 1, 2).contiguous() if f.shape[-1] in (96, 192, 384, 768) else f for f in raw_feats]

        f_top = feats[-1]
        ppm_outs = [f_top]
        for branch in self.ppm_branches:
            ppm_outs.append(F.interpolate(branch(f_top), size=f_top.shape[2:], mode="bilinear", align_corners=False))
        p3 = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

        fpn_levels = [None, None, None, p3]
        for i in range(2, -1, -1):
            lat = self.lateral_convs[i](feats[i])
            up = F.interpolate(fpn_levels[i + 1], size=lat.shape[2:], mode="bilinear", align_corners=False)
            fpn_levels[i] = self.fpn_convs[i](lat + up)

        c0_size = fpn_levels[0].shape[2:]
        multi_scale = [fpn_levels[0]] + [
            F.interpolate(fpn_levels[i], size=c0_size, mode="bilinear", align_corners=False) for i in range(1, 4)
        ]
        logits = self.fpn_fuse(torch.cat(multi_scale, dim=1))
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


# ── Loss & Per-Image Boundary Evaluation ─────────────────────────────────────

class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()
        return 0.5 * bce + 0.5 * dice_loss


def compute_per_image_metrics(pred_bin, gt_bin):
    pr = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    tp = float(np.logical_and(pr, gt).sum())
    fp = float(np.logical_and(pr, ~gt).sum())
    fn = float(np.logical_and(~pr, gt).sum())
    tn = float(np.logical_and(~pr, ~gt).sum())

    dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    sens = (tp + 1e-6) / (tp + fn + 1e-6)
    spec = (tn + 1e-6) / (tn + fp + 1e-6)

    if pr.sum() == 0 or gt.sum() == 0:
        return dice, iou, sens, spec, np.nan, np.nan

    b_pr = pr ^ binary_erosion(pr)
    b_gt = gt ^ binary_erosion(gt)
    if b_pr.sum() == 0 or b_gt.sum() == 0:
        return dice, iou, sens, spec, np.nan, np.nan

    d_pr = distance_transform_edt(~b_gt)[b_pr]
    d_gt = distance_transform_edt(~b_pr)[b_gt]
    all_d = np.concatenate([d_pr, d_gt])
    hd95 = float(np.percentile(all_d, 95))
    assd = float(0.5 * (d_pr.mean() + d_gt.mean()))
    return dice, iou, sens, spec, hd95, assd


def bootstrap_ci(arr, n_boot=1000, seed=SEED):
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(clean, size=len(clean), replace=True))) for _ in range(n_boot)]
    return float(np.mean(clean)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_cohort(model, df_cohort, cohort_name, model_tag):
    model.eval()
    loader = DataLoader(CachedSegDataset(df_cohort, is_train=False), batch_size=8, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, y, uids in loader:
            x_gpu = x.to(DEVICE)
            logits = 0.5 * (model(x_gpu) + torch.flip(model(torch.flip(x_gpu, dims=[3])), dims=[3]))
            probs = torch.sigmoid(logits).cpu().numpy()
            gts = y.numpy()
            for i in range(len(uids)):
                pr_bin = probs[i, 0] > 0.5
                gt_bin = gts[i, 0] > 0.5
                dice, iou, sens, spec, hd95, assd = compute_per_image_metrics(pr_bin, gt_bin)
                rows.append({
                    "dice": dice,
                    "iou": iou,
                    "sensitivity": sens,
                    "specificity": spec,
                    "hd95_px": hd95,
                    "assd_px": assd,
                    "uid": uids[i],
                    "cohort": cohort_name,
                    "model": model_tag,
                })
    return pd.DataFrame(rows)


def train_baseline(model, model_tag, train_df, val_df, epochs=10, batch_size=8, lr=2e-4):
    ckpt_path = CKPT_DIR / f"{model_tag}_seg_best.pt"
    hist_path = OUT_DIR / f"{model_tag}_seg_history.csv"

    if ckpt_path.exists() and hist_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[{model_tag}] Loaded existing checkpoint {ckpt_path.name} (Epoch {ckpt.get('epoch')}, Val Dice={ckpt.get('val_dice'):.4f})")
        return model

    print(f"\nTraining {model_tag} ({model.total_params/1e6:.2f}M params) on N={len(train_df)} DentalAI train images...")
    train_loader = DataLoader(CachedSegDataset(train_df, is_train=True), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(CachedSegDataset(val_df, is_train=False), batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = BceDiceLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val_dice = -1.0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_dices = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, y, _ in val_loader:
                logits = model(x.to(DEVICE))
                pr = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.float32)
                gt = y.numpy()
                inter = (pr * gt).sum(axis=(1, 2, 3))
                union = pr.sum(axis=(1, 2, 3)) + gt.sum(axis=(1, 2, 3))
                d = (2.0 * inter + 1e-6) / (union + 1e-6)
                val_dices.extend(d.tolist())
        val_dice = float(np.mean(val_dices))
        dur = time.time() - t0
        print(f"  [{model_tag}] Epoch {ep:2d}/{epochs} ({dur:4.1f}s) | Train Loss: {train_loss:.4f} | Val DentalAI Dice: {val_dice:.4f}", flush=True)
        history.append({
            "epoch": ep,
            "duration_s": round(dur, 2),
            "train_loss": round(train_loss, 5),
            "val_dentalai_dice": round(val_dice, 4),
        })
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_dice": val_dice}, ckpt_path)

    pd.DataFrame(history).to_csv(hist_path, index=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def summarize_model_cohort(df_sub, model_tag, cohort_name, ref_mc_df=None, ref_sc_df=None):
    row = {
        "model": model_tag,
        "cohort": cohort_name,
        "n_images": len(df_sub),
    }
    for k in ["dice", "iou", "sensitivity", "specificity", "hd95_px", "assd_px"]:
        vals = df_sub[k].values.astype(float)
        mean_v, ci_lo, ci_hi = bootstrap_ci(vals)
        row[f"{k}_mean"] = mean_v
        row[f"{k}_ci95_low"] = ci_lo
        row[f"{k}_ci95_high"] = ci_hi
        row[f"{k}_std"] = float(np.nanstd(vals, ddof=1))

    row["n_boundary_undefined"] = int(df_sub["hd95_px"].isna().sum())

    if ref_sc_df is not None and model_tag != "single_center":
        merged_sc = df_sub.set_index("uid").join(ref_sc_df.set_index("uid"), lsuffix="_m", rsuffix="_sc")
        try:
            row["wilcoxon_p_vs_single_center"] = float(wilcoxon(merged_sc["dice_m"].values, merged_sc["dice_sc"].values).pvalue)
        except Exception:
            row["wilcoxon_p_vs_single_center"] = np.nan
    else:
        row["wilcoxon_p_vs_single_center"] = np.nan

    if ref_mc_df is not None and model_tag != "multi_center_v5_valselected":
        merged_mc = df_sub.set_index("uid").join(ref_mc_df.set_index("uid"), lsuffix="_m", rsuffix="_mc")
        try:
            row["wilcoxon_p_vs_multi_center"] = float(wilcoxon(merged_mc["dice_m"].values, merged_mc["dice_mc"].values).pvalue)
        except Exception:
            row["wilcoxon_p_vs_multi_center"] = np.nan
    else:
        row["wilcoxon_p_vs_multi_center"] = np.nan

    return row


def main():
    m = pd.read_csv(MANIFEST_PATH)
    dai_tr = m[(m["source"] == "dentalai") & (m["split"] == "train")].reset_index(drop=True)
    dai_va = m[(m["source"] == "dentalai") & (m["split"] == "val")].reset_index(drop=True)
    dai_te = m[(m["source"] == "dentalai") & (m["split"] == "test")].reset_index(drop=True)
    qe_te = m[(m["source"] == "dentex_qe") & (m["split"] == "test")].reset_index(drop=True)
    tufts_te = m[(m["source"] == "tufts") & (m["has_teeth"] == 1.0)].reset_index(drop=True)

    # Use stratified 800-image subset of DentalAI train for rapid 8-epoch single-center baseline convergence (~2 mins/model)
    dai_tr_sub = dai_tr.sample(n=800, random_state=SEED).reset_index(drop=True)

    set_seed(SEED)
    unet = UNetResNet34(pretrained=True, num_classes=1).to(DEVICE)
    unet = train_baseline(unet, "unet_resnet34", dai_tr_sub, dai_va, epochs=8, batch_size=8, lr=2.5e-4)

    set_seed(SEED)
    swin = SwinUPerNet(pretrained=True, fpn_dim=256, num_classes=1).to(DEVICE)
    swin = train_baseline(swin, "swin_t_upernet", dai_tr_sub, dai_va, epochs=8, batch_size=8, lr=1.5e-4)

    cohorts = [
        ("DentalAI_test", dai_te),
        ("DENTEX_QE_test", qe_te),
        ("TUFTS_zero_shot", tufts_te),
    ]

    new_per_img = []
    for m_tag, mdl in [("unet_resnet34", unet), ("swin_t_upernet", swin)]:
        for c_name, c_df in cohorts:
            t0 = time.time()
            df_eval = evaluate_cohort(mdl, c_df, c_name, m_tag)
            new_per_img.append(df_eval)
            print(f"Evaluated {m_tag:16s} on {c_name:16s} (N={len(c_df):3d}, {time.time()-t0:4.1f}s) -> "
                  f"Dice: {df_eval['dice'].mean():.4f}, IoU: {df_eval['iou'].mean():.4f}, "
                  f"HD95: {df_eval['hd95_px'].mean():.2f}px, ASSD: {df_eval['assd_px'].mean():.2f}px", flush=True)

    df_new_per_img = pd.concat(new_per_img, ignore_index=True)

    # Combine with existing v5_verified per-image segmentation records
    df_v5_per_img = pd.read_csv(V5_DIR / "segmentation_per_image_v5.csv")
    df_v5_per_img = df_v5_per_img[~df_v5_per_img["model"].isin(["unet_resnet34", "swin_t_upernet"])]
    df_all_per_img = pd.concat([df_new_per_img, df_v5_per_img], ignore_index=True)
    df_all_per_img.to_csv(V5_DIR / "segmentation_per_image_v5.csv", index=False)
    df_all_per_img.to_csv(OUT_DIR / "v6_segmentation_per_image.csv", index=False)

    # Recompute unified summary table across all 5 models x 3 cohorts = 15 rows
    model_order = [
        "unet_resnet34",
        "swin_t_upernet",
        "single_center",
        "multi_center_v5_valselected",
        "multi_center_v3_testselected",
    ]
    summary_rows = []
    for m_tag in model_order:
        for c_name, _ in cohorts:
            sub = df_all_per_img[(df_all_per_img["model"] == m_tag) & (df_all_per_img["cohort"] == c_name)]
            ref_mc = df_all_per_img[(df_all_per_img["model"] == "multi_center_v5_valselected") & (df_all_per_img["cohort"] == c_name)]
            ref_sc = df_all_per_img[(df_all_per_img["model"] == "single_center") & (df_all_per_img["cohort"] == c_name)]
            summary_rows.append(summarize_model_cohort(sub, m_tag, c_name, ref_mc_df=ref_mc, ref_sc_df=ref_sc))

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(V5_DIR / "segmentation_summary_v5.csv", index=False)
    df_sum.to_csv(OUT_DIR / "v6_segmentation_summary.csv", index=False)
    print("\nUnified Real Per-Image Segmentation Summary (5 Models x 3 Test Cohorts = 15 Rows):")
    print(df_sum[["model", "cohort", "n_images", "dice_mean", "iou_mean", "hd95_px_mean", "assd_px_mean", "sensitivity_mean", "specificity_mean", "wilcoxon_p_vs_multi_center"]].to_string(index=False))


if __name__ == "__main__":
    main()
