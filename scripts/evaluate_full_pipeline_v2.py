"""
Dental AI v2.0 — Phase 5: Full Evaluation & Publication Artifact Generation
===========================================================================
Aggregates all evaluation results from Phase 2, 3, and 4, generates updated
publication-quality figures, and updates the Overleaf manuscript and CSV reports.

Usage:
    python scripts/evaluate_full_pipeline_v2.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(r"j:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
SEG_DIR = ROOT / "outputs" / "v2_segmentation"
DISEASE_DIR = ROOT / "outputs" / "v2_disease_classification"
MC_DIR = ROOT / "outputs" / "v2_mc_dropout"
FIG_DIR = ROOT / "outputs" / "paper_figures_v2"
OVERLEAF_FIG_DIR = ROOT / "Dental_AI_Overleaf_Package" / "figures"

# Style configuration for Q1 journal
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

DISEASES = ["Impacted", "Caries", "Periapical", "DeepCaries"]
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]


def generate_figures():
    print("=" * 70)
    print("Dental AI v2.0 — Generating Publication Figures from Real Data")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrices Heatmap
    disease_results_csv = DISEASE_DIR / "disease_test_per_class_results.csv"
    if disease_results_csv.exists():
        df_dis = pd.read_csv(disease_results_csv)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for idx, row in df_dis.iterrows():
            ax = axes[idx]
            cm = np.array([
                [row["tn"], row["fp"]],
                [row["fn"], row["tp"]]
            ])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            annot = np.empty_like(cm, dtype=object)
            for i in range(2):
                for j in range(2):
                    annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

            sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False,
                        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax,
                        annot_kws={"size": 11, "weight": "bold"})
            ax.set_title(f"{row['disease']}\n(F1: {row['f1']:.4f}, Rec: {row['recall']:.4f})", fontweight="bold")
            ax.set_xlabel("Predicted")
            if idx == 0:
                ax.set_ylabel("True Condition")

        plt.tight_layout()
        out_p = FIG_DIR / "Fig19_confusion_matrix_heatmap.png"
        plt.savefig(out_p)
        plt.savefig(OVERLEAF_FIG_DIR / "Fig19_confusion_matrix_heatmap.png")
        plt.close()
        print(f"  --> Saved Fig19: {out_p}")

    # 2. SegFormer Training Convergence
    seg_hist_csv = SEG_DIR / "segformer_tooth_seg_epoch_history.csv"
    if seg_hist_csv.exists():
        df_seg = pd.read_csv(seg_hist_csv)
        fig, ax1 = plt.subplots(figsize=(8, 4.5))

        ax1.plot(df_seg["epoch"], df_seg["train_dice"], color="#2b5c8f", lw=2, label="Train Dice")
        ax1.plot(df_seg["epoch"], df_seg["val_dice"], color="#d95f02", lw=2, label="Val Dice")
        ax1.fill_between(df_seg["epoch"], df_seg["train_dice"], df_seg["val_dice"], alpha=0.15, color="#2b5c8f")
        ax1.set_xlabel("Epoch", fontweight="bold")
        ax1.set_ylabel("Dice Similarity Coefficient", fontweight="bold")
        ax1.set_ylim(0.5, 1.0)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(df_seg["epoch"], df_seg["train_loss"], color="#7570b3", lw=1.5, ls="--", label="Train Loss")
        ax2.set_ylabel("Composite BCE + Dice Loss", fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        plt.title("SegFormer-B2 Tooth Segmentation Convergence", fontweight="bold")
        out_p = FIG_DIR / "Fig24_convergence_analysis.png"
        plt.savefig(out_p)
        plt.savefig(OVERLEAF_FIG_DIR / "Fig24_convergence_analysis.png")
        plt.close()
        print(f"  --> Saved Fig24: {out_p}")

    # 3. MC Dropout Clinical Triage Distribution
    mc_summary_csv = MC_DIR / "mc_dropout_triage_summary.csv"
    if mc_summary_csv.exists():
        df_mc = pd.read_csv(mc_summary_csv)
        auto_row = df_mc[df_mc["stratum"].str.contains("Low")].iloc[0]
        ref_row = df_mc[df_mc["stratum"].str.contains("High")].iloc[0]

        labels = [f"Auto-Approved\n({auto_row['percentage']:.1f}%)\nAcc: {auto_row['concordance_accuracy']:.1f}%",
                  f"Radiologist Referral\n({ref_row['percentage']:.1f}%)\nAcc: {ref_row['concordance_accuracy']:.1f}%"]
        sizes = [auto_row["count"], ref_row["count"]]
        colors = ["#2ca02c", "#d62728"]
        explode = (0, 0.08)

        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=140, shadow=True, textprops={"fontsize": 11}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")

        plt.title("Epistemic Uncertainty Risk Stratification (N=20 Passes)", fontweight="bold")
        out_p = FIG_DIR / "Fig17_clinical_triage_pie.png"
        plt.savefig(out_p)
        plt.savefig(OVERLEAF_FIG_DIR / "Fig17_clinical_triage_pie.png")
        plt.close()
        print(f"  --> Saved Fig17: {out_p}")

    print("\nFigure generation complete!")


if __name__ == "__main__":
    generate_figures()
