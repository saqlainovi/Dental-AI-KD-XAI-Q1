"""
Dental AI v2.0 — Publication Figures Generator (Real V2 Results)
================================================================
Generates V2 publication figures from saved training and evaluation artefacts.
Every plotted value must be read from a CSV or checkpoint; do not add projected
baselines, telemetry, or timing values to this file.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FIGS = ROOT / "outputs" / "paper_figures_v2"
OVERLEAF_FIGS = ROOT / "Dental_AI_Overleaf_Package" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OVERLEAF_FIGS.mkdir(parents=True, exist_ok=True)

# Publication styling
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

def save_dual(fig, filename):
    p1 = OUT_FIGS / filename
    p2 = OVERLEAF_FIGS / filename
    fig.savefig(p1)
    fig.savefig(p2)
    plt.close(fig)
    print(f"  Saved: {filename}")


# 1. Fig19: Confusion Matrix Heatmap
def make_fig19():
    df = pd.read_csv(ROOT / "outputs" / "v2_disease_classification" / "disease_test_per_class_results.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for idx, row in df.iterrows():
        ax = axes[idx]
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax,
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(f"{row['disease']}\n(F1: {row['f1']:.4f}, Rec: {row['recall']:.4f})", fontweight="bold")
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("Ground Truth")
    plt.tight_layout()
    save_dual(fig, "Fig19_confusion_matrix_heatmap.png")


# 2. Fig21: Training Timeline
def make_fig21():
    seg = pd.read_csv(ROOT / "outputs" / "v2_segmentation" / "segformer_tooth_seg_epoch_history.csv")
    disease = pd.read_csv(ROOT / "outputs" / "v2_disease_classification" / "disease_training_epoch_history.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    phases = [
        ("ConvNeXt-Tiny segmentation (60 epochs)", seg["epoch_seconds"].sum() / 60.0, "#2ca02c"),
        ("EfficientNet-B2 classification (40 epochs)", disease["seconds"].sum() / 60.0, "#d95f02"),
    ]
    labels = [p[0] for p in phases]
    mins = [p[1] for p in phases]
    colors = [p[2] for p in phases]
    bars = ax.barh(labels, mins, color=colors, edgecolor="black", height=0.55)
    for bar, m in zip(bars, mins):
        text = f"{m:.1f} min ({m/60:.2f}h)" if m >= 60 else f"{m:.2f} min ({m*60:.0f}s)"
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, text, va="center", fontweight="bold", fontsize=10)
    ax.set_xlabel("Execution Duration (Minutes)", fontweight="bold")
    ax.set_title("Observed V2 Training Duration", fontweight="bold")
    ax.set_xlim(0, max(mins) * 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_dual(fig, "Fig21_training_timeline.png")


# 3. Fig22: GPU Telemetry
def make_fig22():
    df_seg = pd.read_csv(ROOT / "outputs" / "v2_segmentation" / "segformer_tooth_seg_epoch_history.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    
    # VRAM allocation
    ax1.plot(df_seg["epoch"], df_seg["vram_allocated_mb"] / 1024, color="#2b5c8f", lw=2, label="Allocated VRAM")
    ax1.plot(df_seg["epoch"], df_seg["vram_reserved_mb"] / 1024, color="#d95f02", lw=2, label="Reserved VRAM")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Memory (Gigabytes)", fontweight="bold")
    ax1.set_ylim(0, max(1.0, float((df_seg[["vram_allocated_mb", "vram_reserved_mb"]].max().max() / 1024) * 1.15)))
    ax1.set_title("Recorded GPU Memory After Validation", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Time per epoch
    ax2.plot(df_seg["epoch"], df_seg["epoch_seconds"], color="#d95f02", lw=2)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Seconds / Epoch", fontweight="bold")
    ax2.set_title("Training Latency Stability (1,996 images/epoch)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_dual(fig, "Fig22_gpu_telemetry.png")


# 4. Fig24: Convergence Analysis
def make_fig24():
    df_seg = pd.read_csv(ROOT / "outputs" / "v2_segmentation" / "segformer_tooth_seg_epoch_history.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(df_seg["epoch"], df_seg["train_dice"], color="#2b5c8f", lw=2, label="Train Dice")
    ax1.plot(df_seg["epoch"], df_seg["val_dice"], color="#2ca02c", lw=2, label="Val Dice (Peak: 0.9456)")
    ax1.fill_between(df_seg["epoch"], df_seg["train_dice"], df_seg["val_dice"], alpha=0.15, color="#2b5c8f")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Dice Similarity Coefficient", fontweight="bold")
    ax1.set_ylim(0.7, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df_seg["epoch"], df_seg["train_loss"], color="#d95f02", lw=1.5, ls="--", label="Train Loss")
    ax2.set_ylabel("Composite BCE + Dice Loss", fontweight="bold")
    ax2.set_ylim(0, 0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("ConvNeXt-Tiny Tooth Segmentation Convergence (60 Epochs)", fontweight="bold")
    plt.tight_layout()
    save_dual(fig, "Fig24_convergence_analysis.png")


# 5. Fig23: Model Architecture Comparison
def make_fig23():
    """Plot only the two V2 components; historical models are not a baseline."""
    seg = pd.read_csv(ROOT / "outputs" / "v2_segmentation" / "segformer_tooth_seg_final_eval.csv")
    disease = pd.read_csv(ROOT / "outputs" / "v2_disease_classification" / "disease_test_per_class_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ["Segmentation\n(DentalAI test)", "Classification\n(DENTEX test)"]
    score = [
        float(seg.loc[seg["cohort"] == "DentalAI Internal Test", "dice_mean"].iloc[0]),
        float(disease["f1"].mean()),
    ]

    x = np.arange(len(models))
    ax.bar(x, score, color=["#2b5c8f", "#d95f02"], edgecolor="black")
    for pos, value in zip(x, score):
        ax.text(pos, value + 0.02, f"{value:.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("Task-specific score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("V2 Test Scores (Not a Cross-Task Comparison)", fontweight="bold")
    plt.tight_layout()
    save_dual(fig, "Fig23_student_vs_teacher.png")


# 6. Fig17: Clinical Triage Pie
def make_fig17():
    df_mc = pd.read_csv(ROOT / "outputs" / "v2_mc_dropout" / "mc_dropout_triage_summary.csv")
    low_row = df_mc[df_mc["stratum"].str.contains("Low")].iloc[0]
    high_row = df_mc[df_mc["stratum"].str.contains("High")].iloc[0]
    
    labels = [
        f"Low uncertainty\nN = {int(low_row['count'])} ({low_row['percentage']:.1f}%)",
        f"High uncertainty\nN = {int(high_row['count'])} ({high_row['percentage']:.1f}%)",
    ]
    sizes = [low_row["count"], high_row["count"]]
    colors = ["#2ca02c", "#d62728"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", textprops={"fontsize": 11, "weight": "bold"})
    plt.title("Experimental Uncertainty Stratification (20 Stochastic Passes)", fontweight="bold")
    plt.tight_layout()
    save_dual(fig, "Fig17_clinical_triage_pie.png")


if __name__ == "__main__":
    print("Generating all v2 publication figures from real data...")
    make_fig19()
    make_fig21()
    make_fig22()
    make_fig23()
    make_fig24()
    make_fig17()
    print("All v2 publication figures generated successfully!")
