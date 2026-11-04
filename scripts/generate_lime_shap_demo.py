"""
Run LIME and KernelSHAP Explanations on authentic Dental X-ray crops.
Produces high-resolution visual explainability panels for the Dental AI research paper.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from full_pipeline.models.swin_specialist import SwinCropClassifier
from full_pipeline.evaluation.lime_shap_explainer import DentalLimeShapExplainer

OUTPUT_DIR = r"outputs\paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Model
print("[Setup] Loading Swin-T Disease Model on GPU/CPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SwinCropClassifier(num_classes=4, bbox_features=True)
ckpt = torch.load(r"F:\dental_ai_checkpoints\v3_disease_swin_t_full.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

explainer = DentalLimeShapExplainer(model, grid_size=(6, 6), n_samples=100, device=device)
disease_names = ["Impacted Tooth", "Caries", "Periapical Lesion", "Deep Caries"]

# 2. Select two distinct real cases from DENTEX
# Case A: Impacted 3rd Molar (train_673.png, tooth 48)
img673 = cv2.imread(r"data\raw\dentex\training_data\quadrant-enumeration-disease\xrays\train_673.png")
box_imp = [542, 698, 220, 271]
crop_imp = img673[box_imp[1]:box_imp[1]+box_imp[3], box_imp[0]:box_imp[0]+box_imp[2]]

# Case B: Deep Caries Premolar (train_95.png, tooth 35)
img95 = cv2.imread(r"data\raw\dentex\training_data\quadrant-enumeration-disease\xrays\train_95.png")
box_deep = [1727, 795, 102, 341]
crop_deep = img95[box_deep[1]:box_deep[1]+box_deep[3], box_deep[0]:box_deep[0]+box_deep[2]]

cases = [
    ("Case 1: Impacted Mandibular Molar (FDI #48)", crop_imp, box_imp, img673.shape, 0),
    ("Case 2: Deep Pulpal Caries Premolar (FDI #35)", crop_deep, box_deep, img95.shape, 3)
]

fig = plt.figure(figsize=(18, 11), dpi=300, facecolor="#ffffff")
gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.4], hspace=0.35, wspace=0.25)

fig.suptitle("LIME & KernelSHAP Explainability Analysis for Dental Pathology Diagnosis\n(Local Surrogate Models & Shapley Value Feature Attribution on Swin-Transformer)", 
             fontsize=14, fontweight="bold", y=0.98, color="#1a252f")

for row_idx, (title, crop, box, full_shape, target_cls) in enumerate(cases):
    print(f"Explaining {title}...")
    H, W = full_shape[:2]
    norm_bbox = torch.tensor([[box[0]/W, box[1]/H, (box[0]+box[2])/W, (box[1]+box[3])/H]], dtype=torch.float32)
    
    # Run LIME and KernelSHAP
    res = explainer.explain(crop, bbox=norm_bbox, target_class=target_cls)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # 1. Col 0: Original Tooth Crop
    ax0 = fig.add_subplot(gs[row_idx, 0])
    ax0.imshow(crop_rgb)
    ax0.set_title(f"A. Original Radiograph Crop\n{disease_names[target_cls]} (p = {res['base_prob']*100:.1f}%)", fontsize=10, fontweight="bold")
    ax0.axis("off")
    
    # 2. Col 1: LIME Attribution Heatmap Overlay
    lime_overlay = explainer.render_overlay(crop_rgb, res["lime_map"], alpha=0.55)
    ax1 = fig.add_subplot(gs[row_idx, 1])
    ax1.imshow(lime_overlay)
    ax1.set_title("B. LIME Feature Attribution\n(Superpixel Ridge Regression)", fontsize=10, fontweight="bold", color="#c0392b")
    ax1.axis("off")
    
    # 3. Col 2: KernelSHAP Attribution Heatmap Overlay
    shap_overlay = explainer.render_overlay(crop_rgb, res["shap_map"], alpha=0.55)
    ax2 = fig.add_subplot(gs[row_idx, 2])
    ax2.imshow(shap_overlay)
    ax2.set_title("C. KernelSHAP Shapley Values\n(Combinatorial Feature Importance)", fontsize=10, fontweight="bold", color="#2980b9")
    ax2.axis("off")
    
    # 4. Col 3: SHAP Feature Importance Waterfall / Bar Chart
    ax3 = fig.add_subplot(gs[row_idx, 3])
    top10 = res["top_features"][:10]
    seg_names = [f"Region #{seg}" for seg, _ in top10][::-1]
    shap_vals = [val for _, val in top10][::-1]
    
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]
    bars = ax3.barh(seg_names, shap_vals, color=colors, height=0.6, edgecolor="#333333", alpha=0.85)
    
    ax3.axvline(0, color="#7f8c8d", linestyle="--", lw=1.2)
    ax3.set_title("D. Top-10 Regional Shapley Contributions (φ)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Shapley Attribution Value (φ_k)", fontsize=8.5, fontweight="bold")
    ax3.grid(axis="x", alpha=0.4)
    
    # Value annotations on bars
    for b, val in zip(bars, shap_vals):
        offset = 0.003 if val >= 0 else -0.003
        ha = "left" if val >= 0 else "right"
        ax3.text(val + offset, b.get_y() + b.get_height()/2, f"{val:+.3f}", va="center", ha=ha, fontsize=7.5, fontweight="bold")

fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "Fig14_lime_shap_tooth_explanation.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[Success] LIME & SHAP explanation figure saved to: {out_path}")

# Copy to artifact directory
art_dir = r"C:\Users\Admin\.gemini\antigravity\brain\90121f81-2c05-455b-9ee1-61651cc38147\paper_figures"
import shutil
shutil.copy(out_path, os.path.join(art_dir, "Fig14_lime_shap_tooth_explanation.png"))
print("[Done] Copied to artifacts directory.")
