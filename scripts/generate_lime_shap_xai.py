from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageFilter
from sklearn.linear_model import Ridge
from skimage.segmentation import slic


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.training.swin_upernet import build_model_from_checkpoint, segmentation_logits


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    return path


def load_clean_rows(clean_root: Path, task: str) -> list[dict]:
    summary_path = clean_root / task / "summary.json"
    if not summary_path.exists():
        return []
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        if row.get("status") != "kept":
            continue
        item = dict(row)
        item["image_path"] = str(resolve_path(item["image_path"]))
        if "clean_mask" in item:
            item["clean_mask"] = str(resolve_path(item["clean_mask"]))
        out.append(item)
    return out

def load_internal_test_rows_from_checkpoint(checkpoint: dict, limit: int) -> list[dict]:
    manifest_path = checkpoint.get("split_manifest")
    if not manifest_path:
        return []
    path = resolve_path(manifest_path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "split" not in df.columns:
        return []
    rows = []
    for row in df[df["split"] == "internal_test"].head(limit).to_dict("records"):
        image_path = resolve_path(row.get("image_path", ""))
        mask_path = resolve_path(row.get("mask_path", ""))
        if image_path.exists() and mask_path.exists():
            rows.append(
                {
                    "file_name": row.get("file_name", image_path.name),
                    "image_path": str(image_path),
                    "clean_mask": str(mask_path),
                    "status": "kept",
                }
            )
    return rows


def build_tufts_pairs(raw_root: Path) -> list[dict]:
    image_dir = raw_root / "Radiographs" / "Radiographs"
    mask_dir = raw_root / "Segmentation" / "Segmentation" / "teeth_mask"
    mask_by_stem = {path.stem.lower(): path for path in mask_dir.glob("*") if path.is_file()}
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_by_stem.get(image_path.stem.lower())
        if mask_path:
            rows.append({"image_path": str(image_path), "mask_path": str(mask_path), "name": image_path.name})
    return rows


def load_student(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def image_to_tensor(image: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)


@torch.no_grad()
def predict_probability(model: torch.nn.Module, image: Image.Image, size: int, device: torch.device) -> np.ndarray:
    tensor = image_to_tensor(image, size, device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = segmentation_logits(model(tensor))
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
    return prob.squeeze(0).detach().cpu().numpy().astype(np.float32)


def entropy_map(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    ent = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob)) / math.log(2)
    return ent.astype(np.float32)


def normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    denom = float(values.max()) or 1.0
    return values / denom


def perturb_image(base: np.ndarray, blurred: np.ndarray, segments: np.ndarray, active: np.ndarray) -> Image.Image:
    mask = active[segments]
    mixed = np.where(mask[..., None], base, blurred)
    return Image.fromarray(np.clip(mixed * 255, 0, 255).astype(np.uint8))


def shap_kernel_weights(z: np.ndarray, n_features: int) -> np.ndarray:
    counts = z.sum(axis=1)
    weights = np.zeros(len(z), dtype=np.float64)
    for idx, count in enumerate(counts):
        if count <= 0 or count >= n_features:
            weights[idx] = 1e3
        else:
            weights[idx] = (n_features - 1) / max(1e-6, count * (n_features - count))
    return weights


def local_linear_explanations(
    model: torch.nn.Module,
    image: Image.Image,
    size: int,
    device: torch.device,
    n_segments: int,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    image_small = image.convert("RGB").resize((size, size), Image.BILINEAR)
    base = np.asarray(image_small).astype(np.float32) / 255.0
    blurred = np.asarray(image_small.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32) / 255.0
    segments = slic(base, n_segments=n_segments, compactness=12, sigma=1, start_label=0, channel_axis=-1)
    n_features = int(segments.max()) + 1

    masks = rng.integers(0, 2, size=(n_samples, n_features), dtype=np.int8)
    masks[0, :] = 1
    masks[1, :] = 0
    scores = []
    for mask in masks:
        perturbed = perturb_image(base, blurred, segments, mask.astype(bool))
        prob = predict_probability(model, perturbed, size, device)
        scores.append(float(prob.mean()))
    scores_arr = np.asarray(scores, dtype=np.float32)

    distances = np.sqrt(((1 - masks) ** 2).sum(axis=1) / max(1, n_features))
    lime_weights = np.exp(-(distances**2) / 0.25)
    lime_model = Ridge(alpha=1.0, fit_intercept=True)
    lime_model.fit(masks, scores_arr, sample_weight=lime_weights)

    shap_weights = shap_kernel_weights(masks, n_features)
    shap_model = Ridge(alpha=0.05, fit_intercept=True)
    shap_model.fit(masks, scores_arr, sample_weight=shap_weights)

    lime_values = lime_model.coef_.astype(np.float32)
    shap_values = shap_model.coef_.astype(np.float32)
    lime_map = np.zeros_like(segments, dtype=np.float32)
    shap_map = np.zeros_like(segments, dtype=np.float32)
    for segment_id in range(n_features):
        lime_map[segments == segment_id] = max(0.0, float(lime_values[segment_id]))
        shap_map[segments == segment_id] = max(0.0, float(shap_values[segment_id]))

    prob = predict_probability(model, image_small, size, device)
    return {
        "prob": prob,
        "uncertainty": entropy_map(prob),
        "lime": normalize_heatmap(lime_map),
        "kernel_shap": normalize_heatmap(shap_map),
        "mean_tooth_probability": float(prob.mean()),
        "n_segments": n_features,
    }


def overlay_probability(image: Image.Image, prob: np.ndarray) -> np.ndarray:
    base = np.asarray(image.resize(prob.shape[::-1], Image.BILINEAR).convert("RGB")).astype(np.float32) / 255.0
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    mask = prob[..., None]
    return np.clip(base * 0.65 + red * mask * 0.45, 0, 1)


def load_target_mask(row: dict, size: int) -> np.ndarray | None:
    mask_path = row.get("clean_mask") or row.get("mask_path")
    if not mask_path:
        return None
    path = resolve_path(mask_path)
    if not path.exists():
        return None
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(mask) > 0


def attribution_overlap(attribution: np.ndarray, region: np.ndarray | None) -> float:
    if region is None:
        return float("nan")
    weights = np.maximum(attribution.astype(np.float32), 0.0)
    denom = float(weights.sum())
    if denom <= 1e-8:
        return 0.0
    return float(weights[region].sum() / denom)


def deletion_drop(
    model: torch.nn.Module,
    image: Image.Image,
    attribution: np.ndarray,
    size: int,
    device: torch.device,
    top_fraction: float = 0.20,
) -> float:
    image_small = image.convert("RGB").resize((size, size), Image.BILINEAR)
    base = np.asarray(image_small).astype(np.float32) / 255.0
    blurred = np.asarray(image_small.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32) / 255.0
    threshold = np.quantile(attribution, 1.0 - top_fraction)
    delete_mask = attribution >= threshold
    perturbed = np.where(delete_mask[..., None], blurred, base)
    original = predict_probability(model, image_small, size, device).mean()
    deleted = predict_probability(model, Image.fromarray(np.clip(perturbed * 255, 0, 255).astype(np.uint8)), size, device).mean()
    return float(original - deleted)


def make_panel(rows: list[dict], model: torch.nn.Module, args: argparse.Namespace, device: torch.device, source_name: str, output_path: Path) -> pd.DataFrame:
    summary = []
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, row in enumerate(rows):
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        explanation = local_linear_explanations(
            model,
            image,
            args.xai_size,
            device,
            args.segments,
            args.samples,
            args.seed + row_idx,
        )
        prob = explanation["prob"]
        pred_region = prob >= args.threshold
        target_region = load_target_mask(row, args.xai_size)
        lime_overlap_target = attribution_overlap(explanation["lime"], target_region)
        shap_overlap_target = attribution_overlap(explanation["kernel_shap"], target_region)
        lime_overlap_pred = attribution_overlap(explanation["lime"], pred_region)
        shap_overlap_pred = attribution_overlap(explanation["kernel_shap"], pred_region)
        lime_drop = deletion_drop(model, image, explanation["lime"], args.xai_size, device)
        shap_drop = deletion_drop(model, image, explanation["kernel_shap"], args.xai_size, device)
        axes[row_idx, 0].imshow(image.resize((args.xai_size, args.xai_size), Image.BILINEAR), cmap="gray")
        axes[row_idx, 0].set_title(f"{source_name} image")
        axes[row_idx, 1].imshow(overlay_probability(image, prob))
        axes[row_idx, 1].set_title("Prediction overlay")
        axes[row_idx, 2].imshow(explanation["lime"], cmap="magma")
        axes[row_idx, 2].set_title("LIME-style")
        axes[row_idx, 3].imshow(explanation["kernel_shap"], cmap="viridis")
        axes[row_idx, 3].set_title("KernelSHAP-style")
        axes[row_idx, 4].imshow(explanation["uncertainty"], cmap="inferno")
        axes[row_idx, 4].set_title("Entropy uncertainty")
        for ax in axes[row_idx]:
            ax.axis("off")
        summary.append(
            {
                "source": source_name,
                "image_path": row["image_path"],
                "mean_tooth_probability": explanation["mean_tooth_probability"],
                "lime_target_overlap": lime_overlap_target,
                "kernel_shap_target_overlap": shap_overlap_target,
                "lime_pred_overlap": lime_overlap_pred,
                "kernel_shap_pred_overlap": shap_overlap_pred,
                "lime_top20_deletion_drop": lime_drop,
                "kernel_shap_top20_deletion_drop": shap_drop,
                "n_segments": explanation["n_segments"],
                "samples": args.samples,
                "xai_size": args.xai_size,
                "threshold": float(args.threshold),
                "note": "Overlap closer to 1 means attribution concentrates inside mask/predicted tooth region; positive deletion drop means top attribution removal reduces tooth probability.",
            }
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LIME-style and KernelSHAP-style XAI panels for the DENTEX student.")
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/dental_ai_checkpoints/fresh_rerun_20260518_112354/student_tinyunet_dentex_pseudo_full_best.pt"))
    parser.add_argument("--clean-root", type=Path, default=Path("E:/dental_ai_processed/pseudo_masks/fresh_rerun_20260518_112354/dentex_xray_teacher_clean"))
    parser.add_argument("--tufts-root", type=Path, default=ROOT / "data" / "raw" / "tufts")
    parser.add_argument("--output-root", type=Path, default=Path("E:/dental_ai_results/fresh_rerun_20260518_112354/lime_shap_xai_v1"))
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--segments", type=int, default=70)
    parser.add_argument("--xai-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    tables_dir = args.output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_student(args.checkpoint, device)

    dentex_rows = load_clean_rows(args.clean_root, "validation_full")[: args.examples]
    if not dentex_rows:
        dentex_rows = load_internal_test_rows_from_checkpoint(checkpoint, args.examples)
        print(f"validation_full has no kept masks; using internal_test rows for LIME/SHAP examples={len(dentex_rows)}", flush=True)
    tufts_rows = build_tufts_pairs(args.tufts_root)[: args.examples]
    if not dentex_rows:
        raise RuntimeError(f"No DENTEX validation rows found under {args.clean_root}")
    if not tufts_rows:
        raise RuntimeError(f"No TUFTS rows found under {args.tufts_root}")

    dentex_summary = make_panel(dentex_rows, model, args, device, "DENTEX", figures_dir / "dentex_lime_shap_panels.png")
    tufts_summary = make_panel(tufts_rows, model, args, device, "TUFTS", figures_dir / "tufts_lime_shap_panels.png")
    summary = pd.concat([dentex_summary, tufts_summary], ignore_index=True)
    summary["checkpoint_epoch"] = checkpoint.get("epoch")
    summary["checkpoint"] = str(args.checkpoint)
    summary.to_csv(tables_dir / "lime_shap_xai_summary.csv", index=False)
    print(f"LIME_SHAP_XAI_DONE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
