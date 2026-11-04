from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def probability_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("prob_")]


def target_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("target_")]


def load_average_probabilities(paths: list[Path]) -> pd.DataFrame:
    frames = []
    base_prob_cols: list[str] = []
    for idx, path in enumerate(paths):
        df = pd.read_csv(path)
        prob_cols = probability_columns(df)
        if idx == 0:
            base_prob_cols = prob_cols
        keep = ["file_name", *target_columns(df), *prob_cols]
        frame = df[keep].copy()
        frame = frame.rename(columns={col: f"{col}__model{idx}" for col in prob_cols})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        target_dupes = [col for col in target_columns(frame) if col in merged.columns]
        frame = frame.drop(columns=target_dupes)
        merged = merged.merge(frame, on="file_name", how="inner")
    for prob_col in base_prob_cols:
        model_cols = [col for col in merged.columns if col.startswith(f"{prob_col}__model")]
        merged[prob_col] = merged[model_cols].mean(axis=1)
    return merged[["file_name", *target_columns(merged), *base_prob_cols]].copy()


def threshold_to_array(threshold: float | list[float], n_classes: int) -> np.ndarray:
    if isinstance(threshold, list):
        return np.asarray(threshold, dtype=np.float32).reshape(1, n_classes)
    return np.full((1, n_classes), float(threshold), dtype=np.float32)


def threshold_to_text(threshold: float | list[float]) -> str:
    if isinstance(threshold, list):
        return "|".join(f"{value:.3f}" for value in threshold)
    return f"{float(threshold):.3f}"


def macro_f1(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tp = np.logical_and(pred_b, target_b).sum(axis=0)
    fp = np.logical_and(pred_b, ~target_b).sum(axis=0)
    fn = np.logical_and(~pred_b, target_b).sum(axis=0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def per_class_metrics(prob: np.ndarray, target: np.ndarray, class_names: list[str], threshold: float | list[float]) -> pd.DataFrame:
    pred = prob >= threshold_to_array(threshold, len(class_names))
    rows = []
    for idx, class_name in enumerate(class_names):
        p = pred[:, idx].astype(bool)
        t = target[:, idx].astype(bool)
        tp = int(np.logical_and(p, t).sum())
        fp = int(np.logical_and(p, ~t).sum())
        fn = int(np.logical_and(~p, t).sum())
        tn = int(np.logical_and(~p, ~t).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        rows.append(
            {
                "class": class_name,
                "support": int(t.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def tune_threshold(prob: np.ndarray, target: np.ndarray, class_names: list[str], mode: str) -> tuple[float | list[float], pd.DataFrame]:
    grid = np.arange(0.05, 0.96, 0.05)
    rows = []
    if mode == "per_class":
        thresholds = []
        for class_idx, class_name in enumerate(class_names):
            best_threshold = 0.5
            best_f1 = -1.0
            for threshold in grid:
                pred = prob[:, class_idx] >= threshold
                target_c = target[:, class_idx].astype(bool)
                tp = np.logical_and(pred, target_c).sum()
                fp = np.logical_and(pred, ~target_c).sum()
                fn = np.logical_and(~pred, target_c).sum()
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                rows.append({"class": class_name, "threshold": round(float(threshold), 3), "class_f1": f1})
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_threshold = float(threshold)
            thresholds.append(best_threshold)
        pred = (prob >= threshold_to_array(thresholds, len(class_names))).astype(np.uint8)
        rows.append(
            {
                "class": "__overall__",
                "threshold": threshold_to_text(thresholds),
                "macro_f1": macro_f1(pred, target),
                "exact_match": float((pred == target).all(axis=1).mean()),
            }
        )
        return thresholds, pd.DataFrame(rows)
    for threshold in grid:
        pred = (prob >= threshold).astype(np.uint8)
        rows.append(
            {
                "class": "__overall__",
                "threshold": round(float(threshold), 3),
                "macro_f1": macro_f1(pred, target),
                "exact_match": float((pred == target).all(axis=1).mean()),
            }
        )
    df = pd.DataFrame(rows).sort_values(["macro_f1", "exact_match"], ascending=False)
    return float(df.iloc[0]["threshold"]), df


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prob_cols = probability_columns(df)
    target_cols = [f"target_{col.removeprefix('prob_')}" for col in prob_cols]
    class_names = [col.removeprefix("prob_") for col in prob_cols]
    return df[prob_cols].to_numpy(dtype=np.float32), df[target_cols].to_numpy(dtype=np.float32), class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Average DENTEX disease classifier probabilities and tune thresholds.")
    parser.add_argument("--val-probabilities", nargs="+", type=Path, required=True)
    parser.add_argument("--test-probabilities", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    val_df = load_average_probabilities(args.val_probabilities)
    test_df = load_average_probabilities(args.test_probabilities)
    val_prob, val_target, class_names = arrays(val_df)
    test_prob, test_target, _ = arrays(test_df)
    threshold, threshold_df = tune_threshold(val_prob, val_target, class_names, args.threshold_mode)
    val_pred = (val_prob >= threshold_to_array(threshold, len(class_names))).astype(np.uint8)
    test_pred = (test_prob >= threshold_to_array(threshold, len(class_names))).astype(np.uint8)
    summary = {
        "models": len(args.val_probabilities),
        "threshold": threshold_to_text(threshold),
        "threshold_mode": args.threshold_mode,
        "val_macro_f1": macro_f1(val_pred, val_target),
        "test_macro_f1": macro_f1(test_pred, test_target),
        "val_exact_match": float((val_pred == val_target).all(axis=1).mean()),
        "test_exact_match": float((test_pred == test_target).all(axis=1).mean()),
        "val_images": len(val_df),
        "test_images": len(test_df),
        "class_names": "|".join(class_names),
    }
    pd.DataFrame([summary]).to_csv(args.output_root / "disease_ensemble_summary.csv", index=False)
    threshold_df.to_csv(args.output_root / "disease_ensemble_threshold_tuning_validation.csv", index=False)
    per_class_metrics(val_prob, val_target, class_names, threshold).to_csv(args.output_root / "disease_ensemble_val_per_class.csv", index=False)
    per_class_metrics(test_prob, test_target, class_names, threshold).to_csv(args.output_root / "disease_ensemble_test_per_class.csv", index=False)
    val_df.to_csv(args.output_root / "disease_ensemble_val_probabilities.csv", index=False)
    test_df.to_csv(args.output_root / "disease_ensemble_test_probabilities.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
