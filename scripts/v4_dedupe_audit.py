"""
v4 dataset-integrity audit: pixel-level near-duplicate detection across ALL raw radiographs.

Motivation (found while fixing Reviewer #7 / #1): file *names* inside DENTEX subsets collide
("train_N.png" is used by several subsets for different images), so counting images by file
name or by folder is unreliable. We therefore fingerprint every raw radiograph by a
mean/std-normalised 64x32 grey thumbnail and cluster near-identical thumbnails
(normalised squared distance < 0.02, a threshold that sits inside a clear gap of the
nearest-neighbour distance distribution).

Outputs (outputs/v4_reviewer_fixes/):
  dedupe_cluster_map.csv      one row per raw radiograph with cohort, path, cluster_id
  dedupe_summary.csv          per-cohort and global counts: files, unique, duplicated-with-other-cohort
  dedupe_pair_matrix.csv      cohort x cohort count of radiographs that have a near-duplicate in the other cohort
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "v4_reviewer_fixes"
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "DentalAI": [RAW / "dentalai" / s / "img" for s in ("train", "valid", "test")],
    "DENTEX-QED": [RAW / "dentex" / "training_data" / "quadrant-enumeration-disease" / "xrays"],
    "DENTEX-QE": [RAW / "dentex" / "training_data" / "quadrant_enumeration" / "xrays"],
    "DENTEX-Quadrant": [RAW / "dentex" / "training_data" / "quadrant" / "xrays"],
    "DENTEX-Unlabelled": [RAW / "dentex" / "training_data" / "unlabelled" / "xrays"],
    "DENTEX-OfficialVal": [RAW / "dentex" / "validation_data" / "quadrant_enumeration_disease" / "xrays"],
    "TUFTS": [RAW / "tufts" / "Radiographs" / "Radiographs"],
}
EXT = (".png", ".jpg", ".jpeg", ".bmp")
THR = 0.02


def fingerprint(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None or im.size == 0:
        return None
    t = cv2.resize(im, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def main():
    rows = []
    for cohort, dirs in COHORTS.items():
        for d in dirs:
            if not d.exists():
                print(f"[WARN] missing {d}")
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXT):
                    rows.append({"cohort": cohort, "path": str(d / f), "name": f})
    df = pd.DataFrame(rows)
    print(f"Fingerprinting {len(df)} raw radiographs ...", flush=True)
    with Pool(6) as pool:
        fps = pool.map(fingerprint, df["path"].tolist(), chunksize=16)
    ok = np.array([fp is not None for fp in fps])
    print(f"Unreadable files skipped: {int((~ok).sum())}")
    df = df[ok].reset_index(drop=True)
    F = np.stack([fp for fp in fps if fp is not None]).astype(np.float32)

    n = len(df)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = F.shape[1]
    sq = (F ** 2).sum(1)
    B = 512
    nn_all = np.zeros(n, dtype=np.float32)
    for s in range(0, n, B):
        blk = (sq[s:s + B, None] + sq[None] - 2.0 * F[s:s + B] @ F.T) / d
        for k in range(blk.shape[0]):
            blk[k, s + k] = 9.0
        nn_all[s:s + B] = blk.min(1)
        ii, jj = np.where(blk < THR)
        for a, b in zip(ii, jj):
            ra, rb = find(s + a), find(b)
            if ra != rb:
                parent[ra] = rb
    df["cluster_id"] = [find(i) for i in range(n)]
    df["nn_dist"] = nn_all
    df.to_csv(OUT / "dedupe_cluster_map.csv", index=False)

    cl = df.groupby("cluster_id")
    df["cluster_size"] = cl["cluster_id"].transform("size")
    df["cluster_n_cohorts"] = cl["cohort"].transform("nunique")

    summ = []
    for cohort in COHORTS:
        sub = df[df["cohort"] == cohort]
        if len(sub) == 0:
            continue
        summ.append({
            "cohort": cohort,
            "files": len(sub),
            "unique_within_cohort": sub["cluster_id"].nunique(),
            "with_near_duplicate_in_other_cohort": int((sub["cluster_n_cohorts"] > 1).sum()),
        })
    summ.append({
        "cohort": "ALL (global)",
        "files": len(df),
        "unique_within_cohort": df["cluster_id"].nunique(),
        "with_near_duplicate_in_other_cohort": int((df["cluster_n_cohorts"] > 1).sum()),
    })
    pd.DataFrame(summ).to_csv(OUT / "dedupe_summary.csv", index=False)

    names = list(COHORTS.keys())
    mat = pd.DataFrame(0, index=names, columns=names)
    cl_members = df.groupby("cluster_id")["cohort"].apply(set).to_dict()
    for _, r in df.iterrows():
        for other in cl_members[r["cluster_id"]]:
            if other != r["cohort"]:
                mat.loc[r["cohort"], other] += 1
    mat.to_csv(OUT / "dedupe_pair_matrix.csv")

    pd.set_option("display.width", 200)
    print(pd.DataFrame(summ).to_string(index=False))
    print("\nRows = cohort whose radiographs have a near-duplicate in column cohort:")
    print(mat.to_string())


if __name__ == "__main__":
    main()
