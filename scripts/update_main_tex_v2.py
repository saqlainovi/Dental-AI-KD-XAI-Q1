"""Validate the hand-maintained V2 manuscript against its source artefacts.

The previous version of this file overwrote ``main.tex`` with stale,
hard-coded claims.  The manuscript is now deliberately hand-maintained and
this script performs read-only checks instead of regenerating prose.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "Dental_AI_Overleaf_Package" / "main.tex"
BIB_PATH = ROOT / "Dental_AI_Overleaf_Package" / "references.bib"
SEG_PATH = ROOT / "outputs" / "v2_segmentation" / "segformer_tooth_seg_final_eval.csv"
DISEASE_PATH = ROOT / "outputs" / "v2_disease_classification" / "disease_test_per_class_results.csv"


def cited_keys(tex: str) -> set[str]:
    return {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }


def bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", bib))


def main() -> int:
    tex = TEX_PATH.read_text(encoding="utf-8")
    bib = BIB_PATH.read_text(encoding="utf-8")
    seg = pd.read_csv(SEG_PATH)
    disease = pd.read_csv(DISEASE_PATH)

    missing_citations = cited_keys(tex) - bib_keys(bib)
    figure_paths = set(re.findall(r"\{(figures/[^}]+)\}", tex))
    missing_figures = [path for path in figure_paths if not (TEX_PATH.parent / path).exists()]

    internal_dice = float(seg.loc[seg["cohort"] == "DentalAI Internal Test", "dice_mean"].iloc[0])
    macro_f1 = float(disease["f1"].mean())
    expected_numbers = [f"{internal_dice:.4f}", f"{macro_f1:.4f}"]
    absent_numbers = [value for value in expected_numbers if value not in tex]

    if "Distillation" in tex or "Knowledge Distillation" in tex:
        print("FAIL: the V2 manuscript contains an unsupported distillation claim.")
        return 1
    if "Auto-Approved" in tex or "enables autonomous triage" in tex:
        print("FAIL: the V2 manuscript contains an unsupported autonomous-triage claim.")
        return 1
    if missing_citations:
        print("FAIL: missing bibliography keys:", ", ".join(sorted(missing_citations)))
        return 1
    if missing_figures:
        print("FAIL: missing figures:", ", ".join(sorted(missing_figures)))
        return 1
    if absent_numbers:
        print("FAIL: manuscript does not contain source metrics:", ", ".join(absent_numbers))
        return 1

    print("PASS: V2 manuscript citation, figure, and core-metric checks passed.")
    print(f"Verified internal Dice={internal_dice:.4f}; macro-F1={macro_f1:.4f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
