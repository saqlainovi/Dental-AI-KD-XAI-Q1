from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dental_ai.evaluation.checkpoint_summary import summarize_checkpoint_dir, summaries_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dental AI training checkpoints.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("E:/dental_ai_checkpoints"))
    parser.add_argument("--pattern", default="v3_*_swin_t_full.pt")
    args = parser.parse_args()

    summaries = summarize_checkpoint_dir(args.checkpoint_root, args.pattern)
    if not summaries:
        print(f"No checkpoints found in {args.checkpoint_root} with pattern {args.pattern}")
        return
    print(summaries_to_markdown(summaries))


if __name__ == "__main__":
    main()
