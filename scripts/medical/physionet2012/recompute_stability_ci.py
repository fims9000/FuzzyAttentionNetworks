"""Recalculate corrected CIs from frozen evaluations without training models.

This command deliberately does not overwrite the published report tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from conceptfan_realdata.stability import aggregate_stability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    tables = args.artifacts_dir / "TABLES"
    episode = tables / "episode_pairwise_stability.parquet"
    pairs = tables / "stability_pair_summary.parquet"
    if not episode.is_file() or not pairs.is_file():
        parser.error("Missing frozen pairwise artifacts; do not reconstruct from aggregate CSVs")
    result = aggregate_stability(episode, pd.read_parquet(pairs), args.repetitions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print("Wrote corrected model-level intervals to", args.out)
    print("No frozen tables, checkpoints, or previously published metrics were modified")


if __name__ == "__main__":
    main()
