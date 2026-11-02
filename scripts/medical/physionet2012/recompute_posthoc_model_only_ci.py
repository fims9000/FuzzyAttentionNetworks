"""Model-level CIs from the checked-in post-hoc pairwise summary CSV.

This is a complete reanalysis conditional on the *fixed* 600 held-out
patients. It is deliberately distinct from the two-way bootstrap requiring
per-episode frozen artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from conceptfan_realdata.model_pair_bootstrap import model_pair_bootstrap_ci

METRICS = ("spearman", "kendall", "top1_agreement", "top3_jaccard", "sign_agreement", "cosine_similarity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "reports/physionet2012/posthoc_attribution_audit/TABLES/posthoc_pairwise_stability.csv"
    ))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261102)
    a = parser.parse_args()
    if a.out.exists():
        parser.error(f"Refusing to overwrite: {a.out}")
    src = pd.read_csv(a.input)
    src = src[(src["feature_space"] == "proxy_concept_v_only") &
              (src["value_view"] == "signed")]
    rows = []
    for method, group in src.groupby("method", sort=True):
        group = group.sort_values(["model_a", "model_b"])
        pair_keys = list(zip(group["model_a"], group["model_b"]))
        if len(pair_keys) != 435:
            raise ValueError(f"{method}: expected all 435 pairs")
        for metric_index, metric in enumerate(METRICS):
            values = group[metric].to_numpy(dtype=float)
            low, high = model_pair_bootstrap_ci(
                values, pair_keys, a.repetitions,
                np.random.default_rng(a.seed + metric_index),
            )
            rows.append({
                "method": method, "metric": metric,
                "mean": float(values.mean()),
                "model_only_ci95_low": low, "model_only_ci95_high": high,
                "model_pairs": 435, "n_models": 30, "heldout_patient_count_fixed": 600,
                "bootstrap_repetitions": a.repetitions,
                "bootstrap_scope": "model-only, conditional on fixed held-out cohort",
            })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Wrote {len(rows)} model-only CI rows to {a.out}")


if __name__ == "__main__":
    main()
