"""Integration test: published per-episode table -> model-level uncertainty."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from conceptfan_realdata.stability import METRICS, aggregate_stability


def test_aggregate_stability_constant_episode_data(tmp_path):
    models = ("run_a", "run_b", "run_c", "run_d")
    rows = []
    for left, right in itertools.combinations(models, 2):
        for record_id in range(5):
            rows.append({
                "model_arm": "ConceptFAN-NoAlpha",
                "representation": "signed",
                "run_a": left,
                "run_b": right,
                "RecordID": record_id,
                **{metric: 0.2 for metric in METRICS},
            })
    pairs = pd.DataFrame(rows)
    episode = tmp_path / "pairs.parquet"
    pairs.to_parquet(episode, index=False)
    result = aggregate_stability(episode, pd.DataFrame(), repetitions=40)
    assert len(result) == len(METRICS)
    assert set(result["model_pairs"]) == {6}
    assert set(result["patients"]) == {5}
    np.testing.assert_allclose(result["mean"], 0.2, atol=1e-12)
    np.testing.assert_allclose(result["ci95_low_hierarchical"], 0.2, atol=1e-12)
    np.testing.assert_allclose(result["ci95_high_hierarchical"], 0.2, atol=1e-12)
