"""Model-level bootstrap for dependent all-pairs retraining comparisons.

The sampling unit is a trained model, never one of its overlapping pairs.
For episode-level arrays, patients are resampled *once per replicate* across
all selected pairs, preserving the shared evaluation-cohort dependence.
"""

from __future__ import annotations

import math
from typing import Hashable, Sequence

import numpy as np


def model_pair_bootstrap_ci(
    values: np.ndarray,
    model_pairs: Sequence[tuple[Hashable, Hashable]],
    repetitions: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 32,
) -> tuple[float, float]:
    """95% percentile CI for the mean over distinct pairs of fitted models.

    ``values`` is (pairs,) or (pairs, patients). Every unordered pair of the
    same model set must occur exactly once, in the same order as the rows.
    Resampled copies of the *same* model are never treated as independent
    comparison pairs: their self-pairs are excluded from the statistic.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != len(model_pairs):
        raise ValueError("values must have one row per model pair")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be nonempty and finite")
    if repetitions < 2 or chunk_size < 1:
        raise ValueError("repetitions must be >=2 and chunk_size >=1")
    if values.ndim == 2 and values.shape[1] < 1:
        raise ValueError("episode dimension must not be empty")

    ids = sorted({identifier for pair in model_pairs for identifier in pair}, key=str)
    n_models = len(ids)
    if n_models < 2 or len(model_pairs) != math.comb(n_models, 2):
        raise ValueError("all distinct model pairs must be present")
    index = {identifier: k for k, identifier in enumerate(ids)}
    endpoints = np.asarray([(index[a], index[b]) for a, b in model_pairs], dtype=np.int64)
    if np.any(endpoints[:, 0] == endpoints[:, 1]):
        raise ValueError("self-pairs are not allowed")
    normalized = [tuple(sorted(edge)) for edge in endpoints.tolist()]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate model pair")

    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        sampled = rng.integers(n_models, size=(count, n_models))
        counts = np.zeros((count, n_models), dtype=np.int64)
        np.add.at(counts, (np.arange(count)[:, None], sampled), 1)
        weights = counts[:, endpoints[:, 0]] * counts[:, endpoints[:, 1]]
        totals = weights.sum(axis=1)
        # Rare when a bootstrap draws the same model n times. Redraw rather
        # than assign an artificial self-comparison with perfect agreement.
        while np.any(totals == 0):
            missing = np.flatnonzero(totals == 0)
            sampled[missing] = rng.integers(n_models, size=(len(missing), n_models))
            counts[missing] = 0
            np.add.at(counts, (missing[:, None], sampled[missing]), 1)
            weights[missing] = counts[missing][:, endpoints[:, 0]] * counts[missing][:, endpoints[:, 1]]
            totals[missing] = weights[missing].sum(axis=1)
        if values.ndim == 1:
            estimates[start : start + count] = weights @ values / totals
        else:
            weighted_patient_scores = (weights @ values) / totals[:, None]
            patient_ids = rng.integers(values.shape[1], size=(count, values.shape[1]))
            estimates[start : start + count] = np.take_along_axis(
                weighted_patient_scores, patient_ids, axis=1
            ).mean(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))
