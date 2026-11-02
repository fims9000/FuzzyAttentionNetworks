"""Empirical agreement nulls for frozen independently trained concept models.

Permuting patient correspondence *within each run* keeps each model's
concept-label and sign frequencies intact. It removes cross-run alignment
for identical patients. This is a descriptive alignment null, not a claim
that concept ranks are uniformly distributed or a test of causal validity.
"""
from __future__ import annotations

import itertools

import numpy as np


def agreement_nulls(
    contributions: np.ndarray,
    repetitions: int = 1000,
    seed: int = 20260717,
) -> dict:
    """Compare shared-patient Top-1/sign agreement against shuffled patients.

    The tensor must have shape [models, held-out patients, concepts] with
    identical patient ordering for every model.
    """
    a = np.asarray(contributions, dtype=np.float64)
    if a.ndim != 3 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite [model, episode, concept] tensor")
    if repetitions < 10:
        raise ValueError("At least ten null permutations required")
    n_models, n_episodes, n_concepts = a.shape
    left, right = np.triu_indices(n_models, k=1)
    top = np.argmax(np.abs(a), axis=-1)
    signs = np.sign(a).astype(np.int8)
    observed_top = float(np.mean(top[left] == top[right]))
    observed_sign = float(np.mean(signs[left] == signs[right]))

    # Independent-episode expectation conditional on each model's own
    # empirical label/sign frequencies. This need not equal 1/K or 1/2.
    top_freq = np.stack([(top == k).mean(axis=1) for k in range(n_concepts)], axis=1)
    expected_top = float((top_freq[left] * top_freq[right]).sum(axis=1).mean())
    sign_freq = np.stack([(signs == v).mean(axis=1) for v in (-1, 0, 1)], axis=-1)
    expected_sign = float(
        (sign_freq[left] * sign_freq[right]).sum(axis=(2, 3)).mean() / n_concepts
    )

    rng = np.random.default_rng(seed)
    null_top = np.empty(repetitions, dtype=np.float64)
    null_sign = np.empty(repetitions, dtype=np.float64)
    model_axis = np.arange(n_models)[:, None]
    for b in range(repetitions):
        patient_shuffle = np.stack([rng.permutation(n_episodes) for _ in range(n_models)])
        shuffled_top = top[model_axis, patient_shuffle]
        shuffled_sign = signs[model_axis, patient_shuffle, :]
        null_top[b] = np.mean(shuffled_top[left] == shuffled_top[right])
        null_sign[b] = np.mean(shuffled_sign[left] == shuffled_sign[right])
    def summarise(observed: float, expectation: float, null: np.ndarray) -> dict:
        return {
            "observed": observed,
            "marginal_frequency_null": expectation,
            "permutation_null_mean": float(np.mean(null)),
            "permutation_null_q025_q975": [float(v) for v in np.quantile(null, (0.025, 0.975))],
            "observed_minus_null_mean": float(observed - np.mean(null)),
            "permutation_p_greater": float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
        }
    return {
        "models": n_models,
        "model_pairs": len(left),
        "patients": n_episodes,
        "concepts": n_concepts,
        "permutations": repetitions,
        "seed": seed,
        "control": "independent within-model patient permutation; preserves each model's per-concept marginals",
        "top1": summarise(observed_top, expected_top, null_top),
        "sign": summarise(observed_sign, expected_sign, null_sign),
        "interpretation": "exploratory episode-alignment null, not evidence of clinical or causal validity",
    }
