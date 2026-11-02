"""Two-way bootstrap of residual predictive gains across fixed model and patient sets."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss


def hierarchical_delta_ci(
    y: np.ndarray, p_base: np.ndarray, p_extended: np.ndarray,
    repetitions: int = 2000, seed: int = 20260717,
) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p_base = np.asarray(p_base, dtype=np.float64)
    p_extended = np.asarray(p_extended, dtype=np.float64)
    if y.ndim != 1 or p_base.shape != y.shape or p_extended.ndim != 2 or p_extended.shape[1] != len(y):
        raise ValueError("Expected labels/p_base [patients] and p_extended [models,patients]")
    if p_extended.shape[0] < 2 or np.unique(y).size != 2 or repetitions < 10:
        raise ValueError("Require >=2 models, both classes, >=10 bootstrap replicates")
    if not (np.isfinite(p_base).all() and np.isfinite(p_extended).all()):
        raise ValueError("Non-finite predictions")
    if (p_base <= 0).any() or (p_base >= 1).any() or (p_extended <= 0).any() or (p_extended >= 1).any():
        raise ValueError("Probabilities must be strictly between 0 and 1")

    def per_model_deltas(indices: np.ndarray, models: np.ndarray) -> tuple[float, float]:
        labels = y[indices]
        pb = p_base[indices]
        base_ap = average_precision_score(labels, pb)
        base_nll = log_loss(labels, pb, labels=[0, 1])
        ap = np.mean([average_precision_score(labels, p_extended[k, indices]) for k in models])
        nll = np.mean([log_loss(labels, p_extended[k, indices], labels=[0, 1]) for k in models])
        return float(ap - base_ap), float(nll - base_nll)

    n_models, n_patients = p_extended.shape
    observed = per_model_deltas(np.arange(n_patients), np.arange(n_models))
    rng = np.random.default_rng(seed)
    boot = np.empty((repetitions, 2), dtype=np.float64)
    for b in range(repetitions):
        while True:
            patients = rng.integers(n_patients, size=n_patients)
            if np.unique(y[patients]).size == 2:
                break
        models = rng.integers(n_models, size=n_models)
        boot[b] = per_model_deltas(patients, models)
    return {
        "model_count": int(n_models), "patient_count": int(n_patients),
        "bootstrap_repetitions": int(repetitions), "seed": int(seed),
        "delta_auprc_mean": observed[0],
        "delta_auprc_ci95": [float(v) for v in np.quantile(boot[:, 0], [.025, .975])],
        "delta_nll_mean": observed[1],
        "delta_nll_ci95": [float(v) for v in np.quantile(boot[:, 1], [.025, .975])],
        "sampling_unit": "trained models and shared held-out patients",
        "interpretation": "exploratory diagnostic, not causal identification or evidence of clinical readiness",
    }
