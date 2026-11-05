"""Synthetic-oracle audit: omitted concept evidence vs concept estimation error.

This is not an estimator of causal/semantic completeness on clinical data.
The result measures incremental held-out predictive information under a
predeclared logistic probe family. No test labels are used for fitting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class OracleCoverageAudit:
    n_train: int
    n_test: int
    concept_only_logloss: float
    concept_plus_residual_logloss: float
    oracle_residual_gain: float
    oracle_residual_gain_ci95: tuple[float, float]
    predicted_concept_logloss: float
    corrected_concept_logloss: float
    concept_estimation_gain: float
    concept_estimation_gain_ci95: tuple[float, float]
    residual_gain_positive_ci: bool
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_matrix(x: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite [N,features] matrix")
    return out


def _frozen_probe(x_train: np.ndarray, labels_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(x_train, labels_train)
    return np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)


def _per_sample_logloss(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return -(labels * np.log(probability) + (1.0 - labels) * np.log1p(-probability))


def _paired_bootstrap_ci(
    individual_gain: np.ndarray, *,
    seed: int, replicates: int,
) -> tuple[float, float]:
    if replicates < 199:
        raise ValueError("At least 199 bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    sample_size = len(individual_gain)
    draws = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        sample = rng.integers(0, sample_size, size=sample_size)
        draws[b] = float(individual_gain[sample].mean())
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def audit_oracle_concept_coverage(
    *,
    labels: np.ndarray,
    predicted_concepts: np.ndarray,
    true_concepts: np.ndarray,
    residual_candidate: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int = 20261105,
    bootstrap_replicates: int = 999,
) -> OracleCoverageAudit:
    """Separately inspect two possible failure modes on synthetic ground truth.

    `concept_estimation_gain` is the benefit of restoring true concepts from
    predicted concepts. `oracle_residual_gain` is incremental risk information
    in a candidate residual AFTER true concepts have already been provided.
    Only the second comparison could motivate investigating an omitted factor;
    even then it does not establish a new physical concept or causality.
    """
    pred = _as_matrix(predicted_concepts, "predicted_concepts")
    gold = _as_matrix(true_concepts, "true_concepts")
    candidate = _as_matrix(residual_candidate, "residual_candidate")
    if gold.shape != pred.shape or len(candidate) != len(gold):
        raise ValueError("Mismatched cohort or concept dimensions")
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != (len(gold),) or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Expected aligned binary outcomes")
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if train.ndim != 1 or test.ndim != 1 or len(train) < 100 or len(test) < 100:
        raise ValueError("TRAIN and TEST must each contain at least 100 distinct cases")
    if (not np.array_equal(np.unique(train), np.sort(train))
        or not np.array_equal(np.unique(test), np.sort(test))
        or np.intersect1d(train, test).size > 0
        or min(train.min(), test.min()) < 0
        or max(train.max(), test.max()) >= len(y)):
        raise ValueError("Disjoint, unique, in-bounds TRAIN and TEST indices are required")
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        raise ValueError("Both classes are required on TRAIN and TEST")
    if bootstrap_replicates < 199:
        raise ValueError("Insufficient bootstrap repetitions")

    oracle_extra = np.concatenate((gold, candidate), axis=1)
    corrected_pred = np.concatenate((pred, gold - pred), axis=1)
    p_gold = _frozen_probe(gold[train], y[train], gold[test])
    p_gold_extra = _frozen_probe(oracle_extra[train], y[train], oracle_extra[test])
    p_pred = _frozen_probe(pred[train], y[train], pred[test])
    p_pred_corrected = _frozen_probe(corrected_pred[train], y[train], corrected_pred[test])

    oracle_loss = _per_sample_logloss(y[test], p_gold)
    oracle_extra_loss = _per_sample_logloss(y[test], p_gold_extra)
    predicted_loss = _per_sample_logloss(y[test], p_pred)
    corrected_loss = _per_sample_logloss(y[test], p_pred_corrected)
    oracle_gain = oracle_loss - oracle_extra_loss
    estimation_gain = predicted_loss - corrected_loss
    oracle_ci = _paired_bootstrap_ci(oracle_gain, seed=seed, replicates=bootstrap_replicates)
    estimation_ci = _paired_bootstrap_ci(estimation_gain, seed=seed+1, replicates=bootstrap_replicates)
    positive = bool(oracle_ci[0] > 0.0)
    return OracleCoverageAudit(
        n_train=len(train), n_test=len(test),
        concept_only_logloss=float(oracle_loss.mean()),
        concept_plus_residual_logloss=float(oracle_extra_loss.mean()),
        oracle_residual_gain=float(oracle_gain.mean()),
        oracle_residual_gain_ci95=oracle_ci,
        predicted_concept_logloss=float(predicted_loss.mean()),
        corrected_concept_logloss=float(corrected_loss.mean()),
        concept_estimation_gain=float(estimation_gain.mean()),
        concept_estimation_gain_ci95=estimation_ci,
        residual_gain_positive_ci=positive,
        interpretation=(
            "Held-out candidate evidence beyond known oracle concepts under a fixed "
            "logistic probe; investigate omitted factors and probe misspecification."
            if positive else
            "No positive held-out oracle-residual gain resolved by this probe; "
            "absence of evidence is not proof of a complete concept dictionary."
        ),
    )
