"""Frozen-split conditional diagnostic: do residuals add information beyond proxies?

No causal interpretation follows from a positive conditional prediction gain.
Never use held-out test data for feature selection, calibration, or C tuning.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def proxy_summary(concepts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-concept last, mean, max-abs, std, time slope, missing fraction.

    Uses the same six summary families as the residual diagnostic, including
    observed masks, so residual probes cannot win merely by seeing missingness.
    """
    if concepts.shape != mask.shape or concepts.ndim != 3:
        raise ValueError("concepts and masks must have shape [N,T,K]")
    count = mask.astype(bool).sum(axis=1).clip(min=1)
    observed = np.where(mask > 0, concepts, 0.0).astype(np.float64)
    hours = np.arange(concepts.shape[1], dtype=np.float64)
    features = []
    for concept in range(concepts.shape[2]):
        x = observed[:, :, concept]
        is_observed = mask[:, :, concept].astype(bool)
        n = count[:, concept]
        mean = x.sum(axis=1) / n
        largest = np.max(np.where(is_observed, np.abs(x), -np.inf), axis=1)
        largest[~np.isfinite(largest)] = 0.0
        centered_time = hours[None] - (is_observed * hours).sum(axis=1)[:, None] / n[:, None]
        centered = x - mean[:, None]
        denominator = (centered_time**2 * is_observed).sum(axis=1)
        slope = np.divide((centered_time * centered * is_observed).sum(axis=1), denominator,
                          out=np.zeros(len(x)), where=denominator > 1e-12)
        last_position = np.maximum(concepts.shape[1] - 1 - np.argmax(is_observed[:, ::-1], axis=1), 0)
        last = x[np.arange(len(x)), last_position]
        std = np.sqrt(((centered**2) * is_observed).sum(axis=1) / n)
        missing = (concepts.shape[1] - is_observed.sum(axis=1)) / concepts.shape[1]
        features.extend((last, mean, largest, std, slope, missing))
    result = np.column_stack(features)
    if not np.isfinite(result).all():
        raise ValueError("non-finite concept summary")
    return result


def fit_probe_and_predict(
    x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray,
    x_calibration: np.ndarray, y_calibration: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Select L2 strength on validation; calibrate on calibration; predict test."""
    best_score = -np.inf
    selected = None
    selected_c = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", max_iter=2000))
        model.fit(x_train, y_train)
        score = average_precision_score(y_validation, model.predict_proba(x_validation)[:, 1])
        if score > best_score:
            selected, selected_c, best_score = model, c, score
    if selected is None:
        raise RuntimeError("No logistic model selected")
    cal_scores = selected.decision_function(x_calibration).reshape(-1, 1)
    calibration = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
    calibration.fit(cal_scores, y_calibration)
    if calibration.coef_[0, 0] <= 0:
        return np.full(len(x_test), float(np.mean(y_calibration))), float(selected_c)
    probability = calibration.predict_proba(selected.decision_function(x_test).reshape(-1, 1))[:, 1]
    return np.clip(probability, 1e-8, 1.0 - 1e-8), float(selected_c)


def paired_patient_bootstrap(
    y: np.ndarray, base_probability: np.ndarray, extended_probability: np.ndarray,
    repetitions: int = 2000, seed: int = 20260717,
) -> dict:
    """Patient-level paired bootstrap: delta AUPRC and NLL for one frozen model."""
    y = np.asarray(y, dtype=np.int64)
    base_probability = np.asarray(base_probability, dtype=np.float64)
    extended_probability = np.asarray(extended_probability, dtype=np.float64)
    if not (y.ndim == base_probability.ndim == extended_probability.ndim == 1 and
            len(y) == len(base_probability) == len(extended_probability)):
        raise ValueError("Lengths of labels and probabilities must agree")
    if np.unique(y).size != 2 or not np.isfinite(base_probability).all() or not np.isfinite(extended_probability).all():
        raise ValueError("Both classes and finite probabilities are required")
    rng = np.random.default_rng(seed)
    delta_ap, delta_nll = [], []
    for _ in range(repetitions):
        index = rng.integers(len(y), size=len(y))
        if np.unique(y[index]).size != 2:
            continue
        yt, p0, p1 = y[index], base_probability[index], extended_probability[index]
        delta_ap.append(float(average_precision_score(yt, p1) - average_precision_score(yt, p0)))
        delta_nll.append(float(log_loss(yt, p1, labels=[0, 1]) - log_loss(yt, p0, labels=[0, 1])))
    if not delta_ap:
        raise RuntimeError("No valid two-class bootstrap resamples")
    return {
        "delta_auprc": float(average_precision_score(y, extended_probability) - average_precision_score(y, base_probability)),
        "delta_auprc_ci95": [float(x) for x in np.quantile(delta_ap, [0.025, 0.975])],
        "delta_nll": float(log_loss(y, extended_probability, labels=[0, 1]) - log_loss(y, base_probability, labels=[0, 1])),
        "delta_nll_ci95": [float(x) for x in np.quantile(delta_nll, [0.025, 0.975])],
        "bootstrap_valid": len(delta_ap),
        "interpretation": "associational, not causal; one frozen model",
    }
