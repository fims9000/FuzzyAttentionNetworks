"""Prediction-operator Hankel rank: train-only dynamic complexity diagnostic.

For a stationary observable linear state-space model
    s[t+1] = A s[t] + process_noise
    x[t]   = B s[t] + measurement_noise
the positive-lag cross-covariance block-Hankel matrix has rank <= latent
state dimension, with equality under suitable observability and excitation.
This is NOT an identifiability theorem for nonlinear or missing clinical data.

The empirical noise floor is evaluated by permuting patient correspondence
between future and past trajectories. Outcomes and oracle latents are unused.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictiveRankAudit:
    rank_estimate: int
    singular_values: np.ndarray
    null_spectral_threshold: float
    null_spectra: np.ndarray
    valid_pair_coverage: float
    assumptions: tuple[str, ...]
    passed_signal_test: bool

    def summary(self) -> dict:
        return {
            "rank_estimate": int(self.rank_estimate),
            "singular_values": self.singular_values.tolist(),
            "null_spectral_threshold": float(self.null_spectral_threshold),
            "leading_null_singular_values_q95": float(np.quantile(self.null_spectra[:, 0], .95)),
            "valid_observed_pair_coverage": float(self.valid_pair_coverage),
            "model": "positive-lag cross-covariance block-Hankel",
            "interpretation": "train-only estimated detectable predictive state rank, not true physiological factor count",
            "assumptions": list(self.assumptions),
        }


def _normalize_measured_values(
    x: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    count = mask.sum(axis=(0, 1))
    if np.any(count < 3):
        raise ValueError("Every observed channel needs >=3 measurements")
    center = (x * mask).sum(axis=(0, 1)) / count
    centered = np.where(mask > 0, x - center[None, None, :], 0.)
    mean_sq = (centered**2 * mask).sum(axis=(0, 1)) / count
    scale = np.sqrt(np.maximum(mean_sq, 1e-8))
    return centered / scale[None, None, :]


def _positive_lag_covariances(
    centered: np.ndarray, mask: np.ndarray,
    max_lag: int, future_permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lags, count_fractions = [], []
    n = len(centered)
    for lag in range(1, max_lag + 1):
        prior = centered[:, :-lag, :]
        prior_mask = mask[:, :-lag, :]
        if future_permutation is None:
            future = centered[:, lag:, :]
            future_mask = mask[:, lag:, :]
        else:
            future = centered[future_permutation, lag:, :]
            future_mask = mask[future_permutation, lag:, :]
        numerator = np.einsum("ntj,ntk->jk", future, prior, optimize=True)
        counts = np.einsum("ntj,ntk->jk", future_mask, prior_mask, optimize=True)
        valid = counts > max(4, n * (centered.shape[1] - lag) * .002)
        cov = np.divide(numerator, counts, out=np.zeros_like(numerator), where=valid)
        lags.append(cov)
        count_fractions.append(valid.mean())
    return np.stack(lags), np.asarray(count_fractions)


def _block_hankel(covariances: np.ndarray, block_lags: int) -> np.ndarray:
    return np.block([
        [covariances[i+j] for j in range(block_lags)]
        for i in range(block_lags)
    ])


def estimate_predictive_hankel_rank(
    observed_values: np.ndarray,
    observed_mask: np.ndarray,
    *,
    block_lags: int = 3,
    null_repetitions: int = 100,
    max_rank: int = 16,
    significance_level: float = 0.05,
    seed: int = 20261102,
) -> PredictiveRankAudit:
    x = np.asarray(observed_values, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=np.float64)
    if (x.ndim != 3 or x.shape != mask.shape or x.shape[0] < 30
            or x.shape[1] < 2 * block_lags + 1 or x.shape[2] < 2):
        raise ValueError("Expected >=30 time series and sufficient channels/timestamps")
    if not np.isfinite(x).all() or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("Nonfinite measurements or invalid observation masks")
    if block_lags < 2 or null_repetitions < 20 or not 0 < significance_level < .5 or max_rank < 1:
        raise ValueError("Invalid lag/replication/alpha/rank parameters")

    centered = _normalize_measured_values(x, mask)
    covs, coverage = _positive_lag_covariances(centered, mask, 2 * block_lags - 1)
    spectrum = np.linalg.svd(_block_hankel(covs, block_lags), compute_uv=False)
    rng = np.random.default_rng(seed)
    null_spectra = []
    for _ in range(null_repetitions):
        perm = rng.permutation(len(x))
        covs_shuffled, _ = _positive_lag_covariances(
            centered, mask, 2 * block_lags - 1, future_permutation=perm
        )
        null_spectra.append(np.linalg.svd(_block_hankel(covs_shuffled, block_lags), compute_uv=False))
    null = np.asarray(null_spectra)
    # Comparing every singular value with a single null spectral-norm
    # threshold is a conservative simultaneous familywise rule.
    threshold = float(np.quantile(null[:, 0], 1. - significance_level))
    rank = int(min(max_rank, np.count_nonzero(spectrum > threshold)))
    return PredictiveRankAudit(
        rank_estimate=rank,
        singular_values=spectrum,
        null_spectral_threshold=threshold,
        null_spectra=null,
        valid_pair_coverage=float(np.mean(coverage)),
        assumptions=(
            "stationarity or small temporal drift",
            "approximately low-dimensional linear dynamics over measured window",
            "observable persistent modes and sufficient randomized excitation",
            "patient trajectories independent in train",
            "measurement process reasonably captured by explicit masks",
            "false positives possible from nonstationarity, latent confounding or informative missingness",
        ),
        passed_signal_test=rank>0,
    )
