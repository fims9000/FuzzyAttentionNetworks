"""V44: measured partial-pair cross-view fluctuation concept.

Reuses TRAIN-estimated V37 source directions, never simulator H/S/E or Y.
The reconstructed direction is not independently physically identified.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from scipy.special import expit, logit

PAIR_ORDER = ("AB", "AC", "BC")
MIN_CROSS_TIMES = 3
MIN_SENSOR_TIMES = 4
MIN_TRAIN_OVERLAP = 60


@dataclass(frozen=True)
class PairCalibrationV44:
    name: str
    center: float
    scale: float
    train_supported: int
    gauge_slope: float
    gauge_intercept: float
    gauge_valid: bool
    overlap_with_ab: int
    train_agreement_correlation: float


@dataclass(frozen=True)
class PartialViewConceptFitV44:
    v37_fit: Any
    calibrations: dict[str, PairCalibrationV44]
    disagreement_thresholds: dict[str, float | None]
    min_cross_times: int = MIN_CROSS_TIMES
    min_sensor_times: int = MIN_SENSOR_TIMES
    uses_outcomes_to_form_concept: bool = False
    contains_oracle_mixing_or_latent_source: bool = False
    independently_physically_certified: bool = False

    def audit(self) -> dict:
        return {
            "pairs": {k: asdict(v) for k, v in self.calibrations.items()},
            "pairwise_disagreement_thresholds": self.disagreement_thresholds,
            "requires_all_four_cross_products_measured": True,
            "per_pair_minimum_cross_times": self.min_cross_times,
            "per_sensor_minimum_times": self.min_sensor_times,
            "computed_only_from_observed_ON_minus_OFF": True,
            "generated_missing_observations": False,
            "outcome_labels_used_in_source_identification": False,
            "physical_semantics_certified": False,
        }


def _observed_inputs(x: np.ndarray, mask: np.ndarray):
    a = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask)
    if (a.ndim != 3 or a.shape[1:] != (16, 12) or a.shape != m.shape
            or m.dtype != np.bool_ or len(a) < 1
            or not bool(np.isfinite(a[m]).all())):
        raise ValueError("Expected strictly masked [N,16,12] measured tensor")
    return a, m


def partial_pair_energy_v44(pair_fit: Any, x, mask) -> dict:
    """Four cross moments from four distinct *actually observed* time sets.

    Each pair is centered on its own observed intersection and uses n-1.
    This trades exact V37 complete-case synchronization for a noisier
    pairwise-deletion estimator; it is not a valid MNAR guarantee.
    """
    a, m = _observed_inputs(x, mask)
    ia = tuple(pair_fit.view_a)
    ib = tuple(pair_fit.view_b)
    if len(ia) != 2 or len(ib) != 2 or len(set(ia + ib)) != 4:
        raise ValueError("Pair must contain four different channel indices")
    u = np.asarray(pair_fit.inv_left, dtype=np.float64)[0]
    v = np.asarray(pair_fit.inv_right, dtype=np.float64)[0]
    if (u.shape != (2,) or v.shape != (2,) or not np.isfinite(u).all()
            or not np.isfinite(v).all()):
        raise ValueError("Frozen TRAIN source axes are invalid")
    n = len(a)
    cross = np.zeros((n, 2, 2), dtype=np.float64)
    counts = np.zeros((n, 2, 2), dtype=np.int64)
    for i, ci in enumerate(ia):
        for j, cj in enumerate(ib):
            together = m[:, :, ci] & m[:, :, cj]
            count = together.sum(axis=1)
            counts[:, i, j] = count
            xa = np.where(together, a[:, :, ci], 0.)
            xb = np.where(together, a[:, :, cj], 0.)
            ma = xa.sum(axis=1) / np.maximum(count, 1)
            mb = xb.sum(axis=1) / np.maximum(count, 1)
            centered = (xa - ma[:, None]) * (xb - mb[:, None])
            product = np.where(together, centered, 0.).sum(axis=1)
            cross[:, i, j] = product / np.maximum(count - 1, 1)
    actual_sensor_counts = m[:, :, ia + ib].sum(axis=1)
    supported = ((counts >= MIN_CROSS_TIMES).all(axis=(1, 2))
                 & (actual_sensor_counts >= MIN_SENSOR_TIMES).all(axis=1))
    energy = np.einsum("i,nij,j->n", u, cross, v, optimize=True)
    supported &= np.isfinite(energy) & (energy >= .10)
    return {
        "energy": np.where(supported, energy, 0.),
        "support": supported,
        "pairwise_observed_times": counts,
        "minimum_observed_cross_times": counts.min(axis=(1, 2)),
        "sensor_observed_times": actual_sensor_counts,
        "all_cross_products_measured": (counts >= MIN_CROSS_TIMES).all(axis=(1, 2)),
        "no_imputation": True,
        "not_physical_semantics": True,
    }


def _q(energy, support, calibration: PairCalibrationV44):
    q = expit((energy - calibration.center) / calibration.scale)
    if calibration.name != "AB":
        if not calibration.gauge_valid:
            support = np.zeros_like(support, dtype=bool)
        q = expit(calibration.gauge_slope * logit(np.clip(q, 1e-5, 1-1e-5))
                  + calibration.gauge_intercept)
    return np.where(support, q, .5), support


def _fit_gauge(qref, qother, overlap):
    n = int(overlap.sum())
    if n < MIN_TRAIN_OVERLAP:
        return 0., 0., False, n, 0.
    xx = logit(np.clip(qother[overlap], 1e-5, 1-1e-5))
    yy = logit(np.clip(qref[overlap], 1e-5, 1-1e-5))
    xa, ya = xx-xx.mean(), yy-yy.mean()
    denom = float(np.mean(xa*xa))
    if denom < 1e-8 or float(np.mean(ya*ya)) < 1e-8:
        return 0., 0., False, n, 0.
    slope = float(np.mean(xa*ya) / (denom + 1e-6))
    intercept = float(yy.mean() - slope * xx.mean())
    correlation = float(np.mean(xa*ya) /
                        np.sqrt(float(np.mean(xa*xa)*np.mean(ya*ya))))
    passed = (np.isfinite(slope) and np.isfinite(intercept)
              and correlation >= .2 and .25 <= slope <= 4.)
    return slope, intercept, bool(passed), n, correlation


def fit_partial_view_concept_v44(v37_fit: Any, train_x, train_mask):
    """Calibrate scores and multi-pair agreement only on observed TRAIN."""
    x, mask = _observed_inputs(train_x, train_mask)
    if len(x) < 300 or set(v37_fit.pairs) != set(PAIR_ORDER):
        raise ValueError("Needs frozen three-pair V37 fit and >=300 TRAIN cases")
    energies = {name: partial_pair_energy_v44(v37_fit.pairs[name], x, mask)
                for name in PAIR_ORDER}
    base = {}
    for name in PAIR_ORDER:
        p = energies[name]
        valid = p["support"]
        if int(valid.sum()) < 60:
            raise ValueError(f"ABSTAIN: insufficient TRAIN measured evidence for {name}")
        z = p["energy"][valid]
        med = float(np.median(z))
        scale = float(max(.2, (np.quantile(z, .75)-np.quantile(z, .25))/1.349))
        base[name] = PairCalibrationV44(name, med, scale, int(valid.sum()),
                                        1., 0., name == "AB", 0, 1. if name == "AB" else 0.)
    qa, _ = _q(energies["AB"]["energy"], energies["AB"]["support"], base["AB"])
    for name in ("AC", "BC"):
        p = energies[name]
        other, _ = _q(p["energy"], p["support"],
                      PairCalibrationV44(name, base[name].center,
                                         base[name].scale, 0, 1., 0., True, 0, 0.))
        overlap = energies["AB"]["support"] & p["support"]
        slope, intercept, passed, number, correlation = _fit_gauge(qa, other, overlap)
        base[name] = PairCalibrationV44(name, base[name].center, base[name].scale,
                                        int(p["support"].sum()), slope, intercept,
                                        passed, number, correlation)
    qs = {}
    supports = {}
    for name in PAIR_ORDER:
        qs[name], supports[name] = _q(energies[name]["energy"],
                                      energies[name]["support"], base[name])
    thresholds = {}
    for i, first in enumerate(PAIR_ORDER):
        for second in PAIR_ORDER[i+1:]:
            pair_key = f"{first}|{second}"
            eligible = supports[first] & supports[second]
            if int(eligible.sum()) < MIN_TRAIN_OVERLAP:
                thresholds[pair_key] = None
            else:
                discrepancy = np.abs(qs[first][eligible]-qs[second][eligible])
                thresholds[pair_key] = float(np.clip(np.quantile(discrepancy, .95), .12, .38))
    return PartialViewConceptFitV44(v37_fit, base, thresholds)


def partial_view_concept_v44(fit: PartialViewConceptFitV44, x, mask) -> dict:
    """Select from observed pairs only, rejecting missing calibration/conflicts."""
    a, m = _observed_inputs(x, mask)
    pair = {name: partial_pair_energy_v44(fit.v37_fit.pairs[name], a, m)
            for name in PAIR_ORDER}
    q = {}
    supported = {}
    for name in PAIR_ORDER:
        q[name], supported[name] = _q(pair[name]["energy"],
                                       pair[name]["support"], fit.calibrations[name])
    n = len(a)
    result = np.full(n, .5)
    selected = np.full(n, "--", dtype="<U2")
    reason = np.full(n, "NO_MEASURED_PAIR", dtype="<U55")
    refused = np.zeros(n, dtype=bool)
    multiple = np.zeros(n, dtype=bool)
    min_times = np.zeros(n, dtype=int)
    for episode in range(n):
        names = [name for name in PAIR_ORDER if supported[name][episode]]
        if not names:
            continue
        if len(names) > 1:
            multiple[episode] = True
            conflict = False
            for i, xname in enumerate(names):
                for yname in names[i+1:]:
                    tolerance = fit.disagreement_thresholds[f"{xname}|{yname}"]
                    if tolerance is None or abs(q[xname][episode]-q[yname][episode]) > tolerance:
                        conflict = True
            if conflict:
                refused[episode] = True
                reason[episode] = "MULTIVIEW_DISAGREEMENT_OR_NO_TRAIN_CAL"
                continue
        choice = max(names, key=lambda p: (int(pair[p]["minimum_observed_cross_times"][episode]),
                                           -PAIR_ORDER.index(p)))
        selected[episode] = choice
        result[episode] = q[choice][episode]
        min_times[episode] = pair[choice]["minimum_observed_cross_times"][episode]
        reason[episode] = ("MULTIVIEW_CORROBORATED" if len(names) > 1
                           else "SINGLE_PAIR_NOT_CORROBORATED")
    use = selected != "--"
    return {
        "q": result,
        "support": use.astype(float),
        "selected_pair": selected,
        "selected_joint_measured_times": min_times,
        "single_pair_uncorroborated": use & ~multiple,
        "refused_due_to_disagreement": refused,
        "decision_reason": reason,
        "pair_memberships": np.stack([q[name] for name in PAIR_ORDER], axis=1),
        "pair_minimum_cross_times": np.stack(
            [pair[name]["minimum_observed_cross_times"] for name in PAIR_ORDER], axis=1),
        "original_sensor_observation_fraction": m[:, :, 4:10].mean(axis=(1, 2)),
        "method_uses_only_observed_values": True,
        "no_new_physical_semantics_certification": True,
    }
