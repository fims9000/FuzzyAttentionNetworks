"""V45: measured OFF-channel condition sentinel; no extra physical sensor or truth."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

SOURCE_OFF_CHANNELS = (4, 5, 6, 7, 8, 9)
MIN_TIMES = 4
MIN_TRAIN = 300
MIN_SUPPORTED_FRACTION = .90
MAX_FALSE_ALARM_VALID = .05
TRAIN_THRESHOLD_QUANTILE = .99


def measured_off_dynamic_energy(off, observed_mask):
    """Actual OFF samples only. Episode channel variance, n-1 denominator."""
    a = np.asarray(off, dtype=np.float64)
    m = np.asarray(observed_mask)
    if (a.ndim != 3 or a.shape[1:] != (16, 12) or a.shape != m.shape
            or len(a) == 0 or m.dtype != np.bool_
            or not np.isfinite(a[m]).all()):
        raise ValueError("Expected 12 strictly masked OFF channels, including 6 monitored")
    values = np.where(m[:, :, SOURCE_OFF_CHANNELS], a[:, :, SOURCE_OFF_CHANNELS], 0.)
    observed = m[:, :, SOURCE_OFF_CHANNELS]
    count = observed.sum(axis=1)
    avg = values.sum(axis=1) / np.maximum(count, 1)
    centered = np.where(observed, values - avg[:, None, :], 0.)
    variance = (centered * centered).sum(axis=1) / np.maximum(count - 1, 1)
    score = variance.mean(axis=1)
    supported = (count >= MIN_TIMES).all(axis=1)
    return {
        "score": np.where(supported, score, 0.),
        "support": supported,
        "min_actual_OFF_times": count.min(axis=1),
        "n_observed_OFF_samples": count.sum(axis=1),
        "uses_only_observed_OFF_measurements": True,
        "not_physical_identity_or_measurement_correctness_certificate": True,
    }


@dataclass(frozen=True)
class MeasuredOFFGuardV45:
    threshold: float
    train_supported: int
    train_count: int
    train_healthy_quantile: float
    valid1_supported: float
    valid2_supported: float
    valid1_false_alarm: float
    valid2_false_alarm: float
    verified_on_two_independent_healthy_VALID: bool
    no_Y_R_H_S_U_used: bool = True
    physical_identity_certified: bool = False

    def audit(self):
        return asdict(self)


def fit_off_guard_v45(train_off, train_mask, valid1_off, valid1_mask,
                      valid2_off, valid2_mask) -> MeasuredOFFGuardV45:
    """99th healthy TRAIN percentile and TWO disjoint healthy VALID checks."""
    train = measured_off_dynamic_energy(train_off, train_mask)
    n = len(train["support"])
    if n < MIN_TRAIN or train["support"].mean() < MIN_SUPPORTED_FRACTION:
        raise ValueError("INSUFFICIENT_HEALTHY_TRAIN_OFF_OBSERVATIONS")
    threshold = float(np.quantile(train["score"][train["support"]],
                                   TRAIN_THRESHOLD_QUANTILE))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("INVALID_HEALTHY_TRAIN_OFF_THRESHOLD")

    measures = []
    for off, mask in ((valid1_off, valid1_mask), (valid2_off, valid2_mask)):
        r = measured_off_dynamic_energy(off, mask)
        support = r["support"]
        observed_fraction = float(support.mean())
        false_alarms = (float(np.mean(r["score"][support] > threshold))
                        if support.any() else 1.)
        measures.append((observed_fraction, false_alarms))
    admitted = all(x >= MIN_SUPPORTED_FRACTION and y <= MAX_FALSE_ALARM_VALID
                   for x, y in measures)
    return MeasuredOFFGuardV45(
        threshold=threshold,train_supported=int(train["support"].sum()),
        train_count=n,train_healthy_quantile=TRAIN_THRESHOLD_QUANTILE,
        valid1_supported=measures[0][0], valid2_supported=measures[1][0],
        valid1_false_alarm=measures[0][1], valid2_false_alarm=measures[1][1],
        verified_on_two_independent_healthy_VALID=admitted,
    )


def apply_off_guard_v45(fit:MeasuredOFFGuardV45, off, observed_mask):
    evidence = measured_off_dynamic_energy(off, observed_mask)
    support = evidence["support"]
    alarm = support & (evidence["score"] > fit.threshold)
    usable = support & ~alarm & fit.verified_on_two_independent_healthy_VALID
    reasons = np.where(
        ~support, "NO_CURRENT_OFF_REFERENCE",
        np.where(~fit.verified_on_two_independent_healthy_VALID,
                 "NO_HEALTHY_DUAL_VALID_CERTIFICATE",
                 np.where(alarm, "OFF_BACKGROUND_OUT_OF_TRAIN_RANGE",
                          "CURRENT_OFF_BACKGROUND_WITHIN_TRAIN_RANGE")))
    return {
        **evidence,
        "usable_for_additional_concept": usable,
        "off_background_alarm": alarm,
        "decision_reason": reasons,
        "healthy_train_threshold": fit.threshold,
        "source_reliability_not_source_identity": True,
    }


def gate_partial_view_source_v45(source:dict, monitor:dict):
    """Unreliable OFF implies exactly neutral concept q=.5 and source support 0."""
    q = np.asarray(source["q"], dtype=np.float64)
    support = np.asarray(source["support"]) > 0
    usable = np.asarray(monitor["usable_for_additional_concept"], dtype=bool)
    if q.ndim != 1 or support.shape != q.shape or usable.shape != q.shape:
        raise ValueError("Misaligned predicted concept and OFF background readings")
    accepted = support & usable
    result = dict(source)
    result["q"] = np.where(accepted, q, .5)
    result["support"] = accepted.astype(float)
    result["off_monitor_rejected"] = support & ~usable
    result["monitor_reason"] = monitor["decision_reason"]
    result["new_source_is_not_physically_certified"] = True
    return result
