"""Independent reference-device calibration for a restricted IT3 membership.

A credible boundary uncertainty cannot be inferred from a shift of the
unknown physical source itself: its distribution may have changed. Require
a KNOWN reference input read by the device in two independently registered
calibration regimes. This source has no access to Y, synthetic latent
physiology, or TEST. For experiments only, a deterministic artificial
reference generator produces both reference readings; this is NOT evidence
that real devices offer the same calibration channel.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CalibrationEvidenceV34:
    reference_name: str
    reference_input_level: float
    medians: tuple[float,float]
    n_per_regime: tuple[int,int]
    independent_regime_identifiers: tuple[str,str]
    half_difference_boundary_uncertainty: float
    reference_calibration_verified_by_simulator: bool
    calibration_is_physical_concept_identity: bool=False


def fit_two_reference_regimes(
    first:np.ndarray,second:np.ndarray,
    *,
    known_reference_value:float,
    regime_names:tuple[str,str],
    independently_verified:bool,
)->CalibrationEvidenceV34:
    x=np.asarray(first,dtype=np.float64).reshape(-1)
    y=np.asarray(second,dtype=np.float64).reshape(-1)
    if (not independently_verified or regime_names[0]==regime_names[1]
        or not all(regime_names) or len(x)<100 or len(y)<100
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or not np.isfinite(known_reference_value)):
        raise ValueError("Independent known-reference device calibration is required")
    m1=float(np.median(x))
    m2=float(np.median(y))
    if (abs(m1-known_reference_value)>2. or abs(m2-known_reference_value)>2.):
        raise ValueError("Calibration references clearly outside trusted sensor range")
    return CalibrationEvidenceV34(
        reference_name="independently known synthetic device reference",
        reference_input_level=float(known_reference_value),
        medians=(m1,m2),
        n_per_regime=(len(x),len(y)),
        independent_regime_identifiers=regime_names,
        half_difference_boundary_uncertainty=abs(m1-m2)/2,
        reference_calibration_verified_by_simulator=True,
    )


def synthetic_known_reference_training_only(*,seed:int,disable:bool=False):
    """Artificial reference exists only in registered synthetic world."""
    if disable:
        return None
    rng=np.random.default_rng(seed)
    nominal=0.
    low=nominal-.18+rng.normal(0,.025,size=240)
    high=nominal+.18+rng.normal(0,.025,size=240)
    return fit_two_reference_regimes(
        low,high,known_reference_value=nominal,
        regime_names=("calibration_A_TRAIN","calibration_B_TRAIN"),
        independently_verified=True,
    )
