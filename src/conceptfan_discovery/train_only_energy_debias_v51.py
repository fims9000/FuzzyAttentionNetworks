"""V51: TRAIN-only energy magnitude calibration against noisy independent assay.

Assay reference is NOT hidden latent truth: it is observed external source1
with separately known metrology variance sigma_eta^2. A scale and an
affine correction are both admitted/refused on two disjoint VALID cohorts.
No TEST source truth, Y or hidden H/S is used for fitting/candidate choice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from sklearn.linear_model import Ridge

from .independent_anchor_certificate_v50 import (
    AttestedSemanticCalibrationV50,infer_calibrated_source_energy_v50,
)

ASSAY_NOISE_STD=.16
ASSAY_NOISE_VARIANCE=ASSAY_NOISE_STD**2
VALID_ERROR_SLACK=.015
MIN_VALID_ASSAY_EPISODES=80
SLOPE_LIMITS=(.5,2.)
AFFINE_INTERCEPT_LIMITS=(-.30,.30)


def energy_reference_from_independent_assay_v51(assay):
    a=np.asarray(assay,dtype=np.float64)
    if (a.ndim!=2 or a.shape[1]!=16 or len(a)<1
            or not np.isfinite(a).all()):
        raise ValueError("Expected physically independent 16-step external energy assay")
    return np.maximum(0.,np.var(a,axis=1,ddof=1)-ASSAY_NOISE_VARIANCE)


@dataclass(frozen=True)
class TrainOnlyEnergyDebiasV51:
    physical_device_id:str
    external_reference_id:str
    train_anchor_count:int
    valid1_anchor_count:int
    valid2_anchor_count:int
    raw_energy_train_mean:float
    reference_energy_train_mean:float
    scale_alpha:float
    affine_alpha:float
    affine_intercept:float
    valid1_raw_assay_MAE:float
    valid2_raw_assay_MAE:float
    valid1_scale_assay_MAE:float
    valid2_scale_assay_MAE:float
    valid1_affine_assay_MAE:float
    valid2_affine_assay_MAE:float
    scale_approved_two_VALID:bool
    affine_approved_two_VALID:bool
    external_assay_noise_variance:float=ASSAY_NOISE_VARIANCE
    calibrated_only_from_TRAIN_and_two_independent_VALID:bool=True
    target_is_noisy_assay_not_hidden_true_energy:bool=True
    physical_identity_not_proven:bool=True

    def audit(self):
        return asdict(self)


def _correct(raw,*,kind:str,scale:float,affine_alpha:float,intercept:float):
    r=np.asarray(raw,dtype=np.float64)
    if kind=="V50_RAW":
        return r.copy()
    if kind=="V51_SCALE":
        return scale*r
    if kind=="V51_AFFINE":
        return np.maximum(0.,affine_alpha*r+intercept)
    raise ValueError("Unknown frozen energy correction")


def fit_train_only_energy_debias_v51(
    source_calibration:AttestedSemanticCalibrationV50,
    train_x,train_mask,train_independent_assay,
    valid1_x,valid1_mask,valid1_independent_assay,
    valid2_x,valid2_mask,valid2_independent_assay,
    *,train_reference_budget_fraction:float,seed:int,
):
    """Must use SAME paid physical TRAIN samples as source trajectory V50."""
    if train_reference_budget_fraction not in (.05,.20,1.):
        raise ValueError("Only preregistered 5/20/100 percent source assay budgets")
    if not source_calibration.dual_VALID_admitted:
        raise ValueError("Cannot debias a rejected or unknown physical source calibration")
    tr=np.asarray(train_x)
    atr=np.asarray(train_independent_assay,dtype=np.float64)
    if len(tr)<500 or atr.shape!=(len(tr),16):
        raise ValueError("Incomplete and incompatible TRAIN reference")
    ntr=int(np.floor(train_reference_budget_fraction*len(tr)))
    if ntr!=source_calibration.train_assay_episodes:
        raise AssertionError("Energy correction must NOT buy additional TRAIN assay samples")
    tr_ids=np.sort(np.random.default_rng(seed+1).choice(len(tr),size=ntr,replace=False))
    predicted=infer_calibrated_source_energy_v50(
        source_calibration,tr[tr_ids],train_mask[tr_ids])["inferred_target_energy"]
    reference=energy_reference_from_independent_assay_v51(atr[tr_ids])
    denom=float(np.sum(predicted*predicted)+1e-8)
    alpha=float(np.clip(np.sum(predicted*reference)/denom,*SLOPE_LIMITS))
    ridge=Ridge(alpha=5.,fit_intercept=True)
    ridge.fit(predicted.reshape(-1,1),reference)
    affine_slope=float(np.clip(float(ridge.coef_[0]),*SLOPE_LIMITS))
    affine_intercept=float(np.clip(float(ridge.intercept_),*AFFINE_INTERCEPT_LIMITS))
    valid_errors=[]
    for fold,(x,m,a) in enumerate((
        (valid1_x,valid1_mask,valid1_independent_assay),
        (valid2_x,valid2_mask,valid2_independent_assay),
    ),start=1):
        if len(x)<150 or len(a)!=len(x):
            raise ValueError("Incomplete independent source assay VALID")
        idx=np.sort(np.random.default_rng(seed+fold*41).choice(
            len(x),size=MIN_VALID_ASSAY_EPISODES,replace=False))
        raw=infer_calibrated_source_energy_v50(
            source_calibration,x[idx],m[idx])["inferred_target_energy"]
        physical=energy_reference_from_independent_assay_v51(
            np.asarray(a)[idx])
        valid_errors.append({
            name:float(np.mean(np.abs(_correct(
                raw,kind=name,scale=alpha,affine_alpha=affine_slope,
                intercept=affine_intercept)-physical)))
            for name in ("V50_RAW","V51_SCALE","V51_AFFINE")
        })
    scale_approved=all(fold["V51_SCALE"]<=fold["V50_RAW"]+VALID_ERROR_SLACK
                       for fold in valid_errors)
    affine_approved=all(fold["V51_AFFINE"]<=fold["V50_RAW"]+VALID_ERROR_SLACK
                        for fold in valid_errors)
    return TrainOnlyEnergyDebiasV51(
        physical_device_id=source_calibration.physical_device_id,
        external_reference_id=source_calibration.independent_reference_id,
        train_anchor_count=ntr,valid1_anchor_count=MIN_VALID_ASSAY_EPISODES,
        valid2_anchor_count=MIN_VALID_ASSAY_EPISODES,
        raw_energy_train_mean=float(np.mean(predicted)),
        reference_energy_train_mean=float(np.mean(reference)),
        scale_alpha=alpha,
        affine_alpha=affine_slope,
        affine_intercept=affine_intercept,
        valid1_raw_assay_MAE=valid_errors[0]["V50_RAW"],
        valid2_raw_assay_MAE=valid_errors[1]["V50_RAW"],
        valid1_scale_assay_MAE=valid_errors[0]["V51_SCALE"],
        valid2_scale_assay_MAE=valid_errors[1]["V51_SCALE"],
        valid1_affine_assay_MAE=valid_errors[0]["V51_AFFINE"],
        valid2_affine_assay_MAE=valid_errors[1]["V51_AFFINE"],
        scale_approved_two_VALID=scale_approved,
        affine_approved_two_VALID=affine_approved,
    )


def infer_energy_with_train_only_debias_v51(
    source_calibration:AttestedSemanticCalibrationV50,
    energy_debias:TrainOnlyEnergyDebiasV51,x,mask,
    *,mode:str,
):
    if (source_calibration.physical_device_id!=energy_debias.physical_device_id
            or source_calibration.independent_reference_id!=energy_debias.external_reference_id):
        raise ValueError("External physical assay energy correction cannot move across devices")
    raw=infer_calibrated_source_energy_v50(
        source_calibration,x,mask)["inferred_target_energy"]
    if mode=="V51_SCALE":
        approved=energy_debias.scale_approved_two_VALID
    elif mode=="V51_AFFINE":
        approved=energy_debias.affine_approved_two_VALID
    elif mode=="V50_RAW":
        approved=True
    else:
        raise ValueError("Unknown frozen registered source energy variant")
    corrected=_correct(
        raw,kind=mode,scale=energy_debias.scale_alpha,
        affine_alpha=energy_debias.affine_alpha,
        intercept=energy_debias.affine_intercept) if approved else raw.copy()
    if not approved and not np.array_equal(raw,corrected):
        raise AssertionError("Failed two-VALID energy correction MUST be exact V50 source")
    return {
        "energy":corrected,
        "original_uncorrected_energy":raw,
        "requested_correction":mode,
        "correction_admitted":bool(approved),
        "correction_actually_used":bool(approved and mode!="V50_RAW"),
        "decision":("TRAIN_AND_BOTH_VALID_APPROVED" if approved else
                    "CORRECTION_REJECTED_EXACT_RAW_FALLBACK"),
        "oracle_latent_true_energy_used":False,
        "predictor_or_native_ConceptFAN_risk_modified":False,
    }
