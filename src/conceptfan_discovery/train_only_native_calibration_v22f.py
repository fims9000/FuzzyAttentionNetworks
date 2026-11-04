"""V22f: TRAIN-only two-scalar probability calibration of a NATIVE risk logit.

A medically interpretable concept cannot be retroactively created by a
probability calibration layer. This module preserves the ENTIRE original
native concept decision and multiplies all native *signed* direct
contributions by one positive common scalar; it changes calibration, not
concept identity, intervention semantics or a physical causal claim.

Source and model fitting and 200-label independent task-risk screen must be
complete BEFORE calling fit_train_only_native_affine_calibration on the 150
separately reserved TRAIN calibration episodes.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

MIN_SLOPE=.05
MAX_SLOPE=50.
MAX_ABS_INTERCEPT=6.
REGULARIZATION=.002
OPTIMIZATION_STEPS=200


@dataclass(frozen=True)
class NativeAffineProbabilityCalibration:
    slope: float
    intercept: float
    n_independent_train_calibration_cases: int
    initial_train_bce: float
    fitted_train_bce: float
    optimizer_success: bool
    regularization: float = REGULARIZATION
    standard_positive_logit_slope: bool = True
    uses_original_native_logit_only: bool = True
    new_concept_head_created: bool = False
    clinically_meaningful_concepts_certified: bool = False

    def transform_logits(self,logits:np.ndarray)->np.ndarray:
        z=np.asarray(logits,dtype=np.float64)
        if not np.isfinite(z).all():
            raise ValueError("Cannot calibrate nonfinite native score")
        return self.slope*z+self.intercept

    def probabilities(self,logits:np.ndarray)->np.ndarray:
        return expit(self.transform_logits(logits))

    def decomposed_native_contributions(self,out)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """Return (calibrated logit, calibrated bias, physical, observation).

        The original native forward pass is REQUIRED: external logits alone
        cannot recreate concept explanations. There is no additional input
        predictor, latent hidden vector or modified slot interpretation.
        """
        raw=out.logits
        if float(out.decomposition_error().detach())>2e-5:
            raise AssertionError("Native additive logit invalid before calibration")
        bias=self.slope*out.physical.decision_bias+self.intercept
        physical=self.slope*out.physical.contributions
        process=self.slope*out.observation_contributions
        calibrated=bias+physical.sum(dim=-1)+process.sum(dim=-1)
        direct=self.slope*raw+self.intercept
        if not torch.allclose(calibrated,direct,atol=2e-5,rtol=2e-6):
            raise AssertionError("Affine calibration bypassed the native concept risk decision")
        return calibrated,bias,physical,process


def fit_train_only_native_affine_calibration(
    native_logits:np.ndarray,
    independent_train_outcomes:np.ndarray,
    *,
    independent_calibration_episode_ids:np.ndarray,
    native_model_fit_episode_ids:np.ndarray,
    source_risk_screen_episode_ids:np.ndarray,
)->NativeAffineProbabilityCalibration:
    z=np.asarray(native_logits,dtype=np.float64).reshape(-1)
    y=np.asarray(independent_train_outcomes,dtype=np.float64).reshape(-1)
    calibration_ids=list(map(int,independent_calibration_episode_ids))
    model_fit_ids=set(map(int,native_model_fit_episode_ids))
    risk_ids=set(map(int,source_risk_screen_episode_ids))
    if (len(z)!=150 or len(y)!=150 or len(calibration_ids)!=150
            or not np.isfinite(z).all() or not np.isfinite(y).all()
            or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2
            or np.std(z)<1e-10 or len(set(calibration_ids))!=len(calibration_ids)
            or not model_fit_ids or not risk_ids
            or set(calibration_ids)&(model_fit_ids|risk_ids)
            or model_fit_ids & risk_ids):
        raise ValueError(
            "ABSTAIN: calibration must use 150 independent labelled TRAIN episodes"
        )

    def objective(parameters:np.ndarray)->float:
        log_scale,bias=map(float,parameters)
        scale=float(np.exp(log_scale))
        lp=scale*z+bias
        nll=float(np.mean(np.logaddexp(0.,lp)-y*lp))
        return nll+REGULARIZATION*(log_scale**2+bias**2)

    initial=float(np.mean(np.logaddexp(0.,z)-y*z))
    optim=minimize(
        objective,x0=np.array([0.,0.],dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(np.log(MIN_SLOPE),np.log(MAX_SLOPE)),
                (-MAX_ABS_INTERCEPT,MAX_ABS_INTERCEPT)],
        options={"maxiter":OPTIMIZATION_STEPS,"ftol":1e-11},
    )
    if not optim.success or not np.isfinite(optim.x).all():
        raise ValueError("ABSTAIN: native probability calibration optimizer failed")
    slope=float(np.exp(optim.x[0]))
    shift=float(optim.x[1])
    fitted=float(np.mean(np.logaddexp(0.,slope*z+shift)-y*(slope*z+shift)))
    if (not MIN_SLOPE<=slope<=MAX_SLOPE
            or not -MAX_ABS_INTERCEPT<=shift<=MAX_ABS_INTERCEPT
            or not np.isfinite(fitted)):
        raise AssertionError("Calibrated native risk parameters out of preregistered bounds")
    return NativeAffineProbabilityCalibration(
        slope=slope,
        intercept=shift,
        n_independent_train_calibration_cases=150,
        initial_train_bce=initial,
        fitted_train_bce=fitted,
        optimizer_success=True,
    )


def native_calibration_passport(value:NativeAffineProbabilityCalibration)->dict:
    return {
        **asdict(value),
        "all_source_and_observation_concept_contributions_common_positive_slope":True,
        "native_direct_contributions_remain_exactly_additive":True,
        "train_only_reserved_calibration_not_risk_screen_or_model_gradients":True,
        "no_concept_source_identity_or_disease_certified":True,
        "not_a_novel_concept_learning_or_calibration_algorithm":True,
    }
