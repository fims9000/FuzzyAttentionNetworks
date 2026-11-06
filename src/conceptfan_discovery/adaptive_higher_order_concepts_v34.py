"""V34: audited fuzzy concept membership, T1 / interval T2 / restricted interval T3.

IMPORTANT: This is a finite alpha-plane representation of a *restricted*
interval Type-3 fuzzy set, not a general IT3 fuzzy inference system and NOT
a novel type-reduction theorem. The two uncertainties have SEPARATE origins:
- delta_measure: source amplitude ambiguity under missing observations;
- delta_boundary: uncertainty in the concept's boundary across INDEPENDENT,
  externally anchored measurement regimes. Source distribution shift alone
  cannot certify an instrument-induced boundary shift.

At alpha=1 the IT3 footprint reduces to the base IT2 interval; as alpha
decreases, the third-order uncertainty expands this footprint. Every
alpha-plane is persisted and can be inspected. Membership is not a class
probability, and these intervals are not frequentist confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

FuzzyOrder=Literal["type1","interval_type2","restricted_interval_type3","auto"]
ALPHAS=(.25,.50,.75,1.00)


@dataclass(frozen=True)
class FuzzySourceDefinition:
    source_id: str
    sensor_indices: tuple[int,...]
    threshold_train: float
    width_train: float
    train_missingness_spread: float
    boundary_spread_anchor: float
    certified_calibration_environments: int
    external_anchor_verified: bool
    fitted_on_train_only: bool = True
    threshold_used_outcome_labels: bool = False
    verified_physical_semantics: bool = False

    def __post_init__(self):
        if not self.source_id or not self.sensor_indices or min(self.sensor_indices)<0:
            raise ValueError("Every concept needs actual source sensor provenance")
        if (not np.isfinite([self.threshold_train,self.width_train,
                              self.train_missingness_spread,
                              self.boundary_spread_anchor]).all()
            or self.width_train<=0 or self.train_missingness_spread<0
            or self.boundary_spread_anchor<0
            or self.certified_calibration_environments<0):
            raise ValueError("Invalid independent membership uncertainties")
        if self.external_anchor_verified and self.certified_calibration_environments<2:
            raise ValueError("One environment cannot verify Type-3 boundary calibration")
        if not self.fitted_on_train_only or self.threshold_used_outcome_labels:
            raise ValueError("Concept membership definitions must be outcome-blind")


def fit_measured_definition(
    source_id:str,sensor_indices:tuple[int,...],
    train_amplitudes:np.ndarray,
    train_observed_uncertainty:np.ndarray,
    *,
    independently_calibrated_boundary_spread:float=0.,
    certified_environments:int=0,
    trusted_external_anchor:bool=False,
)->FuzzySourceDefinition:
    """Robust unsupervised boundary from TRAIN-only measured amplitude proxies.

    No synthetic true state or Y enters here. trusted_external_anchor must
    refer to independent device calibration, not source median drift.
    """
    a=np.asarray(train_amplitudes,dtype=np.float64).reshape(-1)
    u=np.asarray(train_observed_uncertainty,dtype=np.float64).reshape(-1)
    if (a.shape!=u.shape or len(a)<100 or not np.isfinite(a).all()
        or not np.isfinite(u).all() or (u<0).any()):
        raise ValueError("Unaligned measured TRAIN amplitudes and uncertainty")
    med=float(np.median(a))
    scale=float(max(.15,(np.quantile(a,.75)-np.quantile(a,.25))/1.349))
    uncertainty=float(np.median(u))
    if not trusted_external_anchor:
        if certified_environments!=0 or independently_calibrated_boundary_spread!=0:
            raise ValueError("Do not invent independent boundary calibration")
    return FuzzySourceDefinition(
        source_id=source_id,
        sensor_indices=sensor_indices,
        threshold_train=med,
        width_train=scale,
        train_missingness_spread=uncertainty,
        boundary_spread_anchor=float(independently_calibrated_boundary_spread),
        certified_calibration_environments=certified_environments,
        external_anchor_verified=trusted_external_anchor,
    )


def select_order_from_train(
    definition:FuzzySourceDefinition,
    *,delta2_threshold:float=.08,delta3_threshold:float=.08,
)->str:
    """Order decision based ONLY on frozen TRAIN and independently trusted anchor."""
    if delta2_threshold!=.08 or delta3_threshold!=.08:
        raise ValueError("Frozen adaptive-order thresholds changed")
    if (definition.external_anchor_verified
        and definition.certified_calibration_environments>=2
        and definition.boundary_spread_anchor>delta3_threshold):
        return "restricted_interval_type3"
    if definition.train_missingness_spread>delta2_threshold:
        return "interval_type2"
    return "type1"


@dataclass
class MembershipBundle:
    center: torch.Tensor
    degree_lower: torch.Tensor
    degree_upper: torch.Tensor
    alpha_intervals: torch.Tensor
    delta_measure: torch.Tensor
    delta_boundary: torch.Tensor
    source_observed_information: torch.Tensor
    selected_order: str
    source_id: str
    requires_external_anchor: bool
    boundary_physically_verified: bool = False
    tertiary_full_IT3_type_reduction_performed: bool = False
    interval_is_a_statistical_confidence_interval: bool = False


def source_membership(
    amplitude:torch.Tensor,
    amplitude_uncertainty:torch.Tensor,
    observed_information:torch.Tensor,
    definition:FuzzySourceDefinition,
    *,
    order:FuzzyOrder="auto",
    minimum_information_fraction:float=.50,
)->MembershipBundle:
    """Torch-differentiable fuzzy membership and 4 inspectable alpha-planes.

    This function is a deterministic transform of a measured source and a
    SEPARATELY estimated uncertainty. The default source-amplitude posterior
    mean is NOT a physical/casual concept label. Without adequate measured
    support the caller must set exact zero contribution.
    """
    a=torch.as_tensor(amplitude,dtype=torch.float32).reshape(-1)
    s=torch.as_tensor(amplitude_uncertainty,dtype=torch.float32).reshape(-1)
    info=torch.as_tensor(observed_information,dtype=torch.float32).reshape(-1)
    if (len(a)!=len(s) or len(a)!=len(info) or len(a)<1
        or not bool(torch.isfinite(a).all())
        or not bool(torch.isfinite(s).all())
        or not bool(torch.isfinite(info).all())
        or bool((s<0).any()) or bool((info<0).any()) or bool((info>1).any())):
        raise ValueError("Expected finite amplitude, nonnegative uncertainty, support [0,1]")
    if minimum_information_fraction!=.50:
        raise ValueError("Prespecified minimum measured-source information is .50")
    mode=select_order_from_train(definition) if order=="auto" else order
    if mode not in ("type1","interval_type2","restricted_interval_type3"):
        raise ValueError("Unknown fuzzy concept representation")
    if (mode=="restricted_interval_type3" and not
        (definition.external_anchor_verified
         and definition.certified_calibration_environments>=2)):
        raise ValueError("Restricted Type-3 requires independent anchored calibration")
    tau=definition.threshold_train
    width=definition.width_train
    delta2=torch.zeros_like(a) if mode=="type1" else s.clamp(0,4)
    delta3=(torch.full_like(a,definition.boundary_spread_anchor)
            if mode=="restricted_interval_type3" else torch.zeros_like(a))
    alphas=torch.tensor(ALPHAS,device=a.device,dtype=a.dtype)
    # Finite alpha-slice representation: intervals shrink as alpha rises.
    extra=delta3[:,None]*(1-alphas[None,:])
    radius=delta2[:,None]+extra
    lo=torch.sigmoid((a[:,None]-tau-radius)/width)
    hi=torch.sigmoid((a[:,None]-tau+radius)/width)
    if mode=="type1":
        lo=hi=torch.sigmoid((a[:,None]-tau)/width).expand(-1,len(ALPHAS))
    if (not bool((lo<=hi+1e-7).all())
        or not bool((lo[:,:-1]<=lo[:,1:]+1e-7).all())
        or not bool((hi[:,:-1]>=hi[:,1:]-1e-7).all())):
        raise AssertionError("Nested IT3 alpha-plane membership was violated")
    # Constrained algebraic type reduction for this PREDECLARED symmetric
    # alpha-plane family. This is NOT KM iteration or general IT3 inference.
    weights=alphas**2
    weights=weights/weights.sum()
    mid=.5*(lo+hi)
    core=torch.sum(mid*weights[None,:],dim=1)
    envelope_lo=lo[:,0]
    envelope_hi=hi[:,0]
    return MembershipBundle(
        center=core,
        degree_lower=envelope_lo,
        degree_upper=envelope_hi,
        alpha_intervals=torch.stack((lo,hi),dim=-1),
        delta_measure=delta2,
        delta_boundary=delta3,
        source_observed_information=info,
        selected_order=mode,
        source_id=definition.source_id,
        requires_external_anchor=mode=="restricted_interval_type3",
    )


class AdditiveFuzzySourceSlot(nn.Module):
    """New native torch concept slot with exactly additive signed contribution.

    The existing temporal known-concept encoder/head are passed as FROZEN
    tensors; no raw sensor input or downstream shortcut exists in this module.
    Gamma is a separately trainable signed coefficient from TRAIN labels,
    and approval must come from two heldout VALID environments.
    """
    def __init__(self,definition:FuzzySourceDefinition,*,order:FuzzyOrder="auto"):
        super().__init__()
        self.definition=definition
        self.order=order
        self.gamma=nn.Parameter(torch.zeros((),dtype=torch.float32))

    def forward(self,base_logit:torch.Tensor,amplitude:torch.Tensor,
                uncertainty:torch.Tensor,information:torch.Tensor,
                *,approved:bool=False):
        base=torch.as_tensor(base_logit,dtype=torch.float32).reshape(-1)
        bundle=source_membership(amplitude,uncertainty,information,
                                 self.definition,order=self.order)
        if base.shape!=bundle.center.shape:
            raise ValueError("One base logit is required per measured sample")
        eligible=(bundle.source_observed_information>=.50)&bool(approved)
        contribution=torch.where(
            eligible,self.gamma*(bundle.center-.5),torch.zeros_like(base))
        new=base+contribution
        # Signed lower/upper bound on risk effect; membership degrees NOT
        # probabilities. This is an algebraic envelope, not risk calibration.
        endpoint1=self.gamma*(bundle.degree_lower-.5)
        endpoint2=self.gamma*(bundle.degree_upper-.5)
        delta_lo=torch.where(eligible,torch.minimum(endpoint1,endpoint2),
                             torch.zeros_like(base))
        delta_hi=torch.where(eligible,torch.maximum(endpoint1,endpoint2),
                             torch.zeros_like(base))
        if not bool((delta_lo<=contribution+1e-6).all()) or not bool(
            (contribution<=delta_hi+1e-6).all()):
            raise AssertionError("Point contribution escaped fuzzy membership envelope")
        return {
            "logit":new,
            "original_logit":base,
            "source_contribution":contribution,
            "source_contribution_lower":delta_lo,
            "source_contribution_upper":delta_hi,
            "source_used":eligible,
            "source_membership":bundle,
            "no_raw_input_prediction_bypass":True,
            "true_physical_source_id_confirmed":False,
        }


def membership_grade_alpha_plane(
    alpha_intervals:torch.Tensor,
    trial_membership_grade:torch.Tensor,
)->torch.Tensor:
    """Highest discrete alpha whose interval includes tested grade u.

    The resulting value is the alpha-plane evidence of the restricted
    three-level set. It is NOT an empirical probability of grade u.
    """
    bands=torch.as_tensor(alpha_intervals,dtype=torch.float32)
    trial=torch.as_tensor(trial_membership_grade,dtype=torch.float32).reshape(-1)
    if bands.ndim!=3 or bands.shape[1:]!=(len(ALPHAS),2) or len(trial)!=len(bands):
        raise ValueError("Expected [N,alpha,2] membership representation")
    levels=torch.tensor(ALPHAS,dtype=bands.dtype,device=bands.device)
    support=(trial[:,None]>=bands[:,:,0])&(trial[:,None]<=bands[:,:,1])
    return (support*levels[None,:]).amax(dim=-1)
