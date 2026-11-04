"""V22b: screen measured K2 native source axes on disjoint TRAIN task labels.

This is a faithful adaptation of V18's existing point-biserial permutation,
patient bootstrap and split-half risk association policy to *already
registered V22 PCA/BSS source directions*. It does not use an oracle physical
state, assign a clinical meaning or create a parallel risk classifier.

Accepted sources receive a train-screened ±.05 INITIAL native raw weight,
with unrestricted signed optimization. Rejected sources remain train-native
descriptive concepts but carry serialized V21d source risk-policy code 2.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Iterable

import numpy as np

from .risk_screen_v18 import _point_biserial


@dataclass(frozen=True)
class NativeTwoSourceRiskEvidence:
    slot: int
    status: str
    n_disjoint_screen_episodes: int
    observed_point_biserial_r: float
    permutation_p_two_sided: float
    bootstrap_95pct_ci: tuple[float,float]
    two_split_correlations: tuple[float,float]
    train_only_signed_orientation: int
    physical_source_or_clinical_cause_certified: bool = False


@dataclass(frozen=True)
class NativeSourceRiskPlan:
    active_native_raw_decision_initialization: dict[int,float]
    structurally_descriptive_only_native_slots: tuple[int,...]
    per_source_evidence: tuple[NativeTwoSourceRiskEvidence,...]
    all_risk_cohort_episodes_disjoint_from_source_training: bool
    no_oracle_factor_or_validation_label_used: bool = True


def screen_v22_native_sources_on_disjoint_train_labels(
    measured_screen_histories:np.ndarray,
    measured_screen_masks:np.ndarray,
    screen_outcomes:np.ndarray,
    basis:np.ndarray,
    *,
    original_source_train_episode_ids:Iterable[int],
    screen_episode_ids:Iterable[int],
    seed:int=522231,
    permutations:int=199,
    bootstraps:int=199,
    alpha:float=.01,
    min_abs_r:float=.15,
    first_last_window:int=6,
    initial_raw_weight_magnitude:float=.05,
)->NativeSourceRiskPlan:
    x=np.asarray(measured_screen_histories,dtype=np.float64)
    mask=np.asarray(measured_screen_masks,dtype=np.float64)
    labels=np.asarray(screen_outcomes,dtype=np.float64)
    b=np.asarray(basis,dtype=np.float64)
    train_ids=set(map(int,original_source_train_episode_ids))
    test_ids=list(map(int,screen_episode_ids))
    if (x.ndim!=3 or mask.shape!=x.shape or len(x)<180
            or labels.shape!=(len(x),) or len(test_ids)!=len(x)
            or b.shape!=(x.shape[-1],2)
            or not np.allclose(b.T@b,np.eye(2),atol=1e-6)
            or not np.isfinite(x).all() or not np.isfinite(b).all()
            or not np.array_equal(mask,np.ones_like(mask))
            or not np.isin(labels,[0,1]).all()
            or len(np.unique(labels))!=2
            or not 2<=first_last_window<=x.shape[1]):
        raise ValueError("ABSTAIN: invalid independent TRAIN source-risk screen cohort")
    if not train_ids or len(set(test_ids))!=len(test_ids) or set(test_ids)&train_ids:
        raise ValueError("ABSTAIN: source TRAIN and outcome screen IDs must be disjoint")
    if (permutations<99 or bootstraps<99 or not 0<alpha<=.05
            or not .02<=min_abs_r<1
            or not .005<=initial_raw_weight_magnitude<=.1):
        raise ValueError("Invalid predeclared V22b source-risk screening thresholds")

    rng=np.random.default_rng(seed)
    scores=np.mean(x[:,-first_last_window:,:]@b,axis=1)
    accepted={}
    descriptive=[]
    details=[]
    for slot in (0,1):
        source=scores[:,slot]
        observed=_point_biserial(source,labels)
        halves=rng.permutation(len(labels))
        a,j=np.array_split(halves,2)
        signed_halves=(_point_biserial(source[a],labels[a]),
                       _point_biserial(source[j],labels[j]))
        null=np.asarray([
            abs(_point_biserial(source,rng.permutation(labels)))
            for _ in range(permutations)
        ])
        p=float((1+np.count_nonzero(null>=abs(observed)))/(1+permutations))
        boot=np.asarray([
            _point_biserial(source[idx],labels[idx])
            for idx in (rng.integers(0,len(labels),size=len(labels))
                        for _ in range(bootstraps))
        ])
        lo,hi=map(float,np.quantile(boot,[.025,.975]))
        direction=int(np.sign(observed))
        halves_agree=(direction!=0 and
                      all(int(np.sign(c))==direction and abs(c)>=.035
                          for c in signed_halves))
        ci_good=lo>0 if direction==1 else hi<0 if direction==-1 else False
        ok=(halves_agree and ci_good and p<=alpha and abs(observed)>=min_abs_r)
        if ok:
            accepted[slot]=float(direction*initial_raw_weight_magnitude)
        else:
            descriptive.append(slot)
        details.append(NativeTwoSourceRiskEvidence(
            slot=slot,
            status=("ACCEPT_OBSERVATIONAL_PREDICTIVE_SOURCE"
                    if ok else "ABSTAIN_NO_STABLE_PREDICTIVE_ASSOCIATION"),
            n_disjoint_screen_episodes=len(labels),
            observed_point_biserial_r=float(observed),
            permutation_p_two_sided=p,
            bootstrap_95pct_ci=(lo,hi),
            two_split_correlations=tuple(map(float,signed_halves)),
            train_only_signed_orientation=direction if ok else 0,
        ))
    if set(accepted)&set(descriptive) or len(accepted)+len(descriptive)!=2:
        raise AssertionError("Every native source must be risk-screened or no-risk")
    return NativeSourceRiskPlan(
        active_native_raw_decision_initialization=accepted,
        structurally_descriptive_only_native_slots=tuple(descriptive),
        per_source_evidence=tuple(details),
        all_risk_cohort_episodes_disjoint_from_source_training=True,
    )


def risk_plan_passport(plan:NativeSourceRiskPlan)->dict:
    return {
        "initial_native_raw_risk_weights":{
            str(k):v for k,v in plan.active_native_raw_decision_initialization.items()
        },
        "permanent_native_descriptive_no_risk_slots":list(
            plan.structurally_descriptive_only_native_slots
        ),
        "source_screens":[asdict(x) for x in plan.per_source_evidence],
        "outcome_cohort_disjoint_from_unmixing_and_model_fit":True,
        "artifact_cannot_be_identified_by_risk_screen":True,
        "risk_screen_negative_is_not_nonphysical_label":True,
        "neural_inference_uses_only_existing_native_concept_weights":True,
        "no_clinical_or_causal_source_label_proven":True,
    }
