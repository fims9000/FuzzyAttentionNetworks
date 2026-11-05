"""Outcome-blind independent anchor discrepancy and explicit repair/fallback.

**Assumption**: channel6 is an externally calibrated, genuinely independent
readout of the SAME underlying source as channels4/5. This assumption is
supplied by the synthetic generator, not discovered by ConceptFAN. Replacing
corrupt source evidence with anchor evidence is only justified while this
assumption holds. Anchor-only failures are a required negative test.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import numpy as np
from scipy.special import expit

from conceptfan_discovery.isolated_source_admission_v28 import (
    measured_candidate,
)
from conceptfan_discovery.sensor_pair_provenance_v29 import (
    PairIntegrityPolicy,pair_is_credible,
)


@dataclass(frozen=True)
class AnchorPolicy:
    max_sigmoid_mean_discrepancy: float
    min_pair_observed_fraction: float
    min_anchor_observed_fraction: float
    train_episode_count: int
    valid_train_episode_count: int
    quantile: float
    independent_calibrated_reference_is_external_assumption: bool = True
    may_fail_when_anchor_is_faulty: bool = True
    cannot_detect_coordinated_common_mode_transformation: bool = True
    label_information_used_for_threshold: bool = False

    def as_dict(self):
        return asdict(self)


def candidate_and_anchor(x:np.ndarray,m:np.ndarray)->dict:
    values=np.asarray(x,dtype=np.float64)
    mask=np.asarray(m)
    if values.ndim!=3 or values.shape!=mask.shape or values.shape[-1]!=7 or values.shape[1]<2:
        raise ValueError("Anchor requires [episode,time,7] and aligned mask")
    if not np.logical_or(mask==0,mask==1).all():
        raise ValueError("Anchor observation mask must be binary")
    observed=mask.astype(bool)
    if not np.isfinite(values[observed]).all():
        raise ValueError("Observed sensor payload is nonfinite")
    qpair,support_pair=measured_candidate(values[:,:,:6],mask[:,:,:6])
    anchor_mask=observed[:,:,6]
    nav=anchor_mask.sum(axis=1)
    anchor_mean=np.where(anchor_mask,values[:,:,6],0.).sum(axis=1)/np.maximum(1,nav)
    anchor_support=nav/values.shape[1]
    anchor_activation=0.5+anchor_support*(expit(anchor_mean)-0.5)
    pair_mask=observed[:,:,4:6]
    n_pair=pair_mask.sum(axis=(1,2))
    pair_mean=np.where(pair_mask,values[:,:,4:6],0.).sum(axis=(1,2))/np.maximum(1,n_pair)
    discrepancy=np.abs(expit(pair_mean)-expit(anchor_mean))
    if (np.any(qpair<0) or np.any(qpair>1)
        or np.any(anchor_activation<0) or np.any(anchor_activation>1)):
        raise AssertionError("Out of bounds concept candidate or anchor")
    return {
        "pair_concept":qpair,
        "pair_support":support_pair,
        "anchor_concept":anchor_activation,
        "anchor_support":anchor_support,
        "sigmoid_mean_discrepancy":discrepancy,
        "pair_raw_observed_mean":pair_mean,
        "anchor_raw_observed_mean":anchor_mean,
    }


def fit_anchor_policy(x:np.ndarray,m:np.ndarray,*,min_support:float=.5,
                      quantile:float=.99)->AnchorPolicy:
    """Source-pair/anchor threshold is TRAIN-measured only, outcome-blind."""
    if not 0<min_support<=1 or not .5<quantile<1:
        raise ValueError("Invalid frozen anchor policy settings")
    vals=candidate_and_anchor(x,m)
    available=(vals["pair_support"]>=min_support)&(vals["anchor_support"]>=min_support)
    if available.sum()<100:
        raise ValueError("Insufficient jointly observable TRAIN history")
    threshold=float(np.quantile(vals["sigmoid_mean_discrepancy"][available],quantile))
    if not (0<threshold<1):
        raise ValueError("Unidentifiable or degenerate anchor threshold")
    return AnchorPolicy(
        max_sigmoid_mean_discrepancy=threshold,
        min_pair_observed_fraction=min_support,
        min_anchor_observed_fraction=min_support,
        train_episode_count=len(x),
        valid_train_episode_count=int(available.sum()),
        quantile=quantile,
    )


def anchored_predictions(
    original_base_logit:np.ndarray,
    values:np.ndarray,masks:np.ndarray,
    *,
    gamma:float,
    validated_global_source:bool,
    pair_policy:PairIntegrityPolicy,
    anchor_policy:AnchorPolicy,
)->dict:
    """Compare frozen K2, source-only, V29 reject, V30 reject, V30 repair.

    All acceptance and repair decisions are label blind and only inspect the
    episode's observed *historical* sensor readings. The original K2 risk is
    unchanged for each method.
    """
    base=np.asarray(original_base_logit,dtype=np.float64).reshape(-1)
    if not np.isfinite(base).all() or len(base)!=len(values):
        raise ValueError("Frozen K2 logit must be finite and patient aligned")
    values_a=candidate_and_anchor(values,masks)
    pair_integrity=pair_is_credible(pair_policy,values,masks)["integrity_passed"]
    pair_support=values_a["pair_support"]>=anchor_policy.min_pair_observed_fraction
    anchor_support=values_a["anchor_support"]>=anchor_policy.min_anchor_observed_fraction
    agree=(values_a["sigmoid_mean_discrepancy"]<=anchor_policy.max_sigmoid_mean_discrepancy)
    pair_base_available=bool(validated_global_source) & pair_support
    pair_signed=gamma*(values_a["pair_concept"]-.5)
    anchor_signed=gamma*(values_a["anchor_concept"]-.5)

    # V28 gate, no external integrity evidence.
    used28=pair_base_available
    src28=np.where(used28,pair_signed,0.)

    # V29: internal pair sensor agreement, no independent reference.
    used29=used28 & pair_integrity
    src29=np.where(used29,pair_signed,0.)

    # V30: stricter disqualify if the external anchor disagrees or is
    # unavailable. This is *abstention*, never a semantic claim of completeness.
    used_reject=used29 & anchor_support & agree
    src_reject=np.where(used_reject,pair_signed,0.)

    # V30 repair: if a trustworthy external anchor is measured, substitute
    # it when either pair internal credibility or pair-to-anchor agreement
    # breaks. Otherwise retain pair estimate, or fall back to exact K2.
    trusted_anchor=bool(validated_global_source) & anchor_support
    repair_needed=(~pair_integrity)|(~agree)|(~pair_support)
    repaired=trusted_anchor & repair_needed
    keep_pair=trusted_anchor & pair_support & pair_integrity & agree
    src_repair=np.where(keep_pair,pair_signed,0.)+np.where(repaired,anchor_signed,0.)
    used_repair=keep_pair|repaired
    np.testing.assert_array_equal(src_repair[~used_repair],np.zeros((~used_repair).sum()))
    if np.any(keep_pair&repaired):
        raise AssertionError("Source repair and retention choices overlap")

    results={
        "frozen_k2":base.copy(),
        "v28_pair_candidate":base+src28,
        "v29_pair_reject":base+src29,
        "v30_anchor_reject":base+src_reject,
        "v30_anchor_repair":base+src_repair,
    }
    for name, risk in results.items():
        if not np.isfinite(risk).all():
            raise AssertionError(f"Nonfinite risk for {name}")
    np.testing.assert_array_equal(results["v30_anchor_repair"][~used_repair],
                                  base[~used_repair])
    return {
        "logits":results,
        "contributions":{
            "v28_pair_candidate":src28,
            "v29_pair_reject":src29,
            "v30_anchor_reject":src_reject,
            "v30_anchor_repair":src_repair,
        },
        "decisions":{
            "v28_pair_candidate":used28,
            "v29_pair_reject":used29,
            "v30_anchor_reject":used_reject,
            "v30_anchor_repair":used_repair,
            "v30_replaced_by_anchor":repaired,
            "v30_kept_pair":keep_pair,
        },
        "summary":{
            "globally_admitted":bool(validated_global_source),
            "internal_pair_agreement_fraction":float(pair_integrity.mean()),
            "external_anchor_agreement_fraction":float(agree.mean()),
            "anchor_support_fraction":float(anchor_support.mean()),
            "pair_support_fraction":float(pair_support.mean()),
            "repair_fraction":float(repaired.mean()),
            "refusal_fraction":float((~used_repair).mean()),
            "source_decision_has_no_access_to_outcome_labels":True,
            "repaired_source_has_explicit_sensor6_provenance":True,
        },
    }
