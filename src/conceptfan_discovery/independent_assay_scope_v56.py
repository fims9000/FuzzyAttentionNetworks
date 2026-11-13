"""V56: bounded independent-assay corroboration of automatically discovered concepts.

This is a conditional measurement-evidence gate for a fixed V55 operator.
It is NOT a proof of physical/causal/medical semantic identity. "Device" and
"reference" IDs are declared identifiers, NOT cryptographic credentials.
All assay samples below are explicitly purchased TRAIN/VALID observations;
there is no TEST assay or TEST Y entrypoint.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import numpy as np

from .constrained_joint_operator_v55 import (
    JointMeasuredConceptFitV55,infer_joint_operator_v55,
)

MIN_TRAIN_PURCHASED=150
MIN_VALID_PURCHASED=80
MIN_VALID_SUPPORTED=65
MIN_TRAIN_SUPPORT=120
MIN_VALID_CORRELATION=.55
MIN_VALID_MAE_IMPROVEMENT=.10
TRAIN_MONITOR_SAMPLE=200
TRAIN_MONITOR_BOOTSTRAP=399
MIN_MONITOR_THRESHOLD=.025
MIN_COHORT_SUPPORTED=100
BOOTSTRAP_ALPHA=.995


@dataclass(frozen=True)
class AssayEnvironmentV56:
    name:str
    assay_purchased:int
    supported_assay_episodes:int
    pearson:float
    reference_MAE:float
    constant_MAE:float
    improvement_fraction:float
    passed:bool


@dataclass(frozen=True)
class AssayCorroborationV56:
    candidate_id:str
    device_id:str
    reference_id:str
    train_assay_purchased:int
    train_assay_supported:int
    positive_train_slope:bool
    reference_intercept:float
    reference_slope:float
    reference_constant_train:float
    valid1:AssayEnvironmentV56
    valid2:AssayEnvironmentV56
    dual_VALID_corroborated:bool
    train_source_mean:float
    train_source_n_supported:int
    mean_shift_limit:float
    monitor_bootstrap_subjects:int=TRAIN_MONITOR_SAMPLE
    independent_assay_only_on_train_and_valid:bool=True
    reference_id_is_not_cryptographic_authentication:bool=True
    not_a_causal_or_physical_identity_proof:bool=True
    semantic_label:str="ASSAY_CORROBORATED_STATISTICAL_MEASURED_ASSOCIATION"

    def passport(self):
        return asdict(self)


def _purchased(train_values,train_mask,purchased_indices,purchased_assay,minimum):
    x=np.asarray(train_values)
    mask=np.asarray(train_mask)
    ids=np.asarray(purchased_indices)
    assay=np.asarray(purchased_assay,dtype=np.float64)
    if (x.ndim!=3 or mask.shape!=x.shape or
        ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or
        len(ids)<minimum or len(np.unique(ids))!=len(ids) or
        ids.min()<0 or ids.max()>=len(x) or
        assay.shape!=(len(ids),) or not np.isfinite(assay).all() or
        (assay<0.).any()):
        raise ValueError("Only explicitly purchased and aligned assay references permitted")
    return ids,assay


def _q_in_purchased(fit,x,m,ids):
    pred=infer_joint_operator_v55(fit,x[ids],m[ids])
    return pred["summary"],pred["support"]>=.50


def _correlation(a,b):
    if len(a)<2 or np.std(a)<1e-8 or np.std(b)<1e-8:
        return 0.
    return float(np.corrcoef(a,b)[0,1])


def _environment(fit,x,m,ids,reference,name,slope,intercept,mean_const,train_positive):
    q,used=_q_in_purchased(fit,x,m,ids)
    n=int(used.sum())
    if n<2:
        corr=0.
        mae_pred=mae_null=float("inf")
        improvement=-float("inf")
    else:
        true=reference[used]
        qobs=q[used]
        corr=_correlation(qobs,true)
        mae_pred=float(np.mean(np.abs(slope*qobs+intercept-true)))
        mae_null=float(np.mean(np.abs(mean_const-true)))
        improvement=1.-mae_pred/max(mae_null,1e-12)
    ok=bool(train_positive and n>=MIN_VALID_SUPPORTED and
            corr>=MIN_VALID_CORRELATION and
            improvement>=MIN_VALID_MAE_IMPROVEMENT)
    return AssayEnvironmentV56(
        name=name,assay_purchased=len(ids),supported_assay_episodes=n,
        pearson=corr,reference_MAE=mae_pred,constant_MAE=mae_null,
        improvement_fraction=improvement,passed=ok)


def fit_independent_assay_gate_v56(
    fit:JointMeasuredConceptFitV55,
    train_x,train_mask,train_ids,train_assay,
    valid1_x,valid1_mask,valid1_ids,valid1_assay,
    valid2_x,valid2_mask,valid2_ids,valid2_assay,
    *,device_id,reference_id,seed=20265626,
):
    """Use only independently purchased assay episodes and raw observed signals.

    TEST signals, TEST labels and simulator latent factors are not arguments.
    The regression is fitted on bought TRAIN assay readings only; two
    disjoint VALID groups both must independently corroborate it.
    """
    if (not isinstance(device_id,str) or not device_id.strip() or
        not isinstance(reference_id,str) or not reference_id.strip()):
        raise ValueError("Declared reference and device IDs are required")
    i0,a0=_purchased(train_x,train_mask,train_ids,train_assay,MIN_TRAIN_PURCHASED)
    i1,a1=_purchased(valid1_x,valid1_mask,valid1_ids,valid1_assay,MIN_VALID_PURCHASED)
    i2,a2=_purchased(valid2_x,valid2_mask,valid2_ids,valid2_assay,MIN_VALID_PURCHASED)
    train_q,train_used=_q_in_purchased(fit,train_x,train_mask,i0)
    observed_q=train_q[train_used]
    observed_a=a0[train_used]
    if len(observed_q)<MIN_TRAIN_SUPPORT:
        # Invalid independent measurement: safe refusal rather than
        # fitting a source coefficient on unmeasured values.
        slope=0.
        intercept=float(observed_a.mean()) if len(observed_a) else 0.
    else:
        centered_q=observed_q-observed_q.mean()
        denom=float(np.dot(centered_q,centered_q))
        slope=float(np.dot(centered_q,observed_a-observed_a.mean())/max(denom,1e-12))
        intercept=float(observed_a.mean()-slope*observed_q.mean())
    train_positive=bool(slope>1e-8 and len(observed_q)>=MIN_TRAIN_SUPPORT)
    constant=float(np.mean(observed_a)) if len(observed_a) else 0.
    v1=_environment(fit,valid1_x,valid1_mask,i1,a1,
                    "VALID1",slope,intercept,constant,train_positive)
    v2=_environment(fit,valid2_x,valid2_mask,i2,a2,
                    "VALID2",slope,intercept,constant,train_positive)
    train_all=infer_joint_operator_v55(fit,train_x,train_mask)
    eligible=np.asarray(train_all["support"]>=.50)
    q_all=np.asarray(train_all["summary"])[eligible]
    if len(q_all)<TRAIN_MONITOR_SAMPLE:
        raise ValueError("Not enough observed training episodes for source-only monitor")
    mean=float(q_all.mean())
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(q_all),size=(TRAIN_MONITOR_BOOTSTRAP,TRAIN_MONITOR_SAMPLE))
    deviations=np.abs(q_all[draws].mean(axis=1)-mean)
    threshold=float(max(MIN_MONITOR_THRESHOLD,
                        1.5*np.quantile(deviations,BOOTSTRAP_ALPHA)))
    return AssayCorroborationV56(
        candidate_id=fit.candidate.candidate_id,
        device_id=device_id,reference_id=reference_id,
        train_assay_purchased=len(i0),train_assay_supported=len(observed_q),
        positive_train_slope=train_positive,
        reference_intercept=intercept,reference_slope=slope,
        reference_constant_train=constant,valid1=v1,valid2=v2,
        dual_VALID_corroborated=bool(v1.passed and v2.passed),
        train_source_mean=mean,train_source_n_supported=len(q_all),
        mean_shift_limit=threshold,
    )


def assess_observed_source_scope_v56(
    fit:JointMeasuredConceptFitV55,
    certificate:AssayCorroborationV56|None,
    observed_values,observed_mask,
    *,device_id,reference_id,
):
    """Cohort-scoped, unlabeled, fail-closed decision before native risk.

    A certificate does NOT grant a physical name. Any missing/mismatched
    reference or unsupported shift yields zero admitted source decisions.
    """
    inferred=infer_joint_operator_v55(fit,observed_values,observed_mask)
    q=np.asarray(inferred["summary"])
    support=np.asarray(inferred["support"])
    available=support>=.50
    n=int(available.sum())
    difference=None
    status="ABSTAIN_NO_INDEPENDENT_REFERENCE"
    approved=False
    if certificate is None:
        pass
    elif (not device_id or not reference_id or
          certificate.device_id!=device_id or certificate.reference_id!=reference_id or
          certificate.candidate_id!=fit.candidate.candidate_id):
        status="REVOKED_REFERENCE_OR_DEVICE_MISMATCH"
    elif not certificate.dual_VALID_corroborated:
        status="ABSTAIN_ASSAY_NOT_REPLICATED"
    elif n<MIN_COHORT_SUPPORTED:
        status="ABSTAIN_INSUFFICIENT_COHORT_SOURCE_SUPPORT"
    else:
        difference=float(q[available].mean()-certificate.train_source_mean)
        if abs(difference)>certificate.mean_shift_limit:
            status="REVOKED_UNLABELED_SOURCE_COHORT_SHIFT"
        else:
            status="ASSAY_CORROBORATED_MEASURED_PROXY"
            approved=True
    return {
        "source_q":q,
        "source_support":support,
        "source_used":available & approved,
        "approved":approved,
        "status":status,
        "n_cohort_supported":n,
        "cohort_q_mean_shift":difference,
        "cohort_mean_shift_limit":None if certificate is None else certificate.mean_shift_limit,
        "physical_or_causal_identity_proven":False,
        "external_assay_read_at_test":False,
    }
