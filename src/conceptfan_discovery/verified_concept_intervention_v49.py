"""V49: explicit, provenance-checked test-time concept correction API.

No Y, true source, or sensor access in prediction intervention. Expert
reference is supplied explicitly for audited reject/replace; no implicit
semantic physical guarantees are inferred from the model.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Literal
import numpy as np
from scipy.special import expit,logit

from .bounded_source_evidence_v48 import risk_concept_input_v48

ActionV49=Literal["keep","confirm","reject","replace"]


@dataclass(frozen=True)
class IndependentAssayCalibrationV49:
    center_energy:float
    scale_energy:float
    slope:float
    intercept:float
    reject_threshold:float
    supported_train_count:int
    source_energy_train_correlation:float
    certified_available:bool
    reference_is_ideal_synthetic_assay:bool=True
    not_automatically_identified_semantics:bool=True

    def audit(self):
        return asdict(self)


def fit_independent_reference_v49(train_q,train_source_support,train_energy):
    """External physical assay exists only on TRAIN; fail closed if nonidentifiable.

    Per-device expected q = sigmoid(slope * standardized external energy + b).
    No Y or TEST is accepted here.
    """
    q=np.asarray(train_q,dtype=np.float64)
    supported=np.asarray(train_source_support,dtype=bool)
    energy=np.asarray(train_energy,dtype=np.float64)
    if (q.ndim!=1 or q.shape!=supported.shape or q.shape!=energy.shape
            or len(q)<300 or not np.isfinite(q).all()
            or not np.isfinite(energy).all()
            or (q<0).any() or (q>1).any()):
        raise ValueError("Expected original TRAIN membership and separate TRAIN assay")
    number=int(supported.sum())
    if number<200:
        return IndependentAssayCalibrationV49(0.,1.,0.,0.,0.,number,0.,False)
    e=energy[supported]
    centre=float(np.median(e))
    spread=float(max(.2,(np.quantile(e,.75)-np.quantile(e,.25))/1.349))
    e_norm=(e-centre)/spread
    q_supported=q[supported]
    clipped=np.clip(q_supported,.001,.999)
    logit_q=logit(clipped)
    x=e_norm-e_norm.mean()
    y=logit_q-logit_q.mean()
    denom=float(np.mean(x*x))
    slope=float(np.mean(x*y)/(denom+1e-8))
    intercept=float(logit_q.mean()-slope*e_norm.mean())
    corr=float(np.corrcoef(e_norm,q_supported)[0,1]) if (
        np.std(e_norm)>1e-9 and np.std(q_supported)>1e-9) else 0.
    usable=bool(np.isfinite(slope) and np.isfinite(intercept) and
                corr>=.35 and .15<=slope<=10.)
    reference_q=expit(slope*e_norm+intercept)
    errors=np.abs(q_supported-reference_q)
    threshold=float(max(.05,np.quantile(errors,.95)))
    return IndependentAssayCalibrationV49(
        centre,spread,slope,intercept,threshold,number,corr,usable)


def synthetic_expert_verdict_v49(calibration,measured_q,true_assay_energy,selected):
    """Synthetic IDEAL independent reviewer, never called during model fitting.

    Only selected patients' physical oracle energy is read; other values must
    not be used in candidate selection. Returns confirm/reject, not labels Y.
    """
    q=np.asarray(measured_q,dtype=np.float64)
    e=np.asarray(true_assay_energy,dtype=np.float64)
    chosen=np.asarray(selected,dtype=bool)
    if (q.ndim!=1 or q.shape!=e.shape or q.shape!=chosen.shape
            or not np.isfinite(q).all() or (q<0).any() or (q>1).any()
            or not np.isfinite(e[chosen]).all()):
        raise ValueError("Assay review requires selected measured concept and energy")
    actions=np.repeat("keep",len(q)).astype("<U8")
    if not calibration.certified_available:
        return actions,{"certified_available":False,"reviewed":0,"rejected":0,"confirmed":0}
    expected=np.full(len(q),np.nan,dtype=np.float64)
    expected[chosen]=expit(calibration.slope*
                          ((e[chosen]-calibration.center_energy)/calibration.scale_energy)
                          +calibration.intercept)
    invalid=chosen & (np.abs(q-expected)>calibration.reject_threshold)
    actions[chosen]="confirm"
    actions[invalid]="reject"
    return actions,{"certified_available":True,"reviewed":int(chosen.sum()),
                    "rejected":int(invalid.sum()),
                    "confirmed":int((chosen & ~invalid).sum())}


def choose_review_patients_v49(
    concept_probability,original_k2_probability,source_used, *,
    budget:int,selection:Literal["active","random"],
    seed:int=0):
    """Fixed-count Y- and reference-free review selection."""
    p=np.asarray(concept_probability,dtype=np.float64)
    k=np.asarray(original_k2_probability,dtype=np.float64)
    eligible=np.asarray(source_used,dtype=bool)
    if (p.ndim!=1 or p.shape!=k.shape or p.shape!=eligible.shape
            or budget<0 or budget>len(p) or
            not np.isfinite(p).all() or not np.isfinite(k).all()
            or (p<=0).any() or (p>=1).any()
            or (k<=0).any() or (k>=1).any()):
        raise ValueError("Selection needs aligned TEST probabilities and source flags")
    candidates=np.flatnonzero(eligible)
    n=min(int(budget),len(candidates))
    choice=np.zeros(len(p),dtype=bool)
    if n==0:
        return choice
    if selection=="active":
        influence=np.abs(logit(p)-logit(k))
        order=np.lexsort((candidates,-influence[candidates]))
        chosen=candidates[order[:n]]
    elif selection=="random":
        chosen=np.random.default_rng(seed).choice(candidates,size=n,replace=False)
    else:
        raise ValueError("Unregistered review selection mode")
    choice[chosen]=True
    return choice


def apply_verified_concept_interventions_v49(
    original_k2_probability,concept_probability,
    original_q,source_used,actions, *,link:str,gamma:float,
    corrected_membership=None,reference_ids=None):
    """Only signed concept contribution changes, or EXACT frozen K2 fallback.

    Rejected source -> unchanged original K2 prediction; verified replacement
    -> recompute gamma*e(q) delta, without retraining hidden/raw paths.
    """
    base=np.asarray(original_k2_probability,dtype=np.float64)
    current=np.asarray(concept_probability,dtype=np.float64)
    q=np.asarray(original_q,dtype=np.float64)
    used=np.asarray(source_used,dtype=bool)
    decisions=np.asarray(actions)
    if (base.ndim!=1 or base.shape!=current.shape or base.shape!=q.shape
            or base.shape!=used.shape or base.shape!=decisions.shape
            or not np.isfinite(base).all() or not np.isfinite(current).all()
            or not np.isfinite(q).all() or not np.isfinite(gamma)
            or (q<0).any() or (q>1).any()
            or (base<=0).any() or (base>=1).any()
            or (current<=0).any() or (current>=1).any()):
        raise ValueError("Misaligned or invalid frozen concept risk input")
    legal=np.isin(decisions,["keep","confirm","reject","replace"])
    if not legal.all():
        raise ValueError("Unknown expert concept intervention action")
    changed=np.isin(decisions,["confirm","reject","replace"])
    if (changed&~used).any():
        raise ValueError("Cannot intervene on source without original measured support")
    refs=np.asarray([""]*len(q) if reference_ids is None else reference_ids,dtype=str)
    if refs.shape!=q.shape or (changed&(refs=="")).any():
        raise ValueError("Expert action requires independent nonempty reference ID")
    result=current.copy()
    reject=decisions=="reject"
    result[reject]=base[reject]
    replace=decisions=="replace"
    if replace.any():
        newq=np.asarray(corrected_membership,dtype=np.float64)
        if (newq.shape!=q.shape or not np.isfinite(newq[replace]).all()
                or (newq[replace]<0).any() or (newq[replace]>1).any()):
            raise ValueError("Verified replacement must provide valid 1D membership")
        prior=risk_concept_input_v48(q,link=link)
        proposed=risk_concept_input_v48(
            np.where(replace,newq,q),link=link)
        old_logit=logit(current[replace])
        result[replace]=expit(old_logit+float(gamma)*
                              (proposed[replace]-prior[replace]))
    if not np.array_equal(result[reject],base[reject]):
        raise AssertionError("Verified reject failed exact frozen K2 fallback")
    if not np.array_equal(result[decisions=="keep"],current[decisions=="keep"]):
        raise AssertionError("Unreviewed risk was silently altered")
    if not np.array_equal(result[decisions=="confirm"],current[decisions=="confirm"]):
        raise AssertionError("Confirmed membership was silently altered")
    return {
        "probability":result,
        "action":decisions.copy(),
        "changed_prediction":reject|replace,
        "n_verified_reviews":int(changed.sum()),
        "n_rejected_sources":int(reject.sum()),
        "n_membership_replacements":int(replace.sum()),
        "exact_frozen_K2_on_reject":True,
        "physical_identity_proof_not_inferred":True,
    }
