"""V57: independent measurement anchor verifies ONE native fuzzy source.

This gate is *external measurement consistency*, not autonomous physical/clinical
identification. The anchor is supplied by a separately calibrated sensing
process and is never a risk-prediction feature nor a concept-forming channel.

Unlabeled TRAIN source/anchor readings alone determine:
- which already discovered source has reproducible anchor association;
- an affine reference alignment;
- a 99th-percentile within-TRAIN episode residual bound.

On each new episode, a discordant or absent independent reference neutralizes
only the matched source's signed logit contribution. Other V56 concepts and all
task coefficients are frozen. Joint corruption of both fault domains can escape.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.special import expit

from conceptfan_discovery.autonomous_temporal_bank_v53 import check_measurements
from conceptfan_discovery.native_temporal_fuzzy_v56 import NativeSelfSupervisedConceptFAN


@dataclass(frozen=True)
class AnchorEvidence:
    anchor_provenance: str
    detected_sensor_group: tuple[int,...] | None
    matched_concept_index: int | None
    min_abs_train_fold_correlation: float | None
    signed_train_fold_correlations: tuple[float,...]
    affine_intercept: float | None
    affine_slope: float | None
    train_median_residual_cutoff_99pct: float | None
    n_reference_supported_train_subjects: int
    n_train_subjects: int
    train_subjects_per_correlation_fold: tuple[int,int,int]
    source_definition_sha256: str
    outcome_labels_used: bool = False
    latent_oracle_states_used: bool = False
    independent_device_provenance_externally_assumed: bool = True
    physical_or_clinical_semantics_proven: bool = False
    verification_status: str = "NOT_FITTED"

    def as_dict(self)->dict:
        return asdict(self)


def _check_anchor(a:np.ndarray,m:np.ndarray,shape:tuple[int,int]):
    values=np.asarray(a,dtype=np.float64)
    mask=np.asarray(m)
    if values.shape!=shape or mask.shape!=shape or not np.isin(mask,[0,1]).all():
        raise ValueError("External reference and mask must have [N,T] shape")
    seen=mask.astype(bool)
    if not np.isfinite(values[seen]).all():
        raise ValueError("Nonfinite observed reference measurement")
    # Never inspect the payload of unmeasured cells, including hidden NaN.
    return np.where(seen,values,0.),seen


def _paired_corr(source:np.ndarray,support:np.ndarray,
                 anchor:np.ndarray,anchor_m:np.ndarray,subjects:np.ndarray,
                 *,min_cells:int=250) -> float | None:
    usable=(support[subjects]>0)&anchor_m[subjects]
    count=int(usable.sum())
    if count<min_cells:
        return None
    left=source[subjects][usable]
    right=anchor[subjects][usable]
    left=left-left.mean()
    right=right-right.mean()
    den=np.sqrt(float(np.dot(left,left))*float(np.dot(right,right)))
    if den<1e-12:
        return None
    return float(np.clip(np.dot(left,right)/den,-1.,1.))


def _episode_reference_errors(source:np.ndarray,support:np.ndarray,
                              anchor:np.ndarray,anchor_mask:np.ndarray,
                              slope:float,intercept:float,
                              *,min_paired_times:int=6):
    available=(support>0)&anchor_mask
    expected=intercept+slope*anchor
    diff=np.abs(source-expected)
    n=available.sum(axis=1)
    result=np.full(len(n),np.inf,dtype=float)
    good=n>=min_paired_times
    for i in np.flatnonzero(good):
        result[i]=float(np.median(diff[i,available[i]]))
    return result,good,n


class IndependentAnchorVerifier:
    """Bounded source-level fallback for a frozen native V56 concept network.

    fit_from_unlabeled() never accepts Y/latent truth. predict() uses an
    external measurement for a consistency decision ONLY. It can reject the
    source association; it cannot authenticate sensor provenance on its own.
    """
    def __init__(self,model:NativeSelfSupervisedConceptFAN,
                 *,anchor_provenance:str="EXTERNALLY_VERIFIED_INDEPENDENT_SENSOR",
                 seed:int=20261471,
                 minimum_fold_abs_corr:float=.65,
                 min_cells_per_fold:int=250,
                 min_paired_times:int=6):
        if not model.is_source_frozen:
            raise ValueError("Native V56 source must be trained before verification")
        if not anchor_provenance:
            raise ValueError("Source provenance must be explicit")
        if not (.5<=minimum_fold_abs_corr<1.) or min_cells_per_fold<100:
            raise ValueError("Invalid precommitted source-correlation evidence threshold")
        if min_paired_times<3:
            raise ValueError("Invalid minimum temporal coverage")
        self.model=model
        self.anchor_provenance=anchor_provenance
        self.seed=int(seed)
        self.minimum_fold_abs_corr=float(minimum_fold_abs_corr)
        self.min_cells_per_fold=int(min_cells_per_fold)
        self.min_paired_times=int(min_paired_times)
        self.evidence:AnchorEvidence|None=None
        self.evidence_hash:str|None=None

    def fit_from_unlabeled(self,train_x,train_mask,anchor_x,anchor_mask):
        x,m=check_measurements(train_x,train_mask)
        if len(x)<300:
            raise ValueError("Source reference requires adequate independent TRAIN subjects")
        anchor,am=_check_anchor(anchor_x,anchor_mask,x.shape[:2])
        rng=np.random.default_rng(self.seed)
        folds=np.array_split(rng.permutation(len(x)),3)
        source_sha=self.model.bank.sha
        candidate_id=None
        candidate_score=-1.
        correlations=()
        if self.model.bank.route=="MEASURED_GROUPS":
            level,support=self.model.bank.transform(x,m)
            for k in range(self.model.bank.k):
                corr=[_paired_corr(level[:,:,k],support[:,:,k],
                                   anchor,am,fold,min_cells=self.min_cells_per_fold)
                      for fold in folds]
                if any(v is None for v in corr):
                    continue
                cc=np.asarray(corr,dtype=float)
                if (min(abs(v) for v in cc)<self.minimum_fold_abs_corr
                      or abs(np.sum(np.sign(cc)))!=3):
                    continue
                strength=float(min(np.abs(cc)))
                if strength>candidate_score:
                    candidate_score=strength
                    candidate_id=k
                    correlations=tuple(float(v) for v in corr)

        if candidate_id is None:
            self.evidence=AnchorEvidence(
                self.anchor_provenance,None,None,None,(),None,None,None,
                0,len(x),tuple(len(f) for f in folds),source_sha,
                verification_status="UNVERIFIED_NO_REPRODUCIBLE_ASSOCIATION",
            )
        else:
            k=candidate_id
            observed=(support[:,:,k]>0)&am
            s=level[:,:,k][observed]
            a=anchor[observed]
            aa=a-a.mean();ss=s-s.mean()
            slope=float(np.dot(aa,ss)/max(np.dot(aa,aa),1e-12))
            intercept=float(s.mean()-slope*a.mean())
            errors,valid,count=_episode_reference_errors(
                level[:,:,k],support[:,:,k],anchor,am,slope,intercept,
                min_paired_times=self.min_paired_times)
            usable=errors[valid]
            if len(usable)<100:
                raise ValueError("Insufficient individual TRAIN source/reference agreements")
            cutoff=float(max(1e-8,np.quantile(usable,.99)))
            self.evidence=AnchorEvidence(
                self.anchor_provenance,
                tuple(self.model.bank.source.groups[k].sensor_indices),
                k,candidate_score,correlations,
                intercept,slope,cutoff,len(usable),len(x),
                tuple(len(f) for f in folds),source_sha,
                verification_status="TRAIN_ASSOCIATION_AND_RESIDUAL_FROZEN",
            )
        self.evidence_hash=sha256(json.dumps(
            self.evidence.as_dict(),sort_keys=True,allow_nan=False
        ).encode("utf8")).hexdigest()
        return self

    def predict(self,x,mask,anchor_x,anchor_mask):
        if self.evidence is None:
            raise RuntimeError("Calibrate independent reference on TRAIN first")
        a,m=check_measurements(x,mask)
        anchor,am=_check_anchor(anchor_x,anchor_mask,a.shape[:2])
        base=self.model.predict(a,m)
        old=base["direct_contributions"]
        corrected=old.copy()
        statuses=np.full(len(a),"UNVERIFIED",dtype=object)
        rejection=np.zeros(len(a),dtype=bool)
        error=np.full(len(a),np.nan,dtype=float)
        n_paired=np.zeros(len(a),dtype=int)
        if self.evidence.matched_concept_index is not None:
            if self.model.bank.sha != self.evidence.source_definition_sha256:
                raise AssertionError("Unexpected change to source definitions after reference fit")
            k=self.evidence.matched_concept_index
            level,support=self.model.bank.transform(a,m)
            errors,valid,n=_episode_reference_errors(
                level[:,:,k],support[:,:,k],anchor,am,
                self.evidence.affine_slope,self.evidence.affine_intercept,
                min_paired_times=self.min_paired_times)
            n_paired=n
            error=np.where(valid,errors,np.nan)
            rejection=~valid|(errors>self.evidence.train_median_residual_cutoff_99pct)
            statuses=np.where(~valid,"NO_REFERENCE",
                              np.where(rejection,"DISCORDANT","SUPPORTED"))
            # Only a mathematical source-level fallback, never a risk
            # contribution from the reference channel itself.
            corrected[rejection,k]=0.
        amended=self.model.bias+corrected.sum(axis=1)
        unchanged=(corrected==old)
        if not np.isfinite(amended).all() or not np.isfinite(corrected).all():
            raise AssertionError("Non-finite source-verified logit")
        if np.max(np.abs(amended-(self.model.bias+corrected.sum(axis=1))))>1e-12:
            raise AssertionError("Non-additive reference-protected prediction")
        if self.evidence.matched_concept_index is None:
            np.testing.assert_array_equal(amended,base["logits"])
        else:
            k=self.evidence.matched_concept_index
            other=np.ones(old.shape[1],bool);other[k]=False
            np.testing.assert_array_equal(corrected[:,other],old[:,other])
        return {
            "baseline_logits":base["logits"],
            "baseline_probability":base["probability"],
            "baseline_contributions":old,
            "corrected_contributions":corrected,
            "corrected_logits":amended,
            "probability":expit(amended),
            "correction_to_logit":amended-base["logits"],
            "reference_rejected":rejection,
            "reference_status":statuses,
            "median_reference_residual":error,
            "paired_measurements":n_paired,
            "matched_concept_index":self.evidence.matched_concept_index,
            "anchor_used_as_risk_feature":False,
            "source_model_sha256":self.evidence.source_definition_sha256,
            "frozen_verifier_sha256":self.evidence_hash,
            "max_additive_error":float(np.max(np.abs(
                amended-(self.model.bias+corrected.sum(axis=1))))),
        }

    def passport(self):
        if self.evidence is None:
            raise RuntimeError("No independent reference calibration")
        return {"reference_fit":self.evidence.as_dict(),
                "reference_fit_sha256":self.evidence_hash,
                "underlying_frozen_native_model_sha256":self.model.frozen_network_sha,
                "task_labels_used_to_match_reference":False,
                "reference_raw_values_used_as_risk_predictor":False,
                "provenance_of_reference_is_external_assumption_not_algorithmic_proof":True,
                "joint_reference_and_source_corruption_can_escape":True}
