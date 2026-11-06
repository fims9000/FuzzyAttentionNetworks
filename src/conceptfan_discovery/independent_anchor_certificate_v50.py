"""V50: explicitly conditional source-name calibration from attested external assay.

Without an independent reference, the component MUST label the existing
concept as a statistical proxy. Even with it, a TRAIN/VALID certificate
is environment-scoped, not a per-episode physical truth or causal proof.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge

from .observational_twins_v50 import energy_from_trajectory_v50

MIN_VALID_ASSAY_EPISODES=80
CORRELATION_LIMIT=.75
RIDGE_ALPHA=10.
COV_BOOTSTRAP=48
COV_BOOTSTRAP_EPISODES=250


def _measured_features(x,mask):
    x=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask)
    if (x.ndim!=3 or x.shape[1:]!=(16,4) or x.shape!=m.shape
            or m.dtype!=np.bool_ or len(x)==0
            or not np.isfinite(x[m]).all()):
        raise ValueError("Four real [N,16,4] sensors plus independent boolean masks required")
    return np.concatenate((np.where(m,x,0.),m.astype(np.float64)),axis=2)


def _paired_covariance(x,mask):
    a=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    if a.shape!=m.shape or a.ndim!=3 or a.shape[1:]!=(16,4):
        raise ValueError("Invalid measured covariance cohort")
    x=a.reshape(-1,4)
    obs=m.reshape(-1,4)
    cov=np.empty((4,4),dtype=np.float64)
    for i in range(4):
        for j in range(4):
            both=obs[:,i]&obs[:,j]
            if int(both.sum())<50:
                raise ValueError("Insufficient actual observed channel-pair covariance")
            xi=x[both,i]
            xj=x[both,j]
            cov[i,j]=np.sum((xi-xi.mean())*(xj-xj.mean()))/(len(xi)-1)
    return cov


def _cov_shift(cov_current,cov_train):
    return float(np.linalg.norm(cov_current-cov_train,ord="fro")/
                 max(1e-8,np.linalg.norm(cov_train,ord="fro")))


def _assay_valid_arrays(x,mask,assay):
    x=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask)
    reference=np.asarray(assay,dtype=np.float64)
    _measured_features(x,m)
    if reference.shape!=x.shape[:2] or not np.isfinite(reference).all():
        raise ValueError("External assay must provide independent [N,16] measured time series")
    return x,m,reference


def _masked_anchor_subset(x,mask,assay,n,seed):
    if len(x)<n or n<MIN_VALID_ASSAY_EPISODES:
        raise ValueError("80 fresh independent VALID assay episodes required")
    ids=np.sort(np.random.default_rng(seed).choice(len(x),size=n,replace=False))
    return x[ids],mask[ids],assay[ids],ids


@dataclass(frozen=True)
class AttestedSemanticCalibrationV50:
    physical_device_id:str
    independent_reference_id:str
    declared_target:str
    train_assay_episodes:int
    validation1_assay_episodes:int
    validation2_assay_episodes:int
    ridge_coefficients:tuple[float,...]
    ridge_intercept:float
    VALID1_assay_pearson:float
    VALID2_assay_pearson:float
    reference_residual_95pct:float
    train_sensor_covariance:tuple[tuple[float,...],...]
    train_only_covariance_shift_limit:float
    dual_VALID_admitted:bool
    independently_attested_external_sensor:bool=True
    estimated_source_not_directly_measured_on_TEST:bool=True
    is_universal_semantic_identity_proof:bool=False

    def audit(self):
        return asdict(self)


def fit_attested_source_calibration_v50(
    train_x,train_mask,train_external_assay,
    valid1_x,valid1_mask,valid1_external_assay,
    valid2_x,valid2_mask,valid2_external_assay,
    *,physical_device_id:str,independent_reference_id:str,
    train_reference_budget_fraction:float,seed:int,
):
    """Only assayed TRAIN/VALID, never hidden S,H,TEST Y/source energy."""
    if (not physical_device_id or not independent_reference_id
            or train_reference_budget_fraction not in (.05,.20,1.)):
        raise ValueError("Requires attested device/reference and frozen TRAIN budget")
    tr,mtr,atr=_assay_valid_arrays(train_x,train_mask,train_external_assay)
    v1,m1,a1=_assay_valid_arrays(valid1_x,valid1_mask,valid1_external_assay)
    v2,m2,a2=_assay_valid_arrays(valid2_x,valid2_mask,valid2_external_assay)
    if len(tr)<500 or len(v1)<150 or len(v2)<150:
        raise ValueError("Train and two disjoint valid cohorts required")
    ntr=max(1,int(np.floor(train_reference_budget_fraction*len(tr))))
    tr_ids=np.sort(np.random.default_rng(seed+1).choice(len(tr),size=ntr,replace=False))
    selected_features=_measured_features(tr[tr_ids],mtr[tr_ids]).reshape(-1,8)
    selected_assay=atr[tr_ids].reshape(-1)
    model=Ridge(alpha=RIDGE_ALPHA,fit_intercept=True)
    model.fit(selected_features,selected_assay)
    if len(model.coef_)!=8 or not np.isfinite(model.coef_).all():
        raise FloatingPointError("Invalid TRAIN anchor-to-sensor readout fit")
    validation_receipts=[]
    residuals=[]
    for data,mask,assay,fold in (
        (v1,m1,a1,1),(v2,m2,a2,2),
    ):
        xx,mm,rr,ids=_masked_anchor_subset(
            data,mask,assay,MIN_VALID_ASSAY_EPISODES,seed+fold*41)
        pred=model.predict(_measured_features(xx,mm).reshape(-1,8))
        actual=rr.reshape(-1)
        corr=float(np.corrcoef(pred,actual)[0,1]) if (
            np.std(pred)>1e-8 and np.std(actual)>1e-8) else 0.
        validation_receipts.append(corr)
        residuals.append(np.abs(pred-actual))
    admitted=bool(all(x>=CORRELATION_LIMIT for x in validation_receipts))
    train_cov=_paired_covariance(tr,mtr)
    if len(tr)<COV_BOOTSTRAP_EPISODES:
        raise ValueError("Missing calibration cohort for TEST-time covariance change")
    rng=np.random.default_rng(seed+1510)
    null_scores=[]
    for _ in range(COV_BOOTSTRAP):
        idx=rng.choice(len(tr),size=COV_BOOTSTRAP_EPISODES,replace=True)
        sample_cov=_paired_covariance(tr[idx],mtr[idx])
        null_scores.append(_cov_shift(sample_cov,train_cov))
    shift_limit=float(max(.04,np.quantile(null_scores,.99)*1.15))
    return AttestedSemanticCalibrationV50(
        physical_device_id=str(physical_device_id),
        independent_reference_id=str(independent_reference_id),
        declared_target="named_source_1",
        train_assay_episodes=ntr,
        validation1_assay_episodes=MIN_VALID_ASSAY_EPISODES,
        validation2_assay_episodes=MIN_VALID_ASSAY_EPISODES,
        ridge_coefficients=tuple(float(v) for v in model.coef_),
        ridge_intercept=float(model.intercept_),
        VALID1_assay_pearson=float(validation_receipts[0]),
        VALID2_assay_pearson=float(validation_receipts[1]),
        reference_residual_95pct=float(np.quantile(
            np.concatenate(residuals),.95)),
        train_sensor_covariance=tuple(tuple(float(x) for x in row)
                                     for row in train_cov),
        train_only_covariance_shift_limit=shift_limit,
        dual_VALID_admitted=admitted,
    )


def infer_calibrated_source_energy_v50(fit,x,mask):
    """Inferred from X; does not use real target assay at TEST."""
    t=_measured_features(x,mask)
    coef=np.asarray(fit.ridge_coefficients,dtype=np.float64)
    if coef.shape!=(8,) or not np.isfinite(coef).all():
        raise ValueError("Invalid TRAIN-only measurement-to-source coefficients")
    predicted=np.einsum("ntk,k->nt",t,coef)+fit.ridge_intercept
    energy=energy_from_trajectory_v50(predicted)
    return {
        "inferred_trajectory":predicted,
        "inferred_target_energy":energy,
        "target_natively_measured_on_TEST":False,
        "model_used_TEST_source_truth":False,
    }


def assess_semantic_scope_v50(fit,x,mask,*,device_id:str,external_reference_id:str):
    """Reject physical name on mismatched device or unlabeled readout shift."""
    measured=_measured_features(x,mask)
    if fit is None:
        return {
            "status":"STATISTICAL_PROXY_UNIDENTIFIABLE",
            "physical_label_allowed":False,"covariance_shift":None,
            "requires_external_sensor_or_intervention":True,
        }
    if (device_id!=fit.physical_device_id or
            external_reference_id!=fit.independent_reference_id):
        return {
            "status":"DEVICE_OR_ATTESTED_REFERENCE_MISMATCH",
            "physical_label_allowed":False,"covariance_shift":None,
            "requires_external_sensor_or_intervention":True,
        }
    if not fit.dual_VALID_admitted:
        return {
            "status":"DUAL_VALID_ASSAY_REJECTED",
            "physical_label_allowed":False,"covariance_shift":None,
            "requires_external_sensor_or_intervention":True,
        }
    cov=_paired_covariance(x,mask)
    shift=_cov_shift(cov,np.asarray(fit.train_sensor_covariance))
    acceptable=bool(shift<=fit.train_only_covariance_shift_limit)
    return {
        "status":("ASSAY_CALIBRATED_SAME_ENVIRONMENT" if acceptable else
                  "READOUT_DISTRIBUTION_SHIFT_REVOKED"),
        "physical_label_allowed":acceptable,
        "covariance_shift":float(shift),
        "train_only_shift_threshold":float(fit.train_only_covariance_shift_limit),
        "per_patient_physical_truth_certified":False,
        "cannot_detect_gauge_symmetries_that_leave_observed_X_unchanged":True,
    }


def audited_concept_risk_view_v50(
    frozen_risk_probability,frozen_k2_probability,concept_proxy_membership,
    source_measured_support,*,scope,mode:Literal["predictive","strict_semantic"]="predictive",
):
    """Attach honest semantics WITHOUT silently modifying a risk predictor.

    'predictive' keeps statistical q/risk but never invents a physical name.
    'strict_semantic' returns exact K2 if external semantic evidence absent.
    """
    p=np.asarray(frozen_risk_probability,dtype=np.float64)
    k=np.asarray(frozen_k2_probability,dtype=np.float64)
    q=np.asarray(concept_proxy_membership,dtype=np.float64)
    support=np.asarray(source_measured_support,dtype=bool)
    if (p.ndim!=1 or p.shape!=k.shape or p.shape!=q.shape or p.shape!=support.shape
            or not np.isfinite(p).all() or not np.isfinite(k).all()
            or (p<0).any() or (p>1).any() or
            not np.isfinite(q).all() or (q<0).any() or (q>1).any()):
        raise ValueError("Invalid source/risk/scoping evidence")
    named=bool(scope["physical_label_allowed"])
    if mode=="strict_semantic":
        output=np.where(named,p,k)
    elif mode=="predictive":
        output=p.copy()
    else:
        raise ValueError("Unregistered concept risk policy")
    return {
        "probability":output,
        "concept_membership":q.copy(),
        "concept_is_measured_statistical_proxy":support.copy(),
        "physical_source_name":("калиброванный по внешнему датчику источник 1"
                                if named else None),
        "concept_description_ru":(
            "Оценка источника 1 калибрована по независимому датчику "
            "при проверенных условиях; достоверность каждого случая не доказана."
            if named else
            "Дополнительный статистический признак из измерений ON/OFF; "
            "физическая принадлежность источнику не установлена."),
        "semantic_evidence_status":scope["status"],
        "no_physical_claim_without_attested_external_measurement":True,
        "per_episode_identity_certified":False,
        "exact_K2_fallback_if_strict_and_uncertified":mode=="strict_semantic" and not named,
        "predictive_risk_does_not_imply_physical_truth":True,
    }
