"""V42 historical physical-reference alignment of V41 measured source pairs.

Extra information: a SEPARATELY SIMULATED instrument readout R = true
slow-source energy + independent noise, available ONLY for fixed TRAIN
and VALID1/VALID2 calibration subjects, never for TEST inference.

This is **not** unsupervised physical/casual identification, nor a medical
method. V41's unlabelled measured-source learner itself sees no R, H, S or
outcome Y. The interval is split-calibrated for noisy reference R only
under exchangeability, not a per-subject truth guarantee for latent E.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass,fields
from pathlib import Path
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from .threeview_centered_v37 import (
    CenteredPairFitV37,ThreeViewFitV37,pair_feature_v37,
)
from .adaptive_pair_consensus_v41 import (
    AdaptivePairFitV41,PairGaugeV41,ORDER,
    adaptive_pair_feature_v41,_gauge_transform,
)

REFERENCE_TRAIN=300
REFERENCE_VALID=200
TRAIN_MIN=80
VALID_MIN=50
TRAIN_MIN_RHO=.35
VALID2_MIN_RHO=.25
VALID2_MIN_COVERAGE=.75
VALID2_MAX_NORMALIZED_MAE=.65
INTERVAL_TARGET=.90


@dataclass(frozen=True)
class PairReferenceCertificateV42:
    pair_name:str
    trained_with_external_reference:bool
    two_VALIDs_approved:bool
    reason:str
    n_train_reference:int
    n_valid1_reference:int
    n_valid2_reference:int
    train_spearman:float|None
    valid2_spearman:float|None
    valid2_reference_coverage:float|None
    valid2_normalized_reference_mae:float|None
    x_knots:tuple[float,...]
    y_reference_energy_knots:tuple[float,...]
    interval_half_width_for_noisy_reference_R:float|None

    def as_dict(self)->dict:
        d=asdict(self)
        d["x_knots"]=list(self.x_knots)
        d["y_reference_energy_knots"]=list(self.y_reference_energy_knots)
        return d


@dataclass(frozen=True)
class ReferenceAnchoredV42:
    source_fit:AdaptivePairFitV41
    pair_certs:dict[str,PairReferenceCertificateV42]
    train_reference_iqr:float
    reference_train_budget:int
    reference_valid1_budget:int
    reference_valid2_budget:int
    physical_reference_is_external_instrument_assumption:bool=True
    physical_identity_unverified_real_world:bool=True
    current_healthy_sensor_inference_not_certified:bool=True

    def audit(self)->dict:
        return {
            "pair_certs":{k:v.as_dict() for k,v in self.pair_certs.items()},
            "source_fit":self.source_fit.audit(),
            "reference_train_IQR_physical_units":self.train_reference_iqr,
            "reference_train_budget":self.reference_train_budget,
            "reference_valid1_budget":self.reference_valid1_budget,
            "reference_valid2_budget":self.reference_valid2_budget,
            "reference_privileged_supervision":True,
            "frozen_train_and_two_valid_before_test":True,
            "coverage_claim_only_for_noisy_R_under_exchangeability":True,
            "not_conditional_coverage_or_true_source_E_guarantee":True,
            "not_clinical_or_realworld_physical_certification":True,
        }


def _validate_reference(values,expected):
    r=np.asarray(values,dtype=np.float64)
    if (r.ndim!=1 or len(r)!=expected or not np.isfinite(r).all()):
        raise ValueError(
            "Independent reference readout must cover EXACT registered TRAIN/VALID cohort")
    return np.maximum(r,0.)


def _pair_gauge_q(fit:AdaptivePairFitV41,x,m,name):
    f=pair_feature_v37(fit.v37_fit.pairs[name],x,m)
    if not fit.gauges[name].calibrated:
        return np.full(len(x),.5),np.zeros(len(x),dtype=bool)
    aligned=_gauge_transform(f["q"],fit.gauges[name])
    return aligned,np.asarray(f["support"])>0


def _spearman(x,y):
    if len(x)<4 or np.ptp(x)<1e-9 or np.ptp(y)<1e-9:
        return None
    rho=float(spearmanr(x,y).statistic)
    return rho if np.isfinite(rho) else None


def _refuse(name,reason,*,ntrain=0,nv1=0,nv2=0,rho=None,rho2=None,
            cov=None,mae=None):
    return PairReferenceCertificateV42(
        name,False,False,reason,ntrain,nv1,nv2,rho,rho2,
        cov,mae,(),(),None)


def _conformal_radius(residuals):
    e=np.sort(np.asarray(residuals,dtype=np.float64))
    if len(e)<VALID_MIN or not np.isfinite(e).all() or np.any(e<0):
        raise ValueError("Insufficient held-out noisy reference residuals")
    rank=min(len(e),int(np.ceil((len(e)+1)*INTERVAL_TARGET)))
    return float(e[rank-1])


def fit_physical_reference_v42(
    source:AdaptivePairFitV41,
    train_x,train_m,train_reference,
    valid1_x,valid1_m,valid1_reference,
    valid2_x,valid2_m,valid2_reference,
    *,
    smoke:bool=False,
)->ReferenceAnchoredV42:
    """No TEST reference, source S/H or Y argument. Exactly 3 separated cohorts."""
    ntr=160 if smoke else REFERENCE_TRAIN
    nv=100 if smoke else REFERENCE_VALID
    if (len(train_x)!=ntr or len(valid1_x)!=nv or len(valid2_x)!=nv
        or len(train_m)!=ntr or len(valid1_m)!=nv
        or len(valid2_m)!=nv):
        raise ValueError("Reference budget mismatch; cannot pass TEST as calibration")
    tr=_validate_reference(train_reference,ntr)
    v1=_validate_reference(valid1_reference,nv)
    v2=_validate_reference(valid2_reference,nv)
    iqr=float(np.quantile(tr,.75)-np.quantile(tr,.25))
    if not np.isfinite(iqr) or iqr<=.05:
        raise ValueError("Invalid independent reference energy dynamic range")
    certs={}
    for name in ORDER:
        if not source.gauges[name].calibrated:
            certs[name]=_refuse(
                name,"NO_TRAIN_BLIND_GAUGE",ntrain=0)
            continue
        qtrain,s_train=_pair_gauge_q(source,train_x,train_m,name)
        ct=int(s_train.sum())
        if ct<TRAIN_MIN:
            certs[name]=_refuse(
                name,"TOO_FEW_TRAIN_RECORDED_REFERENCE_PAIRS",ntrain=ct)
            continue
        rho=_spearman(qtrain[s_train],tr[s_train])
        if rho is None or rho<TRAIN_MIN_RHO:
            certs[name]=_refuse(
                name,"INDEPENDENT_REFERENCE_DISAGREES_WITH_MEASURED_CONCEPT",
                ntrain=ct,rho=rho)
            continue
        iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=0.)
        iso.fit(qtrain[s_train],tr[s_train])
        xx=np.asarray(iso.X_thresholds_,dtype=np.float64)
        yy=np.asarray(iso.y_thresholds_,dtype=np.float64)
        q1,s1=_pair_gauge_q(source,valid1_x,valid1_m,name)
        n1=int(s1.sum())
        if n1<VALID_MIN:
            certs[name]=_refuse(
                name,"TOO_FEW_VALID1_REFERENCE_PAIRS",
                ntrain=ct,nv1=n1,rho=rho)
            continue
        pred1=np.interp(q1[s1],xx,yy)
        radius=_conformal_radius(np.abs(pred1-v1[s1]))
        q2,s2=_pair_gauge_q(source,valid2_x,valid2_m,name)
        n2=int(s2.sum())
        if n2<VALID_MIN:
            certs[name]=_refuse(
                name,"TOO_FEW_VALID2_REFERENCE_PAIRS",
                ntrain=ct,nv1=n1,nv2=n2,rho=rho)
            continue
        p2=np.interp(q2[s2],xx,yy)
        error=np.abs(p2-v2[s2])
        coverage=float((error<=radius).mean())
        mae=float(error.mean()/iqr)
        rho2=_spearman(q2[s2],v2[s2])
        passed=(coverage>=VALID2_MIN_COVERAGE
                and rho2 is not None and rho2>=VALID2_MIN_RHO
                and mae<=VALID2_MAX_NORMALIZED_MAE)
        certs[name]=PairReferenceCertificateV42(
            name,True,passed,
            ("HISTORICAL_REFERENCE_TRAIN_AND_BOTH_VALID_SUPPORTED"
             if passed else "VALID2_PHYSICAL_REFERENCE_NOT_REPRODUCIBLE"),
            ct,n1,n2,rho,rho2,coverage,mae,
            tuple(float(a) for a in xx),
            tuple(float(a) for a in yy),
            radius,
        )
    return ReferenceAnchoredV42(
        source,certs,iqr,ntr,nv,nv,
    )


def predict_physical_reference_v42(
    calibrated:ReferenceAnchoredV42,x,observed_mask,
)->dict:
    """Measured-only TEST: no independent reference R or true latent S input."""
    empirical=adaptive_pair_feature_v41(
        calibrated.source_fit,x,observed_mask)
    n=len(empirical["q"])
    used=np.zeros(n,dtype=bool)
    energy=np.full(n,np.nan,dtype=np.float64)
    low=np.full(n,np.nan,dtype=np.float64)
    high=np.full(n,np.nan,dtype=np.float64)
    radius=np.full(n,np.nan,dtype=np.float64)
    reason=np.asarray(empirical["decision_reason"],dtype="U90").copy()
    for name in ORDER:
        selected=(empirical["selected_pair"]==name)&(
            empirical["support"]>0)
        if not selected.any():
            continue
        certificate=calibrated.pair_certs[name]
        if not certificate.two_VALIDs_approved:
            reason[selected]="REFUSED_NO_HISTORICAL_PHYSICAL_REFERENCE_VALIDITY"
            continue
        fitted=np.interp(
            empirical["q"][selected],
            np.asarray(certificate.x_knots),
            np.asarray(certificate.y_reference_energy_knots))
        half=float(certificate.interval_half_width_for_noisy_reference_R)
        energy[selected]=np.maximum(fitted,0.)
        radius[selected]=half
        low[selected]=np.maximum(0.,energy[selected]-half)
        high[selected]=energy[selected]+half
        used[selected]=True
        reason[selected]=np.where(
            empirical["single_pair_uncorroborated"][selected],
            "REFERENCE_HISTORICAL_ONLY_SINGLE_PAIR_NOT_CORROBORATED",
            "REFERENCE_HISTORICAL_MULTI_PAIR_CURRENT_AGREEMENT")
    assert not np.any(np.isfinite(energy[~used]))
    return {
        **empirical,
        "V41_observational_q_not_physical_energy":empirical["q"].copy(),
        "physical_reference_estimated_energy":energy,
        "physical_reference_interval_lower":low,
        "physical_reference_interval_upper":high,
        "physical_reference_interval_radius":radius,
        "physical_reference_allowed":used,
        "physical_reference_reason":reason,
        "external_reference_R_NOT_used_at_inference":True,
        "true_physical_source_S_NOT_used_at_inference":True,
        "historical_reference_does_NOT_certify_current_faults":True,
        "uncertainty_interval_targets_noisy_R_not_true_E":True,
    }


def _serial_v37(source:AdaptivePairFitV41):
    v=source.v37_fit
    pairs={}
    for name,p in v.pairs.items():
        d=asdict(p)
        for key in (
            "left_eigenvectors","right_mixing","inv_left",
            "inv_right","eigenvalues",
        ):
            d[key]=np.asarray(d[key]).tolist()
        pairs[name]=d
    return {
        "pairs":pairs,
        "train_max_pairwise_q_discrepancy_q95":
            v.train_max_pairwise_q_discrepancy_q95,
        "train_calibration_supported_episodes":
            v.train_calibration_supported_episodes,
        "train_fit_episodes":v.train_fit_episodes,
        "train_calibration_episodes":v.train_calibration_episodes,
        "view_partition_externally_assumed":
            v.view_partition_externally_assumed,
        "independent_view_noise_externally_assumed":
            v.independent_view_noise_externally_assumed,
        "true_physical_semantics_certified":
            v.true_physical_semantics_certified,
    }


def save_physical_reference_v42(path:Path,fit:ReferenceAnchoredV42):
    """Explicit JSON-only, no arbitrary pickled Python modules or test data."""
    path=Path(path)
    if path.exists():
        raise FileExistsError("Do not overwrite scientific reference certificate")
    root={
        "schema":"conceptfan-v42-reference-validity-v1",
        "v37":_serial_v37(fit.source_fit),
        "v41":{
            "gauges":{k:asdict(v) for k,v in fit.source_fit.gauges.items()},
            "train_pairwise_threshold":fit.source_fit.train_pairwise_threshold,
            "train_pairwise_overlap":fit.source_fit.train_pairwise_overlap,
        },
        "v42":{
            "pair_certs":{k:v.as_dict() for k,v in fit.pair_certs.items()},
            "train_reference_iqr":fit.train_reference_iqr,
            "reference_train_budget":fit.reference_train_budget,
            "reference_valid1_budget":fit.reference_valid1_budget,
            "reference_valid2_budget":fit.reference_valid2_budget,
        },
        "no_source_S_or_H_or_patient_Y_stored":True,
        "not_clinically_or_causally_certified":True,
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(
        root,indent=2,ensure_ascii=False,allow_nan=False))
    return path


def load_physical_reference_v42(path:Path)->ReferenceAnchoredV42:
    state=json.loads(Path(path).read_text())
    if (state.get("schema")!="conceptfan-v42-reference-validity-v1"
        or state.get("no_source_S_or_H_or_patient_Y_stored") is not True
        or state.get("not_clinically_or_causally_certified") is not True):
        raise ValueError("Unsafe/missing V42 research reference certificate")
    pair_fits={}
    for name in ORDER:
        data=state["v37"]["pairs"][name].copy()
        for key in (
            "left_eigenvectors","right_mixing","inv_left",
            "inv_right","eigenvalues",
        ):
            data[key]=np.asarray(data[key],dtype=np.float64)
        for key in ("view_a","view_b"):
            data[key]=tuple(data[key])
        pair_fits[name]=CenteredPairFitV37(**data)
    v37=ThreeViewFitV37(
        pairs=pair_fits,
        **{k:v for k,v in state["v37"].items() if k!="pairs"})
    if v37.true_physical_semantics_certified:
        raise ValueError("Cannot load physically certified source claim")
    v41=AdaptivePairFitV41(
        gauges={k:PairGaugeV41(**state["v41"]["gauges"][k]) for k in ORDER},
        train_pairwise_threshold=state["v41"]["train_pairwise_threshold"],
        train_pairwise_overlap=state["v41"]["train_pairwise_overlap"],
        v37_fit=v37,
    )
    certs={}
    for name in ORDER:
        d=state["v42"]["pair_certs"][name].copy()
        d["x_knots"]=tuple(d["x_knots"])
        d["y_reference_energy_knots"]=tuple(
            d["y_reference_energy_knots"])
        p=PairReferenceCertificateV42(**d)
        if (p.pair_name!=name or
            (p.two_VALIDs_approved and (
                not p.trained_with_external_reference
                or len(p.x_knots)<2
                or len(p.x_knots)!=len(p.y_reference_energy_knots)
                or p.interval_half_width_for_noisy_reference_R is None))):
            raise ValueError("Malformed or forged pair certificate")
        certs[name]=p
    obj=ReferenceAnchoredV42(
        v41,certs,
        float(state["v42"]["train_reference_iqr"]),
        int(state["v42"]["reference_train_budget"]),
        int(state["v42"]["reference_valid1_budget"]),
        int(state["v42"]["reference_valid2_budget"]),
    )
    if not np.isfinite(obj.train_reference_iqr) or obj.train_reference_iqr<=0:
        raise ValueError("Invalid original reference calibration scale")
    return obj
