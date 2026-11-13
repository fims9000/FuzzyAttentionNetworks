"""V57: new *numeric* calibrated fuzzy concept with a matching TRAIN risk head.

Cannot substitute an externally calibrated q into a V55 head trained on
statistical q: it is a different scale. V57 explicitly retrains the one signed
source coefficient on that exact q and separately validates it on VALID1/2.
The external assay is synthetic, costly, and independently measured; there is
NO clinical, causal, physical-identity or crypto-attestation guarantee.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import numpy as np
from scipy.special import expit

from .constrained_joint_operator_v55 import (
    JointMeasuredConceptFitV55,infer_joint_operator_v55,
)
from .independent_assay_scope_v56 import (
    AssayCorroborationV56,assess_observed_source_scope_v56,
)
from .auto_measured_concept_admission import (
    CandidateGate,evaluate_two_independent_environments,
)
from .isolated_source_admission_v28 import train_source_coefficient

ASSAY_IQR_FLOOR=.10
MIN_PURCHASED_TRAIN=150


@dataclass(frozen=True)
class CalibratedConceptValueContractV57:
    """A TRAIN-/VALID-scoped **measurement-scale** source passport."""
    source_candidate_id:str
    device_id:str
    reference_id:str
    measured_target_name:str
    train_assay_count:int
    train_reference_median:float
    train_reference_iqr:float
    source_to_assay_slope:float
    source_to_assay_intercept:float
    scale_kind:str="AFFINE_ASSAY_ENERGY_TO_SIGMOID"
    source_numeric_not_physical_truth:bool=True
    external_assay_unavailable_at_test:bool=True
    reference_ids_not_authenticated:bool=True

    def passport(self):
        return asdict(self)


@dataclass(frozen=True)
class RefitSourceRiskContractV57:
    source_contract:CalibratedConceptValueContractV57
    risk_gamma_train_only:float
    risk_gate:CandidateGate
    uses_same_q_at_train_and_inference:bool=True
    source_gamma_does_not_reuse_v55:bool=True
    no_direct_raw_to_risk_shortcut:bool=True
    source_physical_identity_not_proven:bool=True

    def passport(self):
        return {
            **asdict(self),
            "risk_gate":self.risk_gate.as_dict(),
        }


def map_to_external_measurement_v57(
    fit:JointMeasuredConceptFitV55,
    reference:CalibratedConceptValueContractV57,
    values,mask
):
    """TRAIN-fitted assay energy + *matching* normalized q for the same source."""
    if (fit.candidate.candidate_id!=reference.source_candidate_id or
        not reference.device_id or not reference.reference_id or
        reference.scale_kind!="AFFINE_ASSAY_ENERGY_TO_SIGMOID" or
        not np.isfinite([
            reference.train_reference_median,reference.train_reference_iqr,
            reference.source_to_assay_slope,reference.source_to_assay_intercept]).all()
        or reference.train_reference_iqr<=0 or
        reference.source_to_assay_slope<=0):
        raise ValueError("Incompatible or uncorroborated source TRAIN calibration")
    original=infer_joint_operator_v55(fit,values,mask)
    q=np.asarray(original["summary"],dtype=np.float64)
    support=np.asarray(original["support"],dtype=np.float64)
    valid=support>=.50
    e=np.maximum(0.,reference.source_to_assay_slope*q+
                 reference.source_to_assay_intercept)
    numeric=expit((e-reference.train_reference_median)/
                  reference.train_reference_iqr)
    numeric=np.where(valid,numeric,.5)
    return {
        "q_assay":numeric,
        "energy_estimate":np.where(valid,e,np.nan),
        "support":support,
        "source_q_v55":q,
        "source_available":valid,
        "reference_is_inference_only_train_calibration":True,
        "not_exact_physical_energy_at_test":True,
    }


def fit_calibrated_concept_and_native_head_v57(
    fit:JointMeasuredConceptFitV55,
    certificate:AssayCorroborationV56,
    purchased_train_assay,
    train_x,train_mask,train_y,frozen_k2_train_logits,
    valid1_x,valid1_mask,valid1_y,frozen_k2_valid1_logits,
    valid2_x,valid2_mask,valid2_y,frozen_k2_valid2_logits,
    *,device_id,reference_id,
):
    """Does not accept TEST values/Y/assay; does not modify frozen K2."""
    if (certificate is None or not certificate.dual_VALID_corroborated or
        certificate.device_id!=device_id or
        certificate.reference_id!=reference_id or
        certificate.candidate_id!=fit.candidate.candidate_id):
        raise ValueError("Independent TRAIN+dual VALID source assay required")
    purchased=np.asarray(purchased_train_assay,dtype=np.float64)
    if (purchased.ndim!=1 or len(purchased)<MIN_PURCHASED_TRAIN or
        len(purchased)!=certificate.train_assay_purchased or
        not np.isfinite(purchased).all() or np.any(purchased<0)):
        raise ValueError("Only purchased independent TRAIN assay is permitted")
    q1,q2,q3=np.quantile(purchased,[.25,.5,.75])
    qscale=float(max(ASSAY_IQR_FLOOR,q3-q1))
    mapping=CalibratedConceptValueContractV57(
        source_candidate_id=fit.candidate.candidate_id,
        device_id=device_id,reference_id=reference_id,
        measured_target_name="independent_synthetic_source_energy",
        train_assay_count=len(purchased),
        train_reference_median=float(q2),
        train_reference_iqr=qscale,
        source_to_assay_slope=certificate.reference_slope,
        source_to_assay_intercept=certificate.reference_intercept,
    )
    qtrain=map_to_external_measurement_v57(fit,mapping,train_x,train_mask)
    qvalid1=map_to_external_measurement_v57(fit,mapping,valid1_x,valid1_mask)
    qvalid2=map_to_external_measurement_v57(fit,mapping,valid2_x,valid2_mask)
    gamma=float(train_source_coefficient(
        frozen_k2_train_logits,qtrain["q_assay"],train_y))
    receipt=evaluate_two_independent_environments(
        fit.candidate.candidate_id,gamma,
        ("VALID1",valid1_y,frozen_k2_valid1_logits,
         qvalid1["q_assay"],qvalid1["support"]),
        ("VALID2",valid2_y,frozen_k2_valid2_logits,
         qvalid2["q_assay"],qvalid2["support"]),
        train_label_count=len(train_y),seed=20265721)
    return RefitSourceRiskContractV57(
        source_contract=mapping,risk_gamma_train_only=gamma,
        risk_gate=receipt)


def predict_calibrated_source_scope_v57(
    fit:JointMeasuredConceptFitV55,
    certificate:AssayCorroborationV56|None,
    risk_contract:RefitSourceRiskContractV57|None,
    values,mask,*,device_id,reference_id
):
    """No TEST assay; missing/gate failure -> bitwise original K2 downstream."""
    q0=infer_joint_operator_v55(fit,values,mask)
    n=len(q0["summary"])
    neutral={"q_assay":np.full(n,.5,np.float64),
             "source_used":np.zeros(n,bool),
             "source_support":np.asarray(q0["support"]),
             "energy_estimate":np.full(n,np.nan),
             "approved":False,
             "physical_or_causal_identity_proven":False,
             "external_test_assay_used":False}
    if certificate is None or risk_contract is None:
        return {**neutral,"status":"ABSTAIN_NO_ASSAY_RISK_CONTRACT"}
    mapfit=risk_contract.source_contract
    if (mapfit.source_candidate_id!=fit.candidate.candidate_id or
        mapfit.device_id!=certificate.device_id or
        mapfit.reference_id!=certificate.reference_id or
        mapfit.device_id!=device_id or mapfit.reference_id!=reference_id):
        return {**neutral,"status":"REVOKED_CONTRACT_DEVICE_REFERENCE_MISMATCH"}
    if not risk_contract.risk_gate.approved:
        return {**neutral,"status":"ABSTAIN_RECALIBRATED_RISK_VALID_FAILED"}
    scope=assess_observed_source_scope_v56(
        fit,certificate,values,mask,
        device_id=device_id,reference_id=reference_id)
    if not scope["approved"]:
        return {**neutral,"status":scope["status"]}
    expected=map_to_external_measurement_v57(fit,mapfit,values,mask)
    return {
        "q_assay":expected["q_assay"],
        "source_used":scope["source_used"],
        "source_support":expected["support"],
        "energy_estimate":expected["energy_estimate"],
        "approved":True,
        "status":"ASSAY_SCALED_MEASURED_PROXY",
        "physical_or_causal_identity_proven":False,
        "external_test_assay_used":False,
    }
