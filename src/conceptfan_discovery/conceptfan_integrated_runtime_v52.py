"""Integrated ConceptFAN inference: V47 signed bottleneck + V50/V51 physical
source calibration + V49 exact expert interventions.

This module deliberately does NOT turn a statistical q into physical truth:
without explicit, same-device external calibration it falls back to the
original independent K2 risk. All source-to-risk transformations must be
included during TRAIN joint optimisation, not substituted after training.

This is an integration runtime, not a new validated SOTA experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from scipy.special import expit

from .anchored_joint_bottleneck_v47 import (
    AnchoredJointConceptBottleneckV47,predict_joint_v47,safe_joint_logits_v47,
)
from .independent_anchor_certificate_v50 import (
    AttestedSemanticCalibrationV50,assess_semantic_scope_v50,
)
from .train_only_energy_debias_v51 import (
    TrainOnlyEnergyDebiasV51,energy_reference_from_independent_assay_v51,
    infer_energy_with_train_only_debias_v51,
)
from .verified_concept_intervention_v49 import (
    apply_verified_concept_interventions_v49,
)

_ALLOWED_ENERGY_METHODS=("V50_RAW","V51_SCALE","V51_AFFINE")
_MIN_OBSERVED_TIMES_PER_SENSOR=4


@dataclass(frozen=True)
class AttestedEnergyMembershipMapV52:
    physical_device_id: str
    external_reference_id: str
    reference_assay_episodes: int
    source_reference_energy_median: float
    source_reference_energy_IQR_scale: float
    external_assay_noise_variance: float=.0256
    trained_with_source_specific_external_assay: bool=True
    not_a_per_episode_identity_certificate: bool=True


@dataclass(frozen=True)
class ConceptRiskTrainingContractV52:
    """The actual source semantics used to TRAIN the joint model.

    A pre-existing V47 head trained on V44 source memberships does NOT meet
    this contract and must not be called with V51 energy memberships.
    """
    physical_device_id: str
    external_reference_id: str
    source_training_kind: str
    energy_correction_method: str
    source_membership_median: float
    source_membership_scale: float

    def check(self,source_calibration,energy_calibration,source_map):
        if self.source_training_kind!="attested_energy_v51":
            raise ValueError("V47/V44 risk-head source distribution is not an attested V51 source")
        if self.energy_correction_method not in _ALLOWED_ENERGY_METHODS:
            raise ValueError("Unregistered energy debias contract")
        ids=(self.physical_device_id,self.external_reference_id)
        if (ids!=(source_map.physical_device_id,source_map.external_reference_id)
                or ids!=(source_calibration.physical_device_id,
                         source_calibration.independent_reference_id)
                or ids!=(energy_calibration.physical_device_id,
                         energy_calibration.external_reference_id)):
            raise ValueError("Source identity/physical device mismatch between TRAIN and inference")
        if (self.source_membership_median!=source_map.source_reference_energy_median
                or self.source_membership_scale!=source_map.source_reference_energy_IQR_scale):
            raise ValueError("Inconsistent TRAIN energy membership normalisation")


def fit_attested_energy_membership_map_v52(
    selected_train_external_assay,*,physical_device_id,external_reference_id):
    """Independent TRAIN assay episodes already purchased for V50/V51.

    Caller must provide ONLY the already-selected costly TRAIN assay episodes,
    not all unpurchased/TEST physical truth.
    """
    if not physical_device_id or not external_reference_id:
        raise ValueError("No independent attested device and reference")
    energies=energy_reference_from_independent_assay_v51(selected_train_external_assay)
    if len(energies)<25:
        raise ValueError("At least 25 independently measured TRAIN episodes")
    lo,hi=np.quantile(energies,[.25,.75])
    return AttestedEnergyMembershipMapV52(
        physical_device_id=physical_device_id,
        external_reference_id=external_reference_id,
        reference_assay_episodes=int(len(energies)),
        source_reference_energy_median=float(np.median(energies)),
        source_reference_energy_IQR_scale=float(max(.1,hi-lo)),
    )


def membership_from_attested_energy_v52(energy,mapping):
    e=np.asarray(energy,dtype=np.float64)
    if (e.ndim!=1 or not np.isfinite(e).all()
            or (e<0).any() or mapping.source_reference_energy_IQR_scale<=0):
        raise ValueError("Invalid measured-source energy or TRAIN-only mapping")
    return expit((e-mapping.source_reference_energy_median)/
                 mapping.source_reference_energy_IQR_scale)


def _observed_support(measurements,mask):
    x=np.asarray(measurements,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    if (x.ndim!=3 or x.shape[1:]!=(16,4) or x.shape!=m.shape
            or not np.isfinite(x[m]).all()):
        raise ValueError("A physical source requires real 16x4 masked sensor measurements")
    return (m.sum(axis=1)>=_MIN_OBSERVED_TIMES_PER_SENSOR).all(axis=1)


def prepare_integrated_concept_v52(
    measurements,mask,*,source_calibration,energy_calibration,membership_map,
    training_contract,physical_device_id,external_reference_id):
    """Attested cohort-scoped source candidate, without TEST source truth/Y."""
    observed_support=_observed_support(measurements,mask)
    training_contract.check(source_calibration,energy_calibration,membership_map)
    scope=assess_semantic_scope_v50(
        source_calibration,measurements,mask,
        device_id=physical_device_id,
        external_reference_id=external_reference_id)
    n=len(observed_support)
    if not scope["physical_label_allowed"]:
        return {
            "q":np.full(n,.5,dtype=np.float64),
            "source_energy":np.full(n,np.nan,dtype=np.float64),
            "source_support":np.zeros(n,dtype=bool),
            "semantic_scope":scope,
            "physical_name":None,
            "value_was_inferred_not_measured_at_TEST":True,
        }
    pred=infer_energy_with_train_only_debias_v51(
        source_calibration,energy_calibration,measurements,mask,
        mode=training_contract.energy_correction_method)
    e=np.asarray(pred["energy"],dtype=np.float64)
    q=membership_from_attested_energy_v52(e,membership_map)
    # A cohort certificate is NOT enough for an individual with no sensor data.
    used=observed_support.copy()
    return {
        "q":np.where(used,q,.5),
        "source_energy":np.where(used,e,np.nan),
        "source_support":used,
        "semantic_scope":scope,
        "physical_name":"Источник 1 (условно откалиброван по внешнему датчику)",
        "value_was_inferred_not_measured_at_TEST":True,
        "energy_correction_admitted":pred["correction_admitted"],
        "source_membership_is_not_physical_probability":True,
    }


def predict_integrated_conceptfan_v52(
    model:AnchoredJointConceptBottleneckV47,
    measurements,mask,original_frozen_k2_logits, *,
    source_calibration,energy_calibration,membership_map,training_contract,
    physical_device_id,external_reference_id):
    """Complete inference; no direct hidden-state/raw-to-risk bypass.

    Model MUST have been trained with q produced by this exact anchored
    source mapping, not with V44 q from a different experimental domain.
    """
    candidate=prepare_integrated_concept_v52(
        measurements,mask,source_calibration=source_calibration,
        energy_calibration=energy_calibration,membership_map=membership_map,
        training_contract=training_contract,
        physical_device_id=physical_device_id,
        external_reference_id=external_reference_id)
    x=np.asarray(measurements,dtype=np.float32)
    m=np.asarray(mask,dtype=bool)
    frozen=np.asarray(original_frozen_k2_logits,dtype=np.float64)
    if frozen.shape!=(len(x),) or not np.isfinite(frozen).all():
        raise ValueError("Original independently fitted K2 fallback is mandatory")
    predicted=predict_joint_v47(
        model,x,m,candidate["q"].astype(np.float32),
        candidate["source_support"])
    logits,used=safe_joint_logits_v47(
        frozen,predicted["logits"],candidate["source_support"],
        approved=candidate["semantic_scope"]["physical_label_allowed"])
    if not np.array_equal(logits[~used],frozen[~used]):
        raise AssertionError("Exact K2 fallback broken")
    # Auditable signed decomposition: require trained V47 bottleneck output
    # to agree with the computed source and known concept contributions.
    with torch.no_grad():
        w=model.known_weight.detach().cpu().numpy().astype(np.float64)
        gamma=float(model.gamma.detach().cpu())
        bias=float(model.bias.detach().cpu())
    known=(predicted["known_concepts"]-.5)*w[None,:]
    source=np.where(candidate["source_support"],
                    gamma*(candidate["q"].astype(np.float32).astype(np.float64)-.5),0.)
    # Float32 source tensor in native head produces expected roundoff;
    # verify actual runtime logits against same signed calculation.
    recomposed=bias+known.sum(axis=1)+source
    if np.max(np.abs(predicted["logits"]-recomposed))>2e-5:
        raise AssertionError("Unexplained hidden bypass in signed concept risk")
    return {
        "probability":expit(logits),
        "logits":logits,
        "original_k2_probability":expit(frozen),
        "original_k2_logits":frozen.copy(),
        "candidate_probability_before_fallback":expit(predicted["logits"]),
        "candidate_logits":predicted["logits"].copy(),
        "source_membership":candidate["q"].copy(),
        "source_energy":candidate["source_energy"].copy(),
        "source_used":used,
        "known_signed_contributions":known,
        "external_source_signed_contribution":source,
        "signed_decomposition_bias":bias,
        "semantic_scope":candidate["semantic_scope"],
        "concept_name_ru":candidate["physical_name"],
        "source_description_ru":(
            "Оценена по измерениям и калибровке внешнего датчика; "
            "истинность каждого случая не гарантирована."
            if candidate["physical_name"] else
            "Нет допустимой физической аттестации; применена исходная модель K2."
        ),
        "physical_reference_used_at_TEST":False,
        "model_source_training_contract_checked":True,
        "risk_only_uses_signed_explicit_concepts":True,
    }


def apply_integrated_expert_review_v52(
    prediction,*,actions:Sequence[str],reference_ids:Sequence[str],
    corrected_membership=None,gamma:float,training_contract):
    """V49 action with EXPLICIT same attested external reference ID.

    Caller is responsible for physically validating the evidence ID; this is
    not a cryptographic authentication or automatically measured truth.
    """
    action=np.asarray(actions,dtype=str)
    reference=np.asarray(reference_ids,dtype=str)
    source_used=np.asarray(prediction["source_used"],dtype=bool)
    if action.shape!=source_used.shape or reference.shape!=source_used.shape:
        raise ValueError("Wrong per-episode action and external physical reference ID")
    changed=np.isin(action,["confirm","reject","replace"])
    if np.any(changed&(reference!=training_contract.external_reference_id)):
        raise ValueError("Expert did not provide the certificate's independent reference")
    result=apply_verified_concept_interventions_v49(
        prediction["original_k2_probability"],prediction["probability"],
        prediction["source_membership"],source_used,action,
        link="linear_membership",gamma=gamma,
        corrected_membership=corrected_membership,reference_ids=reference)
    result["physical_reference_attestation_is_external"]=True
    result["not_a_cryptographic_measurement_validation"]=True
    return result
