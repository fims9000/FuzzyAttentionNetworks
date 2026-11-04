"""V21d: evidence-bounded decision to admit a MEASURED native concept.

This is NOT a clinical or physical semantics classifier. Risk association,
independent measurement and paired acquisition repeatability are separate
sources of evidence. A shared persistent instrument artifact can spoof
ALL THREE; therefore no evidence combination has PHYSICALLY_VALIDATED status.

A candidate-level external instrument-artifact judgement is a hard veto,
not a train-negative event label. V16 interval YES/NO/UNSURE/ARTIFACT
must not be conflated with whole-source provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

from .candidate_admission_v17 import (
    ProposedObservableConcept,
    admit_candidate_without_prediction_drift,
)
from .risk_screen_v18 import RiskAssociationEvidence
from .measurement_corroboration_v21 import MeasurementCorroboration
from .instrument_repeatability_v21c import PairedSourceRepeatability


REVIEW_PROVENANCE_CHOICES=frozenset((
    "unknown","measured_pattern","instrument_artifact",
))
ALLOW_RISK="ALLOW_EXPERIMENTAL_MEASURED_RISK_CANDIDATE"
ALLOW_NO_RISK="ALLOW_EXPERIMENTAL_MEASURED_DESCRIPTIVE_CANDIDATE"
ABSTAIN_REFERENCE="ABSTAIN_NO_INDEPENDENT_REFERENCE"
ABSTAIN_REPEAT="ABSTAIN_NONREPEATABLE_ACQUISITION"
BLOCK_ARTIFACT="BLOCK_DOCUMENTED_INSTRUMENT_ARTIFACT"
NO_CANDIDATE="ABSTAIN_NO_REGISTERED_CANDIDATE"


@dataclass(frozen=True)
class BoundedConceptEvidenceDecision:
    candidate_id: str
    status: str
    reviewer_source_provenance_choice: str
    independent_outcome_risk_accepted: bool
    independent_device_measurement_corroborated: bool
    controlled_instrument_reset_repeatable: bool
    can_insert_provisional_native_slot: bool
    can_train_native_risk_after_insertion: bool
    is_physically_certified: bool = False
    is_clinically_certified: bool = False
    commonmode_artifact_ruled_out: bool = False
    user_review_authenticity_verified: bool = False
    proposed_new_physio_mechanism_discovered: bool = False


def review_candidate_evidence_for_native_admission(
    candidate:ProposedObservableConcept,
    risk:RiskAssociationEvidence|None,
    measurement:MeasurementCorroboration|None,
    paired:PairedSourceRepeatability|None,
    *,
    reviewer_source_provenance_choice:str="unknown",
)->BoundedConceptEvidenceDecision:
    if not isinstance(candidate,ProposedObservableConcept):
        raise ValueError("Expected registered V17 candidate with measured source")
    if reviewer_source_provenance_choice not in REVIEW_PROVENANCE_CHOICES:
        raise ValueError("Unrecognized candidate-level source provenance assertion")
    for evidence,typ in (
        (risk,RiskAssociationEvidence),
        (measurement,MeasurementCorroboration),
        (paired,PairedSourceRepeatability),
    ):
        if evidence is not None and not isinstance(evidence,typ):
            raise ValueError("Incorrect evidence type for candidate")
        if evidence is not None and evidence.candidate_id!=candidate.candidate_id:
            raise ValueError("ABSTAIN: evidence identity must match registered candidate")
    risk_ok=(risk is not None
             and risk.status=="ACCEPT_PREDICTIVE_ASSOCIATION"
             and risk.signed_orientation in (-1,1))
    measurement_ok=(measurement is not None
                    and measurement.status=="ACCEPT_MEASUREMENT_CORROBORATION"
                    and measurement.reference_orientation_from_train_only in (-1,1))
    paired_ok=(paired is not None and
               paired.status=="ACCEPT_ACQUISITION_REPEATABILITY")
    if reviewer_source_provenance_choice=="instrument_artifact":
        status=BLOCK_ARTIFACT
    elif not measurement_ok:
        status=ABSTAIN_REFERENCE
    elif not paired_ok:
        status=ABSTAIN_REPEAT
    elif risk_ok:
        status=ALLOW_RISK
    else:
        status=ALLOW_NO_RISK
    can_insert=status in (ALLOW_RISK,ALLOW_NO_RISK)
    return BoundedConceptEvidenceDecision(
        candidate_id=candidate.candidate_id,
        status=status,
        reviewer_source_provenance_choice=reviewer_source_provenance_choice,
        independent_outcome_risk_accepted=bool(risk_ok),
        independent_device_measurement_corroborated=bool(measurement_ok),
        controlled_instrument_reset_repeatable=bool(paired_ok),
        can_insert_provisional_native_slot=can_insert,
        can_train_native_risk_after_insertion=status==ALLOW_RISK,
    )


def expand_original_native_model_if_provisionally_measured(
    trained_model,
    candidate:ProposedObservableConcept,
    decision:BoundedConceptEvidenceDecision,
):
    """Only permits native K→K+1 zero-init expansion; no physical certification.

    The caller may subsequently choose authorized TRAIN-only model learning,
    but this function never starts risk training or changes previous logits.
    """
    if not isinstance(decision,BoundedConceptEvidenceDecision):
        raise ValueError("Expected auditable concept evidence decision")
    if decision.candidate_id!=candidate.candidate_id:
        raise ValueError("Candidate/decision identity mismatch")
    if not decision.can_insert_provisional_native_slot or decision.status not in (
        ALLOW_RISK,ALLOW_NO_RISK,
    ):
        raise ValueError("ABSTAIN: insufficient measured-source evidence or artifact veto")
    if decision.is_physically_certified or decision.is_clinically_certified:
        raise ValueError("Unsupported auto-physical certification assertion")
    expanded=admit_candidate_without_prediction_drift(
        trained_model,candidate,expert_decision="approve",
    )
    # The original V17 primitive creates a zero-effect slot. This is an
    # *experimental source insertion*, not a risk-oracle sign promotion.
    new=expanded.physical
    if (float(new.decision_weights()[-1].detach())!=0
            or int(new.signed_risk_policy[-1].detach())!=0):
        raise AssertionError("Source insertion changed native risk before training")
    return expanded


def bounded_evidence_passport(value:BoundedConceptEvidenceDecision)->dict:
    return {
        "candidate_id":value.candidate_id,
        "status":value.status,
        "source_reviewer_provenance_claim":value.reviewer_source_provenance_choice,
        "native_candidate_slot_admissible":value.can_insert_provisional_native_slot,
        "native_outcome_risk_training_eligible":value.can_train_native_risk_after_insertion,
        "independent_outcome_task_association":value.independent_outcome_risk_accepted,
        "disjoint_device_measurement_corroboration":value.independent_device_measurement_corroborated,
        "controlled_paired_instrument_repeatability":value.controlled_instrument_reset_repeatable,
        "physical_causal_meaning_certified":False,
        "clinically_validated_by_doctor":False,
        "persistent_environmental_instrument_interference_excluded":False,
        "external_reviewer_identity_verified":False,
        "automatic_known_physiological_mechanism_label_assigned":False,
        "no_patient_level_evidence_exported":True,
    }
