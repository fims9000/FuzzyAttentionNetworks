"""V21d: prospective source evidence boundary and native zero-risk mechanism."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from conceptfan_discovery.candidate_admission_v17 import ProposedObservableConcept
from conceptfan_discovery.risk_screen_v18 import RiskAssociationEvidence
from conceptfan_discovery.measurement_corroboration_v21 import MeasurementCorroboration
from conceptfan_discovery.instrument_repeatability_v21c import PairedSourceRepeatability
from conceptfan_discovery.concept_evidence_policy_v21d import (
    review_candidate_evidence_for_native_admission,
    expand_original_native_model_if_provisionally_measured,
    bounded_evidence_passport,
    ALLOW_RISK,ALLOW_NO_RISK,ABSTAIN_REFERENCE,ABSTAIN_REPEAT,BLOCK_ARTIFACT,
)
from conceptfan_discovery.experiment_admission_policy_v21d import _new_k2_research_model
from conceptfan_discovery.provenance_data_v21 import (
    make_v21_provenance_cohort,CHANNELS,
)
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.native_head_refit_v19e import fit_train_only_native_risk_head


def evidence(status):
    candidate=ProposedObservableConcept(
        candidate_id="measured-source-test",
        direction=np.eye(4)[:,2].copy(),
        train_episode_count=350,positive_lag_strength=.5,
        permutation_p=.005,bootstrap_abs_cosine_q10=.94,
        top_signed_sensors=((CHANNELS[2],1.),),
    )
    risk=RiskAssociationEvidence(
        candidate_id=candidate.candidate_id,
        status=(
            "ACCEPT_PREDICTIVE_ASSOCIATION"
            if status=="risk" else "ABSTAIN_UNSTABLE_OR_WEAK_ASSOCIATION"
        ),
        signed_orientation=(1 if status=="risk" else 0),
        n_screen_episodes=300,
        point_biserial_r=.42 if status=="risk" else .03,
        bootstrap_ci95=(.3,.52) if status=="risk" else (-.1,.13),
        permutation_p_two_sided=.005 if status=="risk" else .85,
        two_split_correlations=(.39,.45) if status=="risk" else (.02,-.01),
        minimum_abs_r=.15,
    )
    reference=MeasurementCorroboration(
        candidate_id=candidate.candidate_id,
        status="ACCEPT_MEASUREMENT_CORROBORATION",
        n_train_provenance_episodes=250,
        observed_signed_pearson_r=.99,
        absolute_pearson_r=.99,
        permutation_p_two_sided=.005,
        signed_bootstrap_ci95=(.97,.995),
        two_halves_signed_correlations=(.98,.99),
        reference_orientation_from_train_only=1,
        reference_device_id="device_c",
    )
    paired=PairedSourceRepeatability(
        candidate_id=candidate.candidate_id,
        status="ACCEPT_ACQUISITION_REPEATABILITY",
        n_independent_calibration_episodes=240,
        first_repeat_pearson_r=.995,
        permutation_p_two_sided=.005,
        bootstrap_ci95=(.994,.998),
        paired_two_half_correlations=(.99,.999),
        before_reference_signed_r=.99,
        after_reference_signed_r=.99,
        external_reference_repeat_r=.98,
        acquisition_reset_declared=True,
    )
    return candidate,risk,reference,paired


@pytest.mark.parametrize("risk_kind,expected,risk_ok",[
    ("risk",ALLOW_RISK,True),
    ("no_risk",ALLOW_NO_RISK,False),
])
def test_separate_descriptive_and_predictive_measured_candidates(risk_kind,expected,risk_ok):
    candidate,risk,reference,paired=evidence(risk_kind)
    decision=review_candidate_evidence_for_native_admission(
        candidate,risk,reference,paired,
        reviewer_source_provenance_choice="measured_pattern",
    )
    assert decision.status==expected
    assert decision.can_insert_provisional_native_slot
    assert decision.can_train_native_risk_after_insertion==risk_ok
    assert not decision.is_physically_certified
    passport=bounded_evidence_passport(decision)
    assert not passport["physical_causal_meaning_certified"]
    assert not passport["clinically_validated_by_doctor"]
    assert not passport["persistent_environmental_instrument_interference_excluded"]
    assert not passport["external_reviewer_identity_verified"]


def test_reference_or_reset_failure_abstains_regardless_of_predictive_risk():
    candidate,risk,reference,paired=evidence("risk")
    unknown=review_candidate_evidence_for_native_admission(
        candidate,risk,None,paired,
    )
    assert unknown.status==ABSTAIN_REFERENCE
    assert not unknown.can_insert_provisional_native_slot
    measurement_failure=review_candidate_evidence_for_native_admission(
        candidate,risk,replace(reference,status="ABSTAIN_UNCORROBORATED_SENSOR_DIRECTION"),paired,
    )
    assert measurement_failure.status==ABSTAIN_REFERENCE
    reset_failure=review_candidate_evidence_for_native_admission(
        candidate,risk,reference,replace(paired,status="ABSTAIN_UNSTABLE_UNDER_INSTRUMENT_RESET"),
    )
    assert reset_failure.status==ABSTAIN_REPEAT
    assert not reset_failure.can_train_native_risk_after_insertion


def test_external_instrument_artifact_veto_overrides_all_statistical_tests():
    candidate,risk,reference,paired=evidence("risk")
    base=_new_k2_research_model()
    veto=review_candidate_evidence_for_native_admission(
        candidate,risk,reference,paired,
        reviewer_source_provenance_choice="instrument_artifact",
    )
    assert veto.status==BLOCK_ARTIFACT
    assert not veto.can_insert_provisional_native_slot
    assert not veto.is_physically_certified
    original={k:v.clone() for k,v in base.state_dict().items()}
    with pytest.raises(ValueError,match="ABSTAIN"):
        expand_original_native_model_if_provisionally_measured(base,candidate,veto)
    assert all(torch.equal(base.state_dict()[k],value) for k,value in original.items())


def test_native_measured_descriptive_slot_is_permanently_zero_for_risk_after_refitting():
    candidate,risk,reference,paired=evidence("no_risk")
    decision=review_candidate_evidence_for_native_admission(
        candidate,risk,reference,paired,
    )
    source=_new_k2_research_model()
    expanded=expand_original_native_model_if_provisionally_measured(
        source,candidate,decision,
    )
    assert expanded.physical.n_slots==3
    assert int(expanded.physical.signed_risk_policy[2])==2
    cohort=make_v21_provenance_cohort("physical_nontask")
    train=cohort.discovery.train
    x=torch.from_numpy(train.x[:12])
    mask=torch.from_numpy(train.mask[:12])
    with torch.no_grad():
        before=source(x,mask,process_mask=mask)
        added=expanded(x,mask,process_mask=mask)
    torch.testing.assert_close(before.logits,added.logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(
        before.physical.future_prediction,added.physical.future_prediction,
        atol=1e-6,rtol=1e-6,
    )
    with torch.no_grad():
        expanded.physical.decision_raw_weight[2]=1.5
    assert float(expanded.physical.decision_weights()[2])==0
    expanded.zero_grad(set_to_none=True)
    out=expanded(x,mask,process_mask=mask)
    out.logits.sum().backward()
    assert expanded.physical.decision_raw_weight.grad is not None
    assert float(expanded.physical.decision_raw_weight.grad[2])==0
    # Even the V19e train-only LBFGS step cannot smuggle descriptive-only
    # information into the target logit after arbitrary head weight changes.
    receipt=fit_train_only_native_risk_head(
        expanded,train,declared_train_episode_ids=train.episode_ids,
        max_iter=35,
    )
    assert receipt.final_objective<=receipt.original_objective+1e-6
    assert float(expanded.physical.decision_weights()[2])==0
    assert int(expanded.physical.signed_risk_policy[2])==2
    reloaded=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.column_stack([np.eye(4,2),candidate.direction]),
        channels=CHANNELS,forecast_horizon=6,
        n_observation_concepts=2,scales=(1,4,12),
        adaptive_feature_gradients=True,
    )
    reloaded.load_state_dict(expanded.state_dict(),strict=True)
    assert int(reloaded.physical.signed_risk_policy[2])==2
    assert float(reloaded.physical.decision_weights()[2])==0
    assert float(reloaded(x,mask,process_mask=mask).decomposition_error())<1e-6
    passport=reloaded.physical.multiscale_passport()
    assert passport["native_risk_policy_code_legend"]["2"]=="descriptive_only_exact_zero_risk"
    assert passport["native_descriptive_only_risk_exclusion_serialized"]
    # V20's alternative optimizer must also respect the serialized native
    # descriptive-only gate; it cannot turn that slot into risk signal.
    from conceptfan_discovery.native_head_factorial_v20 import fit_native_head_factorial_v20
    extra=fit_native_head_factorial_v20(
        reloaded,train,declared_train_episode_ids=train.episode_ids,
        method="adamw",pos_weighted=True,adamw_steps=45,
    )
    assert extra.final_objective<=extra.initial_objective+1e-6
    assert float(reloaded.physical.decision_weights()[2])==0
    assert int(reloaded.physical.signed_risk_policy[2])==2


def test_experimental_predictive_source_enters_with_zero_weight_until_native_training():
    candidate,risk,reference,paired=evidence("risk")
    decision=review_candidate_evidence_for_native_admission(
        candidate,risk,reference,paired,
    )
    original=_new_k2_research_model()
    expanded=expand_original_native_model_if_provisionally_measured(
        original,candidate,decision,
    )
    assert int(expanded.physical.signed_risk_policy[2])==0
    assert float(expanded.physical.decision_weights()[2])==0
    assert decision.can_train_native_risk_after_insertion
    assert not decision.is_physically_certified


def test_mismatched_source_identity_or_fake_reviewer_claim_rejected():
    candidate,risk,reference,paired=evidence("risk")
    with pytest.raises(ValueError,match="identity"):
        review_candidate_evidence_for_native_admission(
            candidate,replace(risk,candidate_id="other"),reference,paired,
        )
    with pytest.raises(ValueError,match="Unrecognized"):
        review_candidate_evidence_for_native_admission(
            candidate,risk,reference,paired,
            reviewer_source_provenance_choice="clinical_physio_certified",
        )
    decision=review_candidate_evidence_for_native_admission(
        candidate,risk,reference,paired,
    )
    with pytest.raises(ValueError,match="certification"):
        expand_original_native_model_if_provisionally_measured(
            _new_k2_research_model(),candidate,
            replace(decision,is_physically_certified=True),
        )
    with pytest.raises(ValueError,match="inconsistent"):
        expand_original_native_model_if_provisionally_measured(
            _new_k2_research_model(),candidate,
            replace(decision,can_train_native_risk_after_insertion=False),
        )
