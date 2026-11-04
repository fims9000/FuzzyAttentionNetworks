"""V21: falsify physical identity from source persistence and risk prediction."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.provenance_data_v21 import (
    FAMILIES,CHANNELS,DISCOVERY_N,SCREEN_N,CORROBORATION_N,
    make_v21_provenance_cohort,load_oracle_source_truth_for_postfit_audit,
)
from conceptfan_discovery.candidate_admission_v17 import (
    ProposedObservableConcept,propose_observable_residual_concept,
)
from conceptfan_discovery.risk_screen_v18 import screen_risk_association
from conceptfan_discovery.measurement_corroboration_v21 import (
    corroborate_candidate_on_disjoint_measured_reference,provenance_passport,
)
from conceptfan_discovery.experiment_provenance_v21 import _select_reviews
from conceptfan_discovery.expert_feedback_v16 import native_expert_loss
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


def candidate_for_sensor(sensor:int):
    unit=np.eye(4,dtype=np.float64)[:,sensor]
    return ProposedObservableConcept(
        candidate_id=f"test-sensor-{sensor}",
        direction=unit,
        train_episode_count=350,
        positive_lag_strength=.5,
        permutation_p=.005,
        bootstrap_abs_cosine_q10=.99,
        top_signed_sensors=((CHANNELS[sensor],1.),),
    )


def _gate(data,candidate,*,seed=91):
    training=data.discovery.train
    discovery=slice(0,DISCOVERY_N)
    risk=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
    gate=slice(DISCOVERY_N+SCREEN_N,DISCOVERY_N+SCREEN_N+CORROBORATION_N)
    return corroborate_candidate_on_disjoint_measured_reference(
        candidate,training.x[gate],data.independent_train_reference[gate],
        provenance_episode_ids=training.episode_ids[gate],
        discovery_episode_ids=training.episode_ids[discovery],
        risk_screen_episode_ids=training.episode_ids[risk],
        model_channel_devices=data.model_sensor_devices,
        reference_device_id=data.reference_device_id,
        seed=seed,permutations=99,bootstraps=99,
    )


@pytest.mark.parametrize("family",FAMILIES)
def test_v21_five_sources_serialize_only_measurements_to_neural_splits(family):
    data=make_v21_provenance_cohort(family)
    again=make_v21_provenance_cohort(family)
    train=data.discovery.train
    val=data.discovery.validation
    assert train.x.shape==(960,24,4)
    assert val.x.shape==(240,24,4)
    assert len(train.y)==960 and len(val.y)==240
    assert np.array_equal(train.x,again.discovery.train.x)
    assert np.array_equal(val.x,again.discovery.validation.x)
    assert np.array_equal(train.mask,np.ones_like(train.mask))
    assert train.oracle_states is None and val.oracle_states is None
    assert set(train.episode_ids).isdisjoint(set(val.episode_ids))
    assert data.independent_train_reference.shape==(960,24)
    assert data.independent_validation_reference.shape==(240,24)
    assert "device_c" not in data.model_sensor_devices
    truth=load_oracle_source_truth_for_postfit_audit(family)
    assert truth["synthetic_physical_new"].shape==(240,24)
    assert truth["synthetic_instrument_artifact"].shape==(240,24)


def test_clean_physical_source_is_correlated_with_independent_reference():
    data=make_v21_provenance_cohort("physical_predictive")
    result=_gate(data,candidate_for_sensor(2))
    assert result.status=="ACCEPT_MEASUREMENT_CORROBORATION"
    assert result.absolute_pearson_r>.5
    assert not result.clinical_or_physical_causal_identity_established
    assert not result.cross_device_common_mode_artifact_ruled_out
    assert not provenance_passport(result)["causal_physiological_mechanism_identified"]


def test_instrument_artifact_is_not_clean_reference_corroborated_even_when_predictive():
    data=make_v21_provenance_cohort("artifact_predictive")
    result=_gate(data,candidate_for_sensor(3))
    assert result.status=="ABSTAIN_UNCORROBORATED_SENSOR_DIRECTION"
    risk=data.discovery.train
    chosen=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
    evidence=screen_risk_association(
        candidate_for_sensor(3),risk.x[chosen],risk.mask[chosen],
        risk.y[chosen],
        screening_episode_ids=risk.episode_ids[chosen],
        discovery_episode_ids=risk.episode_ids[:DISCOVERY_N],
        seed=2112,permutations=99,bootstraps=99,
    )
    assert evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION"
    assert result.status!="ACCEPT_MEASUREMENT_CORROBORATION"


def test_common_mode_instrument_artifact_can_spoof_independent_device():
    data=make_v21_provenance_cohort("artifact_common_mode_reference")
    proof=_gate(data,candidate_for_sensor(3))
    assert proof.status=="ACCEPT_MEASUREMENT_CORROBORATION"
    assert proof.absolute_pearson_r>.5
    assert not proof.cross_device_common_mode_artifact_ruled_out
    assert not proof.clinical_or_physical_causal_identity_established


def test_provenance_requires_disjoint_train_ids_and_different_device():
    data=make_v21_provenance_cohort("physical_predictive")
    train=data.discovery.train
    sl=slice(650,900)
    with pytest.raises(ValueError,match="disjoint"):
        corroborate_candidate_on_disjoint_measured_reference(
            candidate_for_sensor(2),
            train.x[sl],data.independent_train_reference[sl],
            provenance_episode_ids=train.episode_ids[sl],
            discovery_episode_ids=train.episode_ids[:700],
            risk_screen_episode_ids=train.episode_ids[700:900],
            model_channel_devices=data.model_sensor_devices,
            reference_device_id=data.reference_device_id,
        )
    with pytest.raises(ValueError,match="independently"):
        corroborate_candidate_on_disjoint_measured_reference(
            candidate_for_sensor(2),
            train.x[sl],data.independent_train_reference[sl],
            provenance_episode_ids=train.episode_ids[sl],
            discovery_episode_ids=train.episode_ids[:350],
            risk_screen_episode_ids=train.episode_ids[350:650],
            model_channel_devices=data.model_sensor_devices,
            reference_device_id="device_a",
        )


def test_v21_train_review_requires_corroboration_and_no_oracle():
    physical=make_v21_provenance_cohort("physical_predictive")
    accepted=_gate(physical,candidate_for_sensor(2))
    good,manifest=_select_reviews(
        physical,accepted,arm="measured_reference_feedback",
    )
    bad,bad_manifest=_select_reviews(
        physical,accepted,arm="intentionally_inverted_feedback",
    )
    assert len(good.accepted)==len(bad.accepted)==80
    assert manifest["review_effective_binary_supervision"]==80
    assert not manifest["review_is_real_doctor"]
    assert not manifest["audit"]["clinical_validity_established"]
    assert all(
        a[4] != b[4] for a,b in zip(good.accepted,bad.accepted)
    )
    artifact=make_v21_provenance_cohort("artifact_predictive")
    refused=_gate(artifact,candidate_for_sensor(3))
    unsure,receipt=_select_reviews(
        artifact,refused,arm="measured_reference_feedback",
    )
    assert len(unsure.accepted)==0
    assert receipt["audit"]["uncertain"]==80
    assert receipt["audit"]["unknown_is_negative"] is False


def test_native_review_never_creates_an_external_classifier_or_sensor_shortcut():
    data=make_v21_provenance_cohort("physical_predictive")
    accepted=_gate(data,candidate_for_sensor(2))
    prepared,_=_select_reviews(data,accepted,arm="measured_reference_feedback")
    basis=np.eye(4,3,dtype=np.float64)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=basis,channels=CHANNELS,
        forecast_horizon=6,n_observation_concepts=2,
        adaptive_feature_gradients=True,
        scales=(1,4,12),
    )
    x=torch.from_numpy(data.discovery.train.x[:64])
    mask=torch.from_numpy(data.discovery.train.mask[:64])
    model.train()
    output=model(x,mask,process_mask=mask)
    loss,receipt=native_expert_loss(
        output.physical,
        torch.from_numpy(data.discovery.train.episode_ids[:64]),
        prepared,
    )
    loss.backward()
    assert receipt["supervised_slots"]==1
    assert float(model.physical.feature_logits.grad.abs().sum())>0
    assert float(output.decomposition_error())<1e-6
