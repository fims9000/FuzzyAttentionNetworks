"""V13 theorem-boundary and executable negative controls; synthetic ONLY."""
from __future__ import annotations

import inspect
import numpy as np
import pytest
import torch

from conceptfan_discovery.intervention_identifiability_v13 import (
    estimate_intervention_certificate, InterventionAnchoredConceptModel,
    train_intervention_concept_model,
)
from conceptfan_discovery.intervention_stress_v13 import (
    simulate_randomized_intervention_trials, same_emission_new_episodes,
    oracle_direction_alignment, observable_pca_axis_overlap, run_study,
)


def _fixture(regime="exclusive",k=3,n=120,seed=104):
    train,oracle=simulate_randomized_intervention_trials(
        k=k,regime=regime,seed=seed,n_per_environment=n,
    )
    return train,oracle


def _cert(train, *, verified=True, bootstraps=32):
    return estimate_intervention_certificate(
        train.x,train.mask,train.environment,
        exclusive_single_target_provenance=verified,
        no_downstream_propagation_provenance=verified,
        bootstraps=bootstraps,seed=25,
    )


def test_recover_targeted_response_only_under_verified_exclusive_assumptions():
    train,oracle=_fixture(k=3,n=140)
    cert=_cert(train)
    assert cert.status=="CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE"
    assert cert.response.shape==(7,3)
    assert cert.rank_estimate==3
    assert min(cert.bootstrap_q10_direction_cosine)>.8
    assert oracle_direction_alignment(cert,oracle)["oracle_target_order_column_cosine"]>.97
    assert cert.receipt["no_outcome_or_latent_labels_in_contrast_estimation"]
    assert cert.to_json()["external_exclusivity_is_an_assumption_NOT_testable_from_contrasts"]
    # PCA sees only uninfluenced training controls, cannot demand true labels.
    assert .4<observable_pca_axis_overlap(train,oracle)<1.01
    assert "oracle" not in str(inspect.signature(estimate_intervention_certificate))


def test_unverified_targets_abstain_even_when_contrasts_full_rank_and_stable():
    train,oracle=_fixture("multitarget",k=5,n=180,seed=20261604)
    safe=_cert(train,verified=False)
    dishonest=_cert(train,verified=True)
    assert safe.status=="ABSTAIN_UNVERIFIED_SINGLE_TARGET_PROVENANCE"
    assert safe.receipt["empirical_response_resolution_independent_of_provenance"]=="RESOLVED_OBSERVABLE_ENVIRONMENT_RESPONSES"
    assert dishonest.status=="CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE"
    # This is the negative-control theorem boundary: a false experimental
    # metadata assertion CAN produce a strong yet non-identifiable 'concept'.
    assert oracle_direction_alignment(dishonest,oracle)["oracle_target_order_column_cosine"]<.85
    with pytest.raises(ValueError,match="Only verified"):
        InterventionAnchoredConceptModel(safe)


def test_direct_target_labels_alone_cannot_certify_absence_of_propagation():
    train,_=_fixture(k=3,n=120)
    cert=estimate_intervention_certificate(
        train.x,train.mask,train.environment,
        exclusive_single_target_provenance=True,
        no_downstream_propagation_provenance=False,
        bootstraps=32,seed=25,
    )
    assert cert.status=="ABSTAIN_UNVERIFIED_NO_DOWNSTREAM_PROPAGATION"
    assert cert.receipt["empirical_response_resolution_independent_of_provenance"]=="RESOLVED_OBSERVABLE_ENVIRONMENT_RESPONSES"
    assert not cert.to_json()["no_downstream_propagation_verified_externally"]
    with pytest.raises(ValueError,match="Only verified"):
        InterventionAnchoredConceptModel(cert)


@pytest.mark.parametrize("regime",["duplicated","sham"])
def test_insufficient_or_duplicated_interventions_abstain(regime):
    train,_=_fixture(regime,k=3,n=150)
    cert=_cert(train,verified=True)
    assert cert.status.startswith("ABSTAIN")
    assert cert.receipt["empirical_response_resolution_independent_of_provenance"]!="RESOLVED_OBSERVABLE_ENVIRONMENT_RESPONSES"
    with pytest.raises(ValueError):
        InterventionAnchoredConceptModel(cert)


def test_shuffled_train_environment_labels_cannot_claim_identification():
    train,_=_fixture(k=3,n=150)
    rng=np.random.default_rng(81)
    perm=rng.permutation(train.environment)
    cert=estimate_intervention_certificate(train.x,train.mask,perm,
        exclusive_single_target_provenance=True,
        no_downstream_propagation_provenance=True,bootstraps=32,seed=3)
    assert cert.status!="CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE"


def test_trainable_native_concept_path_exact_signed_logit_and_finite_gradients():
    train,_=_fixture(k=3,n=100)
    cert=_cert(train)
    torch.set_num_threads(1)
    model,receipt=train_intervention_concept_model(
        train.x,train.mask,train.y,cert,seed=11,epochs=8,batch_size=48,
    )
    assert receipt["train_binary_outcome_optimizes_native_concept_definitions"]
    assert receipt["definition_movement_from_intervention_initializer"]>0
    assert not any("teacher" in name or "student" in name for name,_ in model.named_parameters())
    x=torch.from_numpy(train.x[:9])
    m=torch.from_numpy(train.mask[:9])
    model.train()
    out=model(x,m)
    assert out["concepts"].shape==(9,3)
    assert out["contributions"].shape==(9,3)
    assert out["logits"].shape==(9,)
    assert torch.allclose(out["logits"],model.risk_bias+out["contributions"].sum(dim=-1),atol=1e-6)
    loss=model.loss(out,torch.from_numpy(train.y[:9]))
    loss.backward()
    assert torch.isfinite(model.definition.grad).all()
    assert torch.isfinite(model.risk_weight.grad).all()
    assert model.definition.grad.norm()>0
    assert torch.isfinite(model.regularizers()["intervention_jacobian"])


def test_unobserved_placeholders_do_not_create_signal():
    train,_=_fixture(k=3,n=130)
    cert=_cert(train)
    model=InterventionAnchoredConceptModel(cert)
    model.eval()
    x=torch.full((4,36,train.x.shape[-1]),999_999.,dtype=torch.float32)
    mask=torch.zeros_like(x)
    out=model(x,mask)
    assert torch.count_nonzero(out["concepts"])==0
    assert torch.count_nonzero(out["contributions"])==0
    assert torch.allclose(out["logits"],model.risk_bias.expand(4))
    with pytest.raises(ValueError):
        model(x,mask[:,:,:-1])
    with pytest.raises(ValueError):
        model(x,mask+2)


def test_train_only_and_validation_cases_disjoint_and_hidden_oracle_not_needed():
    train,oracle=_fixture(k=3,n=100,seed=77)
    val,oracle_val=same_emission_new_episodes(oracle,seed=2007,n=100)
    cert=_cert(train)
    model,_=train_intervention_concept_model(train.x,train.mask,train.y,cert,seed=3,epochs=3)
    with torch.no_grad():
        pred=model(torch.from_numpy(val.x),torch.from_numpy(val.mask))
    assert torch.isfinite(pred["logits"]).all()
    assert val.environment.sum()==0
    assert oracle_val.emission.shape==(val.x.shape[-1],3)


def test_invalid_experimental_protocol_is_rejected():
    train,_=_fixture(k=3,n=80)
    with pytest.raises(ValueError):
        estimate_intervention_certificate(train.x,train.mask,train.environment,
            no_downstream_propagation_provenance=True,
            exclusive_single_target_provenance="yes",bootstraps=16)
    with pytest.raises(ValueError):
        estimate_intervention_certificate(train.x,train.mask,train.environment,
            no_downstream_propagation_provenance=True,
            exclusive_single_target_provenance=True,bootstraps=1)
    missing=train.mask.copy()
    missing[0,0,0]=np.nan
    with pytest.raises(ValueError):
        estimate_intervention_certificate(train.x,missing,train.environment,
            no_downstream_propagation_provenance=True,
            exclusive_single_target_provenance=True)
    with pytest.raises(ValueError):
        estimate_intervention_certificate(train.x,train.mask,train.environment[:25],
            no_downstream_propagation_provenance=True,
            exclusive_single_target_provenance=True)


def test_synthetic_end_to_end_exports_aggregate_only():
    frame,certs=run_study(sizes=(3,),n_train_per_env=70,n_validation=100,
                          bootstraps=16,train_epochs=5)
    assert len(frame)==4
    assert frame.iloc[0]["certificate_status"]=="CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE"
    assert all(not x.startswith("CERTIFIED") for x in frame.iloc[1:]["certificate_status"])
    assert frame.iloc[0]["min_signed_same_patient_spearman"]>.8
    assert "K3_exclusive" in certs
    assert "train_input_and_environment_sha256" in certs["K3_exclusive"]
    assert all(not any("patient_id" in col or "patient_label" in col for col in frame.columns) for _ in [0])
