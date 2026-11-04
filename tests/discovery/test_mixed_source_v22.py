"""V22: multisensor identifiability and native concept training contracts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.mixed_source_data_v22 import (
    FAMILIES,MIXING,MODEL_CHANNELS,
    make_observable_v22,train_only_whitening,build_native_v22_data,
    load_oracle_v22_validation_for_postfit_audit,
)
from conceptfan_discovery.environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from conceptfan_discovery.experiment_native_risk_v18 import _config
from conceptfan_discovery.clinical_multiscale_v10 import fit_multiscale
from conceptfan_discovery.experiment_mixed_source_v22 import _oracle_alignment_no_training_feedback


@pytest.mark.parametrize("family",FAMILIES)
def test_four_dense_mixture_families_sealed_truth_and_train_only_scaler(family):
    observed=make_observable_v22(family)
    again=make_observable_v22(family)
    assert observed.train_x_full.shape==(800,30,5)
    assert observed.validation_x_full.shape==(300,30,5)
    assert observed.train_environment.shape==(800,)
    assert np.array_equal(observed.train_x_full,again.train_x_full)
    assert np.array_equal(observed.train_environment,again.train_environment)
    assert not hasattr(observed,"latent")
    assert not hasattr(observed,"true_physical_source")
    assert observed.no_oracle_latent_trajectories_exposed
    assert 0.4<abs(np.dot(MIXING[:,0],MIXING[:,1]))<.9
    assert np.all(np.abs(MIXING)>.15)
    whitener=train_only_whitening(observed)
    assert whitener.trained_episodes==800
    discovery=build_native_v22_data(observed,whitener)
    assert discovery.train.x.shape==(800,24,5)
    assert discovery.validation.x.shape==(300,24,5)
    assert np.all(discovery.train.x[:,:,2:]==0)
    assert np.all(discovery.validation.x[:,:,2:]==0)
    assert discovery.train.oracle_states is None
    assert discovery.validation.oracle_states is None
    assert set(discovery.train.episode_ids).isdisjoint(discovery.validation.episode_ids)
    assert discovery.receipt["both_arms_receive_identical_whitened_observable_input"]
    assert not np.isnan(discovery.train.x).any()


@pytest.mark.parametrize("family,accepted_reason",[
    ("mixed_distinct_variance","ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"),
    ("mixed_proportional_variance","ABSTAIN_NONIDENTIFIABLE_PROPORTIONAL_ENVIRONMENT_SIGNATURES"),
    ("mixed_sensor_gain_drift","ABSTAIN_ENVIRONMENT_MIXING_CHANGE"),
    ("mixed_artifact_only_variance","ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"),
])
def test_unsupervised_identifiability_and_necessary_negative_controls(family,accepted_reason):
    observed=make_observable_v22(family)
    fit=fit_train_only_environment_source_directions(
        observed,permutations=99,bootstraps=24,
    )
    assert fit.status==accepted_reason
    assert fit.train_sample_count==800
    assert fit.environment_counts==(400,400)
    assert fit.source_filters_original_sensor_space.shape==(5,2)
    assert not fit.source_origin_known_physically
    assert not fit.task_labels_used_for_source_fit
    assert not fit.oracle_latent_states_used_for_fit
    passport=separation_passport(fit)
    assert not passport["physical_vs_instrument_disambiguation_by_statistics_alone"]
    if fit.status.startswith("ACCEPT"):
        basis=native_basis_from_separation(fit)
        assert basis.shape==(5,2)
        np.testing.assert_allclose(basis.T@basis,np.eye(2),atol=1e-8)
        assert fit.bootstrap_min_direction_cosine_q10>=.8
        assert fit.environment_permutation_p<=.01
    else:
        with pytest.raises(ValueError,match="ABSTAIN"):
            native_basis_from_separation(fit)


def test_source_estimator_cannot_use_val_outcomes_or_val_observation_variations():
    observed=make_observable_v22("mixed_distinct_variance")
    initial=fit_train_only_environment_source_directions(
        observed,permutations=99,bootstraps=24,
    )
    from dataclasses import replace
    altered=replace(
        observed,
        validation_x_full=np.full_like(observed.validation_x_full,42),
        validation_labels=1.-observed.validation_labels,
        validation_environment=1-observed.validation_environment,
    )
    second=fit_train_only_environment_source_directions(
        altered,permutations=99,bootstraps=24,
    )
    assert initial.candidate_id==second.candidate_id
    assert initial.status==second.status
    np.testing.assert_allclose(
        initial.source_filters_original_sensor_space,
        second.source_filters_original_sensor_space,rtol=0,atol=0,
    )


def test_independent_training_labels_cannot_influence_source_unmixing():
    from dataclasses import replace
    observed=make_observable_v22("mixed_distinct_variance")
    one=fit_train_only_environment_source_directions(
        observed,permutations=99,bootstraps=24,
    )
    altered=replace(observed,train_labels=1-observed.train_labels)
    two=fit_train_only_environment_source_directions(
        altered,permutations=99,bootstraps=24,
    )
    assert one.candidate_id==two.candidate_id
    np.testing.assert_allclose(
        one.source_vectors_whitened,two.source_vectors_whitened,rtol=0,atol=0,
    )


def test_explicit_semantic_truth_remains_eval_only_and_identifies_both_mixed_sources():
    observed=make_observable_v22("mixed_distinct_variance")
    whitener=train_only_whitening(observed)
    proposal=fit_train_only_environment_source_directions(
        observed,whiten=whitener,permutations=99,bootstraps=24,
    )
    assert proposal.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
    split=build_native_v22_data(observed,whitener).validation
    truth=load_oracle_v22_validation_for_postfit_audit(observed.family)
    values=split.x[:,-6:,:2].mean(axis=1)
    estimate=values@proposal.source_vectors_whitened
    phys=truth["generated_physical_source"][:,-6:].mean(1)
    artifact=truth["generated_instrument_source"][:,-6:].mean(1)
    receipt=_oracle_alignment_no_training_feedback(
        estimate,phys,artifact,observed.validation_environment,
    )
    assert receipt["mean_abs_oracle_source_spearman_eval_only"]>.90
    assert receipt["worst_of_two_oracle_source_abs_spearman_eval_only"]>.85
    assert receipt["frozen_assignment_never_used_in_model"]


def test_native_v10_uses_proposed_mixed_basis_in_actual_decision_graph():
    cohort=make_observable_v22("mixed_distinct_variance")
    w=train_only_whitening(cohort)
    proposed=fit_train_only_environment_source_directions(
        cohort,whiten=w,permutations=99,bootstraps=24,
    )
    native=build_native_v22_data(cohort,w)
    config,dual,v10=_config()
    config["training"]["epochs"]=1
    config["training"]["batch_size"]=256
    model,receipt=fit_multiscale(
        native,native_basis_from_separation(proposed),seed=11,
        base_config=config,v7_config=dual,v10_config=v10,
        consistency=False,feature_exploration=False,
    )
    model.eval()
    x=torch.from_numpy(native.validation.x[:12])
    mask=torch.from_numpy(native.validation.mask[:12])
    with torch.no_grad():
        out=model(x,mask,process_mask=mask)
    assert model.physical.n_slots==2
    assert tuple(model.physical.channels)==MODEL_CHANNELS
    assert out.physical.raw_activation.shape==(12,24,2)
    assert float(out.decomposition_error())<1e-6
    assert torch.isfinite(out.logits).all()
