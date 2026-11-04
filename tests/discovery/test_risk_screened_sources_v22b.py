"""V22b: independent TRAIN outcome association vs intrinsic source formation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from conceptfan_discovery.mixed_source_data_v22 import (
    make_observable_v22,train_only_whitening,build_native_v22_data,
)
from conceptfan_discovery.environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,
)
from conceptfan_discovery.native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,
    risk_plan_passport,
)
from conceptfan_discovery.clinical_multiscale_v10 import fit_multiscale
from conceptfan_discovery.experiment_native_risk_v18 import _config


def _two_train_cohorts(family="mixed_distinct_variance"):
    full=make_observable_v22(family)
    core=replace(
        full,
        train_x_full=full.train_x_full[:600].copy(),
        train_labels=full.train_labels[:600].copy(),
        train_episode_ids=full.train_episode_ids[:600].copy(),
        train_environment=full.train_environment[:600].copy(),
    )
    w=train_only_whitening(core)
    target=build_native_v22_data(core,w)
    xs=w.transform(full.train_x_full[600:])[:,:24]
    ids=full.train_episode_ids[600:]
    assert not set(ids)&set(target.train.episode_ids)
    return full,core,w,target,xs,ids


def _riskplan(full,core,w,target,x,ids,basis):
    return screen_v22_native_sources_on_disjoint_train_labels(
        x,np.ones_like(x),full.train_labels[600:],
        basis,original_source_train_episode_ids=target.train.episode_ids,
        screen_episode_ids=ids,
        permutations=99,bootstraps=99,
    )


def test_two_train_cohorts_and_environment_source_are_different_from_labels():
    full,core,w,data,x,ids=_two_train_cohorts()
    assert len(data.train.episode_ids)==600
    assert len(ids)==200
    assert len(data.validation.y)==300
    assert data.train.oracle_states is None
    assert data.validation.oracle_states is None
    assert w.trained_episodes==600
    bss=fit_train_only_environment_source_directions(
        core,whiten=w,permutations=99,bootstraps=24,
    )
    assert bss.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
    assert bss.train_sample_count==600
    assert bss.source_vectors_whitened.shape==(2,2)
    # Change independent risk labels: unsupervised source proposals cannot
    # read them, and its original candidate ID must remain identical.
    modified=replace(full,train_labels=1-full.train_labels)
    modified_core=replace(core,train_labels=1-core.train_labels)
    other=fit_train_only_environment_source_directions(
        modified_core,whiten=w,permutations=99,bootstraps=24,
    )
    assert other.candidate_id==bss.candidate_id
    assert not bss.oracle_latent_states_used_for_fit
    assert np.array_equal(modified.train_x_full,full.train_x_full)


@pytest.mark.parametrize("family",[
    "mixed_distinct_variance","mixed_artifact_only_variance",
])
def test_task_screen_has_train_only_native_signed_and_no_risk_roles(family):
    full,core,w,data,x,ids=_two_train_cohorts(family)
    proposal=fit_train_only_environment_source_directions(
        core,whiten=w,permutations=99,bootstraps=24,
    )
    assert proposal.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
    plan=_riskplan(full,core,w,data,x,ids,native_basis_from_separation(proposal))
    assert len(plan.per_source_evidence)==2
    assert plan.all_risk_cohort_episodes_disjoint_from_source_training
    assert set(plan.active_native_raw_decision_initialization).isdisjoint(
        plan.structurally_descriptive_only_native_slots
    )
    assert (len(plan.active_native_raw_decision_initialization)
            +len(plan.structurally_descriptive_only_native_slots))==2
    assert all(not item.physical_source_or_clinical_cause_certified
               for item in plan.per_source_evidence)
    assert not risk_plan_passport(plan)["no_clinical_or_causal_source_label_proven"] is False
    for source in plan.active_native_raw_decision_initialization.values():
        assert abs(source)==pytest.approx(.05)


def test_screen_refuses_ids_from_source_training_or_nonbinary_outcomes():
    full,core,w,data,x,ids=_two_train_cohorts()
    basis=np.eye(5,2)
    kw=dict(
        measured_screen_histories=x,
        measured_screen_masks=np.ones_like(x),
        screen_outcomes=full.train_labels[600:],
        basis=basis,
        original_source_train_episode_ids=data.train.episode_ids,
        screen_episode_ids=ids,
        permutations=99,bootstraps=99,
    )
    with pytest.raises(ValueError,match="disjoint"):
        screen_v22_native_sources_on_disjoint_train_labels(
            **{**kw,"screen_episode_ids":np.arange(500,700)},
        )
    with pytest.raises(ValueError,match="invalid"):
        screen_v22_native_sources_on_disjoint_train_labels(
            **{**kw,"screen_outcomes":np.full(200,.5)},
        )
    with pytest.raises(ValueError,match="Invalid"):
        screen_v22_native_sources_on_disjoint_train_labels(
            **{**kw,"permutations":30},
        )


def test_native_no_risk_slot_is_permanently_zero_while_forecasting_stays_native():
    full,core,w,data,x,ids=_two_train_cohorts("mixed_artifact_only_variance")
    source=fit_train_only_environment_source_directions(
        core,whiten=w,permutations=99,bootstraps=24,
    )
    basis=native_basis_from_separation(source)
    plan=_riskplan(full,core,w,data,x,ids,basis)
    config,dual,v10=_config()
    config["training"]["epochs"]=1
    config["training"]["batch_size"]=256
    source_weights=plan.active_native_raw_decision_initialization
    descriptive=plan.structurally_descriptive_only_native_slots
    model,receipt=fit_multiscale(
        data,basis,11,config,dual,v10,
        consistency=False,feature_exploration=False,
        initial_native_raw_decision_weights=(
            source_weights if source_weights else None
        ),
        initial_native_descriptive_only_slots=(
            descriptive if descriptive else None
        ),
    )
    assert receipt["descriptive_only_policy_means_zero_risk_not_nonphysical"]
    assert receipt["train_initial_native_descriptive_only_slot_policy"]==(
        list(descriptive) if descriptive else None
    )
    inputs=torch.from_numpy(data.validation.x[:8])
    mask=torch.from_numpy(data.validation.mask[:8])
    model.eval()
    with torch.no_grad():
        prediction=model(inputs,mask,process_mask=mask)
    assert float(prediction.decomposition_error())<1e-6
    assert torch.isfinite(prediction.logits).all()
    for slot in descriptive:
        assert int(model.physical.signed_risk_policy[slot])==2
        assert float(model.physical.decision_weights()[slot])==0
        with torch.no_grad():
            model.physical.decision_raw_weight[slot]=2.1
        assert float(model.physical.decision_weights()[slot])==0


@pytest.mark.parametrize("bad",[
    ((),None),
    ((0,0),None),
    ((0,2),None),
    ((True,),None),
    ((0,),{0:.05}),
])
def test_initial_native_descriptive_policy_invalid_or_conflicting_slots_rejected(bad):
    slots,weights=bad
    full,core,w,data,x,ids=_two_train_cohorts()
    config,dual,v10=_config()
    config["training"]["epochs"]=0
    with pytest.raises(ValueError,match="Invalid native|must be disjoint"):
        fit_multiscale(
            data,np.eye(5,2),11,config,dual,v10,
            consistency=False,feature_exploration=False,
            initial_native_raw_decision_weights=weights,
            initial_native_descriptive_only_slots=slots,
        )
