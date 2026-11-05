"""Independent checks for masked source-group proposals, K2 proxy training and gate."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit

from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world, masked_histories, intervene, apply_missing_source,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups, measured_group_concept, group_provenance_fingerprint,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments, select_at_most_one_candidate,
    fixed_unobserved_pca,pca_unknown_values,admitted_test_logit,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from scripts.experiment_automatic_measured_concepts import (
    known_observed_proxy,train_masked_measured_known_concepts,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,
)


def _demo(ntrain=420,nval=300,ntest=300):
    z=extended_nonlinear_world(seed=20261113,
             train_n=ntrain,valid_n=nval,test_n=ntest)
    tr=z["train"]
    observed=masked_histories(z["x"][tr],"MCAR20",seed=20267201)
    return z,observed


def test_masked_train_source_grouping_discovers_multiple_stable_groups():
    d,m=_demo()
    x=d["x"][d["train"]]
    proposal=form_train_measured_groups(x,m,
                  known_sensor_indices=(0,1,2,3),seed=20267204)
    groups={frozenset(q.sensor_indices) for q in proposal.candidates}
    assert frozenset((4,5,6)) in groups
    assert frozenset((7,8)) in groups
    assert frozenset((9,10)) in groups
    assert 11 in proposal.ungrouped_sensors
    assert proposal.excluded_known_sensors==(0,1,2,3)
    assert proposal.outcomes_used_for_grouping is False
    assert all(not q.true_latent_concepts_used_to_form for q in proposal.candidates)
    assert proposal.ambiguous_components==()


def test_proposed_concept_is_neutral_when_unobserved_and_never_reads_hidden_payload():
    d,m=_demo()
    p=form_train_measured_groups(d["x"][d["train"]],m,
                  known_sensor_indices=(0,1,2,3))
    q=p.candidates[0]
    old=d["x"][d["test"]].copy()
    testing_m=masked_histories(old,"MCAR40",seed=123)
    base,coverage=measured_group_concept(q,old,testing_m)
    modified=old.copy()
    modified[~testing_m]=float("nan")
    observed,cov2=measured_group_concept(q,modified,testing_m)
    np.testing.assert_array_equal(base,observed)
    np.testing.assert_array_equal(coverage,cov2)
    testing_m[:,:,q.sensor_indices]=False
    q0,s0=measured_group_concept(q,modified,testing_m)
    np.testing.assert_array_equal(q0,np.full(len(q0),.5))
    np.testing.assert_array_equal(s0,np.zeros(len(s0)))


def test_group_fingerprint_is_reproducible_and_does_not_depend_on_outcome_law():
    d,m=_demo()
    p=form_train_measured_groups(d["x"][d["train"]],m,
                   known_sensor_indices=(0,1,2,3))
    second=form_train_measured_groups(d["x"][d["train"]],m,
                   known_sensor_indices=(0,1,2,3))
    assert group_provenance_fingerprint(p)==group_provenance_fingerprint(second)
    assert d["outcomes"]["physical_extra"].shape==d["outcomes"]["complete_nuisance"].shape
    assert not np.array_equal(d["outcomes"]["physical_extra"],
                              d["outcomes"]["complete_nuisance"])


def test_known_k2_target_observed_proxy_does_not_need_hidden_oracle_states():
    d,m=_demo()
    x=d["x"][d["train"],:,:4]
    knownm=m[:,:,:4]
    targets,support=known_observed_proxy(x,knownm)
    assert targets.shape==(len(x),16,2)
    assert support.shape==(len(x),16,2)
    assert float(targets.min())>=0 and float(targets.max())<=1
    changed=x.copy()
    changed[~knownm]=np.nan
    tgt2,s2=known_observed_proxy(changed,knownm)
    np.testing.assert_array_equal(targets.numpy(),tgt2.numpy())
    np.testing.assert_array_equal(support.numpy(),s2.numpy())


def test_measured_k2_can_train_actual_native_core_without_oracle_targets():
    d,m=_demo(ntrain=180,nval=200,ntest=160)
    x=d["x"][d["train"],:,:4]
    knownm=m[:,:,:4]
    core=make_firewalled_k2(11)
    assert core.encoder.input_proj.in_features==8
    receipt=train_masked_measured_known_concepts(
        core,x,knownm,seed=11,epochs=1)
    assert receipt["target_labels_used_in_native_concept_encoder"] is False
    assert receipt["synthetic_latent_concept_targets_used"] is False
    trained=sha_state(core)
    tested=predict_native(core,d["x"][d["test"],:,:4],
           masked_histories(d["x"][d["test"]],"MCAR20",seed=7)[:,:,:4])
    assert tested["max_additive_error"] < 2e-5
    assert tested["concept_summary"].shape==(160,2)
    assert sha_state(core)==trained


def test_candidate_admission_two_valid_99pct_bands_and_no_label_leak():
    rng=np.random.default_rng(16)
    fit=rng.normal(size=800)
    qtrain=expit(fit)
    ytrain=rng.binomial(1,expit(8*(qtrain-.5)))
    gamma=train_source_coefficient(np.zeros(len(fit)),qtrain,ytrain)
    assert gamma>0
    envs=[]
    for idx in range(2):
        q=expit(rng.normal(size=600))
        y=rng.binomial(1,expit(8*(q-.5)))
        envs.append((f"VAL{idx}",y,np.zeros(600),q,np.ones(600)))
    positive=evaluate_two_independent_environments(
        "candidate1",gamma,envs[0],envs[1],
        train_label_count=len(fit),seed=20)
    assert positive.approved
    assert positive.n_total_valid_labels==1200
    negative=list(envs[1])
    negative[3]=1.0-negative[3]
    rejected=evaluate_two_independent_environments(
        "candidate1",gamma,envs[0],tuple(negative),
        train_label_count=len(fit),seed=20)
    assert not rejected.approved
    assert rejected.environment_receipts[0].passed
    assert not rejected.environment_receipts[1].passed
    assert select_at_most_one_candidate((rejected,positive))=="candidate1"


def test_rejected_unmeasured_candidate_preserves_exact_frozen_logits():
    d,m=_demo()
    p=form_train_measured_groups(d["x"][d["train"]],m,
                    known_sensor_indices=(0,1,2,3))
    group=next(s for s in p.candidates if set(s.sensor_indices)=={4,5,6})
    x=d["x"][d["test"]]
    mask=masked_histories(x,"MCAR20",seed=21)
    mask=apply_missing_source(mask,"source_completely_unobserved")
    q,s=measured_group_concept(group,x,mask)
    before=np.linspace(-2,2,len(q))
    after,used=admitted_test_logit(before,q,s,gamma=7.,approved=True)
    np.testing.assert_array_equal(after,before)
    assert not used.any()
    refused,used2=admitted_test_logit(before,np.ones_like(q),np.ones_like(s),
                                      gamma=7.,approved=False)
    np.testing.assert_array_equal(refused,before)
    assert not used2.any()


def test_pca_baseline_uses_train_statistics_only():
    d,m=_demo()
    p=fixed_unobserved_pca(d["x"][d["train"]],m,tuple(range(4,12)))
    x=d["x"][d["test"]].copy()
    mt=masked_histories(x,"MCAR40",seed=9)
    q,s=pca_unknown_values(p,x,mt)
    x[~mt]=np.nan
    q2,s2=pca_unknown_values(p,x,mt)
    np.testing.assert_array_equal(q,q2)
    np.testing.assert_array_equal(s,s2)


def test_future_shift_does_not_change_frozen_training_group():
    d,m=_demo()
    p=form_train_measured_groups(d["x"][d["train"]],m,
                    known_sensor_indices=(0,1,2,3))
    source=next(q for q in p.candidates if set(q.sensor_indices)=={4,5,6})
    x=d["x"][d["test"]]
    corrupted=intervene(x,"source_common_sign_flip")
    assert np.array_equal(x[:,:,:4],corrupted[:,:,:4])
    assert group_provenance_fingerprint(p)==group_provenance_fingerprint(p)
    good,s=measured_group_concept(source,x,
             masked_histories(x,"MCAR20",seed=21))
    bad,_=measured_group_concept(source,corrupted,
             masked_histories(corrupted,"MCAR20",seed=21))
    assert np.mean(np.abs(good-bad))>.08


@pytest.mark.parametrize("badindex",[-1,20])
def test_invalid_known_sensor_spec_is_rejected(badindex):
    d,m=_demo()
    with pytest.raises(ValueError):
        form_train_measured_groups(d["x"][d["train"]],m,
                     known_sensor_indices=(0,1,2,badindex))
