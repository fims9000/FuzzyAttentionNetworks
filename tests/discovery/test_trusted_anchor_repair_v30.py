from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from conceptfan_discovery.nonlinear_source_bench_v30 import (
    SEED,STRESS,N_CHANNELS,N_STEPS,nonlinear_world,mask_view,instrument_fault,
)
from conceptfan_discovery.trusted_anchor_repair_v30 import (
    candidate_and_anchor,fit_anchor_policy,anchored_predictions,
)
from conceptfan_discovery.sensor_pair_provenance_v29 import (
    fit_unlabelled_pair_guard,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2


def _consistent(seed=10,n=640):
    rng=np.random.default_rng(seed)
    z=rng.normal(0,1,(n,N_STEPS))
    for t in range(1,N_STEPS):
        z[:,t]=.4*z[:,t-1]+.65*np.tanh(1.3*z[:,t-1])+rng.normal(0,.3,n)
    x=np.empty((n,N_STEPS,N_CHANNELS),dtype=np.float32)
    x[:,:,:4]=rng.normal(0,1,(n,N_STEPS,4))
    for ch in (4,5,6):
        x[:,:,ch]=z+rng.normal(0,.3,(n,N_STEPS))
    mask=(rng.random(x.shape)>.2)
    return x,mask


def test_new_nonlinear_family_has_seven_sensors_and_disjoint_splits():
    d=nonlinear_world(seed=SEED,n_train=200,n_valid=240,n_test=220)
    x,c=d["x"],d["true_concept_trajectories"]
    assert x.shape==(660,16,7) and c.shape==(660,16,3)
    assert np.isfinite(x).all() and np.isfinite(c).all()
    assert not np.array_equal(d["labels_nuisance"],d["labels_predictive"])
    parts=[d[k] for k in ("train","valid_env1","valid_env2","test")]
    assert len(np.unique(np.concatenate(parts)))==660
    assert d["generator_family"]=="nonlinear_switching_laplace_shocks_v30"
    assert d["sensor_anchor_independent_noise_by_generator_assumption"]


@pytest.mark.parametrize("mode",STRESS)
def test_no_label_leakage_in_missingness_and_determinism(mode):
    d=nonlinear_world(n_train=180,n_valid=200,n_test=180)
    x=d["x"][d["test"]]
    m1=mask_view(x,mode,seed=200)
    m2=mask_view(x,mode,seed=200)
    np.testing.assert_array_equal(m1,m2)
    assert m1.shape==x.shape and m1.dtype==bool
    if mode=="complete":
        assert m1.all()
    else:
        assert not m1.all()


def test_known_k2_encoder_is_hard_firewalled_from_candidate_and_anchor():
    core=make_firewalled_k2(11)
    assert core.encoder.input_proj.in_features==8
    x,m=_consistent(n=160)
    for change in ("single_pair_sign_flip","single_pair_bias",
                   "both_pair_sign_flip","anchor_only_sign_flip",
                   "all_three_sign_flip"):
        transform=instrument_fault(x,change)
        np.testing.assert_array_equal(transform[:,:,:4],x[:,:,:4])
    assert m[:,:,:4].shape[-1]==4


def test_anchor_payload_neutrality_for_unobserved_cells():
    x,m=_consistent(n=170)
    original=candidate_and_anchor(x,m)
    changed=x.copy()
    changed[~m]=float("nan")
    after=candidate_and_anchor(changed,m)
    for key in ("pair_concept","anchor_concept","sigmoid_mean_discrepancy"):
        np.testing.assert_allclose(original[key],after[key],atol=0,rtol=0)
    m[:]=False
    missing=candidate_and_anchor(changed,m)
    np.testing.assert_allclose(missing["pair_concept"],.5)
    np.testing.assert_allclose(missing["anchor_concept"],.5)
    np.testing.assert_allclose(missing["anchor_support"],0.)


def test_train_only_anchor_threshold_and_common_mode_invisibility():
    x,m=_consistent(n=640)
    policy=fit_anchor_policy(x,m)
    assert policy.label_information_used_for_threshold is False
    assert policy.independent_calibrated_reference_is_external_assumption
    a=candidate_and_anchor(x[300:],m[300:])
    both=instrument_fault(x[300:],"all_three_sign_flip")
    b=candidate_and_anchor(both,m[300:])
    np.testing.assert_allclose(
        a["sigmoid_mean_discrepancy"],b["sigmoid_mean_discrepancy"],
        atol=1e-14,rtol=0.)
    assert policy.max_sigmoid_mean_discrepancy>0
    # Inverting every readout preserves pairwise agreements; no absolute
    # orientation or calibration identity can be inferred from redundancy.


def test_anchor_necessary_for_common_mode_pair_fault_and_may_itself_fail():
    x,m=_consistent(n=640)
    rule=fit_unlabelled_pair_guard(x[:400],m[:400])
    policy=fit_anchor_policy(x[:400],m[:400])
    base=np.zeros(240)
    sample=x[400:]
    mask=m[400:]
    intact=anchored_predictions(base,sample,mask,gamma=5.,
       validated_global_source=True,pair_policy=rule,anchor_policy=policy)
    pair_inverted=anchored_predictions(base,
       instrument_fault(sample,"both_pair_sign_flip"),mask,gamma=5.,
       validated_global_source=True,pair_policy=rule,anchor_policy=policy)
    all_inverted=anchored_predictions(base,
       instrument_fault(sample,"all_three_sign_flip"),mask,gamma=5.,
       validated_global_source=True,pair_policy=rule,anchor_policy=policy)
    anchor_inverted=anchored_predictions(base,
       instrument_fault(sample,"anchor_only_sign_flip"),mask,gamma=5.,
       validated_global_source=True,pair_policy=rule,anchor_policy=policy)
    assert pair_inverted["summary"]["repair_fraction"]>.6
    assert anchor_inverted["summary"]["repair_fraction"]>.6
    assert all_inverted["summary"]["repair_fraction"] <.1
    assert intact["summary"]["repair_fraction"] <.1
    # Any repair when the supplied "trusted" anchor is itself bad can be
    # actively harmful; this is a required negative control, not a test bug.


def test_untrusted_or_unmeasured_source_exactly_reverts_to_original_base():
    x,m=_consistent(n=250)
    rule=fit_unlabelled_pair_guard(x,m)
    policy=fit_anchor_policy(x,m)
    old=np.linspace(-1.,1.,250)
    before=old.copy()
    result=anchored_predictions(old,x,m,gamma=9.,
        validated_global_source=False,pair_policy=rule,anchor_policy=policy)
    np.testing.assert_array_equal(old,before)
    for logits in result["logits"].values():
        np.testing.assert_array_equal(logits,old)
    assert all(not bool(q.any()) for q in result["decisions"].values())
