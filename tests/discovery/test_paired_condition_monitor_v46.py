"""V46 sensor-pair condition guard smoke and exact-masking falsifiers."""
import numpy as np
from conceptfan_discovery.paired_condition_monitor_v46 import (
    measured_paired_mismatch_v46,fit_paired_condition_v46,
    apply_paired_condition_v46,
)


def _healthy(n=720,seed=46):
    rng=np.random.default_rng(seed)
    base=rng.normal(0,.15,(n,16,12))
    u=rng.normal(0,.20,(n,16,1))
    off=base.copy()
    on=base.copy()
    off[:,:,4:10]=u+rng.normal(0,.15,(n,16,6))
    source=rng.normal(0,.65,(n,16,6))
    on[:,:,4:10]=off[:,:,4:10]+source
    m=np.ones(on.shape,dtype=bool)
    return on,off,m


def _fit():
    on,off,m=_healthy()
    fit=fit_paired_condition_v46(
        on[:360],off[:360],m[:360],m[:360],
        on[360:540],off[360:540],m[360:540],m[360:540],
        on[540:],off[540:],m[540:],m[540:],
    )
    return fit,on,off,m


def test_healthy_fit_cannot_claim_physical_identity():
    fit,on,off,m=_fit()
    assert fit.approved_two_healthy_VALID
    assert fit.n_train_sufficient==360
    assert fit.uses_outcomes_Y is False
    assert fit.physical_concept_identity_certified is False
    probe=apply_paired_condition_v46(fit,on[360:540],off[360:540],m[360:540],m[360:540])
    assert probe["usable_for_additional_concept"].mean()>.95


def test_safe_matched_background_vs_true_off_only_drift():
    fit,on,off,m=_fit()
    baseline_on=on[360:540].copy()
    baseline_off=off[360:540].copy()
    nuisance=4.*np.linspace(-1.,1.,16)[None,:,None]
    common_on=baseline_on.copy()
    common_off=baseline_off.copy()
    common_on[:,:,4:10]+=nuisance
    common_off[:,:,4:10]+=nuisance
    safe=apply_paired_condition_v46(fit,common_on,common_off,m[360:540],m[360:540])
    assert safe["usable_for_additional_concept"].mean()>.90
    bad_off=baseline_off.copy()
    bad_off[:,:,4:10]+=nuisance
    unsafe=apply_paired_condition_v46(fit,baseline_on,bad_off,m[360:540],m[360:540])
    assert unsafe["joint_condition_alarm"].mean()>.90
    assert unsafe["usable_for_additional_concept"].mean()<.10


def test_missing_and_hidden_sensor_values_never_become_evidence():
    fit,on,off,m=_fit()
    use=m[360:540].copy()
    use[:,:,4:10]=False
    poison_on=on[360:540].copy()
    poison_off=off[360:540].copy()
    poison_off[~use]=np.nan
    unsafe=apply_paired_condition_v46(fit,poison_on,poison_off,use,use)
    assert not unsafe["usable_for_additional_concept"].any()
    assert set(unsafe["decision_reason"])=={"MISSING_REAL_ON_OFF_PAIR"}


def test_missing_one_view_keeps_other_four_channels_a_valid_measurement():
    on,off,m=_healthy(400)
    mon=m.copy()
    mon[:,:,4:6]=False
    result=measured_paired_mismatch_v46(on,off,mon,m)
    assert result["support"].all()
    assert (result["n_paired_channels"]==4).all()
    noise=on.copy()
    noise[~mon]=np.nan
    same=measured_paired_mismatch_v46(noise,off,mon,m)
    np.testing.assert_array_equal(result["score"],same["score"])
