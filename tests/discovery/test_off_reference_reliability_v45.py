import numpy as np
import pytest

from conceptfan_discovery.off_reference_reliability_v45 import (
    measured_off_dynamic_energy, fit_off_guard_v45, apply_off_guard_v45,
    gate_partial_view_source_v45,
)


def _synthetic(n=720, seed=45):
    rng = np.random.default_rng(seed)
    off = rng.normal(0, .15, (n, 16, 12))
    u = rng.normal(size=(n, 16)) * .22
    off[:, :, 4:10] += u[:, :, None]
    m = np.ones(off.shape, dtype=bool)
    return off, m


def _fit():
    x, m = _synthetic()
    return fit_off_guard_v45(x[:360],m[:360],x[360:540],m[360:540],
                             x[540:],m[540:]), x, m


def test_off_guard_is_train_only_with_two_disjoint_healthy_valids():
    fit, x, m = _fit()
    assert fit.train_supported == 360
    assert fit.verified_on_two_independent_healthy_VALID
    assert 0 <= fit.valid1_false_alarm <= .05
    assert 0 <= fit.valid2_false_alarm <= .05
    assert fit.physical_identity_certified is False
    assert fit.no_Y_R_H_S_U_used


def test_drift_detected_and_exact_neutral_source():
    fit, x, m = _fit()
    good=apply_off_guard_v45(fit,x[360:540],m[360:540])
    assert good["usable_for_additional_concept"].mean() >= .95
    broken=x[360:540].copy()
    broken[:,:,4:10]+=3*np.linspace(-1,1,16)[None,:,None]
    bad=apply_off_guard_v45(fit,broken,m[360:540])
    assert bad["off_background_alarm"].all()
    source={"q":np.full(len(bad["score"]),.9),"support":np.ones(len(bad["score"]))}
    safe=gate_partial_view_source_v45(source,bad)
    assert (safe["q"]==.5).all()
    assert (safe["support"]==0).all()
    assert safe["off_monitor_rejected"].all()


def test_missing_off_refusal_and_invariance_to_hidden_nan():
    fit,x,m=_fit()
    mask=m[360:540].copy()
    mask[:,:,4:10]=False
    poison=x[360:540].copy()
    poison[~mask]=np.nan
    out=apply_off_guard_v45(fit,poison,mask)
    assert not out["usable_for_additional_concept"].any()
    assert set(out["decision_reason"])=={"NO_CURRENT_OFF_REFERENCE"}
    with pytest.raises(ValueError):
        apply_off_guard_v45(fit,poison,~mask)


def test_constant_common_shift_not_a_general_identity_check():
    fit,x,m=_fit()
    before=measured_off_dynamic_energy(x[360:540],m[360:540])
    changed=x[360:540].copy()
    changed[:,:,4:10]+=7.
    after=measured_off_dynamic_energy(changed,m[360:540])
    np.testing.assert_allclose(before["score"],after["score"],rtol=0,atol=1e-13)
    assert fit.physical_identity_certified is False
