"""V36 falsifiers: no privileged H, tied modes, missing views, unsigned invariance."""
import numpy as np
import pytest
from conceptfan_discovery.gauge_source_v36 import (
    fit_blind_gauge_source_v36,source_energy_v36,
)
from conceptfan_discovery.crossview_world_v36 import (
    generate_world_v36,mask_views_v36,corrupt_views_v36,
)


def _fit():
    d=generate_world_v36(seed=20261206,n_train=1000,n_valid=400,n_test=240)
    x=d["x"][d["train"]]
    m=mask_views_v36(x,"MCAR20",seed=17)
    fit=fit_blind_gauge_source_v36(
        x,m,independent_view_errors_documented=True,bootstrap=24)
    return d,fit


def test_two_views_recover_distinct_spectral_axis_without_synthetic_H():
    d,f=_fit()
    assert f.lag_ratios[0]>.4 and f.lag_ratios[1]<.2
    assert f.lag_gap>.25 and f.boot_abs_cos_q10>.8
    x=d["x"][d["test"]]
    m=mask_views_v36(x,"MCAR20",seed=55)
    r=source_energy_v36(f,x,m)
    good=r["support"]>0
    truth=d["true_slow_energy_AUDIT_ONLY"][d["test"]]
    assert good.mean()>.60
    assert np.corrcoef(truth[good],r["estimated_crossview_energy"][good])[0,1]>.6
    assert not f.physical_semantics_certified
    assert not f.receipt()["signed_source_identity_certified"]
    assert not f.receipt()["original_mixing_matrix_supplied"]


def test_simultaneous_sign_reversal_exact_invariance_not_physical_sign():
    d,f=_fit()
    x=d["x"][d["test"]]
    m=mask_views_v36(x,"MCAR20",seed=31)
    a=source_energy_v36(f,x,m)
    b=source_energy_v36(f,corrupt_views_v36(x,"both_flip"),m)
    np.testing.assert_array_equal(a["q"],b["q"])
    np.testing.assert_array_equal(a["support"],b["support"])
    assert not a["physically_named_signed_source_is_identified"]


def test_missing_either_view_abstains_and_never_reads_nan_payload():
    d,f=_fit()
    x=d["x"][d["test"]]
    for mode in ("missing_view_A","missing_view_B","no_views"):
        m=mask_views_v36(x,mode,seed=9)
        r=source_energy_v36(f,x,m)
        assert not r["support"].any()
        np.testing.assert_array_equal(r["q"],np.full(len(x),.5))
    m=mask_views_v36(x,"MCAR20",seed=9)
    masked=x.copy()
    masked[~m]=np.nan
    a=source_energy_v36(f,x,m)
    b=source_energy_v36(f,masked,m)
    np.testing.assert_array_equal(a["q"],b["q"])


def test_no_independent_instrument_provenance_means_refusal():
    d=generate_world_v36(n_train=1000,n_valid=400,n_test=240)
    x=d["x"][d["train"]]
    m=mask_views_v36(x,"MCAR20",seed=17)
    with pytest.raises(ValueError,match="independence"):
        fit_blind_gauge_source_v36(
            x,m,independent_view_errors_documented=False)


def test_equal_temporal_signatures_are_not_separate_concepts():
    d=generate_world_v36(
        n_train=1000,n_valid=400,n_test=240,matched_rho=True)
    x=d["x"][d["train"]]
    m=mask_views_v36(x,"MCAR20",seed=17)
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_blind_gauge_source_v36(
            x,m,independent_view_errors_documented=True,bootstrap=24)


def test_flip_one_view_causes_widespread_rejection_not_certified_detection():
    d,f=_fit()
    x=d["x"][d["test"]]
    m=mask_views_v36(x,"MCAR20",seed=9)
    before=source_energy_v36(f,x,m)
    flipped=source_energy_v36(f,corrupt_views_v36(x,"A_flip"),m)
    assert flipped["support"].mean()<.3*before["support"].mean()


def test_invariance_does_not_extend_to_one_view_flip():
    d,f=_fit()
    x=d["x"][d["test"]]
    m=mask_views_v36(x,"complete",seed=9)
    a=source_energy_v36(f,x,m)
    b=source_energy_v36(f,corrupt_views_v36(x,"A_flip"),m)
    assert np.mean(abs(a["estimated_crossview_energy"]-
                       b["estimated_crossview_energy"]))>.3
