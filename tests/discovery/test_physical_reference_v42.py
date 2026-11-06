"""V42 physical reference acceptance tests; no hidden truth used for fitting."""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path

from conceptfan_discovery.physical_reference_world_v42 import (
    REGISTERED_DEVICES,generate_device_v42,
    render_observed_v42,masks_v42,measured_difference_v42,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,
)
from conceptfan_discovery.adaptive_pair_consensus_v41 import (
    fit_adaptive_source_v41,adaptive_pair_feature_v41,
)
from conceptfan_discovery.physical_reference_certificate_v42 import (
    fit_physical_reference_v42,predict_physical_reference_v42,
    save_physical_reference_v42,load_physical_reference_v42,
    _conformal_radius,
)


@pytest.fixture(scope="module")
def ready():
    w=generate_device_v42(
        device_seed=20261607,n_train=380,n_valid=300,n_test=230)
    def observed(indices,stress="MCAR20",fault="none",seed=111):
        x,z=render_observed_v42(w,indices,fault=fault,seed=seed)
        a,b=masks_v42(
            x,z,stress,device_seed=20261607,mask_seed=211)
        if fault=="no_reference":
            b[:,:,4:10]=False
        return measured_difference_v42(x,z,a,b)
    tr,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    xd,md=observed(tr)
    v37=fit_centered_threeview_v37(
        xd,md,independent_view_noise_documented=True,
        seed=20264742,bootstrap=24)
    src=fit_adaptive_source_v41(v37,xd,md)
    R=w["independent_reference_readout_CALIBRATION_ONLY"]
    t,m=observed(tr[:160])
    a,ma=observed(v1[:100])
    b,mb=observed(v2[:100])
    cert=fit_physical_reference_v42(
        src,t,m,R[tr[:160]],a,ma,R[v1[:100]],b,mb,R[v2[:100]],smoke=True)
    return w,src,cert,observed


def test_fresh_four_devices_and_strict_reference_budget():
    assert set(REGISTERED_DEVICES)=={
        20261607,20261609,20261621,20261623}


def test_no_Y_H_S_reference_TEST_in_source_learner_and_certificate(ready):
    w,src,cert,_=ready
    assert not src.uses_hidden_truth_H_or_S
    assert not src.uses_outcome_Y_for_source_definition
    assert cert.reference_train_budget==160
    assert cert.reference_valid1_budget==100
    assert cert.reference_valid2_budget==100
    assert cert.physical_reference_is_external_instrument_assumption
    assert cert.current_healthy_sensor_inference_not_certified
    for c in cert.pair_certs.values():
        assert c.n_train_reference<=160
        assert c.n_valid1_reference<=100
        assert c.n_valid2_reference<=100
        if c.two_VALIDs_approved:
            assert c.trained_with_external_reference
            assert c.train_spearman>=.35
            assert c.valid2_spearman>=.25
            assert c.valid2_reference_coverage>=.75
            assert c.valid2_normalized_reference_mae<=.65


def test_real_external_reference_and_no_target_energy_not_in_prediction_signature(ready):
    w,src,cert,observed=ready
    x,mask=observed(w["test"])
    a=predict_physical_reference_v42(cert,x,mask)
    assert a["external_reference_R_NOT_used_at_inference"]
    assert a["true_physical_source_S_NOT_used_at_inference"]
    assert a["uncertainty_interval_targets_noisy_R_not_true_E"]
    supported=a["physical_reference_allowed"]
    assert np.isfinite(a["physical_reference_estimated_energy"][supported]).all()
    assert np.isnan(a["physical_reference_estimated_energy"][~supported]).all()
    assert np.all(a["physical_reference_estimated_energy"][supported]>=0)
    assert np.all(a["physical_reference_interval_lower"][supported]<=
                  a["physical_reference_estimated_energy"][supported])
    assert np.all(a["physical_reference_interval_upper"][supported]>=
                  a["physical_reference_estimated_energy"][supported])


def test_proved_physical_reference_readout_has_nonzero_noise_not_oracle_truth(ready):
    w,src,cert,_=ready
    r=w["independent_reference_readout_CALIBRATION_ONLY"]
    e=w["true_slow_source_energy_POSTFIT_AUDIT_ONLY"]
    assert np.std(r-e)>.001
    assert w["reference_is_privileged_not_unsupervised"]
    assert w["simulated_reference_error_SD"]>.001


def test_missing_OFF_reference_never_returns_named_physical_source(ready):
    w,src,cert,observed=ready
    d,m=observed(w["test"],fault="no_reference")
    result=predict_physical_reference_v42(cert,d,m)
    assert not result["physical_reference_allowed"].any()
    assert not result["support"].any()
    assert np.isnan(result["physical_reference_estimated_energy"]).all()
    assert set(np.unique(result["selected_pair"]))=={"--"}


def test_missing_A_may_use_only_BC_and_single_uncorroborated(ready):
    w,src,cert,observed=ready
    d,m=observed(w["test"],stress="missing_view_A")
    result=predict_physical_reference_v42(cert,d,m)
    assert set(np.unique(result["selected_pair"])).issubset({"BC","--"})
    assert not result["two_or_three_pairs_corroborated"].any()
    assert np.all(result["single_pair_uncorroborated"][
        result["physical_reference_allowed"]])
    assert result["historical_reference_does_NOT_certify_current_faults"]


def test_isotonic_reference_model_saved_and_reopened_bitwise_probability_equivalent(
    ready,tmp_path,
):
    w,src,cert,observed=ready
    path=tmp_path/"measured_reference_v42.json"
    save_physical_reference_v42(path,cert)
    restored=load_physical_reference_v42(path)
    assert restored.audit()==cert.audit()
    d,m=observed(w["test"][:29])
    a=predict_physical_reference_v42(cert,d,m)
    b=predict_physical_reference_v42(restored,d,m)
    for field in (
        "physical_reference_allowed","selected_pair",
        "physical_reference_reason","q","support",
    ):
        np.testing.assert_array_equal(a[field],b[field])
    for field in (
        "physical_reference_estimated_energy",
        "physical_reference_interval_lower",
        "physical_reference_interval_upper",
    ):
        np.testing.assert_allclose(a[field],b[field],atol=0,rtol=0,
                                   equal_nan=True)
    assert '"outcomes"' not in path.read_text()
    assert '"hidden_source"' not in path.read_text()
    with pytest.raises(FileExistsError):
        save_physical_reference_v42(path,cert)


def test_refuse_corrupted_history_not_silently_accept(ready):
    w,src,cert,observed=ready
    tr,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    r=w["independent_reference_readout_CALIBRATION_ONLY"]
    td,tm=observed(tr[:160])
    a,am=observed(v1[:100])
    b,bm=observed(v2[:100])
    # A constant disconnected reference has no physical scale.
    with pytest.raises(ValueError,match="dynamic range"):
        fit_physical_reference_v42(
            src,td,tm,np.ones(160),a,am,r[v1[:100]],
            b,bm,r[v2[:100]],smoke=True)
    # In an unknown VALID2 instrument failure all three pairs may fail.
    shuffled=np.random.default_rng(20264787).permutation(r[v2[:100]])
    damaged=fit_physical_reference_v42(
        src,td,tm,r[tr[:160]],a,am,r[v1[:100]],
        b,bm,shuffled,smoke=True)
    assert not all(v.two_VALIDs_approved
                   for v in damaged.pair_certs.values())
    assert sum(v.two_VALIDs_approved
               for v in damaged.pair_certs.values())<=sum(
               v.two_VALIDs_approved for v in cert.pair_certs.values())


def test_conformal_style_radius_is_order_statistic_for_noisy_R():
    residuals=np.arange(1,101,dtype=np.float64)
    # ceil((100+1)*.9)=91; zero-index 90 -> radius=91
    assert _conformal_radius(residuals)==91.


def test_hidden_nan_inputs_never_expose_unobserved_true_sensor_values(ready):
    w,src,cert,_=ready
    on,off=render_observed_v42(w,w["test"],fault="none",seed=118)
    mo,mr=masks_v42(
        on,off,"MCAR40",device_seed=20261607,mask_seed=211)
    d,mask=measured_difference_v42(on,off,mo,mr)
    a=predict_physical_reference_v42(cert,d,mask)
    n_on=on.copy();n_off=off.copy()
    n_on[~mo]=np.nan;n_off[~mr]=np.nan
    d2,m2=measured_difference_v42(n_on,n_off,mo,mr)
    b=predict_physical_reference_v42(cert,d2,m2)
    np.testing.assert_array_equal(a["q"],b["q"])
    np.testing.assert_array_equal(a["physical_reference_allowed"],
                                  b["physical_reference_allowed"])
    np.testing.assert_allclose(
        a["physical_reference_estimated_energy"],
        b["physical_reference_estimated_energy"],
        rtol=0,atol=0,equal_nan=True)
    single=predict_physical_reference_v42(cert,d2[:1],m2[:1])
    assert single["physical_reference_reason"][0]==a["physical_reference_reason"][0]


def test_conditional_physical_reference_cert_does_not_solve_unknown_OFF_drift(
    ready,
):
    w,src,cert,observed=ready
    healthy,hmask=observed(w["test"],fault="none",seed=119)
    _,_=observed(w["test"],fault="off_drift",seed=119)
    on0,off0=render_observed_v42(
        w,w["test"],fault="none",seed=119)
    on1,off1=render_observed_v42(
        w,w["test"],fault="off_drift",seed=119)
    np.testing.assert_array_equal(on0,on1)
    assert np.mean(abs(off1[:,:,4:10]-off0[:,:,4:10]))>.1
    # A historically valid calibration cannot, by itself, establish
    # that TEST OFF drift is absent.
    assert cert.current_healthy_sensor_inference_not_certified



def test_external_npz_to_json_runtime_is_generator_free_and_matches_native_reference(
    ready,tmp_path,
):
    from scripts.infer_physical_reference_v42 import (
        inference_physical_reference_npz_v42,
    )
    from conceptfan_discovery.physical_reference_certificate_v42 import (
        measured_external_on_off_difference_v42,
    )
    import scripts.infer_physical_reference_v42 as cli_module
    # No import of physical_reference_world_v42 or hidden simulators in
    # the actual public inference module.
    import inspect
    assert "physical_reference_world_v42" not in inspect.getsource(cli_module)
    w,src,fit,_=ready
    model=tmp_path/"trained_safe_reference.json"
    save_physical_reference_v42(model,fit)
    on,off=render_observed_v42(
        w,w["test"][:17],fault="none",seed=119)
    mo,mr=masks_v42(
        on,off,"MCAR20",device_seed=20261607,mask_seed=211)
    d_old,mask_old=measured_difference_v42(on,off,mo,mr)
    direct=predict_physical_reference_v42(fit,d_old,mask_old)
    hidden_on=on.copy();hidden_off=off.copy()
    hidden_on[~mo]=np.nan;hidden_off[~mr]=np.nan
    d_new,mask_new=measured_external_on_off_difference_v42(
        hidden_on,hidden_off,mo,mr)
    np.testing.assert_array_equal(d_old,d_new)
    np.testing.assert_array_equal(mask_old,mask_new)
    input_npz=tmp_path/"external_17_onoff_only.npz"
    np.savez_compressed(
        input_npz,on=hidden_on,off=hidden_off,
        observed_on=mo,observed_off=mr)
    outfile=tmp_path/"physical_source_estimates.json"
    result=inference_physical_reference_npz_v42(
        model_path=model,input_npz=input_npz,output_json=outfile)
    assert result["research_only_not_clinical_diagnostic"]
    assert result["number_of_input_episodes"]==17
    assert result==json.loads(outfile.read_text())
    for i,row in enumerate(result["episodes"]):
        accepted=bool(direct["physical_reference_allowed"][i])
        assert row["historically_calibrated_source_value_available"]==accepted
        assert row["realworld_physical_identity_or_current_fault_exclusion_PROVEN"] is False
        assert row["single_pair_not_independently_confirmed_for_this_episode"]==bool(
            direct["single_pair_uncorroborated"][i])
        assert row["physical_reference_decision"]==str(
            direct["physical_reference_reason"][i])
        if accepted:
            assert abs(row["model_estimated_physical_reference_energy"]-
                direct["physical_reference_estimated_energy"][i])<1e-10
            assert row["selected_real_ON_OFF_sensor_pair"]==str(
                direct["selected_pair"][i])
        else:
            assert row["model_estimated_physical_reference_energy"] is None
    with pytest.raises(FileExistsError):
        inference_physical_reference_npz_v42(
            model_path=model,input_npz=input_npz,output_json=outfile)
    bad=tmp_path/"invalid_observed_uint8.npz"
    np.savez_compressed(
        bad,on=on,off=off,
        observed_on=mo.astype(np.uint8),observed_off=mr)
    with pytest.raises(ValueError,match="strict BOOL masks"):
        inference_physical_reference_npz_v42(
            model_path=model,input_npz=bad,output_json=tmp_path/"bad.json")
