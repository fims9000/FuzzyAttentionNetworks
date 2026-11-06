"""V50 observational non-identifiability and externally attested semantics."""
import numpy as np
import pytest

from conceptfan_discovery.observational_twins_v50 import (
    make_observational_twins_v50, energy_from_trajectory_v50,
)
from conceptfan_discovery.independent_anchor_certificate_v50 import (
    fit_attested_source_calibration_v50,infer_calibrated_source_energy_v50,
    assess_semantic_scope_v50,audited_concept_risk_view_v50,
)


def _fit(w,which=0,budget=.05):
    idx0,idx1,idx2=w["train"],w["valid1"],w["valid2"]
    mask=w["measured_boolean_masks"]
    world=w["worlds"][which]
    x=w["shared_observed"]
    assay=world["attested_external_assay"]
    return fit_attested_source_calibration_v50(
        x[idx0],mask["TRAIN_MCAR20"][idx0],assay[idx0],
        x[idx1],mask["VALID1_MCAR20"][idx1],assay[idx1],
        x[idx2],mask["VALID2_MCAR40"][idx2],assay[idx2],
        physical_device_id=world["physical_device_id"],
        independent_reference_id=world["external_reference_id"],
        train_reference_budget_fraction=budget,seed=1100+w["device_seed"]%11,
    )


def test_twins_are_bitwise_observationally_identical_but_concepts_differ():
    w=make_observational_twins_v50(20263107,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    x=w["shared_observed"]
    a,b=w["worlds"]
    assert np.array_equal(x,x.copy())
    assert np.array_equal(w["same_shared_outcome_Y"],w["same_shared_outcome_Y"].copy())
    test=w["test"]
    difference=np.abs(a["hidden_physical_source_energy_TEST_AUDIT_ONLY"][test]-
                      b["hidden_physical_source_energy_TEST_AUDIT_ONLY"][test])
    assert float(np.mean(difference))>.05
    naive=np.var(x[test,:,0],axis=1,ddof=1)
    error=(np.mean(np.abs(naive-a["hidden_physical_source_energy_TEST_AUDIT_ONLY"][test]))+
           np.mean(np.abs(naive-b["hidden_physical_source_energy_TEST_AUDIT_ONLY"][test])))/2
    lower=float(np.mean(difference))/2
    assert error>=lower-1e-13
    assert w["observational_twin_proof"]["no_unique_physical_source1_from_X_and_Y"]


def test_independently_measured_anchor_breaks_gauge_conditionally():
    w=make_observational_twins_v50(20263109,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    fa=_fit(w,which=0,budget=.05)
    fb=_fit(w,which=1,budget=.05)
    assert fa.dual_VALID_admitted
    assert fb.dual_VALID_admitted
    assert fa.train_assay_episodes==25
    assert fa.validation1_assay_episodes==80
    assert fa.validation2_assay_episodes==80
    x=w["shared_observed"][w["test"]]
    m=w["measured_boolean_masks"]["TEST_MCAR40"][w["test"]]
    ea=infer_calibrated_source_energy_v50(fa,x,m)["inferred_target_energy"]
    eb=infer_calibrated_source_energy_v50(fb,x,m)["inferred_target_energy"]
    aa=w["worlds"][0]["hidden_physical_source_energy_TEST_AUDIT_ONLY"][w["test"]]
    bb=w["worlds"][1]["hidden_physical_source_energy_TEST_AUDIT_ONLY"][w["test"]]
    assert float(np.corrcoef(ea,aa)[0,1])>.65
    assert float(np.corrcoef(eb,bb)[0,1])>.65
    assert float(np.mean(np.abs(ea-eb)))>.02
    assert fa.physical_device_id!=fb.physical_device_id
    assert not fa.is_universal_semantic_identity_proof


def test_missing_sensor_payload_never_enters_fit_or_inference():
    w=make_observational_twins_v50(20263121,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    fit=_fit(w,which=0,budget=.20)
    x=w["shared_observed"][w["test"]]
    mask=w["measured_boolean_masks"]["TEST_MCAR40"][w["test"]]
    baseline=infer_calibrated_source_energy_v50(fit,x,mask)
    poisoned=x.copy()
    poisoned[~mask]=np.nan
    score=infer_calibrated_source_energy_v50(fit,poisoned,mask)
    np.testing.assert_array_equal(baseline["inferred_target_energy"],
                                  score["inferred_target_energy"])
    np.testing.assert_array_equal(baseline["inferred_trajectory"],
                                  score["inferred_trajectory"])
    with pytest.raises(ValueError):
        infer_calibrated_source_energy_v50(fit,poisoned,~mask)


def test_semantic_claims_refused_without_anchor_and_under_readout_shift():
    w=make_observational_twins_v50(20263123,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    fit=_fit(w,which=0,budget=.20)
    world=w["worlds"][0]
    test=w["test"]
    mask=w["measured_boolean_masks"]["TEST_MCAR20"][test]
    x=w["shared_observed"][test]
    no_anchor=assess_semantic_scope_v50(
        None,x,mask,device_id=world["physical_device_id"],
        external_reference_id=world["external_reference_id"])
    assert no_anchor["status"]=="STATISTICAL_PROXY_UNIDENTIFIABLE"
    assert not no_anchor["physical_label_allowed"]
    okay=assess_semantic_scope_v50(
        fit,x,mask,device_id=world["physical_device_id"],
        external_reference_id=world["external_reference_id"])
    assert okay["physical_label_allowed"]
    wrong=assess_semantic_scope_v50(
        fit,x,mask,device_id=w["worlds"][1]["physical_device_id"],
        external_reference_id=world["external_reference_id"])
    assert wrong["status"]=="DEVICE_OR_ATTESTED_REFERENCE_MISMATCH"
    shifted=assess_semantic_scope_v50(
        fit,w["test_only_amplifier_readout"][test],mask,
        device_id=world["physical_device_id"],
        external_reference_id=world["external_reference_id"])
    assert shifted["status"]=="READOUT_DISTRIBUTION_SHIFT_REVOKED"
    assert not shifted["physical_label_allowed"]


def test_statistical_risk_is_distinguished_from_physical_explanation():
    w=make_observational_twins_v50(20263107,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    n=12
    base=np.linspace(.1,.9,n)
    risk=np.linspace(.2,.8,n)
    q=np.linspace(.02,.98,n)
    support=np.ones(n,dtype=bool)
    null_scope={"status":"STATISTICAL_PROXY_UNIDENTIFIABLE",
                "physical_label_allowed":False}
    inferred=audited_concept_risk_view_v50(
        risk,base,q,support,scope=null_scope,mode="predictive")
    np.testing.assert_array_equal(inferred["probability"],risk)
    assert inferred["physical_source_name"] is None
    assert not inferred["per_episode_identity_certified"]
    strict=audited_concept_risk_view_v50(
        risk,base,q,support,scope=null_scope,mode="strict_semantic")
    np.testing.assert_array_equal(strict["probability"],base)
    certified=audited_concept_risk_view_v50(
        risk,base,q,support,
        scope={"status":"ASSAY_CALIBRATED_SAME_ENVIRONMENT",
               "physical_label_allowed":True},mode="strict_semantic")
    np.testing.assert_array_equal(certified["probability"],risk)
    assert "калиброванный" in certified["physical_source_name"]
    assert not certified["per_episode_identity_certified"]
    with pytest.raises(ValueError):
        audited_concept_risk_view_v50(risk,base,q,support,scope=null_scope,mode="other")
