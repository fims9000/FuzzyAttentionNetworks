"""V57 source-value / native-risk compatibility, independent reference and fallback."""
import numpy as np
import pytest

from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,OP_EVEN,
)
from conceptfan_discovery.constrained_joint_operator_v55 import (
    fit_joint_operator_v55,infer_joint_operator_v55,
)
from conceptfan_discovery.independent_assay_scope_v56 import (
    fit_independent_assay_gate_v56,
)
from conceptfan_discovery.assay_calibrated_concept_v57 import (
    fit_calibrated_concept_and_native_head_v57,
    map_to_external_measurement_v57,
    predict_calibrated_source_scope_v57,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from scripts.experiment_conceptfan_assay_scope_v56 import (
    generate,DEVICE_ID,REFERENCE_ID,
)
from scripts.experiment_conceptfan_assay_value_v57 import (
    DATA_SEED,_native_calibrated,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    masked,native_k2,native_logits,
)


@pytest.fixture(scope="module")
def reference_model():
    w=generate(seed=DATA_SEED)
    x=w["x"]
    tr,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    tm,m1,m2=(w["masks"][k] for k in ("train","valid1","valid2"))
    c=next(k for k in discover_operator_library_v54(
        x[tr],tm).candidates if k.family==OP_EVEN and
        {k.source_channel,k.readout_channel}=={4,5})
    yy=w["y"]["nonlinear_energy"]
    fit=fit_joint_operator_v55(c,x[tr],tm,np.zeros(len(tr)),yy[tr])
    purchased=w["purchased_assay"]
    cert=fit_independent_assay_gate_v56(
        fit,x[tr],tm,purchased["train"]["indices"],
        purchased["train"]["readings"],
        x[v1],m1,purchased["valid1"]["indices"],
        purchased["valid1"]["readings"],
        x[v2],m2,purchased["valid2"]["indices"],
        purchased["valid2"]["readings"],
        device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    assert cert.dual_VALID_corroborated
    contract=fit_calibrated_concept_and_native_head_v57(
        fit,cert,purchased["train"]["readings"],
        x[tr],tm,yy[tr],np.zeros(len(tr)),
        x[v1],m1,yy[v1],np.zeros(len(v1)),
        x[v2],m2,yy[v2],np.zeros(len(v2)),
        device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    return w,fit,cert,contract


def test_external_energy_calibration_changes_numeric_concept_and_refits_gamma(reference_model):
    w,fit,cert,contract=reference_model
    tr=w["train"]
    q=map_to_external_measurement_v57(
        fit,contract.source_contract,w["x"][tr],w["masks"]["train"])
    uncal=infer_joint_operator_v55(fit,w["x"][tr],w["masks"]["train"])
    assert np.mean(np.abs(q["q_assay"]-uncal["summary"]))>.01
    assert contract.source_contract.train_assay_count==190
    assert contract.source_contract.train_reference_iqr>=.1
    assert contract.uses_same_q_at_train_and_inference
    assert contract.source_gamma_does_not_reuse_v55
    expected=train_source_coefficient(
        np.zeros(len(tr)),q["q_assay"],w["y"]["nonlinear_energy"][tr])
    assert contract.risk_gamma_train_only==pytest.approx(expected,abs=1e-9)
    assert np.isfinite(contract.risk_gamma_train_only)
    assert not contract.source_physical_identity_not_proven is False


def test_wrong_device_or_reference_cannot_use_the_calibrated_concept(reference_model):
    w,fit,cert,contract=reference_model
    te=w["test"]
    x=w["x"][te]
    m=masked(x,"MCAR20",seed=134)
    wrong=predict_calibrated_source_scope_v57(
        fit,cert,contract,x,m,device_id="other",reference_id=REFERENCE_ID)
    assert wrong["status"]=="REVOKED_CONTRACT_DEVICE_REFERENCE_MISMATCH"
    assert not wrong["source_used"].any()
    assert not wrong["external_test_assay_used"]
    missing=predict_calibrated_source_scope_v57(
        fit,None,contract,x,m,device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    assert missing["status"]=="ABSTAIN_NO_ASSAY_RISK_CONTRACT"
    assert not missing["source_used"].any()


def test_wrong_contract_rejected_during_calibration(reference_model):
    w,fit,cert,contract=reference_model
    tr,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    x=w["x"];m=w["masks"];y=w["y"]["nonlinear_energy"]
    assays=w["purchased_assay"]["train"]["readings"]
    with pytest.raises(ValueError,match="Independent"):
        fit_calibrated_concept_and_native_head_v57(
            fit,cert,assays,x[tr],m["train"],y[tr],np.zeros(len(tr)),
            x[v1],m["valid1"],y[v1],np.zeros(len(v1)),
            x[v2],m["valid2"],y[v2],np.zeros(len(v2)),
            device_id=DEVICE_ID,reference_id="other")


def test_missing_signal_neutral_and_native_exact_K2(reference_model):
    w,fit,cert,contract=reference_model
    te=w["test"]
    x=w["x"][te]
    m=masked(x,"MCAR20",seed=137)
    removed=m.copy()
    removed[:,:,fit.candidate.source_channel]=False
    removed[:,:,fit.candidate.readout_channel]=False
    q=map_to_external_measurement_v57(
        fit,contract.source_contract,x,removed)
    np.testing.assert_array_equal(q["q_assay"],np.full(len(te),.5))
    assert np.isnan(q["energy_estimate"]).all()
    model=native_k2(11)
    base=native_logits(model,x,removed)
    z,used,_,_,reason=_native_calibrated(
        model,base,fit,cert,contract,x,removed)
    assert not used.any()
    np.testing.assert_array_equal(z,base)
    assert reason["status"] in (
        "ABSTAIN_INSUFFICIENT_COHORT_SOURCE_SUPPORT",
        "ABSTAIN_RECALIBRATED_RISK_VALID_FAILED")


def test_masked_nan_payload_ignored_and_sign_invariance(reference_model):
    w,fit,cert,contract=reference_model
    tr=w["train"][:50]
    x=w["x"][tr]
    m=w["masks"]["train"][:50]
    before=map_to_external_measurement_v57(
        fit,contract.source_contract,x,m)
    dirty=x.copy();dirty[~m]=np.nan
    after=map_to_external_measurement_v57(
        fit,contract.source_contract,dirty,m)
    np.testing.assert_array_equal(before["q_assay"],after["q_assay"])
    neg=x.copy()
    neg[:,:,fit.candidate.source_channel]*=-1
    neg[:,:,fit.candidate.readout_channel]*=-1
    result=map_to_external_measurement_v57(
        fit,contract.source_contract,neg,m)
    np.testing.assert_array_equal(before["q_assay"],result["q_assay"])
