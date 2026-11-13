"""V56 independent reference semantics, coverage, revocation and adversarial controls.

Uses a *new* seeded generator. All reference values supplied to calibration
are explicitly purchased TRAIN/VALID only; no TEST assay exists in fixture.
"""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,OP_EVEN,
)
from conceptfan_discovery.constrained_joint_operator_v55 import (
    fit_joint_operator_v55,
)
from conceptfan_discovery.independent_assay_scope_v56 import (
    fit_independent_assay_gate_v56,assess_observed_source_scope_v56,
    MIN_VALID_CORRELATION,
)
from scripts.experiment_conceptfan_assay_scope_v56 import (
    generate,DEVICE_ID,REFERENCE_ID,SEED,
    _native_selected_with_reference,
)
from scripts.experiment_conceptfan_joint_operator_v55 import (
    native_k2,native_logits,
)
from scripts.experiment_conceptfan_operator_library_v54 import masked


@pytest.fixture(scope="module")
def context():
    world=generate()
    train=world["train"]
    bank=discover_operator_library_v54(
        world["x"][train],world["masks"]["train"])
    useful=next(c for c in bank.candidates if
                c.family==OP_EVEN and {c.source_channel,c.readout_channel}=={4,5})
    artifact=next(c for c in bank.candidates if
                  c.family==OP_EVEN and {c.source_channel,c.readout_channel}=={8,9})
    xtr=world["x"][train];mtr=world["masks"]["train"]
    useful_fit=fit_joint_operator_v55(
        useful,xtr,mtr,np.zeros(len(train)),world["y"]["nonlinear_energy"][train])
    artifact_fit=fit_joint_operator_v55(
        artifact,xtr,mtr,np.zeros(len(train)),
        world["y"]["predictive_recording_artifact"][train])
    def cert(fit,permute=False):
        assay={}
        for label in ("train","valid1","valid2"):
            entry=world["purchased_assay"][label]
            vals=entry["readings"].copy()
            if permute:
                vals=np.random.default_rng(SEED+{
                    "train":401,"valid1":402,"valid2":403}[label]).permutation(vals)
            assay[label]=(entry["indices"],vals)
        tr,v1,v2=(world[name] for name in ("train","valid1","valid2"))
        return fit_independent_assay_gate_v56(
            fit,world["x"][tr],world["masks"]["train"],*assay["train"],
            world["x"][v1],world["masks"]["valid1"],*assay["valid1"],
            world["x"][v2],world["masks"]["valid2"],*assay["valid2"],
            device_id=DEVICE_ID,reference_id=REFERENCE_ID,seed=SEED+124)
    return world,useful_fit,artifact_fit,cert


def test_new_corroboration_is_not_generated_from_test_assay(context):
    world,useful,artifact,cert=context
    assert world["test_assay_available_to_inference"] is False
    assert world["generator_latent_sources_returned"] is False
    assert "test" not in world["purchased_assay"]
    c=cert(useful)
    assert c.train_assay_purchased==190
    assert c.valid1.assay_purchased==100
    assert c.valid2.assay_purchased==100
    assert c.dual_VALID_corroborated
    assert c.valid1.pearson>=MIN_VALID_CORRELATION
    assert c.valid2.pearson>=MIN_VALID_CORRELATION
    assert c.reference_id_is_not_cryptographic_authentication
    assert c.not_a_causal_or_physical_identity_proof


def test_predictive_artifact_without_independent_assay_fails(context):
    _,useful,artifact,cert=context
    c=cert(artifact)
    assert not c.dual_VALID_corroborated
    assert not(c.valid1.passed and c.valid2.passed)


def test_independent_shuffled_assay_breaks_candidate_certification(context):
    _,useful,_,cert=context
    c=cert(useful,permute=True)
    assert not c.dual_VALID_corroborated
    assert not(c.valid1.passed and c.valid2.passed)


def test_missing_or_wrong_reference_revokes_with_exact_k2(context):
    world,useful,_,cert=context
    te=world["test"]
    xt=world["x"][te]
    m=masked(xt,"MCAR20",seed=20265699)
    c=cert(useful)
    model=native_k2(11)
    original=native_logits(model,xt,m)
    for kwargs in (
        {"device_id":"wrong-device","reference_id":REFERENCE_ID},
        {"device_id":DEVICE_ID,"reference_id":"wrong-reference"},
        {"device_id":"","reference_id":REFERENCE_ID}
    ):
        z,used,q,s,scope=_native_selected_with_reference(
            model,original,useful,c,xt,m,**kwargs)
        assert not used.any()
        np.testing.assert_array_equal(z,original)
        assert scope["status"]=="REVOKED_REFERENCE_OR_DEVICE_MISMATCH"
    none=assess_observed_source_scope_v56(
        useful,None,xt,m,device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    assert not none["approved"]
    assert not none["source_used"].any()


def test_novel_unlabeled_gain_shift_and_forced_missing(context):
    world,useful,_,cert=context
    te=world["test"]
    x=world["x"][te].copy()
    m=masked(x,"MCAR20",seed=20265699)
    c=cert(useful)
    healthy=assess_observed_source_scope_v56(
        useful,c,x,m,device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    assert healthy["approved"]
    assert healthy["source_used"].sum()>=100
    flipped=x.copy()
    for j in (useful.candidate.source_channel,useful.candidate.readout_channel):
        flipped[:,:,j]*=-1.
    inverted=assess_observed_source_scope_v56(
        useful,c,flipped,m,device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    np.testing.assert_array_equal(inverted["source_q"],healthy["source_q"])
    large=x.copy()
    for j in (useful.candidate.source_channel,useful.candidate.readout_channel):
        large[:,:,j]*=3.
    shifted=assess_observed_source_scope_v56(
        useful,c,large,m,device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    assert shifted["status"]=="REVOKED_UNLABELED_SOURCE_COHORT_SHIFT"
    assert not shifted["source_used"].any()
    absent=m.copy()
    for j in (useful.candidate.source_channel,useful.candidate.readout_channel):
        absent[:,:,j]=False
    refused=assess_observed_source_scope_v56(
        useful,c,x,absent,device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    assert refused["status"]=="ABSTAIN_INSUFFICIENT_COHORT_SOURCE_SUPPORT"
    assert not refused["source_used"].any()


def test_purchased_assay_contract_cannot_accept_invalid_ids(context):
    world,useful,_,cert=context
    tr,v1,v2=(world[k] for k in ("train","valid1","valid2"))
    m=world["masks"]
    a=world["purchased_assay"]
    ids=a["train"]["indices"].copy()
    ids[0]=ids[1]
    with pytest.raises(ValueError,match="purchased"):
        fit_independent_assay_gate_v56(
            useful,world["x"][tr],m["train"],
            ids,a["train"]["readings"],
            world["x"][v1],m["valid1"],
            a["valid1"]["indices"],a["valid1"]["readings"],
            world["x"][v2],m["valid2"],
            a["valid2"]["indices"],a["valid2"]["readings"],
            device_id=DEVICE_ID,reference_id=REFERENCE_ID)
