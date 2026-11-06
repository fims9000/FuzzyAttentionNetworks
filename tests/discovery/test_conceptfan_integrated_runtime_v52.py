"""Integration tests: native concept-only risk, attested source, fallback and review."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from conceptfan_discovery.observational_twins_v51 import make_observational_twins_v51
from conceptfan_discovery.independent_anchor_certificate_v50 import fit_attested_source_calibration_v50
from conceptfan_discovery.train_only_energy_debias_v51 import fit_train_only_energy_debias_v51
from conceptfan_discovery.anchored_joint_bottleneck_v47 import AnchoredJointConceptBottleneckV47
from conceptfan_discovery.conceptfan_integrated_runtime_v52 import (
    ConceptRiskTrainingContractV52,fit_attested_energy_membership_map_v52,
    prepare_integrated_concept_v52,predict_integrated_conceptfan_v52,
    apply_integrated_expert_review_v52,
)


class TinyNativeConceptEncoder(nn.Module):
    """A real gradient-carrying torch 2-concept encoder, no risk bypass."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(821)
        self.linear=nn.Linear(8,2)

    def forward(self,features):
        trajectory=torch.sigmoid(self.linear(features))
        return SimpleNamespace(
            concept_trajectories=trajectory,
            concept_summaries=trajectory.mean(dim=1),
        )


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(2)
    w=make_observational_twins_v51(
        20263307,n_train=500,n_valid1=200,n_valid2=200,n_test=240)
    data=w["shared_observed"]; masks=w["measured_boolean_masks"]
    t,v1,v2=(w[k] for k in ("train","valid1","valid2"))
    gauge=w["worlds"][0]
    assay=gauge["attested_external_assay"]
    seed=20265041+w["device_seed"]%101
    calibrated=fit_attested_source_calibration_v50(
        data[t],masks["TRAIN_MCAR20"][t],assay[t],
        data[v1],masks["VALID1_MCAR20"][v1],assay[v1],
        data[v2],masks["VALID2_MCAR40"][v2],assay[v2],
        physical_device_id=gauge["physical_device_id"],
        independent_reference_id=gauge["external_reference_id"],
        train_reference_budget_fraction=.05,seed=seed)
    debias=fit_train_only_energy_debias_v51(
        calibrated,
        data[t],masks["TRAIN_MCAR20"][t],assay[t],
        data[v1],masks["VALID1_MCAR20"][v1],assay[v1],
        data[v2],masks["VALID2_MCAR40"][v2],assay[v2],
        train_reference_budget_fraction=.05,seed=seed)
    selected=np.sort(np.random.default_rng(seed+1).choice(len(t),25,replace=False))
    mapping=fit_attested_energy_membership_map_v52(
        assay[t][selected],physical_device_id=gauge["physical_device_id"],
        external_reference_id=gauge["external_reference_id"])
    contract=ConceptRiskTrainingContractV52(
        physical_device_id=gauge["physical_device_id"],
        external_reference_id=gauge["external_reference_id"],
        source_training_kind="attested_energy_v51",
        energy_correction_method="V51_SCALE",
        source_membership_median=mapping.source_reference_energy_median,
        source_membership_scale=mapping.source_reference_energy_IQR_scale)
    model=AnchoredJointConceptBottleneckV47(
        TinyNativeConceptEncoder(),[1.4,-.9],-.17,initial_gamma=1.5)
    return w,calibrated,debias,mapping,contract,model


def _predict(setup,*,shift=False,mask_override=None):
    w,c,d,mapping,contract,model=setup
    x=(w["test_only_amplifier_readout"][w["test"]]
       if shift else w["shared_observed"][w["test"]])
    mask=(w["measured_boolean_masks"]["TEST_MCAR20"][w["test"]].copy()
          if mask_override is None else mask_override)
    original_logits=np.linspace(-.9,.9,len(x))
    return predict_integrated_conceptfan_v52(
        model,x,mask,original_logits,
        source_calibration=c,energy_calibration=d,
        membership_map=mapping,training_contract=contract,
        physical_device_id=contract.physical_device_id,
        external_reference_id=contract.external_reference_id)


def test_integrates_native_torch_signed_concept_risk_with_real_v51_source(setup):
    result=_predict(setup)
    assert result["semantic_scope"]["physical_label_allowed"]
    assert result["source_used"].mean()>.9
    assert result["risk_only_uses_signed_explicit_concepts"]
    assert result["model_source_training_contract_checked"]
    assert not result["physical_reference_used_at_TEST"]
    assert result["source_energy"].shape==(240,)
    assert (result["source_energy"][result["source_used"]]>=0).all()
    assert np.any(result["probability"]!=result["original_k2_probability"])
    assert result["concept_name_ru"].startswith("Источник 1")
    assert result["known_signed_contributions"].shape==(240,2)
    assert result["external_source_signed_contribution"].shape==(240,)


def test_unseen_instrument_drift_revokes_physical_semantics_and_exact_K2(setup):
    result=_predict(setup,shift=True)
    assert not result["semantic_scope"]["physical_label_allowed"]
    assert result["semantic_scope"]["status"]=="READOUT_DISTRIBUTION_SHIFT_REVOKED"
    assert result["concept_name_ru"] is None
    assert not result["source_used"].any()
    np.testing.assert_array_equal(
        result["probability"],result["original_k2_probability"])


def test_missing_all_observations_for_one_case_falls_back_without_oracle(setup):
    w,_,_,_,_,_=setup
    m=w["measured_boolean_masks"]["TEST_MCAR20"][w["test"]].copy()
    m[0]=False
    result=_predict(setup,mask_override=m)
    assert not result["source_used"][0]
    assert result["probability"][0]==result["original_k2_probability"][0]
    assert result["concept_name_ru"] is not None
    assert result["source_energy"][0]!=result["source_energy"][0]


def test_declines_mixed_source_train_contract_and_wrong_device(setup):
    w,c,d,mapping,contract,model=setup
    x=w["shared_observed"][w["test"]]
    mask=w["measured_boolean_masks"]["TEST_MCAR20"][w["test"]]
    wrong_kind=replace(contract,source_training_kind="statistical_V44")
    with pytest.raises(ValueError,match="source distribution"):
        prepare_integrated_concept_v52(
            x,mask,source_calibration=c,energy_calibration=d,
            membership_map=mapping,training_contract=wrong_kind,
            physical_device_id=contract.physical_device_id,
            external_reference_id=contract.external_reference_id)
    wrong_calibration=replace(contract,external_reference_id="unregistered_other_physics")
    with pytest.raises(ValueError,match="identity"):
        prepare_integrated_concept_v52(
            x,mask,source_calibration=c,energy_calibration=d,
            membership_map=mapping,training_contract=wrong_calibration,
            physical_device_id=contract.physical_device_id,
            external_reference_id=contract.external_reference_id)
    missing_external=prepare_integrated_concept_v52(
        x,mask,source_calibration=c,energy_calibration=d,
        membership_map=mapping,training_contract=contract,
        physical_device_id=contract.physical_device_id,
        external_reference_id="")
    assert not missing_external["source_support"].any()
    assert missing_external["physical_name"] is None


def test_attested_expert_reject_and_replace_only_signed_source_contribution(setup):
    result=_predict(setup)
    _,_,_,_,contract,model=setup
    action=np.array(["keep"]*240,dtype=object)
    action[1]="reject"
    action[2]="replace"
    refs=np.array([""]*240,dtype=object)
    refs[1]=refs[2]=contract.external_reference_id
    gamma=float(model.gamma.detach())
    correction=np.full(240,np.nan)
    correction[2]=.95
    updated=apply_integrated_expert_review_v52(
        result,actions=action,reference_ids=refs,
        corrected_membership=correction,gamma=gamma,
        training_contract=contract)
    assert updated["probability"][1]==result["original_k2_probability"][1]
    assert updated["probability"][2]!=result["probability"][2]
    np.testing.assert_array_equal(updated["probability"][[0,3,4]],
                                  result["probability"][[0,3,4]])
    assert updated["n_membership_replacements"]==1
    badrefs=refs.copy();badrefs[1]="unrelated_assay"
    with pytest.raises(ValueError,match="independent reference"):
        apply_integrated_expert_review_v52(
            result,actions=action,reference_ids=badrefs,
            corrected_membership=correction,gamma=gamma,
            training_contract=contract)


def test_frozen_original_K2_requirement_is_explicit(setup):
    w,c,d,mapping,contract,model=setup
    x=w["shared_observed"][w["test"]]
    m=w["measured_boolean_masks"]["TEST_MCAR20"][w["test"]]
    with pytest.raises(ValueError,match="K2 fallback"):
        predict_integrated_conceptfan_v52(
            model,x,m,np.zeros(10),
            source_calibration=c,energy_calibration=d,membership_map=mapping,
            training_contract=contract,
            physical_device_id=contract.physical_device_id,
            external_reference_id=contract.external_reference_id)
