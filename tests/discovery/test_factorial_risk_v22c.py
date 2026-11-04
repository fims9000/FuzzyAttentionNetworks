"""V22c: risk-sign and risk-mask are isolated changes to the native K2 graph."""
import numpy as np
import pytest
import torch
from dataclasses import replace

from conceptfan_discovery.mixed_source_data_v22 import (
    make_observable_v22,train_only_whitening,build_native_v22_data,
)
from conceptfan_discovery.environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,native_basis_from_separation,
)
from conceptfan_discovery.native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,
)
from conceptfan_discovery.clinical_multiscale_v10 import fit_multiscale
from conceptfan_discovery.experiment_native_risk_v18 import _config
from conceptfan_discovery.experiment_factorial_risk_v22c import _valid_spearman


def _source_risk_data():
    full=make_observable_v22("mixed_artifact_only_variance")
    core=replace(
        full,
        train_x_full=full.train_x_full[:600],
        train_environment=full.train_environment[:600],
        train_labels=full.train_labels[:600],
        train_episode_ids=full.train_episode_ids[:600],
    )
    whiten=train_only_whitening(core)
    src=fit_train_only_environment_source_directions(
        core,whiten=whiten,permutations=99,bootstraps=24,
    )
    assert src.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
    basis=native_basis_from_separation(src)
    data=build_native_v22_data(core,whiten)
    screen=whiten.transform(full.train_x_full[600:])[:,:24]
    plan=screen_v22_native_sources_on_disjoint_train_labels(
        screen,np.ones_like(screen),full.train_labels[600:],basis,
        original_source_train_episode_ids=data.train.episode_ids,
        screen_episode_ids=full.train_episode_ids[600:],
        permutations=99,bootstraps=99,
    )
    return data,basis,plan


def test_correctly_screened_mixed_source_has_one_risk_and_one_descriptive_slot():
    _,_,risk=_source_risk_data()
    assert len(risk.active_native_raw_decision_initialization)==1
    assert len(risk.structurally_descriptive_only_native_slots)==1
    assert set(risk.active_native_raw_decision_initialization).isdisjoint(
        risk.structurally_descriptive_only_native_slots
    )
    assert all(not r.physical_source_or_clinical_cause_certified
               for r in risk.per_source_evidence)


def test_exact_factorial_initial_native_parameters_differ_only_on_risk_policy_and_weights():
    data,basis,risk=_source_risk_data()
    config,dual,v10=_config()
    config["training"]["epochs"]=0
    config["training"]["batch_size"]=256
    mods={}
    for arm in ("random_original","signed_start_only",
                "descriptive_mask_only","signed_start_and_mask"):
        use_sign=arm in ("signed_start_only","signed_start_and_mask")
        use_mask=arm in ("descriptive_mask_only","signed_start_and_mask")
        model,_=fit_multiscale(
            data,basis,11,config,dual,v10,
            consistency=False,feature_exploration=False,
            initial_native_raw_decision_weights=(
                risk.active_native_raw_decision_initialization if use_sign else None
            ),
            initial_native_descriptive_only_slots=(
                risk.structurally_descriptive_only_native_slots if use_mask else None
            ),
        )
        mods[arm]=model
    original=mods["random_original"].state_dict()
    changed_expected={
        "signed_start_only":{"physical.decision_raw_weight"},
        "descriptive_mask_only":{"physical.signed_risk_policy"},
        "signed_start_and_mask":{
            "physical.decision_raw_weight","physical.signed_risk_policy",
        },
    }
    for name,model in mods.items():
        if name=="random_original":continue
        modified={
            key for key,value in model.state_dict().items()
            if not torch.equal(value,original[key])
        }
        assert modified==changed_expected[name]
        if "mask" in name:
            for slot in risk.structurally_descriptive_only_native_slots:
                assert int(model.physical.signed_risk_policy[slot])==2
                assert float(model.physical.decision_weights()[slot])==0
        x=torch.from_numpy(data.validation.x[:8])
        m=torch.from_numpy(data.validation.mask[:8])
        model.eval()
        with torch.no_grad():
            out=model(x,m,process_mask=m)
        assert float(out.decomposition_error())<1e-6
        assert torch.isfinite(out.logits).all()


def test_constant_descriptive_contribution_is_undefined_not_zero_stability():
    common=np.ones(300)
    assert _valid_spearman(common,common) is None
    assert _valid_spearman(common,np.linspace(-1,1,300)) is None
    x=np.linspace(-3,3,300)
    assert _valid_spearman(x,x)==pytest.approx(1.,abs=1e-10)
    assert _valid_spearman(x,-x)==pytest.approx(-1.,abs=1e-10)
    with pytest.raises(ValueError,match="paired"):
        _valid_spearman(x,x[:10])
