"""V21b: signed native effects are audited without post-hoc sign rewriting."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.provenance_data_v21 import (
    make_v21_provenance_cohort,CHANNELS,
)
from conceptfan_discovery.clinical_multiscale_v10 import fit_multiscale
from conceptfan_discovery.experiment_native_risk_v18 import _config
from conceptfan_discovery.experiment_sign_stability_v21b import _rho


def _make_original_and_signed():
    data=make_v21_provenance_cohort("physical_predictive").discovery
    basis=np.eye(len(CHANNELS),3)
    config,dual,v10=_config()
    config["training"]["epochs"]=0
    baseline,_=fit_multiscale(
        data,basis,seed=23,base_config=config,
        v7_config=dual,v10_config=v10,
        consistency=False,feature_exploration=True,
    )
    oriented,receipt=fit_multiscale(
        data,basis,seed=23,base_config=config,
        v7_config=dual,v10_config=v10,
        consistency=False,feature_exploration=True,
        initial_native_raw_decision_weights={2:-.05},
    )
    return data,baseline,oriented,receipt


def test_only_one_original_native_head_parameter_changes_at_initialization():
    data,old,signed,receipt=_make_original_and_signed()
    assert receipt["train_initial_native_raw_decision_weight_overrides"]=={"2":-.05}
    assert receipt["train_risk_initialization_is_not_physical_source_certification"]
    assert not signed.training
    original=old.state_dict()
    revised=signed.state_dict()
    assert set(original)==set(revised)
    changed=[]
    for k in original:
        if not torch.equal(original[k],revised[k]):
            changed.append(k)
    assert changed==["physical.decision_raw_weight"]
    torch.testing.assert_close(
        original["physical.decision_raw_weight"][:2],
        revised["physical.decision_raw_weight"][:2],
        rtol=0,atol=0,
    )
    assert float(revised["physical.decision_raw_weight"][2])==pytest.approx(-.05)
    assert int(revised["physical.signed_risk_policy"][2])==0
    assert float(signed.physical.decision_weights()[2])<0


def test_default_training_is_identical_under_no_override():
    data=make_v21_provenance_cohort("physical_predictive").discovery
    basis=np.eye(len(CHANNELS),3)
    config,dual,v10=_config()
    config["training"]["epochs"]=1
    a,r1=fit_multiscale(
        data,basis,11,config,dual,v10,
        consistency=False,feature_exploration=True,
    )
    b,r2=fit_multiscale(
        data,basis,11,config,dual,v10,
        consistency=False,feature_exploration=True,
        initial_native_raw_decision_weights=None,
    )
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    assert r1["train_initial_native_raw_decision_weight_overrides"] is None
    assert r2["train_initial_native_raw_decision_weight_overrides"] is None


@pytest.mark.parametrize("initial",[
    {},{2:float("nan")},{2:float("inf")},{2:.8},{3:-.05},{-1:.05},
    {True:.05},{"2":.05},{2:True},
])
def test_invalid_source_risk_initialization_refuses(initial):
    data=make_v21_provenance_cohort("physical_predictive").discovery
    config,dual,v10=_config()
    config["training"]["epochs"]=0
    with pytest.raises(ValueError,match="Invalid native"):
        fit_multiscale(
            data,np.eye(len(CHANNELS),3),11,config,dual,v10,
            consistency=False,feature_exploration=True,
            initial_native_raw_decision_weights=initial,
        )


def test_sign_flipped_contribution_is_observably_not_stable_and_never_auto_aligned():
    raw=np.linspace(-2,2,240)
    assert _rho(raw,raw)==pytest.approx(1.,abs=1e-10)
    assert _rho(raw,-raw)==pytest.approx(-1.,abs=1e-10)


def test_unknown_concept_source_is_never_given_physical_sign_policy():
    _,_,model,_=_make_original_and_signed()
    assert (model.physical.signed_risk_policy==0).all()
