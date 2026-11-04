"""V21b: TRAIN-screened initial risk sign changes no native architecture path."""
import numpy as np
import pytest
import torch

from conceptfan_discovery.provenance_data_v21 import (
    make_v21_provenance_cohort,CHANNELS,
)
from conceptfan_discovery.experiment_native_risk_v18 import _config
from conceptfan_discovery.clinical_multiscale_v10 import fit_multiscale


def _fit(override):
    cohort=make_v21_provenance_cohort("physical_predictive")
    config,dual,v10=_config()
    config["training"]["epochs"]=1
    config["training"]["batch_size"]=128
    return fit_multiscale(
        cohort.discovery,
        np.eye(len(CHANNELS),3),
        11,config,dual,v10,
        consistency=False,feature_exploration=True,
        initial_native_raw_decision_weights=override,
    )


def test_train_screen_native_initial_weight_opt_in_and_exact_additivity():
    base,plain=_fit(None)
    oriented,signed=_fit({2:.05})
    assert plain["train_initial_native_raw_decision_weight_overrides"] is None
    assert signed["train_initial_native_raw_decision_weight_overrides"]=={"2":.05}
    assert signed["train_risk_initialization_is_not_physical_source_certification"]
    assert int(oriented.physical.signed_risk_policy[2])==0
    assert oriented.physical.n_slots==base.physical.n_slots==3
    xx=torch.from_numpy(make_v21_provenance_cohort("physical_predictive").discovery.validation.x[:12])
    mm=torch.ones_like(xx)
    base.eval();oriented.eval()
    with torch.no_grad():
        a=base(xx,mm,process_mask=mm)
        b=oriented(xx,mm,process_mask=mm)
    assert float(a.decomposition_error())<1e-6
    assert float(b.decomposition_error())<1e-6
    assert not torch.equal(a.logits,b.logits)


@pytest.mark.parametrize("bad",[
    {},
    {3:.05},
    {-1:.05},
    {2:float("nan")},
    {2:.8},
    {2:True},
    {False:.05},
    {"2":.05},
])
def test_unsigned_or_invalid_native_initial_policy_refused(bad):
    with pytest.raises(ValueError,match="risk initialization"):
        _fit(bad)
