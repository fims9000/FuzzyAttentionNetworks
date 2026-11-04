"""V20: optimizer and class weighting only alter the existing ConceptFAN risk head."""
import copy

import numpy as np
import pytest
import torch

from conceptfan_discovery.data import DiscoverySplit
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.native_head_factorial_v20 import fit_native_head_factorial_v20


def _fixture(n=128):
    torch.manual_seed(120)
    rng=np.random.default_rng(222)
    x=rng.normal(size=(n,24,5)).astype(np.float32)
    mask=np.ones_like(x)
    y=(x[:,-6:,2].mean(1)+.2*rng.normal(size=n)>.25).astype(np.float32)
    if len(np.unique(y))!=2:
        raise AssertionError("test labels degenerate")
    train=DiscoverySplit(
        x=x,mask=mask,
        future=np.zeros((n,3,5),dtype=np.float32),
        future_mask=np.ones((n,3,5),dtype=np.float32),
        y=y,oracle_states=None,episode_ids=np.arange(5000,5000+n),
    )
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
        scales=(1,4,12),adaptive_feature_gradients=True,
    )
    model.eval()
    return model,train


@pytest.mark.parametrize("method",["lbfgs","adamw"])
@pytest.mark.parametrize("weighted",[False,True])
def test_factored_native_refit_changes_no_other_parameters_and_improves_train_loss(method,weighted):
    model,train=_fixture()
    original={key:v.detach().clone() for key,v in model.state_dict().items()}
    xx=torch.from_numpy(train.x)
    mm=torch.from_numpy(train.mask)
    with torch.no_grad():
        initial=model(xx,mm,process_mask=mm)
    out=fit_native_head_factorial_v20(
        model,train,declared_train_episode_ids=train.episode_ids,
        method=method,pos_weighted=weighted,adamw_steps=120,
        lbfgs_max_iter=60,
    )
    with torch.no_grad():
        after=model(xx,mm,process_mask=mm)
    assert out.pos_weighted==weighted
    assert out.positive_weight_from_train>0
    assert out.final_objective<=out.initial_objective+1e-6
    assert float(after.decomposition_error())<1e-6
    torch.testing.assert_close(
        after.physical.concept_trajectories,initial.physical.concept_trajectories,
    )
    torch.testing.assert_close(
        after.physical.future_prediction,initial.physical.future_prediction,
    )
    torch.testing.assert_close(
        after.physical.reconstruction,initial.physical.reconstruction,
    )
    torch.testing.assert_close(after.observation_contributions,initial.observation_contributions)
    for key,previous in original.items():
        if key not in ("physical.decision_raw_weight","physical.decision_bias"):
            assert torch.equal(model.state_dict()[key],previous)


def test_optimizer_arms_start_identical_but_weighted_objectives_are_not_identical():
    model,train=_fixture()
    a=copy.deepcopy(model)
    b=copy.deepcopy(model)
    u=fit_native_head_factorial_v20(
        a,train,declared_train_episode_ids=train.episode_ids,
        method="lbfgs",pos_weighted=False,lbfgs_max_iter=70,
    )
    w=fit_native_head_factorial_v20(
        b,train,declared_train_episode_ids=train.episode_ids,
        method="lbfgs",pos_weighted=True,lbfgs_max_iter=70,
    )
    assert u.initial_signed_coefficients==w.initial_signed_coefficients
    assert u.positive_weight_from_train==w.positive_weight_from_train
    assert not np.isclose(u.final_unweighted_bce,w.final_unweighted_bce,atol=1e-7)


def test_train_only_oracle_and_hyperparameter_refusal():
    model,train=_fixture()
    with pytest.raises(ValueError,match="TRAIN-only"):
        fit_native_head_factorial_v20(
            model,train,declared_train_episode_ids=np.arange(7000,7128),
            method="lbfgs",pos_weighted=False,
        )
    train.oracle_states=np.zeros((len(train.y),2,3))
    with pytest.raises(ValueError,match="oracle"):
        fit_native_head_factorial_v20(
            model,train,declared_train_episode_ids=train.episode_ids,
            method="lbfgs",pos_weighted=False,
        )
    train.oracle_states=None
    with pytest.raises(ValueError,match="Invalid native head"):
        fit_native_head_factorial_v20(
            model,train,declared_train_episode_ids=train.episode_ids,
            method="lbfgs",pos_weighted=False,effective_beta_l2=-.01,
        )


def test_signed_risk_policy_unchanged_by_optimization():
    model,train=_fixture()
    with torch.no_grad():
        model.physical.signed_risk_policy[2]=-1
        model.physical.decision_raw_weight[2]=.05
    fit_native_head_factorial_v20(
        model,train,declared_train_episode_ids=train.episode_ids,
        method="adamw",pos_weighted=False,adamw_steps=50,
    )
    assert int(model.physical.signed_risk_policy[2])==-1
    assert float(model.physical.decision_weights()[2])<=0
