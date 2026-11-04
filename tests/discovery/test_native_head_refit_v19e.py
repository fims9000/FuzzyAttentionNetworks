"""V19e regressions: native refit changes only original risk coefficients."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from conceptfan_discovery.data import DiscoverySplit
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.native_head_refit_v19e import fit_train_only_native_risk_head


def setup_model_and_train(n=128):
    rng=np.random.default_rng(20611)
    x=rng.normal(size=(n,24,5)).astype(np.float32)
    mask=np.ones_like(x)
    y=(x[:,-6:,2].mean(axis=1)>0).astype(np.float32)
    data=DiscoverySplit(
        x=x,mask=mask,
        future=np.zeros((n,3,5),dtype=np.float32),
        future_mask=np.ones((n,3,5),dtype=np.float32),
        y=y,oracle_states=None,
        episode_ids=np.arange(100,100+n,dtype=np.int64),
    )
    torch.manual_seed(207)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
        scales=(1,4,12),
        adaptive_feature_gradients=True,
        temporal_readout_mode="learned_shape",
    )
    model.eval()
    return model,data


def test_native_head_refit_train_objective_and_exact_decomposition():
    model,data=setup_model_and_train()
    x=torch.from_numpy(data.x)
    m=torch.from_numpy(data.mask)
    with torch.no_grad():
        before=model(x,m,process_mask=m)
    original={name:t.detach().clone() for name,t in model.state_dict().items()}
    receipt=fit_train_only_native_risk_head(
        model,data,declared_train_episode_ids=data.episode_ids,
        max_iter=120,
    )
    with torch.no_grad():
        after=model(x,m,process_mask=m)
    assert receipt.n_train_episodes==len(data.y)
    assert receipt.final_objective < receipt.original_objective
    assert receipt.final_train_bce < receipt.original_train_bce
    assert float(after.decomposition_error()) < 1e-6
    assert float((after.logits-before.logits).abs().max()) > 1e-5
    torch.testing.assert_close(after.physical.concept_trajectories,before.physical.concept_trajectories)
    torch.testing.assert_close(after.physical.future_prediction,before.physical.future_prediction)
    torch.testing.assert_close(after.physical.reconstruction,before.physical.reconstruction)
    torch.testing.assert_close(after.observation_contributions,before.observation_contributions)
    for name,previous in original.items():
        if name not in ("physical.decision_raw_weight","physical.decision_bias"):
            torch.testing.assert_close(model.state_dict()[name],previous,rtol=0,atol=0)
    assert model.temporal_readout_mode=="learned_shape"


def test_nontrain_episode_or_oracle_is_refused():
    model,data=setup_model_and_train()
    with pytest.raises(ValueError,match="TRAIN IDs"):
        fit_train_only_native_risk_head(
            model,data,declared_train_episode_ids=np.arange(300,428),
        )
    data.oracle_states=np.ones((len(data.y),1,1),dtype=np.float32)
    with pytest.raises(ValueError,match="oracle"):
        fit_train_only_native_risk_head(
            model,data,declared_train_episode_ids=data.episode_ids,
        )


def test_existing_signed_policy_is_honored_not_relabelled():
    model,data=setup_model_and_train()
    with torch.no_grad():
        model.physical.signed_risk_policy[2]=-1
        model.physical.decision_raw_weight[2]=.12
    receipt=fit_train_only_native_risk_head(
        model,data,declared_train_episode_ids=data.episode_ids,
        max_iter=60,
    )
    assert receipt.final_objective <= receipt.original_objective+1e-6
    assert float(model.physical.decision_weights()[2].detach()) <= 0
    assert int(model.physical.signed_risk_policy[2])==-1


def test_hyperparameter_and_label_contracts_fail_closed():
    model,data=setup_model_and_train()
    with pytest.raises(ValueError,match="controls"):
        fit_train_only_native_risk_head(
            model,data,declared_train_episode_ids=data.episode_ids,
            weight_l2=-.1,
        )
    data.y[:]=1
    with pytest.raises(ValueError,match="both outcome classes"):
        fit_train_only_native_risk_head(
            model,data,declared_train_episode_ids=data.episode_ids,
        )
