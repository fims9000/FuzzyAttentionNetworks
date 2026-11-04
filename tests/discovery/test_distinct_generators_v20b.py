"""V20b generator partitions, non-leakage and irregular-measurement contracts."""
import numpy as np
import pytest
import torch

from conceptfan_discovery.distinct_synthetic_v20b import (
    FAMILIES,SEEDS,SIZES,make_distinct_generator_data,
)
from conceptfan_discovery.experiment_distinct_families_v20b import _measured_history_features
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


@pytest.mark.parametrize("family",FAMILIES)
def test_distinct_families_are_reproducible_disjoint_and_have_no_oracle(family):
    x=make_distinct_generator_data(family)
    y=make_distinct_generator_data(family)
    assert x.family==family
    splits=(x.discovery.train,x.discovery.validation,x.test)
    reloaded=(y.discovery.train,y.discovery.validation,y.test)
    for split,again,name in zip(splits,reloaded,("train","validation","test")):
        assert len(split.y)==SIZES[name]
        assert split.x.shape==(SIZES[name],24,5)
        assert split.mask.shape==split.x.shape
        assert split.future.shape==(SIZES[name],6,5)
        assert split.oracle_states is None
        assert np.array_equal(split.x,again.x)
        assert np.array_equal(split.mask,again.mask)
        assert np.array_equal(split.y,again.y)
        assert np.isin(split.y,[0,1]).all()
        assert len(np.unique(split.y))==2
        assert np.all(split.x[split.mask==0]==0)
        assert not np.isnan(split.x).any()
        assert _measured_history_features(split).shape==(SIZES[name],30)
    assert set(splits[0].episode_ids).isdisjoint(splits[1].episode_ids)
    assert set(splits[0].episode_ids).isdisjoint(splits[2].episode_ids)
    assert set(splits[1].episode_ids).isdisjoint(splits[2].episode_ids)
    if family=="informative_sampling":
        assert float(x.discovery.train.mask.mean())<.9
        assert float(x.discovery.train.mask.mean())>.35
    else:
        assert float(x.discovery.train.mask.mean())==1.
    assert x.discovery.receipt["no_latent_oracle_exported"]


def test_native_v10_mask_only_process_and_unknown_physical_neutral_on_real_missingness():
    family=make_distinct_generator_data("informative_sampling")
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=6,n_observation_concepts=2,
        adaptive_feature_gradients=True,
        scales=(1,4,12),
    )
    model.eval()
    split=family.discovery.validation
    x=torch.from_numpy(split.x[:5])
    mask=torch.from_numpy(split.mask[:5])
    with torch.no_grad():
        output=model(x,mask,process_mask=mask)
    assert torch.isfinite(output.logits).all()
    assert float(output.decomposition_error())<1e-6
    assert output.observation_contributions.shape==(5,2)
    assert output.physical.concept_trajectories.shape==(5,24,3)


def test_distinct_families_do_not_share_identical_measurement_generators():
    nonlinear=make_distinct_generator_data("nonlinear_delayed")
    regime=make_distinct_generator_data("regime_switch")
    assert not np.array_equal(nonlinear.discovery.train.x,regime.discovery.train.x)
    assert nonlinear.discovery.receipt["different_data_generator_from_V18"]
