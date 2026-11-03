import numpy as np
import pytest

from conceptfan_discovery.variable_factor_data import generate_variable_factor_data
from conceptfan_discovery.evaluation import oracle_correspondence


@pytest.mark.parametrize("k",[3,5,7])
def test_true_k_only_available_to_external_evaluator(k):
    data=generate_variable_factor_data(
        n_true_factors=k,seed=417+k,split_seed=703+k,
        n_samples=240,n_channels=8,input_steps=18,horizon=6,
    )
    assert data.train.oracle_states.shape[-1]==k
    assert data.validation.oracle_states.shape[-1]==k
    assert len(next(iter(data.train.loader(12,False,0))))==5
    assert data.receipt["latent_ground_truth_used_for_training"] is False
    assert data.receipt["test_inspected_for_model_selection"] is False
    assert data.train.x.shape[-1]==8
    assert np.isfinite(data.train.x).all()
    assert np.unique(data.train.y).size==2


def test_generator_dimension_changes_underlying_transition_not_just_labels():
    a=generate_variable_factor_data(n_true_factors=3,seed=421,split_seed=705,n_samples=240)
    b=generate_variable_factor_data(n_true_factors=7,seed=421,split_seed=705,n_samples=240)
    assert a.receipt["latent_transition_sha256"]!=b.receipt["latent_transition_sha256"]
    assert a.receipt["observable_mixing_sha256"]!=b.receipt["observable_mixing_sha256"]


def test_unacceptable_variable_factor_counts_rejected():
    with pytest.raises(ValueError):
        generate_variable_factor_data(n_true_factors=1,seed=11,split_seed=12,n_samples=240)
    with pytest.raises(ValueError):
        generate_variable_factor_data(n_true_factors=7,n_channels=4,seed=11,split_seed=12,n_samples=240)
