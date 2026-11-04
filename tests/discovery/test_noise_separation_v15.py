"""V15 synthetic positive recovery, abstention, and constructive false certificates."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from conceptfan_discovery.lag_synthetic_v15 import make_lag_scm
from conceptfan_discovery.lag_contrast_v15 import fit_lagged_contrast,project_lag_concepts
from conceptfan_discovery.crossview_synthetic_v15 import make_crossview_scm
from conceptfan_discovery.crossview_v15 import fit_crossview_environment

@pytest.mark.parametrize("k",[3,5])
def test_lag_recovers_when_real_latent_variances_change(k):
    s=make_lag_scm(k=k,seed=991+k,per_env=950,mode="latent_shift")
    c=fit_lagged_contrast(s.x,s.env,sensor_temporal_white_documented=True,
                          seed=30+k,permutations=99,bootstraps=24)
    assert c.source_rank==k
    estimated=project_lag_concepts(s.x,c)
    truth=s.true_sources[:,-4:].mean(axis=1)
    mat=np.array([[abs(spearmanr(estimated[:,i],truth[:,j]).statistic)
                   for j in range(k)] for i in range(k)])
    a,b=linear_sum_assignment(-mat)
    assert mat[a,b].mean()>.90

@pytest.mark.parametrize("mode",["white_sensor_shift","no_shift"])
def test_lag_abstains_when_latent_dynamics_are_unchanged(mode):
    s=make_lag_scm(k=3,seed=994,per_env=950,mode=mode)
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_lagged_contrast(s.x,s.env,sensor_temporal_white_documented=True,
                            seed=33,permutations=99,bootstraps=24)

def test_temporal_sensor_noise_provenance_is_mandatory():
    s=make_lag_scm(k=3,seed=994,per_env=950)
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_lagged_contrast(s.x,s.env)

def test_colored_sensor_noise_can_falsely_certify_source_shifts():
    s=make_lag_scm(k=3,seed=994,per_env=950,mode="colored_sensor_shift")
    np.testing.assert_array_equal(s.source_variances[0],s.source_variances[1])
    c=fit_lagged_contrast(s.x,s.env,sensor_temporal_white_documented=True,
                          seed=33,permutations=99,bootstraps=24)
    assert c.source_rank==3
    assert c.to_json()["white_sensor_noise_verified_from_observations"] is False

def test_crossview_true_shift_and_shared_noise_false_identity():
    for mode in ("latent_shift","shared_colored_noise"):
        s=make_crossview_scm(k=3,seed=519,per_env=1000,mode=mode)
        c=fit_crossview_environment(s.a,s.b,s.env,
             independently_noisy_views_documented=True,
             seed=16,permutations=99,bootstraps=24)
        assert c.source_rank==3
        if mode=="shared_colored_noise":
            assert not c.to_json()["independent_sensor_views_verified_from_data"]

def test_independent_colored_sensor_noise_rejected_by_crossview():
    s=make_crossview_scm(k=3,seed=519,per_env=1000,
                         mode="independent_colored_noise")
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_crossview_environment(s.a,s.b,s.env,
             independently_noisy_views_documented=True,
             seed=16,permutations=99,bootstraps=24)

def test_crossview_requires_externally_verified_independent_errors():
    s=make_crossview_scm(k=3,seed=519,per_env=1000)
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_crossview_environment(s.a,s.b,s.env)

def test_colored_noise_equivalence_across_lag_and_crossview():
    rng=np.random.default_rng(111)
    b=rng.normal(size=(6,3))
    base=np.eye(3)*.7;shift=np.diag([.4,.8,1.2]);rho=.7
    np.testing.assert_allclose(
        b@(rho*(base+shift))@b.T,
        b@(rho*base)@b.T+b@(rho*shift)@b.T)
    a=rng.normal(size=(6,3));c=rng.normal(size=(7,3))
    np.testing.assert_allclose(
        a@(base+shift)@c.T,
        a@base@c.T+a@shift@c.T)
