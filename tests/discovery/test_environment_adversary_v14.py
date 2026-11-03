"""Fundamental counterexample: environment-dependent sensor noise is not ICA."""
import numpy as np
from conceptfan_discovery.environment_contrast_v14 import (
    fit_environment_contrast, project_environment_concepts,
)
from conceptfan_discovery.environment_synthetic_v14 import make_environment_scm


def test_source_aligned_sensor_noise_shift_fools_observational_certificate():
    # Actual Z variance is identical in all three environments. Only sensor
    # noise covariance changes, but its shifts are aligned with true B columns.
    simulation=make_environment_scm(
        k=3,seed=57,per_environment=1000,mode='noise_shift_aligned')
    np.testing.assert_array_equal(
        simulation.source_variances[0],simulation.source_variances[1])
    np.testing.assert_array_equal(
        simulation.source_variances[1],simulation.source_variances[2])
    certificate=fit_environment_contrast(
        simulation.observations,simulation.environment,
        seed=33,permutations=99,bootstraps=24)
    # Deliberate false positive, NOT evidence of successfully discovered factors.
    assert certificate.selected_rank==3
    assert min(certificate.response_overlap_q10)>.9
    observed=project_environment_concepts(simulation.observations,certificate)
    ground_truth=simulation.true_sources[:,-4:].mean(axis=1)
    corr=np.abs(np.corrcoef(observed.T,ground_truth.T)[:3,3:])
    assert np.max(corr.min(axis=1))<.60
    assert not certificate.to_json()['biological_or_causal_names_identified']
    assert 'invariant sensor noise' in certificate.to_json()['not_testable_from_covariance']


def test_covariance_noise_shift_is_algebraically_non_identifiable():
    # For any environment-varying diagonal D_e, both models give identical
    # observable covariance, so this is NOT fixable by another bootstrap.
    rng=np.random.default_rng(991)
    b=rng.normal(size=(6,3))
    omega=np.eye(6)*.2
    base=np.array([.7,.9,.6])
    deltas=np.array([[0.,0.,0.],[1.,.8,.6],[.4,1.5,2.1]])
    for delta in deltas:
        # Model A: latent variance changes, sensor noise invariant.
        model_a=b@np.diag(base+delta)@b.T+omega
        # Model B: latent variance invariant, sensor noise changes.
        model_b=b@np.diag(base)@b.T+(omega+b@np.diag(delta)@b.T)
        np.testing.assert_allclose(model_a,model_b,rtol=1e-12,atol=1e-12)
