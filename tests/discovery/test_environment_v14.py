"""V14 synthetic-only falsification: unknown action map, unknown K, no ICU data."""
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from conceptfan_discovery.environment_contrast_v14 import (
    fit_environment_contrast,project_environment_concepts,
)
from conceptfan_discovery.environment_synthetic_v14 import make_environment_scm
from conceptfan_discovery.instrumented_v13 import InstrumentedConceptBottleneck


def _fit(s,seed=17):
    return fit_environment_contrast(s.observations,s.environment,
        permutations=99,bootstraps=24,seed=seed)


def test_recovers_three_latent_components_without_action_names_or_oracle():
    simulation=make_environment_scm(k=3,seed=121,per_environment=1400)
    certificate=_fit(simulation)
    assert certificate.selected_rank==3
    assert certificate.to_json()["no_outcomes_no_oracle_no_intervention_targets"]
    assert certificate.n_train_per_environment==(1400,1400,1400)
    estimated=project_environment_concepts(simulation.observations,certificate)
    truth=simulation.true_sources[:,-4:].mean(axis=1)
    ranks=np.array([[np.corrcoef(estimated[:,i],truth[:,j])[0,1]
        for j in range(3)] for i in range(3)])
    rr,cc=linear_sum_assignment(-np.abs(ranks))
    assert np.mean(np.abs(ranks[rr,cc]))>.90
    assert np.linalg.norm(certificate.signature,axis=0).min()==pytest.approx(1)
    assert certificate.selected_rank < simulation.observations.shape[-1]
    assert np.allclose(certificate.signature.T@certificate.demixing,np.eye(3),atol=1e-6)


@pytest.mark.parametrize('mode',["no_shift","mean_only_shift"])
def test_observation_changes_without_covariance_shift_abstain(mode):
    s=make_environment_scm(k=3,seed=17,per_environment=650,mode=mode)
    with pytest.raises(ValueError,match='ABSTAIN'):
        _fit(s)


def test_missing_measurements_or_environment_metadata_abstain():
    s=make_environment_scm(k=3,seed=19,per_environment=650,
                           missing_fraction=.12)
    with pytest.raises(ValueError,match='ABSTAIN'):
        _fit(s)
    valid=make_environment_scm(k=3,seed=19,per_environment=650)
    with pytest.raises(ValueError,match='ABSTAIN'):
        fit_environment_contrast(valid.observations,np.zeros(len(valid.environment)),
                                 permutations=99,bootstraps=24)


def test_observational_latent_rotation_nonidentifiability():
    rng=np.random.default_rng(19)
    z=rng.normal(size=(200,4))
    b=rng.normal(size=(9,4))
    orth,_=np.linalg.qr(rng.normal(size=(4,4)))
    np.testing.assert_allclose(z@b.T,(z@orth)@(b@orth).T,rtol=1e-12,atol=1e-12)
    assert not np.allclose(b,b@orth)


def test_equal_profiles_and_noise_shift_are_not_guaranteed_detectable():
    """A deliberate limitation: this does NOT assert 100% abstention."""
    for mode in ('tied_profiles','sensor_noise_shift'):
        s=make_environment_scm(k=3,seed=50,per_environment=650,mode=mode)
        try:
            c=_fit(s)
        except ValueError:
            continue
        assert not c.to_json()['biological_or_causal_names_identified']
        assert 'invariant sensor noise' in c.to_json()['not_testable_from_covariance']


def test_native_and_simple_logistic_are_exact_additive_on_same_train_sources():
    s=make_environment_scm(k=3,seed=119,per_environment=650)
    c=_fit(s)
    feat=project_environment_concepts(s.observations,c)
    logit=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000)
    logit.fit(feat,s.labels)
    terms=feat[:20]*logit.coef_[0]
    np.testing.assert_allclose(terms.sum(1)+logit.intercept_[0],
                               logit.decision_function(feat[:20]),atol=1e-10)
    model=InstrumentedConceptBottleneck(c.signature)
    x=torch.tensor(s.observations[:20],dtype=torch.float32)
    out=model(x)
    assert out['contributions'].shape==(20,3)
    assert float((out['logits']-out['contributions'].sum(-1)-model.bias).detach().abs().max())<1e-6
    out['logits'].square().mean().backward()
    assert model.raw_beta.grad is not None
    assert model.threshold.grad is not None
