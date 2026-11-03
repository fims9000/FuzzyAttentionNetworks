"""V13 randomized-action identification: positive and negative controls."""
import numpy as np
import pytest
import torch
from conceptfan_discovery.instrumented_v13 import (
    InterventionProtocol, identify_action_anchored_concepts,
    InstrumentedConceptBottleneck, fit_instrumented_bottleneck,
)
from conceptfan_discovery.intervention_synthetic_v13 import make_intervention_scm

def _design(k,**overrides):
    kwargs=dict(randomized_assignment=True,individually_targeted_actions=True,
        known_positive_action_effects=True,observation_complete=True,
        action_names=tuple(f"randomized_action_{i}" for i in range(k)))
    kwargs.update(overrides)
    return InterventionProtocol(**kwargs)

def _fit(sim,*,seed=41,**kwargs):
    return identify_action_anchored_concepts(
        sim.observations[:,:sim.history_steps],
        sim.actions[:,:sim.history_steps-1],
        _design(sim.actions.shape[-1]),seed=seed,
        bootstraps=24,permutations=99,**kwargs)

@pytest.mark.parametrize('k',[3,5,7])
def test_identified_signatures_match_oracle_sensor_directions(k):
    sim=make_intervention_scm(n=420,k=k,d=k+3,seed=105+k)
    c=_fit(sim)
    true=sim.planted_mixing/np.linalg.norm(sim.planted_mixing,axis=0)
    assert min(np.sum(c.signature*true,axis=0))>.96
    assert min(c.worst_bootstrap_oriented_cosine_q10)>.94
    assert c.pretreatment_balance_permutation_p>=.02
    assert c.n_action_channels==k
    assert c.to_json()['latent_scale_identified'] is False
    assert c.to_json()['causal_risk_effects_identified'] is False

def test_protocol_assumptions_are_external_not_inferred():
    sim=make_intervention_scm(n=360,k=3,d=6,seed=14)
    for invalid in (_design(3,randomized_assignment=False),
         _design(3,individually_targeted_actions=False),
         _design(3,known_positive_action_effects=False),
         _design(3,observation_complete=False),
         _design(3,time_alignment="simultaneous")):
        with pytest.raises(ValueError,match="ABSTAIN"):
            identify_action_anchored_concepts(sim.observations[:,:12],
                    sim.actions[:,:11],invalid)

def test_confounding_rejected_by_pre_action_placebo():
    sim=make_intervention_scm(n=380,k=3,d=6,seed=59,action_mode="confounded")
    with pytest.raises(ValueError,match="pre-action sensor placebo"):
        _fit(sim)

@pytest.mark.parametrize("bad",["collinear","none"])
def test_no_individual_identification_from_invalid_actions(bad):
    sim=make_intervention_scm(n=360,k=3,d=6,seed=21,action_mode=bad)
    with pytest.raises(ValueError,match="ABSTAIN"):
        _fit(sim)

def test_missingness_cannot_be_masqueraded_as_observed():
    sim=make_intervention_scm(n=360,k=3,d=6,seed=22,missing_fraction=.10)
    with pytest.raises(ValueError,match="ABSTAIN"):
        _fit(sim)

def test_observational_equivalence_under_latent_rotation():
    rng=np.random.default_rng(12)
    z=rng.normal(size=(160,4));b=rng.normal(size=(8,4));beta=rng.normal(size=4)
    q,_=np.linalg.qr(rng.normal(size=(4,4)))
    np.testing.assert_allclose(z@b.T,(z@q)@(b@q).T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(z@beta,(z@q)@(q.T@beta),rtol=1e-12,atol=1e-12)
    assert np.max(np.abs(b-b@q))>.2

def test_exact_native_bottleneck_and_gradient():
    sim=make_intervention_scm(n=360,k=3,d=6,seed=97)
    cert=_fit(sim);model=InstrumentedConceptBottleneck(cert.signature)
    x=torch.tensor(sim.observations[:9,:12],dtype=torch.float32)
    out=model(x)
    assert out["concepts"].shape==(9,3)
    assert (out["logits"]-out["contributions"].sum(-1)-model.bias).abs().max()<1e-6
    out["logits"].square().mean().backward()
    assert model.threshold.grad is not None and model.raw_beta.grad is not None
    assert len(list(model.named_parameters()))==4
    assert not model.dual.requires_grad
    with pytest.raises(ValueError,match="Incompletely observed"):
        model(x.clone().fill_(float("nan")))

def test_train_only_outcome_path_and_action_selected_K():
    sim=make_intervention_scm(n=360,k=3,d=6,seed=57)
    cert=_fit(sim)
    model,receipt=fit_instrumented_bottleneck(
        cert,sim.observations[:,:12],sim.labels,seed=7,epochs=5)
    assert receipt["task_head_has_no_bypass"]
    assert receipt["concept_count_from_independent_action_channels"]==3
    x=torch.tensor(sim.observations[:8,:12],dtype=torch.float32)
    assert model(x)["contributions"].shape==(8,3)


def test_mislabeled_action_target_cannot_be_discovered_from_randomization_alone():
    """Deliberate blind spot: undocumented multi-target actions can fool V13.

    This control is supposed to SUCCEED at estimating *response* columns
    but FAIL at matching them to true latent sources, demonstrating why
    external single-target provenance is a necessary, non-testable assumption.
    """
    rng=np.random.default_rng(44)
    n,t,k,d=380,12,3,6
    mixing=np.zeros((d,k))
    mixing[:k]=np.eye(k)
    # Each action actually hits the NEXT source, not its declared source.
    unknown_action_mixing=np.array([
        [0.,1.,0.],[0.,0.,1.],[1.,0.,0.],
    ])
    z=np.zeros((n,t,k),dtype=np.float64)
    actions=rng.choice([-1.,1.],size=(n,t-1,k))
    for time in range(t-1):
        z[:,time+1]=(.7*z[:,time]
                     +.5*(actions[:,time]@unknown_action_mixing.T)
                     +rng.normal(scale=.06,size=(n,k)))
    measured=z@mixing.T+rng.normal(scale=.03,size=(n,t,d))
    # Incorrect metadata falsely claims individually targeted latent axes.
    claimed=_design(k)
    result=identify_action_anchored_concepts(
        measured,actions,claimed,seed=66,bootstraps=24,permutations=99,
    )
    true_sources=mixing/np.linalg.norm(mixing,axis=0)
    self_label_similarity=np.abs(np.sum(result.signature*true_sources,axis=0))
    # All measured action responses are identifiable, but their latent names
    # are WRONG; the estimator cannot detect false metadata from this data.
    assert np.max(self_label_similarity)<.1
    assert result.pretreatment_balance_permutation_p>=.02
    assert result.to_json()['latent_scale_identified'] is False
