"""V18: independent predictive risk screening is not physical certification."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from conceptfan_discovery.candidate_admission_v17 import (
    ProposedObservableConcept, admit_candidate_without_prediction_drift,
)
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.risk_screen_v18 import (
    screen_risk_association, configure_native_risk_orientation,
    begin_native_risk_training, risk_evidence_passport, native_observable_grounding_loss,
)


def _candidate():
    return ProposedObservableConcept(
        candidate_id="synthetic-measured-direction",
        direction=np.array([0.,0.,1.,0.,0.]),
        train_episode_count=300,positive_lag_strength=.3,
        permutation_p=.005,bootstrap_abs_cosine_q10=.95,
        top_signed_sensors=(("sensor_2",1.),),
    )


def _screen_inputs(seed=71,n=260):
    rng=np.random.default_rng(seed)
    x=rng.normal(scale=.1,size=(n,18,5))
    hidden=rng.normal(size=n)
    x[:,:,2] += .85*hidden[:,None]
    pos=(rng.random(n)<1/(1+np.exp(-3*hidden))).astype(float)
    return x,np.ones_like(x),pos,np.arange(600,600+n)


def _screen(y_override=None):
    x,m,y,ids=_screen_inputs()
    if y_override is not None:
        y=y_override(x,m,y,ids)
    return screen_risk_association(
        _candidate(),x,m,y,screening_episode_ids=ids,
        discovery_episode_ids=np.arange(0,300),
        seed=809,permutations=199,bootstraps=199,
    )


def test_positive_train_association_accepted_without_physical_causality():
    evidence=_screen()
    assert evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION"
    assert evidence.signed_orientation == 1
    assert evidence.permutation_p_two_sided <= .01
    assert evidence.bootstrap_ci95[0] > 0
    assert not evidence.physiological_identity_certified
    assert risk_evidence_passport(evidence)["observational_prediction_only"]


def test_negative_observable_association_orients_opposite_direction():
    evidence=_screen(lambda x,m,y,ids: 1-y)
    assert evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION"
    assert evidence.signed_orientation == -1
    assert evidence.bootstrap_ci95[1] < 0


def test_observational_null_and_unknown_information_abstain():
    x,m,y,ids=_screen_inputs()
    rng=np.random.default_rng(104)
    null=rng.binomial(1,.5,len(y)).astype(float)
    evidence=_screen(lambda x,m,y,ids: null)
    assert evidence.status.startswith("ABSTAIN")
    assert evidence.signed_orientation == 0
    with pytest.raises(ValueError,match="ABSTAIN"):
        screen_risk_association(
            _candidate(),x,m,y,screening_episode_ids=ids,
            discovery_episode_ids=ids[:1],seed=809,
            permutations=199,bootstraps=199,
        )
    m[0,0,2]=0
    with pytest.raises(ValueError,match="ABSTAIN"):
        screen_risk_association(
            _candidate(),x,m,y,screening_episode_ids=ids,
            discovery_episode_ids=range(300),seed=809,
            permutations=199,bootstraps=199,
        )


def test_native_orientation_does_not_create_a_posthoc_classifier_or_immediate_drift():
    torch.manual_seed(101)
    basis=np.eye(5,2,dtype=np.float64)
    names=tuple(f"sensor_{i}" for i in range(5))
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=basis,channels=names,
        forecast_horizon=3,n_observation_concepts=2,
    )
    model.eval()
    candidate=_candidate()
    expanded=admit_candidate_without_prediction_drift(
        model,candidate,expert_decision="approve",
    )
    evidence=_screen()
    x=torch.randn(6,20,5)
    mask=torch.ones_like(x)
    with torch.no_grad():
        before=model(x,mask,process_mask=mask)
    configure_native_risk_orientation(
        expanded,candidate,evidence,slot=expanded.physical.n_slots-1,
    )
    with torch.no_grad():
        unchanged=expanded(x,mask,process_mask=mask)
    torch.testing.assert_close(unchanged.logits,before.logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(
        unchanged.physical.future_prediction,before.physical.future_prediction,
        atol=1e-6,rtol=1e-6,
    )
    assert float(unchanged.decomposition_error().detach()) < 1e-6
    begin_native_risk_training(expanded,slot=2)
    weights=expanded.physical.decision_weights()
    assert float(weights[2].detach()) > 0
    x=torch.randn(6,20,5)
    mask=torch.ones_like(x)
    expanded.train()
    out=expanded(x,mask,process_mask=mask)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(
        out.logits,torch.tensor([0.,1.,0.,1.,0.,1.])
    )
    loss.backward()
    assert expanded.physical.decision_raw_weight.grad is not None
    assert float(expanded.physical.decision_raw_weight.grad[2].abs()) > 0
    assert float(out.decomposition_error().detach()) < 1e-6
    # The statistical orientation lives on the actual risk graph.
    with torch.no_grad():
        expanded.physical.decision_raw_weight[2].fill_(-.2)
    assert float(expanded.physical.decision_weights()[2]) > 0


def test_negative_policy_enforced_and_unproven_candidate_refused():
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
    )
    expanded=admit_candidate_without_prediction_drift(
        model,_candidate(),expert_decision="approve",
    )
    positive=_screen()
    negative=_screen(lambda x,m,y,ids: 1-y)
    with pytest.raises(ValueError,match="ABSTAIN"):
        configure_native_risk_orientation(
            expanded,_candidate(),replace(positive,status="ABSTAIN"),slot=2,
        )
    configure_native_risk_orientation(expanded,_candidate(),negative,slot=2)
    begin_native_risk_training(expanded,slot=2)
    assert float(expanded.physical.decision_weights()[2]) < 0
    # Previously trained concepts remain unrestricted.
    assert (expanded.physical.signed_risk_policy[:2] == 0).all()


def test_sign_orientation_is_not_independent_cross_seed_evidence():
    evidence=_screen()
    assert risk_evidence_passport(evidence)[
        "sign_is_fixed_from_train_evidence_NOT_new_cross_seed_discovery"
    ]
    assert not evidence.causal_treatment_effect_certified


def test_native_grounding_supervises_actual_v10_concept_not_a_posthoc_head():
    torch.manual_seed(911)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
        adaptive_feature_gradients=True,
    )
    expanded=admit_candidate_without_prediction_drift(
        model,_candidate(),expert_decision="approve",
    )
    values=torch.randn(5,20,5)
    mask=torch.ones_like(values)
    out=expanded(values,mask,process_mask=mask)
    penalty=native_observable_grounding_loss(
        out.physical,values,mask,_candidate(),slot=2,window=6,
    )
    penalty.backward()
    assert float(penalty.detach()) > 0
    assert expanded.physical.feature_logits.grad is not None
    assert float(expanded.physical.feature_logits.grad[-1].abs().sum()) > 0
    assert float(out.decomposition_error().detach()) < 1e-6


def test_native_grounding_refuses_missing_oracle_and_zero_measurement():
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
    )
    expanded=admit_candidate_without_prediction_drift(
        model,_candidate(),expert_decision="approve",
    )
    x=torch.randn(4,20,5)
    absent=torch.zeros_like(x)
    out=expanded(x,absent,process_mask=absent)
    val=native_observable_grounding_loss(out.physical,x,absent,_candidate(),slot=2)
    assert float(val.detach())==0
    with pytest.raises(ValueError,match="shapes"):
        native_observable_grounding_loss(
            out.physical,x[:,:-1],absent[:,:-1],_candidate(),slot=2,
        )
