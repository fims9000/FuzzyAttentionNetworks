"""Executable V17 positive and failure controls; hidden-source labels are NOT fit inputs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.candidate_admission_v17 import (
    ProposedObservableConcept,
    propose_observable_residual_concept,
    admit_candidate_without_prediction_drift,
    candidate_passport,
)
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


def _history(mode,seed=2507,n=440,t=17,d=5):
    rng=np.random.default_rng(seed)
    x=rng.normal(scale=.09,size=(n,t,d))
    # Two currently modeled sources occupy channel 0 and 1.
    for i in (0,1):
        x[:, :, i] += rng.normal(size=(n,t))*.3
    shock=rng.normal(size=(n,t))
    state=np.zeros((n,t))
    if mode in ("latent_extra","colored_sensor_error"):
        for j in range(t):
            state[:,j]=(shock[:,j] if j==0 else
                         .78*state[:,j-1]+np.sqrt(1-.78**2)*shock[:,j])
    elif mode=="white_noise_only":
        state=shock.copy()
    else:
        raise ValueError("Unknown fixture")
    # In colored_sensor_error this is explicitly an INSTRUMENT noise process
    # even though its observational law is indistinguishable from the latent.
    if mode=="white_noise_only":
        x[:,:,2] += .18*state
    else:
        x[:,:,2] += .9*state
    return x,np.ones_like(x),np.eye(d,2),state


def test_v17_recovers_observable_residual_direction_without_latent_oracle():
    x,m,b,true_source = _history("latent_extra")
    c=propose_observable_residual_concept(
        x,m,b,tuple(f"sensor_{i}" for i in range(5)),
        seed=701,permutations=99,bootstraps=24,
    )
    assert c.permutation_p <= .01
    assert c.bootstrap_abs_cosine_q10 >= .86
    assert abs(c.direction[2]) > .95
    assert c.top_signed_sensors[0][0]=="sensor_2"
    assert candidate_passport(c)["physical_cause_identified"] is False
    assert c.expert_approved is False


def test_white_residual_noise_abstains():
    x,m,b,_ = _history("white_noise_only",seed=221)
    with pytest.raises(ValueError,match="ABSTAIN"):
        propose_observable_residual_concept(
            x,m,b,tuple(f"sensor_{i}" for i in range(5)),
            seed=701,permutations=99,bootstraps=24,
        )


def test_colored_sensor_noise_still_produces_false_physical_candidates():
    x,m,b,_ = _history("colored_sensor_error",seed=2507)
    c=propose_observable_residual_concept(
        x,m,b,tuple(f"sensor_{i}" for i in range(5)),
        seed=701,permutations=99,bootstraps=24,
    )
    assert c.permutation_p <= .01
    assert candidate_passport(c)["physical_cause_identified"] is False


def test_missingness_refuses_unjustified_inference():
    x,m,b,_ = _history("latent_extra")
    m[0,4,2]=0
    with pytest.raises(ValueError,match="measurement gaps"):
        propose_observable_residual_concept(
            x,m,b,tuple(f"sensor_{i}" for i in range(5)),
        )


@pytest.mark.parametrize("answer",["reject","unsure"])
def test_reviewer_can_block_admission(answer):
    torch.manual_seed(39)
    base=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
    )
    x,m,b,_ = _history("latent_extra")
    c=propose_observable_residual_concept(
        x,m,b,tuple(f"sensor_{i}" for i in range(5)),
        seed=701,permutations=99,bootstraps=24,
    )
    with pytest.raises(ValueError,match="ABSTAIN"):
        admit_candidate_without_prediction_drift(base,c,expert_decision=answer)


def test_approved_candidate_becomes_native_slot_with_no_old_prediction_drift():
    torch.manual_seed(39)
    base=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
    )
    base.eval()
    x,m,b,_ = _history("latent_extra")
    c=propose_observable_residual_concept(
        x,m,b,tuple(f"sensor_{i}" for i in range(5)),
        seed=701,permutations=99,bootstraps=24,
    )
    new=admit_candidate_without_prediction_drift(base,c,expert_decision="approve")
    assert new.physical.n_slots == base.physical.n_slots + 1
    values=torch.randn(4,20,5)
    observed=torch.ones_like(values)
    with torch.no_grad():
        before=base(values,observed,process_mask=observed)
        after=new(values,observed,process_mask=observed)
    torch.testing.assert_close(before.logits,after.logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(
        before.physical.future_prediction,after.physical.future_prediction,
        atol=1e-6,rtol=1e-6,
    )
    torch.testing.assert_close(
        before.physical.reconstruction,after.physical.reconstruction,
        atol=1e-6,rtol=1e-6,
    )
    torch.testing.assert_close(
        before.observation_contributions,after.observation_contributions,
        atol=1e-7,rtol=1e-7,
    )
    assert float(after.physical.contributions[:,-1].abs().max()) < 1e-7
    assert float(after.decomposition_error()) < 1e-6
    # The new slot is *actually on* the native decision path once optimized.
    with torch.no_grad():
        new.physical.decision_raw_weight[-1].fill_(.5)
        altered=new(values,observed,process_mask=observed)
    assert float((altered.logits-before.logits).abs().max()) > 1e-6
    assert float(altered.decomposition_error()) < 1e-6


def test_candidate_duplicating_existing_basis_must_be_rejected():
    base=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
    )
    fake=ProposedObservableConcept(
        "fake",np.array([1.,0,0,0,0]),200,.2,.01,.99,(("sensor_0",1.0),),
    )
    with pytest.raises(ValueError,match="ABSTAIN"):
        admit_candidate_without_prediction_drift(base,fake,expert_decision="approve")
