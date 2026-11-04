"""V19: native concept-only temporal readout vs fixed / legacy controls."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conceptfan_discovery.multiscale_physiology_v10 import (
    MultiScaleDualProcessConceptFAN,
)
from conceptfan_discovery.candidate_admission_v17 import (
    ProposedObservableConcept, admit_candidate_without_prediction_drift,
)


def build(mode: str):
    return MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
        temporal_readout_mode=mode,adaptive_feature_gradients=True,
        scales=(1,4,12),
    )


def test_legacy_v10_remains_default_without_new_readout():
    torch.manual_seed(110)
    base=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(5)),
        forecast_horizon=3,n_observation_concepts=2,
        adaptive_feature_gradients=True,scales=(1,4,12),
    )
    assert base.temporal_readout_mode=="legacy"
    assert not hasattr(base.physical,"temporal_readout_logits")
    x=torch.randn(5,24,5)
    m=torch.ones_like(x)
    out=base(x,m,process_mask=m)
    original=out.physical.concept_trajectories[:,-6:].mean(1)
    torch.testing.assert_close(out.physical.concept_summary,original)
    assert float(out.decomposition_error()) < 1e-6


@pytest.mark.parametrize("mode",["fixed_shape","learned_shape"])
def test_new_model_uses_same_native_concepts_and_exact_additive_bottleneck(mode):
    torch.manual_seed(17)
    base=build("legacy")
    model=build(mode)
    old_state=base.state_dict()
    missing,unexpected=model.load_state_dict(old_state,strict=False)
    assert set(missing)=={"physical.temporal_readout_logits"}
    assert not unexpected
    x=torch.randn(6,24,5)
    m=torch.ones_like(x)
    model.eval();base.eval()
    with torch.no_grad():
        native=base(x,m,process_mask=m)
        changed=model(x,m,process_mask=m)
    torch.testing.assert_close(changed.physical.concept_trajectories,native.physical.concept_trajectories)
    torch.testing.assert_close(changed.physical.future_prediction,native.physical.future_prediction)
    torch.testing.assert_close(changed.physical.reconstruction,native.physical.reconstruction)
    torch.testing.assert_close(changed.observation_contributions,native.observation_contributions)
    assert float(changed.decomposition_error()) < 1e-6
    assert model.physical.temporal_readout_logits.requires_grad == (mode=="learned_shape")
    # The only new learned decision state is a softmax mixture over known
    # native concept shape statistics; there is no direct sensor-to-logit head.
    passport=model.physical.temporal_readout_passport()
    assert "only existing native concept trajectories" in passport["source"]
    assert len(passport["operator_names"])==8


def test_native_shape_summary_gradients_and_exact_interventions():
    torch.manual_seed(102)
    model=build("learned_shape")
    x=torch.randn(12,24,5)
    mask=torch.ones_like(x)
    out=model(x,mask,process_mask=mask)
    target=torch.tensor([0.,1.]*6)
    loss=F.binary_cross_entropy_with_logits(out.logits,target)
    loss.backward()
    assert model.physical.temporal_readout_logits.grad is not None
    assert float(model.physical.temporal_readout_logits.grad.abs().sum()) > 0
    assert float(out.decomposition_error()) < 1e-6
    # The native intervention recomputes the same actual prediction with the
    # intervened summary; it is not a physiological causal claim.
    old_summary=out.physical.concept_summary[:,0]
    revised=torch.full_like(old_summary,.7)
    changed,_=model.physical.intervene(out.physical,0,revised)
    expected=(out.logits
              +(revised-old_summary)*model.physical.decision_weights()[0])
    torch.testing.assert_close(changed+out.observation_contributions.sum(-1),expected)


def test_absent_measurements_remain_physically_unknown_not_negative_risk():
    model=build("learned_shape")
    x=torch.randn(5,24,5)
    missing=torch.zeros_like(x)
    out=model(x,missing,process_mask=missing)
    neutral=.5*out.physical.gates.view(1,-1)
    torch.testing.assert_close(out.physical.concept_summary,neutral.expand(5,-1),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out.physical.contributions,torch.zeros_like(out.physical.contributions),atol=1e-6,rtol=1e-6)


def test_shape_feature_values_stay_bounded_and_neutral_on_constant_trajectory():
    model=build("learned_shape")
    gate=torch.ones(3)
    c=torch.full((4,24,3),.5)
    bank=model.physical.temporal_shape_bank(c,gate)
    assert bank.shape==(4,3,8)
    torch.testing.assert_close(bank,torch.full_like(bank,.5))
    changing=torch.rand(4,24,3)
    bank=model.physical.temporal_shape_bank(changing,gate)
    assert bool(torch.all((bank>=-1e-6)&(bank<=1+1e-6)))


def test_v17_new_slot_admission_preserves_existing_temporal_readout():
    torch.manual_seed(19)
    model=build("learned_shape")
    model.eval()
    with torch.no_grad():
        model.physical.temporal_readout_logits[:,2] += .5
    c=ProposedObservableConcept(
        candidate_id="measured-new",
        direction=np.array([0,0,0,1.,0]),
        train_episode_count=300,positive_lag_strength=.2,
        permutation_p=.005,bootstrap_abs_cosine_q10=.99,
        top_signed_sensors=(("sensor_3",1.),),
    )
    expanded=admit_candidate_without_prediction_drift(
        model,c,expert_decision="approve",
    )
    assert expanded.temporal_readout_mode=="learned_shape"
    torch.testing.assert_close(
        expanded.physical.temporal_readout_weights()[:3],
        model.physical.temporal_readout_weights(),
    )
    x=torch.randn(5,24,5)
    m=torch.ones_like(x)
    expanded.eval()
    with torch.no_grad():
        before=model(x,m,process_mask=m)
        after=expanded(x,m,process_mask=m)
    torch.testing.assert_close(before.logits,after.logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(before.physical.future_prediction,after.physical.future_prediction,atol=1e-6,rtol=1e-6)
    assert float(after.decomposition_error()) < 1e-6


def test_invalid_mode_abstains():
    with pytest.raises(ValueError,match="readout"):
        build("not_a_temporal_readout")
