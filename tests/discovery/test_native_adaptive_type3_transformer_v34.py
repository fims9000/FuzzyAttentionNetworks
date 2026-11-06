"""V34 exact in-graph native Temporal ConceptFAN fuzzy slot and human explanation."""
from __future__ import annotations

import numpy as np
import torch

from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import (
    FuzzySourceDefinition,
)
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import (
    NativeAuditedFuzzyConceptFANV34,readable_source_explanation,
)


def _native():
    core=make_firewalled_k2(123)
    definition=FuzzySourceDefinition(
        source_id="train_measured_sensors4_5_6",
        sensor_indices=(4,5,6),
        threshold_train=.10,width_train=.75,
        train_missingness_spread=.18,boundary_spread_anchor=.16,
        certified_calibration_environments=2,external_anchor_verified=True,
    )
    wrapper=NativeAuditedFuzzyConceptFANV34(
        core,np.array([2.,-3.]),-.15,definition,
        fuzzy_order="restricted_interval_type3")
    return wrapper


def test_original_native_transformer_forward_is_real_and_exactly_additive():
    model=_native().eval()
    assert any("encoder" in k for k in model.core.state_dict())
    rng=np.random.default_rng(123)
    x=torch.from_numpy(rng.normal(size=(12,16,4)).astype(np.float32))
    m=torch.from_numpy((rng.random(size=(12,16,4))>.2).astype(np.float32))
    amp=torch.tensor(np.linspace(-1,1,12),dtype=torch.float32)
    uncertainty=torch.full((12,),.15)
    info=torch.ones(12)
    sha_before=model.known_model_sha256()
    out=model(x,m,extra_source_amplitude=amp,
              extra_source_uncertainty=uncertainty,
              extra_source_observed_information=info,
              extra_source_gate_approved=True)
    assert out["known_concepts"].shape==(12,2)
    assert out["known_concept_trajectories"].shape==(12,16,2)
    assert out["known_temporal_attention"].shape==(12,16,2)
    torch.testing.assert_close(
        out["old_k2_logit"],
        -.15+2*out["known_concepts"][:,0]-3*out["known_concepts"][:,1],
    )
    torch.testing.assert_close(
        out["logit"],out["old_k2_logit"]+out["new_concept_contribution"])
    assert sha_before==model.known_model_sha256()
    assert not any(p.requires_grad for p in model.core.parameters())
    assert not out["physical_concept_identity_certified"]
    text=readable_source_explanation(out,0,
                 has_independent_external_device_calibration=True)
    assert text["source_sensor_channels"]==[4,5,6]
    assert "физический смысл" in text["explanation_in_plain_russian"]
    assert text["algebraic_interval_is_not_statistical_confidence"]


def test_no_group_observations_exact_old_K2_and_hidden_sensor_payload_ignored():
    model=_native().eval()
    with torch.no_grad():
        model.source_slot.gamma.fill_(2.)
    x=torch.randn(5,16,4)
    mask=torch.ones_like(x)
    first=model(x,mask,extra_source_amplitude=torch.ones(5),
                extra_source_uncertainty=torch.ones(5)*.2,
                extra_source_observed_information=torch.zeros(5),
                extra_source_gate_approved=True)
    torch.testing.assert_close(first["old_k2_logit"],first["logit"],rtol=0,atol=0)
    assert not first["new_concept_used"].any()
    second=model(x,mask,extra_source_amplitude=torch.ones(5),
                extra_source_uncertainty=torch.ones(5)*.2,
                extra_source_observed_information=torch.ones(5),
                extra_source_gate_approved=False)
    torch.testing.assert_close(second["old_k2_logit"],second["logit"],rtol=0,atol=0)
    hidden=x.clone()
    mask[:,5:9,:]=0
    hidden[:,5:9,:]=float("nan")
    z=model(hidden,mask,extra_source_amplitude=torch.ones(5),
            extra_source_uncertainty=torch.ones(5)*.2,
            extra_source_observed_information=torch.zeros(5))
    clean=model(torch.where(mask.bool(),hidden,torch.zeros_like(hidden)),mask,
                extra_source_amplitude=torch.ones(5),
                extra_source_uncertainty=torch.ones(5)*.2,
                extra_source_observed_information=torch.zeros(5))
    torch.testing.assert_close(z["logit"],clean["logit"],rtol=0,atol=0)


def test_gamma_gradient_updates_only_source_slot_while_old_transformer_frozen():
    model=_native()
    with torch.no_grad():
        model.source_slot.gamma.fill_(0.01)
    x=torch.randn(9,16,4)
    mask=torch.ones_like(x)
    before=model.known_model_sha256()
    opt=torch.optim.Adam(model.source_slot.parameters(),lr=.05)
    for _ in range(4):
        z=model(x,mask,extra_source_amplitude=torch.ones(9),
                extra_source_uncertainty=torch.ones(9)*.1,
                extra_source_observed_information=torch.ones(9),
                extra_source_gate_approved=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(
            z["logit"],torch.ones(9))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert model.known_model_sha256()==before
    assert float(model.source_slot.gamma)>0.01
