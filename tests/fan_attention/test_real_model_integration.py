"""Runs against native models in the GitHub repository; skipped in mini-package."""
from __future__ import annotations

import pytest
import torch

from fan.concept.observability import ObservationAwareConceptFAN


def test_actual_multiset_native_all_observed_and_missing_explanation():
    mod = pytest.importorskip("fan.concept.temporal")
    Model = mod.MultiSetAdditiveTemporalConceptFANModel
    torch.manual_seed(45)
    core = Model(
        input_dim=6, sequence_length=8, latent_dim=16, n_concepts=3,
        n_memberships=3, membership="gaussian", oracle=False,
        temporal_mode="attention", dropout=0.0,
        encoder_layers=1, encoder_heads=2, encoder_ffn=32,
        alpha_mode="no_alpha",
    )
    core.eval()
    wrapper = ObservationAwareConceptFAN(core, torch.eye(3),
                                         min_observation_fraction=.1)
    values = torch.randn(2,8,3)
    present = torch.ones_like(values)
    with torch.no_grad():
        baseline=core(torch.cat([values,present],dim=-1))
        covered=wrapper(values,present)
    torch.testing.assert_close(covered.logit, baseline.logit, atol=1e-6, rtol=1e-6)
    assert not bool(covered.abstain_from_explanation.any())

    absent = torch.zeros_like(present)
    with torch.no_grad():
        unknown=wrapper(torch.full_like(values,float("nan")),absent)
    assert bool(unknown.abstain_from_explanation.all())
    assert bool((unknown.signed_decision_contributions==0).all())
    torch.testing.assert_close(unknown.logit,core.decision_head.bias.expand_as(unknown.logit),atol=1e-6,rtol=1e-6)
