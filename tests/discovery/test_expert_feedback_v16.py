"""V16 regression tests: all annotations are TRAIN-only and act on native V10."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from conceptfan_discovery.expert_feedback_v16 import (
    ExpertReview, prepare_feedback, native_expert_loss, safe_feedback_manifest,
)
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


@dataclass
class MockOutput:
    raw_activation: torch.Tensor
    availability: torch.Tensor


def test_unknown_artifact_and_overlap_conflicts_do_not_turn_into_negatives():
    reviews = [
        ExpertReview(1, 0, 2, 8, "yes"),
        ExpertReview(1, 0, 2, 8, "yes", .6, "second"),
        ExpertReview(2, 0, 2, 8, "no"),
        ExpertReview(3, 0, 2, 8, "unsure"),
        ExpertReview(4, 0, 2, 8, "artifact"),
        ExpertReview(5, 0, 2, 8, "yes"),
        ExpertReview(5, 0, 6, 10, "no"),
    ]
    prepared = prepare_feedback(reviews, range(1, 6), n_slots=2, n_steps=12)
    assert prepared.audit == {
        "received":7,"accepted":2,"uncertain":1,"artifact":1,
        "conflicting":2,"zero_confidence":0,"merged_duplicates":1,
    }
    assert sorted(x[4] for x in prepared.accepted) == [0., 1.]
    assert safe_feedback_manifest(prepared)["individual_records_exported"] is False


def test_feedback_must_be_training_only():
    with pytest.raises(ValueError, match="validation/test"):
        prepare_feedback(
            [ExpertReview(999, 0, 0, 5, "yes")], [1, 2],
            n_slots=2, n_steps=12,
        )


def test_empty_feedback_has_zero_gradient_and_never_negative():
    p = prepare_feedback(
        [ExpertReview(1, 0, 2, 5, "unsure"),
         ExpertReview(2, 0, 2, 5, "artifact")],
        [1, 2], n_slots=2, n_steps=12,
    )
    logit = torch.nn.Parameter(torch.zeros(2, 12, 2))
    out = MockOutput(torch.sigmoid(logit), torch.ones_like(logit))
    loss, stats = native_expert_loss(out, torch.tensor([1, 2]), p)
    loss.backward()
    assert float(loss.detach()) == 0.
    assert stats["supervised_slots"] == 0
    assert float(logit.grad.abs().sum()) == 0.


def test_unmeasured_interval_does_not_mean_physiological_absence():
    p = prepare_feedback(
        [ExpertReview(1, 0, 2, 5, "yes")], [1], n_slots=2, n_steps=12,
    )
    logit = torch.nn.Parameter(torch.ones(1, 12, 2))
    out = MockOutput(torch.sigmoid(logit), torch.zeros_like(logit))
    loss, stats = native_expert_loss(out, torch.tensor([1]), p)
    loss.backward()
    assert stats["unsupported_intervals"] == 1
    assert float(logit.grad.abs().sum()) == 0.


def test_native_v10_training_graph_receives_review_gradient():
    torch.manual_seed(131)
    basis = np.eye(5, 3, dtype=np.float64)
    model = MultiScaleDualProcessConceptFAN(
        observable_loadings=basis,
        channels=tuple(f"sensor_{j}" for j in range(5)),
        forecast_horizon=3,
        n_observation_concepts=2,
        adaptive_feature_gradients=True,
    )
    x = torch.randn(5, 20, 5)
    mask = torch.ones_like(x)
    output = model(x, mask, process_mask=mask)
    p = prepare_feedback(
        [ExpertReview(11, 0, 14, 20, "yes"),
         ExpertReview(12, 0, 14, 20, "no")],
        [11, 12, 13, 14, 15], n_slots=3, n_steps=20,
    )
    loss, stats = native_expert_loss(
        output.physical, torch.tensor([11, 12, 13, 14, 15]), p,
    )
    loss.backward()
    assert stats["supervised_slots"] == 1
    assert model.physical.feature_logits.grad is not None
    assert float(model.physical.feature_logits.grad.abs().sum()) > 0
    assert model.physical.scale_logits.grad is not None
    assert float(model.physical.scale_logits.grad.abs().sum()) > 0
    assert float(output.decomposition_error().detach()) < 1e-6
    assert float(model.physical.expert_exploration_fraction) == .04 or abs(float(model.physical.expert_exploration_fraction)-.04)<1e-6


def test_wrong_expert_can_harm_and_is_not_auto_certified():
    theta = torch.nn.Parameter(torch.tensor([0.0]))
    raw = torch.sigmoid(theta).view(1, 1, 1).expand(1, 12, 1)
    p = prepare_feedback(
        [ExpertReview(1, 0, 3, 7, "no")], [1], n_slots=1, n_steps=12,
    )
    loss, _ = native_expert_loss(
        MockOutput(raw, torch.ones_like(raw)), torch.tensor([1]), p,
    )
    loss.backward()
    assert theta.grad.item() > 0
    assert safe_feedback_manifest(p)["clinical_validity_established"] is False


def test_real_fit_multiscale_accepts_review_and_preserves_bottleneck():
    from conceptfan_discovery.variable_factor_data import generate_variable_factor_data
    from conceptfan_discovery.clinical_multiscale_v10 import fit_multiscale
    data = generate_variable_factor_data(
        n_true_factors=3, seed=291, split_seed=103, n_samples=240,
        n_channels=5, input_steps=20, horizon=3,
    )
    training = {
        "training": {
            "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
            "batch_order_seed": 101, "epochs": 2,
            "main_mask_fraction": .2, "gradient_clip": 2.,
            "spectral_anchor_weight": .1,
            "spectral_axis_independence_weight": .02,
            "loss_weights": {
                "future": .5, "dynamics": .2, "mdl": .007, "support": .008,
                "redundancy": .08, "nonconstant": .25,
                "transition_l1": .003, "denoising": 1.,
            },
        },
    }
    dual = {"dual_head": {
        "n_observation_process_concepts": 2,
        "signed_mask_coefficient_l1": .03,
        "mask_definition_redundancy": .01,
    }}
    scale = {
        "training": {"scale_entropy_weight": .002},
        "architecture": {"scales_hours": [1, 4, 12]},
    }
    chosen = int(data.train.episode_ids[0])
    feedback = prepare_feedback(
        [ExpertReview(chosen, 0, 14, 20, "yes")],
        data.train.episode_ids, n_slots=3, n_steps=20,
    )
    basis = np.eye(5, 3, dtype=np.float64)
    model, receipt = fit_multiscale(
        data, basis, 7, training, dual, scale,
        consistency=False, expert_feedback=feedback,
        expert_feedback_weight=.5,
    )
    assert receipt["expert_feedback_training_enabled"]
    assert receipt["expert_feedback_supervised_batches"] >= 1
    values = torch.from_numpy(data.validation.x[:8])
    mask = torch.from_numpy(data.validation.mask[:8])
    model.eval()
    with torch.no_grad():
        out = model(values, mask, process_mask=mask)
    assert float(out.decomposition_error()) < 1e-6


def test_reliability_roundoff_tolerated_but_real_invalid_refused():
    p = prepare_feedback(
        [ExpertReview(1, 0, 2, 5, "yes")], [1], n_slots=2, n_steps=12,
    )
    raw = torch.full((1, 12, 2), .25, requires_grad=True)
    rounded = torch.full_like(raw, 1.0 + 2e-7)
    loss, stats = native_expert_loss(
        MockOutput(raw, rounded), torch.tensor([1]), p,
    )
    loss.backward()
    assert stats["supervised_slots"] == 1
    assert float(raw.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match="reliability"):
        native_expert_loss(
            MockOutput(raw, torch.full_like(raw, 1.01)),
            torch.tensor([1]), p,
        )
