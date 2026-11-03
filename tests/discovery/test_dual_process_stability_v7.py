"""Process concepts are native signed operators, not a posthoc proxy."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.observation_process_concepts_v7 import DualProcessConceptFAN
from conceptfan_discovery.dual_process_stability_v7 import collect_observation_process_evidence
from conceptfan_discovery.evaluation import compare_independent_discoveries


def _model():
    return DualProcessConceptFAN(
        observable_loadings=np.eye(8)[:,:3],
        channels=tuple(f"v{i}" for i in range(8)),
        forecast_horizon=4,n_observation_concepts=3,
    )


def test_process_audit_preserves_source_learned_beta_and_mask_only_formulas():
    torch.manual_seed(37)
    rng=np.random.default_rng(90)
    model=_model()
    m=(rng.random((110,36,8))>.40).astype(np.float32)
    y=np.asarray([0.,1.]*55,dtype=np.float32)
    evidence=collect_observation_process_evidence(model,m,y)
    assert evidence.metrics["source"]=="mask_only_care_process_concept_operators"
    assert evidence.trajectories.shape==(110,1,3)
    assert evidence.weights.shape==(3,64)
    np.testing.assert_allclose(
        evidence.decision_weights,
        model.process.direct_weights().detach().cpu().numpy(),
        atol=0,rtol=0,
    )
    assert np.isfinite(evidence.contributions).all()
    assert evidence.metrics["clinical_pathology_claim"] is False


def test_identical_model_has_positive_patient_specific_sign_agreement_when_varying():
    torch.manual_seed(19)
    rng=np.random.default_rng(11)
    model=_model()
    m=(rng.random((180,36,8))>.46).astype(np.float32)
    y=np.asarray([0.,1.]*90,dtype=np.float32)
    e=collect_observation_process_evidence(model,m,y)
    if e.metrics["n_active_and_nonconstant_slots"]<2:
        pytest.skip("Random model has fewer than two nonconstant concepts")
    result=compare_independent_discoveries(e,e)
    assert result["matched_slots"]>=2
    assert result["paired_contribution_sign_agreement"]==1.
    assert result["case_specific_excess_sign_agreement"]>=0.
    assert result["excess_over_unmatched_episode_null_spearman"]>=-1e-6


def test_process_evidence_rejects_bad_masks_and_single_class_cohort():
    model=_model()
    mask=np.ones((50,36,8),dtype=np.float32)
    with pytest.raises(ValueError):
        collect_observation_process_evidence(model,mask,np.zeros(50))
