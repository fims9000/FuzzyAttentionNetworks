"""V37 explanation output must expose observations and limitations to humans."""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.threeview_explanation_v37 import (
    explain_threeview_source_v37,
)


def _source():
    return {
        "support":np.array([1.,0.,0.]),
        "pair_timepoints":np.array([[9,8,8],[8,0,0],[6,6,6]]),
        "all_three_pairs_measured":np.array([True,False,True]),
        "pairwise_max_membership_discrepancy":np.array([.15,.40,.60]),
        "pair_memberships":np.array([
            [.51,.57,.48],[.50,.50,.50],[.23,.50,.98],
        ]),
        "pair_energy":np.array([
            [1.3,1.2,1.4],[0.,0.,0.],[.2,.3,.4],
        ]),
    }


def test_human_explanation_uses_measured_view_ids_not_random_fuzzy_grades():
    d=_source()
    r=explain_threeview_source_v37(
        d,np.array([.20,.31,.41]),np.array([.28,.31,.41]),
        approved_by_two_valid_environments=True,
        trained_pair_discrepancy_limit=.37,index=0)
    assert r["source_was_used"]
    assert r["measured_channels"]["канал_C"]==[8,9]
    assert r["measured_joint_timesteps"]["AB"]==9
    assert r["change_in_percentage_points"]==8.
    assert "согласованные" in r["how_model_decided"]
    assert "помехи" in r["limitations"]
    assert "membership" not in r["how_model_decided"]
    assert not r.get("clinical_identity_certified",False)
    assert r["not_a_causal_intervention"]
    assert "technical_audit" in r


def test_missing_third_view_refuses_without_hallucination():
    r=explain_threeview_source_v37(
        _source(),np.array([.20,.31,.41]),
        np.array([.28,.31,.41]),
        approved_by_two_valid_environments=True,
        trained_pair_discrepancy_limit=.37,index=1)
    assert not r["source_was_used"]
    assert r["reason_code"]=="INSUFFICIENT_OBSERVED_PAIRED_TIMES"
    assert "не генерировались" in r["what_was_not_observed"]


def test_disagreement_is_explicitly_different_from_missing_evidence():
    r=explain_threeview_source_v37(
        _source(),np.array([.20,.31,.41]),
        np.array([.28,.31,.41]),
        approved_by_two_valid_environments=True,
        trained_pair_discrepancy_limit=.37,index=2)
    assert r["reason_code"]=="CROSS_VIEW_DISAGREEMENT"
    assert "противоречат" in r["how_model_decided"]


def test_valid_rejection_and_faithfulness_failing_closed():
    r=explain_threeview_source_v37(
        _source(),np.array([.20,.31,.41]),
        np.array([.20,.31,.41]),
        approved_by_two_valid_environments=False,
        trained_pair_discrepancy_limit=.37,index=0)
    assert r["reason_code"]=="FAILED_TWO_VALID_ENVIRONMENTS"
    with pytest.raises(AssertionError,match="Unsupported concept"):
        explain_threeview_source_v37(
            _source(),np.array([.20,.31,.41]),
            np.array([.27,.31,.41]),
            approved_by_two_valid_environments=False,
            trained_pair_discrepancy_limit=.37,index=0)


def test_explanation_rejects_unaligned_or_invalid_predictions():
    with pytest.raises(ValueError):
        explain_threeview_source_v37(
            _source(),np.array([.20,.31,.41]),
            np.array([.28,.31]),
            approved_by_two_valid_environments=True,
            trained_pair_discrepancy_limit=.37,index=0)
    with pytest.raises(ValueError):
        explain_threeview_source_v37(
            _source(),np.array([.20,.31,.41]),
            np.array([.28,.31,.41]),
            approved_by_two_valid_environments=True,
            trained_pair_discrepancy_limit=.37,index=3)
