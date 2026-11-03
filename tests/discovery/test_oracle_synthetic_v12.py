"""Synthetic oracle comparison is evaluation-only and optimistic."""
import numpy as np
import pytest

from conceptfan_discovery.oracle_synthetic_v12 import optimistic_oracle_match


def test_oracle_eval_optimistic_hungarian_and_permuted_case_null():
    rng=np.random.default_rng(52)
    true=rng.normal(size=(160,3))
    learned=true[:,[2,0,1]]+rng.normal(scale=.03,size=(160,3))
    report=optimistic_oracle_match(learned,true,seed=301)
    assert report["rank_matches_true_oracle_count"]
    assert report["observed_mean_abs_spearman_OPTIMISTIC"]>.97
    assert report["shuffled_patient_oracle_mean_abs_spearman_OPTIMISTIC_NULL"]<.20
    assert report["oracle_used_in_optimization"] is False
    assert report["assigned_true_factor_slots_evaluation_only"]==[2,0,1]


def test_oracle_eval_refuses_mismatched_heldout_cases_or_nonfinite():
    rng=np.random.default_rng(9)
    a=rng.normal(size=(60,4))
    with pytest.raises(ValueError):
        optimistic_oracle_match(a,rng.normal(size=(59,3)),seed=9)
    a[0,0]=float("nan")
    with pytest.raises(ValueError):
        optimistic_oracle_match(a,rng.normal(size=(60,3)),seed=9)


def test_rank_mismatch_must_not_be_called_true_k_recovery():
    rng=np.random.default_rng(12)
    true=rng.normal(size=(80,5))
    learned=true[:,:3]
    report=optimistic_oracle_match(learned,true,seed=44)
    assert not report["rank_matches_true_oracle_count"]
    assert len(report["matched_signed_spearman"])==3
    assert report["n_true_factors"]==5
    assert report["n_learned_slots"]==3


def test_seta_exploratory_reporting_cannot_shadow_cli_namespace():
    # The first full study correctly trained every model, but a local name
    # reused the CLI namespace 'a' for a NumPy effect vector just before
    # writing aggregate output. This explicit regression prevents recurrence.
    import ast
    import inspect
    from conceptfan_discovery.clinical_signed_effect_v12 import run
    tree=ast.parse(inspect.getsource(run))
    writes=[
        node for node in ast.walk(tree)
        if isinstance(node,ast.Name) and node.id=="a" and isinstance(node.ctx,ast.Store)
    ]
    assert not writes, "Runner CLI args cannot be overwritten by effect arrays"
