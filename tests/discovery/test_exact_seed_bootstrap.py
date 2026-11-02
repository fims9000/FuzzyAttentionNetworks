"""Exact 5-model bootstrap unit of inference is TRAINED MODELS, not ten pairs."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from conceptfan_discovery.exact_seed_bootstrap import (
    all_pairs_model_bootstrap, exact_paired_run_bootstrap,
)


def _source(metric_name:str="case_specific_excess_sign_agreement")->pd.DataFrame:
    rows=[]
    for method in ("free","shared"):
        for a,b in combinations(range(5),2):
            effect=(.02*(a+b) if method=="free" else .4+.03*(a+b))
            rows.append({"method":method,"init_a":a,"init_b":b,metric_name:effect})
    return pd.DataFrame(rows)


def test_exact_bootstrap_enumerates_model_resamples_not_pair_samples():
    result=all_pairs_model_bootstrap(
        _source(),metric="case_specific_excess_sign_agreement",
        methods=("free","shared"),
    )
    assert result["overlapping_original_pairs"]==10
    assert result["total_ordered_resamples"]==3125
    assert result["rejected_all_same_model_resamples"]==5
    assert result["valid_distinct_model_resamples"]==3120
    assert result["mean_paired_difference"]==pytest.approx(.42,abs=1e-12)
    assert result["model_only_bootstrap_ci95"][0]>.37
    assert result["model_only_bootstrap_ci95"][1]<.48
    assert len(result["leave_one_original_model_out_deltas"])==5


def test_pair_duplicate_or_missing_raises_instead_of_inventing_precision():
    frame=_source()
    with pytest.raises(ValueError):
        all_pairs_model_bootstrap(
            frame.iloc[:-1],metric="case_specific_excess_sign_agreement",
            methods=("free","shared"),
        )
    with pytest.raises(ValueError):
        all_pairs_model_bootstrap(
            frame,metric="does_not_exist",methods=("free","shared"),
        )


def test_paired_outcome_bootstrap_uses_original_seed_identity():
    rows=[]
    for method in ("free","shared"):
        for seed in range(5):
            rows.append({
                "method":method,"init_seed":seed,
                "outer_auprc":.4+.01*seed + (.07 if method=="shared" else 0.),
            })
    result=exact_paired_run_bootstrap(
        pd.DataFrame(rows),metric="outer_auprc",methods=("free","shared"),
    )
    np.testing.assert_allclose(result["model_only_bootstrap_ci95"],[.07,.07],atol=1e-12)
    assert result["resamples_enumerated"]==3125
    assert result["n_independent_init_seeds"]==5


def test_paired_run_missing_seed_is_not_a_valid_comparison():
    rows=[{"method":"a","init_seed":i,"outer_auprc":.5} for i in range(5)]
    rows += [{"method":"b","init_seed":i,"outer_auprc":.6} for i in range(1,6)]
    with pytest.raises(ValueError,match="Paired seed identity"):
        exact_paired_run_bootstrap(pd.DataFrame(rows),metric="outer_auprc",
                                    methods=("a","b"))
