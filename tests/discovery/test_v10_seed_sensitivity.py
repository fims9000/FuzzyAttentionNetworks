import numpy as np
import pandas as pd
import pytest

from conceptfan_discovery.v10_seed_sensitivity import (
    audit_frozen_tables, exact_seed_sensitivity, METHODS, SCENARIOS, METRICS,
)


def test_three_seed_exact_two_sided_p_has_minimum_quarter():
    a=exact_seed_sensitivity(np.asarray([.01,.03,.06]))
    assert a["two_sided_exact_signflip_p"] == .25
    assert a["n_exhaustive_seed_bootstraps"]==27
    assert .01 <= a["model_only_seed_resample_ci95_low"]
    assert a["model_only_seed_resample_ci95_high"] <= .06
    assert not a["patients_resampled"]
    assert not a["confirmatory_inference"]


def test_zero_differences_have_p_one():
    a=exact_seed_sensitivity(np.zeros(3))
    assert a["two_sided_exact_signflip_p"]==1.
    assert a["model_only_seed_resample_ci95_low"]==0.


def test_invalid_seeds_rejected():
    with pytest.raises(ValueError):
        exact_seed_sensitivity(np.ones(6))


def _toy_tables():
    records=[]
    for method in METHODS:
        for scenario in SCENARIOS:
            for seed in (11,23,37):
                row={"method":method,"development_scenario":scenario,
                     "seed":seed}
                for metric in METRICS:
                    row[metric]=float(seed/1000+(0.05 if method!=METHODS[0] else 0))
                records.append(row)
    pairs=[]
    for method in METHODS:
        for family in ("phys","care"):
            for a,b in ((11,23),(11,37),(23,37)):
                pairs.append({"method":method,"concept_family":family,
                              "seed_a":a,"seed_b":b,
                              "definition_cosine_normalized":.8,
                              "excess_over_unmatched_episode_null_spearman":.2})
    return pd.DataFrame(records),pd.DataFrame(pairs)


def test_exhaustive_statistical_audit_complete_and_pair_dependency_preserved():
    r,p=_toy_tables()
    sensitivity,stability=audit_frozen_tables(r,p)
    assert len(sensitivity)==len(SCENARIOS)*2*len(METRICS)
    assert set(sensitivity.two_sided_exact_signflip_p)=={.25}
    assert stability["v10_multiscale"]["phys"]["n_overlapping_model_pairs"]==3
    assert stability["v10_multiscale"]["phys"]["pairwise_p_value_reported"] is False


def test_wrong_matched_cohort_or_duplicate_rejected():
    r,p=_toy_tables()
    with pytest.raises(ValueError):
        audit_frozen_tables(pd.concat([r,r.iloc[:1]]),p)
    with pytest.raises(ValueError):
        audit_frozen_tables(r,p.iloc[:-1])
