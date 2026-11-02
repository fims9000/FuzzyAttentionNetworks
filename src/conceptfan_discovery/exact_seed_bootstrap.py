"""Exact enumeration of five-seed model bootstrap for dependent all-pairs results.

A five-model run generates ten overlapping comparison pairs: treating ten
pairs as independent greatly understates uncertainty. Enumerate ALL
5**5 original-model bootstrap replicates (with replacement) exactly,
with common seed counts for each compared architecture.

This yields descriptive *model-only* uncertainty conditional on one fixed
synthetic generator and validation cohort; not a confirmatory p-value.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd


def all_pairs_model_bootstrap(
    pairs: pd.DataFrame,
    *,
    metric: str,
    methods: tuple[str, str],
    n_fits: int = 5,
) -> dict:
    """Paired bootstrap of methods B-A, using shared original seed identities."""
    if n_fits < 3 or n_fits > 7 or methods[0] == methods[1]:
        raise ValueError("Expected 3..7 source models and two different methods")
    req={"method","init_a","init_b",metric}
    if not req.issubset(pairs):
        raise ValueError("Pairwise table is missing required model IDs/metric")
    seeds=sorted(set(pairs.init_a.tolist()) | set(pairs.init_b.tolist()))
    if len(seeds)!=n_fits:
        raise ValueError(f"Expected {n_fits} fitted models, got {len(seeds)}")
    edges=list(combinations(seeds,2))
    score={}
    for method in methods:
        frame=pairs[pairs.method==method]
        if len(frame)!=len(edges):
            raise ValueError(f"{method}: expected complete pairs")
        mapping={}
        for row in frame.itertuples(index=False):
            a,b=row.init_a,row.init_b
            edge=tuple(sorted((a,b)))
            if a==b or edge in mapping:
                raise ValueError("Self-pair or duplicate trained-model pair")
            mapping[edge]=float(getattr(row,metric))
        if set(mapping)!=set(edges):
            raise ValueError("Missing model pairs, cannot infer from table means")
        score[method]=np.asarray([mapping[e] for e in edges],dtype=float)
        if not np.isfinite(score[method]).all():
            raise ValueError("All metrics must be observed and finite")

    observed=score[methods[1]]-score[methods[0]]
    idx={seed:i for i,seed in enumerate(seeds)}
    e=np.array([(idx[a],idx[b]) for a,b in edges],dtype=int)
    # Enumerate each ordered resample of length N. This is precisely the
    # multinomial N-draw-with-replacement distribution, including its
    # original multiplicity; do not treat unique count vectors as uniform.
    all_draws=np.asarray(list(product(range(n_fits),repeat=n_fits)),dtype=np.int32)
    count=np.zeros((len(all_draws),n_fits),dtype=np.int32)
    for j in range(n_fits):
        count[:,j]=(all_draws==j).sum(axis=1)
    weights=count[:,e[:,0]]*count[:,e[:,1]]
    denominator=weights.sum(axis=1)
    valid=denominator>0
    # All-equal resamples have no distinct original model pairs. They are
    # explicitly rejected; 5 out of 3125 when N=5.
    distribution=weights[valid]@observed/denominator[valid]
    ci=np.quantile(distribution,(.025,.975))
    loo=[]
    for omitted in range(n_fits):
        keep=np.array([omitted not in (i,j) for i,j in e])
        loo.append(float(observed[keep].mean()))
    return {
        "method_reference":methods[0],
        "method_comparator":methods[1],
        "metric":metric,
        "seed_ids":[int(x) if isinstance(x,(int,np.integer)) else str(x) for x in seeds],
        "mean_paired_difference":float(observed.mean()),
        "model_only_bootstrap_ci95":[float(x) for x in ci],
        "n_original_models":n_fits,
        "overlapping_original_pairs":len(edges),
        "total_ordered_resamples":len(all_draws),
        "valid_distinct_model_resamples":int(valid.sum()),
        "rejected_all_same_model_resamples":int((~valid).sum()),
        "leave_one_original_model_out_deltas":loo,
        "uncertainty_scope":"model-identity-only, fixed synthetic generator and fixed validation patients",
        "claim_limit":"descriptive exact bootstrap enumeration; not independent pairs, no p-value",
    }


def exact_paired_run_bootstrap(
    per_run:pd.DataFrame,*,metric:str,methods:tuple[str,str],n_fits:int=5,
)->dict:
    """Paired mean outcome metric resampling by original TRAIN seed."""
    required={"method","init_seed",metric}
    if not required.issubset(per_run):
        raise ValueError("Run-level source missing method/seed/metric")
    data={}
    for method in methods:
        part=per_run[per_run.method==method]
        if len(part)!=n_fits or part.init_seed.nunique()!=n_fits:
            raise ValueError("Wrong number of independently seeded runs")
        data[method]={row.init_seed:float(getattr(row,metric)) for row in part.itertuples(index=False)}
    if set(data[methods[0]])!=set(data[methods[1]]):
        raise ValueError("Paired seed identity mismatch")
    seeds=sorted(data[methods[0]])
    d=np.asarray([data[methods[1]][k]-data[methods[0]][k] for k in seeds])
    if not np.isfinite(d).all():
        raise ValueError("Nonfinite outcome")
    draws=np.asarray(list(product(range(n_fits),repeat=n_fits)))
    distribution=d[draws].mean(axis=1)
    return {
        "metric":metric,
        "method_reference":methods[0],
        "method_comparator":methods[1],
        "mean_paired_difference":float(d.mean()),
        "model_only_bootstrap_ci95":[float(x) for x in np.quantile(distribution,(.025,.975))],
        "n_independent_init_seeds":n_fits,
        "resamples_enumerated":len(draws),
        "uncertainty_scope":"paired model initialization only; same synthetic generator and validation patients",
        "claim_limit":"exploratory model bootstrap, not clinical or causal significance",
    }
