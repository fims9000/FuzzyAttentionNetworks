"""Design-aware seed inference: 10 initialization seeds x 3 data-order seeds."""
from __future__ import annotations
import itertools
import math
from typing import Hashable, Mapping, Sequence
import numpy as np

def exact_cluster_signflip(differences: np.ndarray, axis: int = 0) -> dict:
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 2 or not d.size or not np.all(np.isfinite(d)):
        raise ValueError("expected finite [initialization,order] differences")
    if axis not in (0,1):
        raise ValueError("axis must be 0 or 1")
    clusters = d.mean(axis=1 if axis==0 else 0)
    n = len(clusters)
    if n < 2 or n > 18:
        raise ValueError("2..18 clusters required")
    all_signs = np.asarray(list(itertools.product((-1., 1.), repeat=n)),dtype=np.float64)
    observed = abs(float(clusters.mean()))
    perm = np.abs(all_signs @ clusters / n)
    return {"n_clusters":n,
            "cluster_axis":"initialization" if axis==0 else "data_order",
            "mean_delta":float(d.mean()),
            "median_per_run_delta":float(np.median(d)),
            "two_sided_exact_p":float(np.count_nonzero(perm>=observed-1e-14)/len(perm)),
            "resamples":len(perm),
            "null_assumption":"symmetric independent paired effects across selected seed clusters"}

def holm(pvalues: list[float]) -> list[float]:
    if not pvalues:return []
    order=np.argsort(pvalues)
    result=np.empty(len(order),dtype=np.float64)
    last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(len(order)-rank)*float(pvalues[index])))
        result[index]=last
    return result.tolist()

def crossed_seed_bootstrap_ci(
    values: np.ndarray,
    model_pairs: Sequence[tuple[Hashable,Hashable]],
    model_seeds: Mapping[Hashable,tuple[Hashable,Hashable]],
    repetitions: int,
    rng: np.random.Generator,
    mode: str = "both",
) -> tuple[float,float]:
    values=np.asarray(values,dtype=np.float64)
    if values.ndim not in (1,2) or not values.size or values.shape[0]!=len(model_pairs) or not np.isfinite(values).all():
        raise ValueError("values must be finite [pairs] or [pairs,patients]")
    if mode not in ("both","init","order") or repetitions<2:
        raise ValueError("Invalid mode/repetitions")
    ids=sorted(set(x for p in model_pairs for x in p),key=str)
    if len(ids)<2 or len(model_pairs)!=math.comb(len(ids),2) or set(model_seeds)!=set(ids):
        raise ValueError("all distinct model pairs and matching seed-map required")
    idx={x:i for i,x in enumerate(ids)}
    edges=np.asarray([(idx[a],idx[b]) for a,b in model_pairs],dtype=int)
    if np.any(edges[:,0]==edges[:,1]) or len({tuple(sorted(p)) for p in edges})!=len(edges):
        raise ValueError("duplicate or self pair")
    initial=sorted({a for a,b in model_seeds.values()},key=str)
    order=sorted({b for a,b in model_seeds.values()},key=str)
    ilook={a:i for i,a in enumerate(initial)}
    olook={b:i for i,b in enumerate(order)}
    ix=np.asarray([ilook[model_seeds[k][0]] for k in ids],dtype=int)
    ox=np.asarray([olook[model_seeds[k][1]] for k in ids],dtype=int)
    if len(initial)*len(order)!=len(ids) or len({model_seeds[k] for k in ids})!=len(ids):
        raise ValueError("seed factors must form complete Cartesian grid")
    estimates=np.empty(repetitions)
    ni,no=len(initial),len(order)
    for b in range(repetitions):
        for retry in range(500):
            ci=np.bincount(rng.integers(ni,size=ni),minlength=ni) if mode in ("init","both") else np.ones(ni,dtype=int)
            co=np.bincount(rng.integers(no,size=no),minlength=no) if mode in ("order","both") else np.ones(no,dtype=int)
            count=ci[ix]*co[ox]
            weights=count[edges[:,0]]*count[edges[:,1]]
            total=weights.sum()
            if total:break
        else:raise RuntimeError("No distinct model cells resampled")
        if values.ndim==1:
            estimates[b]=weights@values/total
        else:
            patient_ids=rng.integers(values.shape[1],size=values.shape[1])
            patient_mean=(weights@values)/total
            estimates[b]=patient_mean[patient_ids].mean()
    return tuple(float(v) for v in np.quantile(estimates,[.025,.975]))
