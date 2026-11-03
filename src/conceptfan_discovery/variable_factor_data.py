"""Variable-ground-truth-dimension dynamical benchmark for concept discovery.

Unlike Med-CircuitBench's hard-coded five states, this independent generator
constructs 3, 5 or 7 interacting latent processes. Oracle latent trajectories
are sealed from training and enter the *external evaluator only*.
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np

from med_circuitbench.benchmark.generator import split_ids
from .data import DiscoveryData, DiscoverySplit


def generate_variable_factor_data(
    *,
    n_true_factors: int,
    seed: int,
    split_seed: int,
    n_samples: int = 960,
    n_channels: int = 8,
    input_steps: int = 36,
    horizon: int = 6,
) -> DiscoveryData:
    if n_true_factors not in (3, 5, 7) or n_channels < n_true_factors:
        raise ValueError("Predeclared ground-truth count must be in {3,5,7}, with >=K observed channels")
    if n_samples < 200 or input_steps < 8 or horizon < 1:
        raise ValueError("Insufficient observations or temporal horizon")
    rng = np.random.default_rng(int(seed))
    k, n, steps = int(n_true_factors), int(n_samples), input_steps + horizon
    # Reproducible stable, sparse signed latent coupling. Distinct K changes
    # model identifiability, not only arbitrary label names.
    coupling = np.eye(k) * 0.58
    edges = rng.random((k,k)) < .22
    np.fill_diagonal(edges, False)
    off = rng.uniform(-0.13, .13, size=(k,k)) * edges
    coupling += off
    radius = max(abs(np.linalg.eigvals(coupling)))
    coupling *= min(1., .84 / max(radius, 1e-8))
    mixing = np.zeros((n_channels,k), dtype=np.float64)
    for j in range(k):
        mixing[j,j] = rng.uniform(.72,1.05)
    mixing += rng.normal(scale=.11,size=mixing.shape)
    z = np.zeros((n,steps,k),dtype=np.float64)
    z[:,0] = rng.normal(scale=.35,size=(n,k))
    treatment = rng.binomial(1,.045,size=(n,steps,k)).astype(float)
    for t in range(1,steps):
        shocks = rng.normal(scale=.055,size=(n,k)) + (treatment[:,t]-.045)*.35
        z[:,t] = np.tanh(z[:,t-1] @ coupling.T + shocks)
    raw = z @ mixing.T + rng.normal(scale=.11,size=(n,steps,n_channels))
    # Missingness is independent of the latent ground truth in this control
    # benchmark. The clinical dataset has informative missingness.
    mask = (rng.random(size=raw.shape) < .83).astype(np.float32)
    target_beta = np.linspace(.1,.8,k) * np.where(np.arange(k)%2==0,1.,-1.)
    risk_logit = -0.35 + (z[:,-horizon:].mean(axis=1) * target_beta[None,:]).sum(axis=1)*2.0
    risk_probability = 1./(1.+np.exp(-risk_logit))
    y = (rng.random(n) < risk_probability).astype(np.float32)
    splits = split_ids(n,int(split_seed))
    train_ids = np.asarray(splits["train"],dtype=int)
    validation_ids = np.asarray(splits["validation"],dtype=int)
    test_ids = np.asarray(splits["test"],dtype=int)
    x_train = np.where(mask[train_ids,:input_steps]>0,raw[train_ids,:input_steps],np.nan)
    median = np.nanmedian(x_train,axis=(0,1))
    scale = np.maximum(
        np.nanquantile(x_train,.75,axis=(0,1))-np.nanquantile(x_train,.25,axis=(0,1)),
        1e-3,
    )
    if not np.isfinite(median).all() or not np.isfinite(scale).all():
        raise FloatingPointError("Invalid train-only normalizer")
    def build(ids:np.ndarray)->DiscoverySplit:
        hist=raw[ids,:input_steps]
        mh=mask[ids,:input_steps]
        fut=raw[ids,input_steps:input_steps+horizon]
        mf=mask[ids,input_steps:input_steps+horizon]
        x=np.where(mh>0,np.clip((hist-median)/scale,-6,6),0).astype(np.float32)
        future=np.where(mf>0,np.clip((fut-median)/scale,-6,6),0).astype(np.float32)
        return DiscoverySplit(
            x=x,mask=mh.astype(np.float32),
            future=future,future_mask=mf.astype(np.float32),
            y=y[ids],oracle_states=z[ids,:input_steps].astype(np.float32),
            episode_ids=ids.astype(np.int64),
        )
    train,validation=build(train_ids),build(validation_ids)
    if np.unique(train.y).size<2 or np.unique(validation.y).size<2:
        raise ValueError("Cannot learn binary outcome without two classes")
    names=[f"sensor_{j+1}" for j in range(n_channels)]
    receipt={
        "dataset":"Variable-factor synthetic dynamical system, fully generated",
        "n_true_factors_hidden_from_optimizer":k,
        "n_channel_measurements":n_channels,
        "channel_names":names,
        "n_total":n,"train":len(train_ids),"validation":len(validation_ids),
        "test_ids_sealed":len(test_ids),
        "train_ids_sha256":sha256(np.ascontiguousarray(train_ids).tobytes()).hexdigest(),
        "validation_ids_sha256":sha256(np.ascontiguousarray(validation_ids).tobytes()).hexdigest(),
        "test_ids_sha256":sha256(np.ascontiguousarray(test_ids).tobytes()).hexdigest(),
        "generator_seed":int(seed),"split_seed":int(split_seed),
        "train_only_median":median.tolist(),"train_only_scale":scale.tolist(),
        "latent_transition_sha256":sha256(np.ascontiguousarray(coupling).tobytes()).hexdigest(),
        "observable_mixing_sha256":sha256(np.ascontiguousarray(mixing).tobytes()).hexdigest(),
        "latent_ground_truth_used_for_training":False,
        "masking_depends_on_latent_state":False,
        "test_inspected_for_model_selection":False,
        "samplewise_positives_train":int(train.y.sum()),
        "samplewise_positives_validation":int(validation.y.sum()),
    }
    return DiscoveryData(train,validation,receipt,median,scale)
