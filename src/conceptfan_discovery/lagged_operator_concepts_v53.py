"""V53: TRAIN-only discovery of delayed observable fuzzy concepts.

A lag describes statistical temporal association, NOT a medical causal path.
Only observed TRAIN sensor values enter definition; no labels or latent oracle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class LaggedConceptV53:
    candidate_id: str
    source_channel: int
    readout_channel: int
    lag_steps: int
    readout_sign: int
    source_mean: float
    source_std: float
    readout_mean: float
    readout_std: float
    min_fold_signed_corr: float
    max_abs_zero_lag_corr: float
    train_subjects: int
    observable_definition: str = "lagged masked sensor-pair fuzzy activation"
    source_identity: str = "MEASUREMENT_ASSOCIATION_ONLY"
    medical_identity_verified: bool = False
    generator_oracle_used: bool = False
    outcome_labels_used: bool = False
    uses_future_measurements_at_time_t: bool = False

    def passport(self):
        return asdict(self)


@dataclass(frozen=True)
class LaggedDiscoveryV53:
    candidates: tuple[LaggedConceptV53, ...]
    unassigned_channels: tuple[int, ...]
    excluded_known_channels: tuple[int, ...]
    n_train: int
    max_candidates: int
    proposed_slots_can_have_different_k: bool = True
    trained_on_outcome_labels: bool = False
    statistically_identified_not_physical: bool = True

    def manifest(self):
        return asdict(self)


def _check(values, mask):
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask)
    if x.ndim != 3 or x.shape != m.shape or x.shape[1] < 8 or x.shape[2] < 4:
        raise ValueError("Expected [subjects,time,sensors] with matching availability")
    if not np.isin(m, [0, 1]).all():
        raise ValueError("Measurement mask must be binary")
    m = m.astype(bool)
    if not np.isfinite(x[m]).all():
        raise ValueError("Nonfinite observed measurement")
    return x, m


def _cross_corr(x, m, ids, source, readout, lag, minimum_cells):
    a = x[ids, :-lag, source]
    b = x[ids, lag:, readout]
    mm = m[ids, :-lag, source] & m[ids, lag:, readout]
    if int(mm.sum()) < minimum_cells:
        return None
    v = a[mm]; w = b[mm]
    v = v - v.mean(); w = w - w.mean()
    den = float(np.sqrt((v*v).sum() * (w*w).sum()))
    if den < 1e-12:
        return None
    return float(np.clip(np.dot(v, w)/den, -1., 1.))


def _zero_corr(x, m, ids, i, j, minimum_cells):
    mm = m[ids, :, i] & m[ids, :, j]
    if int(mm.sum()) < minimum_cells:
        return None
    v = x[ids, :, i][mm]
    w = x[ids, :, j][mm]
    v = v - v.mean(); w = w - w.mean()
    den = float(np.sqrt((v*v).sum()*(w*w).sum()))
    if den < 1e-12:
        return None
    return float(np.clip(np.dot(v,w)/den, -1., 1.))


def discover_lagged_concepts_v53(
    train_values, train_mask, *, known_channels=(0, 1, 2, 3),
    seed=20261113, max_lag=3, max_candidates=3,
    min_abs_lag_corr=.65, min_gain_over_instant=.30,
    min_fold_cells=120,
):
    """TRAIN-only 3 subject folds, disjoint channel pairs, no outcome labels."""
    x, m = _check(train_values, train_mask)
    n, t, d = x.shape
    if (n < 300 or not(1 <= max_lag <= min(6,t-2)) or
            not(1 <= max_candidates <= 8) or not(.4 <= min_abs_lag_corr <= .95)
            or not(0. <= min_gain_over_instant <= .8)
            or min_fold_cells < 30 or
            len(set(known_channels)) != len(known_channels) or
            any(z < 0 or z >= d for z in known_channels)):
        raise ValueError("Invalid predeclared discovery parameters or subjects")
    unknown = tuple(j for j in range(d) if j not in known_channels)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
    possibilities = []
    for i, j in combinations(unknown, 2):
        instant = [_zero_corr(x,m,ids,i,j,min_fold_cells) for ids in folds]
        if any(a is None for a in instant):
            continue
        instantaneous = max(abs(z) for z in instant)
        for source, readout in ((i,j),(j,i)):
            for lag in range(1,max_lag+1):
                v = [_cross_corr(x,m,ids,source,readout,lag,min_fold_cells)
                     for ids in folds]
                if any(z is None for z in v):
                    continue
                arr = np.asarray(v, float)
                if np.sign(arr).sum() not in (-3,3):
                    continue
                score = float(np.min(np.abs(arr)))
                if score < min_abs_lag_corr or score - instantaneous < min_gain_over_instant:
                    continue
                sign = int(np.sign(arr[0]))
                possibilities.append((score-instantaneous,score,source,
                                      readout,lag,sign,instantaneous))
    possibilities.sort(key=lambda p: (-p[0],-p[1],p[2],p[3],p[4]))
    taken = set(); chosen = []
    for delta,score,i,j,lag,sign,zero in possibilities:
        if i in taken or j in taken:
            continue
        stats = []
        for channel in (i,j):
            obs = x[:,:,channel][m[:,:,channel]]
            if len(obs) < min_fold_cells or not np.isfinite(obs).all():
                raise AssertionError("Source statistics unavailable")
            stats.append((float(obs.mean()),float(max(obs.std(),.10))))
        mid = sha256((f"{i}:{j}:{lag}:{sign}:"
                      f"{round(score,8)}:{round(zero,8)}").encode()).hexdigest()[:12]
        chosen.append(LaggedConceptV53(
            candidate_id=f"lagged-measured-{mid}",source_channel=i,
            readout_channel=j,lag_steps=lag,readout_sign=sign,
            source_mean=stats[0][0],source_std=stats[0][1],
            readout_mean=stats[1][0],readout_std=stats[1][1],
            min_fold_signed_corr=sign*score,max_abs_zero_lag_corr=zero,
            train_subjects=n,
        ))
        taken |= {i,j}
        if len(chosen) >= max_candidates:
            break
    return LaggedDiscoveryV53(tuple(chosen),
            unassigned_channels=tuple(j for j in unknown if j not in taken),
            excluded_known_channels=tuple(sorted(known_channels)),
            n_train=n,max_candidates=max_candidates)


def infer_lagged_concept_v53(candidate, values, mask, *, last_steps=6):
    """q[t] uses only source[t-lag] and readout[t]; no future-at-t leakage."""
    x, m = _check(values,mask)
    i,j,lag = (candidate.source_channel,candidate.readout_channel,
               candidate.lag_steps)
    if not(0 <= i < x.shape[2] and 0 <= j < x.shape[2] and
           1 <= lag < x.shape[1] and candidate.readout_sign in (-1,1)
           and candidate.source_std > 0 and candidate.readout_std > 0
           and 1 <= last_steps <= x.shape[1]):
        raise ValueError("Incompatible frozen lagged concept definition")
    n,t,_ = x.shape
    paired = m[:,:-lag,i]&m[:,lag:,j]
    a = np.where(m[:,:-lag,i],x[:,:-lag,i],candidate.source_mean)
    b = np.where(m[:,lag:,j],x[:,lag:,j],candidate.readout_mean)
    z = .5*((a-candidate.source_mean)/candidate.source_std +
            candidate.readout_sign*(b-candidate.readout_mean)/candidate.readout_std)
    c = np.full((n,t),.5,dtype=np.float64)
    c[:,lag:] = np.where(paired,expit(z),.5)
    reliable = np.zeros((n,t),dtype=bool)
    reliable[:,lag:] = paired
    recent = reliable[:,-last_steps:]
    observed = recent.sum(axis=1)
    mean = (c[:,-last_steps:]-.5).sum(axis=1)/np.maximum(observed,1)
    coverage = observed/last_steps
    summary = np.clip(.5+coverage*mean,0.,1.)
    return {"trajectory":c,"available":reliable,
            "summary":summary,"support":coverage,
            "not_a_physical_or_causal_concept":True}


def discovery_fingerprint_v53(discovery):
    return sha256(repr(discovery.manifest()).encode()).hexdigest()
