"""TRAIN-only, label-free source grouping from incomplete measured time series.

This extends V17/V22's source-relevance separation to *masked* multicandidate
observations; correlation grouping is established, not a novel identification
theorem. Group identity is an observational source only: sensor artifacts and
physiology may be statistically indistinguishable. K2 physical meaning or
external device calibration is NEVER inferred by this code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class MeasuredGroup:
    candidate_id: str
    sensor_indices: tuple[int, ...]
    sensor_signs: tuple[int, ...]
    train_means: tuple[float, ...]
    train_stds: tuple[float, ...]
    min_fold_abs_corr: float
    train_episodes: int
    concept_type: str = "MEASUREMENT_ASSOCIATION_ONLY"
    target_labels_used_to_form: bool = False
    true_latent_concepts_used_to_form: bool = False
    physical_semantics_certified: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProposalManifest:
    candidates: tuple[MeasuredGroup, ...]
    excluded_known_sensors: tuple[int, ...]
    searched_sensors: tuple[int, ...]
    ungrouped_sensors: tuple[int, ...]
    ambiguous_components: tuple[tuple[int, ...], ...]
    train_subject_count: int
    min_pair_abs_correlation: float
    max_candidate_count: int
    candidate_groups_truncated: bool
    train_only: bool = True
    outcomes_used_for_grouping: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _check(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask)
    # Grouping requires two observed channels, not six. The old >=6 guard
    # was a synthetic benchmark dimension, not a mathematical condition.
    if a.ndim != 3 or a.shape != m.shape or a.shape[1] < 8 or a.shape[2] < 2:
        raise ValueError("Expected aligned [episode,time,channels] arrays")
    if not np.logical_or(m == 0, m == 1).all():
        raise ValueError("Availability must be binary (0 or 1)")
    mm = m.astype(bool)
    if not np.isfinite(a[mm]).all():
        raise ValueError("Nonfinite actually observed sensor reading")
    return a, mm


def _pair_corr(x: np.ndarray, mask: np.ndarray, ids: np.ndarray,
               first: int, second: int, minimum_cells: int) -> float | None:
    seen = mask[ids, :, first] & mask[ids, :, second]
    n = int(seen.sum())
    if n < minimum_cells:
        return None
    a = x[ids, :, first][seen]
    b = x[ids, :, second][seen]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if den < 1e-9:
        return None
    return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def form_train_measured_groups(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    *,
    known_sensor_indices: tuple[int, ...],
    seed: int = 20267204,
    min_pair_abs_corr: float = 0.60,
    min_joint_cells_per_fold: int = 300,
    max_candidates: int = 5,
) -> ProposalManifest:
    """Three disjoint patient folds must corroborate signed association.

    Only unlabeled TRAIN data; no ground-truth source identity, outcomes,
    VALID or TEST measurements. Singletons and sign-inconsistent components
    ABSTAIN rather than manufacture a multisensor concept.
    """
    x, m = _check(train_values, train_mask)
    n, _, d = x.shape
    if n < 300 or len(set(known_sensor_indices)) != len(known_sensor_indices):
        raise ValueError("Insufficient TRAIN subjects or duplicated known channels")
    if any(i < 0 or i >= d for i in known_sensor_indices):
        raise ValueError("Known-concept sensor index out of range")
    if not (.30 <= min_pair_abs_corr <= .95 and
            min_joint_cells_per_fold >= 50 and 1 <= max_candidates <= 5):
        raise ValueError("Invalid frozen proposal stability parameters")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), 3)
    unknown = tuple(i for i in range(d) if i not in known_sensor_indices)
    edges: dict[tuple[int, int], tuple[int, float]] = {}
    for i, j in combinations(unknown, 2):
        correlations = [_pair_corr(x, m, ids, i, j, min_joint_cells_per_fold)
                        for ids in folds]
        if any(v is None for v in correlations):
            continue
        vals = np.asarray(correlations, dtype=np.float64)
        signs = np.sign(vals)
        if (np.min(np.abs(vals)) >= min_pair_abs_corr
                and abs(np.sum(signs)) == 3):
            edges[i, j] = (int(signs[0]), float(np.min(np.abs(vals))))

    graph = {i: set() for i in unknown}
    for i, j in edges:
        graph[i].add(j)
        graph[j].add(i)
    components = []
    visited = set()
    for i in unknown:
        if i in visited:
            continue
        q = [i]
        visited.add(i)
        comp = []
        while q:
            u = q.pop()
            comp.append(u)
            for v in sorted(graph[u]):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        components.append(tuple(sorted(comp)))

    proposals = []
    rejected = []
    singletons = []
    for comp in components:
        if len(comp) < 2:
            singletons.extend(comp)
            continue
        ref = comp[0]
        orientation = {ref: 1}
        for j in comp[1:]:
            e = edges.get((ref, j))
            if e is not None:
                orientation[j] = e[0]
        complete = (len(orientation) == len(comp)
                    and all((i, j) in edges
                            and edges[(i, j)][0] == orientation[i]*orientation[j]
                            for i, j in combinations(comp, 2)))
        if not complete:
            rejected.append(comp)
            continue
        means, scales = [], []
        for j in comp:
            v = x[:, :, j][m[:, :, j]]
            if len(v) < 150 or not np.isfinite(v).all():
                complete = False
                break
            means.append(float(v.mean()))
            scales.append(float(max(v.std(ddof=0), 0.10)))
        if not complete:
            rejected.append(comp)
            continue
        reliability = min(edges[p][1] for p in combinations(comp, 2))
        signs = tuple(int(orientation[i]) for i in comp)
        digest = sha256((str(comp)+str(signs)+repr(round(reliability, 7)))
                        .encode("utf-8")).hexdigest()[:12]
        proposals.append(MeasuredGroup(
            candidate_id=f"measured-group-{digest}",
            sensor_indices=comp,
            sensor_signs=signs,
            train_means=tuple(means),
            train_stds=tuple(scales),
            min_fold_abs_corr=float(reliability),
            train_episodes=n,
        ))
    proposals.sort(key=lambda z: (-z.min_fold_abs_corr, z.sensor_indices))
    return ProposalManifest(
        candidates=tuple(proposals[:max_candidates]),
        excluded_known_sensors=tuple(sorted(known_sensor_indices)),
        searched_sensors=unknown,
        ungrouped_sensors=tuple(sorted(singletons)),
        ambiguous_components=tuple(rejected),
        train_subject_count=n,
        min_pair_abs_correlation=min_pair_abs_corr,
        max_candidate_count=max_candidates,
        candidate_groups_truncated=(len(proposals) > max_candidates),
    )


def measured_group_concept(
    source: MeasuredGroup, values: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded observed-value-only candidate and per-patient support.

    A completely absent group produces exactly q=0.5 and availability=0.
    No hidden payloads, classifier outcomes or oracle concepts enter.
    """
    x, m = _check(values, mask)
    cols = np.asarray(source.sensor_indices, dtype=np.int64)
    if len(cols) < 2 or int(cols.max()) >= x.shape[-1]:
        raise ValueError("Candidate source sensor map incompatible with sample")
    observed = m[:, :, cols]
    sensor = np.where(observed, x[:, :, cols], 0.0)
    norm = (sensor - np.asarray(source.train_means)[None, None, :]) / (
        np.asarray(source.train_stds)[None, None, :])
    signed = np.asarray(source.sensor_signs)[None, None, :]
    measured = np.where(observed, norm*signed, 0.0)
    count = observed.sum(axis=(1, 2))
    score = measured.sum(axis=(1, 2)) / np.maximum(count, 1)
    support = count / (x.shape[1]*len(cols))
    q = 0.5 + support*(expit(score)-0.5)
    if (not np.isfinite(q).all() or not np.logical_and(q >= 0, q <= 1).all()):
        raise AssertionError("Invalid measured candidate activation")
    return q, support


def group_provenance_fingerprint(manifest: ProposalManifest) -> str:
    return sha256(repr(manifest.as_dict()).encode("utf-8")).hexdigest()
