"""V9: reproduce a COMMON observable mask-process subspace without pretending
the unstable individual V8 ranks or eigenvectors define clinical concepts.

The source evidence is SIX independently fitted TRAIN-half observation-mask
subspaces (two disjoint patient groups under three prespecified observation
policies). Their patient policies are nested stresses on the SAME cases, not
six independent hospitals. The expected confidence of the mean projector
is not a bootstrap CI; only deterministic agreement criteria are claimed.

For U_j in R^{D x K_j}, define the orthogonal projector P_j = U_j U_j^T.
The average A = mean_j P_j is invariant to any rotation within each
subspace. The source-independent directions are eigenvectors v of A
with eigenvalue >=.95 AND min_j ||U_j^T v||^2 >=.90.
This yields a conservative *observable common-core dimension*, not the
total number of clinical or care-process causes.

To make actual TRAIN-TIME concept axes deterministic rather than arbitrary
rotations of a degenerate eigenspace, project the predefined mask-operator
coordinate vectors onto the common subspace, then use pivoted Gram-Schmidt.
Every resulting axis is a mathematical combination of named measured-mask
operators and is also exposed through a signed sparsemax dictionary.

Do not present parameter anchoring as proof of patient-specific agreement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .care_process_rank_v8 import _positive_negative_template


@dataclass(frozen=True)
class StableMaskSubspace:
    k_common: int
    rank_by_source: tuple[int,...]
    projector_eigenvalues: np.ndarray
    mean_projector_eigenvectors: np.ndarray
    minimum_source_energy_per_eigenvector: np.ndarray
    stable_orthonormal_operators: np.ndarray
    signed_sparse_operator_definitions: np.ndarray
    selected_named_operator_indices: tuple[int,...]
    selection_residual_margins: tuple[float,...]
    source_ids: tuple[str,...]
    primary_projector_threshold: float
    minimum_source_energy_threshold: float
    sensitivity_counts: dict[str,int]
    no_shared_patient_outcomes_or_oracle: bool

    def receipt(self)->dict:
        return {
            "status":"OBSERVATION_COMMON_SUBSPACE_ONLY" if self.k_common
                     else "ABSTAIN_NO_COMMON_OBSERVATION_SUBSPACE",
            "k_observable_common_core":int(self.k_common),
            "source_ids":list(self.source_ids),
            "ranks_of_source_half_sample_subspaces":list(self.rank_by_source),
            "mean_projector_eigenvalues":self.projector_eigenvalues.tolist(),
            "min_source_energy_per_mean_projector_axis":
                self.minimum_source_energy_per_eigenvector.tolist(),
            "stable_operator_basis":self.stable_orthonormal_operators.tolist(),
            "positive_negative_operator_simplex":
                self.signed_sparse_operator_definitions.tolist(),
            "canonical_named_operator_pivots":list(self.selected_named_operator_indices),
            "canonical_coordinate_projection_margins":
                list(self.selection_residual_margins),
            "projector_threshold":self.primary_projector_threshold,
            "minimum_individual_source_support":self.minimum_source_energy_threshold,
            "threshold_sensitivity_counts":self.sensitivity_counts,
            "source_concepts_or_patients_independent":False,
            "uses_any_outcome_labels":False,
            "uses_planted_latent_factors_or_expert_concept_labels":False,
            "claim_limit":"stable linear subspace under fixed measurement regimes; NOT uniquely identifiable axes, actual medical concept count, causal processes or generalizable clinical physiology",
        }


def find_common_mask_subspace(
    learned_sources:dict[str,np.ndarray],
    *,
    projector_threshold:float=.95,
    min_source_energy:float=.90,
    sensitivity_levels:tuple[float,...]=(.85,.90,.95,.98),
)->StableMaskSubspace:
    if not 0.5<=projector_threshold<=1 or not 0.5<=min_source_energy<=1:
        raise ValueError("Invalid predeclared reproducibility thresholds")
    if len(learned_sources)<4:
        raise ValueError("Need at least four independently trained source subspaces")
    names=tuple(sorted(learned_sources))
    dimensions=[]
    projectors=[]
    dimension=None
    for name in names:
        u=np.asarray(learned_sources[name],dtype=np.float64)
        if u.ndim!=2 or u.shape[0]<2 or not 0<u.shape[1]<=u.shape[0]:
            raise ValueError("Every observed source must have a nonempty [D,K] basis")
        if dimension is None:
            dimension=u.shape[0]
        if u.shape[0]!=dimension or not np.isfinite(u).all():
            raise ValueError("Observed operators must share identical named coordinates")
        if not np.allclose(u.T@u,np.eye(u.shape[1]),atol=1e-5):
            raise ValueError("Each source basis must be orthonormal")
        projectors.append(u@u.T)
        dimensions.append(u.shape[1])
    avg=sum(projectors)/len(projectors)
    avg=(avg+avg.T)*.5
    eig,vecs=np.linalg.eigh(avg)
    order=np.argsort(-eig,kind="stable")
    eig=np.clip(eig[order],0.,1.)
    vecs=vecs[:,order]
    support=np.stack([
        (vecs*(pj@vecs)).sum(axis=0)
        for pj in projectors
    ])
    minimum=support.min(axis=0)
    eligible=(eig>=projector_threshold-1e-10)&(
        minimum>=min_source_energy-1e-10
    )
    # Keep a prefix by decreasing averaged-projector eigenvalue.
    # A selected axis must be reproducible in ALL supplied sources.
    k=0
    for passes in eligible:
        if not passes:
            break
        k+=1
    sensitivity={}
    for level in sensitivity_levels:
        if not .5<=level<=1:
            raise ValueError("Sensitivity grid must be fixed inside [.5,1]")
        n=0
        for value,contribution in zip(eig,minimum):
            if value<level-1e-10 or contribution<min_source_energy-1e-10:
                break
            n+=1
        sensitivity[f"{level:.3f}"]=int(n)

    if k:
        v=vecs[:,:k]
        projection=v@v.T
        axes=[]
        pivots=[]
        margins=[]
        # Construct a named, rotation-invariant reference frame by
        # pivoting the projector of ORIGINAL observable mask operators.
        for j in range(k):
            candidates=np.empty((dimension,dimension),dtype=np.float64)
            scores=np.empty(dimension,dtype=np.float64)
            for sensor in range(dimension):
                candidate=projection[:,sensor].copy()
                for previous in axes:
                    candidate-=previous*np.dot(previous,candidate)
                scores[sensor]=np.linalg.norm(candidate)
                candidates[:,sensor]=candidate
            # When projected coordinate norms are equal to floating
            # precision, LAPACK's arbitrary basis rotations can change
            # their sub-ulp ordering. Deterministic observable coordinate
            # indices MUST break ties, not random eigensolver noise.
            # This is a canonical dictionary convention, not an assertion
            # that close physiological axes are uniquely identifiable.
            ordered_scores=np.round(scores,decimals=11)
            order=np.lexsort((np.arange(dimension),-ordered_scores))
            pivot=int(order[0])
            score=float(scores[pivot])
            if score<1e-8:
                raise ValueError("Unstable canonical axis: projected operator dictionary is rank deficient")
            margin=float(score-scores[order[1]]) if len(order)>1 else score
            axis=candidates[:,pivot]/score
            if axis[pivot]<0:
                axis*=-1
            # deterministic repeated modified Gram-Schmidt
            for previous in axes:
                axis-=previous*np.dot(previous,axis)
            axis/=np.linalg.norm(axis)
            axes.append(axis)
            pivots.append(pivot)
            margins.append(margin)
        canonical=np.stack(axes,axis=1)
        if not np.allclose(canonical.T@canonical,np.eye(k),atol=1e-8):
            raise AssertionError("Stable canonical source operator frame lost orthogonality")
        if not np.allclose(projection@canonical,canonical,atol=1e-8):
            raise AssertionError("Canonical frame left the shared observed source span")
        signed=_positive_negative_template(canonical)
    else:
        canonical=np.empty((dimension,0),dtype=np.float64)
        signed=np.empty((0,2*dimension),dtype=np.float32)
        pivots=[]
        margins=[]

    return StableMaskSubspace(
        k_common=int(k),
        rank_by_source=tuple(dimensions),
        projector_eigenvalues=eig,
        mean_projector_eigenvectors=vecs,
        minimum_source_energy_per_eigenvector=minimum,
        stable_orthonormal_operators=canonical,
        signed_sparse_operator_definitions=signed,
        selected_named_operator_indices=tuple(pivots),
        selection_residual_margins=tuple(margins),
        source_ids=names,
        primary_projector_threshold=float(projector_threshold),
        minimum_source_energy_threshold=float(min_source_energy),
        sensitivity_counts=sensitivity,
        no_shared_patient_outcomes_or_oracle=True,
    )


@dataclass(frozen=True)
class StableCareAnchor:
    """Minimal typed contract for training with V9 projector-derived axes.

    Unlike V8's CarePolicyRank, this is NOT presented as a single
    permutation-calibrated full-cohort eigenvalue rank.
    """
    chosen_k:int
    n_observed_sensor_channels:int
    positive_negative_simplex_anchors:np.ndarray
    source_method:str
    source_workflow:str
    identifiable_medical_concept_names:bool=False

    @classmethod
    def from_v9_status(cls,receipt:dict)->"StableCareAnchor":
        src=receipt.get("cross_environment_common_core",{})
        channels=int(receipt["n_pilot_selected_measurement_channels"])
        k=src.get("k_observable_common_core")
        if (src.get("status")!="OBSERVATION_COMMON_SUBSPACE_ONLY"
            or not isinstance(k,int) or k<1):
            raise ValueError("Cannot force V9 K when observational common core abstained")
        bases=np.asarray(src["positive_negative_operator_simplex"],dtype=np.float64)
        d=4*channels
        if bases.shape!=(k,2*d):
            raise ValueError("V9 signed operator anchors do not match selected measured channels")
        if not np.isfinite(bases).all() or np.any(bases<0) or (
            not np.allclose(bases.sum(axis=1),1,atol=1e-6)
        ):
            raise ValueError("V9 anchors are not valid nonnegative signed-operator mixtures")
        if not receipt["validation_calibration_test_not_used"]:
            raise ValueError("V9 rank must be frozen strictly on TRAIN patients")
        return cls(
            chosen_k=k,
            n_observed_sensor_channels=channels,
            positive_negative_simplex_anchors=bases.astype(np.float32),
            source_method="train_only_half_subsample_mean_projector_common_core",
            source_workflow="https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41609582143",
            identifiable_medical_concept_names=False,
        )
