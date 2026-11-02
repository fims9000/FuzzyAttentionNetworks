"""Train-only rank of *care observation policy*, not patient physiology.

Autonomous mask-process concept count is estimated from correlations among
patient-level measurement-frequency/recency/staleness/transition operators.
The null independently permutes each source operator between TRAIN patients:
feature marginal sampling-policy distributions are preserved while patient-
specific cross-operator dependence is destroyed.

The largest eigenvalue of EACH null draw yields one familywise conservative
noise threshold. Any claimed rank is only the dimension of reproducible
train-measured cross-operator covariance; it need not equal the number of
hospital workflows or treatment causes. Number of process concept slots
is NOT chosen using the already-reused clinical validation.

Signed covariance eigenvectors are anchored into the precise sparsemax
positive/negative source dictionary; direct beta and the threshold/width
remain participating in the task optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .observation_process_concepts_v7 import observation_process_operator_bank


@dataclass(frozen=True)
class CarePolicyRank:
    chosen_k:int
    detected_uncapped_k:int
    raw_eigenvalues:np.ndarray
    null_max_eigenvalue_cutoff:float
    source_operator_center:np.ndarray
    source_operator_scale:np.ndarray
    signed_operator_basis:np.ndarray
    positive_negative_simplex_anchors:np.ndarray
    n_train_patients:int
    n_observed_sensor_channels:int
    patient_permutations:int
    seed:int
    truncated_by_budget:bool

    def receipt(self)->dict:
        return {
            "status":"TRAIN_OBSERVATION_PROCESS_FACTOR_BUDGET",
            "estimated_uncapped_k":int(self.detected_uncapped_k),
            "selected_computational_k":int(self.chosen_k),
            "truncated_by_announced_maximum_capacity":bool(self.truncated_by_budget),
            "raw_eigenvalues":self.raw_eigenvalues.tolist(),
            "null_max_eigenvalue_cutoff":float(self.null_max_eigenvalue_cutoff),
            "source_operator_center":self.source_operator_center.tolist(),
            "source_operator_scale":self.source_operator_scale.tolist(),
            "signed_operator_basis":self.signed_operator_basis.tolist(),
            "trained_model_sparse_source_weights":self.positive_negative_simplex_anchors.tolist(),
            "n_train_patients":int(self.n_train_patients),
            "n_observed_sensor_channels":int(self.n_observed_sensor_channels),
            "permutations":int(self.patient_permutations),
            "seed":int(self.seed),
            "train_only_masks":True,
            "clinical_outcomes_used_for_k":False,
            "physiological_measurement_values_used_for_k":False,
            "clinical_oracle_concepts_supplied":False,
            "meaning":"detectable correlated CARE OBSERVATION operator directions, NOT independently proven number of clinical mechanisms",
        }


def _positive_negative_template(basis:np.ndarray)->np.ndarray:
    matrix=np.asarray(basis,dtype=np.float64)
    d,k=matrix.shape
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix.T@matrix,np.eye(k),atol=2e-4
    ):
        raise ValueError("Source basis must be finite and orthonormal")
    template=np.zeros((k,2*d),dtype=np.float32)
    for i in range(k):
        direction=matrix[:,i]
        amount=np.abs(direction)
        amount/=max(amount.sum(),1e-12)
        template[i,:d]=np.where(direction>=0,amount,0)
        template[i,d:]=np.where(direction<0,amount,0)
    if not np.allclose(template.sum(axis=1),1.,atol=1e-6):
        raise ValueError("Every mask concept must have an inspectable unit mass")
    return template


def fit_care_policy_rank(
    train_masks:np.ndarray,
    *,
    permutations:int=90,
    seed:int=20271117,
    alpha:float=.05,
    max_slots:int=8,
)->CarePolicyRank:
    m=np.asarray(train_masks,dtype=np.float32)
    if m.ndim!=3 or m.shape[0]<100 or m.shape[1]<8 or m.shape[2]<2:
        raise ValueError("Expected >=100 TRAIN mask histories with several channels")
    if not np.isfinite(m).all() or not np.logical_or(m==0,m==1).all():
        raise ValueError("Strict observed/not-observed mask only")
    if permutations<25 or not .001<alpha<.25 or not 1<=max_slots<=16:
        raise ValueError("Permutation and computational capacity must be prespecified")
    with torch.no_grad():
        measured=observation_process_operator_bank(torch.from_numpy(m)).numpy().astype(float)
    n,d=measured.shape
    center=measured.mean(axis=0)
    scale=measured.std(axis=0)
    # A constant measurement policy operator conveys no patient-specific
    # information and must not be inflated by dividing by tiny variance.
    valid=scale>.01
    if valid.sum()<2:
        raise ValueError("ABSTAIN: too few nonconstant observation-policy operators")
    transformed=np.zeros_like(measured)
    transformed[:,valid]=(measured[:,valid]-center[valid])/scale[valid]
    covariance=(transformed.T@transformed)/n
    eigvals,vecs=np.linalg.eigh(covariance)
    eigvals=eigvals[::-1]
    vecs=vecs[:,::-1]
    for i in range(d):
        anchor=int(np.argmax(np.abs(vecs[:,i])))
        if vecs[anchor,i]<0:
            vecs[:,i]*=-1.
    rng=np.random.default_rng(int(seed))
    nullmax=np.empty(permutations,dtype=float)
    for trial in range(permutations):
        # Each sensor-operator column permuted INDEPENDENTLY across
        # patients. Preserves its marginal measurement-policy frequency.
        null=np.empty_like(transformed)
        for col in range(d):
            null[:,col]=transformed[rng.permutation(n),col]
        nullmax[trial]=np.linalg.eigvalsh((null.T@null)/n)[-1]
    threshold=float(np.quantile(nullmax,1-alpha))
    rank=int(np.count_nonzero(eigvals>threshold))
    if rank<1:
        raise ValueError("ABSTAIN: no identifiable correlated mask-policy direction exceeds permutation null")
    k=min(rank,int(max_slots))
    src=vecs[:,:k]
    weights=_positive_negative_template(src)
    return CarePolicyRank(
        chosen_k=k,detected_uncapped_k=rank,
        raw_eigenvalues=eigvals,
        null_max_eigenvalue_cutoff=threshold,
        source_operator_center=center,source_operator_scale=scale,
        signed_operator_basis=src,
        positive_negative_simplex_anchors=weights,
        n_train_patients=n,n_observed_sensor_channels=m.shape[-1],
        patient_permutations=permutations,seed=int(seed),
        truncated_by_budget=rank>max_slots,
    )
