"""Falsification tests: stable observational K is a COMMON subspace, not
the maximum eigenvalue count from one fitted patient cohort.
"""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.stable_care_subspace_v9 import find_common_mask_subspace


def _rotated(basis:np.ndarray,seed:int)->np.ndarray:
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(basis.shape[1],basis.shape[1])))
    return basis@q


def test_common_two_axes_survive_unstable_extra_source_specific_factors():
    dim=12
    base=np.eye(dim)
    sources={
        "policy_0_half_a":_rotated(base[:,[0,1,2]],1),
        "policy_0_half_b":_rotated(base[:,[0,1,3]],2),
        "policy_1_half_a":_rotated(base[:,[0,1,4,5]],3),
        "policy_1_half_b":_rotated(base[:,[0,1]],4),
        "policy_2_half_a":_rotated(base[:,[0,1,6]],5),
        "policy_2_half_b":_rotated(base[:,[0,1,7]],6),
    }
    out=find_common_mask_subspace(sources)
    assert out.k_common==2
    assert out.rank_by_source==(3,3,4,2,3,3)
    assert out.projector_eigenvalues[1]==pytest.approx(1.,abs=1e-11)
    assert out.projector_eigenvalues[2]<.95
    np.testing.assert_allclose(
        out.stable_orthonormal_operators.T@out.stable_orthonormal_operators,
        np.eye(2),atol=1e-10,
    )
    assert out.receipt()["uses_any_outcome_labels"] is False
    assert out.signed_sparse_operator_definitions.shape==(2,24)
    np.testing.assert_allclose(
        out.signed_sparse_operator_definitions.sum(axis=1),1.,atol=1e-7
    )


def test_projector_basis_is_invariant_to_latent_rotations_and_column_permutations():
    basis=np.eye(10)[:,[1,3,5]]
    sources={f"fit{i}":_rotated(basis,10+i) for i in range(6)}
    a=find_common_mask_subspace(sources)
    sources_b={key:value[:,::-1] for key,value in sources.items()}
    b=find_common_mask_subspace(sources_b)
    assert a.k_common==3
    np.testing.assert_allclose(
        a.stable_orthonormal_operators,
        b.stable_orthonormal_operators,atol=1e-8
    )
    assert a.selected_named_operator_indices==b.selected_named_operator_indices


def test_disjoint_source_factors_require_honest_abstention():
    basis=np.eye(8)
    sources={f"x{i}":basis[:,[0,1]] if i%2 else basis[:,[2,3]]
             for i in range(6)}
    out=find_common_mask_subspace(sources)
    assert out.k_common==0
    assert out.receipt()["status"]=="ABSTAIN_NO_COMMON_OBSERVATION_SUBSPACE"


def test_zero_or_invalid_cross_cohort_bases_not_considered_evidence():
    basis=np.eye(7)[:,:3]
    sources={f"fit{i}":basis.copy() for i in range(6)}
    with pytest.raises(ValueError):
        find_common_mask_subspace({"only":basis})
    wrong={**sources,"broken":np.full((7,2),1.)}
    with pytest.raises(ValueError,match="orthonormal"):
        find_common_mask_subspace(wrong)
    with pytest.raises(ValueError):
        find_common_mask_subspace(sources,projector_threshold=.1)
    with pytest.raises(ValueError):
        find_common_mask_subspace(sources,min_source_energy=2.)


def test_shared_one_dimensional_care_operator_is_not_artificially_inflated():
    base=np.eye(9)
    sources={f"half{i}":_rotated(base[:,[3]],i+1) for i in range(6)}
    out=find_common_mask_subspace(sources)
    assert out.k_common==1
    assert len(out.selected_named_operator_indices)==1
    assert out.signed_sparse_operator_definitions.shape==(1,18)
