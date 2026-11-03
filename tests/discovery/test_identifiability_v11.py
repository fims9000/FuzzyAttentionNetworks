"""Falsifiable V11 contracts: identifiable axes versus rotated subspaces."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.identifiability_v11 import (
    certify_from_svd_bootstraps,
    compare_group_attributions,
    fit_predictive_group_certificate,
    grouped_logit_contributions,
    projector_overlap,
    torch_rotation_invariant_group_loss,
)
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data


def _rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    r=np.eye(3)
    r[:2,:2]=[[c,-s],[s,c]]
    return r


def _certify_rotating_degenerate_pair():
    rng=np.random.default_rng(32)
    bases=np.asarray([_rotation(a) for a in rng.uniform(-1.5,1.5,64)])
    sv=np.asarray([[5.0,5.0,.2]+rng.normal(scale=.008,size=3)
                   for _ in range(64)])
    sv=np.sort(sv,axis=1)[:,::-1]
    return certify_from_svd_bootstraps(
        np.eye(3),np.asarray([5.,5.,.2]),bases,sv,
        computational_rank=2,minimum_subspace_overlap=.8,
    )


def test_projector_invariant_to_sign_and_rotation():
    u=np.eye(3)[:,:2]
    v=u@_rotation(.91)[:2,:2]
    assert projector_overlap(u,v)==pytest.approx(1.)
    assert projector_overlap(u,u@np.diag([1,-1]))==pytest.approx(1.)
    other=np.eye(3)[:,1:3]
    assert projector_overlap(u,other)==pytest.approx(.5)


def test_nearly_degenerate_singular_modes_must_be_group_not_two_clinical_axes():
    e=_certify_rotating_degenerate_pair()
    assert e.certified_boundaries==()
    assert len(e.groups)==1
    assert e.groups[0].members==(0,1)
    assert e.groups[0].status=="GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED"
    assert e.groups[0].q10_subspace_overlap>.99
    assert e.groups[0].q10_individual_axis_overlap is None
    assert not e.to_json()["clinical_concepts_identified"]


def test_separated_spectrum_yields_only_observable_singleton_certificates():
    rng=np.random.default_rng(11)
    u=np.tile(np.eye(3)[None,:,:],(32,1,1))
    boot_s=np.tile(np.asarray([6.,3.,.2]),(32,1))
    boot_s+=rng.normal(scale=.01,size=boot_s.shape)
    cert=certify_from_svd_bootstraps(
        np.eye(3),np.asarray([6.,3.,.2]),u,boot_s,computational_rank=2,
    )
    assert cert.certified_boundaries==(1,)
    assert [g.status for g in cert.groups]==[
        "INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
        "INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
    ]
    assert all(g.q10_subspace_overlap>.99 for g in cert.groups)
    assert not cert.to_json()["causal_variables_identified"]


def test_apparent_spectral_gap_alone_does_not_certify_if_projector_flips():
    rng=np.random.default_rng(2)
    boot=np.asarray([np.eye(3) if i%2 else np.eye(3)[:,[1,0,2]]
                     for i in range(40)])
    s=np.asarray([6.,3.,.2])
    result=certify_from_svd_bootstraps(
        np.eye(3),s,boot,np.tile(s,(40,1)),
        computational_rank=2,
    )
    assert 1 not in result.certified_boundaries
    assert result.groups[0].status=="GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED"


def test_subspace_penalty_rotation_invariant_and_has_gradient():
    # Two concepts form a 2D subspace in three source sensors.
    c,k=3,2
    w=torch.zeros((k,8*c),dtype=torch.float32)
    w[0,0]=.5
    w[0,1]=.5
    w[1,0]=.5
    w[1,4*c+1]=.5
    w=w.clone().detach().requires_grad_(True)
    u=torch.eye(c)[:,:k]
    g=_certify_rotating_degenerate_pair().groups
    a=torch_rotation_invariant_group_loss(w,u,g)
    theta=.65
    rotated=torch.tensor(
        [[np.cos(theta),-np.sin(theta)],
         [np.sin(theta),np.cos(theta)]],dtype=torch.float32
    )
    b=torch_rotation_invariant_group_loss(w,u@rotated,g)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    b.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    # A learned span outside the source's 2D subspace is penalized.
    outside=w.detach().clone()
    outside[:,[0,1,4*c+1]]=0.
    outside[:,2]=1.
    assert torch_rotation_invariant_group_loss(outside,u,g)>a


def test_group_sum_exact_even_when_individual_axis_contributions_rotate():
    g=_certify_rotating_degenerate_pair().groups
    first=np.asarray([[.7,.3],[-.2,.5],[.1,-.5]])
    second=np.asarray([[.1,.9],[.3,0.],[-.1,-.3]])
    grouped=grouped_logit_contributions(first,g)
    np.testing.assert_allclose(grouped[:,0],first.sum(axis=1))
    np.testing.assert_allclose(
        grouped_logit_contributions(second,g),second.sum(axis=1,keepdims=True)
    )
    res=compare_group_attributions(first,second,g)
    assert res[0]["individual_concept_identity_claim"] is False
    assert res[0]["slots"]==[0,1]


def test_train_only_patient_bootstrap_certificate_repeats_exactly():
    ds=generate_variable_factor_data(
        n_true_factors=3,seed=614,split_seed=913,n_samples=300,
    )
    x,m=ds.train.x,ds.train.mask
    a=fit_predictive_group_certificate(
        x,m,computational_rank=3,lags=(1,2),bootstraps=10,seed=37,
    )
    b=fit_predictive_group_certificate(
        x,m,computational_rank=3,lags=(1,2),bootstraps=10,seed=37,
    )
    np.testing.assert_allclose(a.bootstrap_singular_values,
                               b.bootstrap_singular_values,rtol=0,atol=0)
    assert a.to_json()==b.to_json()
    assert not a.to_json()["clinical_concepts_identified"]
    assert a.to_json()["input_used"].startswith("training observations")
    assert ds.receipt["latent_ground_truth_used_for_training"] is False


def test_invalid_singular_and_nonorthogonal_source_are_rejected():
    e=_certify_rotating_degenerate_pair()
    with pytest.raises(ValueError):
        certify_from_svd_bootstraps(
            np.eye(3),np.asarray([1.,2.,3.]),
            np.tile(np.eye(3),(8,1,1)),
            np.tile(np.asarray([1.,2.,3.]),(8,1)),computational_rank=2,
        )
    with pytest.raises(ValueError):
        torch_rotation_invariant_group_loss(
            torch.ones(2,24),torch.ones(3,2),e.groups
        )
