"""V61 regression tests: conditional measured novelty, target exclusion, nulls."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.program_compiler_v60 import infer_program_matrix_v60
from conceptfan_discovery.program_compiler_v61 import (
    compile_allowed_programs_v61,select_conditioning_and_novel_programs_v61,
)
from conceptfan_discovery.conditional_innovation_v61 import (
    fit_conditional_sources_v61,infer_conditional_sources_v61,
    patient_stratified_null_v61,conditional_shuffle_gate_v61,
)
from conceptfan_discovery.program_fan_v60 import (
    fit_program_head_v60,infer_program_head_v60,
    apply_program_admission_v60,
)
from conceptfan_discovery.real_future_targets_v61 import (
    _rows,transform_allowed_v61,AllowedSensorScalerV61,
)


def synthetic(seed=661,n=360):
    rng=np.random.default_rng(seed)
    patient=np.repeat(np.arange(18),n//18)
    latent=rng.normal(size=(n,1))
    drift=rng.normal(size=(n,1))
    x=np.zeros((n,16,3))
    for j in range(3):
        x[:,:,j]=(latent*(.5 if j==1 else 1.)+
                   drift*(1. if j==2 else .12)+
                   .5*rng.normal(size=(n,16)))
    mask=rng.random(x.shape)>.14
    return x,mask,patient


def test_excluded_resp_target_cannot_influence_any_source_or_train_quantiles():
    x,m,group=synthetic()
    bank=compile_allowed_programs_v61(
        x,m,group,allowed_channels=(0,2))
    assert bank.candidates
    assert bank.screened_programs==14
    assert all(1 not in c.channels for c in bank.candidates)
    pulse,novel=select_conditioning_and_novel_programs_v61(bank,(0,2))
    assert pulse and novel
    assert all(0 not in bank.candidates[i].channels for i in novel)
    # Deliberate major perturbation of the forbidden outcome-channel:
    # the program bank should remain BITWISE IDENTICAL.
    bad=x.copy()
    bad[:,:,1]=100000.+np.arange(len(x))[:,None]*1234.
    other=compile_allowed_programs_v61(
        bad,m,group,allowed_channels=(0,2))
    assert bank==other
    q1,s1=infer_program_matrix_v60(bank,x,m)
    q2,s2=infer_program_matrix_v60(bank,bad,m)
    np.testing.assert_array_equal(q1,q2)
    np.testing.assert_array_equal(s1,s2)


def test_conditional_residuals_are_train_only_and_exactly_zero_when_pulse_unavailable():
    x,m,group=synthetic()
    bank=compile_allowed_programs_v61(x,m,group)
    p,n=select_conditioning_and_novel_programs_v61(bank,(0,1,2))
    z,s=infer_program_matrix_v60(bank,x,m)
    fit=fit_conditional_sources_v61(z,s,m,group,
                                    pulse_indices=p,source_indices=n)
    assert fit is not None
    assert not fit.fit_used_y and not fit.source_reads_pulse_directly
    assert all(i in n for i in fit.raw_source_indices)
    values,support=infer_conditional_sources_v61(fit,z,s,m)
    assert values.shape==support.shape==(len(x),len(fit.raw_source_indices))
    assert np.isfinite(values).all()
    missing=m.copy();missing[:,:,0]=False
    zz,ss=infer_program_matrix_v60(bank,x,missing)
    v0,s0=infer_conditional_sources_v61(fit,zz,ss,missing)
    np.testing.assert_array_equal(v0,np.zeros_like(v0))
    np.testing.assert_array_equal(s0,np.zeros_like(s0))


def test_conditional_native_head_and_fallback_no_pulse():
    x,m,group=synthetic()
    bank=compile_allowed_programs_v61(x,m,group)
    p,n=select_conditioning_and_novel_programs_v61(bank,(0,1,2))
    z,s=infer_program_matrix_v60(bank,x,m)
    fitted=fit_conditional_sources_v61(z,s,m,group,
                                       pulse_indices=p,source_indices=n)
    r,a=infer_conditional_sources_v61(fitted,z,s,m)
    rng=np.random.default_rng(112)
    y=(rng.random(len(x)) < 1/(1+np.exp(-(
        1.4*x[:,:,1].mean(axis=1)+.6*x[:,:,0].mean(axis=1))))).astype(int)
    pulse=fit_program_head_v60(
        z[:,p],s[:,p],np.zeros(len(x)),y,group,variant="PURE")
    baseline=infer_program_head_v60(
        pulse,z[:,p],s[:,p],np.zeros(len(x)))["logits"]
    fit=fit_program_head_v60(
        r,a,baseline,y,group,variant="RESIDUAL")
    post=infer_program_head_v60(fit,r,a,baseline)
    np.testing.assert_allclose(
        post["logits"],baseline+post["signed_contributions"].sum(axis=1),
        atol=1e-9)
    stopped,used=apply_program_admission_v60(
        baseline,post["logits"],post["used"][:,list(fit.active_indices)],
        approved=False)
    np.testing.assert_array_equal(stopped,baseline)
    assert not used.any()


def test_conditional_source_patient_and_pulse_bin_permutations():
    n=120
    groups=np.repeat(np.arange(6),20)
    baseline=np.tile(np.arange(20),6).astype(float)
    novel=np.arange(n,dtype=float)[:,None]
    supports=np.ones_like(novel)
    changed,available,fraction=patient_stratified_null_v61(
        novel,supports,baseline,groups,seed=61)
    assert fraction>.5
    assert np.array_equal(np.sort(changed.ravel()),np.arange(n))
    # Every shuffled row must originate from same person and same
    # within-person quartile of pulse-only scores.
    from_ix=changed[:,0].astype(int)
    assert np.array_equal(groups[from_ix],groups)
    threshold=np.quantile(baseline[:20],[.25,.5,.75])
    q_before=np.searchsorted(threshold,baseline,side="right")
    q_after=np.searchsorted(threshold,baseline[from_ix],side="right")
    assert np.array_equal(q_before,q_after)


def test_future_resp_label_exclusion_and_true_future_60s_window():
    t=np.arange(500)
    p=92+4*np.sin(t/24.)
    resp=19+3*np.cos(t/18.)
    spo=98+.2*np.sin(t/33.)
    hr=96+3*np.cos(t/21.)
    measured=np.column_stack((p,resp,spo,hr))
    original=_rows(measured,case=14,patient="p",task="future_resp_60s")
    assert original
    first=original[0]
    assert first["target_first"]>=first["input_end_exclusive"]+60
    assert not first["mask"][:,1].any()
    change=measured.copy()
    target_start=first["target_first"]
    change[target_start:target_start+4,1]=45.
    after=_rows(change,case=14,patient="p",task="future_resp_60s")
    np.testing.assert_array_equal(first["raw"],after[0]["raw"])
    np.testing.assert_array_equal(first["mask"],after[0]["mask"])
    assert after[0]["y"]==1
    scale=AllowedSensorScalerV61((90.,0.,98.),(20.,1.,2.),(0,2))
    x=np.stack([first["raw"]])
    m=np.stack([first["mask"]])
    nx,nm=transform_allowed_v61(x,m,scale)
    assert not nm[:,:,1].any()
    assert not nx[:,:,1].any()


def test_null_gate_never_bypasses_patient_groups_or_source_mask():
    x,m,group=synthetic(n=360)
    bank=compile_allowed_programs_v61(x,m,group)
    p,n=select_conditioning_and_novel_programs_v61(bank,(0,1,2))
    z,s=infer_program_matrix_v60(bank,x,m)
    fit=fit_conditional_sources_v61(z,s,m,group,
                                    pulse_indices=p,source_indices=n)
    a,b=infer_conditional_sources_v61(fit,z,s,m)
    rng=np.random.default_rng(91)
    labels=(rng.random(len(x))>.5).astype(int)
    baseline=np.zeros(len(x))
    risk=fit_program_head_v60(a,b,baseline,labels,group,variant="RESIDUAL")
    chosen=infer_program_head_v60(risk,a,b,baseline)["logits"]
    null=conditional_shuffle_gate_v61(
        risk,a,b,baseline,labels,group,observed_proposed=chosen,seed=20266121)
    assert null["conditional_source_subject_stratified_null_repeats"]==49
    assert null["null_preserves_patient_and_pulse_quartile"]
    assert 0<=null["minimum_row_reassignment_fraction"]<=1
    assert not null["autocorrelated_window_exchangeability_not_proven"] is False
