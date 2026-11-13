"""V61R regression: truly seeded PyTorch, deterministic outcome-blind sources.

These are engineering test fixtures, not medical outcome evidence. The
changes to V61R were explicitly declared *after* V61 TEST was inspected.
"""
from __future__ import annotations

import json
import numpy as np

from conceptfan_discovery.program_compiler_v60 import infer_program_matrix_v60
from conceptfan_discovery.program_compiler_v61 import (
    compile_allowed_programs_v61,select_conditioning_and_novel_programs_v61)
from conceptfan_discovery.conditional_innovation_v61 import (
    fit_conditional_sources_v61,infer_conditional_sources_v61)
from conceptfan_discovery.stable_source_contract_v61r import (
    shortlist_conditional_sources_v61r,source_contract_fingerprint_v61r)
from conceptfan_discovery.program_fan_v60 import fit_program_head_v60


def synthetic(seed=49):
    rng=np.random.default_rng(seed)
    n=360
    group=np.repeat(np.arange(18),20)
    u=rng.normal(size=(n,1))
    signal=np.zeros((n,16,3))
    signal[:,:,0]=u+.15*rng.normal(size=(n,16))
    signal[:,:,1]=.7*u+.5*rng.normal(size=(n,16))
    signal[:,:,2]=.4*u+.8*rng.normal(size=(n,16))
    mask=rng.random(signal.shape)>.10
    return signal,mask,group


def test_V61R_shortlist_is_stable_conditioned_and_label_blind():
    x,m,g=synthetic()
    bank=compile_allowed_programs_v61(x,m,g)
    pulse,other=select_conditioning_and_novel_programs_v61(bank,(0,1,2))
    values,support=infer_program_matrix_v60(bank,x,m)
    full=fit_conditional_sources_v61(
        values,support,m,g,pulse_indices=pulse,source_indices=other)
    assert full is not None and len(full.raw_source_indices)>6
    source,contract=shortlist_conditional_sources_v61r(
        full,bank,values,support,m,g)
    assert source and contract
    assert len(source.raw_source_indices)<=6
    assert contract.selection_used_train_Y is False
    assert len(set(contract.selected_candidate_ids))==len(contract.selected_candidate_ids)
    assert all(0 not in bank.candidates[j].channels
               for j in source.raw_source_indices)
    v,s=infer_conditional_sources_v61(source,values,support,m)
    assert v.shape[1]==len(source.raw_source_indices)
    assert np.isfinite(v).all()
    again,identical=shortlist_conditional_sources_v61r(
        full,bank,values,support,m,g)
    assert source==again and contract==identical
    assert (source_contract_fingerprint_v61r(source,contract)==
            source_contract_fingerprint_v61r(again,identical))


def test_stochastic_initialization_creates_distinct_learned_heads_without_TEST():
    x,m,g=synthetic()
    bank=compile_allowed_programs_v61(x,m,g)
    p,_=select_conditioning_and_novel_programs_v61(bank,(0,1,2))
    values,support=infer_program_matrix_v60(bank,x,m)
    yy=np.random.default_rng(19).integers(0,2,len(x))
    base=np.zeros(len(x))
    a=fit_program_head_v60(
        values[:,p],support[:,p],base,yy,g,variant="PURE",
        seed=11,train_initialization_std=.03,
        retain_all_predeclared_sources=True)
    b=fit_program_head_v60(
        values[:,p],support[:,p],base,yy,g,variant="PURE",
        seed=23,train_initialization_std=.03,
        retain_all_predeclared_sources=True)
    assert a.fixed_outcome_blind_source_shortlist
    assert b.fixed_outcome_blind_source_shortlist
    assert a.stochastic_seeded_initialization_std==.03
    assert b.stochastic_seeded_initialization_std==.03
    assert a.active_indices==tuple(range(len(p)))
    assert b.active_indices==tuple(range(len(p)))
    assert not np.array_equal(a.alpha,b.alpha) or not np.array_equal(a.gamma,b.gamma)
    # Old V60 interface unchanged: defaults imply deterministic same start.
    old=fit_program_head_v60(
        values[:,p],support[:,p],base,yy,g,variant="PURE",seed=11)
    assert not old.fixed_outcome_blind_source_shortlist
    assert old.stochastic_seeded_initialization_std==0.


def test_outcome_blind_sources_cannot_read_target_resp():
    x,m,g=synthetic()
    bank=compile_allowed_programs_v61(
        x,m,g,allowed_channels=(0,2))
    p,other=select_conditioning_and_novel_programs_v61(bank,(0,2))
    values,support=infer_program_matrix_v60(bank,x,m)
    full=fit_conditional_sources_v61(
        values,support,m,g,pulse_indices=p,source_indices=other)
    source,contract=shortlist_conditional_sources_v61r(
        full,bank,values,support,m,g)
    assert source
    assert all(set(bank.candidates[j].channels)=={2}
               for j in source.raw_source_indices)
    bad=x.copy()
    bad[:,:,1]=np.random.default_rng(203).normal(size=x[:,:,1].shape)*9999
    a=compile_allowed_programs_v61(
        bad,m,g,allowed_channels=(0,2))
    assert a==bank
