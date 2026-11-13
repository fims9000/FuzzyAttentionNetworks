"""V60 standalone/compiler tests: typed programs, missingness and no Y leakage."""
import numpy as np
import pytest

from conceptfan_discovery.program_compiler_v60 import (
    compile_measured_programs_v60,infer_program_matrix_v60,
    fingerprint_program_bank_v60,_raw_operator,
    UNARY,PAIR,
)
from conceptfan_discovery.real_future_tasks_v60 import _window_rows


def cohort(seed=20,n=360):
    r=np.random.default_rng(seed)
    people=np.repeat(np.arange(18),n//18)
    signal=r.normal(size=(n,1,3))
    x=signal+.12*r.normal(size=(n,16,3))
    # independent trend/window bursts: no generator-specific lag target
    x[:,8:,0]+=.2*np.linspace(0,1,8)[None,:]
    mask=r.random(x.shape)>.2
    return x,mask,people


def test_global_program_compiler_discovers_realizable_levels_without_lag_correlation():
    x,m,people=cohort()
    bank=compile_measured_programs_v60(x,m,people)
    assert 1<=len(bank.candidates)<=24
    assert bank.screened_programs==24
    assert len(set(c.candidate_id for c in bank.candidates))==len(bank.candidates)
    assert any(c.operator=="LEVEL_LAST4" and c.channels==(0,)
               for c in bank.candidates)
    assert set(c.operator for c in bank.candidates).issubset(set(UNARY+PAIR))
    assert not any(c.discovery_uses_outcome_y for c in bank.candidates)
    assert not any(c.physical_or_causal_identity_proven for c in bank.candidates)
    values,support=infer_program_matrix_v60(bank,x,m)
    assert values.shape==support.shape==(len(x),len(bank.candidates))
    assert np.isfinite(values).all()
    assert ((support>=0)&(support<=1)).all()
    assert bank==compile_measured_programs_v60(x,m,people)
    assert fingerprint_program_bank_v60(bank)==fingerprint_program_bank_v60(
        compile_measured_programs_v60(x,m,people))


def test_no_masked_future_or_hidden_value_leakage():
    x,m,p=cohort()
    bank=compile_measured_programs_v60(x,m,p)
    before=infer_program_matrix_v60(bank,x,m)
    dirty=x.copy();dirty[~m]=np.nan
    after=infer_program_matrix_v60(bank,dirty,m)
    for a,b in zip(before,after):
        np.testing.assert_array_equal(a,b)
    # Future values outside the operator's last-four-point window
    # cannot affect that program's measured last-four mean.
    p0=next(c for c in bank.candidates if
            c.operator=="LEVEL_LAST4" and c.channels==(0,))
    changed=x.copy();changed[:,:12,0]+=33.
    z1=infer_program_matrix_v60(bank,x,m)[0][:,list(bank.candidates).index(p0)]
    z2=infer_program_matrix_v60(bank,changed,m)[0][:,list(bank.candidates).index(p0)]
    np.testing.assert_array_equal(z1,z2)


def test_refuse_invalid_binary_mask_and_observed_nan():
    x,m,p=cohort()
    x[0,0,1]=np.nan
    m[0,0,1]=True
    with pytest.raises(ValueError,match="finite"):
        compile_measured_programs_v60(x,m,p)
    x,m,p=cohort()
    m=m.astype(int);m[0,0,1]=2
    with pytest.raises(ValueError,match="binary"):
        compile_measured_programs_v60(x,m,p)


def test_missing_source_is_exactly_zero_support_and_feature_value():
    x,m,p=cohort()
    bank=compile_measured_programs_v60(x,m,p)
    missing=np.zeros_like(m)
    zz,s=infer_program_matrix_v60(bank,x,missing)
    np.testing.assert_array_equal(zz,np.zeros_like(zz))
    np.testing.assert_array_equal(s,np.zeros_like(s))


def test_future_sixty_second_target_never_enters_features():
    rng=np.random.default_rng(44)
    x=np.column_stack([
        84+4*np.sin(np.arange(500)/31.),
        23+.5*np.cos(np.arange(500)/9.),
        np.full(500,98.)
    ])
    hr=85+4*np.sin(np.arange(500)/31.)
    original=_window_rows(x,hr,case=1,patient="p1",task="future_60s")
    assert original
    row=original[0]
    assert row["target_begin"]>=row["input_end_exclusive"]+60
    start=row["start"]
    future_hr=hr.copy()
    future_hr[row["target_begin"]:row["target_begin"]+4]=130.
    after=_window_rows(x,future_hr,case=1,patient="p1",task="future_60s")
    r=next(r for r in after if r["start"]==start)
    np.testing.assert_array_equal(row["raw"],r["raw"])
    np.testing.assert_array_equal(row["mask"],r["mask"])
    assert row["y"]==0 and r["y"]==1
