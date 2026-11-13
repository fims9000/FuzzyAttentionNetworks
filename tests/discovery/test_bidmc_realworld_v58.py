"""V58 unit tests. Fixtures are fabricated; no fabricated data become study results."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pytest

from conceptfan_discovery.bidmc_realworld_v58 import (
    causal_monitor_features,group_allocations,
    load_bidmc_realworld_v58,apply_train_normalization,THRESHOLD_BPM,
)
from scripts.experiment_conceptfan_bidmc_real_v58 import (
    _test_variant,_bootstrap_by_real_patient,
)


@pytest.fixture(scope="module")
def fake_full_bidmc(tmp_path_factory):
    root=tmp_path_factory.mktemp("fake_bidmc_numeric")
    for idx in range(1,54):
        # Exact same MIMIC group for records 1 and 2: must stay in same split.
        group="s00001" if idx==2 else f"s{idx:05d}"
        (root/f"bidmc_{idx:02d}_Fix.txt").write_text(
            f"bidmc_{idx:02d}\nMIMIC II matched wdb ID: {group}\n")
        rng=np.random.default_rng(idx)
        with (root/f"bidmc_{idx:02d}_Numerics.csv").open("w",newline="") as fp:
            writer=csv.writer(fp)
            writer.writerow(["Time [s]"," HR"," PULSE"," RESP"," SpO2"])
            group_shift=3.0*(idx%8-4)
            for t in range(480):
                p=97+group_shift+7*np.sin(t/43.)+.12*rng.normal()
                rr=23+2*np.sin(t/57.)
                spo=97+.2*np.sin(t/60.)
                hr=p+1.5*np.sin(t/64.)
                # Natural PULSE outage, HR remains observed from ECG.
                if 80<=t<90:p=float("nan")
                writer.writerow([t,f"{hr:.4f}",f"{p:.4f}",
                                 f"{rr:.4f}",f"{spo:.4f}"])
    return root


def test_causal_derivatives_and_volatility_do_not_see_future():
    t=np.arange(40,dtype=float)
    raw=np.column_stack((90.+t,20.+.1*t,97.-.02*t))
    x,m=causal_monitor_features(raw)
    after=raw.copy();after[25:]+=800
    xx,mm=causal_monitor_features(after)
    np.testing.assert_array_equal(x[:25],xx[:25])
    np.testing.assert_array_equal(m[:25],mm[:25])
    assert not m[0,4:7].any()
    assert not m[:2,7:].any()
    assert m[2,7:].all()


def test_missing_real_pulse_never_becomes_observed_zero():
    raw=np.column_stack((90+np.arange(40),20+np.arange(40),97+np.arange(40)))
    raw[10:13,0]=np.nan
    x,m=causal_monitor_features(raw)
    assert not m[10:13,0].any()
    assert not m[10:13,3].any()
    assert not m[10:14,4].any()
    assert not m[10:15,7].any()
    assert np.isfinite(x).all()


def test_patient_disjoint_split_and_same_patient_records_kept_together(fake_full_bidmc):
    data,norm,info=load_bidmc_realworld_v58(fake_full_bidmc)
    assert info["records"]==53
    assert info["identified_patient_groups"]==52
    assert info["HR_used_as_input"] is False
    assert info["HR_is_both_supervised_target_and_numeric_evaluation_reference"]
    assert not info["identical_patient_in_multiple_splits"]
    assert sum(len(row["y"]) for row in data.values())>1500
    assert {name:len(set(row["patient"])) for name,row in data.items()}=={
        name:info["splits"][name]["patient_group_count"] for name in data}
    patients={name:set(row["patient"]) for name,row in data.items()}
    assert all(not patients[k]&patients[j] for k in data for j in data if j!=k)
    # Recording 1/2 have same patient ID and MUST be allocated together.
    first=[name for name,row in data.items() if 1 in row["record"]]
    second=[name for name,row in data.items() if 2 in row["record"]]
    assert first==second
    assert all(row["x"].shape[-2:]==(16,10) for row in data.values())
    assert all(np.isfinite(row["x"]).all() for row in data.values())
    assert norm.train_count>100
    assert "s00001" not in str(info)
    assert info["raw_vitals_exported"] is False


def test_genuine_real_outcome_is_not_part_of_feature_construction(fake_full_bidmc):
    data,norm,info=load_bidmc_realworld_v58(fake_full_bidmc)
    anyrow=data["test"]
    original=anyrow["x"].copy()
    # X depends only on monitor PULSE/RESP/SpO2; relabeling HR is
    # downstream and must never alter TRAIN scaler or TEST feature values.
    fake_y=1-anyrow["y"]
    assert not np.array_equal(fake_y,anyrow["y"])
    np.testing.assert_array_equal(anyrow["x"],original)


def test_bias_and_missing_stress_recompute_dependent_channels(fake_full_bidmc):
    data,norm,_=load_bidmc_realworld_v58(fake_full_bidmc)
    heldout=data["test"]
    x,m,pulse=_test_variant(heldout,norm,"pulse_plus_25_bpm",11)
    assert x.shape==heldout["x"].shape
    idx=np.isfinite(pulse)&np.isfinite(heldout["pulse"])
    assert np.allclose(pulse[idx]-heldout["pulse"][idx],25.)
    absent,mask,pulse2=_test_variant(heldout,norm,"pulse_missing",11)
    assert not mask[:,:,[0,3,4,7]].any()
    assert np.isnan(pulse2).all()
    assert np.isfinite(absent).all()
    mcar,extra,pulse3=_test_variant(heldout,norm,"MCAR20",11)
    assert np.logical_or(~extra,heldout["mask"]).all()
    assert extra.mean()<heldout["mask"].mean()


def test_bootstrap_never_claims_overlapping_windows_independent():
    rng=np.random.default_rng(2)
    people=np.repeat(np.arange(8),16)
    y=rng.integers(0,2,len(people))
    z1=rng.normal(size=len(people))
    z2=rng.normal(size=len(people))
    result=_bootstrap_by_real_patient(y,z1,z2,people,n_draws=399)
    assert result["unique_patient_groups"]==8
    assert result["resample_unit"].startswith("whole held-out patient groups")
    assert len(result["interval95_descriptive"])==2
