"""V59 API irregular-time, external metadata enrollment and source-freeze tests."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from conceptfan_discovery.bidmc_realworld_v58 import NormalizationV58
from scripts.download_vitaldb_external_v59 import (
    _align,_numeric_track,_earliest_complete_window,REQUIRED_TRACKS,
)
from scripts.experiment_conceptfan_vitaldb_transfer_v59 import (
    load_external_vitaldb,
)


def test_last_observation_alignment_is_causal_not_future_filled():
    tt=np.array([0.,2.,8.,12.])
    vv=np.array([60.,63.,90.,96.])
    grid=np.arange(16,dtype=float)
    observed=_align(tt,vv,grid)
    assert observed[1]==60.
    assert observed[3]==63.
    assert np.isnan(observed[6])
    assert np.isnan(observed[7])
    assert observed[8]==90.
    assert np.isnan(observed[16-1])


def test_safely_parse_sparse_numeric_track_and_last_duplicate():
    raw=b"Time,Value\n0,75\n2,78\n2,79\n5,82\n8,75\n"
    # 300 minimum record points is a deliberate validity criterion.
    with pytest.raises(ValueError,match="Insufficient"):
        _numeric_track(raw,"Solar8000/HR")
    many="Time,Value\n"+"".join(f"{2*i},{75+i%5}\n" for i in range(305))
    time,value=_numeric_track(many.encode(),"Solar8000/HR")
    assert len(time)==305 and time[0]==0. and time[-1]==608.


def test_first_available_window_selection_never_uses_signal_values():
    time=np.arange(0,1001,2,dtype=float)
    dataset={}
    for k,name in enumerate(REQUIRED_TRACKS):
        dataset[name]=(time,np.full(len(time),90. if k<2 else 97.))
    first=_earliest_complete_window(dataset)
    assert first is not None and first[0]==0
    changed={name:(t,values*1.1) for name,(t,values) in dataset.items()}
    other=_earliest_complete_window(changed)
    assert other[0]==first[0]
    np.testing.assert_array_equal(np.isfinite(first[1]),np.isfinite(other[1]))


def test_real_external_window_loader_preserves_hidden_HR_role(tmp_path):
    source=Path(tmp_path)
    ids=list(range(101,141))
    receipt={"selected_public_case_ordinals":ids,
             "clinical_HR_not_an_input":True}
    (source/"vitaldb_download_receipt.json").write_text(json.dumps(receipt))
    for case in ids:
        time=np.arange(480)
        pulse=90+3*np.sin(time/18.)+case%5
        resp=20+2*np.cos(time/31.)
        oxygen=97+0.1*np.sin(time/22.)
        ecg=pulse+2*np.sin(time/35.)
        np.savez_compressed(source/f"vitaldb_case_{case}.npz",
            signals=np.column_stack([pulse,resp,oxygen,ecg]).astype(np.float32),
            origin_sec=10)
    norm=NormalizationV58(
        mean=(90.,20.,97.,70.,0.,0.,0.,.4,.4,.1),
        scale=(20.,10.,3.,15.,4.,4.,1.,2.,2.,1.),
        train_count=35000)
    observed,_=load_external_vitaldb(source,norm)
    assert len(observed["y"])>800
    assert len(set(observed["patient"]))==40
    assert observed["x"].shape[1:]==(16,10)
    assert observed["mask"].shape==observed["x"].shape
    expected=observed["x"].copy()
    # Changing ECG reference alone changes y but never any FAN feature.
    with np.load(source/"vitaldb_case_101.npz") as z:
        signals=z["signals"].copy()
    signals[:,3]=125.
    np.savez_compressed(source/"vitaldb_case_101.npz",
                        signals=signals,origin_sec=10)
    changed,_=load_external_vitaldb(source,norm)
    np.testing.assert_array_equal(expected,changed["x"])
    assert (changed["y"]!=observed["y"]).any()


def test_unavailable_vitaldb_source_refuses_wrong_track_claim(tmp_path):
    p=tmp_path/"vitaldb_download_receipt.json"
    p.write_text(json.dumps({"selected_public_case_ordinals":[1],
                             "clinical_HR_not_an_input":False}))
    norm=NormalizationV58(
        mean=tuple(np.zeros(10)),scale=tuple(np.ones(10)),train_count=200)
    with pytest.raises(RuntimeError,match="external"):
        load_external_vitaldb(tmp_path,norm)
