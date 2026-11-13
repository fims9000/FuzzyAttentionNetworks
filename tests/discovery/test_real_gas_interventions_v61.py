import io

import numpy as np
import pytest

from scripts.experiment_real_gas_interventions_v61 import (
    parse_run_name,time_median_bins,stratify_whole_runs,
    train_sensor_normalizer,raw_exposure_features,
    run_level_spearman,semantic_audit,
)


def test_real_config_filename_controls_not_predictor_features():
    a=parse_run_name("dataset_twosources_downsampled/007_Et_L_Me_H")
    assert a["run_index"]==7
    assert a["gas_config"]=="L-methane-H"
    assert a["ppm"]=={"ethylene":31.,"co":0.,"methane":131.}
    assert a["targets"]=={"ethylene_present":1,
                          "co_present":0,"methane_present":1}
    b=parse_run_name("015_Et_n_CO_n")
    assert b["targets"]=={"ethylene_present":0,"co_present":0,
                          "methane_present":0}
    with pytest.raises(ValueError):
        parse_run_name("malformed-run-name")


def test_five_second_median_uses_real_time_not_sample_row_index():
    time=np.arange(0.,300.,.1)
    signal=np.column_stack(
        (time,np.full(len(time),22.),np.full(len(time),38.),
         np.column_stack([time*(j+1) for j in range(8)])))
    encoded=io.StringIO()
    np.savetxt(encoded,signal,fmt="%.5f",delimiter=",")
    result,stats=time_median_bins(encoded.getvalue().encode())
    assert result.shape==(60,8)
    assert stats["min_sensor_samples_per_bin"]>=49
    assert 2.<result[0,0]<3.
    assert 60*5==300


def test_missing_time_window_does_not_get_imputed():
    time=np.arange(0.,300.,.1)
    matrix=np.column_stack(
        (time,np.ones(len(time)),np.ones(len(time)),
         np.ones((len(time),8))))
    filtered=matrix[(time<100)|(time>=105)]
    out=io.StringIO()
    np.savetxt(out,filtered,fmt="%.3f",delimiter=",")
    with pytest.raises(ValueError,match="Insufficient measured samples"):
        time_median_bins(out.getvalue().encode())


def test_six_repeated_control_conditions_split_by_whole_recordings():
    records=[{"gas_config":f"cfg_{k}","run_index":i}
             for k in range(30) for i in range(k*6,(k+1)*6)]
    split=stratify_whole_runs(records)
    assert tuple(map(len,split.values()))==(120,30,30)
    all_ids=np.concatenate(list(split.values()))
    assert len(np.unique(all_ids))==180
    for k in range(30):
        indexes=set(range(k*6,(k+1)*6))
        assert len(indexes & set(split["TRAIN"]))==4
        assert len(indexes & set(split["VALID"]))==1
        assert len(indexes & set(split["TEST"]))==1
    with pytest.raises(ValueError):
        stratify_whole_runs(records[:-1])


def test_normalization_uses_train_physical_sensor_readings_only():
    rng=np.random.default_rng(4)
    raw=rng.normal(70,10,size=(180,60,8))
    train=np.arange(120)
    normalized,mask,normalizer=train_sensor_normalizer(raw,train)
    modified=raw.copy()
    modified[120:,:,:]+=20000.
    shifted,_,other=train_sensor_normalizer(modified,train)
    assert normalizer==other
    np.testing.assert_array_equal(normalized[:120],shifted[:120])
    assert mask.all()
    exposure=raw_exposure_features(normalized,mask)
    assert exposure.shape==(180,48)


def test_heldout_semantic_audit_is_run_level_not_high_frequency_label():
    rng=np.random.default_rng(77)
    n=180
    x=rng.normal(size=(n,60,8))
    concepts=rng.random(size=(n,60,2))
    metas=[]
    for i in range(n):
        name=("H" if i%2 else "N")
        metas.append({"ppm":{"ethylene":96. if name=="H" else 0.,
                              "co":0.,"methane":0.}})
        concepts[i,12:48,0]+=(.5 if name=="H" else 0.)
    output=semantic_audit(concepts,x,metas,np.arange(120,180),seed=82)
    assert output["n_independent_whole_test_runs"]==60
    assert output["max_abs_by_gas"]["ethylene"]["concept"]>.6
    assert output["max_abs_by_gas"]["co"]["concept"] is None
    assert run_level_spearman(np.ones(20),np.arange(20))["rho"] is None
