import io
import zipfile

import numpy as np
import pytest

from conceptfan_discovery.train_innovation_ica_bank_v63 import TrainOnlyInnovationBank
from scripts.experiment_real_dynamic_gas_ica_v63 import (
    _read_official_second_medians,split_windows_one_stream,
    train_only_sensor_scaling,_concept_ridge,_partial_rho,reg_metrics,
    semantic_scores,fixed_valid_semantic_alignment,
)


def _unlabeled_two_sources(n=320,seed=88):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(n,2))
    # Distinct non-Gaussian innovation sources, not a specified K.
    a[:,0]=np.sign(a[:,0])*np.abs(a[:,0])**.7
    a[:,1]=np.sign(a[:,1])*np.abs(a[:,1])**1.6
    w=np.zeros((2,16))
    w[0,:8]=rng.normal(1.,.1,size=8)
    w[1,8:]=rng.normal(1.,.1,size=8)
    x=rng.normal(scale=.15,size=(n,120,16))
    change=(np.linspace(0,1,120))[None,:,None]
    x+=change*(a@w)[:,None,:]
    mask=np.ones_like(x,dtype=bool)
    groups=np.asarray([f"session_{i%2}:{i//9}" for i in range(n)])
    return x,mask,groups


def test_blind_ica_source_finds_more_than_one_without_known_gas_labels():
    x,m,groups=_unlabeled_two_sources()
    bank=TrainOnlyInnovationBank(basis="ica",block_bootstraps=5)
    bank.fit_unlabeled(x,m,train_halfhour_blocks=groups)
    assert 2<=bank.k<=8
    assert bank.train_receipt["gas_labels_or_controller_setpoints_used"] is False
    assert bank.train_receipt["n_independent_continuous_recording_sessions"]==2
    assert len(bank.train_receipt["rank_bootstrap_20_blocks"])==5
    assert bank.public_passport()["source_risk_labels_used"] is False


def test_pca_and_ica_share_frozen_train_only_k_selection():
    x,m,groups=_unlabeled_two_sources()
    p=TrainOnlyInnovationBank(basis="pca",block_bootstraps=5).fit_unlabeled(
        x,m,train_halfhour_blocks=groups)
    i=TrainOnlyInnovationBank(basis="ica",block_bootstraps=5).fit_unlabeled(
        x,m,train_halfhour_blocks=groups)
    assert p.k==i.k
    assert i.sha!=p.sha
    assert i.k==len(i.tau)==len(i.width)
    assert i.route=="TRAIN_INNOVATION_ICA"


def test_masked_readings_are_ignored_and_projection_is_causal():
    x,m,groups=_unlabeled_two_sources()
    bank=TrainOnlyInnovationBank(basis="pca",block_bootstraps=5).fit_unlabeled(
        x,m,train_halfhour_blocks=groups)
    mask=m[:8].copy()
    mask[:,10:20,0]=False
    z,s=bank.transform(x[:8],mask)
    hidden=x[:8].copy()
    hidden[~mask]=np.nan
    zz,ss=bank.transform(hidden,mask)
    np.testing.assert_array_equal(z,zz)
    np.testing.assert_array_equal(s,ss)
    changed=x[:8].copy()
    changed[:,60:,:]+=1e5
    future,_=bank.transform(changed,mask)
    np.testing.assert_array_equal(z[:,:60],future[:,:60])


def test_time_split_embargos_windows_and_no_gas_sources_in_input():
    n=24000
    sec=np.arange(n)
    sensor=np.ones((n,16),dtype=float)
    refs=np.zeros((n,2),dtype=float)
    refs[:,0]=np.sin(sec/180)
    refs[:,1]=np.cos(sec/180)
    split,receipt=split_windows_one_stream(
        sec,sensor,refs,"ethylene_CO.txt")
    assert min(len(v) for v in split.values())>=20
    assert receipt["bilateral_embargo_seconds"]==120
    assert receipt["no_windows_crossing_temporal_boundary"]
    stop_train=max(z["first_recorded_second"]+119 for z in split["TRAIN"])
    start_valid=min(z["first_recorded_second"] for z in split["VALID"])
    assert start_valid-stop_train>=241
    assert split["TRAIN"][0]["sensor"].shape==(120,16)
    assert split["TRAIN"][0]["setpoint_last30"].shape==(2,)


def test_stream_rejects_broken_real_archive_line_and_header(tmp_path):
    path=tmp_path/"s.zip"
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("ethylene_CO.txt",
                   "BAD HEADER\n0.00 0.0 0.0 " + "1.0 "*16+"\n")
    with zipfile.ZipFile(path) as z,pytest.raises(ValueError,match="header"):
        _read_official_second_medians(z,"ethylene_CO.txt")


def test_train_sensor_scaling_never_looks_at_future_source():
    x,m,groups=_unlabeled_two_sources(n=320)
    source={"TRAIN":[],"VALID":[],"TEST":[]}
    for i in range(len(x)):
        cohort="TRAIN" if i<260 else "VALID" if i<290 else "TEST"
        source[cohort].append({
            "sensor":x[i],"setpoint_last30":np.array([i%5,i%3]),
            "source_file_code":0 if i%2 else 1,
            "train_halfhour_block":groups[i]})
    xx,mm,yy,ss,block,receipt=train_only_sensor_scaling(source)
    source["TEST"][0]["sensor"]+=99999.
    newer,*rest=train_only_sensor_scaling(source)
    np.testing.assert_array_equal(xx["TRAIN"],newer["TRAIN"])
    assert receipt["normalization_never_reads_valid_or_test"]
    assert len(block)==260


def test_exact_concept_only_ridge_contributions_and_no_ref_shortcut():
    rng=np.random.default_rng(90)
    x=rng.random((300,4))
    y=(x[:,0]-x[:,2])*50
    pred,error,head=_concept_ridge(x[:250],y[:250],x[250:])
    assert error<=1e-12
    assert head["raw_sensor_to_head_bypass"] is False
    assert np.isfinite(pred).all()
    assert np.abs(pred-y[250:]).mean()<2.


def test_semantic_partial_corr_and_valid_mapping_abstention():
    rng=np.random.default_rng(33)
    n=90
    eth=rng.normal(size=n)
    other=rng.normal(size=n)
    c=np.empty((n,120,2))
    c[:,:,0]=eth[:,None]+rng.normal(scale=.1,size=(n,120))
    c[:,:,1]=other[:,None]+rng.normal(scale=.1,size=(n,120))
    real=np.column_stack((other,eth))
    sensor=rng.normal(size=(n,120,16))
    session=np.r_[np.zeros(45,dtype=int),np.ones(45,dtype=int)]
    valid=semantic_scores(c,sensor,real,session)
    assert abs(_partial_rho(eth,eth,other))>.99
    assert valid["CO_recording"]["max_abs_conditional_by_target"][
        "ethylene"]["concept"]>.7
    mapping=fixed_valid_semantic_alignment(valid,valid)
    assert mapping["CO_recording"]["status"].startswith("VALID_LABEL_SELECTED")
    single=semantic_scores(c[:,:,:1],sensor,real,session)
    abstain=fixed_valid_semantic_alignment(single,single)
    assert abstain["CO_recording"]["status"]=="ABSTAIN_K_LT_2"


def test_negative_r2_and_reference_error_not_silently_excluded():
    y=np.array([1.,2.,3.]*10)
    p=np.zeros_like(y)
    a=reg_metrics(y,p,trainstd=1.)
    assert a["R2"]<0
    assert a["MAE_ppm"]>1
