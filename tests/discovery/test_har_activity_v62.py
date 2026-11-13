"""V62 clean separate-source HAR activity tests (all test arrays fabricated).

No real UCI HAR TEST data or target values are accessed in tests. The TEST
dataset remains unread until all native TRAIN/VALID models are frozen.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from conceptfan_discovery.har_activity_v62 import (
    TASKS,SENSOR_NAMES,collapse_raw_axes_v62,load_har_v62,
    load_har_test_v62,stress_har_v62,rename_bank_as_har_v62,
)
from conceptfan_discovery.program_compiler_v61 import (
    compile_allowed_programs_v61,select_conditioning_and_novel_programs_v61,
)
from conceptfan_discovery.program_compiler_v60 import infer_program_matrix_v60
from scripts.experiment_conceptfan_har_v62 import _axis_features


@pytest.fixture(scope="module")
def simulated_official_source(tmp_path_factory):
    root=tmp_path_factory.mktemp("fake_uci_har_full")
    for split,person_ids in (("train",range(1,22)),("test",range(22,31))):
        rng=np.random.default_rng(773 if split=="train" else 774)
        count_per_person=12
        people=[];outcomes=[]
        sources=[]
        for person in person_ids:
            for label in range(1,7):
                for _ in range(count_per_person):
                    # Purely synthetic fixtures, not scientific outcomes.
                    movement=(1.5 if label in (2,3) else .5)
                    acc=rng.normal(size=(128,3))*.25
                    gyro=rng.normal(size=(128,3))*.18
                    acc[:,0]+=movement*(1 if label in (2,5) else -.6)
                    gyro[:,1]+=(.8 if label in (2,5) else -.4)
                    sources.append(np.column_stack((acc,gyro)))
                    people.append(person);outcomes.append(label)
        np.savez_compressed(root/f"private_{split}.npz",
                            axes=np.asarray(sources,dtype=np.float32),
                            subject=np.asarray(people,dtype=np.int32),
                            y=np.asarray(outcomes,dtype=np.int32))
    return root


def test_clean_HAR_raw_axis_to_16_causal_block_sensor_programs():
    n=2
    raw=np.zeros((n,128,6))
    raw[:,:,0]=3.;raw[:,:,1]=4.
    raw[:,:,4]=-6.
    result=collapse_raw_axes_v62(raw)
    assert result.shape==(n,16,3)
    np.testing.assert_array_equal(result[:,:,0],np.full((n,16),5.))
    np.testing.assert_array_equal(result[:,:,1],np.full((n,16),6.))
    np.testing.assert_array_equal(result[:,:,2],np.full((n,16),-6.))
    ft=_axis_features(raw,(0,1,2))
    assert ft.shape==(n,16*3*2+3*2)


def test_no_HAR_TEST_opened_during_training_valid_source_compilation(simulated_official_source):
    root=simulated_official_source
    # Temporarily rename TEST, ensure TRAIN+VALID works before TEST exists.
    source=root/"private_test.npz"
    hidden=root/"private_test.secure"
    source.rename(hidden)
    try:
        dev,norm,receipt=load_har_v62(root,"STAIRS_UP_VS_DOWN")
        assert set(dev)=={"train","valid1","valid2"}
        assert receipt["official_TEST_opened_only_after_ALL_train_VALID_fit"]
        assert {k:len(np.unique(v["patient"])) for k,v in dev.items()}=={
            "train":13,"valid1":4,"valid2":4}
        assert sum(len(v["y"]) for v in dev.values())==21*24
        assert not norm.excluded_target_channel_was_read_for_scaling
    finally:
        hidden.rename(source)
    _,norm,_=load_har_v62(root,"STAIRS_UP_VS_DOWN")
    test=load_har_test_v62(root,"STAIRS_UP_VS_DOWN",norm)
    assert len(test["y"])==9*24
    assert len(np.unique(test["patient"]))==9
    assert np.isfinite(test["x"]).all()


def test_stairs_task_and_static_negative_control_person_split(simulated_official_source):
    root=simulated_official_source
    for task in TASKS:
        data,norm,receipt=load_har_v62(root,task)
        held=load_har_test_v62(root,task,norm)
        assert set(data["train"]["patient"]).isdisjoint(set(held["patient"]))
        assert set(np.unique(held["y"]))=={0,1}
        assert receipt["precomputed_561_vector_not_used"]
        assert data["train"]["x"].shape[1:]==(16,3)
        assert data["train"]["axes"].shape[1:]==(128,6)
        assert np.isfinite(held["x"]).all()
        assert len(held["y"])==9*24


def test_outcome_blind_gyro_concepts_never_include_accel(simulated_official_source):
    data,_,_=load_har_v62(simulated_official_source,"STAIRS_UP_VS_DOWN")
    train=data["train"]
    bank=rename_bank_as_har_v62(
        compile_allowed_programs_v61(
            train["x"],train["mask"],train["patient"],
            allowed_channels=(0,1,2),seed=20266213))
    p,n=select_conditioning_and_novel_programs_v61(bank,(0,1,2))
    assert p and n
    assert all(0 not in bank.candidates[i].channels for i in n)
    assert all(bank.candidates[i].channel_names[0] in SENSOR_NAMES
               for i in range(len(bank.candidates)))
    z,s=infer_program_matrix_v60(
        bank,train["x"],train["mask"])
    assert z.shape[0]==len(train["y"])
    assert np.isfinite(z).all()


def test_gyro_dropout_gain_and_accel_dropout_preserve_correct_source_mask(simulated_official_source):
    dev,norm,_=load_har_v62(simulated_official_source,"STAND_VS_SIT")
    held=load_har_test_v62(simulated_official_source,"STAND_VS_SIT",norm)
    natural,orig,axes,mask=stress_har_v62(held,norm,"natural",seed=12)
    nogyro,m2,ar2,device2=stress_har_v62(
        held,norm,"gyro_missing",seed=12)
    assert not m2[:,:,1:].any()
    np.testing.assert_array_equal(natural[:,:,0],nogyro[:,:,0])
    assert np.isfinite(nogyro).all()
    noacc,m3,ar3,device3=stress_har_v62(
        held,norm,"accel_missing",seed=12)
    assert not m3[:,:,0].any()
    np.testing.assert_array_equal(natural[:,:,1:],noacc[:,:,1:])
    gain,m4,ar4,device4=stress_har_v62(
        held,norm,"gyro_gain_1p5",seed=12)
    np.testing.assert_array_equal(natural[:,:,0],gain[:,:,0])
    np.testing.assert_allclose(ar4[:,:,3:],
                               axes[:,:,3:]*1.5,atol=1.e-6)
    dropped,m5,ar5,device5=stress_har_v62(
        held,norm,"MCAR20",seed=12)
    assert (m5.mean()<orig.mean()) and m5.shape==natural.shape
    assert np.isfinite(dropped).all()


def test_original_har_test_volunteers_cannot_reappear_in_training(simulated_official_source):
    root=simulated_official_source
    with np.load(root/"private_test.npz",allow_pickle=False) as z:
        copied={k:z[k].copy() for k in z.files}
    modified=copied.copy()
    modified["subject"]=np.repeat(np.arange(1,10),72).astype(np.int32)
    invalid=root/"private_test.npz"
    np.savez_compressed(invalid,**modified)
    try:
        dev,norm,_=load_har_v62(root,"STAND_VS_SIT")
        with pytest.raises(AssertionError,match="overlap"):
            load_har_test_v62(root,"STAND_VS_SIT",norm)
    finally:
        np.savez_compressed(invalid,**copied)
