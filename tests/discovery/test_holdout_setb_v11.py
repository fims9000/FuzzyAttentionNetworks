"""Preflight fixtures for the ONE-SHOT Set B evaluation before viewing any B case."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from conceptfan_discovery.holdout_setb_v11 import (
    load_set_b_strict,predict_fixed_concept,_reference_score,
    METHODS,ARMS,SCENARIOS,SEEDS,FIT_ORDER,download_set_b_only_after_all_refit_guards_pass,
)
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.effect_certification_v11 import joint_concept_screen


def _fake_files(tmp_path:Path, *, future_offset:int=0,
                bad_inside_id:bool=False, overlap_id:bool=False):
    zipfile_path=tmp_path/"set-b.zip"
    target=tmp_path/"Outcomes-b.txt"
    ids=[901,902,903]
    rows=[]
    with zipfile.ZipFile(zipfile_path,"w") as zipout:
        for j,ident in enumerate(ids):
            label=int(j==1)
            rows.append({"RecordID":ident,"In-hospital_death":label,
                         "SOFA":10000+ident,"Survival":100000+ident})
            rec=[
                "Time,Parameter,Value",
                f"00:00,RecordID,{ident+1 if bad_inside_id and j==1 else ident}",
                f"00:17,HR,{75+j}",
                f"01:19,MAP,{65+j}",
                f"35:53,HR,{95+j}",
                "35:20,MAP,-1",
                f"36:00,HR,{1e9+future_offset+j}",
                f"46:10,MAP,{2e9+future_offset+j}",
                "48:00,HR,220",
            ]
            zipout.writestr(f"set-b/{ident}.txt","\n".join(rec)+"\n")
    pd.DataFrame(rows).to_csv(target,index=False)
    return zipfile_path,target,{901} if overlap_id else {801,802,803}


def _load(paths):
    zip_file,labels,a_ids=paths
    return load_set_b_strict(
        zip_file,labels,selected_channels=("HR","MAP"),
        medians=np.array([70.,50.]),scales=np.array([20.,10.]),
        set_a_record_ids=a_ids,history_hours=36,expected_n=3,
    )


def test_no_future_36h_to_48h_features_and_unchanged_train_normalization(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";a.mkdir();b.mkdir()
    x,receipt=_load(_fake_files(a,future_offset=0))
    y,another=_load(_fake_files(b,future_offset=999999999))
    np.testing.assert_array_equal(x.x,y.x)
    np.testing.assert_array_equal(x.mask,y.mask)
    np.testing.assert_array_equal(x.y,y.y)
    assert x.x.shape==(3,36,2)
    assert float(x.x[0,0,0])==pytest.approx(.25)
    assert float(x.x[0,1,1])==pytest.approx(1.5)
    assert x.mask[0,35,0]==1
    assert x.mask[0,35,1]==0
    assert x.x[0,35,1]==0
    assert receipt["no_future_labs_used"]
    assert receipt["same_mimic_ii_data_source_not_independent_hospital"]
    assert receipt["set_a_set_b_ids_disjoint"]
    assert receipt["number_positive"]==1
    assert "901" not in str(receipt)


def test_outcome_metadata_sofa_survival_or_future_data_never_enter_features(tmp_path):
    files=_fake_files(tmp_path)
    x,_=_load(files)
    frame=pd.read_csv(files[1])
    frame["SOFA"]=frame["SOFA"]+100000
    frame["Survival"]=frame["Survival"]-200000
    frame.to_csv(files[1],index=False)
    y,_=_load(files)
    np.testing.assert_array_equal(x.x,y.x)
    np.testing.assert_array_equal(x.mask,y.mask)


def test_disallow_a_b_record_overlap_and_wrong_internal_id(tmp_path):
    p=tmp_path/"p";p.mkdir()
    with pytest.raises(ValueError,match="overlap"):
        _load(_fake_files(p,overlap_id=True))
    q=tmp_path/"q";q.mkdir()
    with pytest.raises(ValueError,match="mismatch|disagree"):
        _load(_fake_files(q,bad_inside_id=True))


def test_never_allow_outcome_as_input_or_unnormalized_set_b(tmp_path):
    paths=_fake_files(tmp_path)
    with pytest.raises(ValueError):
        load_set_b_strict(
            paths[0],paths[1],selected_channels=("HR","In-hospital_death"),
            medians=np.array([1.,2.]),scales=np.ones(2),
            set_a_record_ids={801},expected_n=3,
        )
    with pytest.raises(ValueError):
        load_set_b_strict(
            paths[0],paths[1],selected_channels=("HR","MAP"),
            medians=np.array([70.,50.]),scales=np.array([1.,0.]),
            set_a_record_ids={801},expected_n=3,
        )


def test_dev_refit_guard_blocks_any_model_drift_before_set_b_opening():
    previous_v11=pd.DataFrame([
        {"method":"v10_consistency_refit","scenario":"original",
         "seed":11,"auprc_full":.4}
    ])
    other=pd.DataFrame([
        {"model":"transformer_timeseries","scenario":"original",
         "seed":11,"auprc":.37}
    ])
    a=_reference_score(
        "v10_consistency_refit",11,.4008,
        v11_records=previous_v11,external_records=other,tolerance=.002,
    )
    assert a["eligible_for_independent_B_evaluation"]
    with pytest.raises(RuntimeError,match="REFIT DRIFT"):
        _reference_score(
            "v10_consistency_refit",11,.391,
            v11_records=previous_v11,external_records=other,tolerance=.002,
        )
    with pytest.raises(ValueError):
        _reference_score(
            "transformer_timeseries",23,.37,
            v11_records=previous_v11,external_records=other,tolerance=.002,
        )


def test_concept_inference_uses_recorded_mask_in_separate_care_head():
    torch.manual_seed(4)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3,dtype=np.float64),
        channels=("HR","MAP","GCS","RespRate","Temp"),
        forecast_horizon=12,
    )
    x=np.random.default_rng(2).normal(size=(5,36,5)).astype(np.float32)
    source=(np.random.default_rng(4).random((5,36,5))>.15).astype(np.float32)
    visible=source.copy()
    visible[:,1::2,0]=0
    p,effect=predict_fixed_concept(
        model,x,visible,source,batch_size=2,collect_contributions=True,
    )
    assert p.shape==(5,) and effect.shape==(5,3)
    assert np.isfinite(p).all() and np.isfinite(effect).all()
    assert np.all((0<=p)&(p<=1))
    with pytest.raises(ValueError):
        predict_fixed_concept(model,x,source,visible)


def test_exact_registry_and_independent_patient_certificate_label():
    assert len(METHODS)==7
    assert len(ARMS)==3
    assert SCENARIOS==("original","mcar_thin","time_channel_mar")
    assert SEEDS==(11,23,37)
    cert={"groups":[{
        "first_slot_zero_index":0,"stop_exclusive":1,
        "slots_zero_index":[0],"certificate":"INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
    }]}
    pairs=pd.DataFrame([{
        "method":"v11_projector_group_view","seed_a":i,"seed_b":j,
        "slots":"[0]","certificate":"INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
        "same_patient_spearman":.95,"definition_subspace_overlap":.96,
    } for i,j in ((11,23),(11,37),(23,37))])
    out=joint_concept_screen(
        cert,pairs,already_repeatedly_inspected_development=False
    )
    assert out["already_repeatedly_inspected_development"] is False
    assert out["methods"]["v11_projector_group_view"]["n_candidate_individual_effects"]==1


def test_late_download_accepts_only_frozen_official_urls_and_never_overwrites(tmp_path):
    b=tmp_path/"set-b.zip"
    o=tmp_path/"Outcomes-b.txt"
    with pytest.raises(ValueError,match="exact pre-registered"):
        download_set_b_only_after_all_refit_guards_pass(
            destination_zip=b,destination_outcomes=o,
            source_zip_url="https://other.example/set-b.zip",
            outcome_url="https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt",
        )
    assert not b.exists() and not o.exists()
    b.write_bytes(b"EXISTING_HOLDOUT_FORBIDDEN")
    with pytest.raises(FileExistsError,match="absent BEFORE"):
        download_set_b_only_after_all_refit_guards_pass(
            destination_zip=b,destination_outcomes=o,
            source_zip_url="https://physionet.org/files/challenge-2012/1.0.0/set-b.zip",
            outcome_url="https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt",
        )
    assert b.read_bytes()==b"EXISTING_HOLDOUT_FORBIDDEN"
    assert not o.exists()


def test_original_external_train_order_executed_before_v11_new_arms():
    # Source external comparison originally trained V10 -> traditional
    # classifiers -> GRU-D -> Transformer, each with fixed input seeds.
    # The new experimental V11 arms must not affect that baseline's
    # surrounding training sequence or global random state.
    assert FIT_ORDER==(
        "v10_consistency_refit",
        "logistic_history",
        "hist_gradient_boosting",
        "gru_d_style",
        "transformer_timeseries",
        "v11_projector_axis_view",
        "v11_projector_group_view",
    )
    assert set(FIT_ORDER)==set(METHODS)
