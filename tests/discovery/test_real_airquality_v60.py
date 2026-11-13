import zipfile

import numpy as np
import pandas as pd

from scripts.experiment_real_airquality_v60 import (
    SOURCE_COLUMNS,REFERENCE_COLUMNS,read_official_csv,
    build_disjoint_blocks,train_only_normalize,split_labels,
    semantic_association,
)


def fixture(n=6500):
    rng=np.random.default_rng(13)
    date=pd.date_range("2004-03-10 18:00:00",periods=n,freq="h")
    frame=pd.DataFrame({"Date":date.strftime("%d/%m/%Y"),
                        "Time":date.strftime("%H.%M.%S")})
    for name in SOURCE_COLUMNS:
        frame[name]=rng.normal(100,10,size=n)
    for name in REFERENCE_COLUMNS:
        frame[name]=rng.normal(5,1,size=n)
    return frame


def test_official_decimal_comma_and_hidden_missing_tag(tmp_path):
    frame=fixture()
    frame.loc[12,SOURCE_COLUMNS[0]]=-200.
    frame.loc[14,REFERENCE_COLUMNS[0]]=-200.
    archive=tmp_path/"uci.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("dataset/AirQualityUCI.csv",
                   frame.to_csv(sep=";",decimal=",",index=False).encode("latin-1"))
    parsed=read_official_csv(archive)
    assert len(parsed)==len(frame)
    assert np.isnan(parsed.loc[12,SOURCE_COLUMNS[0]])
    assert np.isnan(parsed.loc[14,REFERENCE_COLUMNS[0]])
    assert parsed.loc[0,"timestamp"]==pd.Timestamp("2004-03-10 18:00:00")


def test_disjoint_episode_splits_and_forecast_alignment():
    frame=fixture()
    frame.insert(0,"timestamp",pd.date_range("2004-03-10",periods=len(frame),freq="h"))
    result=build_disjoint_blocks(frame)
    assert result["raw_x"].shape==(500,12,5)
    assert result["reference_history"].shape==(500,12,2)
    assert result["reference_future"].shape==(500,2)
    assert tuple(map(len,result["splits"].values()))==(300,100,100)
    np.testing.assert_allclose(result["raw_x"][0,-1],
        frame.loc[11,list(SOURCE_COLUMNS)].to_numpy(dtype=float))
    np.testing.assert_allclose(result["reference_future"][0],
        frame.loc[12,list(REFERENCE_COLUMNS)].to_numpy(dtype=float))
    np.testing.assert_allclose(result["raw_x"][1,0],
        frame.loc[13,list(SOURCE_COLUMNS)].to_numpy(dtype=float))


def test_non_contiguous_hours_abstain_instead_of_interpolation():
    # Preserve the production >=300 TRAIN minimum even after discarding a bad block.
    frame=fixture(6630)
    frame.insert(0,"timestamp",pd.date_range("2004-03-10",periods=len(frame),freq="h"))
    frame.loc[10,"timestamp"]=frame.loc[10,"timestamp"]+pd.Timedelta(hours=6)
    result=build_disjoint_blocks(frame)
    assert result["n_discarded_noncontiguous_windows"]==1


def test_train_only_features_and_reference_thresholds_unchanged_by_test():
    rng=np.random.default_rng(4)
    raw=rng.normal(size=(500,12,5))*4+20
    raw[0,0,0]=np.nan
    train=np.arange(300)
    x,m,params=train_only_normalize(raw,train)
    alt=raw.copy();alt[300:]+=100000
    changed,_,alt_params=train_only_normalize(alt,train)
    assert params==alt_params
    np.testing.assert_array_equal(x[:300],changed[:300])
    assert x[0,0,0]==0 and not m[0,0,0]
    ref=rng.normal(size=(500,2))
    labels,observed,threshold=split_labels(ref,train,0)
    altered_ref=ref.copy();altered_ref[300:,0]+=99999
    _,_,other=split_labels(altered_ref,train,0)
    assert threshold==other
    assert observed.all()


def test_external_analyzer_permutation_negative_control():
    rng=np.random.default_rng(6)
    n=200
    c=rng.normal(size=(n,2))
    raw=rng.normal(size=(n,12,5))
    ref=np.empty((n,12,2))
    for j in range(2):
        ref[:,:,j]=c[:,j,None]+rng.normal(scale=.15,size=(n,12))
    results=semantic_association(c,raw,ref,np.arange(n))
    assert results["max_abs_spearman_by_reference"]["CO(GT)"]["concept"]>.75
    assert results["max_abs_spearman_by_reference"]["CO(GT)"]["permuted_concept"]<.4
    assert len(results["all_concept_reference_ranks"])==2




def test_five_sensor_unlabeled_source_grouping_not_artificially_six_only():
    from conceptfan_discovery.auto_measured_concept_proposals import form_train_measured_groups
    rng=np.random.default_rng(84)
    n=360;t=12
    latent=rng.normal(size=(n,t))
    x=rng.normal(size=(n,t,5))
    x[:,:,0]=latent+rng.normal(scale=.05,size=(n,t))
    x[:,:,1]=latent+rng.normal(scale=.05,size=(n,t))
    mask=rng.random(x.shape)>.12
    proposed=form_train_measured_groups(
        x,mask,known_sensor_indices=(),seed=17)
    assert any(tuple(g.sensor_indices)==(0,1) for g in proposed.candidates)
    assert proposed.outcomes_used_for_grouping is False

def test_no_evidence_no_invented_semantic_concept():
    rng=np.random.default_rng(9)
    c=np.empty((80,0))
    raw=rng.normal(size=(80,12,5))
    ref=np.full((80,12,2),np.nan)
    result=semantic_association(c,raw,ref,np.arange(80))
    assert result["max_abs_spearman_by_reference"]["CO(GT)"]["concept"] is None
    assert result["max_abs_spearman_by_reference"]["NOx(GT)"]["raw_sensor"] is None
