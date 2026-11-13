import inspect
import io
from zipfile import ZipFile

import numpy as np
import pytest

import conceptfan_discovery.static_gas_drift_concepts_v67 as v
from scripts.experiment_real_six_gas_drift_v67 import (
    exact_concept_head,measure_multiclass
)


def synthetic(n=1300,seed=70):
    rng=np.random.default_rng(seed)
    latent=rng.normal(size=(n,10))
    mixing=rng.normal(size=(10,128))
    return 300.+latent@mixing+rng.normal(size=(n,128))


@pytest.fixture(scope="module")
def fixed_source():
    source=synthetic()
    return source,v.FrozenSixGasSourceBank().fit_unlabeled(source)


def test_exact_uci_libsvm_schema_observed_physical_features():
    row="2;10.000000 "+" ".join(
        f"{i}:{i*.1:.6f}" for i in range(1,129))
    gas,ppm,values=v.parse_uci_record(row.encode())
    assert gas==2 and ppm==10.
    assert values.shape==(128,)
    assert values[0]==.1 and values[-1]==12.8
    with pytest.raises(ValueError):
        v.parse_uci_record(row.replace("128:12.800000","2:12.800000").encode())
    with pytest.raises(ValueError):
        v.parse_uci_record(row.replace("128:12.800000","").encode())
    with pytest.raises(ValueError):
        v.parse_uci_record(row.replace("128:12.800000","128:nan").encode())


def test_all_ten_real_batches_disjoint_time_groups(monkeypatch,tmp_path):
    monkeypatch.setattr(v,"UCI_N",60)
    file=tmp_path/"original_uci270.zip"
    row=" ".join(f"{i}:{i*.1}" for i in range(1,129))
    with ZipFile(file,"w") as zipfile:
        for b in range(1,11):
            records=[f"{k};{10*b:.1f} {row}" for k in range(1,7)]
            zipfile.writestr(f"batch{b}.dat","\n".join(records)+"\n")
    x,y,ppm,batch,groups,receipt=v.read_original_ten_batches(file)
    assert x.shape==(60,128)
    assert [len(groups[k]) for k in ("TRAIN","VALID","TEST")]==[42,12,6]
    assert set(batch[groups["TRAIN"]])==set(range(1,8))
    assert set(batch[groups["VALID"]])=={8,9}
    assert set(batch[groups["TEST"]])=={10}
    assert receipt["per_sensor_features_are_precomputed_not_actual_timestamps"]


def test_fuzzy_train_only_not_label_or_physical_identity(fixed_source):
    x,bank=fixed_source
    sig=inspect.signature(v.FrozenSixGasSourceBank.fit_unlabeled)
    assert list(sig.parameters)==["self","train_x"]
    assert 1<=bank.K<=12
    assert bank.quantiles.shape==(bank.K,3)
    assert bank.uncapped_K>=bank.K
    assert bank.receipt["training_source_labels_or_gas_ppm_used"] is False
    assert bank.receipt["source_axes_have_no_labels_or_identified_chemical_identity"]
    z=bank.source_levels(x[:10])
    c=bank.fuzzy(x[:10])
    assert z.shape==(10,bank.K) and c.shape==(10,3*bank.K)
    assert np.isfinite(c).all() and np.all((c>=0)&(c<=1))


def test_feature_views_are_two_engineered_descriptors_groups_not_time():
    example=np.arange(128,dtype=float).reshape(1,16,8)
    mapped=v.FrozenSixGasSourceBank.sensor_views(example)
    assert mapped.shape==(1,32)
    assert mapped[0,0]==.5
    assert mapped[0,15]==(120.+121.)/2
    assert mapped[0,16]==np.mean(np.arange(2,8))
    assert mapped[0,-1]==np.mean(np.arange(122,128))


def test_train_sensor_fitting_no_test_dependence(fixed_source):
    x,bank=fixed_source
    sha=bank.sha
    original=bank.fuzzy(x[120:140])
    other=x[120:140].copy()
    other[:,0]+=1e6
    assert bank.sha==sha
    assert not np.array_equal(original,bank.fuzzy(other))
    assert bank.sha==sha
    with pytest.raises(ValueError):
        bank.fuzzy(np.zeros((8,127)))
    with pytest.raises(ValueError):
        bank.fuzzy(np.full((8,128),np.nan))


def test_exact_six_class_risk_additivity_without_raw_shortcut(fixed_source):
    x,bank=fixed_source
    y=np.tile(np.arange(1,7),200)
    p,receipt=exact_concept_head(
        bank.fuzzy(x[:1200]),y,bank.fuzzy(x[1200:]))
    assert p.shape==(100,6)
    assert np.allclose(p.sum(axis=1),1.)
    assert receipt["max_abs_probability_additive_reconstruction_error"]<1e-9
    assert receipt["no_raw_input_to_risk_head_bypass"]
    assert len(receipt["per_class_beta_per_fuzzy_state"])==6


def test_semantic_matching_on_VALID_does_not_manual_force_six_axes():
    rng=np.random.default_rng(12)
    y=np.repeat(np.arange(1,7),30)
    z=rng.normal(size=(len(y),3))
    z[y==1,0]+=4.
    z[y==2,1]+=4.
    z[y==3,2]+=4.
    fixed,valid=v.fix_semantic_alignment_from_valid(z,y)
    assert len(fixed)==3
    result=v.audit_fixed_mapping(
        fixed,valid,z,y,np.tile(np.arange(1,7),30).astype(float))
    assert len(result["mapping"])==6
    assert len([q for q in result["mapping"].values()
                if q["status"].startswith("ABSTAIN")])==3
    assert result["valid_labels_used_only_for_postfit_semantic_mapping"]
    assert not result["unique_physical_gas_concepts_certified"]


def test_multiclass_macro_metrics_and_negative_class_support():
    y=np.repeat(np.arange(1,7),4)
    p=np.full((24,6),.02)
    p[np.arange(24),y-1]=.9
    m=measure_multiclass(y,p)
    assert m["accuracy"]==1.
    assert m["macro_F1"]==1.
    assert len(m["confusion_matrix_gas_1_to_6"])==6
    assert m["multiclass_log_loss"]>=0
