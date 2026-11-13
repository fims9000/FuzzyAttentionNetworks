import numpy as np

from conceptfan_discovery.train_innovation_ica_bank_v63 import TrainOnlyInnovationBank
from conceptfan_discovery.dualstream_multifuzzy_bank_v65 import TrainOnlyDualStreamMultiFuzzyBank
from conceptfan_discovery.native_temporal_fuzzy_v56 import NativeSelfSupervisedConceptFAN
from scripts.experiment_dualstream_multifuzzy_v65 import (
    extract_all_concept_features,_fit_predict_with_receipt
)


def demo(n=320,seed=43):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(n,2))
    mix=rng.normal(size=(2,16))
    x=rng.normal(0,.18,size=(n,120,16))
    x+=a[:,None,:]@mix[None,:,:]
    # A second changing source is visible after an onset.
    x[:,30:]+=np.linspace(0,1,90)[None,:,None]*(a@mix)[:,None,:]
    mask=np.ones_like(x,dtype=bool)
    blocks=np.asarray([f"session_{i%2}:{i//8}" for i in range(n)])
    return x,mask,blocks


def bank():
    x,m,groups=demo()
    ica=TrainOnlyInnovationBank(basis="ica",block_bootstraps=5).fit_unlabeled(
        x,m,train_halfhour_blocks=groups)
    dual=TrainOnlyDualStreamMultiFuzzyBank(ica).fit_unlabeled(x,m)
    return x,m,ica,dual


def test_train_only_source_axis_count_does_not_equal_fake_gas_count():
    x,m,ica,dual=bank()
    assert dual.original_observable_k==ica.k
    assert dual.k==6*ica.k
    assert len(dual.tau)==len(dual.width)==dual.k
    assert (dual.width>0).all()
    assert dual.train_receipt["source_and_membership_thresholds_use_no_gas_labels"]
    assert dual.train_receipt["same_three_statistical_axes_are_not_eighteen_gases"]
    assert len(dual.quartiles)==2*ica.k
    original=dual.static_concepts(x[:5],m[:5])
    assert original.shape==(5,120,dual.k)
    assert np.isfinite(original).all()


def test_abs_and_delta_are_distinct_and_prefix_causal():
    x,m,ica,dual=bank()
    before,sm=dual.transform(x[:8],m[:8])
    different=x[:8].copy()
    different[:,65:]+=1e5
    after,_=dual.transform(different,m[:8])
    np.testing.assert_array_equal(before[:,:65],after[:,:65])
    changed=x[:8]+100.
    shifted,_=dual.transform(changed,m[:8])
    k=ica.k
    # First 3 membership duplicates for each of K absolute source axes,
    # then all DELTA sources; absolute changes while differences don't.
    assert not np.allclose(before[:,:,:3*k],shifted[:,:,:3*k])
    np.testing.assert_allclose(before[:,:,3*k:],
                               shifted[:,:,3*k:],atol=2e-5)


def test_missing_sensor_hidden_payload_neutral_states():
    x,m,ica,dual=bank()
    mask=m[:6].copy()
    mask[:,50:80,:]=False
    original,s=dual.transform(x[:6],mask)
    changed=x[:6].copy()
    changed[~mask]=np.nan
    result,ss=dual.transform(changed,mask)
    np.testing.assert_array_equal(original,result)
    np.testing.assert_array_equal(s,ss)
    fuzzy=dual.static_concepts(changed,mask)
    np.testing.assert_array_equal(fuzzy[:,50:80],.5+np.zeros_like(fuzzy[:,50:80]))



def test_missing_initial_baseline_neutralizes_only_delta_stream():
    x,m,ica,dual=bank()
    masked=m[:4].copy()
    masked[:,0,:]=False
    hidden=x[:4].copy()
    hidden[:,0,:]=np.nan
    c=dual.static_concepts(hidden,masked)
    K=ica.k
    np.testing.assert_array_equal(
        c[:,:,3*K:],np.full(c[:,:,3*K:].shape,.5))
    assert np.isfinite(c[:,:,:3*K]).all()



def test_native_fuzzy_training_has_no_target_argument_or_task_gradient():
    x,m,ica,dual=bank()
    model=NativeSelfSupervisedConceptFAN(
        dual,neural_seed=11,epochs=1).fit_source_unlabeled(x,m)
    assert model.is_source_frozen
    assert len(model.loss_receipt)==1
    assert np.isfinite(model.loss_receipt[0]["mean_unlabeled_total_loss"])
    features=extract_all_concept_features(x[:6],m[:6],ica,dual,model)
    assert features["v65_native_dualstream_36"].shape==(6,2*dual.k)
    assert features["v65_static_dualstream_36"].shape==(6,2*dual.k)


def test_exact_additive_risk_head_and_no_hidden_raw_shortcut():
    rng=np.random.default_rng(55)
    f=rng.random((120,36))
    y=rng.normal(size=120)
    pred,receipt=_fit_predict_with_receipt(
        "v65_native_dualstream_36",f[:100],y[:100],f[100:])
    assert pred.shape==(20,)
    assert receipt["max_additive_error"]<=1e-12
    assert receipt["risk_raw_sensor_bypass"] is False
    assert receipt["n_fuzzy_membership_state_time_summaries"]==36
