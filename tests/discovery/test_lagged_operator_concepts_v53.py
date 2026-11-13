import numpy as np
import pytest
from conceptfan_discovery.lagged_operator_concepts_v53 import (
    discover_lagged_concepts_v53,infer_lagged_concept_v53,
    discovery_fingerprint_v53,
)

def fixture(seed=73,n=420):
    rng=np.random.default_rng(seed)
    x=rng.normal(0,.7,(n,16,8))
    src=rng.normal(0,1,(n,16))
    x[:,:,4]=src+rng.normal(0,.08,(n,16))
    x[:,:,5]=rng.normal(0,1,(n,16))
    x[:,2:,5]=src[:,:-2]+rng.normal(0,.08,(n,14))
    x[:,:,6]=rng.normal(size=(n,16))
    x[:,:,7]=rng.normal(size=(n,16))
    return x,rng.random(x.shape)>=.18

def test_discovery_reveals_lag_two_not_instantaneous():
    x,m=fixture()
    d=discover_lagged_concepts_v53(x,m)
    assert len(d.candidates)==1
    c=d.candidates[0]
    assert (c.source_channel,c.readout_channel,c.lag_steps)==(4,5,2)
    assert c.max_abs_zero_lag_corr<.25
    assert c.min_fold_signed_corr>.9
    assert not c.outcome_labels_used and not c.medical_identity_verified
    assert discovery_fingerprint_v53(d)==discovery_fingerprint_v53(
        discover_lagged_concepts_v53(x,m))

def test_missing_payload_and_temporal_prefix_invariant():
    x,m=fixture()
    c=discover_lagged_concepts_v53(x,m).candidates[0]
    orig=infer_lagged_concept_v53(c,x,m)
    filled=x.copy();filled[~m]=np.nan
    after=infer_lagged_concept_v53(c,filled,m)
    for key in ("trajectory","available","summary","support"):
        np.testing.assert_array_equal(orig[key],after[key])
    future=x.copy();future[:,12:,:]+=1000
    changed=infer_lagged_concept_v53(c,future,m)
    np.testing.assert_array_equal(orig["trajectory"][:,:12],
                                  changed["trajectory"][:,:12])

def test_all_missing_yields_exact_neutral_fallback():
    x,m=fixture()
    c=discover_lagged_concepts_v53(x,m).candidates[0]
    r=infer_lagged_concept_v53(c,x,np.zeros_like(m))
    np.testing.assert_array_equal(r["summary"],np.full(len(x),.5))
    np.testing.assert_array_equal(r["support"],np.zeros(len(x)))
    np.testing.assert_array_equal(r["trajectory"],np.full((len(x),16),.5))

def test_shuffling_pair_breaks_train_only_discovery():
    x,m=fixture()
    shuffled=x.copy()
    shuffled[:,:,5]=np.roll(shuffled[:,:,5],1,axis=0)
    assert not discover_lagged_concepts_v53(shuffled,m).candidates

def test_unseen_sign_inversion_is_not_claimed_as_robustness():
    x,m=fixture()
    c=discover_lagged_concepts_v53(x,m).candidates[0]
    before=infer_lagged_concept_v53(c,x,m)["summary"]
    flipped=x.copy();flipped[:,:,4:6]*=-1
    after=infer_lagged_concept_v53(c,flipped,m)["summary"]
    assert np.mean(np.abs(before-after))>.03
    assert c.source_identity=="MEASUREMENT_ASSOCIATION_ONLY"

@pytest.mark.parametrize("invalid",[0,1])
def test_bad_mask_or_observed_nan_rejected(invalid):
    x,m=fixture()
    if invalid==0:
        m=m.astype(int);m[0,0,0]=2
    else:
        x[0,0,0]=np.nan;m[0,0,0]=True
    with pytest.raises(ValueError):
        discover_lagged_concepts_v53(x,m)
