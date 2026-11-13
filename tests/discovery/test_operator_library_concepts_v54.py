"""V54 unit tests: unknown nonlinear concepts, strict masking, parity and controls."""
import numpy as np
import pytest
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,infer_operator_concept_v54,
    fingerprint_v54,OP_EVEN,OP_SIGNED,
)

def synthetic(seed=20265400,n=420,mode="even"):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,16,10))
    source=rng.normal(size=(n,16))
    x[:,:,4]=source+.035*rng.normal(size=(n,16))
    x[:,:2,5]=rng.normal(size=(n,2))
    if mode=="even":
        orientation=rng.choice([-1.,1.],size=(n,14))
        x[:,2:,5]=orientation*source[:,:-2]+.035*rng.normal(size=(n,14))
    else:
        x[:,2:,5]=source[:,:-2]+.035*rng.normal(size=(n,14))
    mask=rng.random(x.shape)>=.2
    return x,mask

def test_even_family_discovered_without_known_sensors_or_y():
    x,m=synthetic()
    bank=discover_operator_library_v54(x,m)
    assert len(bank.candidates)==1
    c=bank.candidates[0]
    assert c.family==OP_EVEN
    assert (c.source_channel,c.readout_channel,c.lag_steps)==(4,5,2)
    assert c.min_fold_signed_correlation>.7
    assert not c.supports_physical_causal_name
    assert not c.discovery_uses_target_y
    assert bank==discover_operator_library_v54(x,m)
    assert fingerprint_v54(bank)==fingerprint_v54(discover_operator_library_v54(x,m))

def test_signed_families_remain_available():
    x,m=synthetic(mode="signed")
    bank=discover_operator_library_v54(x,m)
    assert len(bank.candidates)==1
    assert bank.candidates[0].family==OP_SIGNED

def test_even_sign_inversion_and_missingness_neutrality():
    x,m=synthetic()
    c=discover_operator_library_v54(x,m).candidates[0]
    q=infer_operator_concept_v54(c,x,m)
    flipped=x.copy()
    flipped[:,:,4]*=-1.
    flipped[:,:,5]*=-1.
    other=infer_operator_concept_v54(c,flipped,m)
    for key in ("summary","trajectory","support","available"):
        np.testing.assert_allclose(q[key],other[key],atol=0,rtol=0)
    hidden=x.copy();hidden[~m]=np.nan
    no_leak=infer_operator_concept_v54(c,hidden,m)
    for key in ("summary","trajectory","support","available"):
        np.testing.assert_array_equal(q[key],no_leak[key])
    all_missing=infer_operator_concept_v54(c,x,np.zeros_like(m))
    np.testing.assert_array_equal(all_missing["summary"],np.full(len(x),.5))
    np.testing.assert_array_equal(all_missing["support"],np.zeros(len(x)))

def test_prefix_causality_and_pair_shuffle():
    x,m=synthetic()
    c=discover_operator_library_v54(x,m).candidates[0]
    original=infer_operator_concept_v54(c,x,m)
    future=x.copy()
    future[:,12:,:]+=30.
    changed=infer_operator_concept_v54(c,future,m)
    np.testing.assert_array_equal(original["trajectory"][:,:12],changed["trajectory"][:,:12])
    shuffled=x.copy()
    shuffled[:,:,5]=np.roll(shuffled[:,:,5],1,axis=0)
    assert not discover_operator_library_v54(shuffled,m).candidates

def test_corrupt_observed_and_invalid_mask_are_rejected():
    x,m=synthetic()
    x[0,0,4]=np.nan
    m[0,0,4]=True
    with pytest.raises(ValueError):
        discover_operator_library_v54(x,m)
    x,m=synthetic()
    m=m.astype(int);m[0,0,4]=2
    with pytest.raises(ValueError):
        discover_operator_library_v54(x,m)

def test_irrelevant_noise_does_not_create_pairs():
    rng=np.random.default_rng(76)
    x=rng.normal(size=(420,16,10))
    mask=rng.random(x.shape)>=.20
    assert len(discover_operator_library_v54(x,mask).candidates)==0
