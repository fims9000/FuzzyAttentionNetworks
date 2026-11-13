"""V53 native, pre-TEST generator and separation contracts."""
import numpy as np
import torch
from conceptfan_discovery.lagged_operator_concepts_v53 import (
    discover_lagged_concepts_v53,infer_lagged_concept_v53)
from scripts.experiment_conceptfan_lagged_native_v53 import (
    generate,masked,native_k2,native_logits,TRAIN_N,VALID_N,TEST_N)

def test_splits_and_unsupervised_source_choices():
    world=generate()
    assert len(world["train"])==TRAIN_N
    assert len(world["valid1"])==VALID_N
    assert len(world["valid2"])==VALID_N
    assert len(world["test"])==TEST_N
    ids=[set(world[k]) for k in ("train","valid1","valid2","test")]
    assert all(not ids[i]&ids[j] for i in range(4) for j in range(i+1,4))
    x=world["x"][world["train"]]
    m=world["masks"]["train"]
    c=discover_lagged_concepts_v53(x,m)
    groups={frozenset((z.source_channel,z.readout_channel)):z.lag_steps
            for z in c.candidates}
    assert groups=={
        frozenset((4,5)):2,
        frozenset((6,7)):2,
        frozenset((8,9)):2,
    }
    assert all(not z.medical_identity_verified and z.lag_steps==2
               for z in c.candidates)

def test_label_modes_do_not_affect_unsupervised_concepts():
    world=generate()
    x=world["x"][world["train"]]
    m=world["masks"]["train"]
    a=discover_lagged_concepts_v53(x,m)
    assert not np.array_equal(world["y"]["physical_extra"],
                              world["y"]["irrelevant_source"])
    b=discover_lagged_concepts_v53(x,m)
    assert a==b

def test_native_model_risk_additivity_and_exact_missing_fallback():
    world=generate()
    x=world["x"][world["test"][:8]]
    m=masked(x,"MCAR20",seed=101)
    model=native_k2(11)
    model.eval()
    with torch.no_grad():model.gamma.fill_(1.75)
    baseline=native_logits(model,x,m)
    q=np.linspace(.1,.9,len(x)).astype(np.float32)
    supported=np.array([True,False,True,False,True,False,True,False])
    extended=native_logits(model,x,m,q,supported)
    np.testing.assert_array_equal(extended[~supported],baseline[~supported])
    np.testing.assert_allclose(
        extended,baseline+1.75*np.where(supported,q-.5,0.),
        rtol=0,atol=2e-6)

def test_irrelevant_corrupted_mask_payload_never_enters_native_risk():
    world=generate()
    x=world["x"][world["test"][:6]]
    m=masked(x,"MCAR40",seed=17)
    model=native_k2(11)
    model.eval()
    baseline=native_logits(model,x,m)
    hidden=x.copy();hidden[~m]=np.nan
    second=native_logits(model,hidden,m)
    np.testing.assert_array_equal(baseline,second)
