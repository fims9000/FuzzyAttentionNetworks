"""V54 pre-TEST contracts for separate label worlds and native source head."""
import numpy as np
import torch
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,infer_operator_concept_v54,OP_EVEN,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    generate,energy_tabular,added_logits,TRAIN_N,VALID_N,TEST_N,native_k2,
    native_logits,
)

def test_splits_and_unsupervised_operator_bank():
    w=generate()
    assert [len(w[k]) for k in ("train","valid1","valid2","test")]==[
        TRAIN_N,VALID_N,VALID_N,TEST_N]
    sets=[set(w[k]) for k in ("train","valid1","valid2","test")]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i+1,4))
    x=w["x"][w["train"]];m=w["masks"]["train"]
    c=discover_operator_library_v54(x,m)
    groups={(min(v.source_channel,v.readout_channel),
             max(v.source_channel,v.readout_channel)):v.family for v in c.candidates}
    assert groups[(4,5)]==OP_EVEN
    assert groups[(8,9)]==OP_EVEN
    assert len(groups)==3
    assert c==discover_operator_library_v54(x,m)
    assert not any(v.discovery_uses_target_y for v in c.candidates)

def test_k2_exact_abstention_and_signed_additivity():
    w=generate()
    x=w["x"][w["test"][:8]]
    m=np.ones_like(x,dtype=bool)
    model=native_k2(11)
    baseline=native_logits(model,x,m)
    z,used,q,s=added_logits(baseline,None,0.,x,m,"operator")
    np.testing.assert_array_equal(z,baseline)
    assert not used.any()
    c=discover_operator_library_v54(w["x"][w["train"]],
                                    w["masks"]["train"]).candidates[0]
    z,used,q,s=added_logits(baseline,c,1.75,x,m,"operator")
    np.testing.assert_allclose(z,baseline+np.where(used,1.75*(q-.5),0.),
                               atol=0,rtol=0)
    assert used.any()

def test_engineered_baseline_has_no_simulator_source_inputs():
    w=generate()
    x=w["x"][w["train"][:12]]
    m=w["masks"]["train"][:12]
    feat=energy_tabular(x,m)
    assert feat.shape[0]==12
    assert feat.shape[1]>x.shape[-1]
    assert np.isfinite(feat).all()

def test_y_not_needed_by_operator_discovery():
    w=generate()
    x=w["x"][w["train"]]
    m=w["masks"]["train"]
    discovered=discover_operator_library_v54(x,m)
    assert not np.array_equal(w["y"]["nonlinear_energy"],w["y"]["irrelevant_delays"])
    assert discovered==discover_operator_library_v54(x,m)
