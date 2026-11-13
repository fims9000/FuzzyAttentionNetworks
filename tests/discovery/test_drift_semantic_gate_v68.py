import numpy as np
import pytest

from scripts.experiment_drift_semantic_gate_v68 import (
    source_subspace_agreement,predeclared_early_valid_gate,
    MIN_GAS_SUPPORT_PER_VALID_BATCH
)
from conceptfan_discovery.static_gas_drift_concepts_v67 import (
    fix_semantic_alignment_from_valid
)


def make_synthetic_periods(seed=55):
    rng=np.random.default_rng(seed)
    # six independent physical labels and 4 anonymous statistical directions
    labels=np.tile(np.arange(1,7),30)
    labels=np.concatenate([labels,labels,labels])
    batches=np.repeat([8,9,10],len(labels)//3)
    vals=rng.normal(size=(len(labels),4))
    vals[labels==1,0]+=5.
    vals[labels==2,1]-=4.
    vals[labels==3,2]+=4.
    vals[labels==4,3]-=4.
    return vals,labels,batches


def test_subspace_overlap_invariant_to_orthogonal_axis_reparameterization():
    rng=np.random.default_rng(4)
    v=rng.normal(size=(200,4))
    q,_=np.linalg.qr(rng.normal(size=(4,4)))
    out=source_subspace_agreement(v,v@q)
    assert out["reference_rank"]==4
    np.testing.assert_allclose(
        out["all_squared_canonical_correlations"],
        np.ones(4),atol=1e-12)
    assert out["comparison_invariant_to_source_sign_and_permutation"]
    assert not out["overlap_proves_chemical_identity"]


def test_semantic_gate_only_depends_on_8_and_9_not_test_labels():
    scores,gas,batches=make_synthetic_periods()
    ix8=batches==8
    mapping,_=fix_semantic_alignment_from_valid(scores[ix8],gas[ix8])
    baseline=predeclared_early_valid_gate(
        mapping,scores,gas,batches,rank_is_stable=True)
    changed=gas.copy()
    changed[batches==10]=np.roll(changed[batches==10],17)
    mutated=predeclared_early_valid_gate(
        mapping,scores,changed,batches,rank_is_stable=True)
    assert len(mapping)==4
    for label in baseline:
        assert baseline[label]["status"]==mutated[label]["status"]
        if "VALID_stability_gate_failures" in baseline[label]:
            assert baseline[label]["VALID_stability_gate_failures"]==mutated[label]["VALID_stability_gate_failures"]
            assert baseline[label]["AUC_VALID8"]==mutated[label]["AUC_VALID8"]
            assert baseline[label]["AUC_VALID9_no_rematch"]==mutated[label]["AUC_VALID9_no_rematch"]
    assert any(z["status"].startswith("ABSTAIN_UNMATCHED") for z in baseline.values())


def test_nonstable_historical_rank_forces_matched_gas_abstention():
    score,y,batch=make_synthetic_periods()
    mapping,_=fix_semantic_alignment_from_valid(score[batch==8],y[batch==8])
    outcome=predeclared_early_valid_gate(
        mapping,score,y,batch,rank_is_stable=False)
    for gas in mapping:
        assert outcome[gas]["status"]=="ABSTAIN_EARLY_ASSOCIATION_UNSTABLE"
        assert "RANK_VARIES_ACROSS_TRAIN_HISTORY" in outcome[gas]["VALID_stability_gate_failures"]


def test_sparse_valid_references_do_not_pass_as_semantic_identity():
    score,y,batch=make_synthetic_periods()
    mapping,_=fix_semantic_alignment_from_valid(score[batch==8],y[batch==8])
    outcome=predeclared_early_valid_gate(
        mapping,score,y,batch,rank_is_stable=True)
    assert MIN_GAS_SUPPORT_PER_VALID_BATCH==20
    assert all(not v.get("physical_source_identity_proven",False) for v in outcome.values())


def test_wrong_reference_sensor_shape_is_rejected():
    rng=np.random.default_rng(14)
    x=rng.normal(size=(100,4))
    with pytest.raises(ValueError):
        source_subspace_agreement(x,np.zeros((99,4)))
    with pytest.raises(ValueError):
        source_subspace_agreement(x,np.full((100,4),np.nan))
