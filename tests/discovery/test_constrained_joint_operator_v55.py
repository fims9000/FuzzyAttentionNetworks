"""V55 constrained jointly learned concept: tensor, masking and risk contracts."""
import numpy as np
import pytest
import torch
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,infer_operator_concept_v54,OP_EVEN,
)
from conceptfan_discovery.constrained_joint_operator_v55 import (
    ConstrainedMeasuredOperatorV55,fit_joint_operator_v55,
    infer_joint_operator_v55,TRAIN_STEPS,
)
from scripts.experiment_conceptfan_operator_library_v54 import generate


@pytest.fixture(scope="module")
def sample():
    w=generate(seed=20265505)
    tr=w["train"]
    x=w["x"][tr]
    m=w["masks"]["train"]
    c=discover_operator_library_v54(x,m).candidates[0]
    assert c.family==OP_EVEN
    y=w["y"]["nonlinear_energy"][tr]
    return x,m,c,y


def test_initial_native_measured_membership_is_exact_v54(sample):
    x,m,c,_=sample
    native=ConstrainedMeasuredOperatorV55(c)
    x_safe=np.where(m,x,0)
    with torch.no_grad():
        q,s,_,_=native.memberships(
            torch.as_tensor(x_safe,dtype=torch.float64),
            torch.as_tensor(m,dtype=torch.bool))
    v54=infer_operator_concept_v54(c,x,m)
    np.testing.assert_allclose(q.numpy(),v54["summary"],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(s.numpy(),v54["support"])


def test_train_joint_operator_has_only_bounded_interpretable_parameters(sample):
    x,m,c,y=sample
    baseline=np.zeros(len(y),np.float64)
    fit=fit_joint_operator_v55(c,x,m,baseline,y)
    assert fit.n_train==len(x)
    assert fit.train_objective_last<fit.train_objective_first
    assert fit.source_weight>0 and fit.readout_weight>0
    assert fit.source_weight+fit.readout_weight==pytest.approx(1.,abs=1e-10)
    assert .5<=fit.temperature<=2.
    assert abs(fit.gamma)<=10.
    assert fit.supervised_definition_tuned_on_train_only
    assert not fit.physical_identity_attested
    q=infer_joint_operator_v55(fit,x,m)
    assert (q["summary"]>=0).all() and (q["summary"]<=1).all()
    # Pair sensor program is not changed by supervised risk fitting.
    assert (fit.candidate.source_channel,fit.candidate.readout_channel)==(
            c.source_channel,c.readout_channel)


def test_even_joint_concept_sign_parity_and_mask_neutrality(sample):
    x,m,c,y=sample
    fit=fit_joint_operator_v55(c,x,m,np.zeros(len(y)),y)
    orig=infer_joint_operator_v55(fit,x,m)
    flipped=x.copy()
    flipped[:,:,c.source_channel]*=-1
    flipped[:,:,c.readout_channel]*=-1
    after=infer_joint_operator_v55(fit,flipped,m)
    np.testing.assert_allclose(orig["summary"],after["summary"],atol=0,rtol=0)
    hidden=x.copy();hidden[~m]=np.nan
    ignored=infer_joint_operator_v55(fit,hidden,m)
    np.testing.assert_array_equal(orig["summary"],ignored["summary"])
    absent=infer_joint_operator_v55(fit,x,np.zeros_like(m))
    np.testing.assert_array_equal(absent["summary"],np.full(len(x),.5))
    np.testing.assert_array_equal(absent["support"],np.zeros(len(x)))


def test_observed_corruption_refused_and_no_future_at_t(sample):
    x,m,c,y=sample
    fit=fit_joint_operator_v55(c,x,m,np.zeros(len(y)),y)
    before=infer_joint_operator_v55(fit,x,m)["trajectory"]
    future=x.copy();future[:,12:,:]+=200.
    after=infer_joint_operator_v55(fit,future,m)["trajectory"]
    np.testing.assert_array_equal(before[:,:12],after[:,:12])
    bad=x.copy();mask=m.copy()
    bad[0,0,c.source_channel]=np.nan
    mask[0,0,c.source_channel]=True
    with pytest.raises(ValueError):
        infer_joint_operator_v55(fit,bad,mask)
