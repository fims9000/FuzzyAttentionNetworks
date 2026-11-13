"""V60 native trainable program-concept bottleneck contracts."""
import numpy as np
import pytest
import torch
from scipy.special import expit

from conceptfan_discovery.program_compiler_v60 import (
    compile_measured_programs_v60,infer_program_matrix_v60,
)
from conceptfan_discovery.program_fan_v60 import (
    SparseCompiledConceptHeadV60,fit_program_head_v60,
    infer_program_head_v60,admit_program_head_v60,
    apply_program_admission_v60,group_macro_nll_v60,
)


def cohort(n=360,seed=77):
    rng=np.random.default_rng(seed)
    group=np.repeat(np.arange(18),n//18)
    individual=rng.normal(size=(n,1,3))
    x=individual+.2*rng.normal(size=(n,16,3))
    m=rng.random(x.shape)>.15
    y=(rng.random(n)<expit(1.8*individual[:,0,0]-.35)).astype(np.uint8)
    bank=compile_measured_programs_v60(x,m,group)
    z,s=infer_program_matrix_v60(bank,x,m)
    return x,m,group,y,z,s,bank


def test_torch_native_risk_has_only_signed_bounded_concept_contributions():
    net=SparseCompiledConceptHeadV60(4)
    with torch.no_grad():
        net.gamma.copy_(torch.tensor([2.,-.8,1.,.2]))
        net.alpha.copy_(torch.tensor([1.,2.,1.5,.8]))
    x=np.array([[.8,-1.,.3,0.],[.5,.5,.5,.5]],dtype=float)
    support=np.array([[1.,.7,0.,0.],[0.,0.,0.,0.]])
    base=np.array([-.3,.4],dtype=float)
    pred=net(x,support,base)
    np.testing.assert_allclose(
        pred["logits"].detach().numpy(),
        base+pred["signed_contributions"].detach().numpy().sum(axis=1),
        atol=1e-12,rtol=0)
    assert pred["used"].tolist()==[
        [True,True,False,False],[False,False,False,False]]
    assert pred["logits"].detach().numpy()[1]==base[1]
    assert all(isinstance(name,str) for name in pred)


def test_trainable_pure_program_discovers_source_signal_without_raw_risk_bypass():
    x,m,group,y,z,s,bank=cohort()
    frozen=np.zeros(len(y))
    fit=fit_program_head_v60(z,s,frozen,y,group,variant="PURE")
    assert fit.n_candidates==len(bank.candidates)
    assert 1<=fit.n_active<=6
    assert fit.train_subjects if False else fit.n_train_subjects==18
    assert not fit.raw_to_risk_direct_path
    assert fit.source_programs_fixed_without_Y
    pred=infer_program_head_v60(fit,z,s,frozen)
    assert np.isfinite(pred["logits"]).all()
    assert np.allclose(pred["logits"],
                       pred["baseline"]+pred["signed_contributions"].sum(axis=1),
                       atol=1e-9)
    assert group_macro_nll_v60(y,pred["logits"],group)<.70
    mask_no=np.zeros_like(s)
    absent=infer_program_head_v60(fit,z,mask_no,frozen)
    np.testing.assert_array_equal(
        absent["logits"],np.full(len(y),fit.train_prior_log_odds))


def test_rejected_program_model_is_exact_original_native_K2():
    x,m,group,y,z,s,bank=cohort()
    base=np.full(len(y),-.25)
    fitted=fit_program_head_v60(z,s,base,y,group,variant="RESIDUAL")
    inferred=infer_program_head_v60(fitted,z,s,base)
    used=inferred["used"][:,list(fitted.active_indices)]
    out,eligible=apply_program_admission_v60(
        base,inferred["logits"],used,approved=False)
    np.testing.assert_array_equal(out,base)
    assert not eligible.any()


def test_whole_patient_cluster_gate_refuses_harmful_valid_sources():
    p=np.repeat(np.arange(6),20)
    labels=np.tile([0,1],60)
    baseline=np.zeros(len(labels))
    bad=baseline+np.where(labels==1,-.8,.8)
    used=np.ones((len(labels),2),bool)
    x,m,group,y,z,s,bank=cohort()
    fit=fit_program_head_v60(z,s,np.zeros(len(y)),y,group,variant="PURE")
    report=admit_program_head_v60(
        fit,(labels,baseline,bad,used,p),(labels,baseline,bad,used,p))
    assert not report["approved"]
    assert report["valid1"]["patient_groups"]==6
    assert report["valid2"]["patient_groups"]==6
    assert report["validation_bootstrap_unit"]=="patient"


def test_no_valid_patient_groups_cannot_be_fake_repeated_windows():
    x,m,group,y,z,s,bank=cohort()
    fit=fit_program_head_v60(z,s,np.zeros(len(y)),y,group,variant="PURE")
    labels=np.tile([0,1],60)
    baseline=np.zeros(120)
    proposed=np.ones(120)
    repeated_patient=np.repeat(0,120)
    with pytest.raises(ValueError,match="4 distinct"):
        admit_program_head_v60(
            fit,(labels,baseline,proposed,np.ones((120,1),bool),
                 repeated_patient),
            (labels,baseline,proposed,np.ones((120,1),bool),
             repeated_patient))
