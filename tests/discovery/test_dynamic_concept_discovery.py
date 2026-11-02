from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.data import make_synthetic_discovery_data
from conceptfan_discovery.evaluation import (
    DiscoveryEvidence, compare_independent_discoveries,
    concept_passports, evaluate_discovery, oracle_correspondence,
)
from conceptfan_discovery.model import (
    PredictiveDynamicsConceptFAN, discovery_loss, masked_future_huber, sparsemax,
)
from conceptfan_discovery.operators import CHANNELS, causal_operator_bank


def _inputs(batch=7, t=12):
    rng=np.random.default_rng(17)
    x=torch.from_numpy(rng.normal(size=(batch,t,8)).astype(np.float32))
    mask=torch.from_numpy((rng.random(size=(batch,t,8))>.25).astype(np.float32))
    return x,mask


def test_sparsemax_is_exactly_sparse_and_simplex():
    logits=torch.tensor([[9.,0.,0.,-1.],[.1,.1,.1,.1]],requires_grad=True)
    w=sparsemax(logits)
    torch.testing.assert_close(w.sum(-1),torch.ones(2))
    assert (w>=0).all()
    torch.testing.assert_close(w[0],torch.tensor([1.,0.,0.,0.]))
    w.square().sum().backward()
    assert logits.grad is not None


def test_operator_bank_causal_missingness_and_schema():
    x,mask=_inputs()
    bank=causal_operator_bank(x,mask)
    assert bank.values.shape==(7,12,64)
    assert bank.availability.shape==bank.values.shape
    assert len(set(bank.names))==64
    assert bank.names[0].startswith("temperature|level|positive")
    x_future=x.clone()
    x_future[:,8:,:]+=100.
    prior=causal_operator_bank(x_future,mask)
    torch.testing.assert_close(bank.values[:,:8],prior.values[:,:8])
    masked_change=x.clone()
    masked_change[mask==0]=999.
    torch.testing.assert_close(bank.values,causal_operator_bank(masked_change,mask).values)


def test_model_exact_additivity_and_future_through_concept_gradients():
    torch.manual_seed(10)
    m=PredictiveDynamicsConceptFAN(n_slots=7,forecast_horizon=3)
    m.eval()
    x,mask=_inputs()
    out=m(x,mask)
    assert out.concept_trajectories.shape==(7,12,7)
    assert out.future_prediction.shape==(7,3,8)
    assert out.contributions.shape==(7,7)
    assert float(out.decomposition_error())<1e-6
    assert len(list(m.named_parameters()))>4
    # Both losses depend on the same identifiable concept projector.
    loss=out.logits.square().mean()+out.future_prediction.square().mean()
    loss.backward()
    assert m.feature_logits.grad is not None
    assert torch.isfinite(m.feature_logits.grad).all()


def test_gates_can_disable_slots_without_hidden_prediction_bypass():
    m=PredictiveDynamicsConceptFAN(n_slots=5)
    with torch.no_grad():
        m.gates.logits.fill_(-10.)
        m.gates.logits[2]=10.
    x,mask=_inputs()
    m.eval()
    out=m(x,mask,hard_gates=True)
    torch.testing.assert_close(out.gates,torch.tensor([0.,0.,1.,0.,0.]))
    assert torch.all(out.contributions[:,[0,1,3,4]]==0)
    torch.testing.assert_close(out.logits,out.decision_bias+out.contributions[:,2],atol=1e-6,rtol=1e-6)
    # No classifier logit bypass hidden inside the future decoder.
    with torch.no_grad():
        m.gates.logits.fill_(-10.)
    off=m(x,mask,hard_gates=True)
    torch.testing.assert_close(off.logits,m.decision_bias.expand_as(off.logits))


def test_exact_model_concept_intervention_is_not_fake_causal():
    m=PredictiveDynamicsConceptFAN(n_slots=6)
    m.eval()
    x,mask=_inputs()
    out=m(x,mask)
    override=torch.full_like(out.concept_summary[:,2],.35)
    new_logit,new_contrib=m.intervene(out,2,override)
    rest=[0,1,3,4,5]
    torch.testing.assert_close(new_contrib[:,rest],out.contributions[:,rest])
    torch.testing.assert_close(
        new_logit-out.logits,new_contrib[:,2]-out.contributions[:,2],
        atol=1e-6,rtol=1e-6,
    )
    with pytest.raises(ValueError):
        m.intervene(out,100,override)


def test_loss_is_masked_and_finite_for_all_ablation_modes():
    m=PredictiveDynamicsConceptFAN(n_slots=6,forecast_horizon=3)
    x,mask=_inputs()
    out=m(x,mask)
    y=torch.tensor([0.,1.,0.,1.,0.,1.,0.])
    f=torch.randn(7,3,8)
    fm=torch.ones_like(f)
    weights={"future":.6,"dynamics":.2,"mdl":.007,
             "support":.008,"redundancy":.08,
             "nonconstant":.25,"transition_l1":.003}
    task,_=discovery_loss(m,out,y,f,fm,mode="task_only",pos_weight=torch.tensor(1.),loss_weights=weights)
    multifuture,_=discovery_loss(m,out,y,f,fm,mode="task_future",pos_weight=torch.tensor(1.),loss_weights=weights)
    full,terms=discovery_loss(m,out,y,f,fm,mode="full",pos_weight=torch.tensor(1.),loss_weights=weights)
    assert torch.isfinite(full)
    assert float(multifuture)>float(task)
    assert float(full)>float(multifuture)
    assert "expected_active" in terms
    assert masked_future_huber(f,f,torch.ones_like(f)).item()==0.


def test_synthetic_oracle_never_enters_training_loader():
    data=make_synthetic_discovery_data(generator_seed=741,split_seed=211,n_episodes=120)
    loader=data.train.loader(batch_size=12,shuffle=False,seed=7)
    first=next(iter(loader))
    assert len(first)==5
    assert first[0].shape==(12,36,8)
    assert first[2].shape==(12,6,8)
    assert data.train.oracle_states.shape[-1]==5
    assert data.receipt["test_used_for_training_selection_or_evaluation"] is False
    assert data.receipt["test_n_sealed"]==24
    assert np.isfinite(data.train.x).all()
    assert np.unique(data.train.y).size==2


def test_evaluation_passports_and_oracle_are_posttraining_only():
    data=make_synthetic_discovery_data(generator_seed=743,split_seed=219,n_episodes=120)
    m=PredictiveDynamicsConceptFAN(n_slots=4)
    m.eval()
    e=evaluate_discovery(m,data.validation,batch_size=16)
    passports=concept_passports(e,m,data.validation,data.medians,data.scales)
    assert len(passports)==4
    assert all(p["causal_intervention_claim"] is False for p in passports)
    assert all(p["is_physically_grounded"] is False for p in passports)
    oracle=oracle_correspondence(e,data.validation.oracle_states)
    assert oracle["oracle_truth_used_for_training"] is False
    assert e.metrics["max_additive_error"]<1e-5


def test_stability_does_not_credit_constant_explanation_as_patient_specific():
    n,t,k=80,8,3
    # Fixed per-patient ranking; nonconstant activation due to small jitter.
    rng=np.random.default_rng(13)
    trajectories=np.clip(
        np.asarray([.9,.55,.17])[None,None,:]+rng.normal(scale=.04,size=(n,t,k)),0,1)
    weights=np.eye(k,64)
    values=trajectories.mean(axis=1)
    def evidence():
        return DiscoveryEvidence(
            metrics={"n_active_and_nonconstant_slots":3},
            probabilities=np.full(n,.5),
            contributions=values,
            trajectories=trajectories,
            raw_activation=trajectories,
            reliability=np.ones_like(trajectories),
            weights=weights,
            gates=np.ones(k),
            boundary=np.zeros(k),
            width=np.ones(k),
            decision_weights=np.ones(k),
        )
    result=compare_independent_discoveries(evidence(),evidence())
    assert result["paired_spearman"]>0.999
    assert abs(result["excess_over_unmatched_episode_null_spearman"])<1e-10
    assert result["definition_cosine_normalized"]>.999


def test_native_slot_selection_preserves_exact_logit_and_zeroes_pruned_concept():
    m=PredictiveDynamicsConceptFAN(n_slots=5,forecast_horizon=3)
    m.eval()
    x,mask=_inputs()
    chosen=torch.tensor([1.,0.,1.,0.,1.])
    with torch.no_grad():
        out=m(x,mask,selected_slots=chosen)
    assert out.decomposition_error()<1e-6
    torch.testing.assert_close(out.gates[[1,3]],torch.tensor([0.,0.]))
    torch.testing.assert_close(out.contributions[:,[1,3]],torch.zeros(7,2))
    assert torch.isfinite(out.future_prediction).all()


def test_slot_count_selection_uses_disjoint_validation_halves_only():
    from conceptfan_discovery.selection import (
        select_concept_cardinality, split_validation_for_cardinality,
    )
    data=make_synthetic_discovery_data(generator_seed=741,split_seed=211,n_episodes=120)
    selection,audit=split_validation_for_cardinality(data.validation,random_seed=17)
    assert not set(selection.episode_ids)&set(audit.episode_ids)
    m=PredictiveDynamicsConceptFAN(n_slots=5,forecast_horizon=6)
    result=select_concept_cardinality(
        m,selection,audit,mode="task_only",
        auprc_tolerance=1.0,nll_tolerance=100.,future_rmse_tolerance=100.,
    )
    assert result.selected_mask.shape==(5,)
    assert 1<=int(result.selected_mask.sum())<=5
    assert result.audit_metrics["selection_not_used_for_audit"]
    assert len(result.path)>=1
    with pytest.raises(ValueError):
        select_concept_cardinality(m,selection,selection,mode="full")
