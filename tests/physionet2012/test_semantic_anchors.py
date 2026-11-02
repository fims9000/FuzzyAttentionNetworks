from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_realdata.models import exact_decomposition_error
from conceptfan_realdata.semantic_anchors import (
    AnchoredConceptFAN,
    anchor_evidence,
    fit_semantic_anchor,
    fixed_temporal_weights,
    semantic_concept_regularizers,
)
from scripts.medical.run_semantic_anchor_validation import train_one, agreement


def config(steps=8):
    return {"model": {"n_concepts": 5, "latent_dim": 16, "sequence_length": steps,
                      "dropout": .0, "layers": 1, "heads": 2, "ffn": 32},
            "training":{"epochs": 1, "batch_size": 12,
                        "learning_rate": .001, "weight_decay": 0.,
                        "lambda_concept": 2., "lambda_temporal": .5,
                        "lambda_oracle": 1., "lambda_cross_residual": 25.,
                        "gradient_clip": 1., "data_order_seed": 1001}}


def synthetic(steps=8):
    rng=np.random.default_rng(7)
    x=rng.normal(size=(65,steps,8)).astype(np.float32)
    c=np.clip(.4 + .1*x[:,:,:5] + rng.normal(scale=.08,size=(65,steps,5)),.02,.98).astype(np.float32)
    y=np.array([0,1]*32+[0],dtype=np.float32)
    return {"train":(x[:48],y[:48],c[:48]),
            "validation":(x[48:],y[48:],c[48:])}


def test_temporal_weights_prescribed_not_learned():
    w=fixed_temporal_weights(8,.2)
    assert np.isclose(w.sum(),1.)
    assert np.all(np.diff(w)>0)
    with pytest.raises(ValueError):
        fixed_temporal_weights(1,.5)


def test_frozen_train_anchor_only_uses_training_concepts_and_labels():
    d=synthetic()
    c,y=d["train"][2],d["train"][1]
    a=fit_semantic_anchor(c,y)
    b=fit_semantic_anchor(c.copy(),y.copy())
    assert np.allclose(a.beta,b.beta)
    assert a.bias==b.bias
    assert np.all(a.width>=.05)
    assert np.allclose(anchor_evidence(a.center,a.center,a.width),0.)
    with pytest.raises(ValueError):
        fit_semantic_anchor(np.full_like(c,2.),y)


def test_exact_additive_model_and_frozen_shared_head():
    d=synthetic()
    c,y=d["train"][2],d["train"][1]
    anchor=fit_semantic_anchor(c,y)
    torch.manual_seed(1)
    m=AnchoredConceptFAN(config(),8,anchor)
    outputs=m(torch.from_numpy(d["validation"][0]))
    assert outputs.contributions.shape==(17,5)
    assert outputs.concept_trajectories.shape==(17,8,5)
    assert float(exact_decomposition_error(outputs).max())<1e-6
    assert not m.anchor_beta.requires_grad
    assert not m.temporal_aggregator.value_weight.requires_grad
    loss=outputs.logit.sum()
    loss.backward()
    assert m.projector.net[0].weight.grad is not None
    assert torch.isfinite(m.projector.net[0].weight.grad).all()


def test_oracle_concept_loss_zero_when_predictions_exact():
    d=synthetic()
    c,y=d["train"][2],d["train"][1]
    anchor=fit_semantic_anchor(c,y)
    model=AnchoredConceptFAN(config(),8,anchor)
    true=torch.from_numpy(c[:6])
    latent=torch.zeros((6,8,16))
    summary,evidence=model.anchored_evidence(true)
    from conceptfan_realdata.models import RealModelOutput
    contributions=evidence*model.anchor_beta
    out=RealModelOutput(logit=model.anchor_bias+contributions.sum(-1),
                        probability=torch.zeros(6),
                        concept_trajectories=true,
                        concept_summaries=summary,
                        contributions=contributions,bias=model.anchor_bias,
                        latent_sequence=latent,evidence=evidence)
    reg=semantic_concept_regularizers(out,true,torch.ones_like(true),model)
    for metric in reg.values():
        assert abs(float(metric))<1e-10


def test_one_epoch_arms_finite_without_test_split():
    d=synthetic()
    anchor=fit_semantic_anchor(d["train"][2],d["train"][1])
    for method in ("legacy","anchor_only","anchor_regularized"):
        result=train_one(method,11,config(),anchor,d)
        assert 0 <= result["auprc"] <= 1
        assert 0 <= result["concept_mae"] <= 1
        assert result["max_decomposition_error"]<1e-5
        assert np.isfinite(result["contributions"]).all()


def test_pairwise_agreement_uses_shared_patient_order_and_all_concepts():
    a=np.asarray([[1.,-.5,.2,.1,0.],[.2,.5,-.8,.1,.1]])
    assert agreement(a,a)["top1_abs"]==1.
    assert agreement(a,a)["sign_agreement"]==1.
    with pytest.raises(ValueError):
        agreement(a,a[:1])


def test_concept_intervention_changes_only_its_own_term():
    d=synthetic()
    anchor=fit_semantic_anchor(d["train"][2],d["train"][1])
    model=AnchoredConceptFAN(config(),8,anchor)
    model.eval()
    with torch.no_grad():
        original=model(torch.from_numpy(d["validation"][0][:4]))
        baseline=original.contributions.clone()
        trajectory=original.concept_trajectories
        new_value=torch.full_like(trajectory[:,:,2],.9)
        updated_logit,updated_contrib=model.intervene_on_concept(trajectory,2,new_value)
    unaffected=[0,1,3,4]
    torch.testing.assert_close(updated_contrib[:,unaffected],baseline[:,unaffected])
    torch.testing.assert_close(updated_logit-original.logit,
                               updated_contrib[:,2]-baseline[:,2],atol=1e-5,rtol=1e-5)
    with pytest.raises(ValueError):
        model.intervene_on_concept(trajectory,9,new_value)
    with pytest.raises(ValueError):
        model.intervene_on_concept(trajectory,2,new_value[:1])


def test_global_constant_concept_rank_is_not_patient_specific_stability():
    values=np.repeat(np.array([[3.,2.,1.,.3,-.2]]),100,axis=0)
    result=agreement(values,values)
    assert result["spearman_signed"]==pytest.approx(1.)
    assert result["top1_abs"]==pytest.approx(1.)
    assert result["spearman_excess_episode_alignment"]==pytest.approx(0.,abs=1e-12)
    assert result["top1_excess_episode_alignment"]==pytest.approx(0.,abs=1e-12)


def test_patient_specific_matches_exceed_permutation_chance():
    rng=np.random.default_rng(23)
    values=rng.normal(size=(250,5))
    result=agreement(values,values)
    assert result["spearman_signed"]==pytest.approx(1.,abs=1e-12)
    assert result["spearman_excess_episode_alignment"]>.85
    assert result["top1_excess_episode_alignment"]>.7
    assert result["sign_excess_episode_alignment"]>.35
