"""Falsifying the independent MAGNETS-style BINARY adaptation.

This is NOT the original regression MAGNETS code/paper experiment.
Test data are synthesized in-memory; no PhysioNet cohort required.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset

from conceptfan_discovery.magnets_binary_baseline_v12 import (
    MaskAggregateBinaryConcepts,fit_magnets_binary_adaptation,
    predict_magnets_binary,train_only_concept_permutation,
)


class _ToyTrain:
    def __init__(self,seed=13,n=96):
        rng=np.random.default_rng(seed)
        self.x=rng.normal(size=(n,36,4)).astype(np.float32)
        self.mask=(rng.random(self.x.shape)>.2).astype(np.float32)
        self.x*=self.mask
        self.y=(self.x[:,-6:,0].mean(axis=1)>0).astype(np.float32)
        assert np.unique(self.y).size==2

    def loader(self,batch_size,shuffle,seed):
        n=len(self.x)
        return DataLoader(
            TensorDataset(
                torch.from_numpy(self.x),torch.from_numpy(self.mask),
                torch.zeros(n,3,4),torch.zeros(n,3,4),
                torch.from_numpy(self.y)
            ),batch_size=batch_size,shuffle=shuffle,
            generator=torch.Generator().manual_seed(int(seed)),num_workers=0,
        )


def test_exact_additive_logit_and_no_black_box_bypass():
    torch.manual_seed(1)
    model=MaskAggregateBinaryConcepts(4,3,history_hours=36,n_masks=4)
    x=torch.randn(8,36,4)
    m=(torch.rand_like(x)>.25).float()
    out=model(x,m,m)
    assert out.logits.shape==(8,)
    assert out.contributions.shape==(8,3)
    assert out.masks.shape==(8,4,4,36)
    assert float(out.decomposition_error(model.decision_bias))<1e-6
    assert torch.isfinite(out.logits).all()
    loss=out.logits.square().mean()+sum(model.training_regularizers(out).values())
    loss.backward()
    assert model.masker[0].weight.grad is not None
    assert torch.isfinite(model.masker[0].weight.grad).all()
    assert model.masker[0].weight.grad.abs().sum()>0
    assert model.bottleneck.weight.grad is not None


def test_no_measurements_means_neutral_concept_sum_even_if_placeholders_large():
    model=MaskAggregateBinaryConcepts(4,3,history_hours=36)
    model.eval()
    x=torch.randn(3,36,4)*1e6
    mask=torch.zeros_like(x)
    out=model(x,mask,mask)
    assert torch.allclose(out.concepts,torch.zeros_like(out.concepts))
    assert torch.allclose(out.contributions,torch.zeros_like(out.contributions))
    assert torch.allclose(out.logits,model.decision_bias.expand(3))
    with pytest.raises(ValueError,match="subset"):
        model(torch.zeros_like(x),torch.ones_like(x),mask)


def test_train_and_predict_in_memory_without_future_data_use():
    train=_ToyTrain()
    model,receipt=fit_magnets_binary_adaptation(
        train,seed=11,n_concepts=3,epochs=2,batch_size=24,
    )
    p,c=predict_magnets_binary(model,train.x,train.mask,train.mask)
    assert p.shape==(96,) and c.shape==(96,3)
    assert np.isfinite(p).all() and np.isfinite(c).all()
    assert receipt["training_set"]=="Set A TRAIN ONLY"
    assert receipt["reference_paper_task_regression_not_binary"]
    assert receipt["zero_outcome_holdout_access"]
    assert receipt["n_train_steps"]==8


def test_permutation_must_be_selected_on_training_and_never_flip_risk_sign():
    rng=np.random.default_rng(4)
    a=rng.normal(size=(80,3))
    b=a[:,[2,0,1]].copy()
    b[:,0]*=-1
    mapping=train_only_concept_permutation(a,b)
    assert mapping.tolist()==[1,2,0]
    # Signed effects may point opposite after a legitimate abs matching.
    assert np.corrcoef(a[:,2],b[:,mapping[2]])[0,1]<-.99
    unrelated=rng.normal(size=(80,3))
    none=train_only_concept_permutation(a,unrelated,min_abs_spearman=.99)
    assert (none==-1).all()
    with pytest.raises(ValueError):
        train_only_concept_permutation(a,a[:50])
