from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conceptfan_discovery.selective_explanation_audit_v24 import (
    NativeV10SelectiveExplanationAudit, _supported_temporal_window,
)


@dataclass
class FakeNativeOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    physical: SimpleNamespace
    observation_contributions: torch.Tensor

    def decomposition_error(self):
        return (self.logits - self.physical.decision_bias
                - self.physical.contributions.sum(-1)
                - self.observation_contributions.sum(-1)).abs().max()


class FakePhysical(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("risk_weight", torch.tensor([1.2, 0.0]))

    def decision_weights(self):
        return self.risk_weight


class FakeDualCore(nn.Module):
    def __init__(self, mode="legacy", active=True):
        super().__init__()
        self.physical = FakePhysical()
        self.process = nn.Identity()
        self.temporal_readout_mode = mode
        if not active:
            self.physical.risk_weight.zero_()

    def forward(self, values, observed_mask, *, process_mask=None):
        support = observed_mask[..., :2]
        # Sensor 0 determines physiological risk; sensor 1 is descriptive.
        concept = (values[..., :2] * support).sum(dim=1) / support.sum(dim=1).clamp_min(1)
        phys = concept * self.physical.decision_weights()
        recorded = observed_mask if process_mask is None else process_mask
        policy = recorded.mean(dim=(1, 2)).unsqueeze(-1) * .7
        logits = -.3 + phys.sum(-1) + policy.sum(-1)
        return FakeNativeOutput(
            logits=logits, probabilities=logits.sigmoid(),
            physical=SimpleNamespace(availability=support, gates=torch.ones(2),
                  contributions=phys, decision_bias=logits.new_tensor(-.3)),
            observation_contributions=policy,
        )


def test_all_observed_no_abstention_and_native_output_preserved():
    core=FakeDualCore()
    audit=NativeV10SelectiveExplanationAudit(core,min_recent_reliability=.3)
    values=torch.randn(6,12,2)
    masks=torch.ones_like(values)
    expected=core(values,masks)
    result=audit(values,masks)
    assert not bool(result.abstain_physiological_explanation.any())
    assert result.native_risk_active.tolist()==[True,False]
    torch.testing.assert_close(expected.logits,result.logits)
    torch.testing.assert_close(expected.observation_contributions,result.native.observation_contributions)


def test_missing_risk_sensor_abstains_descriptive_only_sensor_does_not():
    values=torch.randn(4,12,2)
    masks=torch.ones_like(values)
    audit=NativeV10SelectiveExplanationAudit(FakeDualCore(),min_recent_reliability=.5)
    masks[:,:,1]=0
    assert not bool(audit(values,masks).abstain_physiological_explanation.any())
    masks[:,:,0]=0
    result=audit(values,masks)
    assert bool(result.abstain_physiological_explanation.all())
    assert bool(result.unsupported_risk_concepts[:,0].all())
    assert not bool(result.unsupported_risk_concepts[:,1].any())


def test_null_measurements_remain_risk_neutral_and_mask_policy_distinct():
    model=FakeDualCore()
    audit=NativeV10SelectiveExplanationAudit(model)
    mask=torch.zeros(3,10,2)
    values=torch.full_like(mask,float('nan'))
    out=audit(values,mask)
    assert bool(out.abstain_physiological_explanation.all())
    torch.testing.assert_close(out.native.physical.contributions,torch.zeros(3,2))
    torch.testing.assert_close(out.logits,model(torch.zeros_like(values),mask).logits)
    # Process schedule is ORIGINAL, not the artificially thinned view.
    original=torch.ones_like(mask)
    out_with_process=audit(values,mask,process_mask=original)
    torch.testing.assert_close(out_with_process.native.observation_contributions,
                               model(torch.zeros_like(values),mask,process_mask=original).observation_contributions)
    assert not torch.allclose(out.logits,out_with_process.logits)


def test_partial_missing_payload_changes_are_ignored():
    values=torch.randn(3,12,2)
    mask=torch.ones_like(values)
    mask[:,:5,0]=0
    audit=NativeV10SelectiveExplanationAudit(FakeDualCore())
    first=audit(values,mask)
    values[:,:5,0]=float('nan')
    second=audit(values,mask)
    torch.testing.assert_close(first.logits,second.logits)


def test_all_risk_weights_exact_zero_refuses_physio_explanation():
    mask=torch.ones(2,12,2)
    out=NativeV10SelectiveExplanationAudit(FakeDualCore(active=False))(torch.randn_like(mask),mask)
    assert bool(out.abstain_physiological_explanation.all())
    assert not bool(out.native_risk_active.any())


def test_v19_early_and_recent_support_required():
    support=torch.ones(1,12,2)
    support[:,:6,0]=0
    assert _supported_temporal_window(support,"legacy")[0,0].item()==pytest.approx(1.0)
    assert _supported_temporal_window(support,"learned_shape")[0,0].item()==pytest.approx(0.0)


@pytest.mark.parametrize("kind",["nan","nonbinary","original_mismatch"])
def test_invalid_masks_refused(kind):
    audit=NativeV10SelectiveExplanationAudit(FakeDualCore())
    x=torch.zeros(2,12,2)
    m=torch.ones_like(x)
    if kind=="nan":m[0,0,0]=float('nan')
    if kind=="nonbinary":m[0,0,0]=.4
    if kind=="original_mismatch":
        original=torch.zeros_like(m)
        with pytest.raises(ValueError):audit(x,m,process_mask=original)
        return
    with pytest.raises(ValueError):audit(x,m)
