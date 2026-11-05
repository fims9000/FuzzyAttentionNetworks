"""Runs only with the actual canonical V10 branch installed (never fake it)."""
import numpy as np
import pytest
import torch

from conceptfan_discovery.selective_explanation_audit_v24 import NativeV10SelectiveExplanationAudit


def test_real_v10_dual_process_is_not_changed_by_support_audit():
    try:
        from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
    except ImportError:
        pytest.skip("Requires all canonical V10 model dependencies")
    torch.manual_seed(513)
    channels=("temperature","heart_rate","mean_arterial_pressure","lactate")
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(4,2),channels=channels,
        forecast_horizon=2,n_observation_concepts=2,
    ).eval()
    audit=NativeV10SelectiveExplanationAudit(model,min_recent_reliability=.25)
    x=torch.randn(5,12,4)
    mask=torch.ones_like(x)
    with torch.no_grad():
        before=model(x,mask,process_mask=mask)
        out=audit(x,mask,process_mask=mask)
    torch.testing.assert_close(out.logits,before.logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out.native.physical.contributions,before.physical.contributions)
    torch.testing.assert_close(out.native.observation_contributions,before.observation_contributions)
    assert float(out.native.decomposition_error()) < 2e-5
    assert not bool(out.abstain_physiological_explanation.any())

    missing=torch.zeros_like(mask)
    all_nan=torch.full_like(x,float('nan'))
    with torch.no_grad():
        absent=audit(all_nan,missing,process_mask=mask)
    assert bool(absent.abstain_physiological_explanation.all())
    assert float(absent.native.decomposition_error()) < 2e-5
    torch.testing.assert_close(absent.native.physical.contributions,
                               torch.zeros_like(absent.native.physical.contributions),atol=1e-6,rtol=1e-6)
    # Observation-process branch stays the same when physical measurements
    # were temporarily hidden; no corruption of the original recording policy.
    torch.testing.assert_close(absent.native.observation_contributions,
                               before.observation_contributions,atol=1e-6,rtol=1e-6)
