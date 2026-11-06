"""V38 source save/reopen must be faithful, safe and fail closed."""
from __future__ import annotations
import json
import numpy as np
import pytest

from conceptfan_discovery.exogenous_probe_world_v38 import (
    generate_exogenous_world_v38,mask_world_v38,apply_artifact_v38,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,
)
from conceptfan_discovery.exogenous_probe_gate_v38 import (
    fit_sentinel_train_only_v38,fit_probe_intervention_v38,
    source_with_independent_evidence_v38,
)
from conceptfan_discovery.source_persistence_v38 import (
    save_source_state_v38,load_source_state_v38,
)


def _fit():
    d=generate_exogenous_world_v38(
        device_seed=20261231,n_train=460,n_valid=300,
        n_test=240,per_actuator=180)
    x=d["x"]
    tr=d["train"]
    m=mask_world_v38(x[tr],"MCAR20",seed=71,known_seed=72)
    fitted=fit_centered_threeview_v37(
        x[tr],m,independent_view_noise_documented=True,
        seed=20263803)
    sentinel=fit_sentinel_train_only_v38(x[tr],m)
    probe=fit_probe_intervention_v38(
        fitted,d["independent_calibration_onoff"],seed=20263810)
    return d,fitted,sentinel,probe


def test_json_reopen_exact_replay_for_all_masks_and_monitored_fault(tmp_path):
    d,fit,monitor,probe=_fit()
    output=tmp_path/"source-v38.json"
    save_source_state_v38(output,fit,monitor,probe)
    r=load_source_state_v38(output)
    assert r.probe.passed==probe.passed
    assert r.monitor.threshold_train_only==monitor.threshold_train_only
    assert r.source.train_max_pairwise_q_discrepancy_q95==(
        fit.train_max_pairwise_q_discrepancy_q95)
    x=d["x"][d["test"]]
    for mode in ("complete","MCAR20","MCAR40",
                 "missing_sentinel","missing_C"):
        mask=mask_world_v38(x,mode,seed=111,known_seed=17)
        for xx in (
            x,apply_artifact_v38(x,"shared_ABC_monitored",seed=112),
            apply_artifact_v38(x,"shared_ABC_unmonitored",seed=112),
        ):
            old=source_with_independent_evidence_v38(
                fit,monitor,probe,xx,mask)
            fresh=source_with_independent_evidence_v38(
                r.source,r.monitor,r.probe,xx,mask)
            np.testing.assert_array_equal(old["q"],fresh["q"])
            np.testing.assert_array_equal(old["support"],fresh["support"])
    assert "hidden_H" not in output.read_text()
    assert "true_latent" not in output.read_text()


def test_reopen_single_episode_still_readonly_and_hidden_nan_safe(tmp_path):
    d,fit,monitor,probe=_fit()
    path=tmp_path/"source.json"
    save_source_state_v38(path,fit,monitor,probe)
    loaded=load_source_state_v38(path)
    x=d["x"][d["test"]][:1]
    mask=mask_world_v38(x,"MCAR20",seed=22,known_seed=17)
    hidden=x.copy()
    hidden[~mask]=np.nan
    before=source_with_independent_evidence_v38(
        fit,monitor,probe,x,mask)
    after=source_with_independent_evidence_v38(
        loaded.source,loaded.monitor,loaded.probe,hidden,mask)
    np.testing.assert_array_equal(before["q"],after["q"])
    np.testing.assert_array_equal(before["support"],after["support"])


def test_refuse_overwrite_and_falsely_certified_medical_source(tmp_path):
    _,fit,monitor,probe=_fit()
    path=tmp_path/"model.json"
    save_source_state_v38(path,fit,monitor,probe)
    with pytest.raises(FileExistsError):
        save_source_state_v38(path,fit,monitor,probe)
    payload=json.loads(path.read_text())
    payload["source_semantics_medically_certified"]=True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="falsely certified"):
        load_source_state_v38(path)


def test_refuse_corrupt_eigensystem_to_prevent_false_inference(tmp_path):
    _,fit,monitor,probe=_fit()
    path=tmp_path/"source.json"
    save_source_state_v38(path,fit,monitor,probe)
    payload=json.loads(path.read_text())
    payload["source_fits"]["pairs"]["AB"]["inv_left"]=[[0,0],[0,0]]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="inverse"):
        load_source_state_v38(path)
