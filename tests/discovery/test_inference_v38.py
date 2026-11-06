"""V38 public NPZ inference contract, not only internal reproducibility script."""
from __future__ import annotations
import json

import numpy as np
import pytest
import torch

from conceptfan_discovery.exogenous_probe_world_v38 import (
    generate_exogenous_world_v38,mask_world_v38,
)
from conceptfan_discovery.threeview_centered_v37 import fit_centered_threeview_v37
from conceptfan_discovery.exogenous_probe_gate_v38 import (
    fit_sentinel_train_only_v38,fit_probe_intervention_v38,
    source_with_independent_evidence_v38,
)
from conceptfan_discovery.source_persistence_v38 import save_source_state_v38
from conceptfan_discovery.inference_v38 import load_v38_inference_model
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_trained_conceptfan_omission_v25 import predict_native,sha_state
from scripts.infer_exogenous_source_v38 import inference_npz_to_json
from scipy.special import expit


@pytest.fixture(scope="module")
def synthetic_reference():
    world=generate_exogenous_world_v38(
        device_seed=20261231,n_train=460,n_valid=300,
        n_test=240,per_actuator=180)
    x=world["x"]
    tr=world["train"]
    m=mask_world_v38(x[tr],"MCAR20",seed=71,known_seed=72)
    fit=fit_centered_threeview_v37(
        x[tr],m,independent_view_noise_documented=True,
        seed=20263803,bootstrap=24)
    mon=fit_sentinel_train_only_v38(x[tr],m)
    probe=fit_probe_intervention_v38(
        fit,world["independent_calibration_onoff"],seed=20263810)
    native=make_firewalled_k2(11).eval()
    return world,fit,mon,probe,native


def model_dir(tmp_path,source):
    _,fit,mon,probe,native=source
    (tmp_path/"SOURCE_STATES").mkdir()
    (tmp_path/"CHECKPOINTS").mkdir()
    save_source_state_v38(
        tmp_path/"SOURCE_STATES/device20261231_V38_source.json",
        fit,mon,probe)
    model_sha=sha_state(native)
    torch.save({"state_dict":native.state_dict()},
               tmp_path/"CHECKPOINTS/native_K2_device20261231_seed11.pt")
    original={
        "device_seed":20261231,"NN_seed":11,"outcome_case":"healthy",
        "native_checkpoint":"CHECKPOINTS/native_K2_device20261231_seed11.pt",
        "source_model_state":"SOURCE_STATES/device20261231_V38_source.json",
        "two_concept_original_TRAIN_head":{"weights":[1.4,-1.1],"bias":-.08},
        "new_source_gamma_TRAIN_only":1.2,
        "new_source_passed_two_independent_VALID":True,
        "new_source_requires_independent_probe_pass":True,
        "frozen_old_K2_weights_SHA":model_sha,
        "source_physically_certified_realworld":False,
    }
    (tmp_path/"INFERENCE_BUNDLES.json").write_text(
        json.dumps([original]))
    return original


def test_model_inference_is_identical_single_vs_batch_without_hidden_payload(
    tmp_path,synthetic_reference,
):
    world,fit,mon,probe,native=synthetic_reference
    model_dir(tmp_path,synthetic_reference)
    loaded=load_v38_inference_model(
        tmp_path,device_seed=20261231,nn_seed=11,
        outcome_case="healthy")
    x=world["x"][world["test"]][:13]
    m=mask_world_v38(x,"MCAR20",seed=55,known_seed=31)
    full=loaded.predict(x,m)
    nan=x.copy()
    nan[~m]=np.nan
    hidden=loaded.predict(nan,m)
    one=loaded.predict(nan[:1],m[:1])
    assert len(full["episodes"])==13
    for i in range(len(x)):
        assert full["episodes"][i]["model_event_probability"]==(
            hidden["episodes"][i]["model_event_probability"])
    # BLAS reductions on a singleton and a batch can differ by one ULP;
    # this is numerical rounding, not a model/mask semantic difference.
    assert abs(full["episodes"][0]["model_event_probability"]-
               one["episodes"][0]["model_event_probability"])<1e-12
    assert all(r["uncertainty"]["source_physical_identity_unverified"]
               for r in full["episodes"])
    assert all(r["uncertainty"]["no_imputed_values_used"]
               for r in full["episodes"])
    assert full["synthetic_research_not_clinically_validated"]
    assert not full["physical_or_causal_semantics_certified"]
    # Independently reconstruct the same frozen torch + signed risk head.
    q=predict_native(native,x[:,:,:4],m[:,:,:4])["concept_summary"]
    source=source_with_independent_evidence_v38(fit,mon,probe,x,m)
    base=-.08+q@np.asarray([1.4,-1.1])
    expected=expit(base+1.2*(source["q"]-.5)*source["support"])
    np.testing.assert_allclose(
        [r["model_event_probability"] for r in full["episodes"]],
        expected,atol=1e-7,rtol=0)


def test_monitor_is_missing_uses_exact_original_neural_risk(
    tmp_path,synthetic_reference,
):
    model_dir(tmp_path,synthetic_reference)
    loaded=load_v38_inference_model(
        tmp_path,device_seed=20261231,nn_seed=11,
        outcome_case="healthy")
    x=synthetic_reference[0]["x"][synthetic_reference[0]["test"]][:5]
    m=mask_world_v38(x,"missing_sentinel",seed=55,known_seed=31)
    outcome=loaded.predict(x,m)
    assert all(r["decision_code"]=="MONITOR_UNOBSERVED"
               for r in outcome["episodes"])
    assert all(not r["used_extra_source"] for r in outcome["episodes"])
    assert all(r["model_event_probability"]==
               r["model_event_probability_without_extra_source"]
               for r in outcome["episodes"])


def test_cli_npz_data_only_and_fail_closed_model_manifest(tmp_path,synthetic_reference):
    model_dir(tmp_path,synthetic_reference)
    x=synthetic_reference[0]["x"][synthetic_reference[0]["test"]][:3]
    m=mask_world_v38(x,"MCAR20",seed=55,known_seed=31)
    path=tmp_path/"episodes.npz"
    np.savez_compressed(path,x=x,observed=m)
    out=tmp_path/"predictions.json"
    d=inference_npz_to_json(
        tmp_path,device_seed=20261231,nn_seed=11,
        outcome_case="healthy",input_npz=path,output_json=out)
    assert json.loads(out.read_text())==d
    assert d["n_episodes"]==3
    with pytest.raises(FileExistsError):
        inference_npz_to_json(
            tmp_path,device_seed=20261231,nn_seed=11,
            outcome_case="healthy",input_npz=path,output_json=out)
    with pytest.raises(ValueError,match="Boolean"):
        m_bad=m.astype(np.int8)
        loaded=load_v38_inference_model(
            tmp_path,device_seed=20261231,nn_seed=11,
            outcome_case="healthy")
        loaded.predict(x,m_bad)
    original=json.loads((tmp_path/"INFERENCE_BUNDLES.json").read_text())
    original[0]["native_checkpoint"]="../untrusted.pt"
    (tmp_path/"INFERENCE_BUNDLES.json").write_text(json.dumps(original))
    with pytest.raises(ValueError,match="unsafe"):
        load_v38_inference_model(
            tmp_path,device_seed=20261231,nn_seed=11,outcome_case="healthy")


def test_decline_invalid_observation_and_corrupted_native_checksum(
    tmp_path,synthetic_reference,
):
    model_dir(tmp_path,synthetic_reference)
    loaded=load_v38_inference_model(
        tmp_path,device_seed=20261231,nn_seed=11,
        outcome_case="healthy")
    x=synthetic_reference[0]["x"][synthetic_reference[0]["test"]][:1]
    m=mask_world_v38(x,"MCAR20",seed=55,known_seed=31)
    x_bad=x.copy()
    x_bad[m]=np.nan
    with pytest.raises(ValueError,match="finite observed"):
        loaded.predict(x_bad,m)
    with pytest.raises(ValueError):
        loaded.predict(x[:0],m[:0])
    manifest=json.loads((tmp_path/"INFERENCE_BUNDLES.json").read_text())
    manifest[0]["frozen_old_K2_weights_SHA"]="0"*64
    (tmp_path/"INFERENCE_BUNDLES.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="checksum"):
        load_v38_inference_model(
            tmp_path,device_seed=20261231,nn_seed=11,outcome_case="healthy")
