"""V20c: fit-disjoint TRAIN gate for safe native risk-head refit selection.

Four independently generated synthetic families. Core TRAIN 800 patients
trains both the original V10 concept model and its native decision refit.
Independent gate 300 patients is only used for one frozen acceptance rule.
The 300-patient TEST is evaluated only after every selection is finalized.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,roc_auc_score,brier_score_loss,log_loss,
)

from .data import DiscoveryData,DiscoverySplit
from .distinct_synthetic_v20b import (
    _raw_episodes,FAMILIES,CHANNELS,OBSERVATION_WINDOW,HORIZON,
)
from .experiment_native_risk_v18 import _config
from .clinical_multiscale_v10 import fit_multiscale
from .native_head_factorial_v20 import fit_native_head_factorial_v20
from .native_head_refit_gate_v20c import decide_head_refit_from_fit_disjoint_train_gate

SEEDS={"train":305011,"gate":305014,"test":305013}
N={"train":800,"gate":300,"test":300}
MODEL_SEEDS=(11,23,37)
ARMS=("original","refit_candidate","selected_by_train_gate")


def _make_data(family):
    partitions={
        name:_raw_episodes(family,SEEDS[name],N[name])
        for name in ("train","gate","test")
    }
    observed_train,mask_train,_=partitions["train"]
    medians=[]
    scales=[]
    for channel in range(5):
        available=observed_train[:,:,channel][mask_train[:,:,channel]>0]
        medians.append(float(np.median(available)))
        scales.append(float(max(np.quantile(available,.75)-np.quantile(available,.25),.05)))
    medians=np.asarray(medians,dtype=np.float64)
    scales=np.asarray(scales,dtype=np.float64)
    def build(name,offset):
        raw,mask,y=partitions[name]
        observed=mask.astype(np.float32)
        values=(np.clip((raw-medians)/scales,-6,6).astype(np.float32)*observed)
        return DiscoverySplit(
            x=values[:,:OBSERVATION_WINDOW],
            mask=observed[:,:OBSERVATION_WINDOW],
            future=values[:,OBSERVATION_WINDOW:],
            future_mask=observed[:,OBSERVATION_WINDOW:],
            y=y,oracle_states=None,
            episode_ids=np.arange(offset,offset+N[name],dtype=np.int64),
        )
    train=build("train",0)
    gate=build("gate",10000)
    test=build("test",20000)
    if (set(train.episode_ids)&set(gate.episode_ids)
        or set(train.episode_ids)&set(test.episode_ids)):
        raise AssertionError("Original model and acceptance gate must use disjoint episodes")
    data=DiscoveryData(
        train=train,
        validation=gate,
        receipt={
            "channel_names":list(CHANNELS),
            "generator_family":family,
            "gate_is_fit_disjoint_train_holdout":True,
            "all_normalization_from_core_train_only":True,
            "latent_generator_states_not_returned":True,
            "test_generator_seed":SEEDS["test"],
        },
        medians=medians,scales=scales,
    )
    return data,test


def _probabilities(model,split):
    model.eval()
    values=[]
    max_error=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            end=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:end])
            mask=torch.from_numpy(split.mask[start:end])
            output=model(x,mask,process_mask=mask)
            max_error=max(max_error,float(output.decomposition_error()))
            values.extend(output.probabilities.cpu().numpy().tolist())
    if max_error>2e-5:
        raise AssertionError("Native exact-additive contract violated")
    return np.asarray(values,dtype=np.float64),max_error


def _metrics(labels,p):
    return {
        "auprc":float(average_precision_score(labels,p)),
        "auroc":float(roc_auc_score(labels,p)),
        "brier":float(brier_score_loss(labels,p)),
        "logloss":float(log_loss(labels,p,labels=[0,1])),
    }


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("V20c must not overwrite committed scientific evidence")
    torch.set_num_threads(1)
    screening=[]
    test_rows=[]
    for family in FAMILIES:
        data,test=_make_data(family)
        config,dual,v10=_config()
        config["training"]["epochs"]=12
        config["training"]["batch_size"]=64
        basis=np.eye(len(CHANNELS),3)
        prepared=[]
        for seed in MODEL_SEEDS:
            original,_=fit_multiscale(
                data,basis,seed,config,dual,v10,
                consistency=False,feature_exploration=True,
            )
            original.eval()
            candidate=copy.deepcopy(original)
            receipt=fit_native_head_factorial_v20(
                candidate,data.train,
                declared_train_episode_ids=data.train.episode_ids,
                method="lbfgs",pos_weighted=False,
                effective_beta_l2=.001,lbfgs_max_iter=120,
            )
            # The refit is not permitted to reconfigure concept extraction.
            for name,value in original.state_dict().items():
                if name not in ("physical.decision_raw_weight","physical.decision_bias"):
                    if not torch.equal(candidate.state_dict()[name],value):
                        raise AssertionError("Refit candidate mutates concept/forecast graph")
            original_gate,error_old=_probabilities(original,data.validation)
            candidate_gate,error_new=_probabilities(candidate,data.validation)
            gate=decide_head_refit_from_fit_disjoint_train_gate(
                data.validation.y,original_gate,candidate_gate,
                seed=7131+seed,permutations=499,bootstraps=499,
                minimum_absolute_logloss_gain=.005,
            )
            accepted=gate.status=="ACCEPT_TRAIN_ONLY_NATIVE_REFIT"
            screening.append({
                "family":family,"seed":seed,
                "status":gate.status,
                "n_core_train":len(data.train.y),
                "n_fit_disjoint_train_gate":gate.n_gate_patients,
                "n_test":len(test.y),
                "train_positive_rate":float(data.train.y.mean()),
                "gate_positive_rate":float(data.validation.y.mean()),
                "candidate_gate_predictive_r":gate.candidate_risk_point_biserial_r,
                "candidate_gate_permutation_p":gate.candidate_risk_one_sided_permutation_p,
                "mean_gate_logloss_gain":gate.mean_gate_logloss_gain,
                "gate_gain_bootstrap_ci90_lower":gate.gain_bootstrap_ci90[0],
                "gate_gain_bootstrap_ci90_upper":gate.gain_bootstrap_ci90[1],
                "minimum_gain_predeclared":gate.fixed_min_gain,
                "gate_original_auprc":_metrics(data.validation.y,original_gate)["auprc"],
                "gate_candidate_auprc":_metrics(data.validation.y,candidate_gate)["auprc"],
                "native_head_refit_train_loss_decreased":receipt.final_objective<=receipt.initial_objective+1e-6,
                "test_labels_used_to_select_model":False,
                "selected_refit":accepted,
            })
            prepared.append((seed,original,candidate,accepted))
            print(family,seed,"gate",gate.status,
                  "gain",round(gate.mean_gate_logloss_gain,6),
                  "p",round(gate.candidate_risk_one_sided_permutation_p,4),flush=True)

        # TEST access occurs only after every seed's gate decision is frozen.
        for seed,original,candidate,accepted in prepared:
            choices={
                "original":original,
                "refit_candidate":candidate,
                "selected_by_train_gate":candidate if accepted else original,
            }
            for arm,model in choices.items():
                test_p,error=_probabilities(model,test)
                values=_metrics(test.y,test_p)
                test_rows.append({
                    "family":family,"seed":seed,"arm":arm,
                    "test_positive_rate":float(test.y.mean()),
                    "train_gate_accepted":accepted,
                    "model_selection_used_test":False,
                    "max_exact_additive_error":error,
                    "test_auprc":values["auprc"],
                    "test_auroc":values["auroc"],
                    "test_brier":values["brier"],
                    "test_logloss":values["logloss"],
                })
                print(family,seed,arm,"TEST AUPRC",round(values["auprc"],5),flush=True)
    aggregate=[]
    for family in FAMILIES:
        group=[r for r in screening if r["family"]==family]
        for arm in ARMS:
            rows=[r for r in test_rows if r["family"]==family and r["arm"]==arm]
            aggregate.append({
                "family":family,"arm":arm,"n_models":len(rows),
                "n_train_gate_approvals":sum(bool(r["selected_refit"]) for r in group),
                "mean_test_auprc":float(np.mean([r["test_auprc"] for r in rows])),
                "mean_test_auroc":float(np.mean([r["test_auroc"] for r in rows])),
                "mean_test_brier":float(np.mean([r["test_brier"] for r in rows])),
                "mean_test_logloss":float(np.mean([r["test_logloss"] for r in rows])),
                "physical_meaning_verified":False,
            })
    if len(screening)!=12 or len(test_rows)!=36 or len(aggregate)!=12:
        raise AssertionError("Incomplete predeclared V20c independent train-gate study")
    outdir.mkdir(parents=True,exist_ok=False)
    for filename,records in (
        ("ALL_12_FIT_DISJOINT_GATE_DECISIONS.csv",screening),
        ("ALL_36_TEST_POLICY_OUTCOMES.csv",test_rows),
        ("SUMMARY.csv",aggregate),
    ):
        with (outdir/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    manifest={
        "status":"V20C_INDEPENDENT_TRAIN_GATE_SELECTS_OR_REFUSES_NATIVE_RISK_REFIT",
        "number_original_models":12,
        "number_train_only_head_refit_candidates":12,
        "number_independent_train_holdout_decisions":len(screening),
        "number_predeclared_test_arm_evaluations":len(test_rows),
        "all_test_labels_scored_only_after_gate_decisions":True,
        "train_core_episodes":800,"gate_episodes":300,"test_episodes":300,
        "train_and_gate_independently_generated":True,
        "normalization_fit_on_train_core_only":True,
        "source_axes_known_measurement_directions_not_auto_concepts":True,
        "no_raw_input_prediction_shortcut":True,
        "no_physical_causal_or_clinical_claim":True,
        "gate_claims_predictive_improvement_only":True,
        "all_predeclared_cases_reported":True,
        "random_seeds":SEEDS,
        "gate_decisions":screening,
        "test_cases":test_rows,
        "aggregate":aggregate,
    }
    (outdir/"STATUS.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
