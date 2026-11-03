"""Pre-registered ONE-SHOT, previously unseen 4000-patient Set B audit.

The 2012 PhysioNet Set B outcomes, withheld during the original challenge,
are publicly archived in the current v1.0.0 release. Set B represents
additional MIMIC-II ICU stays, NOT a new hospital. This is an independent
patient cohort, with the same historical data source.

Critical order:
  1. load and fit ONLY frozen Set A TRAIN 2400 / original frozen V10 basis
  2. refit & verify each model against already executed Set A DEV receipts
  3. ONLY AFTER ALL DEV verification succeeds, read Set B records/outcomes
  4. evaluate ALL preregistered methods/scenarios ONCE, no retuning
  5. archive aggregate metrics, signed group audit, source SHA256 hashes;
     never publish individual records, record IDs, predictions, or checkpoints.

Any error after step 3 invalidates a claim of uninspected future holdout;
do NOT edit models/hyperparameters and silently rerun Set B in that event.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
import yaml

from conceptfan_realdata.data import (
    FORBIDDEN_INPUTS, KNOWN_PARAMETERS, STATIC_PARAMETERS,
    _aggregate_patient, _read_record, load_outcomes, sha256_file,
)
from .clinical_multiscale_v10 import fit_multiscale
from .clinical_identifiability_v11 import (
    ARMS, _definition_group_overlap,
)
from .clinical_v10_external_benchmark import (
    EXTERNAL_METHODS, _load_and_match_v10_patient_cohort,
    _model_score, _paired_bootstrap, _perturbation_views,
)
from .effect_certification_v11 import joint_concept_screen
from .external_baselines_v10 import (
    fit_neural_baseline, fit_tabular_baseline,
    predict_neural, predict_tabular,
)
from .identifiability_v11 import (
    compare_group_attributions, fit_predictive_group_certificate,
    projector_overlap,
)
from .multiscale_physiology_v10 import expanded_multiscale_definition
from .train_identifiability_v11 import fit_identifiability_aware_v11


SEEDS=(11,23,37)
NEURAL_EXTERNAL=("gru_d_style","transformer_timeseries")
TABULAR_EXTERNAL=("logistic_history","hist_gradient_boosting")
METHODS=ARMS+EXTERNAL_METHODS
SCENARIOS=("original","mcar_thin","time_channel_mar")
REFERENCE_V11_RUN=41689286080
REFERENCE_V10_EXTERNAL_RUN=41626717843


def download_set_b_only_after_all_refit_guards_pass(
    *, destination_zip:Path,destination_outcomes:Path,
    source_zip_url:str,outcome_url:str,
)->None:
    """Actual HOLDOUT source access is deferred until ALL A-only refits pass.

    A failed refit check therefore cannot accidentally access or inspect
    any new cohort labels, even in the GitHub runner filesystem.
    """
    prefix="https://physionet.org/files/challenge-2012/1.0.0/"
    if (source_zip_url!=prefix+"set-b.zip" or
        outcome_url!=prefix+"Outcomes-b.txt"):
        raise ValueError("Only exact pre-registered official Set B source URLs")
    if destination_zip.exists() or destination_outcomes.exists():
        raise FileExistsError("Set B files must be absent BEFORE frozen refit")
    for destination,url in (
        (destination_zip,source_zip_url),(destination_outcomes,outcome_url),
    ):
        destination.parent.mkdir(parents=True,exist_ok=True)
        try:
            request=urllib.request.Request(
                url,headers={"User-Agent":"ConceptFAN-frozen-one-shot-holdout/1.0"}
            )
            with urllib.request.urlopen(request,timeout=180) as source, destination.open("xb") as sink:
                while True:
                    part=source.read(1024*1024)
                    if not part:
                        break
                    sink.write(part)
            if destination.stat().st_size<1000:
                raise IOError("Official Set B source unexpectedly small")
        except Exception:
            destination.unlink(missing_ok=True)
            raise


def load_set_b_strict(
    source_zip:Path,outcomes_b:Path, *,
    selected_channels:tuple[str,...],
    medians:np.ndarray,scales:np.ndarray,
    set_a_record_ids:set[int],
    history_hours:int=36,expected_n:int=4000,
)->tuple[object,dict]:
    """Apply exactly Set-A-fitted preprocessing; never fit anything on Set B.

    The data adapter explicitly removes records >=36h BEFORE aggregation.
    This is stricter than merely omitting future values from model.forward:
    changing Set B's 36-48h records cannot affect evaluation features.
    """
    if not (source_zip.is_file() and outcomes_b.is_file()):
        raise FileNotFoundError("Both public Set B sources are mandatory")
    if history_hours!=36 or expected_n<2:
        raise ValueError("Only pinned 36h Set B protocol supported")
    names=tuple(selected_channels)
    if not names or len(set(names))!=len(names) or (
        not set(names).issubset(KNOWN_PARAMETERS-STATIC_PARAMETERS-FORBIDDEN_INPUTS)
    ):
        raise ValueError("Only frozen independently observed physiological channels allowed")
    mu=np.asarray(medians,dtype=np.float64)
    iq=np.asarray(scales,dtype=np.float64)
    if mu.shape!=(len(names),) or iq.shape!=mu.shape or (
        not np.isfinite(mu).all() or not np.isfinite(iq).all() or np.any(iq<=0)
    ):
        raise ValueError("Set B normalization MUST use positive Set A TRAIN medians/scales")
    if not set_a_record_ids:
        raise ValueError("Must explicitly verify Set A/Set B record ID disjointness")
    # The outcomes file is loaded solely for scoring. No secondary descriptors
    # (SAPS-I, SOFA, length of stay, survival) can enter the model input.
    labels=load_outcomes(outcomes_b)
    if len(labels)!=expected_n:
        raise ValueError("Outcome label count disagrees with preregistered 4000 Set B patients")
    labelmap=dict(zip(labels.RecordID.astype(int),labels["In-hospital_death"].astype(int)))
    x=np.zeros((expected_n,history_hours,len(names)),dtype=np.float32)
    observed=np.zeros_like(x)
    ids=[]
    with zipfile.ZipFile(source_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt Set B ZIP")
        records=sorted(fn for fn in archive.namelist() if fn.lower().endswith(".txt"))
        if len(records)!=expected_n:
            raise ValueError("Set B patient-record count differs from published protocol")
        for row,filename in enumerate(records):
            ident=int(Path(filename).stem)
            frame=_read_record(archive.read(filename))
            internal=frame.loc[frame["Parameter"].eq("RecordID"),"Value"].dropna().astype(int).unique()
            if len(internal)!=1 or int(internal[0])!=ident:
                raise ValueError("Set B filename and patient RecordID disagree")
            # No future measurement can reach a feature builder, even if
            # its value is weird, missing, or spuriously predictive.
            history=frame.loc[frame.hour<history_hours]
            raw,mask,_=_aggregate_patient(history,list(names),hours=history_hours)
            x[row]=np.where(mask>0,np.clip((raw-mu)/iq,-6,6),0).astype(np.float32)
            observed[row]=mask
            ids.append(ident)
    ids_arr=np.asarray(ids,dtype=np.int64)
    if (len(set(ids))!=expected_n or set(ids)!=set(labelmap) or
        set(ids).intersection(set_a_record_ids)):
        raise ValueError("Set B duplicate IDs, unmatched labels, or Set A/Set B patient overlap")
    y=np.asarray([labelmap[i] for i in ids],dtype=np.float32)
    if len(np.unique(y))!=2 or not np.isfinite(x).all() or (
        not np.logical_or(observed==0,observed==1).all()
    ):
        raise ValueError("Invalid features, observation mask, or outcomes in new cohort")
    # Do not return true RecordID anywhere in the scored patient object.
    from types import SimpleNamespace
    split=SimpleNamespace(x=x,mask=observed,y=y)
    receipt={
        "cohort":"2012 PhysioNet Challenge Set B, MIMIC-II, independent patient records",
        "number_of_patients":expected_n,
        "number_positive":int(y.sum()),
        "prevalence":float(y.mean()),
        "history_hours":history_hours,
        "n_selected_channels":len(names),
        "selected_channels":list(names),
        "no_future_labs_used":True,
        "original_set_a_train_only_preprocessing":True,
        "set_b_observation_or_outcome_used_for_training":False,
        "set_a_set_b_ids_disjoint":True,
        "set_b_records_sha256":sha256_file(source_zip),
        "set_b_outcomes_sha256":sha256_file(outcomes_b),
        "set_b_sorted_record_id_digest":sha256(
            np.ascontiguousarray(np.sort(ids_arr),dtype="<i8").tobytes()
        ).hexdigest(),
        "same_mimic_ii_data_source_not_independent_hospital":True,
        "no_patient_level_id_or_prediction_will_be_published":True,
    }
    return split,receipt


def predict_fixed_concept(
    model,values:np.ndarray,visible:np.ndarray,recorded:np.ndarray,
    *, batch_size:int=128,collect_contributions:bool=False,
)->tuple[np.ndarray,np.ndarray|None]:
    if values.shape!=visible.shape or visible.shape!=recorded.shape or (
        not np.isin(visible,[0.,1.]).all() or
        not np.isin(recorded,[0.,1.]).all() or
        np.any(visible>recorded)
    ):
        raise ValueError("Inconsistent visible and original measurement process")
    probabilities=[]
    contributions=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(values),batch_size):
            stop=min(start+batch_size,len(values))
            output=model(
                torch.from_numpy(values[start:stop]*visible[start:stop]),
                torch.from_numpy(visible[start:stop]),
                process_mask=torch.from_numpy(recorded[start:stop]),
            )
            if not torch.isfinite(output.logits).all() or (
                float(output.decomposition_error().detach())>1e-5
            ):
                raise FloatingPointError("Unstable or non-additive concept model")
            probabilities.append(output.probabilities.detach().numpy().astype(np.float64))
            if collect_contributions:
                contributions.append(
                    output.physical.contributions.detach().numpy().astype(np.float64)
                )
    return np.concatenate(probabilities),(
        np.concatenate(contributions) if collect_contributions else None
    )


def _reference_score(
    method:str, seed:int, dev_auprc:float,
    *, v11_records:pd.DataFrame, external_records:pd.DataFrame,
    tolerance:float,
)->dict:
    if method in ARMS:
        rows=v11_records[
            v11_records.method.eq(method)
            & v11_records.scenario.eq("original")
            & v11_records.seed.eq(seed)
        ]
        metric="auprc_full"
    else:
        rows=external_records[
            external_records.model.eq(method)
            & external_records.scenario.eq("original")
            & external_records.seed.eq(seed)
        ]
        metric="auprc"
    if len(rows)!=1:
        raise ValueError(f"Frozen previous DEV model receipt missing {method} seed {seed}")
    ref=float(rows.iloc[0][metric])
    diff=abs(ref-dev_auprc)
    if not np.isfinite(ref) or diff>tolerance:
        raise RuntimeError(
            f"REFIT DRIFT before Set B access: {method} seed {seed}, "
            f"prior dev AUPRC {ref:.7f}, refit {dev_auprc:.7f}, tol {tolerance:.5f}"
        )
    return {"method":method,"seed":seed,"frozen_previous_dev_auprc":ref,
            "retrained_same_dev_auprc":dev_auprc,
            "absolute_difference":diff,
            "eligible_for_independent_B_evaluation":True}


def main()->None:
    p=argparse.ArgumentParser()
    for name in ("config","reference-v6-config","reference-v7-config",
                 "reference-v10-config","reference-v11-config",
                 "frozen-v10-receipt","frozen-v11-artifact",
                 "frozen-external-artifact","set-a-zip","outcomes-a",
                 "set-b-zip","outcomes-b","out-dir"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    if a.out_dir.exists():
        p.error("ONE-SHOT experiment refuses any existing output directory")
    # Forensic A-ONLY controls proved transformer refit AUPRC changes
    # from 0.4603555 when the original Python hash seed is fixed at 0
    # to 0.4522096 under a random hash seed, despite identical model
    # code and torch 2.15.0+cpu. Must reproduce original environment
    # BEFORE any novel Set B patient source can be accessed.
    if os.environ.get("PYTHONHASHSEED")!="0":
        p.error("Original external Transformer baseline requires PYTHONHASHSEED=0")
    # The new cohort must be absent at process entry. The SAME process
    # performs the complete frozen Set A refit check before fetching it.
    if a.set_b_zip.exists() or a.outcomes_b.exists():
        p.error("Set B files must not exist before DEV refit verification")
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    ref=yaml.safe_load(a.reference_v6_config.read_text(encoding="utf-8"))
    v7=yaml.safe_load(a.reference_v7_config.read_text(encoding="utf-8"))
    v10=yaml.safe_load(a.reference_v10_config.read_text(encoding="utf-8"))
    v11=yaml.safe_load(a.reference_v11_config.read_text(encoding="utf-8"))
    v10_receipt=json.loads(a.frozen_v10_receipt.read_text(encoding="utf-8"))
    s=cfg["protocol"]
    if not (s["only_set_a_training"] and s["set_b_access_after_refit_verification"]
            and s["fixed_methods_before_holdout"] and
            s["no_set_a_test_or_calibration_evaluation"] and
            s["set_b_is_same_mimic_ii_source"] and s["one_shot_no_retuning"]
            and s["no_patient_level_export"]
            and v10_receipt["status"]=="EXPLORATORY_V10_MATCHED_TRAINING_COMPLETE"):
        p.error("Frozen cross-cohort audit contract missing")
    if tuple(cfg["methods"])!=METHODS or tuple(cfg["seeds"])!=SEEDS:
        p.error("All 7 predeclared architecture families and three seeds required")
    if tuple(cfg["evaluation"]["scenarios"])!=SCENARIOS:
        p.error("Fixed historical + MCAR + MAR scenarios required")
    if cfg["evaluation"]["holdout_expected_patients"]!=4000:
        p.error("Set B is exactly 4000 independent patient records")
    if cfg["source"]["prior_v11_run"]!=REFERENCE_V11_RUN or (
        cfg["source"]["prior_external_run"]!=REFERENCE_V10_EXTERNAL_RUN
    ):
        p.error("Do not change historical reference evidence")
    torch.set_num_threads(int(ref["training"]["cpu_threads"]))
    v11_source=a.frozen_v11_artifact
    external_source=a.frozen_external_artifact
    if (v11_source/"v11_evidence").is_dir():
        v11_source=v11_source/"v11_evidence"
    if (external_source/"published_external").is_dir():
        external_source=external_source/"published_external"
    v11_receipt=json.loads((v11_source/"STATUS.json").read_text())
    external_receipt=json.loads((external_source/"STATUS.json").read_text())
    if not (
        v11_receipt["status"]=="V11_IDENTIFIABILITY_AWARE_GROUPED_EXPERIMENT_COMPLETE"
        and external_receipt["status"]=="V10_EXTERNAL_ARCHITECTURE_BASELINES_DEV_COMPLETED"
        and v11_receipt["original_calibration_test_never_evaluated"]
        and not external_receipt["test_used"]
    ):
        p.error("Not the genuine previous source model receipts")
    v11_records=pd.read_csv(v11_source/"v11_matched_per_run_and_scenario.csv")
    external_records=pd.read_csv(external_source/"v10_vs_external_all_run_metrics.csv")

    # Phase 1: A-only train and frozen DEV refit audit.
    data,basis,dev,cohort=_load_and_match_v10_patient_cohort(
        source_zip=a.set_a_zip,outcome_file=a.outcomes_a,
        base_cfg=ref,prior_receipt=v10_receipt,
    )
    with zipfile.ZipFile(a.set_a_zip) as source_a:
        old_train_ids={int(Path(path).stem)
                       for path in source_a.namelist()
                       if path.lower().endswith(".txt")}
    if len(old_train_ids)!=4000:
        raise ValueError("Source Set A record manifest unexpectedly altered")
    order=np.random.default_rng(int(ref["data"]["pilot_selection_seed"])).permutation(
        len(data.train.y)
    )[int(ref["data"]["pilot_train_patients"]):]
    c=v11["predictive_certificate"]
    cert=fit_predictive_group_certificate(
        data.train.x[order],data.train.mask[order],
        computational_rank=basis.shape[1],lags=tuple(c["lags"]),
        bootstraps=int(c["patient_bootstraps"]),seed=int(c["seed"]),
        minimum_subspace_overlap=float(c["min_subspace_overlap"]),
        gap_noise_quantile=float(c["gap_noise_quantile"]),
    )
    if projector_overlap(basis,cert.reference_basis[:,:basis.shape[1]])<.999:
        raise ValueError("Source train-only subspace changed before Set B")
    model_objects={}
    refit_audit=[]
    timings=[]
    defs={}
    for method in METHODS:
        seedlist=(0,) if method in TABULAR_EXTERNAL else SEEDS
        model_objects[method]={}
        defs[method]={}
        for seed in seedlist:
            start=time.perf_counter()
            if method==ARMS[0]:
                model,train=fit_multiscale(data,basis,seed,ref,v7,v10,consistency=True)
            elif method in ARMS[1:]:
                model,train=fit_identifiability_aware_v11(
                    data,basis,cert,seed,ref,v7,v10,v11,variant=method,
                )
            elif method in TABULAR_EXTERNAL:
                model=fit_tabular_baseline(
                    method,data.train.x,data.train.mask,data.train.mask,
                    data.train.y,
                    augmentation_fraction=float(cfg["external_training"]["augmentation_fraction"]),
                    seed=int(cfg["external_training"]["tabular_seed"]),
                )
                train={"architecture":method,"tabular_augmentation":True}
            else:
                e=cfg["external_training"]
                model,train=fit_neural_baseline(
                    method,data.train,seed=seed,epochs=int(e["epochs"]),
                    batch_size=int(e["batch_size"]),
                    learning_rate=float(e["learning_rate"]),
                    weight_decay=float(e["weight_decay"]),
                    augmentation_fraction=float(e["augmentation_fraction"]),
                )
            elapsed=time.perf_counter()-start
            if method in ARMS:
                p_dev,_=predict_fixed_concept(
                    model,dev.x,dev.mask,dev.mask,collect_contributions=False,
                )
                defs[method][seed]=expanded_multiscale_definition(model)
            elif method in TABULAR_EXTERNAL:
                p_dev=predict_tabular(model,dev.x,dev.mask,dev.mask)
            else:
                p_dev=predict_neural(model,dev.x,dev.mask,dev.mask)
            dev_score=_model_score(dev.y,p_dev)["auprc"]
            audit=_reference_score(
                method,seed,dev_score,v11_records=v11_records,
                external_records=external_records,
                tolerance=float(cfg["protocol"]["max_prior_dev_auprc_drift"]),
            )
            refit_audit.append(audit)
            model_objects[method][seed]=model
            timings.append({
                "method":method,"seed":seed,"train_seconds":elapsed,
                "epochs":int(ref["training"]["epochs"]) if method in ARMS else (
                    None if method in TABULAR_EXTERNAL else
                    int(cfg["external_training"]["epochs"])
                ),
                "parameters":sum(p.numel() for p in model.parameters())
                             if isinstance(model,torch.nn.Module) else None,
                "previous_receipt_refit_verified":True,
            })
            print("Set A train-only verified",method,seed,"dev AUPRC",
                  round(dev_score,6),flush=True)

    # Phase 2: ONLY NOW do we DOWNLOAD and open the new Set B cohort.
    # The raw source and outcome bytes were absent during every fit/refit.
    download_set_b_only_after_all_refit_guards_pass(
        destination_zip=a.set_b_zip,
        destination_outcomes=a.outcomes_b,
        source_zip_url=cfg["source"]["expected_set_b_url"],
        outcome_url=cfg["source"]["expected_outcomes_b_url"],
    )
    holdout,holdout_meta=load_set_b_strict(
        a.set_b_zip,a.outcomes_b,selected_channels=tuple(data.receipt["channel_names"]),
        medians=data.medians,scales=data.scales,
        set_a_record_ids=old_train_ids,
        history_hours=int(cfg["evaluation"]["history_hours"]),
        expected_n=int(cfg["evaluation"]["holdout_expected_patients"]),
    )
    source_b_seed=int(ref["evaluation"]["measurement_mask_seed"])
    views=_perturbation_views(holdout,source_b_seed)
    all_scores=[]
    predictions=defaultdict(dict)
    contributions=defaultdict(dict)
    for method in METHODS:
        for seed,model in model_objects[method].items():
            predictions[method][seed]={}
            for scenario,(x,m,original_mask) in views.items():
                first=time.perf_counter()
                if method in ARMS:
                    p,phys=predict_fixed_concept(
                        model,x,m,original_mask,
                        collect_contributions=(scenario=="original"),
                    )
                    if scenario=="original":
                        contributions[method][seed]=phys
                elif method in TABULAR_EXTERNAL:
                    p=predict_tabular(model,x,m,original_mask)
                else:
                    p=predict_neural(model,x,m,original_mask)
                elapsed=time.perf_counter()-first
                all_scores.append({
                    "method":method,"seed":seed,"scenario":scenario,
                    "n_set_b_patients":len(holdout.y),
                    "training_set":"Set A official frozen 2400 records",
                    "history_hours":36,"input_channels":holdout.x.shape[-1],
                    "same_mimic_ii_source":True,
                    "set_a_test_and_calibration_remain_sealed":True,
                    "set_b_used_in_training_or_model_selection":False,
                    "inference_seconds_4000":elapsed,
                    **_model_score(holdout.y,p),
                })
                predictions[method][seed][scenario]=p
            print("ONE-SHOT independent Set B",method,seed,
                  "AUPRC",round(all_scores[-3]["auprc"],6),flush=True)
    metrics=pd.DataFrame(all_scores)
    overview=[]
    for method in METHODS:
        for scenario in SCENARIOS:
            group=metrics[metrics.method.eq(method)&metrics.scenario.eq(scenario)]
            overview.append({
                "method":method,"scenario":scenario,
                "independent_model_initializations":len(group),
                "mean_auprc":float(group.auprc.mean()),
                "sd_auprc_across_seeds":float(group.auprc.std(ddof=1))
                        if len(group)>1 else None,
                "mean_auroc":float(group.auroc.mean()),
                "mean_brier_uncalibrated":float(group.brier.mean()),
                "n_new_patients":4000,
                "outcome_prevalence":float(group.prevalence.iloc[0]),
            })
    # New-patient bootstrap, not a cross-hospital uncertainty estimate.
    comparisons=[]
    comp=cfg["comparisons"]
    for candidate,other in comp:
        for scenario in SCENARIOS:
            raw=_paired_bootstrap(
                holdout.y,
                {seed:predictions[candidate][seed][scenario]
                 for seed in predictions[candidate]},
                {seed:predictions[other][seed][scenario]
                 for seed in predictions[other]},
                repetitions=int(cfg["statistics"]["paired_bootstrap_draws"]),
                seed=int(cfg["statistics"]["seed"]) +SCENARIOS.index(scenario)*100
                     +METHODS.index(candidate)*9+METHODS.index(other),
            )
            comparisons.append({
                "scenario":scenario,"candidate":candidate,"comparator":other,
                "mean_auprc_difference_candidate_minus_comparator":
                    raw.pop("delta_auprc_v10_minus_external"),
                "paired_patient_and_model_bootstrap_interval":
                    raw.pop("descriptive_patient_and_training_run_bootstrap_ci95"),
                "same_historical_data_source":True,
                "single_one_shot_external_patient_cohort":True,
                "no_new_hospital_claim":True,
                **raw,
            })
    # Stage-1 source empirical certificate from A; stage-2 signed
    # effect/definition reproducibility now on independent B patients.
    full_group_pairs=[]
    for method in ARMS:
        for a_seed,b_seed in ((11,23),(11,37),(23,37)):
            effect=compare_group_attributions(
                contributions[method][a_seed],
                contributions[method][b_seed],cert.groups,
            )
            definition=_definition_group_overlap(
                defs[method][a_seed],defs[method][b_seed],cert,
            )
            for e,d in zip(effect,definition):
                full_group_pairs.append({
                    "method":method,"seed_a":a_seed,"seed_b":b_seed,
                    **e,
                    "definition_subspace_overlap":d["definition_subspace_overlap"],
                    "definition_comparison_status":d["status"],
                })
    group_frame=pd.DataFrame(full_group_pairs)
    joint=joint_concept_screen(
        cert.to_json(),group_frame,
        expected_seeds=SEEDS,
        min_every_pair_spearman=float(cfg["statistics"]["effect_min_every_pair_spearman"]),
        min_every_pair_definition_overlap=float(
            cfg["statistics"]["definition_min_every_pair_overlap"]
        ),
        already_repeatedly_inspected_development=False,
    )
    joint["evaluation_cohort"]="Set B new patient records, model/threshold decisions frozen on Set A"
    joint["new_hospital"]=False
    joint["new_patient_holdout_evaluated_once"]=True
    joint["clinical_semantics_proven"]=False

    a.out_dir.mkdir(parents=True,exist_ok=False)
    metrics.to_csv(a.out_dir/"SET_B_ONE_SHOT_ALL_METHODS_PER_SEED.csv",index=False)
    pd.DataFrame(overview).to_csv(a.out_dir/"SET_B_ONE_SHOT_AGGREGATE.csv",index=False)
    pd.DataFrame(comparisons).to_csv(a.out_dir/"SET_B_PAIRED_PATIENT_MODEL_BOOTSTRAP.csv",index=False)
    group_frame.to_csv(a.out_dir/"SET_B_SIGNED_GROUP_EFFECT_PAIR_STABILITY.csv",index=False)
    pd.DataFrame(refit_audit).to_csv(a.out_dir/"SET_A_REFIT_VERIFICATION_BEFORE_B_ACCESS.csv",index=False)
    pd.DataFrame(timings).to_csv(a.out_dir/"SET_A_FIT_RUNTIME.csv",index=False)
    (a.out_dir/"SET_B_FUNCTIONAL_IDENTIFIABILITY_AUDIT.json").write_text(
        json.dumps(joint,indent=2,allow_nan=False),encoding="utf-8"
    )
    status={
        "status":"ONE_SHOT_SET_B_INDEPENDENT_PATIENT_COHORT_COMPLETED",
        "not_independent_hospital":True,
        "same_mimic_ii_historical_source":True,
        "set_a_original_calibration_test_sealed":True,
        "not_used_for_model_selection_or_retraining":True,
        "no_set_b_data_access_before_frozen_dev_verification":True,
        "no_patient_id_predictions_or_checkpoints_in_report":True,
        "risk_score_brier_uncalibrated":True,
        "frozen_reference_v11_run":REFERENCE_V11_RUN,
        "frozen_reference_external_run":REFERENCE_V10_EXTERNAL_RUN,
        "train_only_selected_basis_shape":list(basis.shape),
        "previous_set_a_dev_verified_all_models":True,
        "set_b_metadata":holdout_meta,
        "source_set_a_raw_sha256":sha256_file(a.set_a_zip),
        "source_set_a_outcomes_sha256":sha256_file(a.outcomes_a),
        "prior_dev_cohort":cohort,
        "model_seeds":list(SEEDS),
        "methods":list(METHODS),
        "primary_predeclared_comparisons":comp,
        "summary":overview,
        "paired_bootstrap":comparisons,
        "post_frozen_signed_group_audit_counts":{
            method:{k:v for k,v in d.items() if k.startswith("n_")}
            for method,d in joint["methods"].items()
        },
        "limitations":[
            "Set B is from the same MIMIC-II source, not a different hospital",
            "Only three model initialization seeds for neural estimators",
            "Previous 300-patient Set A DEV was repeatedly used in design",
            "Set B was used once for this protocol and must not be recycled for tuning",
            "Bootstrap conditions on this one hospital-source cohort",
            "Signed concept effect stability is not medical causal identification",
            "Test/calibration from original Set A remain sealed",
        ],
    }
    (a.out_dir/"STATUS.json").write_text(json.dumps(
        status,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(pd.DataFrame(overview).query("scenario=='original'").to_string(index=False),flush=True)
    print("FINAL Set B certificate",joint["post_frozen_signed_group_audit_counts"]
          if "post_frozen_signed_group_audit_counts" in joint else
          status["post_frozen_signed_group_audit_counts"],flush=True)


if __name__=="__main__":
    main()
