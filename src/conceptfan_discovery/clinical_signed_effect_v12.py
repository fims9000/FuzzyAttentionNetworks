"""V12 A-only EXPLORATORY matched evaluation; NEVER reads/scans Set B.

Same original 2,400 Set A TRAIN patients, old 300-case repeatedly inspected
DEV, 36h history, 12h forecast, 12 observed channels and three seeds.
This does not make a new clinical validation: actual Set B has been consumed
by the frozen V11 one-shot experiment and cannot tune V12.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from .clinical_identifiability_v11 import _definition_group_overlap
from .clinical_v10_external_benchmark import (
    _load_and_match_v10_patient_cohort,_model_score,_paired_bootstrap,
    _perturbation_views,
)
from .effect_certification_v11 import joint_concept_screen
from .external_baselines_v10 import fit_tabular_baseline,predict_tabular
from .identifiability_v11 import (
    compare_group_attributions,fit_predictive_group_certificate,projector_overlap,
)
from .magnets_binary_baseline_v12 import (
    fit_magnets_binary_adaptation,predict_magnets_binary,
    train_only_concept_permutation,
)
from .multiscale_physiology_v10 import expanded_multiscale_definition
from .signed_effect_v12 import fit_train_only_signed_anchors
from .train_identifiability_v11 import fit_identifiability_aware_v11
from .train_signed_effect_v12 import fit_signed_effect_v12


SEEDS=(11,23,37)
NATIVE=("v11_projector_group_view","v12_observable_anchor","v12_anchor_signed_effect")
METHODS=NATIVE+("magnets_style_binary_adaptation","hist_gradient_boosting")
SCENARIOS=("original","mcar_thin","time_channel_mar")


def predict_native(model,values,visible,recorded,*,batch_size=128):
    scores,terms=[],[]
    model.eval()
    with torch.no_grad():
        for first in range(0,len(values),batch_size):
            last=min(first+batch_size,len(values))
            output=model(
                torch.from_numpy((values[first:last]*visible[first:last]).astype(np.float32)),
                torch.from_numpy(visible[first:last].astype(np.float32)),
                process_mask=torch.from_numpy(recorded[first:last].astype(np.float32)),
            )
            if (not torch.isfinite(output.logits).all()
                    or float(output.decomposition_error().detach())>1e-5):
                raise ValueError("V12 native logit decomposition invalid")
            scores.append(output.probabilities.numpy().astype(np.float64))
            terms.append(output.physical.contributions.numpy().astype(np.float64))
    return np.concatenate(scores),np.concatenate(terms)


def _write(path:Path,obj:dict)->None:
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")


def run(a):
    cfg=yaml.safe_load(a.config.read_text())
    ref=yaml.safe_load(a.reference_v6_config.read_text())
    v7=yaml.safe_load(a.reference_v7_config.read_text())
    v10=yaml.safe_load(a.reference_v10_config.read_text())
    v11=yaml.safe_load(a.reference_v11_config.read_text())
    receipt=json.loads(a.frozen_v10_receipt.read_text())
    p=cfg["protocol"]
    if not (p["set_b_consumed_by_frozen_v11_evaluation_never_used_in_v12"]
            and p["original_set_a_test_and_calibration_remain_sealed"]
            and p["no_raw_patient_data_or_predictions_in_artifacts"]
            and cfg["source"]["forbid_download_or_read_set_b"]
            and cfg["source"]["prior_v11_setb_first_completion"]==41710356212
            and tuple(cfg["training"]["methods"])==METHODS
            and tuple(cfg["training"]["seeds"])==SEEDS
            and tuple(cfg["evaluation"]["scenarios"])==SCENARIOS
            and cfg["evaluation"]["n_previous_development_cases"]==300):
        raise ValueError("Preregistered A-ONLY exploratory V12 contract not satisfied")
    if a.out_dir.exists():
        raise FileExistsError("Never overwrite previous scientific audit")
    if receipt["status"]!="EXPLORATORY_V10_MATCHED_TRAINING_COMPLETE":
        raise ValueError("Original train-only V10 receipt required")
    v11_source=a.frozen_v11_artifact
    if (v11_source/"v11_evidence").is_dir():
        v11_source=v11_source/"v11_evidence"
    prior_status=json.loads((v11_source/"STATUS.json").read_text())
    if (prior_status["status"]!="V11_IDENTIFIABILITY_AWARE_GROUPED_EXPERIMENT_COMPLETE"
            or not prior_status["original_calibration_test_never_evaluated"]):
        raise ValueError("Original frozen V11 comparator receipt invalid")
    prior=pd.read_csv(v11_source/"v11_matched_per_run_and_scenario.csv")
    torch.set_num_threads(int(ref["training"]["cpu_threads"]))
    data,basis,dev,cohort=_load_and_match_v10_patient_cohort(
        source_zip=a.set_a_zip,outcome_file=a.outcomes_a,
        base_cfg=ref,prior_receipt=receipt,
    )
    if len(data.train.y)!=2400 or len(dev.y)!=300:
        raise ValueError("Changed original development sampling contract")
    order=np.random.default_rng(int(ref["data"]["pilot_selection_seed"])).permutation(
        len(data.train.y)
    )[int(ref["data"]["pilot_train_patients"]):]
    c=v11["predictive_certificate"]
    source=fit_predictive_group_certificate(
        data.train.x[order],data.train.mask[order],
        computational_rank=basis.shape[1],lags=tuple(c["lags"]),
        bootstraps=int(c["patient_bootstraps"]),seed=int(c["seed"]),
        minimum_subspace_overlap=float(c["min_subspace_overlap"]),
        gap_noise_quantile=float(c["gap_noise_quantile"]),
    )
    if projector_overlap(basis,source.reference_basis[:,:basis.shape[1]])<.999:
        raise ValueError("Unexpected changed Set A TRAIN predictive subspace")
    conf=cfg["source_anchors"]
    anchors=fit_train_only_signed_anchors(
        data.train.x,data.train.mask,data.train.y,source,
        bootstraps=int(conf["patient_bootstraps"]),
        permutations=int(conf["train_only_label_permutations"]),
        seed=int(conf["seed"]),familywise_alpha=float(conf["bonferroni_familywise_alpha"]),
        min_absolute_class_difference=float(conf["minimum_absolute_train_class_difference"]),
        history_steps=int(cfg["training"]["anchor_history_hours"]),
    )
    views=_perturbation_views(dev,int(ref["evaluation"]["measurement_mask_seed"]))
    scores=[]
    fit_info=[]
    contributions={}
    definitions={}
    probabilities={}
    for method in METHODS:
        contributions[method],definitions[method],probabilities[method]={},{},{}
        method_seeds=(0,) if method=="hist_gradient_boosting" else SEEDS
        for seed in method_seeds:
            start=time.perf_counter()
            if method=="v11_projector_group_view":
                model,training=fit_identifiability_aware_v11(
                    data,basis,source,seed,ref,v7,v10,v11,variant=method,
                )
                definitions[method][seed]=expanded_multiscale_definition(model)
            elif method in NATIVE:
                model,training=fit_signed_effect_v12(
                    data,anchors.basis,source,anchors,seed,
                    ref,v7,v10,v11,cfg,variant=method,
                )
                definitions[method][seed]=expanded_multiscale_definition(model)
            elif method=="magnets_style_binary_adaptation":
                model,training=fit_magnets_binary_adaptation(
                    data.train,seed,n_concepts=basis.shape[1],
                    epochs=int(ref["training"]["epochs"]),
                    batch_size=int(ref["training"]["batch_size"]),
                    learning_rate=float(ref["training"]["learning_rate"]),
                    weight_decay=float(ref["training"]["weight_decay"]),
                    augmentation_fraction=float(ref["training"]["main_mask_fraction"]),
                    n_masks=int(cfg["training"]["mask_aggregate_architecture"]["n_masks"]),
                )
            else:
                model=fit_tabular_baseline(
                    method,data.train.x,data.train.mask,data.train.mask,
                    data.train.y,augmentation_fraction=.20,seed=11,
                )
                training={"model":"independent fixed HGB", "train_only":True}
            fit_elapsed=time.perf_counter()-start
            probabilities[method][seed],contributions[method][seed]={},{}
            for scenario,(x,visible,recorded) in views.items():
                inference_start=time.perf_counter()
                if method in NATIVE:
                    probability,terms=predict_native(model,x,visible,recorded)
                elif method=="magnets_style_binary_adaptation":
                    probability,terms=predict_magnets_binary(model,x,visible,recorded)
                else:
                    probability=predict_tabular(model,x,visible,recorded)
                    terms=None
                seconds=time.perf_counter()-inference_start
                probabilities[method][seed][scenario]=probability
                if terms is not None:
                    contributions[method][seed][scenario]=terms
                row={
                    "method":method,"seed":seed,"scenario":scenario,
                    "previously_inspected_set_a_dev_only":True,
                    "new_set_b_holdout_not_used":True,
                    "training_patients":len(data.train.y),"dev_patients":len(dev.y),
                    "history_hours":dev.x.shape[1],
                    "input_channels":dev.x.shape[-1],
                    "train_seconds":fit_elapsed,
                    "dev_inference_seconds":seconds,
                    **_model_score(dev.y,probability),
                }
                scores.append(row)
            fit_info.append({"method":method,"seed":seed,**training})
            dev_auprc=_model_score(dev.y,probabilities[method][seed]["original"])["auprc"]
            print("EXPLORATORY A-ONLY",method,seed,"DEV AUPRC",round(dev_auprc,6),flush=True)
            if method=="v11_projector_group_view":
                r=prior[(prior.method==method)&(prior.seed==seed)&prior.scenario.eq("original")]
                if len(r)!=1 or abs(float(r.iloc[0]["auprc_full"])-dev_auprc)>.002:
                    raise RuntimeError("Original V11 frozen baseline does not reproduce")
            # Store ONLY aggregate outcomes; never patient-level prediction export.
            if method=="magnets_style_binary_adaptation":
                rng=np.random.default_rng(20261104)
                sample=np.sort(rng.choice(
                    len(data.train.y),size=min(192,len(data.train.y)),replace=False
                ))
                _,train_contributions=predict_magnets_binary(
                    model,data.train.x[sample],data.train.mask[sample],data.train.mask[sample],
                )
                contributions[method][seed]["train_only_anchor"]=train_contributions
    frame=pd.DataFrame(scores)
    summary=frame.groupby(["method","scenario"],sort=False).agg(
        n_independent_fits=("seed","nunique"),
        mean_auprc=("auprc","mean"),
        sd_auprc=("auprc","std"),
        mean_auroc=("auroc","mean"),
        mean_brier=("brier","mean"),
        mean_fit_seconds=("train_seconds","mean"),
    ).reset_index()
    native_pairs=[]
    for method in NATIVE:
        for sa,sb in itertools.combinations(SEEDS,2):
            effect=compare_group_attributions(
                contributions[method][sa]["original"],
                contributions[method][sb]["original"],source.groups,
            )
            definition=_definition_group_overlap(
                definitions[method][sa],definitions[method][sb],source,
            )
            for e,d in zip(effect,definition):
                native_pairs.append({
                    "method":method,"seed_a":sa,"seed_b":sb,**e,
                    "definition_subspace_overlap":d["definition_subspace_overlap"],
                    "definition_comparison_status":d["status"],
                })
    pair_frame=pd.DataFrame(native_pairs)
    certificate=joint_concept_screen(
        source.to_json(),pair_frame,
        already_repeatedly_inspected_development=True,
        expected_seeds=SEEDS,
        min_every_pair_spearman=float(cfg["evaluation"]["signed_pairwise_min_spearman"]),
        min_every_pair_definition_overlap=float(cfg["evaluation"]["learned_definition_min_overlap"]),
    )
    # Different mask-aggregate models have unordered latent concept slots.
    # Assignment is permitted ONLY using train-case contributions. A negative
    # matched correlation stays NEGATIVE, and weak matches abstain.
    magnet=contributions["magnets_style_binary_adaptation"]
    maps={11:np.arange(basis.shape[1],dtype=int)}
    for seed in (23,37):
        maps[seed]=train_only_concept_permutation(
            magnet[11]["train_only_anchor"],magnet[seed]["train_only_anchor"],
            min_abs_spearman=float(cfg["evaluation"]["min_train_magnets_alignment_abs_spearman"]),
        )
    mags=[]
    for sa,sb in itertools.combinations(SEEDS,2):
        for slot in range(basis.shape[1]):
            ia,ib=int(maps[sa][slot]),int(maps[sb][slot])
            if ia<0 or ib<0:
                rho=None
            else:
                a=magnet[sa]["original"][:,ia]
                b=magnet[sb]["original"][:,ib]
                rho=float(spearmanr(a,b).statistic) if (
                    np.std(a)>1e-10 and np.std(b)>1e-10
                ) else None
            mags.append({"seed_a":sa,"seed_b":sb,"train_aligned_slot":slot,
                         "train_supported_matching":ia>=0 and ib>=0,
                         "heldout_DEV_same_patient_signed_spearman":rho})
    mags_df=pd.DataFrame(mags)
    mags_map={str(seed):mapping.tolist() for seed,mapping in maps.items()}
    comparison=[]
    for other in ("v11_projector_group_view","magnets_style_binary_adaptation"):
        for scenario in SCENARIOS:
            raw=_paired_bootstrap(
                dev.y,
                {s:probabilities["v12_anchor_signed_effect"][s][scenario] for s in SEEDS},
                {s:probabilities[other][s][scenario] for s in SEEDS},
                repetitions=int(cfg["evaluation"]["descriptive_bootstrap_draws"]),
                seed=int(cfg["evaluation"]["bootstrap_seed"])+SCENARIOS.index(scenario)
                    +METHODS.index(other)*31,
            )
            comparison.append({"candidate":"v12_anchor_signed_effect",
                               "other":other,"scenario":scenario,
                               "nonconfirmatory_repeated_DEV":True,**raw})
    a.out_dir.mkdir(parents=True,exist_ok=False)
    frame.to_csv(a.out_dir/"SET_A_PREVIOUS_DEV_V12_PER_SEED.csv",index=False)
    summary.to_csv(a.out_dir/"SET_A_PREVIOUS_DEV_V12_SUMMARY.csv",index=False)
    pd.DataFrame(fit_info).to_csv(a.out_dir/"TRAINING_RECEIPTS_V12.csv",index=False)
    pair_frame.to_csv(a.out_dir/"V12_SIGNED_GROUP_PAIRS_PREVIOUS_DEV.csv",index=False)
    mags_df.to_csv(a.out_dir/"MAGNETS_STYLE_TRAIN_MATCHED_DEV_EFFECTS.csv",index=False)
    pd.DataFrame(comparison).to_csv(a.out_dir/"DESCRIPTIVE_ONLY_DEV_BOOTSTRAP.csv",index=False)
    _write(a.out_dir/"TRAIN_ONLY_SOURCE_AND_SIGN_CERTIFICATE.json",anchors.to_json())
    _write(a.out_dir/"SIGNED_EFFECT_AND_DEFINITION_SCREEN.json",certificate)
    _write(a.out_dir/"MASK_AGGREGATE_TRAIN_ONLY_ALIGNMENT.json",maps_json if False else {
        "baseline":"INDEPENDENT MAGNETS_STYLE_BINARY_ADAPTATION_NOT_OFFICIAL",
        "matched_using":"TRAIN only, no DEV or Set B matching",
        "minimum_abs_train_spearman":float(
            cfg["evaluation"]["min_train_magnets_alignment_abs_spearman"]
        ),
        "per_seed_reference_permutations_negative_one_means_abstain":mags_map,
        "never_correct_signed_effect_direction":True,
        "original_MAGNETS_task_is_regression":True,
    })
    _write(a.out_dir/"STATUS.json",{
        "status":"EXPLORATORY_V12_PREVIOUS_DEV_SET_A_COMPLETE",
        "not_new_holdout_or_confirmatory":True,
        "no_set_b_patient_or_label_access":True,
        "original_set_a_test_and_calibration_still_sealed":True,
        "n_train_patients":len(data.train.y),"n_repeatedly_seen_dev_patients":len(dev.y),
        "n_source_predictive_concept_slots":int(basis.shape[1]),
        "n_train_only_certified_risk_signs":int(np.count_nonzero(anchors.risk_sign)),
        "methods":list(METHODS),"seeds":list(SEEDS),
        "first_successful_v11_set_b_run_NOT_REUSED":41710356212,
        "group_signed_effect_audit_counts":{
            method:{key:value for key,value in certificate["methods"][method].items()
                    if key.startswith("n_")}
            for method in NATIVE
        },
        "summary":summary.astype(object).where(pd.notna(summary),None).to_dict("records"),
        "limitations":[
            "All predictive method comparisons are on previously inspected Set A DEV",
            "TRAIN label association imposed risk signs; cannot claim forced sign is independently identified",
            "MAGNETS-style method is classification adaptation, NOT original authors' code",
            "Matched concept slots from TRAIN only; no clinical semantics certified",
            "Three training seeds cannot establish broad distributional confidence",
            "Set B consumed by earlier frozen V11 one-shot and never reused for V12",
            "Set A original test and calibration remain sealed, no external hospital",
        ],
    })
    print(summary[summary.scenario.eq("original")].to_string(index=False),flush=True)
    print("V12 strict signed effect audit",{
        name:certificate["methods"][name]["n_candidate_individual_effects"]
        for name in NATIVE
    },flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("config","reference-v6-config","reference-v7-config",
                 "reference-v10-config","reference-v11-config",
                 "frozen-v10-receipt","frozen-v11-artifact",
                 "set-a-zip","outcomes-a","out-dir"):
        p.add_argument("--"+name,required=True,type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()
