"""V11 matched same-patient study: is grouping reproducible where axes are not?

No reusing the accepted PIERE's test patients, no clinical identification
claims, no artificial axis matching inside unresolved singular modes.

Baseline is V10 with cross-view consistency; new arms replace the
arbitrary SVD axis anchor with rotation-invariant group projectors, and
separately replace the cross-view loss by a rotation-invariant group Gram.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .clinical_multiscale_v10 import fit_multiscale
from .clinical_dual_process_experiment_v7 import evaluate_dual
from .clinical_v10_external_benchmark import _load_and_match_v10_patient_cohort
from .dual_process_stability_v7 import collect_observation_process_evidence
from .evaluation import compare_independent_discoveries, evaluate_discovery
from .identifiability_v11 import (
    PredictiveCertificate, compare_group_attributions,
    fit_predictive_group_certificate, grouped_logit_contributions,
    projector_overlap,
)
from .model_v2 import masked_observation_view
from .multiscale_physiology_v10 import expanded_multiscale_definition
from .train_identifiability_v11 import fit_identifiability_aware_v11
from .passports_v11 import identified_predictive_passports
from .v10_seed_sensitivity import exact_seed_sensitivity


ARMS=("v10_consistency_refit","v11_projector_axis_view","v11_projector_group_view")


def _write_json(path:Path,data:dict|list)->None:
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")


def _definition_group_overlap(
    left:np.ndarray,right:np.ndarray,certificate:PredictiveCertificate,
)->list[dict]:
    if left.shape!=right.shape or left.shape[0]!=certificate.selected_computational_rank:
        raise ValueError("Independent definition supports must align on sensor/time operator space")
    result=[]
    for g in certificate.groups:
        a=left[g.first:g.stop].T
        b=right[g.first:g.stop].T
        # Check numerical rank BEFORE creating a basis. A null/inert feature
        # group is ABSTAIN, not a falsely stable invented singular direction.
        for name,z in (("left",a),("right",b)):
            if np.linalg.svd(z,compute_uv=False)[-1]<1e-7:
                result.append({"slots":list(g.members),
                               "certificate":g.status,
                               "definition_subspace_overlap":None,
                               "status":f"ABSTAIN_RANK_DEFICIENT_{name}"})
                break
        else:
            qa=np.linalg.svd(a,full_matrices=False)[0][:,:g.stop-g.first]
            qb=np.linalg.svd(b,full_matrices=False)[0][:,:g.stop-g.first]
            result.append({
                "slots":list(g.members),
                "certificate":g.status,
                "definition_subspace_overlap":projector_overlap(qa,qb),
                "status":"DEFINITION_SUBSPACE_COMPARABLE",
            })
    return result


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--reference-v6-config",type=Path,required=True)
    parser.add_argument("--reference-v7-config",type=Path,required=True)
    parser.add_argument("--reference-v10-config",type=Path,required=True)
    parser.add_argument("--frozen-v10-receipt",type=Path,required=True)
    parser.add_argument("--raw-zip",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    a=parser.parse_args()
    if a.out_dir.exists():
        parser.error("Never overwrite another exploratory result")
    cfg=yaml.safe_load(a.config.read_text())
    ref=yaml.safe_load(a.reference_v6_config.read_text())
    v7=yaml.safe_load(a.reference_v7_config.read_text())
    v10=yaml.safe_load(a.reference_v10_config.read_text())
    prior=json.loads(a.frozen_v10_receipt.read_text())
    if not (cfg["protocol"]["test_and_calibration_sealed"] and
            cfg["protocol"]["no_medical_concept_identity_claim"] and
            cfg["protocol"]["train_only_predictive_groups"] and
            prior["status"]=="EXPLORATORY_V10_MATCHED_TRAINING_COMPLETE" and
            v10["protocol"]["test_calibration_sealed"]):
        parser.error("Unsafe data/provenance scope: original test must stay sealed")
    if tuple(cfg["training"]["methods"])!=ARMS:
        parser.error("All three preregistered training ablations required")
    seeds=tuple(map(int,cfg["training"]["initialization_seeds"]))
    if seeds!=(11,23,37):
        parser.error("Use identical 3 source initializations")
    torch.set_num_threads(int(ref["training"]["cpu_threads"]))
    data,reference_basis,dev,cohort=_load_and_match_v10_patient_cohort(
        source_zip=a.raw_zip,outcome_file=a.outcomes,
        base_cfg=ref,prior_receipt=prior,
    )
    indices=np.random.default_rng(int(ref["data"]["pilot_selection_seed"])).permutation(
        len(data.train.y)
    )[int(ref["data"]["pilot_train_patients"]):]
    c=cfg["predictive_certificate"]
    cert=fit_predictive_group_certificate(
        data.train.x[indices],data.train.mask[indices],
        computational_rank=reference_basis.shape[1],
        lags=tuple(map(int,c["lags"])),
        bootstraps=int(c["patient_bootstraps"]),
        seed=int(c["seed"]),
        minimum_subspace_overlap=float(c["min_subspace_overlap"]),
        gap_noise_quantile=float(c["gap_noise_quantile"]),
    )
    if cert.selected_computational_rank !=reference_basis.shape[1]:
        raise ValueError("Do not relabel the original computational K")
    full_overlap=projector_overlap(
        reference_basis,cert.reference_basis[:,:cert.selected_computational_rank]
    )
    if full_overlap<.999:
        raise ValueError("New train certificate does not match frozen V10 source subspace")
    a.out_dir.mkdir(parents=True,exist_ok=False)
    _write_json(a.out_dir/"TRAIN_ONLY_PREDICTIVE_CERTIFICATE.json",{
        **cert.to_json(),
        "comparison_to_original_V10_predictive_source_projector":full_overlap,
        "patient_count_used_to_fit_certificate":int(len(indices)),
        "training_labels_used_to_fit_certificate":False,
        "patient_test_labels_or_features_used":False,
        "selected_sensor_names":list(data.receipt["channel_names"]),
    })
    all_runs=[]
    all_pairs=[]
    all_group_pairs=[]
    evidence={}
    definitions={}
    for method in ARMS:
        evidence[method]={}
        definitions[method]={}
        for seed in seeds:
            if method==ARMS[0]:
                model,training=fit_multiscale(
                    data,reference_basis,seed,ref,v7,v10,consistency=True,
                )
            else:
                model,training=fit_identifiability_aware_v11(
                    data,reference_basis,cert,seed,ref,v7,v10,cfg,
                    variant=method,
                )
            model.eval()
            # A trained model-native passport is sufficient to reproduce the
            # actual concept formula and boundaries without patient examples.
            # A source SVD axis is not a medical diagnosis. Report this
            # alongside the separate post-fit signed-effect audit.
            _write_json(
                a.out_dir/f"concept_formulas_{method}_init_{seed}.json",
                identified_predictive_passports(model,cert,max_operator_terms=8)
            )
            physical=evaluate_discovery(model.physical,dev)
            trained_definition=expanded_multiscale_definition(model)
            aligned=replace(physical,weights=trained_definition)
            evidence[method][seed]=aligned
            definitions[method][seed]=trained_definition
            with torch.no_grad():
                xv=torch.from_numpy(dev.x)
                mv=torch.from_numpy(dev.mask)
                hidden_x,visible,withheld=masked_observation_view(
                    xv,mv,
                    fraction=float(ref["training"]["main_mask_fraction"]),
                    generator=torch.Generator().manual_seed(
                        int(ref["evaluation"]["measurement_mask_seed"])+900
                    ),
                )
                reconstruction=model.physical(hidden_x,visible).reconstruction
                masked_error=float(
                    (((reconstruction-xv).square()*withheld).sum()/
                     withheld.sum().clamp_min(1.)).sqrt()
                )
            for i,scenario in enumerate(cfg["evaluation"]["scenarios"]):
                metrics=evaluate_dual(
                    model,dev,scenario=scenario,
                    perturbation_seed=int(ref["evaluation"]["measurement_mask_seed"])+i,
                )
                metrics.pop("_raw_mean_probability",None)
                metrics.pop("_raw_phys_contribution",None)
                metrics.pop("_raw_process_contribution",None)
                all_runs.append({
                    "method":method,"seed":seed,"scenario":scenario,
                    "physical_compute_K":cert.selected_computational_rank,
                    "n_train_only_groups":len(cert.groups),
                    "n_empirically_resolved_singletons":sum(
                        g.status=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
                        for g in cert.groups
                    ),
                    "no_clinical_identity_claim":True,
                    "masked_reconstruction_rmse":masked_error,
                    "future_observation_rmse":
                        float(physical.metrics["future_observed_rmse"]),
                    "physical_effective_K":
                        float(physical.metrics["mean_effective_concept_count"]),
                    **metrics,**training,
                })
            print(method,seed,"AUPRC original",
                  round(all_runs[-3]["auprc_full"],4),flush=True)
        for seed_left,seed_right in itertools.combinations(seeds,2):
            ev_a=evidence[method][seed_left]
            ev_b=evidence[method][seed_right]
            pair=compare_independent_discoveries(ev_a,ev_b)
            all_pairs.append({"method":method,"seed_a":seed_left,
                              "seed_b":seed_right,**pair})
            attr=compare_group_attributions(
                ev_a.contributions,ev_b.contributions,cert.groups
            )
            define=_definition_group_overlap(
                definitions[method][seed_left],
                definitions[method][seed_right],cert,
            )
            for group,dl in zip(attr,define):
                all_group_pairs.append({
                    "method":method,"seed_a":seed_left,"seed_b":seed_right,
                    **group,
                    "definition_subspace_overlap":dl["definition_subspace_overlap"],
                    "definition_comparison_status":dl["status"],
                })
    records=pd.DataFrame(all_runs)
    pair_frame=pd.DataFrame(all_pairs)
    group_frame=pd.DataFrame(all_group_pairs)
    records.to_csv(a.out_dir/"v11_matched_per_run_and_scenario.csv",index=False)
    pair_frame.to_csv(a.out_dir/"v11_individual_explanations_not_certified.csv",index=False)
    group_frame.to_csv(a.out_dir/"v11_empirical_group_explanation_stability.csv",index=False)
    overview=[]
    for method in ARMS:
        part=records[(records.method==method)&records.scenario.eq("original")]
        paired=pair_frame[pair_frame.method==method]
        group=group_frame[group_frame.method==method]
        def mean_valid(series)->float|None:
            v=pd.to_numeric(series,errors="coerce").dropna()
            return float(v.mean()) if len(v) else None
        overview.append({
            "method":method,
            "model_count":len(part),
            "original_auprc":float(part.auprc_full.mean()),
            "mcar_auprc":float(records[(records.method==method)&(
                records.scenario=="mcar_thin")].auprc_full.mean()),
            "mar_auprc":float(records[(records.method==method)&(
                records.scenario=="time_channel_mar")].auprc_full.mean()),
            "physical_only_auprc":float(part.auprc_physical_only.mean()),
            "care_only_auprc":float(part.auprc_observation_process_only.mean()),
            "observed_feature_reconstruction_rmse":float(
                part.masked_reconstruction_rmse.mean()),
            "future_prediction_rmse":float(part.future_observation_rmse.mean()),
            "mean_effective_concept_count":float(part.physical_effective_K.mean()),
            "individual_case_specific_spearman_excess_DESCRIPTIVE":mean_valid(
                paired.excess_over_unmatched_episode_null_spearman),
            "group_case_spearman_DESCRIPTIVE":mean_valid(
                group.same_patient_spearman),
            "group_definition_overlap_DESCRIPTIVE":mean_valid(
                group.definition_subspace_overlap),
            "groups_empirically_individual_certified":sum(
                g.status=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
                for g in cert.groups),
            "groups_only_or_abstained":sum(
                g.status!="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
                for g in cert.groups),
            "max_exact_additive_error":float(part.max_logit_additivity_error.max()),
            "no_independent_hospital_validation":True,
        })
    pd.DataFrame(overview).to_csv(a.out_dir/"v11_matched_aggregate_summary.csv",index=False)
    base=records[records.method.eq(ARMS[0])&
                 records.scenario.eq("original")].sort_values("seed")
    sensitivities=[]
    for name in ARMS[1:]:
        other=records[records.method.eq(name)&
                      records.scenario.eq("original")].sort_values("seed")
        if list(base.seed)!=list(other.seed):
            raise ValueError("Paired model seeds differ")
        stat=exact_seed_sensitivity(
            other.auprc_full.to_numpy(dtype=float)-
            base.auprc_full.to_numpy(dtype=float)
        )
        sensitivities.append({"challenger":name,**stat})
    _write_json(a.out_dir/"STATUS.json",{
        "status":"V11_IDENTIFIABILITY_AWARE_GROUPED_EXPERIMENT_COMPLETE",
        "study":"same repeated development protocol, not independent replication",
        "v10_only_baseline":True,
        "physiology_concepts_trained_end_to_end":True,
        "no_human_or_oracle_concept_targets":True,
        "predictive_groups_fitted_train_only":True,
        "clinical_concept_identity_demonstrated":False,
        "original_calibration_test_never_evaluated":True,
        "number_of_hospitals":1,
        "n_training_seeds":len(seeds),
        "n_development_patients_previously_seen":len(dev.y),
        "predeclared_concept_group_certificates":[g.to_json() for g in cert.groups],
        "reference_train_cohort":cohort,
        "observed_results":overview,
        "exhaustive_three_seed_sensitivity_NOT_population_CI":sensitivities,
        "limitations":[
            "Degenerate predictive directions cannot be assigned unique concepts",
            "A robust projector does not identify true physiological causes",
            "Group stability may be trivial when only one group remains",
            "Repeated 300-case development cohort cannot validate generalization",
            "The 3 model-pair scores overlap by source model; no naive pair bootstrap",
            "Additional artificial MCAR/MAR does not emulate new hospitals",
        ],
    })
    print(pd.DataFrame(overview).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
