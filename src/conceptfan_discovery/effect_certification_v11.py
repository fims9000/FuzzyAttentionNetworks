"""V11 joint structural + functional identifiability screening (descriptive).

Stage 1 structural certificate: train-only positive-lag eigenvalue gaps
and patient-bootstrap projector stability, obtained without outcome labels.

Stage 2 functional audit: on a FIXED held-out development cohort, every
independently trained model pair must agree in SIGNED per-patient direct
logit contributions, and learned definition subspaces must agree. A
reliably oriented source SVD vector is NOT sufficient: a learned
classifier head can reverse its effect in a different local optimum.

All joint decisions are exploratory screening rules with fixed thresholds
and dependent model pairs, NOT population FWER-valid intervals or true
clinical labels. Retrospective classification on already examined DEV
cannot be treated as confirmatory model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def joint_concept_screen(
    structural_certificate:dict,
    pairwise:pd.DataFrame,
    *,
    expected_seeds:tuple[int,...]=(11,23,37),
    min_every_pair_spearman:float=.5,
    min_every_pair_definition_overlap:float=.75,
    already_repeatedly_inspected_development:bool=True,
)->dict:
    if not 0<min_every_pair_spearman<=1 or not 0<min_every_pair_definition_overlap<=1:
        raise ValueError("Strict positive empirical threshold required")
    groups=structural_certificate.get("groups")
    if not isinstance(groups,list) or not groups:
        raise ValueError("Missing trained, train-only structural concept groups")
    expected_pairs={(a,b) for i,a in enumerate(expected_seeds) for b in expected_seeds[i+1:]}
    used_slots=[]
    for item in groups:
        start=int(item["first_slot_zero_index"])
        stop=int(item["stop_exclusive"])
        if start>=stop or start!=len(used_slots):
            raise ValueError("Predictive groups must partition ordered concept slots")
        used_slots.extend(range(start,stop))
    if len(set(used_slots))!=len(used_slots):
        raise ValueError("Duplicate concept group members")
    needed={
        "method","seed_a","seed_b","slots","certificate",
        "same_patient_spearman","definition_subspace_overlap",
    }
    if not needed.issubset(pairwise.columns):
        raise ValueError(f"Group audit missing {needed-set(pairwise.columns)}")
    screens={}
    for method in sorted(set(pairwise.method.astype(str))):
        subset=pairwise[pairwise.method.astype(str).eq(method)]
        results=[]
        for group in groups:
            members=list(map(int,group["slots_zero_index"]))
            descriptor=str(members)
            rows=subset[subset.slots.astype(str).eq(descriptor)]
            pairs={
                tuple(sorted((int(a),int(b))))
                for a,b in rows[["seed_a","seed_b"]].itertuples(index=False,name=None)
            }
            if pairs!=expected_pairs or len(rows)!=len(expected_pairs):
                raise ValueError(f"Missing/duplicate independent model comparisons: {method} {members}")
            rho=rows.same_patient_spearman.to_numpy(dtype=np.float64)
            overlap=rows.definition_subspace_overlap.to_numpy(dtype=np.float64)
            structurally_stable=group["certificate"] in (
                "INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
                "GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED",
            )
            all_finite=bool(np.isfinite(rho).all() and np.isfinite(overlap).all())
            functional=(
                all_finite
                and bool((rho>=min_every_pair_spearman).all())
                and bool((overlap>=min_every_pair_definition_overlap).all())
            )
            if not structurally_stable:
                decision="ABSTAIN_STRUCTURAL_SUBSPACE"
            elif not functional:
                decision="ABSTAIN_CROSS_SEED_EFFECT_OR_DEFINITION"
            elif len(members)>1:
                decision="CANDIDATE_STABLE_GROUP_ONLY_NO_INDIVIDUAL_IDENTITY"
            else:
                decision="CANDIDATE_REPRODUCIBLE_OBSERVABLE_EFFECT"
            results.append({
                "slots":members,
                "dimension":len(members),
                "train_only_structural_status":group["certificate"],
                "all_seed_pairs_signed_effect_and_definition_agree":functional,
                "minimum_signed_same_patient_spearman":
                    float(np.min(rho)) if np.isfinite(rho).all() else None,
                "mean_signed_same_patient_spearman":
                    float(np.mean(rho)) if np.isfinite(rho).all() else None,
                "minimum_learned_definition_subspace_overlap":
                    float(np.min(overlap)) if np.isfinite(overlap).all() else None,
                "mean_learned_definition_subspace_overlap":
                    float(np.mean(overlap)) if np.isfinite(overlap).all() else None,
                "observed_seed_pairs":len(rows),
                "decision":decision,
                "clinical_meaning_verified":False,
            })
        screens[method]={
            "groups":results,
            "n_candidate_individual_effects":sum(
                item["decision"]=="CANDIDATE_REPRODUCIBLE_OBSERVABLE_EFFECT"
                for item in results
            ),
            "n_candidate_group_only":sum(
                item["decision"]=="CANDIDATE_STABLE_GROUP_ONLY_NO_INDIVIDUAL_IDENTITY"
                for item in results
            ),
            "n_abstained":sum(
                item["decision"].startswith("ABSTAIN") for item in results
            ),
            "n_predicted_observable_axes":len(used_slots),
            "n_observable_axis_structural_singletons":sum(
                len(item["slots"])==1 and
                item["train_only_structural_status"]=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
                for item in results
            ),
        }
    return {
        "type":"EXPLORATORY_TWO_STAGE_IDENTIFIABILITY_AUDIT",
        "structural_stage":"training-only SVD spectral separation and bootstrap projector overlap",
        "functional_stage":"signed patient-specific direct logit contribution and learned group definition agreement across complete model pairs",
        "thresholds_selected_for_followup_not_confirmatory_this_cohort":{
            "every_pair_signed_effect_spearman_min":min_every_pair_spearman,
            "every_pair_definition_projector_overlap_min":min_every_pair_definition_overlap,
        },
        "n_independent_source_initializations":len(expected_seeds),
        "n_pair_scores_not_independent":len(expected_pairs),
        "not_causal_identification":True,
        "patient_oracle_clinical_ground_truth_used":False,
        "already_repeatedly_inspected_development":bool(already_repeatedly_inspected_development),
        "statistical_significance_claim":False,
        "methods":screens,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--frozen-v11-artifact",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error("Do not overwrite prior research audit")
    source=args.frozen_v11_artifact
    if (source/"v11_evidence"/"STATUS.json").is_file():
        source=source/"v11_evidence"
    summary=json.loads((source/"STATUS.json").read_text(encoding="utf-8"))
    if (summary["status"]!="V11_IDENTIFIABILITY_AWARE_GROUPED_EXPERIMENT_COMPLETE"
        or summary["n_development_patients_previously_seen"]!=300
        or not summary["original_calibration_test_never_evaluated"]):
        parser.error("Expected pinned historical validation-only source")
    certificate=json.loads((source/"TRAIN_ONLY_PREDICTIVE_CERTIFICATE.json").read_text())
    pairs=pd.read_csv(source/"v11_empirical_group_explanation_stability.csv")
    result=joint_concept_screen(certificate,pairs)
    result["source_status"]="V11_IDENTIFIABILITY_AWARE_GROUPED_EXPERIMENT_COMPLETE"
    result["original_test_calibration_sealed"]=True
    args.out_dir.mkdir(parents=True,exist_ok=False)
    (args.out_dir/"V11_JOINT_EXPLANATION_CERTIFICATE.json").write_text(
        json.dumps(result,indent=2,allow_nan=False),encoding="utf-8"
    )
    table=[]
    for name,report in result["methods"].items():
        for item in report["groups"]:
            table.append({"method":name,**item})
    pd.DataFrame(table).to_csv(args.out_dir/"V11_JOINT_EXPLANATION_CERTIFICATE.csv",index=False)
    print(json.dumps({
        name:{key:value for key,value in report.items() if key.startswith("n_")}
        for name,report in result["methods"].items()
    },indent=2),flush=True)


if __name__=="__main__":
    main()
