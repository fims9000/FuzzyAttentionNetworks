"""Deterministic scientific interpretation of a *completed* one-shot cohort audit.

Reads ONLY aggregated CSV/JSON artifacts; never patient labels, traces,
record identifiers, model checkpoints or raw cohort data.
No post-hoc threshold tuning, invented p-values, or clinical causal claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARCHITECTURES=(
    "v10_consistency_refit","v11_projector_axis_view",
    "v11_projector_group_view","logistic_history",
    "hist_gradient_boosting","gru_d_style","transformer_timeseries",
)
PRIMARY="original"


def format_one_shot_scientific_report(artifact_directory:Path)->str:
    path=Path(artifact_directory)
    status=json.loads((path/"STATUS.json").read_text(encoding="utf-8"))
    if not (
        status.get("status")=="ONE_SHOT_SET_B_INDEPENDENT_PATIENT_COHORT_COMPLETED"
        and status.get("not_independent_hospital") is True
        and status.get("set_a_original_calibration_test_sealed") is True
        and status.get("not_used_for_model_selection_or_retraining") is True
        and status.get("previous_set_a_dev_verified_all_models") is True
        and status.get("no_patient_id_predictions_or_checkpoints_in_report") is True
    ):
        raise ValueError("Only completed, unretuned, fully verified one-shot evidence allowed")
    table=pd.read_csv(path/"SET_B_ONE_SHOT_AGGREGATE.csv")
    primary=table.loc[table.scenario.eq(PRIMARY)]
    if len(primary)!=len(ARCHITECTURES) or set(primary.method)!=set(ARCHITECTURES):
        raise ValueError("Missing or duplicated prespecified method on original holdout")
    if (not np.isfinite(primary[["mean_auprc","mean_auroc",
                              "mean_brier_uncalibrated"]]).all().all()
        or not (primary.n_new_patients==4000).all()):
        raise ValueError("Invalid complete 4000-case new-cohort aggregate")
    compared=pd.read_csv(path/"SET_B_PAIRED_PATIENT_MODEL_BOOTSTRAP.csv")
    joint=json.loads((path/"SET_B_FUNCTIONAL_IDENTIFIABILITY_AUDIT.json").read_text())
    effects=joint.get("methods",{})
    if not all(name in effects for name in ARCHITECTURES[:3]):
        raise ValueError("Missing original Set A structural + new Set B functional audit")
    def row(name:str)->pd.Series:
        return primary[primary.method.eq(name)].iloc[0]
    v10=float(row(ARCHITECTURES[0]).mean_auprc)
    v11=float(row(ARCHITECTURES[2]).mean_auprc)
    transformer=float(row("transformer_timeseries").mean_auprc)
    counts={
        m:int(effects[m]["n_candidate_individual_effects"])
        for m in ARCHITECTURES[:3]
    }
    delta=v11-v10
    gain=counts[ARCHITECTURES[2]]-counts[ARCHITECTURES[0]]
    if delta<-.01 and gain<=0:
        finding="NEGATIVE: V11 loses >0.01 AUPRC vs V10 without increasing the count of reproducible individual computational effects."
    elif delta>=-.01 and gain>0:
        finding="PROMISING BUT INSUFFICIENT: V11 meets the preregistered AUPRC margin and increases observable-effect candidates; this does not validate biological meaning or new-hospital transfer."
    else:
        finding="INCONCLUSIVE OR UNFAVORABLE: V11 does not simultaneously meet both preregistered predictive and interpretability goals."
    lines=[
        "# ConceptFAN V11 — one-shot new-cohort scientific result",
        "",
        "**Population:** 4000 different PhysioNet/CinC 2012 Set B "
        "ICU records from the same historical MIMIC-II source. "
        "This is NOT an independent hospital validation. "
        "Set B was not used in model training, selection or tuning.",
        "",
        "## Frozen evaluation, original 36-hour observations",
        "",
        "| Independently fitted method | Model seeds | AUPRC ↑ | AUROC ↑ | Brier (uncalibrated) ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ARCHITECTURES:
        r=row(name)
        lines.append(
            f"| \`{name}\` | {int(r.independent_model_initializations)} "
            f"| {float(r.mean_auprc):.4f} | {float(r.mean_auroc):.4f} "
            f"| {float(r.mean_brier_uncalibrated):.4f} |"
        )
    lines.extend([
        "",
        f"**Observed Set B mortality prevalence:** {float(row(ARCHITECTURES[0]).outcome_prevalence):.2%}.",
        f"**V11 grouped − V10 original AUPRC:** {delta:+.4f}; "
        f"**V11 grouped − Transformer AUPRC:** {v11-transformer:+.4f}.",
        "",
        "## Frozen signed-effect audit on NEW Set B records",
        "",
        "| Concept learning method | Individual candidate effects | Group-only candidates | Abstained groups |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ARCHITECTURES[:3]:
        j=effects[name]
        lines.append(
            f"| \`{name}\` | {int(j['n_candidate_individual_effects'])} "
            f"| {int(j['n_candidate_group_only'])} "
            f"| {int(j['n_abstained'])} |"
        )
    lines.extend([
        "",
        "The reported 'candidates' are reproducible **model-internal signed logit effects**, "
        "not verified physiological mechanisms, causes or diagnostic entities. "
        "All three initialization pairs must pass the originally locked "
        "signed-rank Spearman >=0.5 and learned-definition projector overlap >=0.75.",
        "",
        "## Preregistered decision",
        "",
        f"**{finding}**",
        "",
        "## Uncertainty and negative controls",
        "",
        "Bootstrap comparisons are paired by patient, with source model seeds "
        "resampled. They condition on this one MIMIC-II-source cohort; "
        "the three model-pair correlations share fitted models and must not "
        "be treated as independent replication.",
        "",
    ])
    for candidate,comparison in (
        (ARCHITECTURES[2],ARCHITECTURES[0]),
        (ARCHITECTURES[2],"transformer_timeseries"),
    ):
        match=compared.loc[
            compared.scenario.eq(PRIMARY)
            & compared.candidate.eq(candidate)
            & compared.comparator.eq(comparison)
        ]
        if len(match)!=1:
            raise ValueError(f"Missing prespecified original cohort comparison: {candidate}/{comparison}")
        rec=match.iloc[0]
        ci=json.loads(rec.paired_patient_and_model_bootstrap_interval.replace("'","\""))
        if len(ci)!=2 or not np.isfinite(ci).all():
            raise ValueError("Invalid paired resampling interval")
        lines.append(
            f"- \`{candidate}\` minus \`{comparison}\`: "
            f"ΔAUPRC {float(rec.mean_auprc_difference_candidate_minus_comparator):+.4f}, "
            f"descriptive paired patient/model 95% interval "
            f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}]."
        )
    lines.extend([
        "",
        "Primary and secondary claims are restricted to this single frozen evaluation. "
        "Other scenarios (additional MCAR and time/channel MAR) are available "
        "unchanged in the aggregate CSV; no new model or decision threshold "
        "was selected by their outcomes.",
        "",
        "**Not established:** a new theory of latent causal identifiability, "
        "external-hospital validity, verified clinical concept semantics, "
        "or superiority to an author-implemented automatic-concept method "
        "such as MAGNETS (not part of this seven-architecture cohort score).",
        "",
        "**Provenance:** original V10/V11 development receipt hashes and "
        "Set B source SHA256s are retained in \`STATUS.json\`. "
        "No raw patient-level predictions are included in this report.",
    ])
    return "\n".join(lines)+"\n"


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--frozen-aggregates",required=True,type=Path)
    p.add_argument("--out-file",required=True,type=Path)
    a=p.parse_args()
    if a.out_file.exists():
        p.error("Refuse to overwrite an existing one-shot report")
    txt=format_one_shot_scientific_report(a.frozen_aggregates)
    a.out_file.parent.mkdir(parents=True,exist_ok=True)
    a.out_file.write_text(txt,encoding="utf-8")
    print(txt,flush=True)


if __name__=="__main__":
    main()
