"""New-seed V15 synthetic acceptance / failure audit, NOT clinical validation.

No patient input, pretrained checkpoints, or true latent variables are passed
to the estimators. Positive-source oracle scores are evaluated after fitting.
All rejects are reported, never silently discarded from denominators.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from .lag_contrast_v15 import fit_lagged_contrast,project_lag_concepts
from .lag_synthetic_v15 import make_lag_scm
from .crossview_v15 import fit_crossview_environment
from .crossview_synthetic_v15 import make_crossview_scm
from .environment_contrast_v14 import fit_environment_contrast

REGIMES=(3,5,7)
NEW_GENERATOR_SEEDS=(20261105,20261205)
LAG_MODES=("latent_shift","white_sensor_shift","colored_sensor_shift","no_shift")
VIEW_MODES=("latent_shift","independent_colored_noise","shared_colored_noise","no_shift")

def _oracle_match(estimated,truth):
    k=truth.shape[-1]
    m=np.array([[abs(spearmanr(estimated[:,i],truth[:,j]).statistic)
        for j in range(k)] for i in range(estimated.shape[1])])
    if not np.isfinite(m).all():
        raise ValueError("Invalid oracle validation correlation")
    a,b=linear_sum_assignment(-m)
    return float(m[a,b].mean()),len(a)

def run(output:Path):
    if output.exists():
        raise FileExistsError("Refuse to overwrite empirical evidence")
    records=[]
    for generator_seed in NEW_GENERATOR_SEEDS:
        for k in REGIMES:
            for mode in LAG_MODES:
                s=make_lag_scm(k=k,seed=generator_seed+100*k,
                               per_env=950,steps=12,mode=mode)
                for method in ("v14_contemporaneous","v15_lagged"):
                    try:
                        if method=="v14_contemporaneous":
                            c=fit_environment_contrast(s.x,s.env,
                                seed=117+k,permutations=99,bootstraps=24)
                            # V14 projects average of last four steps.
                            proj=s.x[:,-4:].mean(axis=1) @ c.demixing
                            rank=c.selected_rank
                        else:
                            c=fit_lagged_contrast(s.x,s.env,
                                sensor_temporal_white_documented=True,
                                seed=117+k,permutations=99,bootstraps=24)
                            proj=project_lag_concepts(s.x,c)
                            rank=c.source_rank
                        # Oracle used for evaluation only, never for fit.
                        score,matched=_oracle_match(
                            proj,s.true_sources[:,-4:].mean(axis=1))
                        accepted=True;reason=""
                    except ValueError as exc:
                        accepted=False;reason=str(exc);rank=0;score=None;matched=0
                    records.append({
                        "generator_seed":generator_seed,"k_true_oracle_only":k,
                        "generator_mode":mode,"estimator":method,
                        "accepted":accepted,"certified_count":rank,
                        "oracle_matched_abs_spearman_eval_only":score,
                        "oracle_slots_matched_eval_only":matched,
                        "failure_reason":reason,
                        "source_variance_actually_changes":mode=="latent_shift",
                        "measurement_noise_is_temporally_white":mode in (
                            "latent_shift","white_sensor_shift","no_shift"),
                        "fit_uses_oracle":False})
                    print(generator_seed,k,mode,method,
                          "ACCEPT" if accepted else "REJECT",rank,flush=True)
            for mode in VIEW_MODES:
                s=make_crossview_scm(k=k,seed=generator_seed+100*k,
                                     per_env=1000,t=12,mode=mode)
                try:
                    c=fit_crossview_environment(s.a,s.b,s.env,
                        independently_noisy_views_documented=True,
                        seed=119+k,permutations=99,bootstraps=24)
                    rank=c.source_rank
                    proj=s.a[:,-4:].mean(axis=1)@c.demixing_a
                    score,matched=_oracle_match(proj,s.latent[:,-4:].mean(axis=1))
                    accepted=True;reason=""
                except ValueError as exc:
                    accepted=False;rank=0;reason=str(exc);score=None;matched=0
                records.append({
                    "generator_seed":generator_seed,"k_true_oracle_only":k,
                    "generator_mode":mode,"estimator":"v15_crossview",
                    "accepted":accepted,"certified_count":rank,
                    "oracle_matched_abs_spearman_eval_only":score,
                    "oracle_slots_matched_eval_only":matched,
                    "failure_reason":reason,
                    "source_variance_actually_changes":mode=="latent_shift",
                    "measurement_noise_is_temporally_white":mode in (
                        "latent_shift","no_shift"),
                    "view_errors_independent":mode!="shared_colored_noise",
                    "fit_uses_oracle":False})
                print(generator_seed,k,mode,"v15_crossview",
                      "ACCEPT" if accepted else "REJECT",rank,flush=True)
    df=pd.DataFrame.from_records(records)
    summary=df.groupby(["estimator","k_true_oracle_only","generator_mode"],dropna=False).agg(
       trials=("accepted","size"),accepted=("accepted","sum"),
       mean_rank_given_accepted=("certified_count",lambda x:float(x[x>0].mean()) if any(x>0) else None),
       mean_matched_oracle_rho_given_accepted=("oracle_matched_abs_spearman_eval_only","mean"),
    ).reset_index()
    receipt={
        "status":"V15_CONDITIONAL_SYNTHETIC_AUDIT_COMPLETE",
        "generator_seeds":list(NEW_GENERATOR_SEEDS),
        "n_environment_labels":3,
        "generator_families":["lagged_AR1","synchronized_two_view_AR1"],
        "denominator_includes_all_rejections":True,
        "true_sources_used_in_estimator":False,
        "sensor_noise_structure_is_external_assumption":True,
        "no_physionet_or_original_PIERE_data_access":True,
        "not_general_clinical_concept_identification":True,
        "n_trials":len(df),
        "summary":summary.where(pd.notna(summary),None).to_dict("records"),
    }
    output.mkdir(parents=True,exist_ok=False)
    df.to_csv(output/"ALL_ACCEPT_AND_ABSTAIN.csv",index=False)
    summary.to_csv(output/"SUMMARY.csv",index=False)
    (output/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
