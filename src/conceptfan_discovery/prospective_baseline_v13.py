"""Third independent synthetic seed comparison with standard sklearn LogisticRegression.

Post-V13 and prior-art novelty audit specified this baseline before generating
seed 20270104. BOTH predictive heads share the exact same TRAIN-estimated
randomized-action source certificate, same 800-subject fitting patients and
same 300 untouched validation subjects. This is NOT clinical validation.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from .intervention_synthetic_v13 import make_intervention_scm
from .instrumented_v13 import (
    InterventionProtocol, identify_action_anchored_concepts,
    fit_instrumented_bottleneck,
)
FACTOR_COUNTS=(3,5,7)
SEEDS=(11,23,37)
GENERATOR_SEED=20270104

def last_four_sensor_projection(x,dual):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=3 or a.shape[1]<4 or a.shape[2]!=dual.shape[1] or not np.isfinite(a).all():
        raise ValueError("Observed input and fixed train-only sensor source signature required")
    return (a[:,-4:,:]@dual.T).mean(axis=1)

def run(out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Never overwrite previously scored new seed")
    rows,pairs=[],[]
    for k in FACTOR_COUNTS:
        data=make_intervention_scm(n=1300,k=k,d=k+3,
            seed=GENERATOR_SEED+100*k,history_steps=12,horizon=2,action_mode="randomized")
        x=data.observations[:,:12].astype(np.float32)
        u=data.actions[:,:11].astype(np.float32)
        y=data.labels.astype(np.float32)
        ids=np.random.default_rng(20261120+k).permutation(len(x))
        train,validation=ids[:1000],ids[1000:]
        protocol=InterventionProtocol(True,True,True,True,
            tuple(f"randomized_target_{j}" for j in range(k)))
        fits={name:{} for name in ("native_concept_head","ordinary_logistic_on_same_action_anchors")}
        for seed in SEEDS:
            selected=np.random.default_rng(seed+1000*k).choice(train,800,replace=False)
            cert=identify_action_anchored_concepts(
                x[selected],u[selected],protocol,seed=seed+3*k,
                permutations=199,bootstraps=48)
            train_features=last_four_sensor_projection(x[selected],cert.dual)
            val_features=last_four_sensor_projection(x[validation],cert.dual)
            baseline=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,random_state=seed)
            baseline.fit(train_features,y[selected])
            base_score=baseline.predict_proba(val_features)[:,1]
            base_terms=val_features*baseline.coef_[0][None,:]
            base_error=np.max(np.abs(baseline.decision_function(val_features)
                                    -base_terms.sum(axis=-1)-baseline.intercept_[0]))
            assert base_error<1e-10
            native,_=fit_instrumented_bottleneck(
                cert,x[selected],y[selected],seed=seed,epochs=85)
            with torch.no_grad():
                output=native(torch.from_numpy(x[validation]))
                native_score=output["probabilities"].numpy()
                native_terms=output["contributions"].numpy()
                native_beta=(3*torch.tanh(native.raw_beta)).numpy()
                native_error=float((output["logits"]-
                    output["contributions"].sum(-1)-native.bias).abs().max())
            assert native_error<1e-6
            for method,score,terms,weights,error in [
                ("ordinary_logistic_on_same_action_anchors",base_score,base_terms,baseline.coef_[0],base_error),
                ("native_concept_head",native_score,native_terms,native_beta,native_error),
            ]:
                fits[method][seed]=(terms,weights,cert.signature)
                rows.append({
                    "generator_seed":GENERATOR_SEED,"k":k,"seed":seed,"method":method,
                    "n_train":len(selected),"n_eval":len(validation),
                    "auprc":float(average_precision_score(y[validation],score)),
                    "auroc":float(roc_auc_score(y[validation],score)),
                    "brier":float(brier_score_loss(y[validation],score)),
                    "max_decomposition_error":float(error),
                    "shared_train_only_intervention_certificate":True,
                    "oracle_used_to_fit_model":False,
                })
            print("K",k,"seed",seed,"baseline",round(rows[-2]["auprc"],5),
                  "native",round(rows[-1]["auprc"],5),flush=True)
        for name,methods in fits.items():
            for a,b in itertools.combinations(SEEDS,2):
                ea,wa,sa=methods[a]
                eb,wb,sb=methods[b]
                for j in range(k):
                    pairs.append({
                        "k":k,"method":name,"seed_a":a,"seed_b":b,"slot":j,
                        "signed_effect_spearman":float(spearmanr(ea[:,j],eb[:,j]).statistic),
                        "signature_cosine":float(np.dot(sa[:,j],sb[:,j])),
                        "both_weights_nonzero":bool(min(abs(wa[j]),abs(wb[j]))>.08),
                        "same_risk_weight_sign":bool(np.sign(wa[j])==np.sign(wb[j])),
                    })
    frame=pd.DataFrame(rows)
    pair_frame=pd.DataFrame(pairs)
    signed=pair_frame.groupby(["k","method","slot"]).agg(
        min_signed_spearman=("signed_effect_spearman","min"),
        min_signature_cosine=("signature_cosine","min"),
        all_weights_nonzero=("both_weights_nonzero","all"),
        all_signs_agree=("same_risk_weight_sign","all"),
    ).reset_index()
    signed["strict_stable_effect"]=(signed.min_signed_spearman>.5)&(
        signed.min_signature_cosine>.9)&signed.all_weights_nonzero&signed.all_signs_agree
    summary=frame.groupby(["k","method"]).agg(
        mean_auprc=("auprc","mean"),sd_auprc=("auprc","std"),
        mean_auroc=("auroc","mean"),mean_brier=("brier","mean"),
    ).reset_index().merge(
        signed.groupby(["k","method"]).strict_stable_effect.sum().rename("strict_signed_effects").reset_index(),
        on=["k","method"],
    )
    assert all(frame.max_decomposition_error<1e-6)
    receipt={
        "status":"POST_V13_THIRD_SEED_ORDINARY_LOGISTIC_COMPARATOR_COMPLETE",
        "generator_seed":GENERATOR_SEED,
        "n_original_train":1000,"n_new_validation":300,
        "source_train_only":True,
        "same_source_certificates_for_both_methods":True,
        "baseline_is_existing_sklearn_logistic_regression":True,
        "no_external_published_crl_baseline_reproduced":True,
        "not_a_clinical_or_out_of_family_external_validation":True,
        "limitations":[
            "Same linearly mixed, fully observed and separately randomized intervention generator",
            "Source identity supplied by known action-target labels",
            "No generator selection on earlier observed synthetic validation outcomes",
            "Only three model seeds; descriptive statistics",
        ],
        "summary":summary.to_dict("records"),
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out_dir/"PER_SEED.csv",index=False)
    pair_frame.to_csv(out_dir/"PAIRWISE.csv",index=False)
    signed.to_csv(out_dir/"SIGNED_SCREEN.csv",index=False)
    summary.to_csv(out_dir/"SUMMARY.csv",index=False)
    (out_dir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print(summary.to_string(index=False),flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    run(p.parse_args().out_dir)

if __name__=="__main__":
    main()
