"""Independent synthetic-dataset replication: 10 initializations x 3 data orders.

Complete exploratory evidence, 435 dependent model pairs per method.
No original PhysioNet or published Med-CircuitBench test examples are read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from conceptfan_realdata.crossed_seed import crossed_seed_bootstrap_ci, exact_cluster_signflip, holm
from conceptfan_realdata.semantic_anchors import fit_semantic_anchor
from scripts.medical.run_semantic_anchor_validation import load_split, train_one, agreement


PER_RUN_METRICS = ("auprc", "auroc", "concept_mae",
                   "concept_dynamics_mae", "mean_effective_concept_count",
                   "top1_concept_diversity", "mean_contribution_variance")
AGREEMENT_METRICS = (
    "spearman_signed", "spearman_excess_episode_alignment",
    "top1_abs", "top1_excess_episode_alignment",
    "sign_agreement", "sign_excess_episode_alignment",
    "cosine_signed",
)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    if a.out_dir.exists():
        p.error("Refusing to overwrite an existing replication")
    config=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if config["evaluation"]["no_test_access"] is not True or config["evaluation"]["no_hyperparameter_selection_using_validation"] is not True:
        p.error("Test access and validation tuning are forbidden")
    torch.set_num_threads(int(config["training"]["cpu_threads"]))
    data, receipt=load_split(config)
    ac=config["anchor"]
    anchor=fit_semantic_anchor(
        data["train"][2],data["train"][1],
        recency_decay=float(ac["recency_decay"]),
        logistic_c=float(ac["logistic_c"]),
        minimum_width=float(ac["minimum_width"]),
    )
    init=[int(i) for i in config["training"]["init_seeds"]]
    order=[int(i) for i in config["training"]["data_order_seeds"]]
    if len(init)!=10 or len(order)!=3 or len(set(init))!=10 or len(set(order))!=3:
        p.error("Expected complete 10 x 3 crossed seed experiment")
    cells=list(itertools.product(init,order))
    ids={f"init_{i}_order_{o}":(i,o) for i,o in cells}
    pairs=list(itertools.combinations(sorted(ids),2))
    methods=list(config["evaluation"]["methods"])
    if set(methods)!={"legacy","anchor_only","anchor_regularized"}:
        p.error("Predefined ablations must all be present")
    outputs={}
    runs=[]
    pairs_report=[]
    for method in methods:
        outputs[method]={}
        for position,(initial,ordering) in enumerate(cells,1):
            identifier=f"init_{initial}_order_{ordering}"
            value=train_one(method,initial,config,anchor,data,order_seed=ordering)
            outputs[method][identifier]=value
            runs.append({"method":method,"run_id":identifier,"init_seed":initial,
                         "data_order_seed":ordering,
                         **{k:float(value[k]) for k in PER_RUN_METRICS},
                         "max_decomposition_error":float(value["max_decomposition_error"])})
            if position%5==0:
                print("completed",method,position,"/30",flush=True)
        for m1,m2 in pairs:
            pair=agreement(outputs[method][m1]["contributions"],
                           outputs[method][m2]["contributions"])
            pairs_report.append({"method":method,"run_a":m1,"run_b":m2,**pair})
    run_frame=pd.DataFrame(runs)
    pair_frame=pd.DataFrame(pairs_report)
    ci_rows=[]
    boot=int(config["evaluation"]["bootstrap_iterations"])
    rngseed=int(config["evaluation"]["bootstrap_seed"])
    for comparator in ("anchor_only","anchor_regularized"):
        for metric_index,metric in enumerate(AGREEMENT_METRICS):
            lookup={}
            for method in methods:
                subset=pair_frame[pair_frame.method==method].set_index(["run_a","run_b"])
                lookup[method]=np.asarray([float(subset.loc[(u,v),metric]) for u,v in pairs])
            delta=lookup[comparator]-lookup["legacy"]
            low,high=crossed_seed_bootstrap_ci(
                delta,pairs,ids,boot,np.random.default_rng(rngseed+metric_index+100*methods.index(comparator)),
                mode="both",
            )
            ci_rows.append({"type":"paired_contribution_agreement","comparator":comparator,
                            "metric":metric,"mean_delta":float(delta.mean()),
                            "ci95_low":low,"ci95_high":high,
                            "statistic":"crossed_init_and_order_cluster_bootstrap",
                            "model_pairs":435,"patients_fixed":receipt["validation_count"],
                            "bootstrap_repetitions":boot})
    infer=[]
    for comparator in ("anchor_only","anchor_regularized"):
        for metric in PER_RUN_METRICS:
            d=np.asarray([[
                outputs[comparator][f"init_{initial}_order_{ordering}"][metric]-
                outputs["legacy"][f"init_{initial}_order_{ordering}"][metric]
                for ordering in order] for initial in init],dtype=np.float64)
            initi=exact_cluster_signflip(d,axis=0)
            orde=exact_cluster_signflip(d,axis=1)
            infer.append({"comparator":comparator,"metric":metric,
                          "mean_delta":float(d.mean()),
                          "p_exact_init_cluster":initi["two_sided_exact_p"],
                          "p_exact_order_cluster":orde["two_sided_exact_p"],
                          "n_initialization_clusters":10,"n_data_order_clusters":3})
    # Holm for all ten non-pre-registered per-run explorations. Two primary
    # hypotheses are reported *without claiming confirmatory test inference*.
    adjusted=holm([row["p_exact_init_cluster"] for row in infer])
    for row,value in zip(infer,adjusted):
        row["p_exact_init_holm_all_10"]=value

    a.out_dir.mkdir(parents=True,exist_ok=False)
    run_frame.to_csv(a.out_dir/"replication_per_run.csv",index=False)
    pair_frame.to_csv(a.out_dir/"replication_435_pairs_per_method.csv",index=False)
    pd.DataFrame(ci_rows).to_csv(a.out_dir/"crossed_seed_stability_deltas.csv",index=False)
    pd.DataFrame(infer).to_csv(a.out_dir/"paired_predictive_seed_tests.csv",index=False)
    manifest={
        "protocol":config["protocol"],
        "config":config,
        "data_receipt":receipt,
        "train_only_anchor":anchor.to_dict(),
        "methods":methods,
        "train_seed_design":"10 independent initialization seeds x 3 shared data-order seeds",
        "statistical_inference":"exploratory independent generator replication; 435 pairs dependent; bootstrap resamples seed factors",
        "scope":"train/validation only; test identifiers sealed; NO clinical inference",
        "no_test_access":True,
        "run_count":len(run_frame),"pairs_per_method":435,
        "result_hashes":{},
    }
    for path in sorted(a.out_dir.glob("*.csv")):
        manifest["result_hashes"][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.out_dir/"provenance.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    summary={
        "runs":len(run_frame),
        "methods":methods,
        "validation_auprc_by_method":run_frame.groupby("method").auprc.mean().to_dict(),
        "mean_signed_spearman_by_method":pair_frame.groupby("method").spearman_signed.mean().to_dict(),
        "mean_top1_by_method":pair_frame.groupby("method").top1_abs.mean().to_dict(),
        "mean_concept_mae_by_method":run_frame.groupby("method").concept_mae.mean().to_dict(),
        "zero_test_evaluations":True,
        "not_a_confirmatory_clinical_result":True,
    }
    (a.out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
