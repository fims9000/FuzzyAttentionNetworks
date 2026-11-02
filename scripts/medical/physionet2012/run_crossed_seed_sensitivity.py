"""Sensitivity checks that respect repeated 10 x 3 random-seed factors.

Does not re-fit any model, never overwrites existing outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from conceptfan_realdata.crossed_seed import exact_cluster_signflip, holm, crossed_seed_bootstrap_ci
from conceptfan_realdata.models import MODEL_ARMS, STABILITY_ANALYSIS_ARMS

METRICS=("AUPRC","AUROC","Brier","NLL")
STABILITY_METRICS=("spearman","top1_agreement","sign_agreement")

def seed_map(runs_root:Path,arm:str)->dict:
    result={}
    for f in sorted((runs_root/arm).glob("run_*/run_manifest.json")):
        manifest=json.loads(f.read_text(encoding="utf-8"))
        if manifest.get("status")!="RUN_COMPLETE":
            raise ValueError("Incomplete frozen run: "+str(f))
        result[f.parent.name]=(manifest["init_seed"],manifest["data_order_seed"])
    if len(result)!=30 or len({v for v in result.values()})!=30:
        raise ValueError(f"Expected 10x3 complete grid for {arm}, found {len(result)}")
    return result

def run_prediction_tests(root:Path,out:Path)->None:
    arms={}
    for arm in MODEL_ARMS:
        seeds=seed_map(root,arm)
        lookup={}
        for name,k in seeds.items():
            path=root/arm/name/"metrics_test.json"
            if not path.is_file():raise FileNotFoundError(path)
            data=json.loads(path.read_text(encoding="utf-8"))
            raw=data["metrics"]["none"]
            lookup[k]={metric:float(raw[metric]) for metric in METRICS}
        arms[arm]=lookup
    keys=sorted(arms[MODEL_ARMS[0]])
    init=sorted({x for x,_ in keys})
    order=sorted({y for _,y in keys})
    if len(init)!=10 or len(order)!=3 or set(keys)!={(x,y) for x in init for y in order}:
        raise ValueError("Expected exact 10x3 crossed seed matrix")
    rows=[]
    for comparator in MODEL_ARMS[1:]:
        for metric in METRICS:
            diff=np.array([[arms[MODEL_ARMS[0]][i,j][metric]-arms[comparator][i,j][metric] for j in order] for i in init])
            by_init=exact_cluster_signflip(diff,axis=0)
            by_order=exact_cluster_signflip(diff,axis=1)
            rows.append({"reference":MODEL_ARMS[0],"comparator":comparator,"metric":metric,
                         "mean_difference":by_init["mean_delta"],
                         "median_difference":by_init["median_per_run_delta"],
                         "p_exact_init_cluster":by_init["two_sided_exact_p"],
                         "p_exact_order_cluster":by_order["two_sided_exact_p"],
                         "n_init_clusters":len(init),"n_order_clusters":len(order),
                         "interpretation":"design sensitivity, tests 10 init clusters or 3 order clusters separately"})
    adj=holm([x["p_exact_init_cluster"] for x in rows])
    for row,value in zip(rows,adj):row["p_exact_init_holm"]=value
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    pd.DataFrame(rows).to_csv(out,index=False)

def run_stability_tests(root:Path,out:Path,repetitions:int)->None:
    source=root/"TABLES"/"episode_pairwise_stability.parquet"
    if not source.is_file():raise FileNotFoundError(source)
    data=pd.read_parquet(source)
    rows=[]
    for arm in STABILITY_ANALYSIS_ARMS:
        seeds=seed_map(root/"runs",arm)
        group=data[(data["model_arm"]==arm)&(data["representation"]=="signed")]
        if group.empty:raise ValueError(f"missing signed rows for {arm}")
        keys=list(group[["run_a","run_b"]].drop_duplicates().itertuples(index=False,name=None))
        patient_ids=np.sort(group.RecordID.unique())
        for metric_idx,metric in enumerate(STABILITY_METRICS):
            pivot=group.pivot(index=["run_a","run_b"],columns="RecordID",values=metric).loc[keys,patient_ids]
            val=pivot.to_numpy(dtype=np.float64)
            for mode_idx,mode in enumerate(("init","order","both")):
                rng=np.random.default_rng(20261102 + 1000*STABILITY_ANALYSIS_ARMS.index(arm) + 50*metric_idx+mode_idx)
                low,high=crossed_seed_bootstrap_ci(val,keys,seeds,repetitions,rng,mode=mode)
                rows.append({"model_arm":arm,"representation":"signed","metric":metric,
                             "mean":float(val.mean()),"ci95_low":low,"ci95_high":high,
                             "bootstrap_mode":mode,"repetitions":repetitions,
                             "model_pairs":len(keys),"patients":len(patient_ids),
                             "n_init_seeds":10,"n_order_seeds":3,
                             "interpretation":"seed-design sensitivity, not independent-cohort inference"})
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    pd.DataFrame(rows).to_csv(out,index=False)

def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--repetitions",type=int,default=2000)
    a=p.parse_args()
    if a.out_dir.exists():p.error("Output directory already exists")
    root=a.artifacts_dir.resolve()
    if not root.is_dir():p.error("Missing frozen release extract")
    # Fail before outputs if required frozen source data are absent.
    for arm in MODEL_ARMS:
        seed_map(root/"runs",arm)
    if not (root/"TABLES"/"episode_pairwise_stability.parquet").exists():
        p.error("Missing frozen per-episode parquet")
    a.out_dir.mkdir(parents=True,exist_ok=False)
    run_prediction_tests(root/"runs",a.out_dir/"raw_paired_seed_sensitivity.csv")
    run_stability_tests(root,a.out_dir/"signed_stability_crossed_seed_ci.csv",a.repetitions)
    print("Complete: 10-init / 3-order design sensitivity. Original evidence unmodified.")

if __name__=="__main__":
    main()
