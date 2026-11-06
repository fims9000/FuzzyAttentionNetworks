"""Run V38 RESEARCH inference on external observed values and masks.

Input .npz is the ONLY patient input, with arrays:
 x:        [N, 16, 12] float32/float64 measured values
 observed: [N, 16, 12] bool mask (False is actually missing)
N>=1, N<=10000. Missing values may contain NaN and are NEVER interpreted
as observations. No simulation, hidden factors, TEST outcomes or retraining.

Example (only trusted, locally produced SOURCE model snapshots):
 PYTHONPATH=src:. python scripts/infer_exogenous_source_v38.py \
   --model-dir V38_EVIDENCE --device-seed 20261231 \
   --nn-seed 11 --outcome-case healthy \
   --input-npz observed_sensor_episodes.npz \
   --output-json research_predictions.json

The output has sensitive per-episode information: secure it appropriately.
This is an EXPLICITLY synthetic research model, not clinical decision support.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import numpy as np

from conceptfan_discovery.inference_v38 import load_v38_inference_model


def inference_npz_to_json(
    model_dir:Path,
    *,
    device_seed:int,
    nn_seed:int,
    outcome_case:str,
    input_npz:Path,
    output_json:Path,
)->dict:
    source=Path(input_npz)
    out=Path(output_json)
    if not source.is_file() or out.exists():
        raise FileExistsError("Missing input NPZ or output already exists; never overwrite")
    # NPZ is a data-only format; no arbitrary pickled objects are loaded.
    with np.load(source,allow_pickle=False) as values:
        if set(values.files)!={"x","observed"}:
            raise ValueError("Input NPZ must contain only x and observed arrays")
        x=values["x"]
        observed=values["observed"]
    model=load_v38_inference_model(
        model_dir,device_seed=device_seed,
        nn_seed=nn_seed,outcome_case=outcome_case,
    )
    results=model.predict(x,observed)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(
        results,indent=2,ensure_ascii=False,allow_nan=False,
    ))
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--device-seed",type=int,required=True)
    p.add_argument("--nn-seed",type=int,required=True)
    p.add_argument("--outcome-case",required=True)
    p.add_argument("--input-npz",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    args=p.parse_args()
    result=inference_npz_to_json(
        args.model_dir,device_seed=args.device_seed,
        nn_seed=args.nn_seed,outcome_case=args.outcome_case,
        input_npz=args.input_npz,output_json=args.output_json,
    )
    reasons={}
    for row in result["episodes"]:
        reasons[row["decision_code"]]=reasons.get(row["decision_code"],0)+1
    print(json.dumps({
        "status":"RESEARCH_INFERENCE_OK",
        "n_episodes":result["n_episodes"],
        "refusal_and_admission_counts":reasons,
        "not_clinically_validated":True,
    },ensure_ascii=False))
