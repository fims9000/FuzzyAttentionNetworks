"""Controlled sanity check only: not publication-grade model evaluation."""
from __future__ import annotations

import json
import numpy as np
from scipy.special import expit
from fan.concept.completeness_audit import audit_oracle_concept_coverage


def synthetic_case(mode, seed=31415, n=4200):
    rng=np.random.default_rng(seed)
    gold=rng.normal(size=(n,3))
    predicted=gold+0.8*rng.normal(size=gold.shape)
    sensor_residual=rng.normal(size=(n,1))
    risk=1.2*gold[:,0]-1.3*gold[:,1]+0.8*gold[:,2]
    if mode == "omitted_predictive_source":
        risk=risk+1.7*sensor_residual[:,0]
    elif mode == "complete_but_noisy_concepts":
        pass
    elif mode == "predictive_measurement_artifact":
        # This is an observational shortcut, not a physiological concept.
        risk=risk+1.7*sensor_residual[:,0]
    else:
        raise ValueError(mode)
    y=rng.binomial(1,expit(risk))
    report=audit_oracle_concept_coverage(
        labels=y,predicted_concepts=predicted,true_concepts=gold,
        residual_candidate=sensor_residual,
        train_indices=np.arange(2800),test_indices=np.arange(2800,n),
        bootstrap_replicates=999,seed=31415)
    result=report.as_dict()
    result["scenario"]=mode
    result["note"]="Synthetic oracle diagnostic only; association does not establish physical meaning"
    return result


if __name__ == "__main__":
    print(json.dumps({m:synthetic_case(m) for m in (
        "complete_but_noisy_concepts", "omitted_predictive_source",
        "predictive_measurement_artifact")},ensure_ascii=False,indent=2))
