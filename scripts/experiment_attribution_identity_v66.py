#!/usr/bin/env python3
"""V66 executable mathematical source identity counterexample, SYNTHETIC only."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np

from conceptfan_discovery.exact_attribution_nonidentifiability_v66 import audit_equivalence


def run(out:Path):
    if out.exists():
        raise FileExistsError("Never overwrite pinned mathematical evidence")
    rows=[]
    for k in (3,18,36):
        rng=np.random.default_rng(20261966+k)
        states=rng.beta(1.5,1.5,size=(1000,k))
        beta=rng.normal(size=k)
        beta[0]+=2.8
        for alpha in (.05,.35,.75):
            r=audit_equivalence(states,beta,-.41,alpha=alpha)
            if r.max_logit_difference>1e-12 or r.bounded_state_violations:
                raise AssertionError("Counterexample failed risk/bound preservation")
            rows.append(r.as_dict())
    final={
        "status":"V66_FORMAL_BOUNDED_CONCEPT_NONIDENTIFIABILITY_EXECUTED",
        "source_type":"DETERMINISTIC_SYNTHETIC_FUZZY_MEMBERSHIP_MATH",
        "n_independent_real_clinical_or_physical_samples_in_this_test":0,
        "theorem":"For A=(1-a)I+a*11^T/K, 0<a<1, CA remains in [0,1], beta'=A^-1 beta, and (CA-.5)^T beta'=(C-.5)^T beta",
        "what_theorem_refutes":"Unique concept semantics from bounded states plus exact signed additive risk alone",
        "what_it_does_not_refute":"Identifiability under additional independently justified physical anchors, ICA assumptions, or restricted source families",
        "stronger_assumptions_can_exclude_cross_concept_mixing":True,
        "no_real_data_or_gas_label_used":True,
        "no_model_accuracy_or_physical_factor_improvement_claimed":True,
        "frozen_protocol":"docs/medical/CONCEPTFAN_V66_FORMAL_ATTRIBUTION_IDENTIFIABILITY_PROTOCOL_20261113.md",
        "rows":rows,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        final,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({
        "event":"V66_FORMAL_COUNTEREXAMPLE_DONE",
        "n_test_cases":len(rows),
        "max_abs_logit_difference":max(x["max_logit_difference"] for x in rows),
        "changed_top1_attribution_by_case":[
           {"K":x["k"],"alpha":x["alpha"],
            "fraction":x["fraction_changed_top_abs_contribution_axis"]}
           for x in rows],
    },ensure_ascii=False,allow_nan=False),flush=True)
    return final

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
