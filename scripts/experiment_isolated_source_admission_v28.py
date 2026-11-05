"""V28: isolated measured-source admission under known vs unknown device shift.

All frozen choices: docs/medical/IOI_V28_FROZEN_ISOLATED_SOURCE_PROTOCOL_20261106.md.
Reuses canonical K2 native temporal ConceptFAN and staged oracle-labelled
synthetic concept formation; NO task label updates K2 concept representation.
The extra candidate is NOT automatically discovered or physically certified.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit

from conceptfan_discovery.isolated_source_admission_v28 import (
    SOURCE_PROVENANCE,
    isolated_source_prediction,
    measured_candidate,
    train_source_coefficient,
    two_environment_admission,
)
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_concept_leakage_guard_v27 import fit_concepts_without_outcome_gradients
from scripts.experiment_trained_conceptfan_omission_v25 import (
    EPOCHS, NN_SEEDS, STRESS,
    external_logistic, flatten_measured_inputs, frozen_world,
    make_model, mask_mechanism, metrics, predict_native, sha_state,
)

DATA_SEED = 20261108
TRAIN_MASK_SEED = 20264101
VALID_MASK1_SEED = 20264102
VALID_MASK2_SEED = 20264103
TEST_MASK_SEED = 20264900
MIN_PER_SUBJECT_SOURCE_SUPPORT = 0.5
VALIDATION_MIN_GAIN = 0.002
VALIDATION_MIN_SUPPORTED_FRACTION = 0.75
TRAIN_N, VALID_N, TEST_N = 1800, 1200, 900
VALID_ENVS = 2
TEST_MODES = ("complete", "MCAR20", "MCAR40", "MNAR", "block8")
ACTUAL_WORLDS = ("stable_physical", "nuisance", "seen_instrument_break",
                 "unseen_instrument_break")


def flip_candidate_instrument(x: np.ndarray, enabled: bool) -> np.ndarray:
    output=np.asarray(x,dtype=np.float32).copy()
    if enabled:
        output[:,:,4:6]*=-1.0
    return output


def altered(world: str, section: str) -> bool:
    if world=="seen_instrument_break":
        return section in ("validation2","test")
    if world=="unseen_instrument_break":
        return section=="test"
    return False


def y_for_world(world: str, data: dict) -> np.ndarray:
    return data["y_complete"] if world=="nuisance" else data["y_omitted"]


def additive_state(head, q:np.ndarray)->np.ndarray:
    logit,contrib=head(q)
    reconstructed=head.bias+contrib.sum(axis=1)
    np.testing.assert_allclose(logit,reconstructed,rtol=0,atol=1e-13)
    return logit


def summary_row(rows:list[dict],world:str,method:str,mode:str)->dict:
    entries=[r for r in rows if r["world"]==world
             and r["method"]==method and r["stress"]==mode]
    expected=1 if method in ("raw_logistic","oracle_true_k2","oracle_true_k3") else None
    if not entries or (expected and len(entries)!=expected):
        raise AssertionError("Missing or duplicated comparator in summary")
    return {
        "world":world,"method":method,"stress":mode,
        "number_of_independent_native_fits":len(entries) if expected is None else 0,
        "mean_ap":float(np.mean([r["metrics"]["auprc"] for r in entries])),
        "mean_auc":float(np.mean([r["metrics"]["auroc"] for r in entries])),
        "mean_nll":float(np.mean([r["metrics"]["nll"] for r in entries])),
        "mean_brier":float(np.mean([r["metrics"]["brier"] for r in entries])),
        "mean_source_usage_fraction":float(np.mean(
            [r.get("source_usage_fraction",0) for r in entries])),
        "mean_source_support_fraction":float(np.mean(
            [r.get("source_support_fraction",0) for r in entries])),
    }


def evaluate(out_dir:Path, *, smoke:bool=False)->dict:
    if out_dir.exists():
        raise FileExistsError("Refuse to overwrite complete or partial V28 evidence")
    torch.set_num_threads(2)
    n_train,n_val,n_test=(240,240,240) if smoke else (TRAIN_N,VALID_N,TEST_N)
    if n_val%2:
        raise AssertionError("Two disjoint validation environments must have identical sizes")
    epochs=2 if smoke else EPOCHS
    seeds=(NN_SEEDS[0],) if smoke else NN_SEEDS
    bootstraps=199 if smoke else 399
    data=frozen_world(seed=DATA_SEED,n_train=n_train,n_cal=n_val,n_test=n_test)
    tr=data["index_train"];va=data["index_cal"];te=data["index_test"]
    va1,va2=va[:n_val//2],va[n_val//2:]
    assert len(set(tr)&set(va))==len(set(tr)&set(te))==len(set(va)&set(te))==0
    X=data["x"];C=data["concept_trajectories"]
    mt=mask_mechanism(X[tr],"MCAR20",seed=TRAIN_MASK_SEED)
    mv1=mask_mechanism(X[va1],"MCAR20",seed=VALID_MASK1_SEED)
    mv2=mask_mechanism(X[va2],"MCAR20",seed=VALID_MASK2_SEED)
    train_candidate,train_support=measured_candidate(X[tr],mt)
    assert len(train_candidate)==n_train
    test_source_mapped={"candidate_name":SOURCE_PROVENANCE["name"],
                       "sensor_names":["sensor_4","sensor_5"],
                       "physical_identity_verified":False}
    all_rows=[]
    all_gates=[]
    all_train=[]
    all_model_hashes=[]
    for world in ACTUAL_WORLDS:
        y=y_for_world(world,data)
        external=external_logistic(X[tr],mt,y[tr])
        oracle_heads={}
        for k in (2,3):
            oracle_heads[k]=refit_linear_concept_head(C[tr,:,:k].mean(axis=1),y[tr])
        for stress in TEST_MODES:
            x_test=flip_candidate_instrument(X[te],altered(world,"test"))
            m_test=mask_mechanism(x_test,stress,seed=TEST_MASK_SEED+STRESS.index(stress))
            raw_prediction=external.predict_proba(
                flatten_measured_inputs(x_test,m_test))[:,1]
            all_rows.append({"world":world,"seed":None,"method":"raw_logistic",
                             "stress":stress,"metrics":metrics(y[te],raw_prediction)})
            for k in (2,3):
                oracle_prediction=expit(additive_state(
                    oracle_heads[k],C[te,:,:k].mean(axis=1)))
                all_rows.append({"world":world,"seed":None,
                                 "method":f"oracle_true_k{k}",
                                 "stress":stress,
                                 "metrics":metrics(y[te],oracle_prediction),
                                 "oracle_only_not_deployable":True})
    # Trained original concept path is *shared across ALL task labels and worlds*.
    for seed in seeds:
        original=make_model("native",2,seed)
        stage=fit_concepts_without_outcome_gradients(
            original,X[tr],mt,C[tr],seed=seed,epochs=epochs)
        h=sha_state(original)
        all_train.append({"seed":seed,"stage":"concept_only",
                          "native_sha256":h,**stage})
        q_train=predict_native(original,X[tr],mt,C[tr])
        train_baseline_concepts=q_train["concept_summary"]
        if q_train["max_additive_error"]>2e-5:
            raise AssertionError("Native stage-one concept pathway broke additive structure")
        valid_prediction_cache={}
        # The two named VALID environments are completely disjoint patients.
        # Reuse exact same model to prevent candidate-label gradients from
        # modifying the original concept definitions.
        for world in ACTUAL_WORLDS:
            y=y_for_world(world,data)
            base=refit_linear_concept_head(train_baseline_concepts,y[tr])
            frozen_training_logit=additive_state(base,train_baseline_concepts)
            coefficient=train_source_coefficient(frozen_training_logit,train_candidate,y[tr])
            trained_augmented=refit_linear_concept_head(
                np.column_stack((train_baseline_concepts,train_candidate)),y[tr])
            old_w=np.asarray(base.weights,dtype=np.float64).copy()
            v_receipts=[]
            for pos,(iv,mv,section) in enumerate(((va1,mv1,"validation1"),
                                                   (va2,mv2,"validation2"))):
                x_val=flip_candidate_instrument(X[iv],altered(world,section))
                q_val=predict_native(original,x_val,mv,C[iv])
                q_source,support=measured_candidate(x_val,mv)
                v_receipts.append((section,y[iv],
                                   additive_state(base,q_val["concept_summary"]),
                                   q_source,support))
            gate=two_environment_admission(
                coefficient, environments=v_receipts,
                minimum_gain=VALIDATION_MIN_GAIN,
                minimum_supported_fraction=VALIDATION_MIN_SUPPORTED_FRACTION,
                per_subject_min_source_support=MIN_PER_SUBJECT_SOURCE_SUPPORT,
                bootstrap_replicates=bootstraps,
                seed=20261108+seed,
            )
            all_gates.append({"world":world,"seed":seed,**gate.as_dict()})
            all_train.append({"world":world,"seed":seed,"stage":"task_risk_heads",
                              "train_labels":n_train,"gate_label_subjects":n_val,
                              "frozen_k2_bias":base.bias,
                              "frozen_k2_weights":list(base.weights),
                              "new_candidate_gamma":coefficient,
                              "risk_full_refit_bias":trained_augmented.bias,
                              "risk_full_refit_weights":list(trained_augmented.weights)})
            # Check that candidate fitting (even rejected) did not touch
            # original concept pathway or K2 decision weights.
            if sha_state(original)!=h or not np.array_equal(np.asarray(base.weights),old_w):
                raise AssertionError("Original concept definitions or baseline coefficients changed")
            if world in ("stable_physical","unseen_instrument_break"):
                valid_prediction_cache[world]=gate.as_dict()
                if (world=="unseen_instrument_break" and
                    valid_prediction_cache["stable_physical"]!=gate.as_dict()):
                    raise AssertionError("Unseen TEST-only shift illegally changed VALID admission")
            x_ref=flip_candidate_instrument(X[te],False)
            for stress in TEST_MODES:
                x_test=flip_candidate_instrument(X[te],altered(world,"test"))
                m_test=mask_mechanism(x_test,stress,
                                      seed=TEST_MASK_SEED+STRESS.index(stress))
                test_native=predict_native(original,x_test,m_test,C[te])
                q_test=test_native["concept_summary"]
                qsrc,src_support=measured_candidate(x_test,m_test)
                old_logit=additive_state(base,q_test)
                outputs=isolated_source_prediction(
                    old_logit,qsrc,src_support,coefficient=coefficient,
                    globally_admitted=gate.admitted,
                    minimum_subject_support=MIN_PER_SUBJECT_SOURCE_SUPPORT)
                full_refit_logit=additive_state(
                    trained_augmented,np.column_stack((q_test,qsrc)))
                if test_native["max_additive_error"]>2e-5:
                    raise AssertionError("Native trained model additive decomposition broken")
                for name,logit in (
                    ("frozen_k2",outputs["base_logit"]),
                    ("forced_isolated_source",outputs["forced_logit"]),
                    ("qualified_isolated_source",outputs["gated_logit"]),
                    ("standard_k3_linear_refit",full_refit_logit),
                ):
                    all_rows.append({
                        "world":world,"seed":seed,"method":name,
                        "stress":stress,"metrics":metrics(y[te],expit(logit)),
                        "source_supported_fraction":outputs["source_support_fraction"],
                        "source_usage_fraction":(
                            outputs["source_usage_fraction"] if name=="qualified_isolated_source"
                            else outputs["source_support_fraction"] if name=="forced_isolated_source"
                            else 0.0),
                        "gate_admitted":gate.admitted,
                        "concept_MSE":test_native["concept_mse"],
                        "unchanged_k2_head_in_isolated_path":name!="standard_k3_linear_refit",
                        "native_max_additive_decomposition_error":test_native["max_additive_error"],
                    })
            if sha_state(original)!=h:
                raise AssertionError("TEST changed original concept tensor parameters")
        all_model_hashes.append({"seed":seed,"native_k2_sha256":h,
                                 "shared_across_outcome_worlds":True})
        print(json.dumps({"event":"V28_SEED","seed":seed,"native_hash":h,
                          "gates":[{"world":g["world"],"admitted":g["admitted"],
                            "nll_gains":[e["mean_paired_logloss_gain"]
                                          for e in g["environments"]]}
                                   for g in all_gates if g["seed"]==seed]}),flush=True)
    # A stable instrument and stable physical source are observationally
    # indistinguishable; do NOT invent new independent training/predictions.
    for records in (all_rows,all_gates,all_train):
        for r in list(records):
            if r.get("world")=="stable_physical":
                records.append({**r,"world":"stable_instrument_interpretation",
                                "reused_identical_observational_evidence":True})
    summary=[]
    for world in ACTUAL_WORLDS+("stable_instrument_interpretation",):
        for stress in TEST_MODES:
            for method in ("raw_logistic","oracle_true_k2","oracle_true_k3",
                           "frozen_k2","forced_isolated_source",
                           "qualified_isolated_source","standard_k3_linear_refit"):
                s=summary_row(all_rows,world,method,stress)
                if method in ("raw_logistic","oracle_true_k2","oracle_true_k3"):
                    assert s["number_of_independent_native_fits"]==0
                else:
                    assert s["number_of_independent_native_fits"]==len(seeds)
                summary.append(s)
    report={
        "status":"V28_SMOKE_ONLY" if smoke else "V28_EXECUTED_NATIVE_CONCEPT_ISOLATED_SOURCE_GATE",
        "protocol":"docs/medical/IOI_V28_FROZEN_ISOLATED_SOURCE_PROTOCOL_20261106.md",
        "generator_seed":DATA_SEED,
        "new_draw_from_same_prior_generator_family":True,
        "train_val_test_disjoint":True,
        "gate_test_labels_used":False,
        "physical_causal_concept_semantics_certified":False,
        "clinically_validated":False,
        "baseline_k2_concept_definitions_frozen":True,
        "unseen_shift_impossible_to_admit_safely_without_new_provenance":True,
        "source_provenance":test_source_mapped,
        "artifact_stable_duplicates_existing_data_not_independent":True,
        "n_unique_k2_native_fit":len(seeds),
        "samples":{"train":n_train,"validation_total":n_val,
                   "validation_per_environment":n_val//2,"test":n_test},
        "nn_seeds":list(seeds),"training_epochs":epochs,
        "stress_modes":list(TEST_MODES),
        "n_validation_label_subjects_extra_for_admission":n_val,
        "source_coefficient_training_only":True,
        "original_concept_activation_source_hashes":all_model_hashes,
        "dataset_sha256":sha256(np.ascontiguousarray(X).tobytes()+
                                np.ascontiguousarray(C).tobytes()+
                                np.ascontiguousarray(data["y_complete"]).tobytes()+
                                np.ascontiguousarray(data["y_omitted"]).tobytes()).hexdigest(),
        "gates":all_gates,"training":all_train,"rows":all_rows,
        "summary":summary,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"STATUS.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (out_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out_dir/"ADMISSION.json").write_text(json.dumps(all_gates,indent=2,allow_nan=False))
    (out_dir/"TRAINING.json").write_text(json.dumps(all_train,indent=2,allow_nan=False))
    print(json.dumps({
        "event":"V28_DONE","status":report["status"],
        "n_unique_native_fits":report["n_unique_k2_native_fit"],
        "n_rows":len(all_rows),"n_summary":len(summary),
        "primary":[s for s in summary
                   if s["stress"]=="MCAR20"
                   and s["world"] not in ("stable_instrument_interpretation",)],
        "gates":[{"world":g["world"],"seed":g["seed"],
                  "admitted":g["admitted"],
                  "validation_gains":[e["mean_paired_logloss_gain"]
                                       for e in g["environments"]]}
                 for g in all_gates
                 if g["world"]!="stable_instrument_interpretation"],
    }),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    evaluate(args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
