"""V30 trained canonical ConceptFAN on nonlinear switching/shock source law.

The known K2 neural concept path sees ONLY sensor0:4, never candidate4:5,
external anchor6, or any Y gradient. Original K2 coefficients are frozen.
Comparators: V28 source, V29 self-agreement, V30 independent-anchor rejection
and repair, raw 7-sensor logistic, privileged oracle K2/K3 ceilings.

An externally calibrated anchor is an ASSUMPTION of this new synthetic law.
No claim of automatic physical or clinical concept identification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fan.concept.temporal import MultiSetAdditiveTemporalConceptFANModel
from conceptfan_discovery.nonlinear_source_bench_v30 import (
    STRESS,TRAIN,VALID,TEST,SEED,
    nonlinear_world,mask_view,instrument_fault,raw_observed_features,
)
from conceptfan_discovery.trusted_anchor_repair_v30 import (
    fit_anchor_policy,anchored_predictions,candidate_and_anchor,
)
from conceptfan_discovery.sensor_pair_provenance_v29 import (
    fit_unlabelled_pair_guard,pair_is_credible,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,two_environment_admission,
)
from scripts.experiment_frozen_concept_head_refit_v26 import (
    refit_linear_concept_head,
)
from scripts.experiment_concept_leakage_guard_v27 import (
    fit_concepts_without_outcome_gradients,
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    NN_SEEDS,EPOCHS,metrics,predict_native,sha_state,
)

M_TRAIN=20266101
M_VALID_1=20266102
M_VALID_2=20266103
M_TEST=20266900
FAULTS=("stable","single_pair_sign_flip","single_pair_bias",
        "both_pair_sign_flip","anchor_only_sign_flip",
        "all_three_sign_flip","nuisance")
METHODS=("frozen_k2","v28_pair_candidate","v29_pair_reject",
         "v30_anchor_reject","v30_anchor_repair",
         "raw_logistic","oracle_true_k2","oracle_true_k3")
SOURCE_METHODS=METHODS[:5]
SINGLE_BASELINES=METHODS[5:]
TEST_MODES=("MCAR20","MCAR40","MNAR","block8")


def make_firewalled_k2(seed:int)->MultiSetAdditiveTemporalConceptFANModel:
    """Model's entire input is restricted to two named K2 sensor pairs."""
    torch.manual_seed(seed)
    return MultiSetAdditiveTemporalConceptFANModel(
        input_dim=8,sequence_length=16,latent_dim=24,n_concepts=2,
        n_memberships=3,membership="sigmoid",oracle=False,
        temporal_mode="attention",dropout=0.,
        encoder_layers=1,encoder_heads=2,encoder_ffn=48,
        alpha_mode="no_alpha",positive_decision_weights=False,
    )


def prefit_raw_baseline(values,observed,labels):
    pipe=make_pipeline(StandardScaler(),
                       LogisticRegression(C=1.,max_iter=1000))
    pipe.fit(raw_observed_features(values,observed),labels)
    return pipe


def additive_logit(head:object,concepts:np.ndarray)->np.ndarray:
    z,contributions=head(concepts)
    np.testing.assert_allclose(
        z,head.bias+contributions.sum(axis=1),atol=1e-12,rtol=0.)
    return z


def evaluate(out_dir:Path,*,smoke:bool=False)->dict:
    if out_dir.exists():
        raise FileExistsError("V30 research evidence is immutable; use a new output directory")
    torch.set_num_threads(2)
    ntr,nva,nte=(240,240,240) if smoke else (TRAIN,VALID,TEST)
    epochs=2 if smoke else EPOCHS
    seeds=(11,) if smoke else NN_SEEDS
    bootstraps=199 if smoke else 399
    data=nonlinear_world(seed=SEED,n_train=ntr,n_valid=nva,n_test=nte)
    x,c=data["x"],data["true_concept_trajectories"]
    tr,v1,v2,te=(data["train"],data["valid_env1"],
                  data["valid_env2"],data["test"])
    train_m=mask_view(x[tr],"MCAR20",seed=M_TRAIN)
    val1_m=mask_view(x[v1],"MCAR20",seed=M_VALID_1)
    val2_m=mask_view(x[v2],"MCAR20",seed=M_VALID_2)
    pair_policy=fit_unlabelled_pair_guard(x[tr],train_m)
    anchor_policy=fit_anchor_policy(x[tr],train_m,min_support=.5,quantile=.99)
    train_anchor=candidate_and_anchor(x[tr],train_m)
    valid_anc1=candidate_and_anchor(x[v1],val1_m)
    valid_anc2=candidate_and_anchor(x[v2],val2_m)
    if train_anchor["pair_support"].mean()<.65:
        raise AssertionError("TRAIN source observation support unexpectedly low")
    y_pred=data["labels_predictive"]
    y_nuis=data["labels_nuisance"]
    results,gates,training=[],[],[]
    reference_first4={}
    first4_mask_cache={
        mode:mask_view(x[te],mode,seed=M_TEST+STRESS.index(mode))[:,:,:4]
        for mode in TEST_MODES
    }
    # Preserve sample identities across physical worlds and all sensor-fault scenarios.
    all_eval_masks={
        (scenario,mode):mask_view(instrument_fault(x[te],scenario),mode,
                                  seed=M_TEST+STRESS.index(mode))
        for scenario in FAULTS for mode in TEST_MODES
    }
    for seed in seeds:
        core=make_firewalled_k2(seed)
        stage=fit_concepts_without_outcome_gradients(
            core,x[tr,:,:4],train_m[:,:,:4],c[tr,:,:2],
            seed=seed,epochs=epochs,
        )
        native_sha=sha_state(core)
        # Verify the K2 parameter/state path shares exactly the same learned
        # representations and never consumes candidate or anchor channels.
        qtrain=predict_native(
            core,x[tr,:,:4],train_m[:,:,:4],c[tr,:,:2])
        qv1=predict_native(core,x[v1,:,:4],val1_m[:,:,:4],c[v1,:,:2])
        qv2=predict_native(core,x[v2,:,:4],val2_m[:,:,:4],c[v2,:,:2])
        qtest={}
        for stress in TEST_MODES:
            m=first4_mask_cache[stress]
            qtest[stress]=predict_native(core,x[te,:,:4],m,c[te,:,:2])
        training.append({"seed":seed,"native_k2_source_sha256":native_sha,
                         "stage_one_label_free":True,
                         "k2_input_dimensions":8,
                         "original_k2_sensor_channels":[0,1,2,3],**stage})
        by_law={}
        for law,y in (("predictive",y_pred),("nuisance",y_nuis)):
            head=refit_linear_concept_head(qtrain["concept_summary"],y[tr])
            original_head=(head.bias,head.weights)
            original_train_logit=additive_logit(head,qtrain["concept_summary"])
            gamma=train_source_coefficient(original_train_logit,
                                           train_anchor["pair_concept"],y[tr])
            receipt=two_environment_admission(
                gamma,
                environments=(
                    ("VALID1",y[v1],additive_logit(head,qv1["concept_summary"]),
                     valid_anc1["pair_concept"],valid_anc1["pair_support"]),
                    ("VALID2",y[v2],additive_logit(head,qv2["concept_summary"]),
                     valid_anc2["pair_concept"],valid_anc2["pair_support"]),
                ),
                minimum_gain=.002,minimum_supported_fraction=.75,
                per_subject_min_source_support=.5,
                bootstrap_replicates=bootstraps,seed=20261110+seed,
            )
            gates.append({"seed":seed,"law":law,"gamma":gamma,
                          "source_gate":receipt.as_dict(),
                          "used_test_labels":False})
            by_law[law]=(head,gamma,receipt.admitted)
            if original_head!=(head.bias,head.weights):
                raise AssertionError("Original K2 coefficients were mutated")
        for scenario in FAULTS:
            law="nuisance" if scenario=="nuisance" else "predictive"
            y=y_nuis if law=="nuisance" else y_pred
            head,gamma,admitted=by_law[law]
            for stress in TEST_MODES:
                observed=instruments=instrument_fault(x[te],scenario)
                m=all_eval_masks[(scenario,stress)]
                # Test K2 NEVER receives sensor4/5/6, even in a fault world.
                if not np.array_equal(observed[:,:,:4],x[te,:,:4]):
                    raise AssertionError("Sensor intervention illegally altered the original K2 inputs")
                q_known=qtest[stress]["concept_summary"]
                base=additive_logit(head,q_known)
                evaluated=anchored_predictions(
                    base,observed,m,gamma=gamma,
                    validated_global_source=admitted,
                    pair_policy=pair_policy,anchor_policy=anchor_policy)
                if sha_state(core)!=native_sha:
                    raise AssertionError("A test-world source changed frozen named K2 representations")
                for method in SOURCE_METHODS:
                    logits=evaluated["logits"][method]
                    used=(evaluated["decisions"][method].mean()
                          if method in evaluated["decisions"] else 0.)
                    signed=(evaluated["contributions"].get(method,
                            np.zeros_like(base)))
                    np.testing.assert_allclose(logits,base+signed,
                                               atol=1e-13,rtol=0.)
                    results.append({
                        "scenario":scenario,"stress":stress,"seed":seed,
                        "method":method,"metrics":metrics(y[te],expit(logits)),
                        "source_usage_fraction":float(used),
                        "repair_fraction":(
                            float(evaluated["decisions"]["v30_replaced_by_anchor"].mean())
                            if method=="v30_anchor_repair" else 0.),
                        "pair_agreement_fraction":evaluated["summary"]["internal_pair_agreement_fraction"],
                        "anchor_agreement_fraction":evaluated["summary"]["external_anchor_agreement_fraction"],
                        "anchor_support_fraction":evaluated["summary"]["anchor_support_fraction"],
                        "source_globally_admitted":bool(admitted),
                        "max_source_additive_error":float(np.max(np.abs(logits-(base+signed)))),
                        "known_concept_MSE":qtest[stress]["concept_mse"],
                    })
            print(json.dumps({"event":"V30_MODEL", "seed":seed,
                    "known_k2_sha256":native_sha,
                    "gates":[{"law":g["law"],"admitted":g["source_gate"]["admitted"],
                    "environment_nll_gains":[e["mean_paired_logloss_gain"]
                         for e in g["source_gate"]["environments"]]}
                         for g in gates if g["seed"]==seed]},ensure_ascii=False),flush=True)
    # Unique single raw/oracle TRAIN fit per outcome law; do NOT falsely
    # count as 3 independently fitted raw/oracle baselines.
    external={}
    oracle={}
    for law,y in (("predictive",y_pred),("nuisance",y_nuis)):
        external[law]=prefit_raw_baseline(x[tr],train_m,y[tr])
        oracle[law]={
            k:refit_linear_concept_head(c[tr,:,:k].mean(axis=1),y[tr])
            for k in (2,3)
        }
    for scenario in FAULTS:
        law="nuisance" if scenario=="nuisance" else "predictive"
        y=y_nuis if law=="nuisance" else y_pred
        test_x=instrument_fault(x[te],scenario)
        for stress in TEST_MODES:
            m=all_eval_masks[(scenario,stress)]
            preds={
                "raw_logistic":external[law].predict_proba(
                    raw_observed_features(test_x,m))[:,1],
                "oracle_true_k2":expit(additive_logit(oracle[law][2],c[te,:,:2].mean(axis=1))),
                "oracle_true_k3":expit(additive_logit(oracle[law][3],c[te,:,:3].mean(axis=1))),
            }
            for method,p in preds.items():
                results.append({
                    "scenario":scenario,"stress":stress,"seed":None,
                    "method":method,"metrics":metrics(y[te],p),
                    "source_usage_fraction":0.,
                    "privileged_oracle_at_TEST_only":method.startswith("oracle"),
                })
    # The physically measured source and an equally predictive sensor artifact
    # are observationally equivalent. Reuse results, not independent models.
    for item in list(results):
        if item["scenario"]=="stable":
            results.append({**item,"scenario":"stable_instrument_interpretation",
                            "observationally_identical_copy":True})
    summary=[]
    for scenario in FAULTS+("stable_instrument_interpretation",):
        for stress in TEST_MODES:
            for method in METHODS:
                group=[r for r in results if r["scenario"]==scenario
                       and r["stress"]==stress and r["method"]==method]
                n_expected=len(seeds) if method in SOURCE_METHODS else 1
                if len(group)!=n_expected:
                    raise AssertionError(f"Incorrect row count: {scenario} {stress} {method}: {len(group)}")
                summary.append({
                    "scenario":scenario,"stress":stress,"method":method,
                    "native_fit_count":n_expected if method in SOURCE_METHODS else 0,
                    "mean_auprc":float(np.mean([s["metrics"]["auprc"] for s in group])),
                    "mean_auroc":float(np.mean([s["metrics"]["auroc"] for s in group])),
                    "mean_nll":float(np.mean([s["metrics"]["nll"] for s in group])),
                    "mean_brier":float(np.mean([s["metrics"]["brier"] for s in group])),
                    "mean_source_usage":float(np.mean([s["source_usage_fraction"] for s in group])),
                    "mean_repair_fraction":float(np.mean([s.get("repair_fraction",0.) for s in group])),
                    "mean_pair_agreement":(float(np.mean([s["pair_agreement_fraction"] for s in group]))
                                           if method in SOURCE_METHODS else None),
                    "mean_anchor_agreement":(float(np.mean([s["anchor_agreement_fraction"] for s in group]))
                                             if method in SOURCE_METHODS else None),
                })
    data_hash=hashlib.sha256(np.ascontiguousarray(x).tobytes()+
                 np.ascontiguousarray(c).tobytes()+
                 np.ascontiguousarray(y_pred).tobytes()+
                 np.ascontiguousarray(y_nuis).tobytes()).hexdigest()
    result={
        "status":"V30_SMOKE_ONLY" if smoke else "V30_NATIVE_NEW_NONLINEAR_FAMILY_TRUSTED_ANCHOR_STUDY",
        "preregistered_protocol":"docs/medical/IOI_V30_FROZEN_NONLINEAR_ANCHOR_PROTOCOL_20261106.md",
        "source_simulator_family":data["generator_family"],
        "not_previous_gaussian_AR_generator":True,
        "sensor_anchor_independent_noise_by_generator_assumption":True,
        "test_outcomes_used_for_training_threshold_admission_or_repair":False,
        "anchor_reliability_is_external_synthetic_assumption":True,
        "physiological_source_identity_certified":False,
        "n_unique_native_known_K2_models":len(seeds),
        "data_sizes":{"train":ntr,"validation_total":nva,
                       "validation_per_environment":nva//2,"test":nte},
        "epochs":epochs,"model_seeds":list(seeds),
        "source_train_labels":ntr,"additional_gate_label_subjects":nva,
        "all_test_modes":list(TEST_MODES),"scenarios":list(FAULTS),
        "pair_policy":pair_policy.as_dict(),
        "anchor_policy":anchor_policy.as_dict(),
        "known_concept_input_sensor_firewall":[0,1,2,3],
        "dataset_sha256":data_hash,
        "source_isolation_and_additive_invariants_validated":True,
        "gates":gates,"model_states":training,
        "rows":results,"summary":summary,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for file,body in (("STATUS.json",result),("SUMMARY.json",summary),
                      ("GATES.json",gates),("MODEL_STATES.json",training)):
        (out_dir/file).write_text(json.dumps(body,indent=2,allow_nan=False))
    primary=[q for q in summary if q["stress"]=="MCAR20"
             and q["scenario"]!="stable_instrument_interpretation"]
    print(json.dumps({
        "event":"V30_COMPLETED","status":result["status"],
        "n_unique_known_k2_models":len(seeds),"n_rows":len(results),
        "n_summary":len(summary),
        "primary":primary,
    },ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    evaluate(out_dir=args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
