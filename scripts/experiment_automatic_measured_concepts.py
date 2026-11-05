"""Full native ConceptFAN: TRAIN-only automatic measured-source formation.

The K2 Transformer is trained from OBSERVED SENSOR PAIR PROXIES, never
privileged hidden simulator concepts or outcome gradients. It is then frozen.
Unknown channels are clustered into multisensor candidate concepts with
TRAIN-only missingness-aware signed correlation; their outcome relevance is
fit with TRAIN labels and gate-approved with two disjoint VALID cohorts.
TEST results deliberately include false semantic source and sensor shift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.auto_measured_concept_benchmark import (
    DATA_SEED, M_TRAIN, M_VAL1, M_VAL2, M_TEST_BASE, N_CHANNELS,
    TRAIN_N, VALID_N, TEST_N, STRESS, apply_missing_source,
    extended_nonlinear_world, intervene, masked_histories,
    raw_measured_features,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups, measured_group_concept,
    group_provenance_fingerprint,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    fixed_unobserved_pca, pca_unknown_values,
    evaluate_two_independent_environments,select_at_most_one_candidate,
    admitted_test_logit,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    measured_candidate,train_source_coefficient,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_trained_conceptfan_omission_v25 import (
    BATCH, EPOCHS, LR, LAMBDA_C, NN_SEEDS, _call,
    metrics, predict_native, sha_state,
)

KNOWN_SENSORS=(0,1,2,3)
GATE_SEED=20267210
WORLDS=("physical_extra","complete_nuisance","artifact_predictive")
TEST_CASES=(
    ("physical_extra","stable"),
    ("physical_extra","source_common_sign_flip"),
    ("physical_extra","source_completely_unobserved"),
    ("complete_nuisance","stable"),
    ("artifact_predictive","stable"),
    ("artifact_predictive","artifact_common_sign_flip"),
)
METHODS=("frozen_K2","auto_selected_source","manual_V30_pair",
         "pca_unknown_rank1","all_measured_groups_CB_head",
         "raw_12channel_logistic","oracle_true_K2",
         "oracle_true_outcome_source")


def known_observed_proxy(x:np.ndarray,m:np.ndarray):
    """Only co-observed readings of preexisting concept sensor0/1 and2/3."""
    xx=torch.from_numpy(np.asarray(x,dtype=np.float32))
    mm=torch.from_numpy(np.asarray(m,dtype=np.float32))
    if xx.ndim!=3 or xx.shape!=mm.shape or xx.shape[-1]!=4:
        raise ValueError("Known K2 sensor firewall expects [episode,16,4]")
    targets=[]
    exists=[]
    for i in (0,2):
        observed=mm[:,:,i:i+2]
        payload=torch.where(observed.bool(),xx[:,:,i:i+2],torch.zeros_like(xx[:,:,i:i+2]))
        counts=observed.sum(dim=-1)
        targets.append(torch.sigmoid(payload.sum(dim=-1)/counts.clamp_min(1)))
        exists.append((counts>0).float())
    return torch.stack(targets,dim=-1),torch.stack(exists,dim=-1)


def train_masked_measured_known_concepts(
    model,x:np.ndarray,m:np.ndarray,*,seed:int,epochs:int
)->dict:
    """No target labels, hidden synthetic source trajectories, or extra sensors."""
    model.train()
    xx=torch.from_numpy(x.astype(np.float32))
    mm=torch.from_numpy(m.astype(np.float32))
    targets,loss_support=known_observed_proxy(x,m)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001)
    rng=np.random.default_rng(82000+seed)
    trace=[]
    for epoch in range(epochs):
        permutation=rng.permutation(len(x))
        losses=[]
        for start in range(0,len(x),BATCH):
            ix=permutation[start:start+BATCH]
            native,_=_call(model,xx[ix],mm[ix])
            if native.concept_trajectories.shape!=targets[ix].shape:
                raise AssertionError("Model receives only K2 observed concept supervision")
            residual=(native.concept_trajectories-targets[ix])**2
            loss=LAMBDA_C*(residual*loss_support[ix]).sum()/loss_support[ix].sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite observed-proxy-only TRAIN objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    return {
        "first_epoch_loss":trace[0],"last_epoch_loss":trace[-1],
        "epochs":len(trace),"supervision":"MEASURED_PAIR_ONLY_NO_ORACLE",
        "target_labels_used_in_native_concept_encoder":False,
        "synthetic_latent_concept_targets_used":False,
        "known_group_mapping_supplied_externally":[[0,1],[2,3]],
    }


def logit_from_head(head,features:np.ndarray)->np.ndarray:
    out,contributions=head(features)
    if float(np.max(np.abs(out-(head.bias+contributions.sum(axis=1)))))>1e-12:
        raise AssertionError("Concept bottleneck no longer strictly additive")
    return out


def candidate_with_support(name, source,
                           values:np.ndarray,m:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    if name=="manual_V30_pair":
        return measured_candidate(values[:,:,:6],m[:,:,:6])
    if name=="pca_unknown_rank1":
        return pca_unknown_values(source,values,m)
    return measured_group_concept(source,values,m)


def one_candidate_gate(
    name, source, x, masks, baseline, labels, splits, *, seed:int
):
    tr,v1,v2=splits
    qtrain,_=candidate_with_support(name,source,x[tr],masks["train"])
    gamma=train_source_coefficient(baseline["train"],qtrain,labels[tr])
    c1,s1=candidate_with_support(name,source,x[v1],masks["val1"])
    c2,s2=candidate_with_support(name,source,x[v2],masks["val2"])
    gate=evaluate_two_independent_environments(
        name,gamma,
        ("VALID1",labels[v1],baseline["val1"],c1,s1),
        ("VALID2",labels[v2],baseline["val2"],c2,s2),
        train_label_count=len(tr),
        seed=seed,
    )
    return gate


def _raw_logistic(train_x,train_m,train_labels):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
    model.fit(raw_measured_features(train_x,train_m),train_labels)
    return model


def _oracle_head_for_outcome(data,world,train_idx):
    c=data["oracle_concepts_for_posthoc_TEST_audit_only"].mean(axis=1)
    if world=="artifact_predictive":
        additional=data["oracle_artifact_for_posthoc_TEST_audit_only"].mean(axis=1)
    else:
        additional=c[:,2]
    expanded=np.column_stack((c[:,:2],additional))
    y=data["outcomes"][world]
    return (refit_linear_concept_head(c[train_idx,:2],y[train_idx]),
            refit_linear_concept_head(expanded[train_idx],y[train_idx]),
            c[:,:2],expanded)


def _summarize(rows,condition,method,stress,n_seeds):
    subset=[z for z in rows if z["condition"]==condition
            and z["method"]==method and z["stress"]==stress]
    n_expected=1 if method in (
        "raw_12channel_logistic","oracle_true_K2","oracle_true_outcome_source"
    ) else n_seeds
    if len(subset)!=n_expected:
        raise AssertionError(f"Expected {n_expected} rows for {condition}/{method}/{stress}, got {len(subset)}")
    return {
        "condition":condition,"method":method,"stress":stress,
        "native_seed_fits":n_expected if n_expected==n_seeds and method not in (
            "raw_12channel_logistic","oracle_true_K2","oracle_true_outcome_source") else 0,
        "mean_auprc":float(np.mean([z["test_metrics"]["auprc"] for z in subset])),
        "mean_auroc":float(np.mean([z["test_metrics"]["auroc"] for z in subset])),
        "mean_nll":float(np.mean([z["test_metrics"]["nll"] for z in subset])),
        "mean_brier":float(np.mean([z["test_metrics"]["brier"] for z in subset])),
        "mean_candidate_usage":float(np.mean([z.get("candidate_usage",0.) for z in subset])),
    }


def run(out_dir:Path,*,smoke:bool=False):
    if out_dir.exists():
        raise FileExistsError("Research evidence must be immutable; choose a new path")
    torch.set_num_threads(2)
    ntr,nva,nte=(300,300,300) if smoke else (TRAIN_N,VALID_N,TEST_N)
    seeds=(11,) if smoke else NN_SEEDS
    epochs=2 if smoke else EPOCHS
    data=extended_nonlinear_world(seed=DATA_SEED,
            train_n=ntr,valid_n=nva,test_n=nte)
    x=data["x"]
    tr,v1,v2,te=(data["train"],data["valid_env1"],
                  data["valid_env2"],data["test"])
    assert len(set(tr)&set(v1))==len(set(v1)&set(v2))==len(set(tr)&set(te))==0
    masks={
        "train":masked_histories(x[tr],"MCAR20",seed=M_TRAIN),
        "val1":masked_histories(x[v1],"MCAR20",seed=M_VAL1),
        "val2":masked_histories(x[v2],"MCAR20",seed=M_VAL2),
    }
    proposals=form_train_measured_groups(
        x[tr],masks["train"],
        known_sensor_indices=KNOWN_SENSORS,
        seed=20267204,
        min_pair_abs_corr=.60,max_candidates=5,
    )
    # No true source groups, Y, validation or TEST data enter proposals.
    pca=fixed_unobserved_pca(x[tr],masks["train"],proposals.searched_sensors)
    sources={c.candidate_id:c for c in proposals.candidates}
    sources["manual_V30_pair"]=None
    sources["pca_unknown_rank1"]=pca
    named=tuple(c.candidate_id for c in proposals.candidates)
    if not named:
        raise AssertionError("No unlabeled source candidates identified: report FAIL")
    test_masks={}
    for world,scenario in TEST_CASES:
        for j,stress in enumerate(STRESS):
            observed=intervene(x[te],scenario)
            m=masked_histories(observed,stress,seed=M_TEST_BASE+j)
            test_masks[(world,scenario,stress)]=apply_missing_source(m,scenario)
    rows,gate_records,training,selection,semantic_audit=[],[],[],[],[]
    # Single-fit conventional raw/oracle references on identical cohort.
    baseline_raw={}
    oracle_heads={}
    for world in WORLDS:
        y=data["outcomes"][world]
        baseline_raw[world]=_raw_logistic(x[tr],masks["train"],y[tr])
        oracle_heads[world]=_oracle_head_for_outcome(data,world,tr)
    for seed in seeds:
        core=make_firewalled_k2(seed)
        stage=train_masked_measured_known_concepts(
            core,x[tr,:,:4],masks["train"][:,:,:4],seed=seed,epochs=epochs)
        k2_sha=sha_state(core)
        training.append({"seed":seed,"known_concept_sha256":k2_sha,**stage})
        qtrain=predict_native(core,x[tr,:,:4],masks["train"][:,:,:4])
        qval1=predict_native(core,x[v1,:,:4],masks["val1"][:,:,:4])
        qval2=predict_native(core,x[v2,:,:4],masks["val2"][:,:,:4])
        val_k2={"train":qtrain["concept_summary"],"val1":qval1["concept_summary"],
                "val2":qval2["concept_summary"]}
        test_known={}
        for j,stress in enumerate(STRESS):
            core_x=x[te,:,:4]
            # The 4 K2 measurement channels are untouched across fault
            # scenarios, and the random mask streams are fixed by stress.
            basicmask=masked_histories(x[te],"complete",seed=M_TEST_BASE+j) if stress=="complete" else (
                masked_histories(x[te],stress,seed=M_TEST_BASE+j))
            test_known[stress]=predict_native(
                core,core_x,basicmask[:,:,:4],
                data["oracle_concepts_for_posthoc_TEST_audit_only"][te,:,:2],
            )
        for world in WORLDS:
            y=data["outcomes"][world]
            head=refit_linear_concept_head(val_k2["train"],y[tr])
            original_head=(head.bias,head.weights)
            baseline_logits={key:logit_from_head(head,q)
                              for key,q in val_k2.items()}
            group_gates=[]
            all_gates={}
            for i,group in enumerate(proposals.candidates):
                gate=one_candidate_gate(
                    group.candidate_id,group,x,masks,baseline_logits,y,
                    (tr,v1,v2),seed=GATE_SEED+10*i,
                )
                group_gates.append(gate)
                all_gates[group.candidate_id]=gate
                gate_records.append({"world":world,"nn_seed":seed,
                                     "source_family":"AUTO_UNLABELED_TRAIN",
                                     **gate.as_dict()})
            selected=select_at_most_one_candidate(tuple(group_gates))
            for i,name in enumerate(("manual_V30_pair","pca_unknown_rank1")):
                gate=one_candidate_gate(
                    name,sources[name],x,masks,baseline_logits,y,(tr,v1,v2),
                    seed=GATE_SEED+100+10*i,
                )
                all_gates[name]=gate
                gate_records.append({"world":world,"nn_seed":seed,
                                     "source_family":"BASELINE",
                                     **gate.as_dict()})
            selection.append({
                "world":world,"nn_seed":seed,
                "selected_candidate_id":selected,
                "no_approved_group":selected is None,
                "approved_unsupervised_groups":[z.candidate_id for z in group_gates if z.approved],
                "source_training_labels":ntr,
                "independent_validation_labels":nva,
                "known_concept_sha256":k2_sha,
                "proposal_hash":group_provenance_fingerprint(proposals),
            })
            # Full risk-head refit on ALL measured proposals: a regular
            # capacity baseline; it does NOT keep old K2 coefficients.
            with_all=np.column_stack(
                [val_k2["train"]]
                +[measured_group_concept(z,x[tr],masks["train"])[0]
                  for z in proposals.candidates])
            all_head=refit_linear_concept_head(with_all,y[tr])
            for original_world,scenario in TEST_CASES:
                if original_world!=world:
                    continue
                for stress in STRESS:
                    shifted=intervene(x[te],scenario)
                    mt=test_masks[(world,scenario,stress)]
                    known=test_known[stress]["concept_summary"]
                    old=logit_from_head(head,known)
                    def from_source(name,group,approved,coeff):
                        q,s=candidate_with_support(name,group,shifted,mt)
                        z,used=admitted_test_logit(
                            old,q,s,gamma=coeff,approved=approved,
                            minimum_fraction=.50,
                        )
                        return z,used
                    results={"frozen_K2":(old,np.zeros(len(old),bool))}
                    if selected is None:
                        results["auto_selected_source"]=(old.copy(),np.zeros(len(old),bool))
                    else:
                        agate=all_gates[selected]
                        results["auto_selected_source"]=from_source(
                            selected,sources[selected],agate.approved,agate.gamma_train_only)
                    for method,name in (("manual_V30_pair","manual_V30_pair"),
                                        ("pca_unknown_rank1","pca_unknown_rank1")):
                        gate=all_gates[name]
                        results[method]=from_source(
                            name,sources[name],gate.approved,gate.gamma_train_only)
                    q_all=np.column_stack(
                        [known]+[measured_group_concept(z,shifted,mt)[0]
                                  for z in proposals.candidates])
                    results["all_measured_groups_CB_head"]=(
                        logit_from_head(all_head,q_all),np.zeros(len(old),bool))
                    for method,(logits,used) in results.items():
                        # For all source-isolated methods the old K2
                        # contribution stays literally unchanged. Only
                        # a separately named signed source is allowed.
                        if method!="all_measured_groups_CB_head":
                            signed=logits-old
                            np.testing.assert_array_equal(logits[~used],old[~used])
                            if not np.isfinite(signed).all():
                                raise AssertionError("Nonfinite extra concept contribution")
                        rows.append({
                            "condition":f"{world}/{scenario}","world":world,
                            "scenario":scenario,"stress":stress,
                            "nn_seed":seed,"method":method,
                            "test_metrics":metrics(y[te],expit(logits)),
                            "candidate_usage":float(used.mean()),
                            "selected_auto_group":selected,
                            "known_K2_source_sha256":k2_sha,
                            "known_oracle_MSE_TEST_AUDIT_ONLY":test_known[stress]["concept_mse"],
                            "exact_additive_K2_plus_candidate":method!="all_measured_groups_CB_head",
                        })
            if (sha_state(core)!=k2_sha or
                original_head!=(head.bias,head.weights)):
                raise AssertionError("Source admission altered frozen K2 model or risk head")
        print(json.dumps({
            "event":"AUTOMATIC_SOURCE_SEED","seed":seed,
            "known_sha256":k2_sha,
            "train_proposals":[z.sensor_indices for z in proposals.candidates],
            "admissions":[{"world":z["world"],"selected":z["selected_candidate_id"],
                           "approved":z["approved_unsupervised_groups"]}
                          for z in selection if z["nn_seed"]==seed],
        }),flush=True)
    for world,scenario in TEST_CASES:
        y=data["outcomes"][world]
        o2,oextra,q2,qextra=oracle_heads[world]
        for stress in STRESS:
            shifted=intervene(x[te],scenario)
            mt=test_masks[(world,scenario,stress)]
            refs={
                "raw_12channel_logistic":baseline_raw[world].predict_proba(
                    raw_measured_features(shifted,mt))[:,1],
                "oracle_true_K2":expit(logit_from_head(o2,q2[te])),
                "oracle_true_outcome_source":expit(logit_from_head(oextra,qextra[te])),
            }
            for method,p in refs.items():
                rows.append({
                    "condition":f"{world}/{scenario}","world":world,
                    "scenario":scenario,"stress":stress,
                    "nn_seed":None,"method":method,
                    "test_metrics":metrics(y[te],p),
                    "candidate_usage":0.,
                    "privileged_oracle_trajectories_at_TEST_ONLY":method.startswith("oracle"),
                })
    summary=[]
    for world,scenario in TEST_CASES:
        for stress in STRESS:
            for method in METHODS:
                summary.append(_summarize(rows,f"{world}/{scenario}",
                                          method,stress,len(seeds)))
    # Entirely observationally equivalent alternative interpretation of
    # the physiological extra source: it uses exact same frozen data/results.
    semantic_fingerprint=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    report={
        "status":"SMOKE_ONLY" if smoke else "AUTO_MEASURED_CONCEPTS_TRAIN_ONLY_PROPOSE_TWO_ENV_GATED",
        "frozen_protocol":"docs/medical/IOI_AUTOMATIC_MEASURED_CONCEPTS_FROZEN_PROTOCOL_20261106.md",
        "generator":"extended_V30_nonlinear_laplace_switching",
        "generator_seed":DATA_SEED,
        "separate_test_generator_draw_but_same_v30_nonlinear_family":True,
        "never_trained_on_simulator_true_concepts":True,
        "initial_known_concept_sensor_pair_mapping_supplied":True,
        "true_concepts_only_posthoc_oracle_baselines_and_audit":True,
        "source_proposals_do_not_use_any_labels":True,
        "source_gating_uses_two_disjoint_VALID_outcome_cohorts":True,
        "candidate_selection_no_TEST_outcomes":True,
        "no_automatic_physical_or_causal_identity":True,
        "source_artifact_semantic_ambiguity_negative_control":True,
        "split_sizes":{"TRAIN":ntr,"VALID_ENV1":nva//2,"VALID_ENV2":nva//2,"TEST":nte},
        "nn_seeds":list(seeds),"epochs":epochs,
        "n_known_neural_fits":len(seeds),"n_unevaluated_test_law_changes":0,
        "proposal_manifest":proposals.as_dict(),
        "proposal_sha256":group_provenance_fingerprint(proposals),
        "raw_data_sha256":semantic_fingerprint,
        "known_concept_training_receipts":training,
        "gates":gate_records,
        "selections":selection,
        "rows":rows,"summary":summary,
        "oracle_true_group_metadata_EVALUATION_ONLY":data["evaluation_only_true_groups"],
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for name,datum in (
        ("STATUS.json",report),("SUMMARY.json",summary),
        ("PROPOSALS.json",proposals.as_dict()),
        ("GATES.json",gate_records),("SELECTIONS.json",selection),
        ("KNOWN_CONCEPT_TRAINING.json",training)
    ):
        (out_dir/name).write_text(json.dumps(datum,indent=2,allow_nan=False))
    print(json.dumps({
        "event":"AUTOMATIC_SOURCE_COMPLETE",
        "status":report["status"],"n_real_native_fits":len(seeds),
        "n_candidate_groups":len(proposals.candidates),
        "groups":[z.sensor_indices for z in proposals.candidates],
        "n_summary":len(summary),"n_test_rows":len(rows),
        "selection":selection,
        "MCAR20_summary":[z for z in summary if z["stress"]=="MCAR20"],
    }),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
