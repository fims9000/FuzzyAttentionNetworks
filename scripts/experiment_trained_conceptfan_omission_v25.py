"""V25: trained canonical ConceptFAN omission vs estimation error falsification.

Protocol: docs/medical/IOI_V25_FROZEN_TRAINED_NETWORK_OMISSION_PROTOCOL_20261106.md
Runs the *actual* repository MultiSetAdditiveTemporalConceptFANModel with
a synthetic oracle-concept supervision term on TRAIN ONLY.
Synthetic complete/omitted/artifact worlds must not be interpreted as
causal discovery or valid clinical concepts. No PIERE patient data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fan.concept.completeness_audit import audit_oracle_concept_coverage
from fan.concept.observability import ObservationAwareConceptFAN
from fan.concept.temporal import MultiSetAdditiveTemporalConceptFANModel

FULL_TRAIN, FULL_CAL, FULL_TEST = 1600, 400, 800
TIME_STEPS, CHANNELS, TRUE_FACTORS = 16, 6, 3
NN_SEEDS = (11, 23, 37)
STRESS = ("complete", "MCAR10", "MCAR20", "MCAR40", "MAR", "MNAR", "block8")
LR, EPOCHS, BATCH, LAMBDA_C, COVERAGE_THRESHOLD = 0.002, 14, 128, 5.0, 0.65


def frozen_world(*, seed: int = 20261106, n_train: int = FULL_TRAIN,
                 n_cal: int = FULL_CAL, n_test: int = FULL_TEST) -> dict[str, np.ndarray]:
    """All three worlds share exactly the same observed sensor law."""
    if min(n_train, n_cal, n_test) < 10:
        raise ValueError("Every partition needs >=10 episodes")
    n = n_train + n_cal + n_test
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, TIME_STEPS, TRUE_FACTORS)).astype(np.float32)
    z = np.empty_like(eps)
    z[:, 0] = eps[:, 0]
    rho = 0.77
    for t in range(1, TIME_STEPS):
        z[:, t] = rho * z[:, t - 1] + np.sqrt(1.0 - rho**2) * eps[:, t]
    concepts = expit(1.2 * z).astype(np.float32)
    x = np.empty((n, TIME_STEPS, CHANNELS), np.float32)
    for k in range(TRUE_FACTORS):
        for sensor in (0, 1):
            x[:, :, 2*k + sensor] = (
                z[:, :, k] + 0.32 * rng.standard_normal((n, TIME_STEPS))
            ).astype(np.float32)
    c = concepts.mean(axis=1)
    common_logit = 6.0 * (c[:, 0] - 0.5) - 5.5 * (c[:, 1] - 0.5) - 0.15
    extra_logit = common_logit + 7.0 * (c[:, 2] - 0.5)
    uniforms = rng.random(n)
    complete = (uniforms < expit(common_logit)).astype(np.int64)
    extra = (uniforms < expit(extra_logit)).astype(np.int64)
    result = {
        "x": x, "concept_trajectories": concepts,
        "y_complete": complete, "y_omitted": extra, "y_artifact": extra.copy(),
        "index_train": np.arange(0, n_train),
        "index_cal": np.arange(n_train, n_train + n_cal),
        "index_test": np.arange(n_train + n_cal, n),
    }
    assert np.array_equal(result["y_omitted"], result["y_artifact"])
    assert np.intersect1d(result["index_train"], result["index_cal"]).size == 0
    assert np.intersect1d(result["index_cal"], result["index_test"]).size == 0
    return result


def mask_mechanism(x: np.ndarray, mode: str, *, seed: int) -> np.ndarray:
    """Evaluation masks never use outcomes, concept ground truth or predictions."""
    if x.ndim != 3 or x.shape[-1] != CHANNELS or x.shape[1] != TIME_STEPS:
        raise ValueError("Expected [episodes,16,6] synthetic observed sensor histories")
    rng = np.random.default_rng(seed)
    n, t, c = x.shape
    mask = np.ones((n, t, c), dtype=np.bool_)
    if mode == "complete":
        return mask
    if mode.startswith("MCAR"):
        fraction = int(mode[4:]) / 100.0
        mask = rng.random(mask.shape) >= fraction
    elif mode == "MAR":
        # Missingness of all but channel0 depends on channel0 that stays observed.
        clock = (np.arange(t)[None, :] >= (t // 2)).astype(float)
        dependence = 1.2 * x[:, :, 0] + 0.7 * clock - 0.4
        p = 0.07 + 0.73 * expit(dependence)
        mask[:, :, 1:] = rng.random((n, t, c - 1)) >= p[:, :, None]
        mask[:, :, 0] = True
    elif mode == "MNAR":
        # Each channel's missingness depends on its own underlying value.
        p = 0.07 + 0.75 * expit(1.7 * x)
        mask = rng.random(mask.shape) >= p
    elif mode == "block8":
        mask = rng.random(mask.shape) >= 0.08
        start = rng.integers(0, t - 8 + 1, size=(n, c))
        for i in range(n):
            for channel in range(c):
                mask[i, start[i, channel]:start[i, channel]+8, channel] = False
    else:
        raise ValueError("Unknown frozen stress mode")
    return mask


def flatten_measured_inputs(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Same observed-channel statistical features for raw-input logistic baseline."""
    zero = np.where(m, x, 0.0)
    count = m.sum(axis=1).clip(min=1)
    recent_count = m[:, -6:].sum(axis=1).clip(min=1)
    means = zero.sum(axis=1) / count
    recent = zero[:, -6:].sum(axis=1) / recent_count
    return np.concatenate([means, recent, m.mean(axis=1)], axis=1)


def measured_witness(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Actually measured third-pair evidence, no hidden-value or oracle shortcut."""
    if x.shape != m.shape:
        raise ValueError("Measured witness values and masks must match")
    usable = np.where(m[:, :, 4:6], x[:, :, 4:6], 0.0)
    denominator = m[:, :, 4:6].sum(axis=(1, 2)).clip(min=1)
    return (usable.sum(axis=(1, 2)) / denominator)[:, None].astype(np.float64)


def provenance_map(k: int) -> torch.Tensor:
    if k not in (2, 3):
        raise ValueError("Only frozen K2 or privileged K3 oracle control")
    p = torch.zeros((k, CHANNELS), dtype=torch.float32)
    for i in range(k):
        p[i, 2*i:2*i+2] = 1.0
    return p


def make_model(kind: str, k: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    core = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=2 * CHANNELS, sequence_length=TIME_STEPS,
        latent_dim=24, n_concepts=k,
        n_memberships=3, membership="sigmoid", oracle=False,
        temporal_mode="attention", dropout=0.0,
        encoder_layers=1, encoder_heads=2, encoder_ffn=48,
        alpha_mode="no_alpha", positive_decision_weights=False,
    )
    if kind == "native" or kind == "oracle_k3":
        return core
    if kind == "observability":
        return ObservationAwareConceptFAN(
            core, provenance_map(k), min_observation_fraction=COVERAGE_THRESHOLD
        )
    raise ValueError(kind)


def _call(model: torch.nn.Module, x: torch.Tensor, m: torch.Tensor):
    if isinstance(model, ObservationAwareConceptFAN):
        out = model(x, m)
        return out.native, out
    masked = torch.where(m.bool(), x, torch.zeros_like(x))
    features = torch.cat((masked, m), dim=-1)
    out = model(features)
    return out, None


def sha_state(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, weight in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(weight.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_native(model: torch.nn.Module, x: np.ndarray, m: np.ndarray,
               true_concepts: np.ndarray, labels: np.ndarray, *,
               seed: int, epochs: int) -> dict[str, float]:
    model.train()
    xx = torch.from_numpy(x.astype(np.float32))
    mm = torch.from_numpy(m.astype(np.float32))
    cc = torch.from_numpy(true_concepts.astype(np.float32))
    yy = torch.from_numpy(labels.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.001)
    order_rng = np.random.default_rng(82000 + seed)
    trace: list[float] = []
    for epoch in range(epochs):
        permutation = order_rng.permutation(len(x))
        losses = []
        for start in range(0, len(permutation), BATCH):
            ix = permutation[start:start+BATCH]
            native, wrapper = _call(model, xx[ix], mm[ix])
            task = F.binary_cross_entropy_with_logits(native.logit, yy[ix])
            target = cc[ix, :, :native.concept_trajectories.shape[-1]]
            concept = F.mse_loss(native.concept_trajectories, target)
            objective = task + LAMBDA_C * concept
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite ConceptFAN objective")
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(objective.detach()))
        trace.append(float(np.mean(losses)))
    return {"first_epoch_objective":trace[0], "final_epoch_objective":trace[-1],
            "epochs":len(trace)}


@torch.no_grad()
def predict_native(model: torch.nn.Module, x: np.ndarray, m: np.ndarray,
                   concepts: np.ndarray | None = None) -> dict[str, Any]:
    model.eval()
    out_logits, out_concepts, out_summaries, out_support, out_abstain = [], [], [], [], []
    max_decomp_error = 0.0
    max_unsupported_effect = 0.0
    for start in range(0, len(x), 256):
        tx = torch.from_numpy(x[start:start+256].astype(np.float32))
        tm = torch.from_numpy(m[start:start+256].astype(np.float32))
        native, wrapper = _call(model, tx, tm)
        if not torch.isfinite(native.logit).all():
            raise ValueError("Nonfinite native ConceptFAN logit")
        core = model.core if wrapper is not None else model
        err = (native.logit - (core.decision_head.bias
                              + native.signed_decision_contributions.sum(dim=1))).abs().max()
        max_decomp_error = max(max_decomp_error, float(err))
        if wrapper is not None:
            unsupported = ~wrapper.supported_concepts
            if unsupported.any():
                max_unsupported_effect = max(
                    max_unsupported_effect,
                    float(native.signed_decision_contributions[unsupported].abs().max()))
            out_abstain.append(wrapper.abstain_from_explanation.cpu().numpy())
            out_support.append(wrapper.supported_concepts.cpu().numpy())
        else:
            out_abstain.append(np.zeros((len(tx),), dtype=bool))
            out_support.append(np.ones((len(tx), native.concept_summaries.shape[1]), dtype=bool))
        out_logits.append(native.logit.cpu().numpy())
        out_concepts.append(native.concept_trajectories.cpu().numpy())
        out_summaries.append(native.concept_summaries.cpu().numpy())
    logits = np.concatenate(out_logits, axis=0).astype(np.float64)
    predicted = np.concatenate(out_concepts, axis=0)
    if max_decomp_error > 2e-5 or max_unsupported_effect > 1e-6:
        raise AssertionError("Broken strict additive concept bottleneck")
    mse = (float(np.mean((predicted-concepts[:,:,:predicted.shape[-1]])**2))
           if concepts is not None else None)
    return {
        "logits":logits, "p":expit(logits), "concept_summary":np.concatenate(out_summaries),
        "abstain":np.concatenate(out_abstain), "support":np.concatenate(out_support),
        "concept_mse":mse, "max_additive_error":max_decomp_error,
        "max_unsupported_effect":max_unsupported_effect,
    }


def metrics(y: np.ndarray, p: np.ndarray, subset: np.ndarray | None = None) -> dict:
    if subset is not None:
        y, p = y[subset], p[subset]
    if len(y) == 0:
        return {"n":0,"positives":0, "auprc":None, "auroc":None,"nll":None,"brier":None}
    p = np.clip(p.astype(np.float64),1e-7,1-1e-7)
    valid = len(np.unique(y))==2
    return {"n":int(len(y)), "positives":int(y.sum()),
            "auprc":float(average_precision_score(y,p)) if valid else None,
            "auroc":float(roc_auc_score(y,p)) if valid else None,
            "nll":float(log_loss(y,p,labels=[0,1])),
            "brier":float(brier_score_loss(y,p))}


def external_logistic(x_train: np.ndarray, m_train: np.ndarray, y_train: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=1000))
    model.fit(flatten_measured_inputs(x_train,m_train), y_train)
    return model


def run(*, out_dir: Path, smoke: bool = False) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError("Never overwrite complete research evidence")
    torch.set_num_threads(min(2,max(1,os.cpu_count() or 1)))
    n_train, n_cal, n_test = ((160,40,160) if smoke
                             else (FULL_TRAIN,FULL_CAL,FULL_TEST))
    epochs = 2 if smoke else EPOCHS
    seeds = (NN_SEEDS[0],) if smoke else NN_SEEDS
    bootstrap = 199 if smoke else 499
    world = frozen_world(n_train=n_train,n_cal=n_cal,n_test=n_test)
    tr, ca, te = (world["index_train"],world["index_cal"],world["index_test"])
    x, c = world["x"], world["concept_trajectories"]
    train_mask = mask_mechanism(x[tr],"MCAR20",seed=20262001)
    # Calibration remains sealed from ALL model/diagnostic fits; this
    # version freezes the support threshold a priori at 0.65.
    assert np.array_equal(world["y_omitted"],world["y_artifact"])
    stress_masks = {mode:mask_mechanism(x[te],mode,seed=20262200+i)
                    for i,mode in enumerate(STRESS)}
    test_reference_mask = stress_masks["MCAR20"]
    rows, diagnostics, training_records = [], [], []
    train_test_fingerprint = hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
        + np.ascontiguousarray(c).tobytes()
        + np.ascontiguousarray(world["y_omitted"]).tobytes()
        + np.ascontiguousarray(world["y_complete"]).tobytes()
    ).hexdigest()
    for family in ("complete","omitted"):
        y = world["y_complete"] if family=="complete" else world["y_omitted"]
        baseline = external_logistic(x[tr],train_mask,y[tr])
        for mode, mt in stress_masks.items():
            probabilities = baseline.predict_proba(flatten_measured_inputs(x[te],mt))[:,1]
            rows.append({"family":family, "seed":None,"architecture":"raw_logistic",
                         "mode":mode,"coverage":1.0,"abstentions":0,
                         "overall":metrics(y[te], probabilities),
                         "selected":metrics(y[te], probabilities),
                         "concept_mse":None,"max_additive_error":None})
        for seed in seeds:
            learned: dict[str, torch.nn.Module] = {}
            for architecture in ("native_k2","observation_aware_k2","privileged_oracle_k3"):
                kind = ("observability" if architecture=="observation_aware_k2"
                        else "oracle_k3" if architecture=="privileged_oracle_k3"
                        else "native")
                k = 3 if architecture=="privileged_oracle_k3" else 2
                model = make_model(kind,k,seed)
                train_receipt = fit_native(model,x[tr],train_mask,c[tr],y[tr],
                                           seed=seed,epochs=epochs)
                trained_hash = sha_state(model)
                learned[architecture] = model
                training_records.append({"family":family, "seed":seed,
                                         "architecture":architecture, "k":k,
                                         "model_sha256":trained_hash,**train_receipt})
                for mode, mt in stress_masks.items():
                    predictions=predict_native(model,x[te],mt,c[te])
                    accepted=~predictions["abstain"]
                    rows.append({
                        "family":family, "seed":seed,"architecture":architecture,
                        "mode":mode,"coverage":float(accepted.mean()),
                        "abstentions":int((~accepted).sum()),
                        "overall":metrics(y[te],predictions["p"]),
                        "selected":metrics(y[te],predictions["p"],accepted),
                        "concept_mse":predictions["concept_mse"],
                        "max_additive_error":predictions["max_additive_error"],
                        "max_unsupported_effect":predictions["max_unsupported_effect"],
                    })
                if sha_state(model) != trained_hash:
                    raise AssertionError("TEST evaluation illegally modified model parameters")
            # Train-set predictions are for fitted TRAIN-only diagnostic
            # probes, not sample-independent model validation.
            k2=learned["native_k2"]
            predicted_train=predict_native(k2,x[tr],train_mask)["concept_summary"]
            predicted_test=predict_native(k2,x[te],test_reference_mask)["concept_summary"]
            true_train=c[tr,:,:2].mean(axis=1)
            true_test=c[te,:,:2].mean(axis=1)
            oracle_report=audit_oracle_concept_coverage(
                labels=np.concatenate([y[tr],y[te]]),
                predicted_concepts=np.concatenate([predicted_train,predicted_test]),
                true_concepts=np.concatenate([true_train,true_test]),
                residual_candidate=np.concatenate([measured_witness(x[tr],train_mask),measured_witness(x[te],test_reference_mask)]),
                train_indices=np.arange(len(tr)),
                test_indices=np.arange(len(tr),len(tr)+len(te)),
                bootstrap_replicates=bootstrap,
                seed=902000+seed,
            )
            diagnostics.append({"family":family,"seed":seed,**oracle_report.as_dict(),
                                "concept_estimation_gain_in_omitted_world_can_include_task_leakage":True})
            print(json.dumps({
                "event":"trained_family_seed","family":family,"seed":seed,
                "oracle_residual_gain":oracle_report.oracle_residual_gain,
                "oracle_residual_ci95":oracle_report.oracle_residual_gain_ci95,
                "estimation_gain":oracle_report.concept_estimation_gain,
                "native_MCAR20_nll":next(row["overall"]["nll"] for row in rows
                    if row["family"]==family and row["seed"]==seed
                    and row["architecture"]=="native_k2" and row["mode"]=="MCAR20"),
            }),flush=True)
    # The artifact world is observationally *identical* to omitted, even at
    # the individual prediction level. Reuse actual fitted models/metrics;
    # never invent independent runs or a distinct TEST cohort.
    old_rows=list(rows)
    for row in old_rows:
        if row["family"]=="omitted":
            rows.append({**row,"family":"artifact","observationally_identical_to_omitted":True})
    for row in list(diagnostics):
        if row["family"]=="omitted":
            diagnostics.append({**row,"family":"artifact",
                                "observationally_identical_to_omitted":True})
    for row in list(training_records):
        if row["family"]=="omitted":
            training_records.append({**row,"family":"artifact",
                                     "same_trained_model_reused":True})
    summary=[]
    for family in ("complete","omitted","artifact"):
        for architecture in ("raw_logistic","native_k2","observation_aware_k2",
                             "privileged_oracle_k3"):
            for mode in STRESS:
                selected=[row for row in rows
                          if row["family"]==family and row["architecture"]==architecture
                          and row["mode"]==mode]
                assert len(selected)==(1 if architecture=="raw_logistic" else len(seeds))
                summary.append({
                    "family":family,"architecture":architecture,"mode":mode,
                    "mean_over_fit_seeds_test_auprc":float(np.mean([r["overall"]["auprc"] for r in selected])),
                    "mean_over_fit_seeds_test_nll":float(np.mean([r["overall"]["nll"] for r in selected])),
                    "mean_over_fit_seeds_test_brier":float(np.mean([r["overall"]["brier"] for r in selected])),
                    "mean_coverage":float(np.mean([r["coverage"] for r in selected])),
                    "mean_selected_test_nll":float(np.mean([r["selected"]["nll"]
                        for r in selected if r["selected"]["nll"] is not None]))
                    if any(r["selected"]["nll"] is not None for r in selected) else None,
                })
    report = {
        "status":"SMOKE_ONLY" if smoke else "V25_TRAINED_CANONICAL_CONCEPTFAN_OMISSION_AUDIT",
        "protocol":"docs/medical/IOI_V25_FROZEN_TRAINED_NETWORK_OMISSION_PROTOCOL_20261106.md",
        "original_observational_artifact_world_equals_omitted":True,
        "artifact_fits_are_reused_not_independent":True,
        "real_patient_data_used":False,"oracle_concepts_supervised_on_train_only":True,
        "no_semantic_or_causal_identification_claim":True,
        "test_outcomes_never_used_for_training_model_selection_or_thresholds":True,
        "generator_seed":20261106,"NN_seeds":list(seeds),
        "sample_sizes":{"train":n_train,"unused_calibration":n_cal,"test":n_test},
        "epochs":epochs,"batch_size":BATCH,"learning_rate":LR,
        "min_support_fraction":COVERAGE_THRESHOLD,
        "fingerprint_sha256":train_test_fingerprint,
        "stress_conditions":list(STRESS),
        "n_trained_distinct_native_models":int(2 * len(seeds) * 3),
        "rows":rows,"diagnostics":diagnostics,"training_receipts":training_records,
        "summary":summary,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"STATUS.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (out_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out_dir/"TRAINING.json").write_text(json.dumps(training_records,indent=2,allow_nan=False))
    print(json.dumps({
        "event":"completed","status":report["status"],
        "n_trained_distinct_native_models":report["n_trained_distinct_native_models"],
        "n_rows":len(rows),"n_summaries":len(summary),
        "short_summary":[r for r in summary if r["family"]!="artifact"
                         and r["mode"] in ("MCAR20","MCAR40","MNAR")],
    }),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(out_dir=args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
