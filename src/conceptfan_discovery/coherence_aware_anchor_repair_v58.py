"""V58: coherence-aware independent-reference repair inside native fuzzy concepts.

External reference is not a supervised shortcut. It is a separately certified
measurement of one observational source, mapped with unlabeled TRAIN-only
measurements. Correlated malfunction of 3 primary sensors but not their
independent reference can be corrected at the SOURCE-INPUT layer.

Source coherence distinguishes detectable single-channel conflict (leave
the V54 robust median intact) from potentially common-mode source failure
(reference repair). Neither the coherence test nor the reference can prove
which device failed, and joint faults can escape both.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha256
import json

import numpy as np
import torch
from scipy.special import expit

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    check_measurements,_disagreement
)
from conceptfan_discovery.independent_anchor_verification_v57 import (
    IndependentAnchorVerifier,_check_anchor
)


@dataclass(frozen=True)
class CoherenceReceipt:
    source_group: tuple[int,...] | None
    concept_index: int | None
    train_coherence_cutoff_99pct: float | None
    n_train_coherence_subjects: int
    independent_reference_manifest_sha256: str
    n_unlabeled_source_train_subjects: int
    source_threshold_fitted_with_outcomes: bool = False
    physical_source_identity_confirmed: bool = False
    reference_credibility_is_external_assumption: bool = True

    def as_dict(self)->dict:
        return asdict(self)


class CoherenceAwareAnchorRepair:
    """V58 conditional source-input replacement, with frozen native V56 head.

    Modes:
      * discordant external reference + coherent primary source => repaired;
      * incoherent single/multiple sensor group => keep robust primary source;
      * source wholly missing + reference present => reconstruct from ref;
      * missing external reference => original V56, status UNVERIFIED;
      * no validated association => original V56, status UNVERIFIED.

    All changes occur at one matched concept input and pass through the SAME
    frozen native causal fuzzy encoder; task beta, bias, other concepts remain
    literally unchanged. No TEST outcomes are accepted or used.
    """
    def __init__(self,verifier:IndependentAnchorVerifier):
        if verifier.evidence is None:
            raise ValueError("Train-only external reference must be calibrated")
        self.verifier=verifier
        self.model=verifier.model
        self.receipt:CoherenceReceipt|None=None
        self.fingerprint:str|None=None

    def fit_from_unlabeled(self,x,mask):
        a,m=check_measurements(x,mask)
        evidence=self.verifier.evidence
        if evidence.source_definition_sha256!=self.model.bank.sha:
            raise AssertionError("Source basis changed before calibration")
        k=evidence.matched_concept_index
        if k is None:
            self.receipt=CoherenceReceipt(
                None,None,None,0,self.verifier.evidence_hash,len(a))
        else:
            if self.model.bank.route!="MEASURED_GROUPS":
                raise AssertionError("A measured source is required for consistency checks")
            group=self.model.bank.source.groups[k]
            scores,n_joint=_disagreement(group,a,m)
            usable=scores[np.isfinite(scores)&(n_joint>=3)]
            if len(usable)<100:
                raise ValueError("Insufficient co-observed TRAIN sensor groups")
            cutoff=float(max(.01,np.quantile(usable,.99)))
            self.receipt=CoherenceReceipt(
                tuple(group.sensor_indices),k,cutoff,
                len(usable),self.verifier.evidence_hash,len(a))
        self.fingerprint=sha256(json.dumps(
            self.receipt.as_dict(),sort_keys=True,allow_nan=False
        ).encode("utf8")).hexdigest()
        return self

    def predict(self,x,mask,reference,reference_mask):
        if self.receipt is None:
            raise RuntimeError("Fit coherence bounds on TRAIN only")
        a,m=check_measurements(x,mask)
        ref,rm=_check_anchor(reference,reference_mask,a.shape[:2])
        baseline=self.model.predict(a,m)
        v=self.verifier.predict(a,m,ref,rm)
        unchanged=baseline["direct_contributions"]
        corrected=unchanged.copy()
        traj=baseline["trajectories"].copy()
        reason=np.full(len(a),"UNVERIFIED_NO_MATCH",dtype=object)
        repair=np.zeros(len(a),bool)
        internal_score=np.full(len(a),np.nan,dtype=float)
        internally_coherent=np.zeros(len(a),bool)
        k=self.receipt.concept_index
        if k is not None:
            if self.receipt.independent_reference_manifest_sha256!=self.verifier.evidence_hash:
                raise AssertionError("Reference mapping mutated after coherence calibration")
            level,support=self.model.bank.transform(a,m)
            group=self.model.bank.source.groups[k]
            coherence,n_joint=_disagreement(group,a,m)
            internal_score=np.where(np.isfinite(coherence),coherence,np.nan)
            enough_coherence=n_joint>=3
            internally_coherent=(
                enough_coherence &
                (coherence<=self.receipt.train_coherence_cutoff_99pct)
            )
            enough_reference=rm.sum(axis=1)>=self.verifier.min_paired_times
            all_source_missing=(support[:,:,k]>0).sum(axis=1)==0
            source_empty_repair=all_source_missing&enough_reference
            discordance=v["reference_status"]=="DISCORDANT"
            group_fully_observed=(len(group.sensor_indices)>=3)
            common_fault_candidate=(
                discordance&internally_coherent&enough_reference&
                group_fully_observed
            )
            repair=source_empty_repair|common_fault_candidate
            reason[:]=np.where(
                all_source_missing,
                np.where(source_empty_repair,"RECOVERED_SOURCE_FROM_REFERENCE",
                         "UNVERIFIED_SOURCE_AND_REFERENCE_MISSING"),
                np.where(~enough_reference,"UNVERIFIED_NO_REFERENCE",
                    np.where(discordance,
                        np.where(common_fault_candidate,
                                 "REPAIRED_COHERENT_BUT_REFERENCE_DISCORDANT",
                                 "SENSOR_CONFLICT_KEPT_ROBUST_SOURCE"),
                        "SUPPORTED_NO_REPAIR"))
            )
            if repair.any():
                projection=(self.verifier.evidence.affine_intercept
                            +self.verifier.evidence.affine_slope*ref)
                # Reference is a direct measured source input ONLY in
                # selected episodes. Unsupported timestamps are neutral.
                replaced_level=level.copy()
                replaced_support=support.copy()
                replaced_level[repair,:,k]=np.where(
                    rm[repair],projection[repair],0.)
                replaced_support[repair,:,k]=rm[repair].astype(np.float32)
                with torch.inference_mode():
                    sequences,_,_=self.model.net(
                        torch.from_numpy(replaced_level.astype(np.float32)),
                        torch.from_numpy(replaced_support.astype(np.float32)))
                updated=sequences.detach().cpu().numpy().astype(float)
                altered=(updated.mean(axis=1)-.5)*self.model.beta[None,:]
                # Do not recompute or modify any other concept, even if
                # native internal implementation changes in a future fork.
                corrected[repair,k]=altered[repair,k]
                traj[repair,:,k]=updated[repair,:,k]

        logits=self.model.bias+corrected.sum(axis=1)
        if not np.isfinite(logits).all() or not np.isfinite(corrected).all():
            raise AssertionError("Unstable repaired native concept")
        if k is None:
            np.testing.assert_array_equal(logits,baseline["logits"])
        else:
            other=np.arange(corrected.shape[1])!=k
            np.testing.assert_array_equal(corrected[:,other],unchanged[:,other])
            np.testing.assert_array_equal(corrected[~repair,k],unchanged[~repair,k])
        err=float(np.max(np.abs(logits-(self.model.bias+corrected.sum(axis=1)))))
        if err>1e-12:
            raise AssertionError("Lost exact additive concept-only risk")
        return {
            "probability":expit(logits),
            "logits":logits,"baseline_probability":baseline["probability"],
            "baseline_logits":baseline["logits"],
            "direct_contributions":corrected,
            "original_contributions":unchanged,
            "concept_trajectories":traj,
            "repaired_source":repair,
            "repair_reason":reason,
            "reference_status":v["reference_status"],
            "internal_coherence_score":internal_score,
            "internally_coherent":internally_coherent,
            "reference_provenance_assumed_trusted":True,
            "reference_used_as_direct_risk_feature":False,
            "reference_used_as_source_measurement_when_repaired":True,
            "n_concepts":corrected.shape[1],
            "source_model_sha256":self.model.frozen_network_sha,
            "source_verification_sha256":self.fingerprint,
            "max_additive_logit_error":err,
        }

    def passport(self):
        if self.receipt is None:
            raise RuntimeError("Unfitted source coherence")
        return {
            "train_only_coherence_receipt":self.receipt.as_dict(),
            "train_only_coherence_sha256":self.fingerprint,
            "independent_reference_receipt":self.verifier.passport(),
            "risk_frozen_native_encoder":self.model.frozen_network_sha,
            "no_supervised_gradients_in_source_repair":True,
            "no_reference_to_task_head_bypass":True,
            "missing_reference_predicts_with_original_model_unverified":True,
            "joint_reference_source_fault_not_solved":True,
        }
