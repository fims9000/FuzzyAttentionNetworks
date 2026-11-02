"""Match native V7 observation-process concept slots across retrainings.

This is NOT a post-hoc concept extractor: each source `ObservationProcessConceptHead`
has the exact learned signed temporal-mask operator coefficients, threshold
and direct logit beta used in mortality training. The audit only packages
those source-native quantities into the same independent-model matching
contract as physiological concepts.

Matching uses definition cosine alone (no patient outcomes, no oracle).
Patient-specific agreement is compared against patient-independent
frequency-preserving nulls. Similarity of parameter definitions alone
does not establish identical patient-level explanation meaning.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .evaluation import DiscoveryEvidence
from .observation_process_concepts_v7 import DualProcessConceptFAN


def collect_observation_process_evidence(
    model:DualProcessConceptFAN,
    measured_mask:np.ndarray,
    reference_outcome:np.ndarray,
)->DiscoveryEvidence:
    m=np.asarray(measured_mask,dtype=np.float32)
    y=np.asarray(reference_outcome,dtype=np.float32)
    if m.ndim!=3 or y.shape!=(len(m),) or np.unique(y).size<2:
        raise ValueError("Mask-only model comparison needs matched clinical cohort")
    model.eval()
    with torch.no_grad():
        concepts, contributions, _, feature_weights=model.process(
            torch.from_numpy(m)
        )
        process_weights=model.process.direct_weights()
        threshold=torch.tanh(model.process.threshold_logits)
        width=.08+torch.nn.functional.softplus(model.process.width_logits)
        # One time-independent summary per 36-hour patient; unlike native
        # physiology trajectories, the observation-process concept describes
        # aggregate sampling behavior of the whole measured input window.
        c=concepts.detach().cpu().numpy()
        contribution=contributions.detach().cpu().numpy()
        chance=torch.sigmoid(
            model.physical.decision_bias+contributions.sum(dim=-1)
        ).detach().cpu().numpy()
        weights=feature_weights.detach().cpu().numpy()
        signed=process_weights.detach().cpu().numpy()
        tau=threshold.detach().cpu().numpy()
        scale=width.detach().cpu().numpy()
    active=(c.var(axis=0)>.0005)
    ones=np.ones((len(c),1,c.shape[-1]),dtype=np.float32)
    score={
        "n_examples":len(c),
        "n_active_and_nonconstant_slots":int(active.sum()),
        "mean_abs_contributions":float(np.abs(contribution).sum(axis=1).mean()),
        "auprc_mask_only_observation_process":float(
            average_precision_score(y,chance)
        ),
        "source":"mask_only_care_process_concept_operators",
        "clinical_pathology_claim":False,
    }
    return DiscoveryEvidence(
        metrics=score,probabilities=chance,
        contributions=contribution,
        trajectories=c[:,None,:],
        raw_activation=c[:,None,:],
        reliability=ones,
        weights=weights,
        gates=np.ones(c.shape[-1],dtype=np.float32),
        boundary=tau,width=scale,
        decision_weights=signed,
    )
