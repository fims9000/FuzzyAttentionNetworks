# V21 offline four-choice review: from a native concept to TRAIN-only supervision

This is a **local research-only workflow**, not a clinical application, consent framework, diagnosis or validated physician annotation study. It makes the previous train-native `yes` / `no` / `unsure` / `artifact` workflow usable without making a reviewer write equations or see the model's predicted outcome.

## Reviewer's workflow
A researcher who has lawful access to a properly deidentified TRAIN cohort and the trained ConceptFAN model calls `create_offline_native_review_packet` with the concept ID, concept slot and observed channel names. The packet includes a random signed review-round nonce, pseudonymous case tokens and short measured historical curves. It does **not** include original episode IDs, outcome labels, model risk predictions or generator latent truth. The cases deliberately span high, low and ambiguous native activation but **their model confidence is hidden from the reviewer**.

`render_local_review_html(packet)` or `write_offline_review_page(packet,output_path)` produces a self-contained HTML file that functions with no network connection. The reviewer sees simple curves, a plain question and four actions: **Да / Нет / Не уверен / Артефакт измерения**. They export `conceptfan_review_answers.json` after selecting the cases they can judge; unanswered cases remain absent, not counted as negatives. The choices in the browser do **not** train a network by themselves.

The researcher imports the JSON with `import_offline_native_review_answers(...)`. It validates the signed packet HMAC, verifies all pseudonymous tokens belong to the authorized original TRAIN episode IDs, blocks duplicate/unknown reviews and wrong slot/window/model dimensions, then creates exactly the existing V16 `PreparedFeedback` object. The usual `fit_multiscale(... expert_feedback=prepared, expert_feedback_weight=1.0, feature_exploration=True)` supervises the same native risk-mediating K3 concept layer.

`unsure` and `artifact` remain **quarantined, not labelled negative**. Contradictory intervals are excluded by V16's existing conflict logic. No clinical or causal identity can be inferred from a click. A bad reviewer can harm performance, as demonstrated by the V16 and V21 inverted-feedback controls.

## Researcher-side integration example

```python
import json, secrets
from pathlib import Path
from conceptfan_discovery.offline_native_review_v21 import (
    create_offline_native_review_packet,
    write_offline_review_page,
    import_offline_native_review_answers,
)

# Persist this private key securely OUTSIDE your Git repo and do NOT
# append it to the packet or reviewer HTML. Re-use it only to import
# the same review round, then follow your data retention policy.
key = secrets.token_bytes(32)
packet = create_offline_native_review_packet(
    trained_model, data.train,
    candidate_id="experimentally-measured-residual-001",
    concept_slot=2,
    channel_names=tuple(data.receipt["channel_names"]),
    private_key=key,
    cases_per_stratum=4,
)
write_offline_review_page(packet, Path("/private/clinician_review.html"))

# After a reviewer has opened the local page and saved their choices:
answers = json.loads(Path("/private/conceptfan_review_answers.json").read_text())
prepared, audit = import_offline_native_review_answers(
    packet, answers, private_key=key,
    original_train_episode_ids=data.train.episode_ids,
    n_slots=trained_model.physical.n_slots,
    n_steps=data.train.x.shape[1],
    reviewer_id="local_authorized_reviewer_01",
)
# 'prepared' is the EXISTING V16 native concept-supervision object.
# Train through fit_multiscale(..., expert_feedback=prepared,
#                              expert_feedback_weight=1.0,
#                              feature_exploration=True).
```

A reviewer HTML file **still contains measurement time series** that may reveal sensitive information despite pseudonyms. This is not encryption, access control, consent, or safe external transfer. Only generate packets on properly deidentified or fully synthetic data, store in an access-controlled local workspace, do not use a public web host, and do not upload HTML/packet/answers to GitHub or CI artifacts. Private HMAC keys must be kept separate from the packet and answers; do not commit keys.

## Source of truth and limitations
Implementation: `src/conceptfan_discovery/offline_native_review_v21.py`.
Security / integration tests: `tests/discovery/test_offline_native_review_v21.py`.
Signed packet means the reviewer cannot silently alter the measurement records without detection; it does **not** authenticate a real human or prove that an answer is medically valid. Browser UI is a deliberately minimal **offline review aid**, not a production clinical EHR application, multi-user server or ethics-approved study.

The executed V21 experiment used **simulated measured-reference labels, not this human GUI**. Its scientific result is an explicit physical-versus-instrument falsification; do not claim clinical reviewers have already been recruited or evaluated.
