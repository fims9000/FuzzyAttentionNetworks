# V17 frozen positive control: does an admitted new native concept improve actual prediction?

This experiment is separate from the already executed V17 source-versus-noise falsification, and from V16. It tests whether a newly proposed, **actually inserted** concept can learn risk and affect the prediction after further training.

## Controlled synthetic task
- New generator seed 20311104; complete 5-channel, 30-time-step sequences, hidden AR state dimension 3. The first two persistent sources are represented by an intentionally underspecified K2 initial observable basis. A third hidden source projects to sensor_2 and drives the outcome in the subsequent six steps. The held-out future trajectories are NEVER passed to concept formation.
- Split: 600 TRAIN episodes, 200 previously unused VALIDATION, 40 sealed TEST. Train-only channel median/IQR normalization. This is an intentionally favorable *positive control*, not a naturalistic hospital setting or general concept-discovery guarantee.
- TRAIN-only residual-lag candidate search K2→K3; 99 null permutations, 24 episode bootstrap draws; no oracle factor state sent to estimator.
- Programmatic approval of the candidate occurs only in this controlled synthetic positive test; real human review was not done.
- For seeds 11/23/37, fit the original K2 V10 for 6 epochs. Then compare its copy continued for 6 epochs, the K2→K3 expanded native model continued for the same 6 epochs, and K3 trained from scratch for 12 epochs. All paths use the same native V10 physical and observation head and same train-only outcomes; no raw-input bypass or post-hoc concepts. Separate optimizer reset at growth is explicitly an experimental difference from continuous scratch training.
- Primary metric: AUPRC on untouched synthetic VALIDATION. Secondary: AUROC, exact additive logit error, average absolute direct logit effect of the added slot. Keep all 9 models, no p-values with 3 seeds.
- Check immediate prediction preservation before retraining. Any newly proposed concept with zero readout is mathematically silent until task learning, which is intentional.
- Do not use this dataset or its validation outcomes to select a final clinical design, sign label, rank or hyperparameters. Positive differences are *demonstrations on an intentionally constructed task*, not proofs of superiority to V10/MAGNETS or semantic factor identifiability.

The code also documents the artifact-level insufficiency of original V16: better concept content is not the same as stable signed risk contribution. Stable effect remains an independent scientific acceptance criterion.
