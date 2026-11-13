import unittest

import numpy as np

from conceptfan_discovery.independent_anchor_verification_v57 import IndependentAnchorVerifier
from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN, TrainOnlySourceBank
)


def independent_data(seed=79,n=360,t=16):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(n,t,2))
    for i in range(1,t):
        z[:,i]=.72*z[:,i-1]+.28*z[:,i]
    e=lambda s: rng.normal(scale=s,size=(n,t))
    x=np.stack([
        z[:,:,0]+e(.16),z[:,:,0]+e(.16),z[:,:,0]+e(.18),
        z[:,:,1]+e(.16),z[:,:,1]+e(.17),
        rng.normal(size=(n,t))],axis=-1)
    m=rng.random(x.shape)>.18
    anchor=z[:,:,0]+e(.15)  # External independent-noise measurement
    am=rng.random(anchor.shape)>.16
    y=(z[:,-5:,0].mean(axis=1)+z[:,-5:,1].mean(axis=1)>.03).astype(int)
    return x,m,anchor,am,y


class V57IndependentMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.m,cls.anchor,cls.am,cls.y=independent_data()
        cls.bank=TrainOnlySourceBank(
            seed=171,permutations=25,bootstraps=5
        ).fit_unlabeled(cls.x,cls.m)
        cls.native=NativeSelfSupervisedConceptFAN(
            cls.bank,neural_seed=11,epochs=2
        ).fit_source_unlabeled(cls.x,cls.m)
        cls.native.fit_risk_head(cls.x,cls.m,cls.y)
        cls.verifier=IndependentAnchorVerifier(
            cls.native,seed=171
        ).fit_from_unlabeled(cls.x,cls.m,cls.anchor,cls.am)

    def test_external_anchor_selects_group_not_oracle_id(self):
        receipt=self.verifier.evidence
        self.assertEqual(receipt.verification_status,
                         "TRAIN_ASSOCIATION_AND_RESIDUAL_FROZEN")
        self.assertEqual(set(receipt.detected_sensor_group),{0,1,2})
        self.assertGreater(receipt.min_abs_train_fold_correlation,.65)
        self.assertFalse(receipt.latent_oracle_states_used)
        self.assertFalse(receipt.outcome_labels_used)

    def test_clean_reference_rarely_rejects_train_distribution(self):
        result=self.verifier.predict(self.x,self.m,self.anchor,self.am)
        self.assertLess(result["reference_rejected"].mean(),.08)
        self.assertLessEqual(result["max_additive_error"],1e-12)

    def test_common_mode_inversion_rejected_by_independent_reference(self):
        x=self.x[:64].copy()
        x[:,:,:3]*=-1.
        result=self.verifier.predict(x,self.m[:64],self.anchor[:64],self.am[:64])
        self.assertGreater(result["reference_rejected"].mean(),.40)
        k=self.verifier.evidence.matched_concept_index
        np.testing.assert_array_equal(
            result["corrected_contributions"][result["reference_rejected"],k],0.)
        unchanged=np.arange(result["corrected_contributions"].shape[1])!=k
        np.testing.assert_array_equal(
            result["baseline_contributions"][:,unchanged],
            result["corrected_contributions"][:,unchanged])

    def test_missing_reference_exact_fallback_not_magic_reconstruction(self):
        x=self.x[:7].copy()
        missing=np.zeros_like(self.am[:7],dtype=bool)
        out=self.verifier.predict(x,self.m[:7],self.anchor[:7],missing)
        self.assertTrue(np.all(out["reference_rejected"]))
        self.assertTrue(np.all(out["reference_status"]=="NO_REFERENCE"))
        k=self.verifier.evidence.matched_concept_index
        np.testing.assert_array_equal(out["corrected_contributions"][:,k],
                                      np.zeros(7))
        changed=self.anchor[:7].copy()
        changed[:]=np.nan
        other=self.verifier.predict(x,self.m[:7],changed,missing)
        np.testing.assert_array_equal(other["corrected_logits"],
                                      out["corrected_logits"])

    def test_joint_source_and_anchor_flip_is_known_counterexample(self):
        x=self.x[:64].copy();x[:,:,:3]*=-1
        wrong_reference=self.anchor[:64].copy()
        wrong_reference*=-1
        separate=self.verifier.predict(x,self.m[:64],self.anchor[:64],self.am[:64])
        joint=self.verifier.predict(x,self.m[:64],wrong_reference,self.am[:64])
        self.assertGreater(separate["reference_rejected"].mean(),
                           joint["reference_rejected"].mean())

    def test_noise_anchor_refuses_to_claim_source_identity(self):
        random_anchor=np.random.default_rng(107).normal(size=self.anchor.shape)
        guard=IndependentAnchorVerifier(self.native,seed=171)
        guard.fit_from_unlabeled(self.x,self.m,random_anchor,self.am)
        self.assertEqual(guard.evidence.verification_status,
                         "UNVERIFIED_NO_REPRODUCIBLE_ASSOCIATION")
        result=guard.predict(self.x[:6],self.m[:6],random_anchor[:6],self.am[:6])
        np.testing.assert_array_equal(result["baseline_logits"],
                                      result["corrected_logits"])
        self.assertFalse(result["reference_rejected"].any())

    def test_no_task_source_retraining_during_verification(self):
        source_sha=self.native.frozen_network_sha
        betas=self.native.beta.copy()
        p=self.verifier.passport()
        self.assertFalse(p["task_labels_used_to_match_reference"])
        self.assertFalse(p["reference_raw_values_used_as_risk_predictor"])
        self.assertEqual(source_sha,self.native.frozen_network_sha)
        np.testing.assert_array_equal(betas,self.native.beta)

if __name__=="__main__":
    unittest.main()
