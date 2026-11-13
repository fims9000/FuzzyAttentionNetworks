import unittest
import numpy as np

from conceptfan_discovery.coherence_aware_anchor_repair_v58 import (
    CoherenceAwareAnchorRepair
)
from conceptfan_discovery.independent_anchor_verification_v57 import (
    IndependentAnchorVerifier
)
from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN,TrainOnlySourceBank
)


def sample(seed=20261491,n=360,t=16):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(n,t,2))
    for step in range(1,t):
        z[:,step]=.73*z[:,step-1]+.27*z[:,step]
    eps=lambda s:rng.normal(scale=s,size=(n,t))
    x=np.stack((
        z[:,:,0]+eps(.15),z[:,:,0]+eps(.15),z[:,:,0]+eps(.18),
        z[:,:,1]+eps(.15),z[:,:,1]+eps(.15),
        rng.normal(size=(n,t))
    ),axis=-1)
    m=rng.random(x.shape)>.18
    ref=z[:,:,0]+eps(.15)
    rm=rng.random(ref.shape)>.15
    y=(z[:,-6:,0].mean(axis=1)>z[:,-6:,1].mean(axis=1)).astype(int)
    return x,m,ref,rm,y


class V58CoherenceRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.m,cls.ref,cls.rm,cls.y=sample()
        bank=TrainOnlySourceBank(seed=17,permutations=25,bootstraps=5)
        bank.fit_unlabeled(cls.x,cls.m)
        cls.model=NativeSelfSupervisedConceptFAN(
            bank,neural_seed=11,epochs=2).fit_source_unlabeled(cls.x,cls.m)
        cls.verifier=IndependentAnchorVerifier(
            cls.model,seed=19).fit_from_unlabeled(
                cls.x,cls.m,cls.ref,cls.rm)
        cls.repair=CoherenceAwareAnchorRepair(cls.verifier).fit_from_unlabeled(
            cls.x,cls.m)
        cls.model.fit_risk_head(cls.x,cls.m,cls.y)

    def test_receipts_have_no_task_labels_or_oracle(self):
        manifest=self.repair.passport()
        self.assertTrue(manifest["no_supervised_gradients_in_source_repair"])
        self.assertFalse(manifest["no_reference_to_task_head_bypass"] is False)
        self.assertEqual(set(self.repair.receipt.source_group),{0,1,2})
        self.assertGreater(self.repair.receipt.n_train_coherence_subjects,100)

    def test_common_mode_fault_repair_and_unchanged_other_concepts(self):
        x=self.x[:70].copy()
        x[:,:,:3]*=-1.
        out=self.repair.predict(x,self.m[:70],self.ref[:70],self.rm[:70])
        self.assertGreater(out["repaired_source"].mean(),.35)
        self.assertLessEqual(out["max_additive_logit_error"],1e-12)
        k=self.repair.receipt.concept_index
        other=np.arange(self.model.bank.k)!=k
        np.testing.assert_array_equal(
            out["direct_contributions"][:,other],
            out["original_contributions"][:,other])

    def test_single_bad_sensor_keeps_robust_group_instead_of_zeroing(self):
        x=self.x[:70].copy()
        x[:,:,0]*=-1
        out=self.repair.predict(x,self.m[:70],self.ref[:70],self.rm[:70])
        old=self.verifier.predict(x,self.m[:70],self.ref[:70],self.rm[:70])
        # Some single faults may be ambiguous under random missingness;
        # the within-group check must reject repair on at least a subset.
        self.assertGreater(np.mean(
            out["repair_reason"]=="SENSOR_CONFLICT_KEPT_ROBUST_SOURCE"),.05)
        self.assertLess(out["repaired_source"].mean(),
                        old["reference_rejected"].mean()+.01)

    def test_missing_external_reference_preserves_original_risk(self):
        absent=np.zeros_like(self.rm[:8],dtype=bool)
        out=self.repair.predict(
            self.x[:8],self.m[:8],self.ref[:8],absent)
        base=self.model.predict(self.x[:8],self.m[:8])
        np.testing.assert_array_equal(out["probability"],base["probability"])
        self.assertFalse(out["repaired_source"].any())
        self.assertTrue(np.all(
            out["repair_reason"]=="UNVERIFIED_NO_REFERENCE"))
        nan_payload=np.full(self.ref[:8].shape,np.nan)
        out2=self.repair.predict(self.x[:8],self.m[:8],nan_payload,absent)
        np.testing.assert_array_equal(out2["logits"],base["logits"])

    def test_fully_missing_source_is_reconstructed_by_independent_sensor(self):
        missing=self.m[:20].copy()
        missing[:,:,:3]=False
        out=self.repair.predict(self.x[:20],missing,self.ref[:20],self.rm[:20])
        self.assertTrue(np.all(out["repaired_source"]))
        self.assertTrue(np.all(
            out["repair_reason"]=="RECOVERED_SOURCE_FROM_REFERENCE"))
        self.assertTrue(np.isfinite(out["probability"]).all())
        k=self.repair.receipt.concept_index
        self.assertGreater(np.mean(np.abs(
            out["direct_contributions"][:,k])),.005)

    def test_joint_reference_source_fault_can_escape(self):
        x=self.x[:70].copy();x[:,:,:3]*=-1
        reference=self.ref[:70].copy()*-1
        out=self.repair.predict(x,self.m[:70],reference,self.rm[:70])
        separate=self.repair.predict(
            x,self.m[:70],self.ref[:70],self.rm[:70])
        self.assertLess(out["repaired_source"].mean(),
                        separate["repaired_source"].mean())

    def test_anchor_only_corruption_remains_ambiguity(self):
        out=self.repair.predict(
            self.x[:60],self.m[:60],-self.ref[:60],self.rm[:60])
        # A single synthetic external device can itself be wrong.
        self.assertGreater(out["repaired_source"].mean(),.20)
        self.assertTrue(out["reference_provenance_assumed_trusted"])

    def test_no_change_to_supervised_weights_or_source_network(self):
        sha=self.model.frozen_network_sha
        beta=self.model.beta.copy()
        self.repair.predict(self.x[:16],self.m[:16],
                            self.ref[:16],self.rm[:16])
        self.assertEqual(self.model.frozen_network_sha,sha)
        np.testing.assert_array_equal(self.model.beta,beta)

if __name__=="__main__":
    unittest.main()
