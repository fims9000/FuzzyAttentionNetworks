import unittest
import numpy as np

from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeCausalFuzzyDynamics,TrainOnlySourceBank,
    NativeSelfSupervisedConceptFAN
)


def measured(n=360,t=16,seed=14):
    rng=np.random.default_rng(seed)
    latent=rng.normal(size=(n,t,2))
    for step in range(1,t):
        latent[:,step]=.73*latent[:,step-1]+.27*latent[:,step]
    x=np.stack([
        latent[:,:,0]+rng.normal(scale=.12,size=(n,t)),
        latent[:,:,0]+rng.normal(scale=.12,size=(n,t)),
        latent[:,:,1]+rng.normal(scale=.12,size=(n,t)),
        latent[:,:,1]+rng.normal(scale=.12,size=(n,t)),
        rng.normal(size=(n,t)),rng.normal(size=(n,t))
    ],axis=-1)
    m=rng.random(x.shape)>.14
    y=(latent[:,-6:,0].mean(axis=1)>latent[:,-6:,1].mean(axis=1)).astype(int)
    return x,m,y


class NativeTemporalV56Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.m,cls.y=measured()
        cls.bank=TrainOnlySourceBank(seed=20261393,permutations=25,
                                     bootstraps=5).fit_unlabeled(cls.x,cls.m)
        cls.model=NativeSelfSupervisedConceptFAN(
            cls.bank,neural_seed=11,epochs=2).fit_source_unlabeled(cls.x,cls.m)
        cls.model.fit_risk_head(cls.x,cls.m,cls.y)

    def test_discovered_k_and_outcome_absent_in_source_fit(self):
        self.assertEqual(self.bank.route,"MEASURED_GROUPS")
        self.assertEqual(self.bank.k,2)
        self.assertFalse(self.bank.train_receipt["outcome_oracle_used_in_source_selection"])
        self.assertFalse(self.model.passport()["outcome_gradients_in_concept_training"])
        self.assertTrue(all(not p.requires_grad for p in self.model.net.parameters()))

    def test_all_missing_source_is_unknown_and_exact_zero(self):
        m=np.zeros_like(self.m[:6],dtype=bool)
        x=self.x[:6].copy()
        a=self.model.predict(x,m)
        np.testing.assert_array_equal(a["direct_contributions"],
                                       np.zeros_like(a["direct_contributions"]))
        np.testing.assert_array_equal(a["logits"],
                                       np.full(6,self.model.bias))
        x[:,:,:]=np.nan
        b=self.model.predict(x,m)
        np.testing.assert_array_equal(b["logits"],a["logits"])

    def test_model_is_causal_for_early_concept_time(self):
        first=self.model.predict(self.x[:5],self.m[:5])["trajectories"]
        later=self.x[:5].copy();later[:,8:]+=10000
        changed=self.model.predict(later,self.m[:5])["trajectories"]
        np.testing.assert_allclose(first[:,:8],changed[:,:8],rtol=0,atol=0)

    def test_additive_decision_no_raw_bypass(self):
        pred=self.model.predict(self.x[:15],self.m[:15])
        np.testing.assert_allclose(
            pred["logits"],self.model.bias+pred["direct_contributions"].sum(axis=1),
            rtol=0,atol=1e-12)
        self.assertLessEqual(pred["max_additive_logit_error"],1e-12)
        self.assertEqual(len(self.model.event_summary(self.x[:12],self.m[:12])),2)

    def test_outcome_permutation_does_not_update_source(self):
        fingerprint=self.model.frozen_network_sha
        original_weights=[
            p.detach().cpu().numpy().copy() for p in self.model.net.parameters()]
        rng=np.random.default_rng(77)
        shuffled=self.y[rng.permutation(len(self.y))]
        self.model.fit_risk_head(self.x,self.m,shuffled)
        self.assertEqual(self.model.frozen_network_sha,fingerprint)
        for previous,parameter in zip(original_weights,self.model.net.parameters()):
            np.testing.assert_array_equal(
                previous,parameter.detach().cpu().numpy())
        self.assertEqual(len(self.model.beta),self.bank.k)

    def test_legal_fuzzy_width_and_finite_unlabeled_training(self):
        self.assertEqual(len(self.model.loss_receipt),2)
        self.assertTrue(all(np.isfinite(r["mean_unlabeled_total_loss"])
                            for r in self.model.loss_receipt))
        self.assertTrue(all(w>0 for w in self.model.passport()["fuzzy_width"]))

    def test_no_task_head_before_source_fit(self):
        other=NativeSelfSupervisedConceptFAN(self.bank,neural_seed=23,epochs=1)
        with self.assertRaises(RuntimeError):
            other.fit_risk_head(self.x,self.m,self.y)

if __name__=="__main__":
    unittest.main()
