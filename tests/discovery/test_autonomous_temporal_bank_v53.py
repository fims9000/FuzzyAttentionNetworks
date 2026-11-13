import unittest

import numpy as np

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    AutonomousTemporalConceptFAN, check_measurements, event_intervals,
    fit_temporal_vocabulary, concepts_from_measurements
)


def sample_dataset(seed=17,n=360,t=12):
    rng=np.random.default_rng(seed)
    s0=rng.normal(size=(n,t))
    s1=rng.normal(size=(n,t))
    x=np.stack((
        s0+rng.normal(scale=.13,size=(n,t)),
        s0+rng.normal(scale=.13,size=(n,t)),
        s1+rng.normal(scale=.13,size=(n,t)),
        s1+rng.normal(scale=.13,size=(n,t)),
        rng.normal(size=(n,t)),
        rng.normal(size=(n,t))
    ),axis=-1)
    mask=rng.random(x.shape)>.15
    return x,mask,((s0.mean(axis=1)+s1.mean(axis=1))>0).astype(int)


class V53ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.m,cls.y=sample_dataset()
        cls.bank=fit_temporal_vocabulary(cls.x,cls.m,seed=678)

    def test_no_initial_manual_K2_and_stable_unsupervised_source_proposals(self):
        self.assertTrue(self.bank.no_known_sensor_group_supplied)
        self.assertTrue(self.bank.no_outcomes_used)
        self.assertEqual({tuple(g.sensor_indices) for g in self.bank.groups},
                         {(0,1),(2,3)})
        self.assertEqual(len(self.bank.groups),2)
        self.assertEqual(len(self.bank.excluded_singletons),2)

    def test_mask_hidden_values_are_never_used(self):
        one=self.bank
        x=self.x[:6].copy()
        m=self.m[:6].copy()
        alt=x.copy()
        alt[~m]=99999.0
        for dynamic in (False,True):
            a=concepts_from_measurements(one,x,m,include_surprise=dynamic)
            b=concepts_from_measurements(one,alt,m,include_surprise=dynamic)
            np.testing.assert_allclose(a["trajectories"],b["trajectories"],rtol=0,atol=0)

    def test_unobserved_group_exact_neutral_and_zero_contribution(self):
        x=self.x[:6]
        m=self.m[:6].copy()
        m[:,:,0:2]=False
        model=AutonomousTemporalConceptFAN(self.bank,include_surprise=True)
        model.fit_risk_head(self.x,self.m,self.y)
        pred=model.predict(x,m)
        np.testing.assert_array_equal(pred["trajectories"][:,:,0],.5)
        np.testing.assert_array_equal(pred["trajectories"][:,:,1],.5)
        np.testing.assert_array_equal(pred["contributions"][:,0],0.)
        np.testing.assert_array_equal(pred["contributions"][:,1],0.)
        self.assertLessEqual(pred["logit_additive_error"],1e-12)

    def test_causality_temporal_prefix(self):
        x=self.x[:8].copy()
        m=self.m[:8].copy()
        later=x.copy()
        later[:,8:]+=1000.
        a=concepts_from_measurements(self.bank,x,m)["trajectories"]
        b=concepts_from_measurements(self.bank,later,m)["trajectories"]
        np.testing.assert_allclose(a[:,:8],b[:,:8],rtol=0,atol=0)

    def test_train_only_source_receipt_and_strict_bottleneck(self):
        model=AutonomousTemporalConceptFAN(self.bank,include_surprise=True)
        model.fit_risk_head(self.x,self.m,self.y)
        p=model.predict(self.x[:12],self.m[:12])
        self.assertEqual(len(model.beta),4)
        self.assertTrue(np.isfinite(p["probability"]).all())
        np.testing.assert_allclose(p["logits"],
                 model.bias+p["contributions"].sum(axis=1),rtol=0,atol=1e-12)
        self.assertTrue(model.passport()["trained_source_without_labels"])

    def test_guard_rejects_single_channel_flip_without_retraining(self):
        model=AutonomousTemporalConceptFAN(self.bank,include_surprise=True,guard=True)
        model.fit_risk_head(self.x,self.m,self.y)
        x=self.x[:64].copy()
        m=np.ones_like(x,dtype=bool)
        x[:,:,0]*=-1
        result=model.predict(x,m)
        self.assertGreater(result["coherence_rejected"][:,0].mean(),.6)

    def test_event_intervals_are_explicit_time_bounds(self):
        self.assertEqual(event_intervals(np.array([.2,.72,.75,.1,.82,.83,.91,.3])),
                         ((1,3),(4,7)))

    def test_reject_nonbinary_mask_and_observed_nan(self):
        x=self.x[:2].copy()
        m=self.m[:2].astype(float)
        m[0,0,0]=2.
        with self.assertRaises(ValueError):
            check_measurements(x,m)
        m=self.m[:2].copy()
        x[0,0,0]=np.nan
        m[0,0,0]=True
        with self.assertRaises(ValueError):
            check_measurements(x,m)

    def test_no_group_source_falls_back_to_prevalence(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(360,12,6))
        m=np.ones_like(x,dtype=bool)
        bank=fit_temporal_vocabulary(x,m,seed=42)
        self.assertEqual(len(bank.groups),0)
        y=(rng.random(360)>.55).astype(int)
        model=AutonomousTemporalConceptFAN(bank).fit_risk_head(x,m,y)
        out=model.predict(x[:5],m[:5])
        self.assertEqual(out["contributions"].shape,(5,0))
        np.testing.assert_array_equal(out["logits"],np.full(5,model.bias))

if __name__=="__main__":
    unittest.main()
