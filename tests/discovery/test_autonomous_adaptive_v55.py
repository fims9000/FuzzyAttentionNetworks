import unittest
import numpy as np

from conceptfan_discovery.autonomous_adaptive_conceptfan_v55 import AdaptiveConceptFAN


def grouped_sample(seed=14,n=360,t=16):
    rng=np.random.default_rng(seed)
    f=rng.normal(size=(n,t,2))
    for j in range(1,t):
        f[:,j]=.75*f[:,j-1]+.25*f[:,j]
    noise=lambda: rng.normal(scale=.12,size=(n,t))
    x=np.stack((
        f[:,:,0]+noise(),f[:,:,0]+noise(),
        f[:,:,1]+noise(),f[:,:,1]+noise(),
        rng.normal(size=(n,t)),rng.normal(size=(n,t))
    ),axis=-1)
    m=rng.random(x.shape)>.18
    y=((f[:,-6:,0].mean(axis=1)-f[:,-6:,1].mean(axis=1))>0).astype(int)
    return x,m,y


def mixed_sample(seed=45,n=210,t=16,k=8):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(n,t,k))
    for step in range(1,t):
        z[:,step]=.77*z[:,step-1]+.23*z[:,step]
    x=z+.07*rng.normal(size=z.shape)
    m=rng.random(x.shape)>.12
    y=(z[:,-4:,0].mean(axis=1)>.05).astype(int)
    return x,m,y


class AdaptiveConceptFANV55Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.m,cls.y=grouped_sample()
        cls.model=AdaptiveConceptFAN(seed=20261331,permutations=25,bootstraps=5)
        cls.model.fit(cls.x,cls.m,cls.y)

    def test_measured_route_has_no_manual_concepts_and_exact_logit(self):
        self.assertEqual(self.model.route,"MEASURED_GROUPS")
        self.assertEqual(self.model.selected_k,2)
        p=self.model.predict(self.x[:12],self.m[:12])
        self.assertLessEqual(p["max_additive_logit_error"],1e-12)
        self.assertEqual(p["contributions"].shape,(12,2))
        self.assertFalse(self.model.public_passport()["clinical_disease_labels_inferred"])
        self.assertTrue(self.model.route_receipt["source_definition_train_only"])

    def test_label_permutation_does_not_change_source_route_or_manifest(self):
        rng=np.random.default_rng(114)
        yy=self.y[rng.permutation(len(self.y))]
        alternative=AdaptiveConceptFAN(
            seed=20261331,permutations=25,bootstraps=5).fit(self.x,self.m,yy)
        self.assertEqual(alternative.route,self.model.route)
        self.assertEqual(alternative.source_hash,self.model.source_hash)

    def test_all_group_measurements_hidden_gives_bias_only(self):
        absent=np.zeros_like(self.m[:8],dtype=bool)
        x=self.x[:8].copy()
        pred=self.model.predict(x,absent)
        np.testing.assert_array_equal(pred["contributions"],
                                       np.zeros_like(pred["contributions"]))
        np.testing.assert_array_equal(pred["logits"],
                                       np.full(8,self.model.model.bias))
        x[~absent]=np.nan  # permitted: no unobserved payload may be evaluated
        same=self.model.predict(x,absent)
        np.testing.assert_array_equal(same["logits"],pred["logits"])

    def test_unknown_events_are_not_diagnoses(self):
        events=self.model.event_summary(self.x[:32],self.m[:32])
        self.assertEqual(len(events),2)
        self.assertTrue(all(0<=v["episode_fraction_with_event"]<=1
                            for v in events.values()))

    def test_mixed_no_groups_routes_into_nonphysical_spectral_subspace(self):
        x,m,y=mixed_sample()
        net=AdaptiveConceptFAN(seed=20261381,permutations=25,bootstraps=5)
        net.fit(x,m,y)
        self.assertEqual(net.route,"PREDICTIVE_SUBSPACE")
        self.assertGreater(net.selected_k,0)
        pred=net.predict(x[:18],m[:18])
        self.assertEqual(pred["contributions"].shape,(18,net.selected_k))
        self.assertEqual(pred["source_semantics"],
                         "OBSERVABLE_PREDICTIVE_SUBSPACE_ONLY")
        self.assertLessEqual(pred["max_additive_logit_error"],1e-12)

    def test_source_absence_in_spectral_model_exact_neutral(self):
        x,m,y=mixed_sample(seed=46)
        net=AdaptiveConceptFAN(seed=20261391,permutations=25,bootstraps=5)
        net.fit(x,m,y)
        self.assertEqual(net.route,"PREDICTIVE_SUBSPACE")
        missing=np.zeros_like(m[:6])
        result=net.predict(x[:6],missing)
        np.testing.assert_array_equal(
            result["contributions"],np.zeros_like(result["contributions"]))
        changed=x[:6].copy()
        changed[:,:,:]=7e8
        p2=net.predict(changed,missing)
        np.testing.assert_array_equal(result["probability"],p2["probability"])

if __name__=="__main__":
    unittest.main()
