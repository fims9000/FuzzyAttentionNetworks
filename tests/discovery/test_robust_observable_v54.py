import unittest
import numpy as np

from conceptfan_discovery.autonomous_temporal_bank_v53 import TemporalGroup, group_signals
from conceptfan_discovery.robust_observable_concepts_v54 import (
    RobustGroupConceptFAN, robust_group_level, fit_robust_vocabulary,
    fit_observable_spectral_definitions, SpectralObservableConceptFAN,
)


class RobustObservableConceptTests(unittest.TestCase):
    def test_majority_reduces_single_flipped_sensor_effect(self):
        rng=np.random.default_rng(7)
        s=rng.normal(size=(12,10))
        x=np.stack((s+.02*rng.normal(size=s.shape),
                    s+.02*rng.normal(size=s.shape),
                    s+.02*rng.normal(size=s.shape)),axis=-1)
        m=np.ones_like(x,dtype=bool)
        source=TemporalGroup(
            "triple",(0,1,2),(1,1,1),(0.,0.,0.),(1.,1.,1.),
            .4,0.,1.,.5,1.,1.,1000,100)
        flipped=x.copy();flipped[:,:,0]*=-1.
        pre,_=robust_group_level(source,x,m)
        after,_=robust_group_level(source,flipped,m)
        old_pre,_,_,_=group_signals(source,x,m)
        old_after,_,_,_=group_signals(source,flipped,m)
        self.assertLess(np.mean(np.abs(after-pre)),
                        np.mean(np.abs(old_after-old_pre))/4.)

    def test_group_missing_is_neutral_even_with_hidden_payload(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(360,12,6))
        z=rng.normal(size=(360,12))
        x[:,:,0]=z;x[:,:,1]=z+.05*rng.normal(size=z.shape)
        x[:,:,2]=z+.06*rng.normal(size=z.shape)
        m=np.ones_like(x,dtype=bool)
        vocab=fit_robust_vocabulary(x,m,seed=19)
        self.assertEqual(len(vocab.groups),1)
        model=RobustGroupConceptFAN(vocab).fit_risk_head(x,m,(z.mean(axis=1)>0).astype(int))
        missing=m[:8].copy();missing[:,:,0:3]=False
        a=model.predict(x[:8],missing)
        x_hidden=x[:8].copy();x_hidden[:,:,0:3]=1.e9
        b=model.predict(x_hidden,missing)
        np.testing.assert_array_equal(a["probability"],b["probability"])
        np.testing.assert_array_equal(a["contributions"],np.zeros_like(a["contributions"]))
        self.assertLessEqual(a["logit_additive_error"],1e-12)

    def test_masked_train_only_spectral_basis_and_exact_logit(self):
        rng=np.random.default_rng(21)
        n,t,d=110,15,5
        z=np.zeros((n,t,2))
        z[:,0]=rng.normal(size=(n,2))
        for step in range(1,t):
            z[:,step]=.78*z[:,step-1]+rng.normal(scale=.15,size=(n,2))
        mixing=rng.normal(size=(d,2))
        x=z@mixing.T+rng.normal(scale=.07,size=(n,t,d))
        m=rng.random(x.shape)>.15
        definition=fit_observable_spectral_definitions(
            x,m,seed=53,permutations=25,bootstraps=5)
        self.assertFalse(definition.true_latent_states_used_to_fit)
        self.assertFalse(definition.target_labels_used_to_fit)
        self.assertGreaterEqual(definition.rank,1)
        y=(z[:,-3:,0].mean(axis=1)>0).astype(int)
        model=SpectralObservableConceptFAN(definition).fit_risk_head(x,m,y)
        p=model.predict(x[:9],m[:9])
        self.assertLessEqual(p["max_additive_error"],1e-12)
        missing=np.zeros_like(m[:5],dtype=bool)
        empty=model.predict(x[:5],missing)
        np.testing.assert_array_equal(empty["contributions"],np.zeros_like(empty["contributions"]))
        x2=x[:5].copy();x2[:,:,:]=900.
        alt=model.predict(x2,missing)
        np.testing.assert_array_equal(alt["logits"],empty["logits"])

    def test_no_future_leakage_in_spectral_trajectory(self):
        rng=np.random.default_rng(31)
        x=rng.normal(size=(100,10,5))
        x[:,1:]=.7*x[:,:-1]+.3*x[:,1:]
        m=np.ones_like(x,bool)
        definition=fit_observable_spectral_definitions(x,m,seed=7,permutations=25,bootstraps=5)
        model=SpectralObservableConceptFAN(definition)
        first=model.extract(x[:3],m[:3])["trajectories"]
        altered=x[:3].copy();altered[:,7:]=100.
        second=model.extract(altered,m[:3])["trajectories"]
        np.testing.assert_allclose(first[:,:7],second[:,:7],rtol=0,atol=0)

if __name__=="__main__":
    unittest.main()
